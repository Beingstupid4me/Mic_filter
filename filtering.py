import os
import re
import json
from pathlib import Path
import logging
import time
from datetime import datetime
import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from accelerate import Accelerator
import gc

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler(); console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(message)s'); console_handler.setFormatter(console_formatter)
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers): logger.addHandler(console_handler)
logger.propagate = False

# --- Warnings Filter ---
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*TypedStorage is deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Unable to register cuFFT factory.*")
warnings.filterwarnings("ignore", message=".*Unable to register cuDNN factory.*")
warnings.filterwarnings("ignore", message=".*Unable to register cuBLAS factory.*")
warnings.filterwarnings("ignore", message=".*`do_sample` is set to `False`.*")
warnings.filterwarnings("ignore", message=".*A decoder-only architecture is being used.*")
warnings.filterwarnings("ignore", message=".*Passing `attn_implementation_preference` is deprecated.*")
warnings.filterwarnings("ignore", message=".*Using the pipeline API to.*")

# --- Input/Output Paths ---
STEP1_INPUT_DIR = Path("../Amartya_tasker/structured_parsed_chunks_combined") # Local input path
TAGGED_OUTPUT_DIR = Path("./filtered_tagged_chunks") # Local output path
TAGGED_OUTPUT_FILE_PATTERN = "tagged_chunk_batch_{batch_num}.jsonl"
FILTER_STATS_CSV = Path("./qwen3_filter_stats.csv") # Local stats path

# --- Model Configuration ---
MODEL_ID = "../Qwen3-0.6B" # <<< USING LOCAL QWEN3 PATH as requested
LLM_FILTER_BATCH_SIZE = 128 # Adjust based on V100 16GB VRAM with FP16
MAX_CONTEXT_LEN_FILTER = 2048
MAX_NEW_TOKENS_FILTER = 10

# --- Text Cleaning & Preparation ---
HEADER_PATTERN = re.compile(r'^Search Strategy.*?Results: \d+\s*(?=Document \d+ of \d+|\n\n|$)', re.DOTALL | re.MULTILINE | re.IGNORECASE)
METADATA_PATTERNS = re.compile(r'^(Author|URL|Publication title|Copyright|Document ID|ProQuest document ID|Database|Last updated|Source type|Language of publication|ISSN|Publication subject|Country of publication|Place of publication|Publisher|Section|Publication date|Publication year|Pages|Title|Company / organization|Location|Subject|Links|Abstract|Document \d+ of \d+):.*$', re.IGNORECASE | re.MULTILINE)
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
WHITESPACE_PATTERN = re.compile(r'\s+')
def strip_proquest_header(text):
    if not isinstance(text, str): return ""
    cleaned_text, num_subs = HEADER_PATTERN.subn('', text, count=1)
    if num_subs == 0: cleaned_text = METADATA_PATTERNS.sub('', text)
    # Added basic check for document start as well
    doc_match = re.search(r'Document \d+ of \d+', cleaned_text)
    if doc_match and doc_match.start() < 100: # If "Document X of Y" is near the start after other removals
         cleaned_text = cleaned_text[doc_match.end():]
    return cleaned_text.strip()

def prepare_text_for_filter_llm(header_stripped_text):
    if not isinstance(header_stripped_text, str): return ""
    try:
        text = URL_PATTERN.sub(' ', header_stripped_text)
        text = WHITESPACE_PATTERN.sub(' ', text).strip()
        # Use character limit based on context length
        char_limit = MAX_CONTEXT_LEN_FILTER * 5 # Heuristic
        return text[:char_limit]
    except Exception as e: logger.warning(f"Cleaning error (filter LLM): {e}. Text: {header_stripped_text[:100]}..."); return ""

# --- LLM Model Loading (Using FP16 as requested) ---
def load_filter_llm_model_and_tokenizer(model_id_path):
    """Loads the specified LLM using FP16 and device_map."""
    model_path = Path(model_id_path)
    if not model_path.exists() or not model_path.is_dir():
         logger.error(f"Model directory not found at: {model_path}")
         raise FileNotFoundError(f"Model directory not found: {model_path}")

    try:
        logger.info(f"--- Loading Filter LLM & Tokenizer ({model_path}) ---")
        if not torch.cuda.is_available(): logger.error("CUDA not available."); raise RuntimeError("CUDA not available")
        logger.info(f"Found {torch.cuda.device_count()} GPU(s).")

        # *** Force FP16 loading as requested ***
        target_dtype = torch.float16
        logger.info(f"Attempting to load model in FP16...")

        # Use device_map="auto"
        model = AutoModelForCausalLM.from_pretrained(
            model_path, # Use the Path object
            torch_dtype=target_dtype, # Force FP16
            device_map="auto",
            trust_remote_code=True # Often needed for community models/Qwen
        )
        logger.info(f"Filter LLM model loaded with forced dtype: {model.dtype}.")
        logger.info(f"Filter LLM model device map: {model.hf_device_map}")

        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side='left')
        logger.info("Tokenizer loaded.")
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
                logger.info(f"Tokenizer pad_token_id set to eos_token_id ({tokenizer.pad_token_id}).")
            else:
                logger.warning("Tokenizer lacks pad/eos. Setting pad_token_id to 0."); tokenizer.pad_token_id = 0
        else: logger.info(f"Tokenizer has pad_token_id: {tokenizer.pad_token_id}.")
        logger.info(f"Padding side: '{tokenizer.padding_side}'.")

        model.eval(); logger.info("--- Filter LLM and tokenizer ready ---")
        return model, tokenizer
    except Exception as e: logger.error(f"Filter LLM load failed: {str(e)}", exc_info=True); raise

# --- LLM Prompt Engineering for Filtering ---
def format_filter_prompt(text_chunk):
    """Creates the prompt asking the LLM to identify non-MIC articles."""
    instructions = """Analyze the following article text. Determine if the text is CLEARLY and DEFINITIVELY **NOT** about a militarized clash or armed conflict between the military forces of two different countries where military personnel died. Examples of non-MIC events include domestic news, sports, finance, accidents, politics without direct military clashes, etc.

If you are **highly confident** the text is **NOT** an MIC event as described, answer ONLY with the word "YES".
Otherwise, if there is any possibility it *could* be an MIC event, or if you are unsure, answer ONLY with the word "NO".

Article Text:
{context}

Answer (ONLY YES or NO):"""
    user_content = instructions.format(context=text_chunk)
    messages = [{"role": "user", "content": user_content}]
    return messages

# --- Data Loading ---
def load_data_only(directory, header_stripper):
    """Loads data, strips headers, returns items."""
    logger.info(f"--- Loading Data ({directory}) ---")
    all_items = []
    if not directory.exists() or not directory.is_dir():
        logger.error(f"Input directory not found: {directory}")
        return []
    file_paths = list(directory.rglob("*.jsonl")); total_chunks_read = 0
    if not file_paths: logger.warning(f"No *.jsonl files found in {directory}."); return []
    logger.info(f"Found {len(file_paths)} files. Reading...")
    for file_path in tqdm(file_paths, desc="Reading Files", unit="file", dynamic_ncols=True):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip()); total_chunks_read += 1
                        if "raw_chunk_text" in data and isinstance(data["raw_chunk_text"], str):
                            raw_text = data["raw_chunk_text"]
                            article_body = header_stripper(raw_text)
                            if not article_body: continue
                            item = { "raw_text": raw_text, "article_body": article_body, "metadata": { "source_file": data.get("original_file"), "original_subdir": data.get("original_subdir"), "representative_year": data.get("representative_year"), "chunk_index_in_file": data.get("chunk_index_in_file"), "inferred_date": data.get("inferred_date"), "publication_date": data.get("publication_date") } }
                            all_items.append(item)
                    except json.JSONDecodeError: logger.warning(f"Skipping invalid JSON line {line_num} in {file_path.name}")
                    except Exception as e: logger.warning(f"Error processing line {line_num} in {file_path.name}: {e}", exc_info=False)
        except Exception as e: logger.error(f"Failed read/process file {file_path}: {e}", exc_info=True)
    logger.info(f"--- Initial Data Load COMPLETE ---")
    logger.info(f"   Total raw chunks read: {total_chunks_read:,}")
    logger.info(f"   Valid items loaded for processing: {len(all_items):,}")
    return all_items

# --- LLM Filtering Execution ---
def run_llm_filtering(model, tokenizer, items_to_process, batch_size, gen_kwargs):
    """Uses LLM to classify items as definitely NOT MIC."""
    logger.info(f"--- LLM Filtering Stage ({len(items_to_process)} items) ---")
    logger.info(f"Batch Size: {batch_size}")
    all_decisions = {} # Store decision for each original index: True=Keep, False=Remove
    num_processed = 0
    num_excluded = 0
    is_dataparallel = isinstance(model, torch.nn.DataParallel) # Check if DP wrapped (might happen with device_map auto indirectly?)
    model_device = next(model.parameters()).device # Get device where model (or main part) is

    # Prepare prompts and original indices
    prompts_and_indices = []
    for idx, item in enumerate(items_to_process):
        # Use the function to clean and truncate text before formatting prompt
        cleaned_body = prepare_text_for_filter_llm(item["article_body"])
        if cleaned_body:
            prompt_messages = format_filter_prompt(cleaned_body)
            prompts_and_indices.append({"messages": prompt_messages, "original_index": idx})
        else:
            all_decisions[idx] = True # Keep items with empty bodies

    if not prompts_and_indices: logger.error("No valid items to filter."); return {}

    total_batches = (len(prompts_and_indices) + batch_size - 1) // batch_size
    logger.info(f"Total batches for LLM filtering: {total_batches}")

    for i in tqdm(range(0, len(prompts_and_indices), batch_size), desc="LLM Filtering", unit="batch"):
        batch_data = prompts_and_indices[i : i+batch_size]
        batch_messages = [item['messages'] for item in batch_data]
        batch_original_indices = [item['original_index'] for item in batch_data]
        tokenized_inputs = None
        batch_num = i // batch_size + 1

        try:
            # Apply chat template with thinking disabled
            batch_inputs_text = [ tokenizer.apply_chat_template( p, tokenize=False, add_generation_prompt=True, enable_thinking=False ) for p in batch_messages ]
            tokenized_inputs = tokenizer( batch_inputs_text, return_tensors="pt", padding=True, truncation=True, max_length=MAX_CONTEXT_LEN_FILTER, return_attention_mask=True ).to(model_device) # Move to primary model device
        except Exception as e: logger.error(f"Tokenization/Template error batch {batch_num}/{total_batches}: {e}", exc_info=True); all_decisions.update({idx: True for idx in batch_original_indices}); continue

        raw_responses = ["ERROR"] * len(batch_messages)
        try:
            with torch.no_grad():
                # Use model.module.generate if DataParallel wrapped (less likely with device_map but check)
                generate_func = model.module.generate if is_dataparallel else model.generate
                outputs = generate_func( input_ids=tokenized_inputs['input_ids'], attention_mask=tokenized_inputs['attention_mask'], **gen_kwargs )
                input_length = tokenized_inputs['input_ids'].shape[1]
                generated_tokens = outputs[:, input_length:]
                raw_responses = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                del outputs
        except RuntimeError as e:
            if "out of memory" in str(e).lower(): logger.warning(f"OOM LLM filter batch {batch_num}. Marking items as KEEP.")
            else: logger.error(f"Runtime error LLM filter batch {batch_num}: {e}", exc_info=False)
            all_decisions.update({idx: True for idx in batch_original_indices}) # Keep on error
        except Exception as e: logger.error(f"Unexpected error LLM filter batch {batch_num}: {e}", exc_info=False); all_decisions.update({idx: True for idx in batch_original_indices}) # Keep on error
        finally:
             if tokenized_inputs is not None: del tokenized_inputs
             gc.collect(); # Optional: torch.cuda.empty_cache()


        # Process responses for this batch
        for idx, response in enumerate(raw_responses):
            original_idx = batch_original_indices[idx]
            # Only update if no error occurred during generation for this item
            if original_idx not in all_decisions or all_decisions[original_idx] is True:
                 response_clean = response.strip().upper()
                 if response_clean == "YES": all_decisions[original_idx] = False; num_excluded += 1 # Exclude
                 else: all_decisions[original_idx] = True # Keep (NO or garbage or error string)
                 if response_clean != "NO" and response_clean != "YES": logger.debug(f"LLM filter got unexpected answer item {original_idx}: '{response}'. Keeping.")
        num_processed += len(batch_messages)
        if batch_num % 50 == 0 or batch_num == total_batches: logger.info(f"Processed filter batch {batch_num}/{total_batches}...")

    logger.info("--- LLM Filtering COMPLETE ---")
    kept_count = sum(1 for status in all_decisions.values() if status)
    pass_rate = (kept_count / len(items_to_process) * 100) if items_to_process else 0
    logger.info(f"   Items marked as candidates (kept): {kept_count:,} ({pass_rate:.2f}%)")
    logger.info(f"   Items confidently excluded as non-MIC: {num_excluded:,}")
    return all_decisions


# --- Function to Save Tagged Data ---
def save_tagged_data(all_original_items, candidate_status, output_dir, file_pattern, batch_size=50000):
    """Saves all items, adding the 'is_mic_event_candidate' flag."""
    logger.info(f"--- Saving Tagged Data to {output_dir} ---")
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_num = 1; items_in_batch = 0
    file_path = output_dir / file_pattern.format(batch_num=batch_num)
    outfile = open(file_path, 'wt', encoding='utf-8')
    logger.info(f"Writing to batch file: {file_path.name}")
    saved_count = 0; excluded_count_saving = 0
    try:
        for i, item_data in enumerate(tqdm(all_original_items, desc="Saving Tagged Items", unit="item", dynamic_ncols=True)):
            is_candidate = candidate_status.get(i, True) # Default keep if index missing
            item_data_copy = item_data.copy()
            item_data_copy['is_mic_event_candidate'] = is_candidate
            # No scores to add in this version
            json_line = json.dumps(item_data_copy); outfile.write(json_line + '\n'); items_in_batch += 1; saved_count += 1
            if not is_candidate: excluded_count_saving +=1
            if items_in_batch >= batch_size and i < len(all_original_items) - 1:
                outfile.close(); logger.info(f"Completed batch file: {file_path.name}")
                batch_num += 1; items_in_batch = 0
                file_path = output_dir / file_pattern.format(batch_num=batch_num)
                outfile = open(file_path, 'wt', encoding='utf-8'); logger.info(f"Writing to batch file: {file_path.name}")
    except Exception as e: logger.error(f"Error saving tagged data: {e}", exc_info=True)
    finally:
        if outfile and not outfile.closed: outfile.close(); logger.info(f"Closed final batch file: {file_path.name}")
    logger.info(f"--- Tagged Data Saving COMPLETE ---")
    logger.info(f"   Total items saved: {saved_count:,}")
    logger.info(f"   Items marked as non-candidate (excluded): {excluded_count_saving:,}")
    kept_saving = saved_count - excluded_count_saving
    logger.info(f"   Items marked as candidate (kept): {kept_saving:,} ({(kept_saving/saved_count*100) if saved_count else 0:.2f}%)")


# --- Main Execution Orchestration (Filtering Session) ---
def main_filtering_session():
    overall_start_time = time.time()
    logger.info("\n=====================================================")
    logger.info("====== Starting MIC Pre-Filtering Pipeline (Local Qwen3) =====")
    logger.info(f"====== Start Time: {datetime.now()} ======")
    logger.info("=====================================================\n")

    # Phase 1: Load Data
    logger.info("--- Phase 1: Loading Data ---")
    phase1_start = time.time()
    all_items_loaded = load_data_only( STEP1_INPUT_DIR, strip_proquest_header )
    phase1_end = time.time(); logger.info(f"Phase 1 (Data Load) finished in {(phase1_end - phase1_start)/60:.2f} minutes.")
    if not all_items_loaded: logger.error("No data loaded. Aborting."); return
    logger.info(f"Loaded {len(all_items_loaded):,} items for filtering.")

    # Phase 2: Load Filter LLM
    logger.info("\n--- Phase 2: Loading Filter LLM ---")
    phase2_start = time.time()
    try: llm_model, llm_tokenizer = load_filter_llm_model_and_tokenizer(MODEL_ID)
    except FileNotFoundError: return # Exit if model dir not found
    except Exception as e: logger.error(f"Fatal: Filter LLM load failed. {e}", exc_info=True); return
    phase2_end = time.time(); logger.info(f"Phase 2 (Filter LLM Load) finished in {phase2_end - phase2_start:.2f} seconds.")
    pad_id = llm_tokenizer.pad_token_id; eos_id = llm_tokenizer.eos_token_id
    # Handle potential None values defensively
    if pad_id is None: pad_id = eos_id
    if pad_id is None or eos_id is None: logger.error("CRITICAL: Could not determine valid pad/eos token IDs."); return
    FILTER_GEN_KWARGS = { "max_new_tokens": MAX_NEW_TOKENS_FILTER, "do_sample": False, "pad_token_id": int(pad_id), "eos_token_id": int(eos_id), }
    logger.info(f"Filter LLM Generation Kwargs configured: {FILTER_GEN_KWARGS}")

    # Phase 3: Run LLM Filtering
    logger.info("\n--- Phase 3: Running LLM Filtering ---")
    phase3_start = time.time()
    candidate_status_dict = run_llm_filtering( llm_model, llm_tokenizer, all_items_loaded, LLM_FILTER_BATCH_SIZE, FILTER_GEN_KWARGS )
    phase3_end = time.time(); logger.info(f"Phase 3 (LLM Filtering) finished in {(phase3_end - phase3_start)/60:.2f} minutes.")
    if not candidate_status_dict: logger.error("LLM Filtering returned empty status. Aborting."); return

    # Cleanup Filter LLM
    logger.info("Cleaning up Filter LLM model and tokenizer...")
    del llm_model, llm_tokenizer; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache(); logger.info("Cleared CUDA cache.")
    logger.info("Filter LLM cleanup complete.")

    # Phase 4: Save Tagged Data
    logger.info("\n--- Phase 4: Saving Tagged Data ---")
    phase4_start = time.time()
    save_tagged_data(all_items_loaded, candidate_status_dict, TAGGED_OUTPUT_DIR, TAGGED_OUTPUT_FILE_PATTERN)
    phase4_end = time.time(); logger.info(f"Phase 4 (Saving) finished in {(phase4_end - phase4_start)/60:.2f} minutes.")

    # Final Cleanup
    del all_items_loaded, candidate_status_dict; gc.collect(); logger.info("Cleaned up final data.")
    overall_end_time = time.time()
    logger.info("\n=====================================================")
    logger.info("====== MIC Pre-Filtering Pipeline Finished ===========")
    logger.info(f"====== End Time: {datetime.now()} ======")
    logger.info(f"====== Total execution time: {(overall_end_time - overall_start_time)/60:.2f} minutes ======")
    logger.info(f"====== Tagged data saved to: {TAGGED_OUTPUT_DIR} ======")
    logger.info("=====================================================\n")


# --- Main Execution ---
if __name__ == "__main__":
    logger.info("--- Script Execution Started (Local Qwen Pre-Filtering) ---")
    # Ensure input directory exists
    if not STEP1_INPUT_DIR.exists() or not STEP1_INPUT_DIR.is_dir():
        logger.error(f"Input directory missing: {STEP1_INPUT_DIR}"); logger.error("Pipeline aborting.")
    else:
        # Ensure CUDA is available
        if not torch.cuda.is_available(): logger.error("CUDA not available. Aborting.")
        # Check compute capability
        elif torch.cuda.get_device_capability(0)[0] < 7: logger.warning("GPU compute capability < 7.0 (Volta). BF16/FP16 support may vary.")
        main_filtering_session() # Run the filtering session
    logger.info("--- Script Execution Finished ---")

# --- REMINDER: SESSION 2 uses the data saved in TAGGED_OUTPUT_DIR ---
