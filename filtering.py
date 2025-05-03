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
STEP1_INPUT_DIR = Path("../Amartya_tasker/structured_parsed_chunks_combined")
TAGGED_OUTPUT_DIR = Path("./filtered_tagged_chunks")
TAGGED_OUTPUT_FILE_PATTERN = "tagged_chunk_batch_{batch_num}.jsonl"
FILTER_STATS_CSV = Path("./qwen_filter_stats.csv")

# --- Model Configuration ---
MODEL_ID = "../Qwen3-1.7B"
LLM_FILTER_BATCH_SIZE = 64
MAX_CONTEXT_LEN_FILTER = 2048
MAX_NEW_TOKENS_FILTER = 700 # Increased slightly more for potentially longer checklist reasoning

# --- Text Cleaning & Preparation ---
HEADER_PATTERN = re.compile(r'^Search Strategy.*?Results: \d+\s*(?=Document \d+ of \d+|\n\n|$)', re.DOTALL | re.MULTILINE | re.IGNORECASE)
METADATA_PATTERNS = re.compile(r'^(Author|URL|Publication title|Copyright|Document ID|ProQuest document ID|Database|Last updated|Source type|Language of publication|ISSN|Publication subject|Country of publication|Place of publication|Publisher|Section|Publication date|Publication year|Pages|Title|Company / organization|Location|Subject|Links|Abstract|Document \d+ of \d+):.*$', re.IGNORECASE | re.MULTILINE)
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
WHITESPACE_PATTERN = re.compile(r'\s+')
def strip_proquest_header(text):
    if not isinstance(text, str): return ""
    cleaned_text, num_subs = HEADER_PATTERN.subn('', text, count=1)
    if num_subs == 0: cleaned_text = METADATA_PATTERNS.sub('', text)
    doc_match = re.search(r'Document \d+ of \d+', cleaned_text)
    if doc_match and doc_match.start() < 100: cleaned_text = cleaned_text[doc_match.end():]
    return cleaned_text.strip()

def prepare_text_for_filter_llm(header_stripped_text):
    if not isinstance(header_stripped_text, str): return ""
    try:
        text = URL_PATTERN.sub(' ', header_stripped_text)
        text = WHITESPACE_PATTERN.sub(' ', text).strip()
        char_limit = MAX_CONTEXT_LEN_FILTER * 5
        return text[:char_limit]
    except Exception as e: logger.warning(f"Cleaning error (filter LLM): {e}. Text: {header_stripped_text[:100]}..."); return ""

# --- LLM Model Loading ---
def load_filter_llm_model_and_tokenizer(model_id_path):
    """Loads the specified LLM using BF16/FP16 and device_map."""
    model_path = Path(model_id_path)
    if not model_path.exists() or not model_path.is_dir():
         logger.error(f"Model directory not found at: {model_path}")
         raise FileNotFoundError(f"Model directory not found: {model_path}")
    try:
        logger.info(f"--- Loading Filter LLM & Tokenizer ({model_path}) ---")
        if not torch.cuda.is_available(): logger.error("CUDA not available."); raise RuntimeError("CUDA not available")
        logger.info(f"Found {torch.cuda.device_count()} GPU(s).")
        target_dtype = torch.float16 # Force FP16 for V100
        logger.info(f"Loading model in FP16...")
        model = AutoModelForCausalLM.from_pretrained( model_path, torch_dtype=target_dtype, device_map="auto", trust_remote_code=True )
        logger.info(f"Filter LLM model loaded with dtype: {model.dtype}.")
        logger.info(f"Filter LLM model device map: {model.hf_device_map}")
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side='left')
        logger.info("Tokenizer loaded.")
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None: tokenizer.pad_token_id = tokenizer.eos_token_id; logger.info(f"Tokenizer pad_token_id set to eos_token_id ({tokenizer.pad_token_id}).")
            else: logger.warning("Tokenizer lacks pad/eos. Setting pad_token_id to 0."); tokenizer.pad_token_id = 0
        else: logger.info(f"Tokenizer has pad_token_id: {tokenizer.pad_token_id}.")
        logger.info(f"Padding side: '{tokenizer.padding_side}'.")
        model.eval(); logger.info("--- Filter LLM and tokenizer ready ---")
        return model, tokenizer
    except Exception as e: logger.error(f"Filter LLM load failed: {str(e)}", exc_info=True); raise

# --- LLM Prompt Engineering for Filtering (Checklist Approach) ---
def format_filter_prompt(text_chunk):
    """Creates the prompt asking the LLM to follow a checklist and classify."""
    # <<< REVISED INSTRUCTIONS with Checklist >>>
    instructions = """You are a defense analyst assessing article relevance. Analyze the article text below by following this checklist step-by-step to determine if the article can be definitively excluded as a non-MIC event. A Militarized Interstate Confrontation (MIC) for this purpose involves military forces of one country causing fatalities to military forces of another country.

**Checklist:**
1.  **Non-MIC Topic Check:** Is the article CLEARLY about unrelated topics like sports, finance, domestic politics (without international military clashes), entertainment, science, non-military accidents, or weather/disasters? (Answer Yes/No)
2.  **Casualty Check:** Does the article mention any deaths, fatalities, or casualties? (Answer Yes/No)
3.  **Military Actor Check:** Does the article involve military/militia/armed state forces (e.g., army, troops, soldiers, police *in a conflict role*, militants if state-backed)? (Answer Yes/No)
4.  **Causality Check:** If casualties and military actors are present, does the text suggest the military actors *caused* the casualties in a conflict situation? (Answer Yes/No/NA)
5.  **Interstate Check:** If criteria 2, 3, and 4 are met, does the conflict involve actors explicitly identified or strongly implied as belonging to **two different countries/states**? (Answer Yes/No/NA)
6.  **State Actor Causality Check:** If criteria 2-5 are met, does the text indicate that the military forces of one country caused the casualties to the military forces of the *other* country? (Answer Yes/No/NA)

**Decision:**
*   If the answer to Check 1 is **YES**, the article is **NOT** an MIC, the decision is made to remove and we are done here no need to check further.
*   If the answer to Check 1 is **NO**, but the answer to *any* of Checks 2, 3, 4, 5, or 6 is **NO** (where applicable), the article is **NOT** clearly an MIC described fully in the text (it might be related but lacks key details for our definition), the decision is made to remove and we are done here no need to check further.
*   Only if Check 1 is NO AND Checks 2, 3, 4, 5, 6 are ALL YES or  if Check 1 and 2 are NO AND Checks 3, 4, 5, 6 are ALL YES  should you consider it potentially a relevant MIC event for further review.

Based on your checklist analysis:
If you are **highly confident** the article can be excluded based on the checklist (e.g., Check 1 is YES, or subsequent checks are NO), your final answer **MUST** be the single word: "REMOVE".
Otherwise, if the checklist suggests it *might* be a relevant MIC event OR if you are unsure about any step, your final answer **MUST** be the single word: "KEEP".

Article Text:
{context}

Final Answer (ONLY REMOVE or KEEP, do not include anything else, final answer should be just one word):"""
    user_content = instructions.format(context=text_chunk)
    messages = [{"role": "user", "content": user_content}]
    return messages

# --- Data Loading ---
def load_data_only(directory, header_stripper):
    """Loads data, strips headers, returns items."""
    logger.info(f"--- Loading Data ({directory}) ---")
    all_items = []
    if not directory.exists() or not directory.is_dir(): logger.error(f"Input directory not found: {directory}"); return []
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
                            raw_text = data["raw_chunk_text"]; article_body = header_stripper(raw_text)
                            if not article_body: continue
                            item = { "raw_text": raw_text, "article_body": article_body, "metadata": { "source_file": data.get("original_file"), "original_subdir": data.get("original_subdir"), "representative_year": data.get("representative_year"), "chunk_index_in_file": data.get("chunk_index_in_file"), "inferred_date": data.get("inferred_date"), "publication_date": data.get("publication_date") } }
                            all_items.append(item)
                    except json.JSONDecodeError: logger.warning(f"Skipping invalid JSON line {line_num} in {file_path.name}")
                    except Exception as e: logger.warning(f"Error processing line {line_num} in {file_path.name}: {e}", exc_info=False)
        except Exception as e: logger.error(f"Failed read/process file {file_path}: {e}", exc_info=True)
    logger.info(f"--- Initial Data Load COMPLETE ---"); logger.info(f"   Total raw chunks read: {total_chunks_read:,}"); logger.info(f"   Valid items loaded: {len(all_items):,}")
    return all_items

# --- LLM Filter Verification Function (Checklist Prompt) ---
def run_filter_verification(model, tokenizer, gen_kwargs):
    """Runs the filter LLM on examples, checks for REMOVE/KEEP after <think>."""
    logger.info("--- Running Filter LLM Verification (Checklist Prompt) ---")
    samples = {
        "Potential MIC": "Reports emerged on Tuesday detailing a border clash between troops from Country A and Country B near the disputed checkpoint XY. Initial accounts suggest three soldiers from Country A were killed during the firefight, while Country B acknowledged suffering 'some casualties'.",
        "Likely Non-MIC (Sports)": "The Lions secured a stunning victory over the Eagles last night with a final score of 28-24. Quarterback Johnson threw three touchdown passes, including the game-winner in the final minute. Fans celebrated wildly as the team clinched the division title."
    }
    results = {}
    is_dataparallel = isinstance(model, torch.nn.DataParallel)
    model_device = next(model.parameters()).device

    for name, text in samples.items():
        logger.info(f"\nVerifying: {name}")
        logger.info(f"Input Text:\n{text}")
        prompt_messages = format_filter_prompt(text) # Uses the checklist prompt
        final_answer = "VERIFICATION_ERROR"
        try:
            input_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True) # Enable thinking
            tokenized_input = tokenizer(input_text, return_tensors="pt").to(model_device)
            with torch.no_grad():
                generate_func = model.module.generate if is_dataparallel else model.generate
                outputs = generate_func(input_ids=tokenized_input['input_ids'], attention_mask=tokenized_input['attention_mask'], max_new_tokens=300, do_sample=False, pad_token_id=gen_kwargs['pad_token_id'], eos_token_id=gen_kwargs['eos_token_id']) # More tokens for verification
                input_length = tokenized_input['input_ids'].shape[1]; generated_tokens = outputs[:, input_length:]
                raw_response = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            logger.info(f"Raw Output (may include think): '{raw_response}'")
            think_end_tag = "</think>"; think_end_index = raw_response.find(think_end_tag)
            final_answer_text = raw_response[think_end_index + len(think_end_tag):].strip() if think_end_index != -1 else raw_response.strip()
            logger.info(f"Extracted final answer: '{final_answer_text}'")
            # Use uppercase for reliable comparison
            results[name] = final_answer_text.upper()
            del outputs, tokenized_input
        except Exception as e: logger.error(f"Error during verification for '{name}': {e}"); results[name] = "VERIFICATION_ERROR"

    logger.info("--- Filter Verification Complete ---")
    # Expected results based on the *new* prompt asking for REMOVE/KEEP
    logger.info(f"Expected 'Potential MIC' -> KEEP (Result: {results.get('Potential MIC', 'ERROR')})")
    logger.info(f"Expected 'Likely Non-MIC (Sports)' -> REMOVE (Result: {results.get('Likely Non-MIC (Sports)', 'ERROR')})")
    mic_correct = results.get('Potential MIC') == 'KEEP'
    non_mic_correct = results.get('Likely Non-MIC (Sports)') == 'REMOVE'
    if mic_correct and non_mic_correct: logger.info("Verification results match expectations."); return True
    else: logger.warning("Verification results DO NOT match expectations! Check model/prompt/extraction."); return False


# --- LLM Filtering Execution (Revised Output Parsing) ---
def run_llm_filtering(model, tokenizer, items_to_process, batch_size, gen_kwargs):
    """Uses LLM to classify items, extracting REMOVE/KEEP after <think>."""
    logger.info(f"--- LLM Filtering Stage ({len(items_to_process)} items) ---")
    logger.info(f"Batch Size: {batch_size}")
    all_decisions = {}
    num_processed = 0; num_excluded = 0
    is_dataparallel = isinstance(model, torch.nn.DataParallel)
    model_device = next(model.parameters()).device

    prompts_and_indices = []
    for idx, item in enumerate(items_to_process):
        cleaned_body = prepare_text_for_filter_llm(item["article_body"])
        if cleaned_body: prompts_and_indices.append({"messages": format_filter_prompt(cleaned_body), "original_index": idx})
        else: all_decisions[idx] = True # Keep empty

    if not prompts_and_indices: logger.error("No valid items to filter."); return {}
    total_batches = (len(prompts_and_indices) + batch_size - 1) // batch_size
    logger.info(f"Total batches for LLM filtering: {total_batches}")

    for i in tqdm(range(0, len(prompts_and_indices), batch_size), desc="LLM Filtering", unit="batch"):
        batch_data = prompts_and_indices[i : i+batch_size]
        batch_messages = [item['messages'] for item in batch_data]
        batch_original_indices = [item['original_index'] for item in batch_data]
        tokenized_inputs = None; batch_num = i // batch_size + 1

        try:
            # Thinking enabled by default
            batch_inputs_text = [ tokenizer.apply_chat_template( p, tokenize=False, add_generation_prompt=True ) for p in batch_messages ]
            tokenized_inputs = tokenizer( batch_inputs_text, return_tensors="pt", padding=True, truncation=True, max_length=MAX_CONTEXT_LEN_FILTER, return_attention_mask=True ).to(model_device)
        except Exception as e: logger.error(f"Tokenization error batch {batch_num}/{total_batches}: {e}", exc_info=True); all_decisions.update({idx: True for idx in batch_original_indices}); continue

        raw_responses = ["ERROR"] * len(batch_messages)
        try:
            with torch.no_grad():
                generate_func = model.module.generate if is_dataparallel else model.generate
                outputs = generate_func( input_ids=tokenized_inputs['input_ids'], attention_mask=tokenized_inputs['attention_mask'], **gen_kwargs )
                input_length = tokenized_inputs['input_ids'].shape[1]; generated_tokens = outputs[:, input_length:]
                raw_responses = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True); del outputs
        except RuntimeError as e:
            if "out of memory" in str(e).lower(): logger.warning(f"OOM filter batch {batch_num}. Marking KEEP.")
            else: logger.error(f"Runtime error filter batch {batch_num}: {e}", exc_info=False)
            all_decisions.update({idx: True for idx in batch_original_indices}) # Keep on error
        except Exception as e: logger.error(f"Unexpected error filter batch {batch_num}: {e}", exc_info=False); all_decisions.update({idx: True for idx in batch_original_indices}) # Keep on error
        finally:
             if tokenized_inputs is not None: del tokenized_inputs
             gc.collect()

        # Process responses: Extract answer after </think> and check for REMOVE/KEEP
        for idx, raw_response in enumerate(raw_responses):
            original_idx = batch_original_indices[idx]
            if original_idx not in all_decisions: # Only process if no prior error
                 think_end_tag = "</think>"
                 think_end_index = raw_response.find(think_end_tag)
                 final_answer_text = ""
                 if think_end_index != -1: final_answer_text = raw_response[think_end_index + len(think_end_tag):].strip()
                 else: final_answer_text = raw_response.strip()

                 response_clean = final_answer_text.upper()
                 # <<< UPDATED Logic: Check for REMOVE, otherwise KEEP >>>
                 if response_clean == "REMOVE":
                     all_decisions[original_idx] = False # Mark for REMOVAL
                     num_excluded += 1
                 else:
                     all_decisions[original_idx] = True # KEEP (if it's KEEP, garbage, or error)
                     if response_clean != "KEEP": logger.debug(f"Unexpected final answer item {original_idx}: '{final_answer_text}'. Keeping.")

        num_processed += len(batch_messages)
        if batch_num % 50 == 0 or batch_num == total_batches: logger.info(f"Processed filter batch {batch_num}/{total_batches}...")

    logger.info("--- LLM Filtering COMPLETE ---")
    kept_count = sum(1 for status in all_decisions.values() if status)
    pass_rate = (kept_count / len(items_to_process) * 100) if items_to_process else 0
    logger.info(f"   Items marked as candidates (kept): {kept_count:,} ({pass_rate:.2f}%)")
    logger.info(f"   Items confidently excluded as non-MIC: {num_excluded:,}")
    return all_decisions


# --- Function to Save Tagged Data ---
# (No changes needed)
def save_tagged_data(all_original_items, candidate_status, output_dir, file_pattern, batch_size=50000):
    logger.info(f"--- Saving Tagged Data to {output_dir} ---")
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_num = 1; items_in_batch = 0
    file_path = output_dir / file_pattern.format(batch_num=batch_num)
    outfile = open(file_path, 'wt', encoding='utf-8')
    logger.info(f"Writing to batch file: {file_path.name}")
    saved_count = 0; excluded_count_saving = 0
    try:
        for i, item_data in enumerate(tqdm(all_original_items, desc="Saving Tagged Items", unit="item", dynamic_ncols=True)):
            is_candidate = candidate_status.get(i, True)
            item_data_copy = item_data.copy()
            item_data_copy['is_mic_event_candidate'] = is_candidate
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
    logger.info("====== Starting MIC Pre-Filtering Pipeline (Qwen Filter w/ Checklist & Think Handling) =====")
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
    except Exception as e: logger.error(f"Fatal: Filter LLM load failed. {e}", exc_info=True); return
    phase2_end = time.time(); logger.info(f"Phase 2 (Filter LLM Load) finished in {phase2_end - phase2_start:.2f} seconds.")
    pad_id = llm_tokenizer.pad_token_id; eos_id = llm_tokenizer.eos_token_id
    if pad_id is None: pad_id = eos_id
    if pad_id is None or eos_id is None: logger.error("CRITICAL: Could not determine valid pad/eos token IDs."); return
    FILTER_GEN_KWARGS = { "max_new_tokens": MAX_NEW_TOKENS_FILTER, "do_sample": False, "pad_token_id": int(pad_id), "eos_token_id": int(eos_id), }
    logger.info(f"Filter LLM Generation Kwargs configured: {FILTER_GEN_KWARGS}")

    # Phase 2b: Run Filter Verification
    logger.info("\n--- Phase 2b: Running Filter Verification ---")
    verification_passed = run_filter_verification(llm_model, llm_tokenizer, FILTER_GEN_KWARGS)
    if not verification_passed: logger.warning("Filter verification failed! Check model/prompt. Continuing...")

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
    logger.info("--- Script Execution Started (Qwen Pre-Filtering with Checklist Prompt) ---")
    if not STEP1_INPUT_DIR.exists() or not STEP1_INPUT_DIR.is_dir():
        logger.error(f"Input directory missing: {STEP1_INPUT_DIR}"); logger.error("Pipeline aborting.")
    else:
        if not torch.cuda.is_available(): logger.error("CUDA not available. Aborting.")
        elif torch.cuda.get_device_capability(0)[0] < 7: logger.warning("GPU compute capability < 7.0 (Volta). BF16/FP16 support may vary.")
        main_filtering_session()
    logger.info("--- Script Execution Finished ---")

# --- REMINDER: SESSION 2 uses the data saved in TAGGED_OUTPUT_DIR ---
