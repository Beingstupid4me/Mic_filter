from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class QwenChatbot:
    def __init__(self, model_name="../Qwen3-1.7B"):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)
            self.model.to('cuda' if torch.cuda.is_available() else 'cpu')  # Load on GPU if available
            self.history = []
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model/tokenizer: {e}")
            raise

    def generate_response(self, user_input):
        try:
            messages = self.history + [{"role": "user", "content": user_input}]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False)
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            response_ids = self.model.generate(**inputs, max_new_tokens=512)[0]
            response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

            # Update history and return response
            self.history.extend([{"role": "user", "content": user_input}, {"role": "assistant", "content": response}])
            return response.split("Assistant:")[-1].strip()  # Extract assistant reply
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Sorry, I couldn't generate a response."

if __name__ == "__main__":
    chatbot = QwenChatbot()

    user_input_1 = "How many r's in strawberries?"
    print(f"User: {user_input_1}")
    response_1 = chatbot.generate_response(user_input_1)
    print(f"Bot: {response_1}")
    print("----------------------")

    user_input_2 = "Then, how many r's in blueberries? /no_think"
    print(f"User: {user_input_2}")
    response_2 = chatbot.generate_response(user_input_2)
    print(f"Bot: {response_2}")
    print("----------------------")

    user_input_3 = "Really? /think"
    print(f"User: {user_input_3}")
    response_3 = chatbot.generate_response(user_input_3)
    print(f"Bot: {response_3}")
