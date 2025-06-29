import torch
import sentencepiece as spm
from training import TinyLLaMAInference

def main():
    model_path = "tinyllama_final.pt"
    tokenizer_path = "tinyllama_tokenizer.model"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    inference = TinyLLaMAInference(model_path, tokenizer_path, device)

    print("TinyLLaMA Interactive Chat. Type 'quit' or 'exit' to stop.")
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in {"quit", "exit"}:
            print("Exiting.")
            break
        response = inference.generate_text(user_input, max_length=100, temperature=1.0, top_k=100)
        print(f"TinyLLaMA: {response}")

if __name__ == "__main__":
    main()
