"""
Test fine-tuned model with llama.cpp locally
Run AFTER converting to GGUF in Colab
"""

from llama_cpp import Llama
import os

# Path to your GGUF model (after downloading from Colab)
MODEL_PATH = r"D:\PrivateAI-Brain\normal-brain\privateai-v2\models\privateai-normal-early-q8.gguf"

if not os.path.exists(MODEL_PATH):
    print(f"Model not found at {MODEL_PATH}")
    print("Download from Colab after conversion, place in models/ folder")
    exit(1)

print("Loading model...")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=4,
    n_gpu_layers=0,  # CPU only
    verbose=False,
)

# Chat format for SmolLM
def format_prompt(user_msg):
    return f"<|system|>\nYou are PrivateAI, a helpful local AI assistant.\n<|user|>\n{user_msg}\n<|assistant|>\n"

# Test queries
queries = [
    "What is PrivateAI?",
    "What are the three brains?",
    "How does it protect privacy?",
    "Write a Python hello world.",
    "Explain machine learning simply.",
]

print("\n" + "="*50)
for q in queries:
    prompt = format_prompt(q)
    output = llm(
        prompt,
        max_tokens=256,
        temperature=0.7,
        top_p=0.9,
        repeat_penalty=1.2,
        repeat_penalty_tokens=64,
        stop=["<|user|>", "<|system|>", "<|endoftext|>"],
        echo=False,
    )
    response = output['choices'][0]['text'].strip()
    print(f"Q: {q}")
    print(f"A: {response}")
    print("-"*40)

print("\nInteractive mode (type 'exit' to quit):")
while True:
    user = input("\nYou: ")
    if user.lower() in ['exit', 'quit']:
        break
    prompt = format_prompt(user)
    output = llm(prompt, max_tokens=256, temperature=0.7, top_p=0.9, repeat_penalty=1.2, repeat_penalty_tokens=64, stop=["<|user|>", "<|system|>"])
    print(f"AI: {output['choices'][0]['text'].strip()}")