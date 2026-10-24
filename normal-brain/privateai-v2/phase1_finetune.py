"""
Phase 1: Fine-tune SmolLM-360M on PrivateAI data (LoRA)
Run in Google Colab (T4 GPU, ~30 min)
"""

# =========================
# 1. INSTALL DEPENDENCIES
# =========================
!pip install -q transformers accelerate peft bitsandbytes trl datasets huggingface_hub

# =========================
# 2. UPLOAD YOUR DATA
# =========================
# In Colab UI: Left sidebar -> Folder icon -> Upload train_final.jsonl
# Or from Drive:
# from google.colab import drive
# drive.mount('/content/drive')
# !cp /content/drive/MyDrive/PrivateAI/train_final.jsonl /content/

# =========================
# 3. CONFIGURATION
# =========================
import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

# Model config
MODEL_NAME = "HuggingFaceTB/SmolLM-360M"  # or "Qwen/Qwen2-0.5B"
OUTPUT_DIR = "./smollm-privateai"
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
EPOCHS = 1
BATCH_SIZE = 4
GRAD_ACCUM = 4
LR = 2e-4
MAX_LENGTH = 1024

# =========================
# 4. LOAD & PREPARE DATA
# =========================
def load_privateai_data(filepath):
    """Load and format PrivateAI dataset"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                messages = item.get('messages', [])
                
                # Format as chat
                text = ""
                for msg in messages:
                    role = msg.get('role', '')
                    content = msg.get('content', '').strip()
                    if not content:
                        continue
                    if role == 'user':
                        text += f"<|user|>\n{content}\n"
                    elif role == 'assistant':
                        text += f"<|assistant|>\n{content}\n"
                
                if text:
                    text = "<|system|>\nYou are PrivateAI, a helpful local AI assistant.\n" + text + "<|endoftext|>"
                    data.append({"text": text})
    
    return Dataset.from_list(data)

print("Loading dataset...")
dataset = load_privateai_data("/content/train_final.jsonl")
print(f"Total examples: {len(dataset)}")

# Split
dataset = dataset.train_test_split(test_size=0.02, seed=42)
train_dataset = dataset['train']
eval_dataset = dataset['test']

# =========================
# 5. LOAD MODEL & TOKENIZER
# =========================
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Add special tokens if needed
special_tokens = {"additional_special_tokens": ["<|user|>", "<|assistant|>", "<|system|>"]}
tokenizer.add_special_tokens(special_tokens)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
model.resize_token_embeddings(len(tokenizer))

# Enable gradient checkpointing
model.gradient_checkpointing_enable()
model.config.use_cache = False

# =========================
# 6. SETUP LoRA
# =========================
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# =========================
# 7. TOKENIZE DATA
# =========================
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )

print("Tokenizing...")
train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# =========================
# 8. TRAINING ARGS
# =========================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    weight_decay=0.01,
    warmup_steps=50,
    lr_scheduler_type="cosine",
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=True,
    report_to="none",
    remove_unused_columns=False,
    dataloader_pin_memory=False,
)

# =========================
# 9. TRAIN
# =========================
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

print("Starting training...")
trainer.train()

# =========================
# 10. SAVE & MERGE
# =========================
print("Saving LoRA adapter...")
model.save_pretrained(f"{OUTPUT_DIR}-lora")
tokenizer.save_pretrained(f"{OUTPUT_DIR}-lora")

# Merge for inference
print("Merging LoRA...")
merged_model = model.merge_and_unload()
merged_model.save_pretrained(f"{OUTPUT_DIR}-merged")
tokenizer.save_pretrained(f"{OUTPUT_DIR}-merged")

# =========================
# 11. EXPORT TO GGUF (for llama.cpp)
# =========================
# Install llama.cpp conversion script
!pip install -q gguf
!git clone -q https://github.com/ggerganov/llama.cpp /content/llama.cpp

# Convert (run after training)
# !python /content/llama.cpp/convert_hf_to_gguf.py /content/smollm-privateai-merged \
#     --outfile /content/privateai-normal-q4.gguf --outtype q4_k_m

print("Done! Model saved to:")
print(f"  LoRA adapter: {OUTPUT_DIR}-lora")
print(f"  Merged: {OUTPUT_DIR}-merged")
print("\nTo convert to GGUF (run after training):")
print(f"  python /content/llama.cpp/convert_hf_to_gguf.py /content/{OUTPUT_DIR}-merged --outfile /content/privateai-normal-q4.gguf --outtype q4_k_m")