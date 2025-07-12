from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset
import torch

# --- Configuration ---
model_name = "microsoft/phi-2"
dataset_path = "./sample_finetune_data.jsonl"
output_dir = "./phi2-script-generator-lora"
lora_r = 16
lora_alpha = 32
lora_target_modules = ["Wqkv", "out_proj", "fc1", "fc2"]
lora_dropout = 0.05
use_4bit_quantization = True

# Training arguments
per_device_train_batch_size = 1
gradient_accumulation_steps = 4
learning_rate = 2e-4
num_train_epochs = 1
logging_steps = 10
save_steps = 50

# --- GPU Check ---
if not torch.cuda.is_available():
    print("WARNING: CUDA not available, training will be very slow on CPU!")
else:
    print("CUDA is available! Training will run on GPU.")

device_map = "auto"

# --- BitsAndBytes Configuration ---
bnb_config = None
if use_4bit_quantization and torch.cuda.is_available():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    print("Using 4-bit quantization (BitsAndBytes).")
elif use_4bit_quantization:
    print("WARNING: 4-bit quantization selected but CUDA is not available. Quantization will be skipped.")

# --- Load Tokenizer and Model ---
print(f"Loading tokenizer for '{model_name}'...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("Set pad_token to eos_token.")

print(f"Loading base model '{model_name}'...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map,
    trust_remote_code=True
)

if use_4bit_quantization and torch.cuda.is_available():
    model = prepare_model_for_kbit_training(model)
    print("Model prepared for k-bit training.")

# --- LoRA Configuration ---
lora_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    target_modules=lora_target_modules,
    lora_dropout=lora_dropout,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
print("LoRA model created.")
model.print_trainable_parameters()

# --- Load Dataset ---
print(f"Loading dataset from '{dataset_path}'...")
try:
    dataset = load_dataset("json", data_files={"train": dataset_path}, split="train")
    print(f"Dataset loaded successfully. Number of examples: {len(dataset)}")
    print(f"First example: {dataset[0]}")
except Exception as e:
    print(f"ERROR: Failed to load dataset from {dataset_path}. Check the path and format.")
    print(f"Details: {e}")
    exit()

# --- Training Arguments ---
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=learning_rate,
    num_train_epochs=num_train_epochs,
    logging_steps=logging_steps,
    save_steps=save_steps,
    fp16=False if not torch.cuda.is_available() else (not (bnb_config and bnb_config.bnb_4bit_compute_dtype == torch.bfloat16)),
    bf16=False if not torch.cuda.is_available() else (bnb_config and bnb_config.bnb_4bit_compute_dtype == torch.bfloat16 and torch.cuda.is_bf16_supported()),
    optim="paged_adamw_8bit" if use_4bit_quantization and torch.cuda.is_available() else "adamw_torch",
    save_total_limit=2,
    report_to="none"
)

# --- Define the Formatting Function ---
def formatting_func(example):
    return example["text"]

# --- SFTTrainer Initialization (with all fixes) ---
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    formatting_func=formatting_func,
    peft_config=lora_config,
)

# --- Start Training ---
print("Starting fine-tuning...")
try:
    trainer.train()
    print("Fine-tuning completed successfully.")

    final_adapter_path = f"{output_dir}/final_adapter"
    trainer.model.save_pretrained(final_adapter_path)
    tokenizer.save_pretrained(final_adapter_path)
    print(f"LoRA adapter and tokenizer saved to {final_adapter_path}")
    print("\nTo use the fine-tuned model in main.py, update the ADAPTER_PATH and set USE_FINE_TUNED_MODEL = True.")

except Exception as e:
    print(f"An error occurred during training: {e}")
    import traceback
    traceback.print_exc()