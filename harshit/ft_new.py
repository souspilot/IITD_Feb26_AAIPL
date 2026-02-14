from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only
import os
import json
import torch
import argparse
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

# ---------------------------
# Environment Setup
# ---------------------------
os.environ["WANDB_DISABLED"] = "true"

# ---------------------------
# Model-Specific Conversion
# ---------------------------
def convert_openai_messages(messages, model_name):
    """
    Convert OpenAI/GPT-style messages to model-specific format.
    
    Args:
        messages: List of dicts with 'role' and 'content'
        model_name: Target model name (e.g., 'gemma', 'qwen', 'llama')
    
    Returns:
        Converted messages list
    """
    model_name_lower = model_name.lower()

    # ----------------
    # GEMMA
    # ----------------
    if "gemma" in model_name_lower:
        result = []
        for msg in messages:
            content = msg["content"]
            result.append({
                "role": msg["role"],
                "content": [{"type": "text", "text": content}]
            })
        return result

    # ----------------
    # QWEN
    # ----------------
    elif "qwen" in model_name_lower:
        system_content = ""
        result = []
        
        for msg in messages:
            if msg["role"] == "system":
                # Accumulate system messages
                system_content += msg["content"] + "\n\n"
            elif msg["role"] == "user":
                # Prepend accumulated system content to first user message
                if system_content:
                    result.append({
                        "role": "user",
                        "content": system_content + msg["content"]
                    })
                    system_content = ""  # Clear after using
                else:
                    result.append(msg)
            else:
                # Assistant or other roles
                result.append(msg)
        
        return result

    # ----------------
    # LLAMA (default)
    # ----------------
    return messages

# ---------------------------
# Data Loading
# ---------------------------
def load_data(file_path):
    """
    Load dataset from either:
    - JSONL (OpenAI-style messages per line)
    - JSON (dict-of-dicts format like CoT_collection_en.json)
    """

    data = []

    # -----------------------------
    # JSONL (existing behavior)
    # -----------------------------
    if file_path.endswith(".jsonl"):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)

                if 'body' in record and 'messages' in record['body']:
                    messages = record['body']['messages']
                elif 'messages' in record:
                    messages = record['messages']
                else:
                    print("Warning: Could not find messages, skipping.")
                    continue

                data.append({
                    'messages': messages,
                    'custom_id': record.get('custom_id', '')
                })

    # -----------------------------
    # JSON (CoT_collection format)
    # -----------------------------
    elif file_path.endswith(".json"):
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        count = 0

        for sample_id, record in raw_data.items():

            source = record.get("source", "")
            target = record.get("target", "")
            rationale = record.get("rationale", "")

            # Choose what you want assistant to learn:
            # Option A: Only answer
            # assistant_content = target

            # Option B (recommended for CoT fine-tuning):
            # answer + reasoning
            assistant_content = f"{rationale}\n\nFinal Answer: {target}"

            messages = [
                {"role": "user", "content": source},
                {"role": "assistant", "content": assistant_content}
            ]

            data.append({
                "messages": messages,
                "custom_id": sample_id
            })

            count += 1
            if count >= 1000:
                break


    else:
        raise ValueError("Unsupported file format. Use .json or .jsonl")

    return data


def split_data(data, train_ratio=0.95, val_ratio=0.05, seed=42):
    """Split data into train/val sets"""
    import random
    random.seed(seed)
    
    total = len(data)
    train_size = int(total * train_ratio)
    
    train_data = data[:train_size]
    val_data = data[train_size:]
    random.shuffle(train_data)
    print(f"Total samples: {total}")
    print(f"Train: {len(train_data)} ({len(train_data)/total*100:.1f}%)")
    print(f"Val: {len(val_data)} ({len(val_data)/total*100:.1f}%)")
    
    return train_data, val_data

# ---------------------------
# Prompt Formatting
# ---------------------------
def format_prompt(example, tokenizer, model_name):
    """Convert messages to model-specific format and apply chat template"""
    # Convert to target model format
    messages = convert_openai_messages(example['messages'], model_name)
    
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=False
    )
    
    return {"text": text}

def filter_by_length(example, tokenizer, max_length):
    """Filter examples that exceed max token length"""
    token_count = len(tokenizer(example['text'])['input_ids'])
    return token_count <= max_length

# ---------------------------
# Response-Only Training Setup
# ---------------------------
def get_chat_markers(model_name):
    """Get instruction and response markers for model"""
    model_name_lower = model_name.lower()
    
    if "qwen" in model_name_lower:
        return {
            "instruction_part": "<|im_start|>user\n",
            "response_part": "<|im_start|>assistant\n"
        }
    elif "llama" in model_name_lower:
        return {
            "instruction_part": "<|start_header_id|>user<|end_header_id|>\n\n",
            "response_part": "<|start_header_id|>assistant<|end_header_id|>\n\n"
        }
    elif "gemma" in model_name_lower:
        return {
            "instruction_part": "<start_of_turn>user\n",
            "response_part": "<start_of_turn>model\n"
        }
    
    # Default fallback
    return {
        "instruction_part": "user:",
        "response_part": "assistant:"
    }

# ---------------------------
# Main Training
# ---------------------------
# Parse command line arguments
parser = argparse.ArgumentParser(description='Fine-tune language model with unified config')
parser.add_argument('--config', type=str, required=True,
                   help='Path to config JSON file')
args = parser.parse_args()

# Load unified config
config_file = args.config
print(f"Loading config from {config_file}")

with open(config_file, 'r') as f:
    config = json.load(f)

model_cfg = config['model']
train_cfg = config['training']
data_cfg = config.get('data', {})

# Load data
input_file = data_cfg.get('input_file', 'training_prompts_openai_format.jsonl')
print(f"\nLoading data from {input_file}")

data = load_data(input_file)
train_data, val_data = split_data(
    data,
    train_ratio=data_cfg.get('train_ratio', 0.95),
    val_ratio=data_cfg.get('val_ratio', 0.05),
    seed=data_cfg.get('seed', 42)
)

# Load model and tokenizer
print("\nLoading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_cfg["base_model"],
    max_seq_length=model_cfg["max_seq_length"],
    load_in_4bit=model_cfg.get("load_in_4bit", True),
)

# Apply LoRA
print("Applying LoRA...")
model = FastLanguageModel.get_peft_model(model, **model_cfg["lora"])

# Convert to datasets
train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

# Format prompts
print("\nFormatting prompts...")
model_name = model_cfg["base_model"]

train_dataset = train_dataset.map(
    lambda x: format_prompt(x, tokenizer, model_name),
    batched=False,
    num_proc=os.cpu_count(),
)

val_dataset = val_dataset.map(
    lambda x: format_prompt(x, tokenizer, model_name),
    batched=False
)

# Filter by length
max_length = model_cfg.get("max_seq_length", 8192)
print(f"\nFiltering by max length: {max_length}")
print(f"Original train size: {len(train_dataset)}")

train_dataset = train_dataset.filter(
    lambda x: filter_by_length(x, tokenizer, max_length)
)

print(f"Filtered train size: {len(train_dataset)}")

val_dataset = val_dataset.filter(
    lambda x: filter_by_length(x, tokenizer, max_length)
)

print(f"Filtered val size: {len(val_dataset)}")

# Create trainer
print("\nInitializing trainer...")
training_args = SFTConfig(**train_cfg)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    dataset_text_field="text",
    args=training_args,
)

# Train on responses only
print("\nConfiguring response-only training...")
markers = get_chat_markers(model_name)

trainer = train_on_responses_only(
    trainer,
    instruction_part=markers["instruction_part"],
    response_part=markers["response_part"],
)

# Train
print("\n" + "="*50)
print("Starting training...")
print("="*50)

trainer.train(resume_from_checkpoint=train_cfg.get("resume_from_checkpoint", False))

print("\n✅ Training completed!")

# Save model
output_dir = train_cfg.get('output_dir', 'results')
print(f"\nSaving model to {output_dir}")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("\n" + "="*50)
print("All tasks completed successfully!")
print("="*50)