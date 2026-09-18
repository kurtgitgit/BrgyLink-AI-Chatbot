import os
import sys
import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

# Limit PyTorch CPU threads to 4 to match i5-7200U without freezing OS
torch.set_num_threads(4)

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "local_model_lora")
LOG_FILE = os.path.join(os.path.dirname(__file__), "train_local.log")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
INTENTS_FILE = os.path.join(os.path.dirname(__file__), "data", "intents_brgylink_curated.json")
# Keep public-service training authoritative. Set to "1" only for an explicitly
# separate experimental language model; generic Alpaca data is not barangay policy.
USE_GENERAL_LANGUAGE_DATA = os.getenv("USE_GENERAL_LANGUAGE_DATA") == "1"

def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
        f.flush()

def extract_ilocano_response(raw_output):
    """Extracts direct Ilocano response from TaCo dataset output if present."""
    if "Response in Ilocano:" in raw_output:
        parts = raw_output.split("Response in Ilocano:")
        return parts[-1].strip()
    return raw_output.strip()

def load_jsonl_dataset(path, max_samples=500, is_ilocano=False):
    """Parses JSONL format line by line safely."""
    items = []
    if not os.path.exists(path):
        log(f"Warning: file not found: {path}")
        return items

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    inst = entry.get("instruction", "").strip()
                    inp = entry.get("input", "").strip()
                    out = entry.get("output", "").strip()

                    if is_ilocano:
                        out = extract_ilocano_response(out)

                    if inst and out:
                        items.append({
                            "instruction": inst,
                            "input": inp if inp != "nan" and inp != "<walang input>" else "",
                            "output": out
                        })
                except json.JSONDecodeError:
                    continue

        if len(items) > max_samples:
            items = random.sample(items, max_samples)
    except Exception as e:
        log(f"Error reading {path}: {e}")

    return items

def load_training_data():
    items = []

    # 1. Load Barangay FAQ data (repeated 5 times for high prioritization)
    if os.path.exists(INTENTS_FILE):
        try:
            with open(INTENTS_FILE, "r", encoding="utf-8") as f:
                intents_data = json.load(f)
                faq_count = 0
                for intent in intents_data.get("intents", []):
                    responses = intent.get("responses", [])
                    resp_text = responses[0] if responses else "Hello! How can I assist you with barangay matters?"
                    for pattern in intent.get("patterns", []):
                        for _ in range(5):
                            items.append({
                                "instruction": pattern,
                                "input": "",
                                "output": resp_text
                            })
                            faq_count += 1
            log(f"✅ Loaded {faq_count} weighted barangay FAQ patterns.")
        except Exception as e:
            log(f"Warning loading intents.json: {e}")

    if not USE_GENERAL_LANGUAGE_DATA:
        random.shuffle(items)
        log(f"🔥 Total verified BrgyLink FAQ instruction pairs: {len(items)}")
        return items

    # 2. Experimental general-language data (opt-in only)
    pangasinan_items = load_jsonl_dataset(os.path.join(DATA_DIR, "alpaca_pangasinan.json"), max_samples=500)
    log(f"✅ Loaded {len(pangasinan_items)} Pangasinan examples.")
    items.extend(pangasinan_items)

    # 3. Load Ilocano (JSONL)
    ilocano_items = load_jsonl_dataset(os.path.join(DATA_DIR, "alpaca_ilocano_taco.json"), max_samples=500, is_ilocano=True)
    log(f"✅ Loaded {len(ilocano_items)} Ilocano examples.")
    items.extend(ilocano_items)

    # 4. Load Filipino / Tagalog (JSONL)
    filipino_items = load_jsonl_dataset(os.path.join(DATA_DIR, "alpaca_filipino_cleaned.json"), max_samples=500)
    log(f"✅ Loaded {len(filipino_items)} Filipino/Tagalog examples.")
    items.extend(filipino_items)

    random.shuffle(items)
    log(f"🔥 Total balanced multilingual instruction pairs: {len(items)}")
    return items


class InstructDataset(Dataset):
    def __init__(self, items, tokenizer, max_len=256):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []

        system_prompt = (
            "You are BrgyLink AI, a friendly and knowledgeable barangay virtual assistant. "
            "You understand and respond accurately in Tagalog, Pangasinan, Ilocano, and English."
        )

        for item in items:
            inst = item.get("instruction", "").strip()
            inp = item.get("input", "").strip()
            out = item.get("output", "").strip()
            if not inst or not out:
                continue

            user_content = f"{inst}\n{inp}".strip() if inp else inst
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": out}
            ]

            try:
                prompt_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                self.samples.append(prompt_text)
            except Exception:
                pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def main():
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n=== Resuming BrgyLink AI Local CPU Training ===\n")

    log(f"Loading base model & tokenizer: {MODEL_NAME} from local cache...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )

    # Load existing checkpoint if available to continue training, else initialize fresh
    adapter_file = os.path.join(OUTPUT_DIR, "adapter_model.safetensors")
    if os.path.exists(adapter_file):
        from peft import PeftModel
        log(f"🔄 Resuming from saved LoRA checkpoint: {OUTPUT_DIR}")
        model = PeftModel.from_pretrained(model, OUTPUT_DIR, is_trainable=True)
    else:
        log("✨ Initializing fresh LoRA adapter...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none"
        )
        model = get_peft_model(model, peft_config)
    trainable_params, all_params = model.get_nb_trainable_parameters()
    log(f"Trainable params: {trainable_params:,} / {all_params:,} ({100 * trainable_params / all_params:.2f}%)")

    # Prepare balanced dataset
    raw_data = load_training_data()
    dataset = InstructDataset(raw_data, tokenizer, max_len=256)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Optimizer & Scheduler
    epochs = 1
    total_steps = len(dataloader) * epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=total_steps)

    log(f"🚀 Starting training: {total_steps} steps on 4 CPU threads...")
    model.train()
    step = 0
    running_loss = 0.0

    for epoch in range(epochs):
        for batch in dataloader:
            step += 1
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            if step % 5 == 0 or step == 1:
                avg_loss = running_loss / (5 if step >= 5 else 1)
                log(f"[Step {step}/{total_steps}] Loss: {avg_loss:.4f} | Progress: {100 * step / total_steps:.1f}%")
                running_loss = 0.0

            # Save checkpoint every 100 steps
            if step % 100 == 0:
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                model.save_pretrained(OUTPUT_DIR)
                tokenizer.save_pretrained(OUTPUT_DIR)
                log(f"💾 Checkpoint saved to {OUTPUT_DIR} (Step {step})")

    # Final Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    log(f"🎉 Training complete! Multilingual Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
