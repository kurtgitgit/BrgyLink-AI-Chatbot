import json
import random
import os

def load_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def inject_data():
    datasets_dir = "datasets"
    intents_path = "intents.json"
    
    # Load existing intents
    with open(intents_path, 'r', encoding='utf-8') as f:
        intents_data = json.load(f)
        
    files = {
        "pangasinan": "alpaca_pangasinan.json",
        "ilocano": "alpaca_ilocano_taco.json",
        "filipino": "alpaca_filipino_cleaned.json"
    }
    
    samples_per_language = 50
    
    for lang, filename in files.items():
        filepath = os.path.join(datasets_dir, filename)
        if not os.path.exists(filepath):
            print(f"Skipping {lang}, file not found: {filepath}")
            continue
            
        print(f"Processing {lang}...")
        data = load_jsonl(filepath)
        
        # Sample random entries
        num_samples = min(samples_per_language, len(data))
        sampled = random.sample(data, num_samples)
        
        for i, item in enumerate(sampled):
            instruction = item.get("instruction", "")
            inp = item.get("input", "")
            output = item.get("output", "")
            
            # Combine instruction and input as pattern
            pattern = instruction
            if inp:
                pattern += " " + inp
                
            if not pattern or not output:
                continue
                
            new_intent = {
                "tag": f"alpaca_{lang}_{i}",
                "patterns": [pattern],
                "responses": [output]
            }
            intents_data["intents"].append(new_intent)
            
    # Save updated intents
    with open(intents_path, 'w', encoding='utf-8') as f:
        json.dump(intents_data, f, indent=4, ensure_ascii=False)
        
    print(f"Successfully injected samples into {intents_path}. Total intents now: {len(intents_data['intents'])}")

if __name__ == "__main__":
    inject_data()
