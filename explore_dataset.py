from datasets import load_dataset
import json

def explore():
    print("Loading Pangasinan dataset...")
    ds = load_dataset("PLTAT/alpaca_pangasinan")
    
    print("Dataset keys:", ds.keys())
    if 'train' in ds:
        print(f"Number of rows in train: {len(ds['train'])}")
        print("\n--- SAMPLE 1 ---")
        print(json.dumps(ds['train'][0], indent=2))
        print("\n--- SAMPLE 2 ---")
        print(json.dumps(ds['train'][10], indent=2))

if __name__ == "__main__":
    explore()
