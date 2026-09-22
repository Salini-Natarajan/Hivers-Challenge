import json
import os
from datasets import load_dataset

def main():
    print("Downloading public dataset from HuggingFace...")
    # We use a popular customer support dataset
    # This dataset contains intents, user utterances (incoming), and responses (reply)
    dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train")
    
    # Shuffle and select 100 random examples to keep the project lightweight
    dataset = dataset.shuffle(seed=42).select(range(100))
    
    print("Formatting and splitting data...")
    formatted_data = []
    
    for i, row in enumerate(dataset):
        # 80% for retrieval (train), 20% for evaluation (test)
        split = "train" if i < 80 else "test"
        
        # We extract the customer's instruction and the bot's response
        formatted_data.append({
            "id": str(i + 1),
            "split": split,
            "intent": row.get("intent", "general"),
            "incoming": row.get("instruction", ""),
            "reply": row.get("response", "")
        })
        
    os.makedirs("data", exist_ok=True)
    out_path = "data/dataset.json"
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(formatted_data, f, indent=2)
        
    print(f"Successfully saved {len(formatted_data)} items to {out_path}")
    print(f"- 80 items for the Retrieval Vector DB")
    print(f"- 20 items for the Evaluation Set")

if __name__ == "__main__":
    main()
