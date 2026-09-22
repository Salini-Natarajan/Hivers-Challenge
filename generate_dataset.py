import os
import json
import time
from openai import OpenAI

CATEGORIES = [
    "Billing & Refunds",
    "Technical Support (Bugs/Crashes)",
    "Feature Requests",
    "Account Management (Password reset, profiles)",
    "Sales & Pricing Inquiries",
    "Integration & API Help",
    "General Feedback"
]

def generate_batch(client, category, count=10):
    prompt = f"""
You are an expert data generator. Generate {count} highly realistic customer support email pairs for a software company (SaaS). 
Focus on the category: "{category}".

Each pair must contain:
1. An "incoming" email from a customer. Make them varied - some polite, some frustrated, some short, some detailed.
2. A "reply" from the support team. The reply MUST be professional, helpful, polite, and follow standard corporate policies (e.g., granting refunds, escalating bugs, providing instructions).

Format the output EXACTLY as a JSON array of objects, with no markdown formatting or backticks outside the array. 
Example structure:
[
  {{
    "incoming": "customer email text...",
    "reply": "support reply text..."
  }}
]
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert data generator."},
                {"role": "user", "content": prompt}
            ]
        )
        raw_text = response.choices[0].message.content.strip()
        
        # Clean up markdown code blocks if present
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        if raw_text.startswith("```"):
            raw_text = raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
            
        return json.loads(raw_text.strip())
    except Exception as e:
        print(f"Error generating for {category}: {e}")
        return []

def main():
    if "OPENAI_API_KEY" not in os.environ:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        return

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    all_data = []
    total_target = 65
    batch_size = 10
    
    print(f"Generating ~{total_target} synthetic email pairs...")
    
    for i, category in enumerate(CATEGORIES):
        print(f"Generating batch {i+1}/{len(CATEGORIES)}: {category}...")
        batch = generate_batch(client, category, count=batch_size)
        all_data.extend(batch)
        time.sleep(15) # Prevent rate limiting (Free tier is 5 requests per minute)
        
    # Format and split the dataset
    formatted_data = []
    for i, item in enumerate(all_data):
        # 80% train (retrieval), 20% test (evaluation)
        split = "train" if (i % 5) != 0 else "test"
        
        formatted_data.append({
            "id": str(i + 1),
            "split": split,
            "incoming": item.get("incoming", ""),
            "reply": item.get("reply", "")
        })
        
    print(f"Successfully generated {len(formatted_data)} items.")
    
    os.makedirs("data", exist_ok=True)
    with open("data/dataset.json", "w") as f:
        json.dump(formatted_data, f, indent=2)
        
    print("Saved to data/dataset.json")

if __name__ == "__main__":
    main()
