import sys
import os
from generator import EmailGenerator

def main():
    if "OPENAI_API_KEY" not in os.environ:
        print("ERROR: Please set the OPENAI_API_KEY environment variable.")
        print("Example (Windows PowerShell): $env:OPENAI_API_KEY='your-key-here'")
        print("Example (Linux/Mac): export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)
        
    print("Initializing AI Email Assistant (loading dataset and building ChromaDB Vector Index)...")
    try:
        generator = EmailGenerator("data/dataset.json")
    except Exception as e:
        print(f"Error initializing generator: {e}")
        sys.exit(1)
        
    print("\n--- AI Email Support Assistant ---")
    print("Type your incoming customer email below. Type 'exit' or 'quit' to stop.")
    
    while True:
        try:
            print("\n" + "="*50)
            incoming = input("\nIncoming Email:\n> ")
            if incoming.strip().lower() in ['exit', 'quit']:
                break
                
            if not incoming.strip():
                continue
                
            print("\nGenerating reply...")
            reply = generator.generate_reply(incoming)
            
            print("\nSuggested Reply:")
            print("-" * 30)
            print(reply)
            print("-" * 30)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

    print("\nGoodbye!")

if __name__ == "__main__":
    main()
