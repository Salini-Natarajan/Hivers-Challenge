import json
import os
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import evaluate as hf_evaluate # HuggingFace evaluate package for BERTScore
from generator import EmailGenerator

class Evaluator:
    def __init__(self, dataset_path, model_name="gpt-4o-mini"):
        # We initialize the generator (which uses OpenAI for drafting)
        self.generator = EmailGenerator(dataset_path, model_name)
        self.dataset_path = dataset_path
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.test_emails = [d for d in data if d['split'] == 'test']
        
        # Initialize deterministic semantic metrics
        print("Loading SentenceTransformer for semantic embedding similarity...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("Loading BERTScore metric...")
        self.bertscore = hf_evaluate.load("bertscore")

    def _calculate_cosine_similarity(self, gen_text, ref_text):
        """Calculates cosine similarity between sentence embeddings."""
        # Encode both sentences
        embeddings = self.embedder.encode([gen_text, ref_text])
        # Calculate similarity (0 to 1)
        sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(sim)

    def run_evaluation(self):
        print(f"\nStarting deterministic evaluation on {len(self.test_emails)} test examples...")
        
        results = []
        generated_replies = []
        reference_replies = []
        
        # 1. Generate all replies
        for idx, item in enumerate(self.test_emails):
            print(f"Generating reply for test example {idx+1}/{len(self.test_emails)}...")
            generated = self.generator.generate_reply(item['incoming'])
            
            generated_replies.append(generated)
            reference_replies.append(item['reply'])
            
            results.append({
                "id": item['id'],
                "incoming": item['incoming'],
                "ground_truth": item['reply'],
                "generated": generated
            })

        # 2. Calculate BERTScore for all items
        print("\nCalculating BERTScore...")
        # Note: using distilbert to keep it fast and lightweight
        bert_results = self.bertscore.compute(
            predictions=generated_replies, 
            references=reference_replies, 
            model_type="distilbert-base-uncased"
        )
        
        # 3. Calculate Embedding Similarity & Compile
        print("Calculating Dense Embedding Cosine Similarities...")
        total_cosine = 0
        
        for i in range(len(results)):
            # Calculate cosine similarity for this specific pair
            cos_sim = self._calculate_cosine_similarity(generated_replies[i], reference_replies[i])
            total_cosine += cos_sim
            
            # Extract BERTScore F1 for this specific pair
            f1_score = bert_results['f1'][i]
            
            results[i]['metrics'] = {
                "cosine_similarity": cos_sim,
                "bert_f1_score": f1_score
            }
            print(f"Item {results[i]['id']} -> Cosine: {cos_sim:.3f}, BERT-F1: {f1_score:.3f}")

        # 4. Overall System Averages
        n = len(results)
        overall_metrics = {
            "average_cosine_similarity": total_cosine / n if n > 0 else 0,
            "average_bert_f1": np.mean(bert_results['f1']) if n > 0 else 0
        }
        
        report = {
            "metrics": overall_metrics,
            "details": results
        }
        
        with open("evaluation_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
            
        print(f"\nEvaluation complete. Report saved to evaluation_report.json")
        print(f"System Average Cosine Similarity: {overall_metrics['average_cosine_similarity']:.3f} (closer to 1.0 is better)")
        print(f"System Average BERT F1-Score: {overall_metrics['average_bert_f1']:.3f} (closer to 1.0 is better)")
        
        return report

if __name__ == "__main__":
    if "OPENAI_API_KEY" not in os.environ:
        print("ERROR: Please set the OPENAI_API_KEY environment variable.")
        exit(1)
        
    evaluator = Evaluator("data/dataset.json")
    evaluator.run_evaluation()
