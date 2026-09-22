import json
import os
import chromadb
from openai import OpenAI

class EmailGenerator:
    def __init__(self, dataset_path, model_name="gpt-4o-mini"):
        """
        Initializes the semantic RAG pipeline using ChromaDB for dense vector retrieval.
        Assumes OPENAI_API_KEY is set in the environment.
        """
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model_name = model_name
        
        # Initialize an in-memory ChromaDB vector store
        self.chroma_client = chromadb.Client()
        # Create or get a collection (which automatically uses an embedding model under the hood)
        self.collection = self.chroma_client.get_or_create_collection(name="support_emails")
        
        # Load dataset
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        self.past_emails = [d for d in data if d['split'] == 'train']
        
        # Populate Vector DB
        self._build_index()

    def _build_index(self):
        """Indexes the training emails into ChromaDB."""
        # Check if already populated to save time
        if self.collection.count() > 0:
            return
            
        documents = []
        metadatas = []
        ids = []
        
        for email in self.past_emails:
            # We index the INCOMING email text so we can match against new incoming emails
            documents.append(email['incoming'])
            # Store the reply in metadata so we can retrieve it
            metadatas.append({"reply": email['reply'], "intent": email.get('intent', '')})
            ids.append(email['id'])
            
        # Add to Chroma (this automatically computes dense vector embeddings)
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def _retrieve_similar(self, query, top_k=2):
        """Finds the most semantically similar past emails using Dense Vector Search."""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        retrieved = []
        # Chroma returns lists of lists
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                retrieved.append({
                    "incoming": results['documents'][0][i],
                    "reply": results['metadatas'][0][i]['reply']
                })
        return retrieved

    def generate_reply(self, incoming_email):
        """Generates a reply by synthesizing past strategies (Semantic RAG)."""
        # 1. Retrieve semantically similar past emails
        similar_emails = self._retrieve_similar(incoming_email, top_k=2)
        
        # 2. Construct Prompt for Strategy Extraction & Application
        prompt = "You are an expert customer support agent.\n"
        prompt += "Your task is to write a reply to a new incoming customer email.\n\n"
        
        if similar_emails:
            prompt += "Here are examples of how we successfully resolved similar semantic intents in the past.\n"
            prompt += "Do NOT just copy these blindly. Extract the resolution strategy and apply it to the new email.\n\n"
            for idx, ex in enumerate(similar_emails):
                prompt += f"--- Past Example {idx+1} ---\n"
                prompt += f"Similar Customer Issue:\n{ex['incoming']}\n\n"
                prompt += f"Our Successful Resolution:\n{ex['reply']}\n"
                prompt += "------------------------\n\n"
                
        prompt += "Now, please write a reply to the following NEW customer email:\n"
        prompt += f"New Customer Email:\n{incoming_email}\n\n"
        prompt += "Your Reply:\n"

        # 3. Call LLM (OpenAI)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful customer support agent."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.choices[0].message.content.strip()
