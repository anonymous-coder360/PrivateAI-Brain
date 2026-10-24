"""
RAG + Model Integration for PrivateAI
Combines local retrieval with llama.cpp generation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from llama_cpp import Llama
from rag import RAGPipeline, LocalVectorStore, MemoryStore, LocalEmbedder


class PrivateAIWithRAG:
    """PrivateAI model with RAG augmentation using llama.cpp"""

    def __init__(
        self,
        model_path: str,
        rag_store_path: str = "./rag_store",
        memory_path: str = "./memory_store",
        n_ctx: int = 2048,
        n_threads: int = 4,
    ):
        # Load tokenizer for prompt formatting (optional, we use simple templates)
        # Load llama.cpp model
        self.model = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=4,
            n_gpu_layers=0,
            verbose=False,
        )

        # Initialize RAG
        self.rag = RAGPipeline()
        self.rag.vector_store = LocalVectorStore(rag_store_path)
        self.rag.memory = MemoryStore(memory_path)
        self.rag.embedder = LocalEmbedder()

    def _build_prompt(self, user_query: str, use_rag: bool = True) -> str:
        """Build prompt with optional RAG context"""
        if use_rag:
            return self.rag.get_context_for_prompt(user_query)
        return f"<|system|>\nYou are PrivateAI, a helpful local AI assistant.\n<|user|>\n{user_query}\n<|assistant|>\n"

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.9,
        use_rag: bool = True,
    ) -> str:
        """Generate response with optional RAG"""
        full_prompt = self._build_prompt(prompt, use_rag)
        
        output = self.model(
            full_prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=1.2,
            repeat_penalty_tokens=64,
            stop=["<|user|>", "<|system|>", "<|endoftext|>"],
            echo=False,
        )
        return output['choices'][0]['text'].strip()

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        use_rag: bool = True,
    ):
        """Generate streaming response with optional RAG"""
        full_prompt = self._build_prompt(prompt, use_rag)
        
        output = self.model(
            full_prompt,
            max_tokens=512,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=1.2,
            repeat_penalty_tokens=64,
            stop=["<|user|>", "<|system|>", "<|endoftext|>"],
            echo=False,
            stream=True,
        )
        
        for chunk in output:
            yield chunk['choices'][0]['text']

    def chat(self, user_msg: str, use_rag: bool = True) -> str:
        """Single chat turn with memory"""
        response = self.generate(user_msg, use_rag=use_rag)
        self.rag.add_conversation(user_msg, response)
        return response

    def chat_stream(self, user_msg: str, use_rag: bool = True):
        """Streaming chat with memory"""
        full_response = ""
        for chunk in self.generate_stream(user_msg, use_rag=use_rag):
            full_response += chunk
            yield chunk
        self.rag.add_conversation(user_msg, full_response)

    def add_knowledge(self, documents: list):
        """Add documents to knowledge base"""
        self.rag.add_documents(documents)

    def add_knowledge_from_directory(self, directory: str):
        """Ingest all documents from directory"""
        from ingest import ingest_directory
        docs = ingest_directory(directory)
        self.rag.add_documents(docs)

    def stats(self) -> dict:
        """Get system stats"""
        return {
            "vector_store": self.rag.vector_store.stats(),
            "model_path": getattr(self.model, 'model_path', 'unknown'),
        }


def demo():
    """Demo with trained model"""
    model_path = r"D:\PrivateAI-Brain\normal-brain\privateai-v2\models\privateai-normal-early-q8.gguf"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Train and convert to GGUF first.")
        return

    print("Loading PrivateAI with RAG...")
    ai = PrivateAIWithRAG(model_path)

    # Add knowledge
    ai.add_knowledge([
        {"text": "PrivateAI runs locally on your device. No data leaves without consent.", "source": "privacy.md"},
        {"text": "The Normal Brain handles chat. Complex Brain does math. Coder Brain writes code.", "source": "arch.md"},
    ])

    print("Stats:", ai.stats())

    # Test queries
    queries = [
        "How does PrivateAI protect my privacy?",
        "What are the three brains?",
        "Can it run offline?",
    ]

    for q in queries:
        print(f"\nUser: {q}")
        response = ai.chat(q, use_rag=True)
        print(f"AI: {response}")


if __name__ == "__main__":
    demo()