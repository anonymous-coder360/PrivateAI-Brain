"""
Local RAG + Memory System for PrivateAI
Privacy-first: all data stays on device
"""

import json
import os
import pickle
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False
    print("sentence-transformers not installed. Run: pip install sentence-transformers")


class LocalVectorStore:
    """Simple local vector store using numpy (no external DB needed)"""

    def __init__(self, store_path: str = "./rag_store"):
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)

        self.embeddings_file = self.store_path / "embeddings.npy"
        self.metadata_file = self.store_path / "metadata.json"
        self.index_file = self.store_path / "index.pkl"

        self.embeddings: np.ndarray = None
        self.metadata: List[Dict] = []
        self._load()

    def _load(self):
        if self.embeddings_file.exists():
            self.embeddings = np.load(self.embeddings_file)
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.embeddings = np.zeros((0, 384), dtype=np.float32)
            self.metadata = []

    def _save(self):
        np.save(self.embeddings_file, self.embeddings)
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def add(self, texts: List[str], metadatas: List[Dict], embeddings: np.ndarray):
        """Add documents to store"""
        if self.embeddings.shape[0] == 0:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])

        for i, meta in enumerate(metadatas):
            meta["id"] = len(self.metadata)
            meta["added_at"] = datetime.now().isoformat()
            self.metadata.append(meta)

        self._save()

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Search for similar documents"""
        if self.embeddings.shape[0] == 0:
            return []

        # Cosine similarity
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        doc_norms = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8)
        scores = np.dot(doc_norms, query_norm)

        top_indices = np.argsort(scores)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result["score"] = float(scores[idx])
                results.append(result)

        return results

    def delete(self, doc_id: int) -> bool:
        """Delete a document by ID"""
        if 0 <= doc_id < len(self.metadata):
            # Mark as deleted (soft delete)
            self.metadata[doc_id]["deleted"] = True
            self._save()
            return True
        return False

    def clear(self):
        """Clear all data"""
        self.embeddings = np.zeros((0, 384), dtype=np.float32)
        self.metadata = []
        self._save()

    def stats(self) -> Dict:
        active = sum(1 for m in self.metadata if not m.get("deleted", False))
        return {
            "total_documents": len(self.metadata),
            "active_documents": active,
            "embedding_dim": self.embeddings.shape[1] if self.embeddings.shape[0] > 0 else 0,
        }


class LocalEmbedder:
    """Local embedding model using sentence-transformers"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not HAS_ST:
            raise RuntimeError("sentence-transformers not installed")
        self.model = SentenceTransformer(model_name)
        # Use get_sentence_embedding_dimension for newer versions
        try:
            self.dim = self.model.get_sentence_embedding_dimension()
        except AttributeError:
            self.dim = self.model.get_embedding_dimension()

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings"""
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


class MemoryStore:
    """Local conversation memory with user consent"""

    def __init__(self, store_path: str = "./memory_store"):
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        self.file = self.store_path / "conversations.jsonl"

    def add(self, user_msg: str, assistant_msg: str, metadata: Dict = None):
        """Add a conversation turn"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "user": user_msg,
            "assistant": assistant_msg,
            "metadata": metadata or {},
        }
        with open(self.file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def get_recent(self, n: int = 10) -> List[Dict]:
        """Get recent conversations"""
        if not self.file.exists():
            return []
        with open(self.file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return [json.loads(line) for line in lines[-n:]]

    def search(self, query: str, n: int = 5) -> List[Dict]:
        """Simple text search in memory"""
        if not self.file.exists():
            return []
        results = []
        query_lower = query.lower()
        with open(self.file, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                if query_lower in entry["user"].lower() or query_lower in entry["assistant"].lower():
                    results.append(entry)
        return results[-n:]

    def clear(self):
        """Clear all memory"""
        if self.file.exists():
            self.file.unlink()

    def export(self, path: str):
        """Export memory for backup/transfer"""
        import shutil
        shutil.copy(self.file, path)

    def import_from(self, path: str):
        """Import memory from backup"""
        import shutil
        shutil.copy(path, self.file)


class RAGPipeline:
    """Complete RAG pipeline: embed -> store -> retrieve -> augment"""

    def __init__(
        self,
        embedder: LocalEmbedder = None,
        vector_store: LocalVectorStore = None,
        memory: MemoryStore = None,
    ):
        self.embedder = embedder or LocalEmbedder()
        self.vector_store = vector_store or LocalVectorStore()
        self.memory = memory or MemoryStore()

    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Add documents to RAG store
        documents: [{"text": "...", "source": "file.pdf", "metadata": {...}}, ...]
        """
        texts = [doc["text"] for doc in documents]
        metadatas = [
            {
                "source": doc.get("source", "unknown"),
                "text": doc["text"][:500],  # Store preview
                **doc.get("metadata", {}),
            }
            for doc in documents
        ]

        embeddings = self.embedder.encode(texts)
        self.vector_store.add(texts, metadatas, embeddings)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve relevant documents for query"""
        query_emb = self.embedder.encode([query])[0]
        return self.vector_store.search(query_emb, top_k)

    def augment_prompt(self, user_query: str, top_k: int = 3) -> str:
        """Augment user prompt with retrieved context"""
        results = self.retrieve(user_query, top_k)

        if not results:
            return user_query

        context_parts = []
        for r in results:
            source = r.get("source", "unknown")
            text = r.get("text", "")
            score = r.get("score", 0)
            context_parts.append(f"[Source: {source}, Relevance: {score:.2f}]\n{text}")

        context = "\n\n---\n\n".join(context_parts)

        augmented = f"""Use the following context to answer the question. If the context doesn't help, say so.

Context:
{context}

Question: {user_query}

Answer:"""

        return augmented

    def add_conversation(self, user_msg: str, assistant_msg: str):
        """Store conversation in memory"""
        self.memory.add(user_msg, assistant_msg)

    def get_context_for_prompt(self, user_query: str, include_memory: bool = True) -> str:
        """Build full prompt with RAG context + recent memory"""
        parts = []

        # RAG context
        rag_prompt = self.augment_prompt(user_query)
        if rag_prompt != user_query:
            parts.append(rag_prompt)

        # Recent memory (last 3 turns)
        if include_memory:
            recent = self.memory.get_recent(3)
            if recent:
                memory_text = "Recent conversation:\n"
                for turn in recent:
                    memory_text += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
                parts.append(memory_text)

        if parts:
            return "\n\n".join(parts) + f"\n\nQuestion: {user_query}\nAnswer:"
        return user_query


def demo():
    """Quick demo of RAG system"""
    print("Initializing RAG system...")
    pipeline = RAGPipeline()

    # Add sample documents
    docs = [
        {
            "text": "PrivateAI is a privacy-first AI ecosystem that runs locally on user devices. It uses specialized brains for different tasks.",
            "source": "privateai_docs.md",
            "metadata": {"category": "overview"},
        },
        {
            "text": "The Normal Brain handles daily conversations, general questions, and common tasks. It's trained on diverse multilingual data.",
            "source": "privateai_docs.md",
            "metadata": {"category": "architecture"},
        },
        {
            "text": "The Complex Brain handles mathematics, science, and reasoning. The Coder Brain handles programming tasks.",
            "source": "privateai_docs.md",
            "metadata": {"category": "architecture"},
        },
        {
            "text": "All user data stays on device. Optional cloud features require explicit consent. No data leaves the device without permission.",
            "source": "privacy_policy.md",
            "metadata": {"category": "privacy"},
        },
    ]

    print("Adding documents...")
    pipeline.add_documents(docs)

    # Test retrieval
    print("\nTesting retrieval...")
    results = pipeline.retrieve("What brains does PrivateAI have?")
    for r in results:
        print(f"  Score: {r['score']:.3f} | Source: {r['source']} | {r['text'][:80]}...")

    # Test augmented prompt
    print("\nAugmented prompt:")
    prompt = pipeline.get_context_for_prompt("How does PrivateAI protect privacy?")
    print(prompt[:500] + "...")

    # Test memory
    print("\nTesting memory...")
    pipeline.add_conversation("What is PrivateAI?", "PrivateAI is a local AI system...")
    pipeline.add_conversation("How does it work?", "It uses three specialized brains...")
    recent = pipeline.memory.get_recent(2)
    for turn in recent:
        print(f"  User: {turn['user'][:50]}...")
        print(f"  Assistant: {turn['assistant'][:50]}...")

    print("\nRAG system ready!")


if __name__ == "__main__":
    demo()