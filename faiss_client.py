import os
import pickle
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️  faiss-cpu not installed. Run: pip install faiss-cpu")


class FAISSClient:
    """Local vector search using FAISS (Facebook AI Similarity Search)"""
    
    def __init__(self, index_path: str = "data/faiss_index", dimension: int = 384):
        """Initialize FAISS index"""
        if not FAISS_AVAILABLE:
            raise RuntimeError("faiss-cpu is not installed.")
        
        self.index_path = index_path
        self.dimension = dimension
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create directory if doesn't exist
        os.makedirs(index_path, exist_ok=True)
        
        # Initialize or load index
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata_store = []  # Store metadata separately
        
        self._load_if_exists()
    
    def _load_if_exists(self):
        """Load existing index from disk"""
        index_file = os.path.join(self.index_path, "index.faiss")
        metadata_file = os.path.join(self.index_path, "metadata.pkl")
        
        if os.path.exists(index_file) and os.path.exists(metadata_file):
            try:
                self.index = faiss.read_index(index_file)
                with open(metadata_file, 'rb') as f:
                    self.metadata_store = pickle.load(f)
                print(f"✅ Loaded FAISS index with {len(self.metadata_store)} vectors")
            except Exception as e:
                print(f"⚠️  Could not load FAISS index: {e}")
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for texts"""
        embeddings = self.embedding_model.encode(texts)
        return embeddings.tolist()
    
    def add_chunks(self, chunks: List[Any]) -> None:
        """Add document chunks to FAISS"""
        vectors = []
        
        for chunk in chunks:
            embedding = self.create_embeddings([chunk.text])[0]
            
            metadata = {
                'chunk_id': chunk.chunk_id,
                'text': chunk.text[:1500],
                **chunk.metadata,
                'type': 'document_chunk'
            }
            
            vectors.append(np.array(embedding, dtype=np.float32))
            self.metadata_store.append(metadata)
        
        # Add vectors to index
        if vectors:
            vectors_array = np.array(vectors, dtype=np.float32)
            self.index.add(vectors_array)
            print(f"✅ Added {len(vectors)} vectors to FAISS")
        
        # Save to disk
        self._save_index()
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar chunks"""
        query_embedding = np.array(
            self.create_embeddings([query])[0], 
            dtype=np.float32
        )
        
        # If index is empty
        if self.index.ntotal == 0:
            return []
        
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1), 
            min(top_k, self.index.ntotal)
        )
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx == -1:  # Invalid index
                continue
            
            metadata = self.metadata_store[idx].copy()
            metadata['score'] = float(1 / (1 + distance))  # Convert distance to similarity
            results.append(metadata)
        
        return results
    
    def _save_index(self):
        """Save index to disk"""
        try:
            index_file = os.path.join(self.index_path, "index.faiss")
            metadata_file = os.path.join(self.index_path, "metadata.pkl")
            
            faiss.write_index(self.index, index_file)
            with open(metadata_file, 'wb') as f:
                pickle.dump(self.metadata_store, f)
            
            print(f"✅ FAISS index saved to {self.index_path}")
        except Exception as e:
            print(f"❌ Failed to save FAISS index: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        return {
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'index_type': 'FlatL2'
        }
