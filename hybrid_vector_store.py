from typing import List, Dict, Any, Optional
from pinecone_client import PineconeClient
from faiss_client import FAISSClient

class HybridVectorStore:
    """Hybrid vector search: Pinecone (primary) + FAISS (fallback)"""
    
    def __init__(self, use_pinecone: bool = True, use_faiss: bool = True):
        self.use_pinecone = use_pinecone
        self.use_faiss = use_faiss
        
        self.pinecone_client = None
        self.faiss_client = None
        
        if use_pinecone:
            try:
                self.pinecone_client = PineconeClient()
                print("✅ Pinecone initialized")
            except Exception as e:
                print(f"⚠️  Pinecone failed: {e}")
                self.pinecone_client = None
        
        if use_faiss:
            try:
                self.faiss_client = FAISSClient()
                print("✅ FAISS initialized")
            except Exception as e:
                print(f"⚠️  FAISS failed: {e}")
                self.faiss_client = None
    
    def add_chunks(self, chunks: List[Any]) -> None:
        """Add chunks to both Pinecone and FAISS"""
        if self.pinecone_client:
            try:
                self.pinecone_client.upsert_chunks(chunks)
            except Exception as e:
                print(f"❌ Pinecone upsert failed: {e}")
        
        if self.faiss_client:
            try:
                self.faiss_client.add_chunks(chunks)
            except Exception as e:
                print(f"❌ FAISS add failed: {e}")
    
    def search(
        self, 
        query: str, 
        top_k: int = 5,
        use_fallback: bool = True
    ) -> List[Dict]:
        """
        Search with smart fallback:
        1. Try Pinecone (fast, cloud)
        2. If fails and fallback enabled, try FAISS (local)
        """
        results = []
        
        # Try Pinecone first
        if self.pinecone_client:
            try:
                pinecone_results = self.pinecone_client.search(query, top_k)
                results = pinecone_results
                print("✅ Results from Pinecone")
                return results
            except Exception as e:
                print(f"⚠️  Pinecone search failed: {e}")
        
        # Fallback to FAISS
        if use_fallback and self.faiss_client:
            try:
                faiss_results = self.faiss_client.search(query, top_k)
                print("✅ Results from FAISS (fallback)")
                return faiss_results
            except Exception as e:
                print(f"❌ FAISS search also failed: {e}")
        
        return []
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of both backends"""
        return {
            'pinecone': 'active' if self.pinecone_client else 'inactive',
            'faiss': 'active' if self.faiss_client else 'inactive',
            'faiss_stats': self.faiss_client.get_stats() if self.faiss_client else None,
        }
