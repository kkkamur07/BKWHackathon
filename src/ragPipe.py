import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import pickle
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentChunker:
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str) -> List[str]:
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk = ' '.join(words[i:i + self.chunk_size])
            chunks.append(chunk)
        
        return chunks
    
    def chunk_dataframe(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        chunked_data = []
        
        for idx, row in df.iterrows():
            text = str(row[text_column])
            chunks = self.chunk_text(text)
            
            for chunk_idx, chunk in enumerate(chunks):
                chunk_row = row.to_dict()
                chunk_row['chunk_text'] = chunk
                chunk_row['chunk_id'] = f"{idx}_{chunk_idx}"
                chunk_row['original_index'] = idx
                chunked_data.append(chunk_row)
        
        return pd.DataFrame(chunked_data)


class VectorStore:
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.index = None
        self.documents = []
        self.metadata = []
    
    def create_index(self, index_type: str = "flat"):
        if index_type == "flat":
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, 100)
        elif index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
    
    def add_vectors(self, embeddings: np.ndarray, documents: List[str], metadata: List[Dict]):
        if self.index is None:
            self.create_index()
        
        if isinstance(self.index, faiss.IndexIVFFlat):
            if not self.index.is_trained:
                self.index.train(embeddings.astype('float32'))
        
        self.index.add(embeddings.astype('float32'))
        self.documents.extend(documents)
        self.metadata.extend(metadata)
        
        logger.info(f"Added {len(embeddings)} vectors to index. Total: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        distances, indices = self.index.search(query_embedding, k)
        return distances[0], indices[0]
    
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        faiss.write_index(self.index, f"{path}.index")
        
        with open(f"{path}.docs", 'wb') as f:
            pickle.dump(self.documents, f)
        
        with open(f"{path}.meta", 'wb') as f:
            pickle.dump(self.metadata, f)
        
        logger.info(f"Saved vector store to {path}")
    
    def load(self, path: str):
        self.index = faiss.read_index(f"{path}.index")
        
        with open(f"{path}.docs", 'rb') as f:
            self.documents = pickle.load(f)
        
        with open(f"{path}.meta", 'rb') as f:
            self.metadata = pickle.load(f)
        
        logger.info(f"Loaded vector store from {path}")


class HybridRetriever:
    def __init__(self, semantic_weight: float = 0.7, keyword_weight: float = 0.3):
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
    
    def bm25_score(self, query: str, document: str) -> float:
        query_terms = set(query.lower().split())
        doc_terms = document.lower().split()
        
        score = 0
        for term in query_terms:
            tf = doc_terms.count(term) / len(doc_terms) if doc_terms else 0
            score += tf
        
        return score
    
    def hybrid_search(self, query: str, vector_distances: np.ndarray, 
                     documents: List[str], k: int = 5) -> List[Tuple[int, float]]:
        
        semantic_scores = 1 / (1 + vector_distances)
        semantic_scores = (semantic_scores - semantic_scores.min()) / (semantic_scores.max() - semantic_scores.min() + 1e-10)
        
        keyword_scores = np.array([self.bm25_score(query, doc) for doc in documents])
        if keyword_scores.max() > 0:
            keyword_scores = keyword_scores / keyword_scores.max()
        
        hybrid_scores = (self.semantic_weight * semantic_scores + 
                        self.keyword_weight * keyword_scores)
        
        top_indices = np.argsort(hybrid_scores)[-k:][::-1]
        results = [(idx, hybrid_scores[idx]) for idx in top_indices]
        
        return results


class RAGPipeline:
    def __init__(self, 
                 model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
                 chunk_size: int = 500,
                 chunk_overlap: int = 50,
                 index_type: str = "flat"):
        
        logger.info(f"Initializing RAG Pipeline with model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.chunker = DocumentChunker(chunk_size, chunk_overlap)
        self.vector_store = VectorStore(embedding_dim=self.model.get_sentence_embedding_dimension())
        self.vector_store.create_index(index_type)
        self.retriever = HybridRetriever()
        
    def ingest_csv(self, csv_path: str, text_column: str, metadata_columns: List[str] = None):
        logger.info(f"Ingesting data from {csv_path}")
        
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} rows")
        
        chunked_df = self.chunker.chunk_dataframe(df, text_column)
        logger.info(f"Created {len(chunked_df)} chunks")
        
        documents = chunked_df['chunk_text'].tolist()
        logger.info("Generating embeddings...")
        embeddings = self.model.encode(documents, show_progress_bar=True, batch_size=32)
        
        if metadata_columns:
            metadata = chunked_df[metadata_columns + ['chunk_id', 'original_index']].to_dict('records')
        else:
            metadata = [{'chunk_id': row['chunk_id'], 'original_index': row['original_index']} 
                       for _, row in chunked_df.iterrows()]
        
        self.vector_store.add_vectors(embeddings, documents, metadata)
        
        return len(chunked_df)
    
    def ingest_documents(self, documents: List[str], metadata: List[Dict] = None):
        logger.info(f"Ingesting {len(documents)} documents")
        
        embeddings = self.model.encode(documents, show_progress_bar=True, batch_size=32)
        
        if metadata is None:
            metadata = [{'doc_id': i} for i in range(len(documents))]
        
        self.vector_store.add_vectors(embeddings, documents, metadata)
    
    def retrieve(self, query: str, k: int = 5, use_hybrid: bool = True) -> List[Dict]:
        logger.info(f"Retrieving top {k} results for query: {query[:50]}...")
        
        query_embedding = self.model.encode([query])[0]
        
        distances, indices = self.vector_store.search(query_embedding, k=k*2 if use_hybrid else k)
        
        if use_hybrid:
            candidate_docs = [self.vector_store.documents[i] for i in indices if i < len(self.vector_store.documents)]
            hybrid_results = self.retriever.hybrid_search(query, distances[:len(candidate_docs)], candidate_docs, k)
            
            results = []
            for rank, (local_idx, score) in enumerate(hybrid_results):
                global_idx = indices[local_idx]
                if global_idx < len(self.vector_store.documents):
                    results.append({
                        'rank': rank + 1,
                        'document': self.vector_store.documents[global_idx],
                        'metadata': self.vector_store.metadata[global_idx],
                        'score': float(score),
                        'distance': float(distances[local_idx])
                    })
        else:
            results = []
            for rank, (idx, dist) in enumerate(zip(indices[:k], distances[:k])):
                if idx < len(self.vector_store.documents):
                    results.append({
                        'rank': rank + 1,
                        'document': self.vector_store.documents[idx],
                        'metadata': self.vector_store.metadata[idx],
                        'distance': float(dist)
                    })
        
        return results
    
    def retrieve_and_aggregate(self, query: str, k: int = 5, 
                              aggregation_method: str = "concatenate") -> Dict:
        results = self.retrieve(query, k)
        
        documents = [r['document'] for r in results]
        metadata_list = [r['metadata'] for r in results]
        
        if aggregation_method == "concatenate":
            context = "\n\n---\n\n".join(documents)
        elif aggregation_method == "summarize":
            context = "\n".join([f"{i+1}. {doc[:200]}..." for i, doc in enumerate(documents)])
        else:
            context = documents[0] if documents else ""
        
        return {
            'query': query,
            'context': context,
            'sources': results,
            'num_sources': len(results)
        }
    
    def save_index(self, path: str):
        self.vector_store.save(path)
        
        config = {
            'model_name': self.model._model_name,
            'chunk_size': self.chunker.chunk_size,
            'chunk_overlap': self.chunker.overlap,
            'embedding_dim': self.vector_store.embedding_dim
        }
        
        with open(f"{path}.config", 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved RAG pipeline to {path}")
    
    def load_index(self, path: str):
        self.vector_store.load(path)
        
        with open(f"{path}.config", 'r') as f:
            config = json.load(f)
        
        logger.info(f"Loaded RAG pipeline from {path}")
        return config


class RoomClassificationRAG(RAGPipeline):
    def __init__(self, reference_csv_path: str):
        super().__init__()
        
        logger.info("Building RAG index for room classification")
        self.reference_df = pd.read_csv(reference_csv_path)
        
        documents = []
        metadata = []
        
        for _, row in self.reference_df.iterrows():
            room_type = row['RoomType']
            
            doc_text = f"Room Type: {room_type}"
            
            for col in self.reference_df.columns:
                if col != 'RoomType':
                    doc_text += f"\n{col}: {row[col]}"
            
            documents.append(doc_text)
            metadata.append(row.to_dict())
        
        self.ingest_documents(documents, metadata)
        logger.info(f"Indexed {len(documents)} room types")
    
    def classify_room(self, room_name: str, area: float, k: int = 3) -> Dict:
        query = f"Classify room: {room_name} with area {area} square meters"
        
        results = self.retrieve(query, k=k)
        
        top_match = results[0] if results else None
        
        return {
            'input_room': room_name,
            'input_area': area,
            'predicted_type': top_match['metadata']['RoomType'] if top_match else None,
            'confidence': top_match['score'] if top_match else 0.0,
            'top_k_matches': [
                {
                    'room_type': r['metadata']['RoomType'],
                    'score': r['score']
                } for r in results
            ]
        }
    
    def classify_from_dataframe(self, input_df: pd.DataFrame) -> pd.DataFrame:
        results = []
        
        for _, row in input_df.iterrows():
            classification = self.classify_room(row['Architect room names'], row['Area'])
            
            results.append({
                'Architect room names': row['Architect room names'],
                'Area': row['Area'],
                'Volume': row['Volume'],
                'BKW_name': classification['predicted_type'],
                'confidence': classification['confidence'],
                'top_3_matches': [m['room_type'] for m in classification['top_k_matches'][:3]]
            })
        
        return pd.DataFrame(results)


def build_rag_index_from_csv(csv_path: str, 
                             text_column: str,
                             save_path: str,
                             metadata_columns: List[str] = None):
    
    rag = RAGPipeline()
    rag.ingest_csv(csv_path, text_column, metadata_columns)
    rag.save_index(save_path)
    
    return rag


def query_rag_index(index_path: str, query: str, k: int = 5):
    
    rag = RAGPipeline()
    rag.load_index(index_path)
    
    results = rag.retrieve_and_aggregate(query, k)
    
    return results