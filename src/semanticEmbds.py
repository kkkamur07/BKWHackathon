from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import re


class SemanticMatcher:
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.model = SentenceTransformer(model_name)
        self.tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=100)
        self.pca = PCA(n_components=50)
        
    def preprocess_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def get_embeddings(self, texts: list) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=True)
    
    def get_tfidf_features(self, texts: list, fit: bool = False) -> np.ndarray:
        if fit:
            return self.tfidf.fit_transform(texts).toarray()
        else:
            return self.tfidf.transform(texts).toarray()
    
    def apply_pca(self, embeddings: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            return self.pca.fit_transform(embeddings)
        else:
            return self.pca.transform(embeddings)
    
    def hybrid_similarity(self, input_text: str, reference_texts: list, 
                         semantic_weight: float = 0.7, tfidf_weight: float = 0.3) -> np.ndarray:
        
        preprocessed_input = self.preprocess_text(input_text)
        preprocessed_refs = [self.preprocess_text(t) for t in reference_texts]
        
        input_embedding = self.model.encode([preprocessed_input])
        reference_embeddings = self.model.encode(preprocessed_refs)
        semantic_sim = cosine_similarity(input_embedding, reference_embeddings)[0]
        
        all_texts = [preprocessed_input] + preprocessed_refs
        tfidf_matrix = self.tfidf.fit_transform(all_texts).toarray()
        tfidf_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
        
        hybrid_scores = (semantic_weight * semantic_sim) + (tfidf_weight * tfidf_sim)
        
        return hybrid_scores
    
    def find_best_match(self, input_text: str, reference_texts: list) -> tuple:
        hybrid_scores = self.hybrid_similarity(input_text, reference_texts)
        
        best_idx = np.argmax(hybrid_scores)
        best_score = hybrid_scores[best_idx]
        
        top_3_indices = np.argsort(hybrid_scores)[-3:][::-1]
        top_3_matches = [(reference_texts[i], hybrid_scores[i]) for i in top_3_indices]
        
        return reference_texts[best_idx], best_score, top_3_matches
    
    def classify_rooms(self, input_df: pd.DataFrame, reference_classes: list) -> pd.DataFrame:
        results = []
        
        print(f"Processing {len(input_df)} rooms with semantic matching...")
        print(f"Reference classes: {len(reference_classes)}")
        
        for idx, row in input_df.iterrows():
            room_name = row['Architect room names']
            area = row['Area']
            
            input_with_context = f"{room_name} area {area} square meters"
            best_match, score, top_3 = self.find_best_match(input_with_context, reference_classes)
            
            results.append({
                'Architect room names': room_name,
                'Area': area,
                'Volume': row['Volume'],
                'BKW_name': best_match,
                'similarity_score': score,
                'confidence': 'high' if score > 0.8 else 'medium' if score > 0.6 else 'low',
                'second_best': top_3[1][0] if len(top_3) > 1 else None,
                'second_score': top_3[1][1] if len(top_3) > 1 else None
            })
            
            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/{len(input_df)} rooms")
        
        return pd.DataFrame(results)
    
    def batch_similarity_matrix(self, texts1: list, texts2: list) -> np.ndarray:
        embeddings1 = self.model.encode(texts1)
        embeddings2 = self.model.encode(texts2)
        
        return cosine_similarity(embeddings1, embeddings2)
    
    def get_feature_importance(self, input_text: str, reference_texts: list) -> dict:
        preprocessed_input = self.preprocess_text(input_text)
        preprocessed_refs = [self.preprocess_text(t) for t in reference_texts]
        
        all_texts = [preprocessed_input] + preprocessed_refs
        tfidf_matrix = self.tfidf.fit_transform(all_texts)
        
        feature_names = self.tfidf.get_feature_names_out()
        input_tfidf = tfidf_matrix[0].toarray()[0]
        
        top_features_idx = np.argsort(input_tfidf)[-10:][::-1]
        top_features = {feature_names[i]: input_tfidf[i] for i in top_features_idx if input_tfidf[i] > 0}
        
        return top_features
    
    def analyze_classification_quality(self, results_df: pd.DataFrame) -> dict:
        analysis = {
            'total_rooms': len(results_df),
            'avg_confidence': results_df['similarity_score'].mean(),
            'high_confidence_count': len(results_df[results_df['similarity_score'] > 0.8]),
            'medium_confidence_count': len(results_df[(results_df['similarity_score'] > 0.6) & (results_df['similarity_score'] <= 0.8)]),
            'low_confidence_count': len(results_df[results_df['similarity_score'] <= 0.6]),
            'unique_classes_assigned': results_df['BKW_name'].nunique(),
            'class_distribution': results_df['BKW_name'].value_counts().to_dict()
        }
        return analysis


class AdvancedSemanticMatcher(SemanticMatcher):
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        super().__init__(model_name)
        self.embedding_cache = {}
        
    def get_cached_embedding(self, text: str) -> np.ndarray:
        if text not in self.embedding_cache:
            self.embedding_cache[text] = self.model.encode([text])[0]
        return self.embedding_cache[text]
    
    def ensemble_matching(self, input_text: str, reference_texts: list) -> tuple:
        semantic_scores = self.hybrid_similarity(input_text, reference_texts, 
                                                 semantic_weight=0.6, tfidf_weight=0.4)
        
        tfidf_scores = self.hybrid_similarity(input_text, reference_texts,
                                             semantic_weight=0.3, tfidf_weight=0.7)
        
        pure_semantic = self.hybrid_similarity(input_text, reference_texts,
                                              semantic_weight=1.0, tfidf_weight=0.0)
        
        ensemble_scores = (0.5 * semantic_scores + 0.3 * tfidf_scores + 0.2 * pure_semantic)
        
        best_idx = np.argmax(ensemble_scores)
        return reference_texts[best_idx], ensemble_scores[best_idx]


def classify_from_csv_semantic(
    input_csv_path: str,
    reference_csv_path: str,
    output_csv_path: str = None,
    advanced: bool = False
) -> pd.DataFrame:
    
    print(f"Loading input data from {input_csv_path}...")
    input_df = pd.read_csv(input_csv_path)
    
    print(f"Loading reference data from {reference_csv_path}...")
    reference_df = pd.read_csv(reference_csv_path)
    reference_classes = reference_df['RoomType'].unique().tolist()
    
    print(f"Initializing {'Advanced' if advanced else 'Standard'} Semantic Matcher...")
    if advanced:
        matcher = AdvancedSemanticMatcher()
    else:
        matcher = SemanticMatcher()
    
    print("Starting room classification with hybrid semantic-TF-IDF approach...")
    result_df = matcher.classify_rooms(input_df, reference_classes)
    
    print("\nGenerating classification quality analysis...")
    quality_analysis = matcher.analyze_classification_quality(result_df)
    
    print(f"\nClassification Summary:")
    print(f"  Total rooms classified: {quality_analysis['total_rooms']}")
    print(f"  Average confidence: {quality_analysis['avg_confidence']:.3f}")
    print(f"  High confidence (>0.8): {quality_analysis['high_confidence_count']}")
    print(f"  Medium confidence (0.6-0.8): {quality_analysis['medium_confidence_count']}")
    print(f"  Low confidence (<0.6): {quality_analysis['low_confidence_count']}")
    
    if output_csv_path:
        result_df.to_csv(output_csv_path, index=False)
        print(f"\nResults saved to {output_csv_path}")
    
    return result_df


def compare_semantic_vs_llm(
    input_csv_path: str,
    reference_csv_path: str,
    llm_results_path: str
) -> pd.DataFrame:
    
    print("Running semantic classification...")
    semantic_results = classify_from_csv_semantic(input_csv_path, reference_csv_path)
    
    print("Loading LLM classification results...")
    llm_results = pd.read_csv(llm_results_path)
    
    comparison = pd.DataFrame({
        'Room': semantic_results['Architect room names'],
        'Semantic_Match': semantic_results['BKW_name'],
        'Semantic_Score': semantic_results['similarity_score'],
        'LLM_Match': llm_results['BKW_name'],
        'Agreement': semantic_results['BKW_name'] == llm_results['BKW_name']
    })
    
    agreement_rate = comparison['Agreement'].mean()
    print(f"\nSemantic vs LLM Agreement Rate: {agreement_rate:.2%}")
    
    return comparison