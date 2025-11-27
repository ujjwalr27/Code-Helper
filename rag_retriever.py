"""
RAG Retriever for Context-Aware Code Remediation
Uses FAISS for efficient similarity search
"""

import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import Optional, List, Dict
import logging

logger = logging.getLogger(__name__)

class RAGRetriever:
    """Retrieval-Augmented Generation component for security guidelines"""
    
    def __init__(self, recipes_dir: str = "recipes", 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize RAG retriever with FAISS index
        
        Args:
            recipes_dir: Directory containing remediation recipe files
            model_name: Sentence transformer model for embeddings
        """
        self.recipes_dir = recipes_dir
        self.model_name = model_name
        
        logger.info(f"Initializing RAG retriever with model: {model_name}")
        
        # Load sentence transformer model
        self.encoder = SentenceTransformer(model_name)
        
        # Load and index recipes
        self.recipes = []
        self.recipe_embeddings = []
        self.index = None
        
        self._load_recipes()
        self._build_index()
        
        logger.info(f"RAG retriever initialized with {len(self.recipes)} recipes")
    
    def _load_recipes(self):
        """Load all recipe files from the recipes directory"""
        if not os.path.exists(self.recipes_dir):
            logger.warning(f"Recipes directory not found: {self.recipes_dir}")
            return
        
        for filename in os.listdir(self.recipes_dir):
            if filename.endswith('.txt'):
                filepath = os.path.join(self.recipes_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    self.recipes.append({
                        'filename': filename,
                        'content': content,
                        'cwe': self._extract_cwe(filename)
                    })
                    logger.info(f"Loaded recipe: {filename}")
                    
                except Exception as e:
                    logger.error(f"Error loading {filename}: {str(e)}")
    
    def _extract_cwe(self, filename: str) -> Optional[str]:
        """Extract CWE identifier from filename if present"""
        # Try to find CWE pattern in filename
        import re
        match = re.search(r'cwe[-_]?(\d+)', filename.lower())
        if match:
            return f"CWE-{match.group(1)}"
        return None
    
    def _build_index(self):
        """Build FAISS index from recipe embeddings"""
        if not self.recipes:
            logger.warning("No recipes to index")
            return
        
        logger.info("Building FAISS index...")
        
        # Generate embeddings for all recipes
        recipe_texts = [r['content'] for r in self.recipes]
        self.recipe_embeddings = self.encoder.encode(
            recipe_texts,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        # Create FAISS index
        dimension = self.recipe_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.recipe_embeddings.astype('float32'))
        
        logger.info(f"FAISS index built with {self.index.ntotal} vectors")
    
    def retrieve(self, cwe: str, language: str, top_k: int = 1) -> Optional[str]:
        """
        Retrieve the most relevant remediation recipe
        
        Args:
            cwe: CWE identifier (e.g., "CWE-89")
            language: Programming language
            top_k: Number of top results to return
            
        Returns:
            Content of the most relevant recipe, or None
        """
        if not self.recipes or self.index is None:
            logger.warning("No recipes available for retrieval")
            return None
        
        # First, try exact CWE match
        for recipe in self.recipes:
            if recipe['cwe'] == cwe:
                logger.info(f"Exact CWE match found: {recipe['filename']}")
                return recipe['content']
        
        # If no exact match, use semantic search
        query = f"{cwe} {language} security vulnerability remediation"
        query_embedding = self.encoder.encode(
            [query],
            convert_to_numpy=True
        ).astype('float32')
        
        # Search in FAISS index
        distances, indices = self.index.search(query_embedding, top_k)
        
        if len(indices[0]) > 0:
            best_idx = indices[0][0]
            best_recipe = self.recipes[best_idx]
            logger.info(f"Retrieved recipe: {best_recipe['filename']} (distance: {distances[0][0]:.4f})")
            return best_recipe['content']
        
        logger.warning("No relevant recipe found")
        return None
    
    def retrieve_multiple(self, cwe: str, language: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve multiple relevant recipes
        
        Returns:
            List of recipe dictionaries with content and metadata
        """
        if not self.recipes or self.index is None:
            return []
        
        query = f"{cwe} {language} security vulnerability remediation"
        query_embedding = self.encoder.encode(
            [query],
            convert_to_numpy=True
        ).astype('float32')
        
        distances, indices = self.index.search(query_embedding, min(top_k, len(self.recipes)))
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            results.append({
                'content': self.recipes[idx]['content'],
                'filename': self.recipes[idx]['filename'],
                'distance': float(dist)
            })
        
        return results
    
    def add_recipe(self, filename: str, content: str):
        """
        Dynamically add a new recipe to the index
        
        Args:
            filename: Name for the recipe file
            content: Recipe content
        """
        # Add to recipes list
        self.recipes.append({
            'filename': filename,
            'content': content,
            'cwe': self._extract_cwe(filename)
        })
        
        # Generate embedding
        embedding = self.encoder.encode(
            [content],
            convert_to_numpy=True
        ).astype('float32')
        
        # Add to index
        if self.index is None:
            dimension = embedding.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
        
        self.index.add(embedding)
        self.recipe_embeddings = np.vstack([self.recipe_embeddings, embedding])
        
        logger.info(f"Added new recipe: {filename}")
    
    def get_stats(self) -> Dict:
        """Get statistics about the RAG retriever"""
        return {
            'total_recipes': len(self.recipes),
            'indexed_vectors': self.index.ntotal if self.index else 0,
            'model': self.model_name,
            'dimension': self.recipe_embeddings.shape[1] if len(self.recipe_embeddings) > 0 else 0
        }