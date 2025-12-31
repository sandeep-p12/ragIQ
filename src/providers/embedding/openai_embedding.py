"""OpenAI embedding provider implementation."""

import time
from typing import Dict, List

from openai import APIError, OpenAI, RateLimitError

from src.config.retrieval import EmbeddingConfig
from src.core.interfaces import EmbeddingProvider
from src.utils.env import get_openai_api_key


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider with batching and retry logic."""
    
    def __init__(self, config: EmbeddingConfig = None):
        """Initialize provider with config.
        
        Args:
            config: EmbeddingConfig (defaults to EmbeddingConfig.from_env())
        """
        self.config = config or EmbeddingConfig.from_env()
        api_key = get_openai_api_key()
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        self.client = OpenAI(api_key=api_key)
        self._cache: Dict[str, List[float]] = {}
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts with batching and retry logic.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Check cache first
        uncached_texts = []
        uncached_indices = []
        cached_results = []
        
        for idx, text in enumerate(texts):
            cache_key = f"{self.config.model}:{text}"
            if cache_key in self._cache:
                cached_results.append((idx, self._cache[cache_key]))
            else:
                uncached_texts.append(text)
                uncached_indices.append(idx)
        
        # Embed uncached texts in batches
        if uncached_texts:
            all_embeddings = []
            for i in range(0, len(uncached_texts), self.config.batch_size):
                batch = uncached_texts[i:i + self.config.batch_size]
                batch_indices = uncached_indices[i:i + self.config.batch_size]
                
                embeddings = self._embed_batch_with_retry(batch)
                
                # Cache results
                for text, embedding in zip(batch, embeddings):
                    cache_key = f"{self.config.model}:{text}"
                    self._cache[cache_key] = embedding
                
                all_embeddings.extend((idx, emb) for idx, emb in zip(batch_indices, embeddings))
        else:
            all_embeddings = []
        
        # Combine cached and new results in correct order
        all_results = cached_results + all_embeddings
        all_results.sort(key=lambda x: x[0])
        
        return [emb for _, emb in all_results]
    
    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string.
        
        Args:
            query: Query string to embed
            
        Returns:
            Embedding vector
        """
        cache_key = f"{self.config.model}:{query}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        embedding = self._embed_batch_with_retry([query])[0]
        self._cache[cache_key] = embedding
        return embedding
    
    def _embed_batch_with_retry(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch with exponential backoff retry.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.config.model,
                    input=texts
                )
                return [item.embedding for item in response.data]
            except RateLimitError as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                else:
                    raise
            except APIError as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                else:
                    raise
        
        raise last_error or Exception("Failed to embed texts")
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self._cache.clear()

