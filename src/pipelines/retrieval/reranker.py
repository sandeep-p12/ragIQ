"""OpenAI LLM reranker implementation with strict JSON output."""

import json
import time
from typing import Dict, List

from openai import OpenAI
from openai import RateLimitError, APIError
from pydantic import BaseModel, Field, field_validator

from src.config.retrieval import RerankConfig
from src.core.dataclasses import CandidateText, RerankResult
from src.core.interfaces import Reranker
from src.utils.env import get_openai_api_key


class RerankItem(BaseModel):
    """Single reranked item from LLM response."""
    chunk_id: str
    relevance_score: int = Field(ge=0, le=100)
    answerability: str
    key_evidence: List[str] = Field(default_factory=list)
    
    @field_validator('answerability')
    @classmethod
    def validate_answerability(cls, v):
        """Validate answerability is yes or no."""
        if v.lower() not in ('yes', 'no'):
            raise ValueError("answerability must be 'yes' or 'no'")
        return v.lower()


class RerankResponse(BaseModel):
    """Rerank response from LLM."""
    ranked: List[RerankItem]


class OpenAIReranker(Reranker):
    """OpenAI LLM reranker with strict JSON output."""
    
    def __init__(self, config: RerankConfig = None):
        """Initialize reranker.
        
        Args:
            config: RerankConfig (defaults to RerankConfig.from_env())
        """
        self.config = config or RerankConfig.from_env()
        api_key = get_openai_api_key()
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        self.client = OpenAI(api_key=api_key)
        self._cache: Dict[str, List[RerankResult]] = {}
    
    def rerank(
        self,
        query: str,
        candidates: List[CandidateText]
    ) -> List[RerankResult]:
        """Rerank candidates using LLM.
        
        Args:
            query: Query string
            candidates: List of candidate texts to rerank
            
        Returns:
            List of reranked results sorted by relevance_score (descending)
        """
        if not candidates:
            return []
        
        # Limit to max_candidates_to_rerank
        candidates = candidates[:self.config.max_candidates_to_rerank]
        
        # Check cache
        cache_key = self._make_cache_key(query, candidates)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Build prompt
        prompt = self._build_prompt(query, candidates)
        
        # Call LLM with retry
        response = self._call_llm_with_retry(prompt)
        
        # Parse and validate response
        results = self._parse_response(response, candidates)
        
        # Cache results
        self._cache[cache_key] = results
        
        return results
    
    def _build_prompt(self, query: str, candidates: List[CandidateText]) -> str:
        """Build reranking prompt.
        
        Args:
            query: Query string
            candidates: List of candidate texts
            
        Returns:
            Formatted prompt string
        """
        # Format candidates
        candidate_list = []
        for idx, candidate in enumerate(candidates):
            # Trim text snippet
            text_snippet = candidate.text_snippet
            if len(text_snippet) > self.config.max_text_chars_per_candidate:
                # Always include first lines if they contain Document/Page/Section prefix
                lines = text_snippet.split('\n')
                prefix_lines = []
                content_lines = []
                
                for line in lines:
                    if any(marker in line for marker in ['Document:', 'Page:', 'Section:']):
                        prefix_lines.append(line)
                    else:
                        content_lines.append(line)
                
                # Keep prefix lines + truncated content
                content_text = '\n'.join(content_lines)
                if len(content_text) > self.config.max_text_chars_per_candidate - len('\n'.join(prefix_lines)):
                    content_text = content_text[:self.config.max_text_chars_per_candidate - len('\n'.join(prefix_lines)) - 10] + "..."
                
                text_snippet = '\n'.join(prefix_lines + [content_text])
            
            # Format candidate metadata
            metadata = candidate.metadata
            page_span = metadata.get('page_span', metadata.get('page_span_start', 0))
            if isinstance(page_span, (list, tuple)):
                page_str = f"Page {page_span[0]}-{page_span[1]}"
            else:
                page_str = f"Page {page_span}"
            
            candidate_str = f"""Candidate {idx + 1}:
- chunk_id: {candidate.chunk_id}
- section_label: {metadata.get('section_label', 'N/A')}
- header_path: {metadata.get('header_path', 'N/A')}
- page_span: {page_str}
- element_type: {metadata.get('element_type', 'N/A')}
- text_snippet:
{text_snippet}"""
            
            candidate_list.append(candidate_str)
        
        candidates_text = "\n\n".join(candidate_list)
        
        prompt = f"""You are a relevance rater for document retrieval. Rate each candidate chunk's relevance to the query.

Query: {query}

Candidates:
{candidates_text}

Scoring Rubric:
- 100: Directly answers query with explicit evidence
- 70: Highly relevant but partial answer
- 40: Tangential context
- 0: Unrelated

Output ONLY valid JSON in this exact format:
{{
  "ranked": [
    {{
      "chunk_id": "...",
      "relevance_score": 0-100,
      "answerability": "yes" or "no",
      "key_evidence": ["quote 1", "quote 2"]
    }}
  ]
}}"""
        
        return prompt
    
    def _call_llm_with_retry(self, prompt: str) -> str:
        """Call LLM with exponential backoff retry.
        
        Args:
            prompt: Prompt string
            
        Returns:
            Response text
        """
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": "You are a relevance rater. Output only valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0
                )
                return response.choices[0].message.content
            except RateLimitError as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                else:
                    raise
            except APIError as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                else:
                    raise
        
        raise last_error or Exception("Failed to call LLM")
    
    def _parse_response(self, response_text: str, candidates: List[CandidateText]) -> List[RerankResult]:
        """Parse and validate LLM response.
        
        Args:
            response_text: Raw response text from LLM
            candidates: Original candidates list
            
        Returns:
            List of reranked results
        """
        # Try to parse JSON
        try:
            response_dict = json.loads(response_text)
            rerank_response = RerankResponse(**response_dict)
        except (json.JSONDecodeError, ValueError) as e:
            # Retry once with repair prompt
            try:
                repair_prompt = f"""Output valid JSON only. Previous output was invalid:
{response_text}

Please output valid JSON in the correct format."""
                response_text = self._call_llm_with_retry(repair_prompt)
                response_dict = json.loads(response_text)
                rerank_response = RerankResponse(**response_dict)
            except Exception:
                # If repair fails, return empty list
                return []
        
        # Convert to RerankResult objects
        results = []
        candidate_map = {c.chunk_id: c for c in candidates}
        
        for item in rerank_response.ranked:
            if item.chunk_id in candidate_map:
                results.append(RerankResult(
                    chunk_id=item.chunk_id,
                    relevance_score=item.relevance_score,
                    answerability=item.answerability,
                    key_evidence=item.key_evidence
                ))
        
        # Sort by relevance_score descending
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return results
    
    def _make_cache_key(self, query: str, candidates: List[CandidateText]) -> str:
        """Make cache key from query and candidates.
        
        Args:
            query: Query string
            candidates: List of candidates
            
        Returns:
            Cache key string
        """
        candidate_ids = ",".join(sorted(c.chunk_id for c in candidates))
        return f"{self.config.model}:{query}:{candidate_ids}"
    
    def clear_cache(self):
        """Clear the rerank cache."""
        self._cache.clear()

