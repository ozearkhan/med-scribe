"""
Gemini LLM Client for Psychiatry AI Scribe.
Handles all LLM operations: reranking, attribute validation, text generation.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google-generativeai not installed. Run: pip install google-generativeai")

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """Result of reranking operation."""
    class_name: str
    score: float
    reasoning: str


@dataclass
class AttributeValidationResult:
    """Result of attribute validation."""
    is_valid: bool
    score: float
    conditions_met: List[str]
    conditions_not_met: List[str]
    explanation: str


class GeminiClient:
    """
    Gemini API client for LLM operations.
    Provides methods for reranking, attribute validation, and text generation.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash-exp",
        temperature: float = 0.1,
        max_tokens: int = 1000
    ):
        """
        Initialize Gemini client.
        
        Args:
            api_key: Gemini API key (or from GEMINI_API_KEY env var)
            model_name: Model to use for generation
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if GEMINI_AVAILABLE and self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(model_name)
        else:
            self.model = None
            if not self.api_key:
                logger.warning("No Gemini API key provided. LLM features will be unavailable.")
    
    def is_available(self) -> bool:
        """Check if Gemini client is properly configured."""
        return GEMINI_AVAILABLE and bool(self.api_key) and self.model is not None
    
    def rerank_candidates(
        self,
        text: str,
        candidates: List[Tuple[str, str, float]],  # (class_name, description, similarity_score)
        task_type: str = "classification"
    ) -> List[RerankResult]:
        """
        Rerank classification candidates using LLM.
        
        Args:
            text: Input text to classify
            candidates: List of (class_name, description, similarity_score) tuples
            task_type: Type of classification task (section, domain, entity, etc.)
        
        Returns:
            List of RerankResult sorted by score descending
        """
        if not self.is_available():
            # Return candidates as-is with similarity scores
            return [
                RerankResult(name, score, "LLM unavailable")
                for name, _, score in candidates
            ]
        
        # Build reranking prompt
        prompt = self._build_rerank_prompt(text, candidates, task_type)
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                )
            )
            
            # Parse response
            return self._parse_rerank_response(response.text, candidates)
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Fallback to similarity scores
            return [
                RerankResult(name, score, f"Error: {str(e)}")
                for name, _, score in candidates
            ]
    
    def validate_attributes(
        self,
        text: str,
        class_name: str,
        attributes: Dict[str, Any]
    ) -> AttributeValidationResult:
        """
        Validate text against class-specific attributes.
        
        Args:
            text: Input text to validate
            class_name: Predicted class name
            attributes: Required attributes for the class
        
        Returns:
            AttributeValidationResult with validation details
        """
        if not self.is_available():
            return AttributeValidationResult(
                is_valid=True,
                score=1.0,
                conditions_met=[],
                conditions_not_met=[],
                explanation="LLM unavailable - validation skipped"
            )
        
        prompt = self._build_validation_prompt(text, class_name, attributes)
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                )
            )
            
            return self._parse_validation_response(response.text)
            
        except Exception as e:
            logger.error(f"Attribute validation failed: {e}")
            return AttributeValidationResult(
                is_valid=True,
                score=1.0,
                conditions_met=[],
                conditions_not_met=[],
                explanation=f"Error: {str(e)}"
            )
    
    def _build_rerank_prompt(
        self,
        text: str,
        candidates: List[Tuple[str, str, float]],
        task_type: str
    ) -> str:
        """Build prompt for reranking candidates."""
        candidates_text = "\n".join([
            f"{i+1}. {name}: {desc} (similarity: {score:.2f})"
            for i, (name, desc, score) in enumerate(candidates)
        ])
        
        return f"""You are an expert psychiatry clinical documentation classifier.

TASK: Rerank the following classification candidates for the given text.

INPUT TEXT:
"{text}"

CANDIDATES (ranked by embedding similarity):
{candidates_text}

INSTRUCTIONS:
1. Analyze the input text carefully
2. Consider which candidate class best matches the text semantically
3. For psychiatric notes, consider clinical context, terminology, and section patterns
4. Return a JSON object with reranked scores (0.0 to 1.0) and brief reasoning

OUTPUT FORMAT (JSON only, no markdown):
{{
  "rankings": [
    {{"class": "ClassName", "score": 0.95, "reasoning": "Brief explanation"}},
    ...
  ]
}}

Return ONLY the JSON, no other text."""

    def _build_validation_prompt(
        self,
        text: str,
        class_name: str,
        attributes: Dict[str, Any]
    ) -> str:
        """Build prompt for attribute validation."""
        attrs_text = json.dumps(attributes, indent=2)
        
        return f"""You are an expert psychiatry clinical documentation validator.

TASK: Validate if the text satisfies the required attributes for class "{class_name}".

INPUT TEXT:
"{text}"

REQUIRED ATTRIBUTES:
{attrs_text}

INSTRUCTIONS:
1. Check each condition in the attributes
2. For AND conditions, all must be met
3. For OR conditions, at least one must be met
4. Consider negation patterns (e.g., "denies SI" vs "reports SI")

OUTPUT FORMAT (JSON only, no markdown):
{{
  "is_valid": true/false,
  "score": 0.0-1.0,
  "conditions_met": ["condition1", "condition2"],
  "conditions_not_met": ["condition3"],
  "explanation": "Brief explanation"
}}

Return ONLY the JSON, no other text."""

    def _parse_rerank_response(
        self,
        response_text: str,
        candidates: List[Tuple[str, str, float]]
    ) -> List[RerankResult]:
        """Parse reranking response from LLM."""
        try:
            # Clean response
            text = response_text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            
            data = json.loads(text)
            rankings = data.get("rankings", [])
            
            results = []
            for item in rankings:
                results.append(RerankResult(
                    class_name=item.get("class", "Unknown"),
                    score=float(item.get("score", 0.0)),
                    reasoning=item.get("reasoning", "")
                ))
            
            # Sort by score descending
            results.sort(key=lambda x: x.score, reverse=True)
            return results
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse rerank response: {e}")
            # Fallback to original candidates
            return [
                RerankResult(name, score, "Parse error")
                for name, _, score in candidates
            ]
    
    def _parse_validation_response(
        self,
        response_text: str
    ) -> AttributeValidationResult:
        """Parse validation response from LLM."""
        try:
            # Clean response
            text = response_text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            
            data = json.loads(text)
            
            return AttributeValidationResult(
                is_valid=data.get("is_valid", True),
                score=float(data.get("score", 1.0)),
                conditions_met=data.get("conditions_met", []),
                conditions_not_met=data.get("conditions_not_met", []),
                explanation=data.get("explanation", "")
            )
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse validation response: {e}")
            return AttributeValidationResult(
                is_valid=True,
                score=1.0,
                conditions_met=[],
                conditions_not_met=[],
                explanation=f"Parse error: {str(e)}"
            )


# Global client instance (initialized lazily)
_gemini_client: Optional[GeminiClient] = None


def get_gemini_client() -> GeminiClient:
    """Get or create global Gemini client."""
    global _gemini_client
    if _gemini_client is None:
        from config import config
        _gemini_client = GeminiClient(
            api_key=config.gemini.api_key,
            model_name=config.gemini.default_model,
            temperature=config.models.default_temperature,
            max_tokens=config.models.default_max_tokens,
        )
    return _gemini_client
