"""
Section Splitter for Psychiatric Clinical Notes.

Splits raw clinical notes into individual sections based on:
- Bullet markers (* or **)
- Section keywords (CC, History, MSE, etc.)
- Line patterns

This allows per-section classification instead of whole-note classification.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# Section keyword patterns (case insensitive)
SECTION_KEYWORDS = {
    # Chief Complaint
    r'\bCC\b': 'Chief Complaint',
    r'\bChief Complaint\b': 'Chief Complaint',
    r'\bPresenting Complaint\b': 'Chief Complaint',
    
    # History of Present Illness
    r'\bHPI\b': 'History of Present Illness',
    r'\bHistory of Present Illness\b': 'History of Present Illness',
    r'\bHistory\s*[-:]': 'History of Present Illness',
    
    # Past Psychiatric History
    r'\bPast Psychiatric History\b': 'Past Psychiatric History',
    r'\bPHx\b': 'Past Psychiatric History',
    r'\bPast History\b': 'Past Psychiatric History',
    
    # Family History
    r'\bFamily History\b': 'Family Psychiatric History',
    r'\bFamily\s+history\b': 'Family Psychiatric History',
    r'\bFHx\b': 'Family Psychiatric History',
    
    # Substance Use
    r'\bSubstance Use\b': 'Substance Use History',
    r'\bSubstance\s+use\b': 'Substance Use History',
    r'\bDrug History\b': 'Substance Use History',
    r'\bAlcohol\s+History\b': 'Substance Use History',
    
    # Social History  
    r'\bSocial History\b': 'Social History',
    r'\bPremorbid\s+Personality\b': 'Social History',
    r'\bPremorbid\b': 'Social History',
    
    # Mental Status Examination
    r'\bMSE\b': 'Mental Status Examination',
    r'\bMental Status\b': 'Mental Status Examination',
    r'\bGeneral\s+appearance\b': 'Mental Status Examination',
    r'\bAppearance\s+and\s+behaviour\b': 'Mental Status Examination',
    r'\bPMA\b': 'Mental Status Examination',
    r'\bTalk\b': 'Mental Status Examination',
    r'\bMood\b': 'Mental Status Examination',
    r'\bAffect\b': 'Mental Status Examination',
    r'\bThought\b': 'Mental Status Examination',
    r'\bPerception\b': 'Mental Status Examination',
    
    # Assessment
    r'\bAssessment\b': 'Assessment',
    r'\bImpression\b': 'Assessment',
    r'\bDiagnosis\b': 'Assessment',
    r'\bDx\b': 'Assessment',
    
    # Plan
    r'\bPlan\b': 'Plan',
    r'\bManagement\b': 'Plan',
    r'\bTreatment\b': 'Plan',
    r'\bRx\b': 'Plan',
}


@dataclass
class NoteSection:
    """A section extracted from a clinical note."""
    section_type: str  # Detected or classified section type
    content: str       # The text content of this section
    start_pos: int     # Character start position in original note
    end_pos: int       # Character end position in original note
    confidence: float  # Confidence in section detection (1.0 for keyword match)
    keywords_found: List[str]  # Keywords that triggered this detection


class SectionSplitter:
    """
    Splits psychiatric clinical notes into sections.
    
    Strategy:
    1. Split by bullet markers (* or **)
    2. Detect section type by keywords
    3. Merge consecutive chunks of same type
    """
    
    def __init__(self):
        # Compile keyword patterns for efficiency
        self.keyword_patterns = [
            (re.compile(pattern, re.IGNORECASE), section_type)
            for pattern, section_type in SECTION_KEYWORDS.items()
        ]
    
    def split_note(self, note_text: str) -> List[NoteSection]:
        """
        Split a clinical note into sections.
        
        Args:
            note_text: Raw clinical note text
            
        Returns:
            List of NoteSection objects
        """
        # Step 1: Split by bullet markers
        chunks = self._split_by_markers(note_text)
        
        # Step 2: Detect section type for each chunk
        sections = []
        for chunk_text, start_pos, end_pos in chunks:
            section_type, confidence, keywords = self._detect_section_type(chunk_text)
            
            sections.append(NoteSection(
                section_type=section_type,
                content=chunk_text.strip(),
                start_pos=start_pos,
                end_pos=end_pos,
                confidence=confidence,
                keywords_found=keywords
            ))
        
        # Step 3: Merge consecutive sections of same type
        merged_sections = self._merge_sections(sections)
        
        logger.info(f"Split note into {len(merged_sections)} sections")
        for s in merged_sections:
            logger.debug(f"  - {s.section_type}: {len(s.content)} chars")
        
        return merged_sections
    
    def _split_by_markers(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Split text by bullet markers (* or **).
        
        Returns list of (chunk_text, start_pos, end_pos)
        """
        # Pattern to match bullet markers at start of line or after newline
        # Handles: *, * *, **, line breaks
        pattern = r'(?:^|\n)\s*\*+\s*\*?\s*'
        
        chunks = []
        last_end = 0
        
        for match in re.finditer(pattern, text):
            # Content before this marker
            if last_end < match.start():
                chunk = text[last_end:match.start()]
                if chunk.strip():  # Only add non-empty chunks
                    chunks.append((chunk, last_end, match.start()))
            
            last_end = match.end()
        
        # Add remaining content after last marker
        if last_end < len(text):
            chunk = text[last_end:]
            if chunk.strip():
                chunks.append((chunk, last_end, len(text)))
        
        # If no markers found, return whole text as one chunk
        if not chunks:
            chunks = [(text, 0, len(text))]
        
        return chunks
    
    def _detect_section_type(self, chunk_text: str) -> Tuple[str, float, List[str]]:
        """
        Detect section type from chunk content using keywords.
        
        Returns:
            (section_type, confidence, keywords_found)
        """
        found_keywords = []
        detected_type = "Unknown"
        
        # Check each keyword pattern
        for pattern, section_type in self.keyword_patterns:
            if pattern.search(chunk_text):
                found_keywords.append(section_type)
                detected_type = section_type
        
        # Confidence based on keyword matches
        if found_keywords:
            # If multiple keywords, use most specific or first found
            confidence = 0.9
            # Use the most common section type if multiple found
            from collections import Counter
            if len(found_keywords) > 1:
                counter = Counter(found_keywords)
                detected_type = counter.most_common(1)[0][0]
        else:
            confidence = 0.0
        
        return detected_type, confidence, found_keywords
    
    def _merge_sections(self, sections: List[NoteSection]) -> List[NoteSection]:
        """
        Merge consecutive sections of the same type.
        """
        if not sections:
            return []
        
        merged = []
        current = sections[0]
        
        for i in range(1, len(sections)):
            next_section = sections[i]
            
            # Merge if same type and both have high confidence
            if (current.section_type == next_section.section_type and 
                current.section_type != "Unknown" and
                current.confidence > 0.5 and next_section.confidence > 0.5):
                # Merge content
                current = NoteSection(
                    section_type=current.section_type,
                    content=current.content + "\n" + next_section.content,
                    start_pos=current.start_pos,
                    end_pos=next_section.end_pos,
                    confidence=min(current.confidence, next_section.confidence),
                    keywords_found=current.keywords_found + next_section.keywords_found
                )
            else:
                merged.append(current)
                current = next_section
        
        merged.append(current)
        return merged
    
    def get_section_summary(self, sections: List[NoteSection]) -> Dict[str, Any]:
        """
        Get a summary of extracted sections.
        """
        return {
            "total_sections": len(sections),
            "sections": [
                {
                    "type": s.section_type,
                    "length": len(s.content),
                    "confidence": s.confidence,
                    "preview": s.content[:100] + "..." if len(s.content) > 100 else s.content
                }
                for s in sections
            ]
        }


# Global instance
_splitter: Optional[SectionSplitter] = None


def get_section_splitter() -> SectionSplitter:
    """Get or create global section splitter instance."""
    global _splitter
    if _splitter is None:
        _splitter = SectionSplitter()
    return _splitter
