
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import spacy
import pytextrank  

from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

SPACY_MODEL_PREFERENCE = ["en_core_web_trf", "en_core_web_sm"]
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"   
USE_COREF = False 
 
@dataclass
class Section:
    heading: Optional[str]
    text: str
    sentences: list[str] = field(default_factory=list)


@dataclass
class PreprocessedDoc:
    raw_text: str
    sections: list[Section]
    all_sentences: list[str]          
    sentence_to_section: list[int]    

@dataclass
class ConceptCandidate:
    text: str
    score: float
    source: str   
    
_HEADING_RE = re.compile(
    r"""^(
        \#{1,6}\s+.+              |   # markdown headings
        [A-Z][A-Za-z0-9 ,'\-]{2,60}$ # short Title-Case-ish standalone line (heuristic)
    )$""",
    re.VERBOSE,
)