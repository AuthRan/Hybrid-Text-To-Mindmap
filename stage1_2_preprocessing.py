"""
Stage 1: Preprocessing & Segmentation
Stage 2: Concept / Keyphrase Extraction

Design notes:
- We use spaCy's `en_core_web_trf` if available (best accuracy for dependency
  parsing, which Stage 3 relies on heavily) and fall back to `en_core_web_sm`
  (faster, slightly less accurate) if trf isn't installed. For short articles
  (500-1500 words) trf is fast enough on CPU to be worth it.
- Coreference resolution is OPTIONAL and gated behind a flag. fastcoref is the
  best currently-maintained lightweight local coref model, but it's an extra
  ~500MB download and extra latency. For short, well-written articles the
  payoff is usually modest because pronoun chains are short. Recommend
  trying WITHOUT it first, turning it on only if you see nodes like "it" /
  "this process" / "these" leaking into your concept set.
- KeyBERT and TextRank are combined (union, not intersection) because they
  have different failure modes: KeyBERT is embedding-similarity based and
  tends to surface thematically central but possibly rare phrases; TextRank
  is graph-centrality based on co-occurrence and tends to surface
  structurally important, frequently-connected phrases. Run a few sample
  texts through both with the diagnostic prints left in here and you'll see
  the lists are usually 60-80% overlapping with a handful of useful
  disagreements.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import spacy
import pytextrank  # noqa: F401 -- import side effect registers the "textrank"
                    # factory onto spaCy's Language class; without this import,
                    # nlp.add_pipe("textrank") fails with E002 even though the
                    # package is installed
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SPACY_MODEL_PREFERENCE = ["en_core_web_trf", "en_core_web_sm"]
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"   # strong general-purpose sentence embedder
USE_COREF = False  # flip to True if you install fastcoref and want it


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Section:
    """A structural unit of the document. If the source text has no markup,
    the whole document is a single Section with heading=None."""
    heading: Optional[str]
    text: str
    sentences: list[str] = field(default_factory=list)


@dataclass
class PreprocessedDoc:
    raw_text: str
    sections: list[Section]
    all_sentences: list[str]          # flattened, in order, across all sections
    sentence_to_section: list[int]    # parallel array: section index per sentence


@dataclass
class ConceptCandidate:
    text: str
    score: float
    source: str   # "keybert" | "textrank" | "both"


# ---------------------------------------------------------------------------
# Stage 1: Preprocessing & Segmentation
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(
    r"""^(
        \#{1,6}\s+.+              |   # markdown headings
        [A-Z][A-Za-z0-9 ,'\-]{2,60}$ # short Title-Case-ish standalone line (heuristic)
    )$""",
    re.VERBOSE,
)


def _looks_like_heading(line: str, next_line_is_blank_or_text: bool) -> bool:
    """Heuristic structural-cue detector. This is intentionally conservative:
    false negatives (missing a real heading) are cheap, since the pipeline
    degrades gracefully to prose mode. False positives (treating a normal
    sentence as a heading) are more damaging since it would wrongly anchor
    a level-1 mindmap node. So we require ALL of:
      - short line (<= 8 words)
      - no terminal punctuation (.?! at the end)
      - either markdown '#' prefix, OR Title Case / ALLCAPS pattern
    """
    line = line.strip()
    if not line:
        return False
    if line.startswith("#"):
        return True
    if len(line.split()) > 8:
        return False
    if line[-1:] in ".?!,;:":
        return False
    words = line.split()
    cap_ratio = sum(1 for w in words if w[:1].isupper()) / max(len(words), 1)
    return cap_ratio >= 0.6


def split_into_sections(raw_text: str) -> list[Section]:
    """Splits on detected headings. If none are found, returns a single
    Section covering the whole document (heading=None) -- this is the
    'unstructured prose' fallback path."""
    lines = raw_text.splitlines()
    sections: list[Section] = []
    current_heading: Optional[str] = None
    current_buf: list[str] = []

    def flush():
        text = "\n".join(current_buf).strip()
        if text:
            sections.append(Section(heading=current_heading, text=text))

    for i, line in enumerate(lines):
        if _looks_like_heading(line, True):
            flush()
            current_heading = line.strip().lstrip("#").strip()
            current_buf = []
        else:
            current_buf.append(line)
    flush()

    if not sections:
        sections = [Section(heading=None, text=raw_text.strip())]
    return sections


def preprocess(raw_text: str, nlp: spacy.Language, use_coref: bool = USE_COREF) -> PreprocessedDoc:
    if use_coref:
        raw_text = _resolve_coref(raw_text)

    sections = split_into_sections(raw_text)

    all_sentences: list[str] = []
    sentence_to_section: list[int] = []

    for sec_idx, sec in enumerate(sections):
        doc = nlp(sec.text)
        sents = [s.text.strip() for s in doc.sents if s.text.strip()]
        sec.sentences = sents
        all_sentences.extend(sents)
        sentence_to_section.extend([sec_idx] * len(sents))

    return PreprocessedDoc(
        raw_text=raw_text,
        sections=sections,
        all_sentences=all_sentences,
        sentence_to_section=sentence_to_section,
    )


def _resolve_coref(raw_text: str) -> str:
    """Optional coref resolution using fastcoref. Replaces pronoun/referring
    mentions with their resolved antecedent text. Only call this if
    USE_COREF=True and fastcoref is installed.

    NOTE: naive string-splice resolution like this can occasionally produce
    slightly awkward grammar (e.g. repeated determiners) -- that's an
    acceptable cost here because this text is feeding extraction stages,
    not being shown to the end user directly.
    """
    try:
        from fastcoref import FCoref
    except ImportError as e:
        raise ImportError(
            "USE_COREF=True but fastcoref isn't installed. "
            "Run: pip install fastcoref"
        ) from e

    model = FCoref()
    preds = model.predict(texts=[raw_text])[0]
    clusters = preds.get_clusters(as_strings=False)

    # For each cluster, pick the longest mention as the canonical referent,
    # then replace every other (shorter) mention's span with it.
    # We do span replacement from right to left so earlier offsets stay valid.
    replacements = []  # (start, end, replacement_text)
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        spans_text = [raw_text[s:e] for s, e in cluster]
        canonical = max(spans_text, key=len)
        for (s, e), mention_text in zip(cluster, spans_text):
            if mention_text != canonical and len(mention_text.split()) <= 2:
                # only replace short referring expressions (pronouns, short
                # NPs like "this process"), never touch the canonical mention
                # itself or other long mentions (avoids mangling the text)
                replacements.append((s, e, canonical))

    replacements.sort(key=lambda x: x[0], reverse=True)
    text_chars = list(raw_text)
    result = raw_text
    for s, e, repl in replacements:
        result = result[:s] + repl + result[e:]

    return result


# ---------------------------------------------------------------------------
# Stage 2: Concept / Keyphrase Extraction
# ---------------------------------------------------------------------------

def extract_keybert_candidates(
    full_text: str,
    kw_model: KeyBERT,
    top_n: int = 25,
) -> list[ConceptCandidate]:
    """Embedding-similarity-based keyphrase extraction. use_mmr trades a bit
    of pure relevance for diversity, which matters a lot for mindmaps --
    without it KeyBERT tends to return near-duplicate phrases
    ("photosynthesis process", "process of photosynthesis", "photosynthetic
    process") that would otherwise collapse into one node anyway, wasting
    your top_n budget."""
    results = kw_model.extract_keywords(
        full_text,
        keyphrase_ngram_range=(1, 3),
        stop_words="english",
        use_mmr=True,
        diversity=0.6,
        top_n=top_n,
    )
    return [ConceptCandidate(text=t, score=s, source="keybert") for t, s in results]


def extract_named_entity_candidates(
    doc: spacy.tokens.Doc,
) -> list[ConceptCandidate]:
    """Extracts named entities via spaCy NER as guaranteed concept candidates.
    This catches specific important terms (enzyme names like RuBisCO, molecules
    like NADPH/ATP, organisms like Carotenoids) that frequency-based methods
    (TextRank) miss because they appear infrequently, and that KeyBERT's MMR
    diversity filter can suppress when a semantically similar broader term
    scores higher.

    We use a fixed score of 0.5 (mid-range, normalized) so NER candidates
    don't crowd out highly-ranked KeyBERT/TextRank results when merged, but
    can survive into the final concept set if not already covered.

    Entity types kept: chemicals, substances, organisms, processes, and
    generic catch-all for domain-specific proper nouns (MISC/PROPN coverage).
    Types excluded: dates, cardinal numbers, ordinals, locations (GPE/LOC),
    persons (PERSON) -- these rarely belong in a science content mindmap.
    """
    KEEP_ENT_TYPES = {
        "CHEM", "CHEMICAL", "ORG", "PRODUCT", "WORK_OF_ART",
        # spaCy en_core_web_sm uses these broader categories:
        "ORG", "GPE", "FAC", "PRODUCT", "EVENT", "LAW",
        # for science text, MISC and bare PROPN tokens are often enzyme/molecule names
    }
    SKIP_ENT_TYPES = {"DATE", "TIME", "PERCENT", "MONEY", "QUANTITY",
                      "ORDINAL", "CARDINAL", "PERSON", "GPE", "LOC", "FAC"}

    seen: set[str] = set()
    candidates: list[ConceptCandidate] = []

    for ent in doc.ents:
        if ent.label_ in SKIP_ENT_TYPES:
            continue
        text = ent.text.strip()
        if len(text) < 2 or text.lower() in seen:
            continue
        seen.add(text.lower())
        candidates.append(ConceptCandidate(text=text, score=0.5, source="ner"))

    # Also scan for ALL-CAPS and mixed-case short tokens (e.g. ATP, NADPH,
    # RuBisCO) that NER might miss because en_core_web_sm's NER is trained
    # on news, not science. These are almost always molecule/enzyme names.
    for token in doc:
        if token.is_stop or token.is_punct or token.is_space:
            continue
        t = token.text.strip()
        if len(t) < 2 or t.lower() in seen:
            continue
        # ALL-CAPS acronym (ATP, NADPH, DNA) or CamelCase molecule (RuBisCO)
        is_allcaps_acronym = t.isupper() and len(t) >= 2
        is_camelcase = (not t.isupper() and not t.islower() and
                        t[0].isupper() and any(c.isupper() for c in t[1:]))
        if is_allcaps_acronym or is_camelcase:
            seen.add(t.lower())
            candidates.append(ConceptCandidate(text=t, score=0.5, source="ner"))

    return candidates


def extract_textrank_candidates(
    doc: spacy.tokens.Doc,
    top_n: int = 25,
) -> list[ConceptCandidate]:
    """Graph-centrality-based extraction via pytextrank. Requires the
    'textrank' pipe to have been added to the spaCy pipeline before calling
    nlp() on this doc -- see build_nlp_pipeline()."""
    candidates = []
    for phrase in doc._.phrases[:top_n]:
        candidates.append(
            ConceptCandidate(text=phrase.text, score=phrase.rank, source="textrank")
        )
    return candidates


def _normalize_phrase(p: str) -> str:
    return re.sub(r"\s+", " ", p.strip().lower())


# Patterns that are structurally non-concepts regardless of KeyBERT/TextRank score.
# Verb phrases, adverbs, loose modifiers, and pronouns should never be mindmap nodes.
_NOISE_PATTERNS = re.compile(
    r"""^(
        # bare pronouns / demonstratives
        (it|this|that|these|those|which|who|what|they|them|their|its)\b
        |
        # adverbs used as concept candidates (common TextRank false positives)
        (specifically|primarily|mainly|generally|also|additionally|however|therefore|thus)\b
        |
        # generic relational phrases that aren't domain concepts
        (critical\s+role|key\s+role|important\s+role|plays?\s+\w+|process\s+occur\w*|
         also\s+known|known\s+as)
        |
        # verb-phrase fragments: first word is a verb base form / gerund
        # e.g. "split water molecules", "membrane drives atp", "rubisco plays"
        # We catch these by checking if the phrase starts with a known verb pattern.
        # This is a heuristic -- not perfect, but much better than no filter.
        (\w+(ing|ed|es|s)\s+\w+\s+\w+|\w+\s+(drive|drives|play|plays|occur|occurs|convert|converts|absorb|absorbs|affect|affects|impact|impacts)\s)
    )$""",
    re.VERBOSE | re.IGNORECASE,
)

_STOP_SINGLE_WORDS = {
    "specifically", "primarily", "mainly", "generally", "also", "additionally",
    "however", "therefore", "thus", "which", "this", "that", "these", "those",
    "it", "they", "them", "place", "stage", "process", "role", "way", "form",
    "type", "kind", "part", "level", "rate", "range", "point", "step",
}


def filter_concept_candidates(
    candidates: list[ConceptCandidate],
    nlp: spacy.Language,
) -> list[ConceptCandidate]:
    """Post-filters merged concept candidates using spaCy POS tags to remove
    phrases that are grammatically non-concepts: verb phrases, adverbs,
    bare function words, and lone generic nouns that carry no domain meaning.

    The strategy: parse each candidate phrase with spaCy and check that the
    HEAD token (the syntactic root of the phrase) is a NOUN or PROPN. Verb
    phrases have a VERB root; adverbial phrases have an ADV root. Single-word
    candidates are additionally checked against a stop-list of generic words.

    This runs on short phrase strings (~1-3 words each), so the overhead is
    negligible even without batching.
    """
    clean: list[ConceptCandidate] = []
    for c in candidates:
        phrase = c.text.strip()

        # single-word stop-list check (fast path)
        if phrase.lower() in _STOP_SINGLE_WORDS:
            continue

        # regex noise pattern check (catches obvious verb/adverb phrases)
        if _NOISE_PATTERNS.match(phrase):
            continue

        # spaCy POS check: the syntactic root of the phrase must be a noun
        doc = nlp(phrase)
        if not doc:
            continue
        root_token = max(doc, key=lambda t: t.head.i == t.i)  # token that is its own head = root
        if root_token.pos_ in ("VERB", "ADV", "ADP", "CCONJ", "SCONJ", "DET", "PRON"):
            continue

        clean.append(c)

    removed = len(candidates) - len(clean)
    if removed:
        print(f"  [stage2] POS filter removed {removed} non-concept phrases")
    return clean


def merge_concept_candidates(
    keybert_cands: list[ConceptCandidate],
    textrank_cands: list[ConceptCandidate],
) -> list[ConceptCandidate]:
    """Union with score reconciliation. Scores from the two methods are on
    different scales (KeyBERT: cosine similarity ~0-1, TextRank: graph rank,
    roughly 0-1 but not directly comparable), so we min-max normalize each
    list independently before merging, then take the max normalized score
    per phrase as its combined score. Phrases found by both methods get
    source='both', which Stage 4 can use as a confidence signal when
    deciding root/level-1 nodes."""

    def _minmax_normalize(cands: list[ConceptCandidate]) -> list[ConceptCandidate]:
        if not cands:
            return cands
        scores = [c.score for c in cands]
        lo, hi = min(scores), max(scores)
        rng = (hi - lo) or 1.0
        return [
            ConceptCandidate(text=c.text, score=(c.score - lo) / rng, source=c.source)
            for c in cands
        ]

    kb = _minmax_normalize(keybert_cands)
    tr = _minmax_normalize(textrank_cands)

    merged: dict[str, ConceptCandidate] = {}
    for c in kb + tr:
        key = _normalize_phrase(c.text)
        if key in merged:
            existing = merged[key]
            new_score = max(existing.score, c.score)
            new_source = "both" if existing.source != c.source else existing.source
            merged[key] = ConceptCandidate(text=existing.text, score=new_score, source=new_source)
        else:
            merged[key] = c

    return sorted(merged.values(), key=lambda c: c.score, reverse=True)


# ---------------------------------------------------------------------------
# Pipeline setup helpers
# ---------------------------------------------------------------------------

def build_nlp_pipeline() -> spacy.Language:
    """Loads the best available spaCy model and attaches pytextrank.
    Tries en_core_web_trf first (transformer-based, much better dependency
    parses -- this matters for Stage 3's SVO extraction), falls back to
    en_core_web_sm if trf isn't installed."""
    nlp = None
    for model_name in SPACY_MODEL_PREFERENCE:
        try:
            nlp = spacy.load(model_name)
            print(f"[stage1-2] loaded spaCy model: {model_name}")
            break
        except OSError:
            continue
    if nlp is None:
        raise OSError(
            "No spaCy model found. Run one of:\n"
            "  python -m spacy download en_core_web_trf   (recommended)\n"
            "  python -m spacy download en_core_web_sm    (faster fallback)"
        )

    nlp.add_pipe("textrank")
    return nlp


def build_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def build_keybert(embedder: SentenceTransformer) -> KeyBERT:
    return KeyBERT(model=embedder)


# ---------------------------------------------------------------------------
# Quick standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SAMPLE_TEXT = """
    Photosynthesis

    Photosynthesis is the process by which green plants, algae, and some
    bacteria convert light energy into chemical energy. This process occurs
    primarily in the chloroplasts of plant cells, specifically within
    structures called thylakoids. Chlorophyll, the green pigment found in
    chloroplasts, absorbs light energy, mainly in the red and blue
    wavelengths of the visible spectrum.

    Light-Dependent Reactions

    The light-dependent reactions take place in the thylakoid membrane.
    During this stage, light energy is absorbed and used to split water
    molecules into oxygen, protons, and electrons. This process is called
    photolysis. The electrons move through the electron transport chain,
    generating ATP and NADPH. Oxygen is released as a byproduct of this
    reaction.

    Calvin Cycle

    The Calvin cycle, also known as the light-independent reactions, occurs
    in the stroma of the chloroplast. It uses the ATP and NADPH produced
    during the light-dependent reactions to convert carbon dioxide into
    glucose. This process is also called carbon fixation. The enzyme RuBisCO
    plays a critical role in catalyzing the first major step of carbon
    fixation.
    """

    nlp = build_nlp_pipeline()
    embedder = build_embedder()
    kw_model = build_keybert(embedder)

    pre = preprocess(SAMPLE_TEXT, nlp, use_coref=False)
    print(f"\n[stage1] {len(pre.sections)} sections, {len(pre.all_sentences)} sentences")
    for sec in pre.sections:
        print(f"  - heading={sec.heading!r}  ({len(sec.sentences)} sentences)")

    full_doc = nlp(pre.raw_text)
    kb_cands = extract_keybert_candidates(pre.raw_text, kw_model)
    tr_cands = extract_textrank_candidates(full_doc)
    merged = merge_concept_candidates(kb_cands, tr_cands)

    print(f"\n[stage2] KeyBERT top candidates:")
    for c in kb_cands[:10]:
        print(f"    {c.score:.3f}  {c.text}")

    print(f"\n[stage2] TextRank top candidates:")
    for c in tr_cands[:10]:
        print(f"    {c.score:.3f}  {c.text}")

    print(f"\n[stage2] MERGED concept candidates ({len(merged)} total):")
    for c in merged[:20]:
        print(f"    {c.score:.3f}  [{c.source:8s}]  {c.text}")