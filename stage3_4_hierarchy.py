"""
Stage 3: Relation Extraction (structure, not flat OpenIE triples)
Stage 4: Hierarchy Construction

Design notes:
- Stage 3 builds TWO graphs that get fused:
    (a) an SVO dependency graph: concept A -[verb]-> concept B, extracted via
        spaCy dependency parsing. This gives explicit, labeled, directional
        relations -- more reliable than OpenIE's generic extraction because
        we constrain matching to sentences containing concept-candidate
        phrases from Stage 2, rather than extracting from every sentence
        indiscriminately. This is the key fix vs your original option 2: we
        don't ask the model (or here, the parser) to extract everything and
        hope it's useful -- we extract relations specifically *between
        concepts we already know matter*.
    (b) a semantic similarity graph: concept-bearing sentences embedded and
        connected by cosine similarity above a threshold. This catches
        relations that aren't expressed as clean SVO (e.g. "Chlorophyll...
        absorbs light energy, mainly in the red and blue wavelengths" has
        useful structure beyond the single SVO triple).
  These are fused into one weighted graph: SVO edges get a confidence boost
  since they're more precise; similarity edges fill gaps SVO misses.

- Stage 4 turns that graph into a tree via:
    1. PageRank over the fused graph -> centrality score per concept node.
       Highest-centrality nodes become root/level-1 candidates.
    2. If section headings were detected in Stage 1, they take priority as
       level-1 nodes (strong structural prior -- the author already told us
       the hierarchy). PageRank is then used only WITHIN each section to
       rank that section's children, and across sections to order them.
    3. If no headings exist (pure prose), PageRank centrality alone
       determines level-1 nodes (top-K by score, K chosen via a simple
       elbow heuristic on the score distribution).
    4. Agglomerative clustering (on concept embeddings) groups remaining
       concepts under their nearest level-1/level-2 parent, using a
       combination of (a) graph proximity in the fused graph and
       (b) embedding similarity to decide parent assignment, since relying
       on embedding similarity alone can group concepts that are thematically
       similar but not actually hierarchically related (e.g. "chlorophyll"
       and "carotenoids" are similar but siblings, not parent/child).

This produces a tree (networkx DiGraph) with node attributes:
  - label: raw text (Stage 5 cleans this up)
  - level: 0=root, 1=branch, 2=sub-branch, ...
  - source_sentences: list[str], for Stage 5/6 grounding
  - centrality: float
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

from stage1_2_preprocessing import ConceptCandidate, PreprocessedDoc


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SIMILARITY_EDGE_THRESHOLD = 0.45     # min cosine sim to draw a similarity edge
SVO_EDGE_WEIGHT_BOOST = 1.5          # SVO edges trusted more than similarity edges
MAX_LEVEL1_NODES_NO_HEADINGS = 6     # cap when inferring root branches from prose
MIN_CONCEPT_LEN_CHARS = 3


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Relation:
    subj: str
    verb: str
    obj: str
    sentence: str


@dataclass
class MindmapNode:
    id: str
    label: str                      # raw label, Stage 5 will clean this
    level: int
    centrality: float = 0.0
    source_sentences: list[str] = field(default_factory=list)
    parent_id: str | None = None


# ---------------------------------------------------------------------------
# Stage 3a: SVO extraction constrained to concept-bearing sentences
# ---------------------------------------------------------------------------

def _phrase_in_text(phrase: str, text_lower: str) -> bool:
    return phrase.lower() in text_lower


def extract_svo_relations(
    sentences: list[str],
    nlp: spacy.Language,
    concept_phrases: set[str],
) -> list[Relation]:
    """Extracts subject-verb-object relations, but ONLY records a relation if
    at least one of subj/obj overlaps with a known concept phrase from
    Stage 2. This is the key constraint that keeps output cleaner than raw
    OpenIE: we're not extracting every fact in the document, only relations
    that connect things we've already identified as important concepts.
    """
    relations: list[Relation] = []
    concept_phrases_lower = {c.lower() for c in concept_phrases}

    for sent_text in sentences:
        doc = nlp(sent_text)
        sent_lower = sent_text.lower()

        # quick skip: if no known concept phrase appears in this sentence at
        # all, don't bother parsing it for relations (saves time, and avoids
        # pulling in irrelevant SVOs from filler sentences)
        if not any(_phrase_in_text(c, sent_lower) for c in concept_phrases_lower):
            continue

        for token in doc:
            if token.pos_ != "VERB":
                continue

            subjects = [c for c in token.children if c.dep_ in ("nsubj", "nsubjpass")]
            objects = [
                c for c in token.children
                if c.dep_ in ("dobj", "pobj", "attr", "oprd")
            ]
            # also walk one hop further for prepositional objects
            # e.g. "split water into oxygen" -> verb 'split', dobj 'water',
            # prep 'into' -> pobj 'oxygen'
            for c in token.children:
                if c.dep_ == "prep":
                    objects.extend(g for g in c.children if g.dep_ == "pobj")

            if not subjects or not objects:
                continue

            for subj_tok, obj_tok in itertools.product(subjects, objects):
                subj_span = _expand_to_noun_phrase(subj_tok)
                obj_span = _expand_to_noun_phrase(obj_tok)

                subj_text = subj_span.text.strip()
                obj_text = obj_span.text.strip()

                if len(subj_text) < MIN_CONCEPT_LEN_CHARS or len(obj_text) < MIN_CONCEPT_LEN_CHARS:
                    continue
                # Block any token that's a pronoun, determiner, or relativizer
                _SKIP_TOKENS = {
                    "it", "this", "that", "these", "those", "which", "who",
                    "what", "they", "them", "their", "its", "he", "she",
                    "place", "stage",  # overly generic SVO objects
                }
                if subj_text.lower() in _SKIP_TOKENS or obj_text.lower() in _SKIP_TOKENS:
                    continue

                relations.append(
                    Relation(
                        subj=subj_text,
                        verb=token.lemma_,
                        obj=obj_text,
                        sentence=sent_text,
                    )
                )

    return relations


def _expand_to_noun_phrase(token: spacy.tokens.Token) -> spacy.tokens.Span:
    """Expands a single token to its full noun phrase by including
    contiguous compound/amod/det children, so 'reactions' inside
    'light-dependent reactions' becomes the full phrase, not just the head
    noun."""
    doc = token.doc
    left = token.i
    right = token.i
    for child in token.children:
        if child.dep_ in ("compound", "amod", "det", "nummod"):
            left = min(left, child.i)
            right = max(right, child.i)
    return doc[left:right + 1]


# ---------------------------------------------------------------------------
# Stage 3b: Semantic similarity graph between concept-bearing sentences
# ---------------------------------------------------------------------------

def build_similarity_edges(
    concept_candidates: list[ConceptCandidate],
    embedder: SentenceTransformer,
    threshold: float = SIMILARITY_EDGE_THRESHOLD,
) -> list[tuple[str, str, float]]:
    """Embeds each concept phrase itself (not full sentences -- phrase-level
    embeddings give cleaner similarity signal for graph-building than
    sentence-level, since two concepts can appear in dissimilar sentences but
    still be semantically close). Returns edges above threshold."""
    phrases = [c.text for c in concept_candidates]
    if len(phrases) < 2:
        return []

    embeddings = embedder.encode(phrases, normalize_embeddings=True)
    sim_matrix = cosine_similarity(embeddings)

    edges = []
    n = len(phrases)
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(sim_matrix[i, j])
            if sim >= threshold:
                edges.append((phrases[i], phrases[j], sim))
    return edges


# ---------------------------------------------------------------------------
# Stage 3c: Fuse into one weighted graph
# ---------------------------------------------------------------------------

def build_fused_graph(
    concept_candidates: list[ConceptCandidate],
    svo_relations: list[Relation],
    similarity_edges: list[tuple[str, str, float]],
) -> nx.DiGraph:
    g = nx.DiGraph()

    concept_set = {c.text for c in concept_candidates}
    for c in concept_candidates:
        g.add_node(c.text, concept_score=c.score, source=c.source)

    # SVO edges (directed, higher trust weight). Only keep edges where BOTH
    # ends are in our concept set -- otherwise we'd reintroduce the same
    # noise problem OpenIE had.
    for rel in svo_relations:
        subj_match = _best_concept_match(rel.subj, concept_set)
        obj_match = _best_concept_match(rel.obj, concept_set)
        if subj_match and obj_match and subj_match != obj_match:
            w = SVO_EDGE_WEIGHT_BOOST
            if g.has_edge(subj_match, obj_match):
                g[subj_match][obj_match]["weight"] += w
                g[subj_match][obj_match]["relations"].append(rel.verb)
            else:
                g.add_edge(subj_match, obj_match, weight=w, relations=[rel.verb], kind="svo")

    # similarity edges (undirected in meaning, but we add both directions
    # with lower weight so PageRank treats them as soft connections)
    for a, b, sim in similarity_edges:
        if g.has_edge(a, b):
            g[a][b]["weight"] += sim
        else:
            g.add_edge(a, b, weight=sim, relations=[], kind="similarity")
        if g.has_edge(b, a):
            g[b][a]["weight"] += sim
        else:
            g.add_edge(b, a, weight=sim, relations=[], kind="similarity")

    return g


def _best_concept_match(phrase: str, concept_set: set[str]) -> str | None:
    """SVO-extracted spans rarely match a Stage-2 concept phrase exactly
    (e.g. 'the light energy' vs 'light energy'), so do fuzzy containment
    matching: prefer exact match, then substring match in either direction,
    picking the longest concept that matches."""
    phrase_lower = phrase.lower().strip()
    if phrase_lower in {c.lower() for c in concept_set}:
        for c in concept_set:
            if c.lower() == phrase_lower:
                return c

    candidates = [
        c for c in concept_set
        if c.lower() in phrase_lower or phrase_lower in c.lower()
    ]
    if not candidates:
        return None
    return max(candidates, key=len)


# ---------------------------------------------------------------------------
# Stage 4: Hierarchy construction
# ---------------------------------------------------------------------------

def compute_centrality(g: nx.DiGraph) -> dict[str, float]:
    if g.number_of_edges() == 0:
        # no edges at all (degenerate case) -- fall back to concept_score
        return {n: g.nodes[n].get("concept_score", 0.0) for n in g.nodes}
    try:
        return nx.pagerank(g, weight="weight", max_iter=200)
    except nx.PowerIterationFailedConvergence:
        # fall back to degree centrality if pagerank doesn't converge
        # (rare, but can happen on small/weird graphs)
        return nx.degree_centrality(g)


def _elbow_cutoff(scores: list[float], max_k: int) -> int:
    """Simple elbow heuristic: sort scores descending, find the largest
    relative drop between consecutive ranks within the first max_k, cut
    there. Falls back to max_k if no clear elbow."""
    if len(scores) <= 1:
        return len(scores)
    sorted_scores = sorted(scores, reverse=True)[: max_k + 1]
    drops = [
        (sorted_scores[i] - sorted_scores[i + 1]) / (sorted_scores[i] + 1e-9)
        for i in range(len(sorted_scores) - 1)
    ]
    if not drops:
        return len(sorted_scores)
    best_cut = max(range(len(drops)), key=lambda i: drops[i]) + 1
    return max(2, min(best_cut, max_k))


def build_hierarchy(
    pre: PreprocessedDoc,
    g: nx.DiGraph,
    centrality: dict[str, float],
    embedder: SentenceTransformer,
) -> nx.DiGraph:
    """Returns a NEW DiGraph representing the mindmap tree: a 'ROOT' node,
    level-1 branch nodes, and deeper levels attached via 'parent' edges.
    Node attrs: level, centrality, source_sentences.
    """
    tree = nx.DiGraph()
    tree.add_node("ROOT", label=_infer_doc_title(pre), level=0)

    has_headings = any(s.heading for s in pre.sections)

    if has_headings:
        _build_hierarchy_from_headings(pre, g, centrality, embedder, tree)
    else:
        _build_hierarchy_from_centrality(g, centrality, embedder, tree)

    return tree


def _infer_doc_title(pre: PreprocessedDoc) -> str:
    """Best-effort root label: use the first section's heading if it reads
    like a document title (short, first in doc), else fall back to a
    placeholder Stage 5/6 should replace with something better (e.g. the
    SLM can synthesize a title from the full text)."""
    if pre.sections and pre.sections[0].heading:
        return pre.sections[0].heading
    return "Document"  # Stage 6 (SLM) should overwrite this with a real title


def _build_hierarchy_from_headings(
    pre: PreprocessedDoc,
    g: nx.DiGraph,
    centrality: dict[str, float],
    embedder: SentenceTransformer,
    tree: nx.DiGraph,
) -> None:
    """Headings become level-1 nodes (strong structural prior). Concepts get
    assigned to whichever section's sentences they predominantly appear in,
    then clustered into level-2/3 within that section using the fused
    graph + embedding similarity."""
    section_concepts: dict[int, list[str]] = {i: [] for i in range(len(pre.sections))}

    # assign each concept node to the section it appears most in
    for node in g.nodes:
        node_lower = node.lower()
        counts = [0] * len(pre.sections)
        for sec_idx, sec in enumerate(pre.sections):
            for sent in sec.sentences:
                if node_lower in sent.lower():
                    counts[sec_idx] += 1
        if max(counts, default=0) > 0:
            best_sec = max(range(len(counts)), key=lambda i: counts[i])
            section_concepts[best_sec].append(node)

    root_label_lower = tree.nodes["ROOT"]["label"].lower()

    for sec_idx, sec in enumerate(pre.sections):
        heading_label = sec.heading or f"Section {sec_idx + 1}"
        heading_lower = heading_label.lower()
        sec_node_id = f"L1_{sec_idx}"
        tree.add_node(
            sec_node_id,
            label=heading_label,
            level=1,
            centrality=1.0,
            source_sentences=sec.sentences[:2],
        )
        tree.add_edge("ROOT", sec_node_id)

        concepts_here = section_concepts.get(sec_idx, [])

        # Drop concepts that duplicate the heading or root title — these
        # produce "Photosynthesis > Photosynthesis" or "Calvin Cycle > Calvin Cycle"
        # self-reference nodes that add nothing to the mindmap.
        concepts_here = [
            c for c in concepts_here
            if c.lower() not in (heading_lower, root_label_lower)
            and not heading_lower.startswith(c.lower())
            and not c.lower().startswith(heading_lower)
        ]

        if not concepts_here:
            continue

        _cluster_and_attach(
            concepts_here, g, centrality, embedder, tree,
            parent_id=sec_node_id, parent_level=1, sentences=sec.sentences,
        )


def _build_hierarchy_from_centrality(
    g: nx.DiGraph,
    centrality: dict[str, float],
    embedder: SentenceTransformer,
    tree: nx.DiGraph,
) -> None:
    """No headings available -- infer level-1 branches purely from PageRank
    centrality on the fused graph, then cluster remaining concepts under
    them."""
    all_nodes = list(g.nodes)
    if not all_nodes:
        return

    scores = [centrality.get(n, 0.0) for n in all_nodes]
    k = _elbow_cutoff(scores, MAX_LEVEL1_NODES_NO_HEADINGS)
    ranked = sorted(all_nodes, key=lambda n: centrality.get(n, 0.0), reverse=True)
    level1_nodes = ranked[:k]
    remaining_nodes = ranked[k:]

    for n in level1_nodes:
        node_id = f"L1_{_safe_id(n)}"
        tree.add_node(node_id, label=n, level=1, centrality=centrality.get(n, 0.0))
        tree.add_edge("ROOT", node_id)

    if remaining_nodes:
        _cluster_and_attach(
            remaining_nodes, g, centrality, embedder, tree,
            parent_id=None, parent_level=1,
            sentences=[], level1_label_to_id={n: f"L1_{_safe_id(n)}" for n in level1_nodes},
        )


def _safe_id(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text)[:40]


def _cluster_and_attach(
    concepts: list[str],
    g: nx.DiGraph,
    centrality: dict[str, float],
    embedder: SentenceTransformer,
    tree: nx.DiGraph,
    parent_id: str | None,
    parent_level: int,
    sentences: list[str],
    level1_label_to_id: dict[str, str] | None = None,
) -> None:
    """Clusters a list of concept nodes and attaches them as children
    (level = parent_level + 1) of parent_id. If parent_id is None, each
    concept is instead routed to its nearest level-1 node by graph proximity
    (used in the no-headings path, where 'parent' isn't fixed up front).

    Clustering uses embeddings, but parent ROUTING (when parent_id is None)
    uses graph shortest-path proximity in the fused graph first, falling
    back to embedding similarity only if the concept is disconnected from
    all level-1 nodes in the graph. This matters because pure embedding
    similarity conflates "similar topic" with "parent-child relationship" --
    e.g. chlorophyll and carotenoids are similar (both pigments) but neither
    is the other's parent; graph proximity (via shared SVO/co-occurrence
    edges) is a better signal for actual hierarchical relatedness.
    """
    if not concepts:
        return

    if parent_id is not None:
        # fixed parent (heading case): just cluster into sub-groups under it
        n_clusters = max(1, min(len(concepts) // 3, 5))
        _attach_clustered_children(concepts, embedder, tree, parent_id, parent_level, n_clusters)
        return

    # no fixed parent: route each concept to nearest level-1 node via graph
    assert level1_label_to_id is not None
    level1_nodes = list(level1_label_to_id.keys())

    undirected = g.to_undirected()
    for concept in concepts:
        best_target, best_dist = None, float("inf")
        for l1 in level1_nodes:
            try:
                dist = nx.shortest_path_length(undirected, concept, l1, weight="weight")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            if dist < best_dist:
                best_dist, best_target = dist, l1

        if best_target is None:
            # disconnected from everything -- fall back to embedding sim
            best_target = _most_similar_by_embedding(concept, level1_nodes, embedder)

        parent_node_id = level1_label_to_id[best_target]
        child_id = f"L2_{_safe_id(concept)}"
        if child_id in tree:
            continue
        tree.add_node(
            child_id, label=concept, level=parent_level + 1,
            centrality=centrality.get(concept, 0.0),
        )
        tree.add_edge(parent_node_id, child_id)


def _attach_clustered_children(
    concepts: list[str],
    embedder: SentenceTransformer,
    tree: nx.DiGraph,
    parent_id: str,
    parent_level: int,
    n_clusters: int,
    max_direct_attach: int = 6,
) -> None:
    """Attaches concept nodes as children of parent_id, optionally grouping
    them into intermediate cluster nodes.

    Key design decision: only create intermediate grouping nodes (the ones
    that get slash placeholder labels) when a cluster is large enough to
    NEED grouping (> max_direct_attach members total, or a single cluster
    has > 4 members). For small concept sets, attach directly -- a 3-node
    intermediate group adds visual clutter and creates the slash-label
    problem without giving students any useful organizational benefit.
    """
    # For small concept sets: attach directly, no clustering needed
    if len(concepts) <= max_direct_attach or n_clusters <= 1:
        for c in concepts:
            child_id = f"{parent_id}_{_safe_id(c)}"
            if child_id not in tree:
                tree.add_node(child_id, label=c, level=parent_level + 1)
                tree.add_edge(parent_id, child_id)
        return

    embeddings = embedder.encode(concepts, normalize_embeddings=True)
    clustering = AgglomerativeClustering(n_clusters=n_clusters, metric="cosine", linkage="average")
    labels = clustering.fit_predict(embeddings)

    clusters: dict[int, list[str]] = {}
    for concept, lbl in zip(concepts, labels):
        clusters.setdefault(lbl, []).append(concept)

    for cluster_idx, members in clusters.items():
        if len(members) <= 3:
            # Small cluster: attach members directly to parent -- no intermediate
            # grouping node. This eliminates the slash-label problem entirely for
            # small clusters since the SLM was reliably skipping group-node relabeling
            # even when asked to do it (observed: SLM edited children correctly but
            # left the "concept1 / concept2 / concept3" group label unchanged).
            for m in members:
                leaf_id = f"{parent_id}_{_safe_id(m)}"
                if leaf_id not in tree:
                    tree.add_node(leaf_id, label=m, level=parent_level + 1)
                    tree.add_edge(parent_id, leaf_id)
        else:
            # Large cluster (4+ members): an intermediate grouping node is
            # genuinely useful here. SLM will rewrite the placeholder label.
            group_id = f"{parent_id}_group{cluster_idx}"
            placeholder_label = " / ".join(members[:3])
            tree.add_node(group_id, label=placeholder_label, level=parent_level + 1, is_placeholder_group=True)
            tree.add_edge(parent_id, group_id)
            for m in members:
                leaf_id = f"{group_id}_{_safe_id(m)}"
                if leaf_id not in tree:
                    tree.add_node(leaf_id, label=m, level=parent_level + 2)
                    tree.add_edge(group_id, leaf_id)


def _most_similar_by_embedding(concept: str, candidates: list[str], embedder: SentenceTransformer) -> str:
    emb = embedder.encode([concept] + candidates, normalize_embeddings=True)
    sims = cosine_similarity([emb[0]], emb[1:])[0]
    return candidates[int(np.argmax(sims))]


# ---------------------------------------------------------------------------
# Pretty-print helper for debugging
# ---------------------------------------------------------------------------

def print_tree(tree: nx.DiGraph, node: str = "ROOT", indent: int = 0) -> None:
    label = tree.nodes[node].get("label", node)
    cent = tree.nodes[node].get("centrality")
    cent_str = f"  (c={cent:.3f})" if isinstance(cent, float) else ""
    print("  " * indent + f"- {label}{cent_str}")
    for child in tree.successors(node):
        print_tree(tree, child, indent + 1)