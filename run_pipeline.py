"""
End-to-end orchestration: text -> mindmap JSON.

Usage:
    python run_pipeline.py input.txt output_mindmap.json
    python run_pipeline.py input.txt output_mindmap.json --no-slm   (skip Stage 6, inspect raw NLP output)
    python run_pipeline.py input.txt output_mindmap.json --slm-model qwen2.5:3b --slm-endpoint http://localhost:11434/api/chat
"""

from __future__ import annotations

import argparse
import json
import sys

from stage1_2_preprocessing import (
    build_embedder,
    build_keybert,
    build_nlp_pipeline,
    extract_keybert_candidates,
    extract_textrank_candidates,
    filter_concept_candidates,
    merge_concept_candidates,
    preprocess,
)
from stage3_4_hierarchy import (
    build_fused_graph,
    build_hierarchy,
    build_similarity_edges,
    compute_centrality,
    extract_svo_relations,
    print_tree,
)
from stage5_6_label_and_slm import (
    SLMConfig,
    apply_slm_edits,
    attach_grounding_sentences,
    call_local_slm_batched_by_branch,
    tree_to_json,
)


def run_pipeline(
    raw_text: str,
    use_slm: bool = True,
    slm_config: SLMConfig | None = None,
    keybert_top_n: int = 25,
    textrank_top_n: int = 25,
) -> dict:
    slm_config = slm_config or SLMConfig()

    print("=" * 70)
    print("STAGE 1-2: Preprocessing & Concept Extraction")
    print("=" * 70)
    nlp = build_nlp_pipeline()
    embedder = build_embedder()
    kw_model = build_keybert(embedder)

    pre = preprocess(raw_text, nlp, use_coref=False)
    print(f"  {len(pre.sections)} section(s), {len(pre.all_sentences)} sentence(s)")
    has_headings = any(s.heading for s in pre.sections)
    print(f"  structural headings detected: {has_headings}")

    full_doc = nlp(pre.raw_text)
    kb_cands = extract_keybert_candidates(pre.raw_text, kw_model, top_n=keybert_top_n)
    tr_cands = extract_textrank_candidates(full_doc, top_n=textrank_top_n)
    concept_candidates = merge_concept_candidates(kb_cands, tr_cands)
    concept_candidates = filter_concept_candidates(concept_candidates, nlp)
    print(f"  {len(concept_candidates)} merged concept candidates (after POS filter)")
    for c in concept_candidates[:15]:
        print(f"    {c.score:.3f}  [{c.source:8s}]  {c.text}")

    print()
    print("=" * 70)
    print("STAGE 3: Relation Extraction")
    print("=" * 70)
    concept_phrase_set = {c.text for c in concept_candidates}
    svo_relations = extract_svo_relations(pre.all_sentences, nlp, concept_phrase_set)
    print(f"  {len(svo_relations)} SVO relations extracted")
    for r in svo_relations[:10]:
        print(f"    {r.subj}  --[{r.verb}]-->  {r.obj}")

    similarity_edges = build_similarity_edges(concept_candidates, embedder)
    print(f"  {len(similarity_edges)} similarity edges (threshold-based)")

    fused_graph = build_fused_graph(concept_candidates, svo_relations, similarity_edges)
    print(f"  fused graph: {fused_graph.number_of_nodes()} nodes, {fused_graph.number_of_edges()} edges")

    print()
    print("=" * 70)
    print("STAGE 4: Hierarchy Construction")
    print("=" * 70)
    centrality = compute_centrality(fused_graph)
    tree = build_hierarchy(pre, fused_graph, centrality, embedder)
    print(f"  tree built: {tree.number_of_nodes()} nodes")
    print()
    print_tree(tree)

    print()
    print("=" * 70)
    print("STAGE 5: Label Grounding")
    print("=" * 70)
    attach_grounding_sentences(tree, pre)
    print("  grounding sentences attached")

    if not use_slm:
        print("\n[--no-slm] skipping Stage 6, returning raw NLP-derived tree")
        return tree_to_json(tree)

    print()
    print("=" * 70)
    print("STAGE 6: SLM Checker / Merger")
    print("=" * 70)
    tree_json = tree_to_json(tree)
    total_nodes = sum(1 for _ in tree.nodes)
    print(f"  {total_nodes} total nodes, batching SLM calls by branch ({slm_config.model})")

    try:
        edits = call_local_slm_batched_by_branch(tree_json, slm_config)
        print(f"  received {len(edits)} total edits from SLM across all batches")
        final_tree = apply_slm_edits(tree, edits)
    except Exception as e:
        print(f"  [stage6] SLM call failed ({e}), falling back to raw NLP tree")
        final_tree = tree

    print()
    print_tree(final_tree)

    return tree_to_json(final_tree)


def main():
    parser = argparse.ArgumentParser(description="Generate a mindmap JSON from a text file.")
    parser.add_argument("input_file", help="Path to a .txt file containing the source text")
    parser.add_argument("output_file", help="Path to write the resulting mindmap JSON")
    parser.add_argument("--no-slm", action="store_true", help="Skip Stage 6 SLM polishing")
    parser.add_argument("--slm-model", default="qwen2.5:3b")
    parser.add_argument("--slm-endpoint", default="http://localhost:11434/api/chat")
    args = parser.parse_args()

    with open(args.input_file, "r", encoding="utf-8") as f:
        raw_text = f.read()

    slm_config = SLMConfig(endpoint=args.slm_endpoint, model=args.slm_model)
    result = run_pipeline(raw_text, use_slm=not args.no_slm, slm_config=slm_config)

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nWrote mindmap JSON to {args.output_file}")


if __name__ == "__main__":
    main()