"""
Stage 5: Label Generation (still non-LLM)
Stage 6: SLM as Checker / Merger

Design notes:
- Stage 5 is deliberately "dumb": it just attaches the best representative
  raw text to each node (closest source sentence, or the concept phrase
  itself). No attempt at fluent labeling here -- that's Stage 6's job.

- Stage 6 is the ONLY place an LLM/SLM touches this pipeline, and its task is
  narrow and bounded on purpose:
      INPUT:  the full tree as JSON (labels, levels, parent/child structure,
              a couple of grounding sentences per node)
      OUTPUT: the SAME tree structure, with:
                (a) labels rewritten as short, clean mindmap phrases
                (b) near-duplicate sibling nodes merged
                (c) obviously misplaced nodes flagged (optionally moved)
                (d) low-value leaf nodes pruned (e.g. stray single words)
  Crucially the SLM is NOT asked to invent structure -- it only edits
  structure that already exists. This is what should make a 3B model
  perform far better here than in your original OpenIE pipeline, where it
  had to do structure-inference AND cleanup at once.

- The prompt enforces strict JSON-in/JSON-out with the same node IDs, so a
  buggy/truncated LLM response can be validated and partially merged back
  (e.g. "if it touched a node, accept the edit; if it dropped a node
  entirely without an explicit prune marker, keep the original" -- never
  silently lose data because of a parsing hiccup).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import networkx as nx
import requests  # for talking to a local Qwen server (e.g. via Ollama / vLLM / llama.cpp server)


# ---------------------------------------------------------------------------
# Stage 5: Raw label / grounding assignment
# ---------------------------------------------------------------------------

def attach_grounding_sentences(
    tree: nx.DiGraph,
    pre,  # PreprocessedDoc
    max_sentences_per_node: int = 2,
) -> None:
    """For every node that doesn't already have source_sentences (headings
    and section nodes get them in Stage 4), find the most relevant original
    sentences containing that node's label text. This grounding is what lets
    Stage 6 produce labels that are faithful to the source rather than
    hallucinated, and is also just generally useful to keep around for the
    final mindmap (e.g. as hover-text / tooltips in the rendered map)."""
    for node_id, data in tree.nodes(data=True):
        if data.get("source_sentences"):
            continue
        label = data.get("label", "")
        label_lower = label.lower()
        matches = [s for s in pre.all_sentences if label_lower in s.lower()]
        data["source_sentences"] = matches[:max_sentences_per_node]


def tree_to_json(tree: nx.DiGraph, root: str = "ROOT") -> dict:
    """Serializes the tree to a nested JSON structure, the format Stage 6
    will operate on (and the format your mindmap renderer presumably already
    expects, matching your option-1 Gemini output shape)."""

    def _node_to_dict(node_id: str) -> dict:
        data = tree.nodes[node_id]
        children = list(tree.successors(node_id))
        return {
            "id": node_id,
            "label": data.get("label", ""),
            "level": data.get("level", 0),
            "centrality": round(data.get("centrality", 0.0), 4),
            "source_sentences": data.get("source_sentences", []),
            "is_placeholder_group": data.get("is_placeholder_group", False),
            "children": [_node_to_dict(c) for c in children],
        }

    return _node_to_dict(root)


# ---------------------------------------------------------------------------
# Stage 6: SLM checker / merger
# ---------------------------------------------------------------------------

SLM_SYSTEM_PROMPT = """You are a mindmap editor. You receive a small hierarchical \
JSON tree (one branch of a larger mindmap, extracted from a document using \
NLP). The PARENT-CHILD STRUCTURE is already correct -- do not change it. \
Your only job is to clean up labels.

Every node has a "label" and "source_sentences" (the original text it came \
from). Some labels look like "word1 / word2 / word3" -- these are \
placeholder cluster labels and MUST be rewritten. Every other label should \
also be reviewed.

For EVERY node in the input, output exactly one edit object. Do not skip \
any node, even if its label already looks fine (in that case, just repeat \
it unchanged in "new_label").

Rules per node:
1. If "label" contains " / " (a placeholder group): replace it with a short \
(2-5 word) phrase that captures what its children have in common, grounded \
in source_sentences. Never just pick one of the slash-separated words.
2. If "label" is a single stray word/phrase with no real meaning on its own \
(e.g. "specifically", "which", "critical role", "this process", a bare \
pronoun, or a sentence fragment that isn't a concept) -- set "prune": true.
3. Otherwise, tighten the label to 2-6 clean words (fix grammar, drop \
filler words like "the"/"a" at the start, fix case) but keep its meaning.
4. If two SIBLING nodes (same parent in this input) clearly mean the same \
thing, set "merge_into" on the worse-labeled one, pointing at the other's id.

Respond with ONLY a valid JSON array, no markdown fences, no commentary, no \
explanation before or after. One object per node:

[
  {"id": "<id>", "new_label": "<clean label>", "prune": false, "merge_into": null}
]

For a pruned node: {"id": "<id>", "prune": true}
For a merged node: {"id": "<id>", "merge_into": "<sibling id>"}
"""


@dataclass
class SLMConfig:
    endpoint: str = "http://localhost:11434/api/chat"  # default Ollama endpoint
    model: str = "qwen2.5:3b"
    temperature: float = 0.2
    timeout_s: int = 120
    debug: bool = False


def _flatten_tree_for_prompt(tree_json: dict) -> list[dict]:
    """Flattens the nested tree into a list of nodes (the LLM finds a flat
    list with explicit parent_id easier to edit reliably than a deeply
    nested structure, and it's much easier for us to validate/merge the
    response back against a flat list too)."""
    flat = []

    def _walk(node: dict, parent_id: str | None):
        flat.append({
            "id": node["id"],
            "label": node["label"],
            "level": node["level"],
            "parent_id": parent_id,
            "source_sentences": node["source_sentences"],
        })
        for child in node["children"]:
            _walk(child, node["id"])

    _walk(tree_json, None)
    return flat


def call_local_slm(flat_nodes: list[dict], config: SLMConfig) -> list[dict]:
    """Sends the flattened tree to a locally-running model (Ollama-style API
    by default; adjust the request shape if you're using vLLM/llama.cpp
    server/text-generation-webui instead -- the OpenAI-compatible
    /v1/chat/completions shape is a one-line change here).

    Splitting note: for short articles (500-1500 words) the flattened tree
    is small enough (typically 20-50 nodes) to send in one shot. If you
    later handle longer documents and the node count grows large, consider
    batching by top-level branch (one SLM call per level-1 subtree) to stay
    well within context and keep the model focused.
    """
    return _call_slm_on_flat_nodes(flat_nodes, config)


def call_local_slm_batched_by_branch(tree_json: dict, config: SLMConfig) -> list[dict]:
    """Calls the SLM once per top-level (level-1) branch instead of once for
    the whole tree. This is the recommended entry point for Stage 6 -- the
    single-shot whole-tree call (call_local_slm) is kept for small trees /
    debugging, but in practice a 3B model reliably under-edits when handed
    30-40+ nodes and a multi-part instruction set at once (observed: 39
    nodes in, 2 edits out). Splitting by branch keeps each call to roughly
    5-15 nodes, which is small enough for the model to actually cover
    completely -- and a coverage warning fires per-branch if it still
    doesn't, so undershoot is visible rather than silent.

    The ROOT node and level-1 nodes themselves are sent together as a small
    final call (they're usually few in number and benefit from seeing
    siblings together for merge decisions).
    """
    all_edits: list[dict] = []

    root_and_l1 = [{
        "id": tree_json["id"], "label": tree_json["label"], "level": tree_json["level"],
        "parent_id": None, "source_sentences": tree_json["source_sentences"],
    }]
    for branch in tree_json["children"]:
        root_and_l1.append({
            "id": branch["id"], "label": branch["label"], "level": branch["level"],
            "parent_id": tree_json["id"], "source_sentences": branch["source_sentences"],
        })

    print(f"  [stage6] batch 0/{ len(tree_json['children'])} (root + level-1 branches, {len(root_and_l1)} nodes)")
    all_edits.extend(_call_slm_on_flat_nodes(root_and_l1, config))

    for i, branch in enumerate(tree_json["children"], start=1):
        branch_nodes = _flatten_tree_for_prompt(branch)
        if len(branch_nodes) <= 1:
            continue  # branch with no children, nothing to clean up below it
        print(f"  [stage6] batch {i}/{len(tree_json['children'])} "
              f"(branch {branch['id']!r} = {branch['label']!r}, {len(branch_nodes)} nodes)")
        all_edits.extend(_call_slm_on_flat_nodes(branch_nodes, config))

    return all_edits


def _call_slm_on_flat_nodes(flat_nodes: list[dict], config: SLMConfig) -> list[dict]:
    """Shared single-call implementation used by both call_local_slm and
    call_local_slm_batched_by_branch."""
    user_payload = json.dumps(flat_nodes, ensure_ascii=False, indent=2)

    response = requests.post(
        config.endpoint,
        json={
            "model": config.model,
            "messages": [
                {"role": "system", "content": SLM_SYSTEM_PROMPT},
                {"role": "user", "content": user_payload},
            ],
            "stream": False,
            "options": {"temperature": config.temperature},
        },
        timeout=config.timeout_s,
    )
    response.raise_for_status()
    raw_content = response.json()["message"]["content"]

    if config.debug:
        print(f"\n[stage6][DEBUG] raw SLM response ({len(raw_content)} chars):")
        print(raw_content)
        print("[stage6][DEBUG] end of raw response\n")

    expected_ids = {n["id"] for n in flat_nodes}
    return _parse_slm_edits(raw_content, expected_node_ids=expected_ids)


def _parse_slm_edits(raw_content: str, expected_node_ids: set[str] | None = None) -> list[dict]:
    """Defensive parsing: strip markdown fences if the model added them
    anyway, and fail soft (return []) rather than crash the whole pipeline
    if the model returns garbage -- the original NLP-derived tree is always
    a valid fallback.

    If expected_node_ids is given, also checks coverage and warns loudly
    when the model silently skipped nodes -- this is the failure mode we
    saw in practice (39 nodes sent, 2 edits back): the model complies with
    the easy part of the instructions (pruning) and quietly drops the hard
    part (relabeling every node) instead of erroring, so without this check
    you'd never know it happened.
    """
    cleaned = raw_content.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        edits = json.loads(cleaned)
        if not isinstance(edits, list):
            print("[stage6] WARNING: SLM response wasn't a JSON list, ignoring edits")
            return []
    except json.JSONDecodeError as e:
        print(f"[stage6] WARNING: failed to parse SLM response as JSON ({e}), ignoring edits")
        print(f"[stage6] raw response was:\n{raw_content[:500]}")
        return []

    if expected_node_ids is not None:
        returned_ids = {e.get("id") for e in edits if isinstance(e, dict)}
        missing = expected_node_ids - returned_ids
        if missing:
            coverage_pct = 100 * len(returned_ids) / max(len(expected_node_ids), 1)
            print(
                f"[stage6] WARNING: SLM only edited {len(returned_ids)}/{len(expected_node_ids)} "
                f"nodes ({coverage_pct:.0f}% coverage). Missing: {sorted(missing)[:10]}"
                + (" ..." if len(missing) > 10 else "")
            )
            print("[stage6] Nodes not mentioned by the SLM keep their original (raw NLP) label.")

    return edits


def apply_slm_edits(tree: nx.DiGraph, edits: list[dict]) -> nx.DiGraph:
    """Applies validated edits back onto the tree. Defensive by design:
      - unknown node ids in the edit list are skipped with a warning, never crash
      - merge targets that don't exist are skipped (better to keep a
        duplicate node than silently drop content)
      - pruning a node also prunes its descendants, since a parent marked
        "too trivial" implies its children are too
    """
    tree = tree.copy()
    valid_ids = set(tree.nodes)

    for edit in edits:
        node_id = edit.get("id")
        if node_id not in valid_ids:
            print(f"[stage6] WARNING: edit references unknown node id {node_id!r}, skipping")
            continue

        if edit.get("merge_into"):
            target_id = edit["merge_into"]
            if target_id not in valid_ids or target_id == node_id:
                print(f"[stage6] WARNING: invalid merge target for {node_id!r}, skipping merge")
                continue
            _merge_node(tree, source_id=node_id, target_id=target_id)
            continue

        if edit.get("prune"):
            if node_id in tree:
                descendants = nx.descendants(tree, node_id)
                tree.remove_nodes_from(descendants | {node_id})
            continue

        if edit.get("new_label") and node_id in tree:
            tree.nodes[node_id]["label"] = edit["new_label"]

    return tree


def _merge_node(tree: nx.DiGraph, source_id: str, target_id: str) -> None:
    if source_id not in tree or target_id not in tree:
        return
    # combine grounding sentences
    src_sents = tree.nodes[source_id].get("source_sentences", [])
    tgt_sents = tree.nodes[target_id].get("source_sentences", [])
    tree.nodes[target_id]["source_sentences"] = list(dict.fromkeys(tgt_sents + src_sents))

    # re-parent source's children onto target, then drop source
    for child in list(tree.successors(source_id)):
        tree.add_edge(target_id, child)
    tree.remove_node(source_id)