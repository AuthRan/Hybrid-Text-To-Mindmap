# NLP-First Mindmap Pipeline

Text -> hierarchical mindmap JSON, using classical NLP for structure
(graph centrality + clustering) and a local SLM (Qwen2.5-3B) only as a final
polish/merge step — never to infer structure from scratch.

## What was actually tested vs. not

Built in a sandboxed environment with no access to GitHub Releases or
HuggingFace Hub, so the ML-model-dependent parts (spaCy's trained pipeline,
sentence-transformers embeddings, KeyBERT) could **not** be run end-to-end
there. What WAS tested in that sandbox, with passing results:

- `split_into_sections` — heading detection, both structured and pure-prose
  fallback paths
- `merge_concept_candidates` — case-insensitive dedup, score normalization,
  `source="both"` tagging
- `_elbow_cutoff` — verified against a clear-elbow distribution, a smooth
  decline, and a too-few-items edge case
- `_parse_slm_edits` — markdown-fence stripping, and fail-soft behavior on
  garbage LLM output (returns `[]` rather than crashing)
- `apply_slm_edits` — relabel, merge (with source-sentence combination),
  prune-with-descendants, and unknown-node-id robustness
- `tree_to_json` / `_flatten_tree_for_prompt` — correct nesting and
  parent-id chains

**Not yet run**: anything touching spaCy dependency parsing
(`extract_svo_relations`, `_expand_to_noun_phrase`), sentence-transformer
embeddings (`build_similarity_edges`, clustering), or an actual call to your
local Qwen server. Run the steps below on your machine and watch the
diagnostic prints — they'll surface real issues fast.

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_trf   # best quality, recommended
# or, faster / smaller:
python -m spacy download en_core_web_sm
```

The first run will also download `all-mpnet-base-v2` from HuggingFace Hub
(~420MB) — needs internet once, then it's cached in `~/.cache/huggingface`.

`requirements.txt` lists `fastcoref` but it's optional — only needed if you
set `USE_COREF = True` in `stage1_2_preprocessing.py`. Leave it off
initially; see the note in that file for when it's worth turning on.

## Running it

```bash
# Make sure your local Qwen server is up first (e.g. `ollama serve`,
# and `ollama pull qwen2.5:3b` if you haven't already)

python run_pipeline.py sample_photosynthesis.txt output.json

# To inspect the raw NLP-only output before SLM polishing (useful for
# debugging Stages 1-4 in isolation):
python run_pipeline.py sample_photosynthesis.txt output_raw.json --no-slm
```

Every stage prints its intermediate output to stdout — concept candidates
with scores and source (keybert/textrank/both), extracted SVO relations,
graph size, the tree structure before and after SLM editing. Read these
before looking at the final JSON; if something looks off in the mindmap,
the stage-by-stage prints will tell you which step introduced it.

## Likely first issues on your machine (and what they mean)

**`OSError: No spaCy model found`** — the `en_core_web_trf`/`en_core_web_sm`
download didn't complete. Re-run the `spacy download` command.

**Ollama response shape mismatch in `call_local_slm`** — the code assumes
Ollama's `/api/chat` endpoint, which returns `{"message": {"content": ...}}`.
If you're running Qwen through vLLM, llama.cpp's server, or
text-generation-webui instead, they typically expose an
OpenAI-compatible `/v1/chat/completions` endpoint, which returns
`{"choices": [{"message": {"content": ...}}]}` instead. Adjust the one line
in `call_local_slm` that does `response.json()["message"]["content"]`
accordingly — everything downstream (`_parse_slm_edits`, `apply_slm_edits`)
is decoupled from the request shape, so this should be a small change.

**Few or zero SVO relations extracted** — if `extract_svo_relations` returns
very little, your text might be using passive voice heavily, or
`en_core_web_sm`'s parses might be too noisy for your domain. Try
`en_core_web_trf` if you're on `sm`. This is also exactly the kind of gap
the similarity-edge graph (Stage 3b) is meant to backstop — check that
`build_similarity_edges` is producing a reasonable edge count too, since the
fused graph should still work even with sparse SVO coverage.

**SLM "fixes" things you didn't want fixed (e.g. moves nodes around)** — the
system prompt explicitly forbids restructuring, but smaller models don't
always follow instructions perfectly. `apply_slm_edits` only honors
`new_label` / `merge_into` / `prune` fields and ignores anything else the
model might try to do (like adding new nodes), so the worst case should be
"some labels/merges/prunes are weird," not "the tree shape changed
unexpectedly." If you see bad merges specifically, that's the first thing
I'd tighten — either via prompt examples (few-shot) or by adding a
similarity-score gate before even offering merge candidates to the SLM.

## File map

- `stage1_2_preprocessing.py` — segmentation, heading detection, KeyBERT +
  TextRank concept extraction
- `stage3_4_hierarchy.py` — SVO + similarity graph construction, PageRank
  centrality, hierarchy building (headings-as-prior or centrality-elbow
  fallback), agglomerative clustering for grouping
- `stage5_6_label_and_slm.py` — grounding-sentence attachment, JSON
  serialization, the SLM prompt + defensive parsing + edit application
- `run_pipeline.py` — orchestrates all six stages end-to-end, CLI entry point
