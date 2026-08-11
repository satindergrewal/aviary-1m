# Upstream PR prep: ggml-alloc leaf check in needs_realloc

**Status: AWAITING HIS GO. Nothing pushed anywhere third-party.**

## The defect (as it would be described upstream)

`ggml_gallocr_needs_realloc` decides whether a previously measured allocation
layout can be reused for a new graph. It validates every node and every node's
direct sources -- but never the graph's leafs. A graph input consumed only
through views is sized at 0 on the consumer side (`view_src` set), so a leaf
whose size GROWS between two topology-matching graphs is invisible to the
check: `ggml_gallocr_alloc_graph` then assigns it the stored (smaller) offset,
and any write of the tensor's full extent legally overruns the next leaf's
allocation.

Observed in the wild on Apple Metal with a DeepSeek-V4-class model: a sliding
window KQ mask leaf stepping 131072 -> 262144 bytes (padded n_kv 256 -> 512)
between two same-topology prefill ubatch graphs kept its old offset and
overwrote the adjacent `inp_pos` leaf with F16 -inf mask bytes, corrupting
rope positions and every downstream KV write. Symptom: deterministic garbage
completions whenever a prompt chunk size made consecutive graphs
topology-identical while the padded KV span grew.

## The fix (10 lines, mirrors the existing node loop)

```c
    // in ggml_gallocr_needs_realloc, before the node loop:
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        struct leaf_alloc * leaf_alloc = &galloc->leaf_allocs[i];

        if (!ggml_gallocr_node_needs_realloc(galloc, leaf, &leaf_alloc->leaf)) {
#ifndef NDEBUG
            GGML_LOG_DEBUG("%s: leaf %s is not valid\n", __func__, leaf->name);
#endif
            return true;
        }
    }
```

## Repro sketch for the PR (to be turned into a test if maintainers want one)

1. Build graph A: leaf L of size S, plus enough nodes for a stable topology.
2. `ggml_gallocr_alloc_graph(galloc, A)` -- layout measured.
3. Build graph B: identical topology, same node/leaf counts, leaf L' of size 2S,
   consumed through a view.
4. Without the fix: `needs_realloc` returns false, L' gets L's offset, and
   writing L' overruns the neighbouring leaf. With the fix: realloc fires.

## Verification already done in our tree

- The exact failing configuration (DSV4 paged, ub=256) went from deterministic
  corruption to byte-exact against the static path at 616, 1018 and 16016
  prompt tokens, and to identical perplexity (3.8875 +/- 0.09889) at ctx 2048.
- No regression across the full local gate battery (ub 128/255/256/384/509/512).

## What must NOT go upstream

The DS4P_* diagnostic prints living near this code in our tree
(DS4P_ALLOC_TRACE etc.) are lane-local. The PR is the leaf loop only,
rebased onto upstream master, no other hunks.

## Apply-ready artifact

`0001-ggml-alloc-check-leafs-in-needs-realloc.patch` sits next to this doc. The hunk header
uses a function-name anchor rather than line numbers -- regenerate offsets against upstream
master at apply time (`git apply --3way` or hand-apply; the insertion point is immediately
after the n_leafs count check, before the node loop).

## Suggested PR title

    ggml-alloc: check leafs in ggml_gallocr_needs_realloc

Co-authored-by credit per the lane's credit-ported-code rule.
