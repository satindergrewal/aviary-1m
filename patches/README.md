# Patches

## rocmfpx-hy3-parser-fixes.patch (630 lines, July 8, 2026)

Working-tree fixes for running Tencent Hy3 (hy_v3) on the ROCmFPX community fork, developed during real Mac sessions and documented here per the publish-everything policy. Covers:

- Hy3 reasoning-tag support in the autoparser (`<think:opensource>` delimiters; the template prefills the opening tag, so delimiter-style parsing is required)
- Content termination at the fullwidth-bar eos token so `<|hy_eos:opensource|>` never leaks into visible output (plain and tools grammar paths, streaming-safe)
- Hy3 tagged tool-call format wiring (`<tool_call:opensource>NAME<tool_sep:opensource>` + `<arg_key/arg_value>` pairs)
- Single-tool-call grammar enforcement when parallel calls are not requested (prevents low-bit quants from emitting a malformed second call that fails strict finalize)
- webui build repairs (async/await misuse and a half-shipped image-capping module)

Apply on top of charlie12345/ROCmFPX. The upstream path for the same functionality is pwilkin's autoparser follow-up to ggml-org/llama.cpp#25395; these patches are the fork-side equivalent, offered upstream to the fork separately.
