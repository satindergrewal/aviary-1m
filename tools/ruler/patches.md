# Local patches applied to RULER (full disclosure)

1. scripts/pred/call_api.py and scripts/eval/evaluate.py: replaced
   `from nemo.collections.asr.parts.utils.manifest_utils import read_manifest(, write_manifest)`
   with a plain jsonl reader/writer (nemo is an ASR framework; the import only read/wrote jsonl lines).
2. scripts/pred/client_wrappers.py: Azure env reads made optional (os.environ[...] to .get(...)); model2length lookup given a default. (These affect only the stock OpenAI client, which we ultimately did not use; the bridge replaced it.)
3. No changes to data generation or scoring logic.
