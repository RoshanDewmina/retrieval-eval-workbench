# Benchmark history

`latest.json` is preserved as the original development receipt. It evaluated an exposed question set and did not record immutable encoder revision or weight hashes. Its model provenance is therefore **unknown**, and its metric names used the earlier document-ID/term-overlap proxies.

The frozen implementation writes any new run to `frozen-run.json`. Root will supply independently authored, frozen evaluation cases before treating a new receipt as independent evaluation evidence. A failed run must remain as a receipt with its nonzero exit status.
