# Benchmark Fixtures

These fixtures are for repeatable MLX vs Ollama comparison work.

## Files
- `tiny-factual.fixture.json` — small control prompt for cold/warm latency and output-cleanliness checks
- `format-restraint.fixture.json` — mid-sized exact-shape fixture for JSON-only output and chatter suppression checks
- `long-context-retrieval.fixture.json` — metadata and expected facts for the long-context retrieval run
- `long-context-retrieval.prompt.txt` — exact shared prompt text for the long-context retrieval run

## Rules
- Use the same fixture text across MLX and Ollama runs
- Do not silently rewrite the prompt between lanes
- Capture latency, correctness, and output cleanliness together
- Treat extra chatter, repeated bullets, leaked reasoning/wrapper text, or JSON-shape violations as output-quality problems even when the facts are correct

## Helper scripts
- `../scripts/list-fixtures` — show the available fixture ids and descriptions
- `../scripts/run-fixture <fixture.json>` — send one fixture to the MLX service and save `request.json`, `response.json`, and `summary.json` under `tmp/fixture-runs/`
