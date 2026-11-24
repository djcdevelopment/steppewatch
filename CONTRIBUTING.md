# Contributing to Steppewatch

Real-world field reports, adversarial benchmarks, bug fixes, and clean adapters are warmly welcomed.

## Guidelines

1. **Preserve the Read-Only & Offline Invariants:**
   - Never mutate host agent transcripts, global configs, or live SQLite stores.
   - Keep runtime dependencies minimal; optional ML models must degrade cleanly and honor offline flags (`--force-hash`, `--offline`, `HF_HUB_OFFLINE=1`).
2. **Synthetic Offline Fixtures Only:**
   - Never commit real private transcripts, prompts, personal home paths, or credentials.
   - Keep unit tests fast, deterministic, and 100% offline.
3. **Keep PRs Focused:**
   - Separate transcript discovery/parsing adapters from scoring, threshold, or model changes so each change can be reviewed and validated cleanly.
4. **Honest Claims & Methodology:**
   - Distinguish synthetic smoke tests from manually adjudicated real-session precision, relevance scores from logical entailment, and character-derived estimates from API billing.

## Running the Test Suite Locally

Python (requires only `numpy` for the offline suite):

```bash
python3 -W error::ResourceWarning -m unittest discover -s tests -v
```

`.NET 10` (optional, if you touched `dotnet/`):

```bash
dotnet test dotnet/tests/Steppewatch.Tests/Steppewatch.Tests.csproj -c Release
dotnet test dotnet/tests/Steppewatch.Onnx.Tests/Steppewatch.Onnx.Tests.csproj -c Release
```

## Field-Testing Credits

- **David Murray (`@thedavidmurray`)** — Hermes default/named-profile & read-only `state.db` adapter ([PR #1](https://github.com/djcdevelopment/steppewatch/pull/1)) and independent Apple M2 Option B real-session evaluation ([Issue #2](https://github.com/djcdevelopment/steppewatch/issues/2)).
