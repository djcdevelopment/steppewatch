# Steppewatch Windows 10 & 12th-Gen Intel Core i5 Cookbook (`.NET 10` Native SIMD)

> **Built for everyday laptops with NO dedicated NPU and NO discrete GPU** — such as a 4-year-old **12th-Gen Intel Core i5 (Alder Lake)** running Windows 10 (`win-x64`).  
> **Zero Python, zero `venv`, zero pip/DLL friction.** Runs as a single native `.exe` or via the `.NET 10` SDK using built-in **AVX-VNNI (`vpdpbusd` `INT8`)** and **AVX2 + FMA** hardware SIMD.

---

## Why a Native `.NET 10` Binary Beats Python on a Windows 10 i5 Laptop

1. **Direct `AVX-VNNI` `INT8` Slab Math (`96.5 MiB`):** 12th-Gen Intel Core i5 (Alder Lake) introduced native **AVX-VNNI** instructions (`vpdpbusd`) directly on its CPU cores. `Steppewatch.Core` exercises `AvxVnni.MultiplyWideningAndAdd` natively inside `.NET 10` (with bit-exact `Avx2` fallback on older x64 CPUs) against the **exact `96.5 MiB` `INT8 + FP16 scales` `[384 × 262,144]` slab** — verified bit-for-bit against a scalar reference in xUnit.
2. **Zero Dependencies:** `Steppewatch.Core` and `Steppewatch.Cli` have **zero NuGet dependencies** and require **no Python install**. Copy one self-contained `steppewatch.exe` onto Windows 10 and run immediately.
3. **Full Windows Transcript Discovery:** Scans `%USERPROFILE%\.claude\projects\` (including `subagents\agent-*.jsonl`), `%USERPROFILE%\.codex\sessions\`, and `%USERPROFILE%\.hermes\` (single and multi-profile paths) read-only.

---

## Option 1: Run the Single-File `steppewatch.exe` (Zero Installs on Windows 10!)

Publish a self-contained single-file Windows 10 binary from Linux, macOS, or Windows:

```bash
dotnet publish dotnet/src/Steppewatch.Cli/Steppewatch.Cli.csproj \
  -c Release -r win-x64 --self-contained true \
  -p:PublishSingleFile=true -p:IncludeNativeLibrariesForSelfExtract=true \
  -o publish/win-x64
```

Copy `publish/win-x64/steppewatch.exe` to your Windows 10 i5 laptop and open PowerShell:

```powershell
# 1. Benchmark REAL INT8 (AVX-VNNI) vs FP32 SIMD against 262,144 vectors + thrash + repo grounding
.\steppewatch.exe bench --repo C:\path\to\any\repo --receipt .\i5_win10_receipt.json

# 2. X-Ray your own local Claude Code / Codex / Hermes transcripts (100% read-only)
.\steppewatch.exe xray --max-sessions 25 --export-episodes .\episodes\i5_history.jsonl

# 3. Train a tiny local JEPA predictor in milliseconds (use --episodes .\episodes\i5_history.jsonl once you have >= 6 runs, or --demo-synthetic)
.\steppewatch.exe train --episodes .\episodes\i5_history.jsonl --out .\checkpoints\my_i5_jepa.json

# 4. Live-watch active sessions using your newly trained local JEPA checkpoint (Ctrl+C to stop)
.\steppewatch.exe watch --follow --jepa-checkpoint .\checkpoints\my_i5_jepa.json

# 5. One-command JSON probe for Claude Code / Codex / Hermes self-tap on Windows
.\steppewatch.exe self-tap --repo C:\path\to\any\repo
```

---

## Option 2: Build & Test Directly on Windows 10 (with `.NET 10` SDK)

```powershell
git clone https://github.com/djcdevelopment/steppewatch.git
cd steppewatch
dotnet test dotnet/tests/Steppewatch.Tests/Steppewatch.Tests.csproj -c Release
dotnet run --project dotnet/src/Steppewatch.Cli/Steppewatch.Cli.csproj -c Release -- bench --repo .
```

---

## Option 3: Optional Neural `all-MiniLM-L6-v2` ONNX Runtime Backend (CPU `INT8` & Intel Iris Xe DirectML)

The core `Steppewatch.Core` and `Steppewatch.Cli` projects remain **100% zero-NuGet-dependency**. When you want true 384-D transformer embeddings (catching zero-word-overlap paraphrases with **`16.8×`** sharper cosine margin than lexical hashing), use the opt-in **`Steppewatch.Onnx`** + **`Steppewatch.Onnx.Cli`** (`steppewatch-onnx.exe`) projects — complete with a pure-C# WordPiece tokenizer (`WordPieceTokenizer.cs`), automatic `DirectML.dll 1.15.4` bundling for Windows 10 19045, and static `64 / 128` sequence buckets for DirectML.

### Publish & Run on Windows 10 (`win-x64`, DirectML + CPU)

```bash
# Publish steppewatch-onnx.exe + onnxruntime.dll + DirectML.dll (1.15.4)
dotnet publish dotnet/src/Steppewatch.Onnx.Cli/Steppewatch.Onnx.Cli.csproj \
  -c Release -r win-x64 -p:UseDirectML=true --self-contained true \
  -p:PublishSingleFile=true -p:IncludeNativeLibrariesForSelfExtract=false \
  -p:PublishTrimmed=false -p:DebugType=none \
  -o publish/win-x64-onnx
```

Run in PowerShell — `--download-model` auto-fetches Apache-2.0 `Xenova/all-MiniLM-L6-v2` (`21.9 MiB` `INT8` + `vocab.txt`) once into `%LOCALAPPDATA%\steppewatch\models\all-MiniLM-L6-v2\`, so subsequent runs need zero flags:

```powershell
# 1. One-command neural X-Ray on your own Claude Code / Codex / Hermes sessions (auto-downloads 22 MiB INT8 model once!)
.\steppewatch-onnx.exe xray --download-model --max-sessions 25 --export-episodes .\episodes\i5_neural_history.jsonl

# 2. Train your neural Tiny JEPA on YOUR real history & live-watch active sessions
.\steppewatch-onnx.exe train --episodes .\episodes\i5_neural_history.jsonl --out .\checkpoints\i5_neural_jepa.json
.\steppewatch-onnx.exe watch --follow --jepa-checkpoint .\checkpoints\i5_neural_jepa.json

# 3. Benchmark neural encoding throughput + zero-overlap paraphrase check + repo claim grounding
.\steppewatch-onnx.exe neural-bench --repo C:\path\to\any\repo

# 4. Explicit Intel Iris Xe / AMD / NVIDIA DX12 offload via DirectML (static 64/128 buckets enabled automatically)
.\steppewatch-onnx.exe neural-bench --download-model --model-variant fp32 --provider dml --repo C:\path\to\any\repo
```

### Live Verified Numbers on Dell Precision 3571 (`i5-12600H` + Intel Iris Xe, Win10 Pro 19045)

| Neural Backend (`all-MiniLM-L6-v2`, 384-D) | Model Size | `B=1` Encode | `B=8` Encode | `B=16` Encode | Stall Check (5 steps) | Full Repo (`239` syms) + Query | Rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`i5-12600H` CPU `INT8`** (`model_quantized.onnx`) | `21.9 MiB` | **`3.19 ms`** (`314/s`) | **`13.99 ms`** (`572/s`) | **`21.79 ms`** (**`734/s`**) | **`12.37 ms`** (`0.0` vs `1.0`) | **`534 ms`** / **`6.2 ms`** | **`#1`** |
| **`i5-12600H` CPU `FP32`** (`model.onnx`) | `86.2 MiB` | `4.04 ms` (`248/s`) | `18.38 ms` (`435/s`) | `28.99 ms` (`552/s`) | `13.62 ms` (`0.0` vs `1.0`) | `728 ms` / `5.8 ms` | **`#1`** |
| **Intel Iris Xe DirectML `FP32`** *(static `64/128`)* | `86.2 MiB` | `54.81 ms` (`18/s`) | `28.00 ms` (`286/s`) | `86.00 ms` (`186/s`) | `45.83 ms` (`0.0` vs `1.0`) | `2,090 ms` / `69.9 ms` | **`#1`** |
| **Intel Iris Xe DirectML `FP16`** *(static `64/128`)* | `43.2 MiB` | `46.09 ms` (`22/s`) | `48.37 ms` (`165/s`) | `109.27 ms` (`146/s`) | `53.97 ms` (`0.0` vs `1.0`) | `2,511 ms` / `52.2 ms` | **`#1`** |

- **Zero-overlap paraphrase check:** On two stall descriptions sharing zero content words, neural `all-MiniLM-L6-v2` scored `0.3273` vs `0.0087` unrelated (**margin `0.3187`**), compared to `0.1015` vs `0.0825` (`0.0190`) for lexical hashing.
- **Three real Windows 10 catches fixed in-tree:**
  1. **Win10 `DirectML.dll` 1.0 trap:** Windows 10 19045 ships `DirectML.dll 1.0` in `System32`, causing ORT 1.24.4 to throw `0x887A0004 (DXGI_ERROR_UNSUPPORTED)`; `Steppewatch.Onnx.csproj` auto-bundles `DirectML.dll 1.15.4` (`x64-win`) alongside `steppewatch-onnx.exe`.
  2. **Dynamic shape DX12 shader stalls:** Snapping DirectML inputs to static `64 / 128` sequence buckets cut Intel Iris Xe `FP32` claim-query latency **`8.9×`** (`623 ms -> 69.9 ms`) and full-repo indexing **`2.3×`** (`4,762 ms -> 2,090 ms`).
  3. **Honest silicon choice:** For a tiny `22 MiB` `INT8` encoder on a 12th-gen i5, **CPU `INT8` beats Iris Xe DirectML by `3.9×–14×`** (so `--provider auto` defaults to CPU for quantized `INT8` models and DirectML for `FP16`/`FP32` when GPU offload is preferred).
