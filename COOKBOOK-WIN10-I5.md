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

# 3. Train a tiny local JEPA predictor in milliseconds (use --demo-synthetic if no logs yet)
.\steppewatch.exe train --demo-synthetic --out .\checkpoints\my_i5_jepa.json

# 4. One-command JSON probe for Claude Code / Codex / Hermes self-tap on Windows
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

## Next Step After Seeing Your Baseline i5 Numbers

Once `i5_win10_receipt.json` proves your exact `AVX-VNNI` `INT8` vs. `FP32` timings on Windows 10, the clean semantic upgrade path behind `ITextEmbedder` is plugging in **ONNX Runtime (`all-MiniLM-L6-v2` quantized `INT8` on CPU AVX-VNNI or DirectML on the i5's Intel Iris Xe iGPU)** — keeping the exact same single-binary C# control plane.
