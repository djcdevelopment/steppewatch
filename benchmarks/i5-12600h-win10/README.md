# i5-12600H / Windows 10: AC vs battery receipts (2026-11-22)

**Host:** Dell Precision 3571, i5-12600H (Alder Lake-H, 4P+8E / 16T), 32 GB, Win10 Pro 19045, .NET 10.0.15, Balanced power plan for every run.
**INT8 kernel on this host:** `x86_avx2_vpmaddubsw`. AVX-VNNI is hidden because Windows runs as a Hyper-V root partition: hypervisor vendor `Microsoft Hv`, `CPUID(7,1).EAX = 0`, so `AvxVnni.IsSupported = false`.
**Tests at `v1.7.1-avx2-exact-20261122` (d8cfc8c):** 9/9 pass.

All timings are the median of 6, in ms, from `steppewatch bench`. Raw receipts are in this folder.

## Headline: 262,144 cols (96.5 MiB INT8+scales vs 384 MiB FP32), default runtime (tiered JIT)

| Run | Commit | INT8 Q=1 | Q=8 | Q=16 | Q=64 | FP32 Q=1 | Q=8 | Q=16 | Q=64 |
|---|---|---|---|---|---|---|---|---|---|
| **AC** run 1 | d8cfc8c | 1.87 | 4.02 | 8.70 | 28.67 | 7.45 | 9.65 | 16.27 | 67.34 |
| **AC** run 2 | d8cfc8c | 1.66 | 3.53 | 6.51 | 27.91 | 6.89 | 8.93 | 14.16 | 62.66 |
| **Battery** run 1 | d8f8586 | 1.95 | 4.62 | 6.70 | 30.78 | 7.68 | 10.71 | 15.35 | 70.51 |
| **Battery** run 2 | d8f8586 | 1.98 | 4.72 | 7.15 | 28.68 | 7.41 | 9.73 | 16.01 | 63.52 |
| Battery, pre-fix (saturating, **wrong results**) | 7e4feb2 | 2.09 | 4.29 | 7.02 | 25.84 | 7.27 | 10.40 | 16.38 | 67.51 |

- **AC vs battery:** AC is about 5–15% faster at Q≤8. At Q=16 and Q=64 the difference is within run-to-run noise. Q=1 is memory-bandwidth-bound, so power limits barely touch it.
- **Exact INT8 vs FP32 on AC:** about **4.1× (Q=1), 2.5× (Q=8), 2.0× (Q=16), 2.3× (Q=64)**. This holds without VNNI; most of the gain comes from moving 4× fewer bytes.
- **Same-code caveat:** the battery rows ran on d8f8586 and the AC rows on d8cfc8c. The hot loop is identical; d8cfc8c only moves the AVX2 path into `DotInt8Avx2OrScalar`. A same-commit battery row needs the laptop unplugged and a rerun.

## Methodology finding: small-N rows measure JIT warmup, not the kernel

`MeasureMedianMs` does one warmup call and then 6 timed calls. Early in the process, the small-N configs run before tier-1/PGO code is installed. Disabling tiering (`DOTNET_TieredCompilation=0`, full optimization from the first call) shows the size of the effect, all on AC at d8cfc8c:

| Config | Tiered (default) run 1 / run 2 | TieredCompilation=0 run 1 / run 2 |
|---|---|---|
| INT8 N=4096 Q=1 | 0.73 / 0.68 | **0.06 / 0.08** |
| INT8 N=4096 Q=64 | 4.44 / 4.45 | **0.68 / 0.85** |
| INT8 N=16384 Q=64 | 11.88 / 11.52 | **2.01 / 1.95** |
| INT8 N=65536 Q=64 | 9.17 / **42.93** (spike) | 8.59 / 8.28 |
| FP32 N=4096 Q=1 | 0.76 / 0.66 | **0.12 / 0.10** |
| INT8 N=262144 Q=64 | 28.67 / 27.91 | 34.39 / 33.61 (slower: no dynamic PGO) |

- The small-N tiered numbers are 5–10× inflated and noisy. Don't publish the N≤65,536 rows as kernel speed.
- Turning tiering off isn't the fix either, because at large N tier-1 with dynamic PGO beats full optimization without PGO (Q=64: 28 vs 34 ms).
- **Suggested fix:**
  - Warm up each (kernel, N, Q) config until it is stable, for example at least 200 ms or at least 50 untimed calls, or one untimed pass over the whole grid first.
  - Take at least 20 timed samples and report median plus p10/p90.
  - Record the power source, power plan, and JIT mode in the receipt.

## Follow-up: `v1.8` Tier-1 Warmup Verification + Optional Neural ONNX (`all-MiniLM-L6-v2` CPU `INT8` vs Intel Iris Xe DirectML)

Running the `v1.8` Tier-1 pre-warmed binary (`0d7399d`, `WarmupTier1` + `15` samples) on the same Dell Precision 3571 eliminated the small-`N` Tier-0 inflation and `42.93 ms` spike while preserving full Dynamic PGO throughput at `N = 262,144`:

| Config | `v1.7.1` Cold Tiered (`run 1 / 2`) | `v1.7.1` `TC=0` (`run 1 / 2`) | **`v1.8` Tier-1 Pre-Warmed (`run 1 / 2`)** |
|---|---:|---:|---:|
| `INT8 N=4096 Q=1` | `0.73 / 0.68 ms` | `0.06 / 0.08 ms` | **`0.21 / 0.27 ms`** |
| `INT8 N=16384 Q=64` | `11.88 / 11.52 ms` | `2.01 / 1.95 ms` | **`1.99 / 1.86 ms`** |
| `INT8 N=65536 Q=64` | `9.17 / 42.93 ms` *(spike)* | `8.59 / 8.28 ms` | **`7.12 / 6.99 ms`** *(spike gone, beats `TC=0`)* |
| `INT8 N=262144 Q=1 / 8 / 16 / 64` | `1.66–1.87 / 3.53–4.02 / 6.51–8.70 / 27.91–28.67 ms` | `1.9 / 4.2 / 8.7 / 33.6–34.4 ms` | **`1.60–1.63 / 3.38–3.60 / 6.78–6.95 / 27.41–28.05 ms`** |

We also benchmarked the opt-in `.NET 10` neural embedder (`dotnet/src/Steppewatch.Onnx`, `all-MiniLM-L6-v2`, `384-D`) on both `i5-12600H` CPU and **Intel Iris Xe Graphics** (`Driver 32.0.101.7088`, `DirectML.dll 1.15.4`):

| Neural Backend (`all-MiniLM-L6-v2`) | Model | `B=1` | `B=8` | `B=16` | Stall Check | Full Repo (`239` syms) / Query | Caught Rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| **CPU `INT8`** (`model_quantized.onnx`) | `21.9 MiB` | **`3.19 ms`** (`314/s`) | **`13.99 ms`** (`572/s`) | **`21.79 ms`** (`734/s`) | **`12.37 ms`** | **`534 ms`** / **`6.2 ms`** | **`#1`** |
| **CPU `FP32`** (`model.onnx`) | `86.2 MiB` | `4.04 ms` (`248/s`) | `18.38 ms` (`435/s`) | `28.99 ms` (`552/s`) | `13.62 ms` | `728 ms` / `5.8 ms` | **`#1`** |
| **Iris Xe DirectML `FP32`** *(static `64/128`)* | `86.2 MiB` | `54.81 ms` (`18/s`) | `28.00 ms` (`286/s`) | `86.00 ms` (`186/s`) | `45.83 ms` | `2,090 ms` / `69.9 ms` | **`#1`** |
| **Iris Xe DirectML `FP16`** *(static `64/128`)* | `43.2 MiB` | `46.09 ms` (`22/s`) | `48.37 ms` (`165/s`) | `109.27 ms` (`146/s`) | `53.97 ms` | `2,511 ms` / `52.2 ms` | **`#1`** |

- **DirectML static bucket speedup:** Before snapping DirectML inputs to static `64 / 128` sequence buckets, dynamic seq lengths triggered DX12 shader recompilation on every batch (`4,762 ms` repo index / `623 ms` query for `FP32`; `5,215 ms` / `308 ms` for `FP16`). Static buckets cut query latency **`5.9×–8.9×`** and indexing **`2.1×–2.3×`**.
- **Takeaway:** For a `22 MiB` `INT8` MiniLM encoder, **`i5-12600H` CPU `INT8` (`3.19 ms` / `734 texts/sec`) decisively beats Intel Iris Xe DirectML** while keeping GPU utilization at `0%`.

## Files

| File | Conditions |
|---|---|
| `battery_pre-fix_saturating_7e4feb2.json` | battery, original kernel (gives wrong INT8 results) |
| `battery_fixed_d8f8586_run{1,2}.json` | battery, exact AVX2 kernel |
| `ac_v1.7.1_d8cfc8c_run{1,2}.json` | AC, default tiered JIT |
| `ac_v1.7.1_d8cfc8c_tc0_run{1,2}.json` | AC, `DOTNET_TieredCompilation=0` |
| `i5_v1.8_warmed_run2.json` | `v1.8` `WarmupTier1` + `15` samples (`run 2` JSON receipt) |
| `i5_onnx_cpu_int8.json` | Neural `all-MiniLM-L6-v2` `INT8` (`21.9 MiB`) on `i5-12600H` CPU |
| `i5_onnx_dml_fp32_static.json` | Neural `all-MiniLM-L6-v2` `FP32` on Intel Iris Xe via DirectML `1.15.4` (static `64/128`) |
| `i5_onnx_dml_fp16_static.json` | Neural `all-MiniLM-L6-v2` `FP16` on Intel Iris Xe via DirectML `1.15.4` (static `64/128`) |
