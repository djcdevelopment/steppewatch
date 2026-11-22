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

## Files

| File | Conditions |
|---|---|
| `battery_pre-fix_saturating_7e4feb2.json` | battery, original kernel (gives wrong INT8 results) |
| `battery_fixed_d8f8586_run{1,2}.json` | battery, exact AVX2 kernel |
| `ac_v1.7.1_d8cfc8c_run{1,2}.json` | AC, default tiered JIT |
| `ac_v1.7.1_d8cfc8c_tc0_run{1,2}.json` | AC, `DOTNET_TieredCompilation=0` |
