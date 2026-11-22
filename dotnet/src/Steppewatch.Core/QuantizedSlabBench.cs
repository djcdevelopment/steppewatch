using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Steppewatch.Core;

public sealed record SlabBenchReport(
    int MaxColumns,
    int EmbedDim,
    double SlabInt8PlusScalesMiB,
    double SlabFp32MiB,
    string Int8Kernel,
    string Fp32Kernel,
    Dictionary<string, Dictionary<string, double>> Int8QuantizedMs,
    Dictionary<string, Dictionary<string, double>> Fp32SimdMs);

public static class QuantizedSlabBench
{
    public const int Dim = Hash384Embedder.EmbedDim;

    public static SlabBenchReport Run(
        int[]? colCounts = null,
        int[]? qCounts = null,
        int iterations = 15)
    {
        colCounts ??= [4096, 16384, 65536, 262144];
        qCounts ??= [1, 8, 16, 64];
        int maxCols = colCounts.Max();

        var rng = new Random(42);
        var weightsFp32 = new float[maxCols * Dim];
        var weightsInt8 = new sbyte[maxCols * Dim];
        var colScales = new Half[maxCols];
        var colSums = new int[maxCols];

        for (int c = 0; c < maxCols; c++)
        {
            int offset = c * Dim;
            float maxAbs = 1e-6f;
            for (int k = 0; k < Dim; k++)
            {
                float v = (float)(rng.NextDouble() * 2.0 - 1.0);
                weightsFp32[offset + k] = v;
                float av = MathF.Abs(v);
                if (av > maxAbs)
                {
                    maxAbs = av;
                }
            }

            float scale = maxAbs / 127.0f;
            colScales[c] = (Half)scale;
            float invScale = 1.0f / scale;
            int sum = 0;
            for (int k = 0; k < Dim; k++)
            {
                int q = Math.Clamp((int)MathF.Round(weightsFp32[offset + k] * invScale), -127, 127);
                weightsInt8[offset + k] = (sbyte)q;
                sum += q;
            }

            colSums[c] = sum;
        }

        double int8MiB = Math.Round((weightsInt8.Length + colScales.Length * 2L) / (1024.0 * 1024.0), 2);
        double fp32MiB = Math.Round((weightsFp32.Length * 4L) / (1024.0 * 1024.0), 2);
        var hw = HardwareProbe.Detect();

        var int8Results = new Dictionary<string, Dictionary<string, double>>();
        var fp32Results = new Dictionary<string, Dictionary<string, double>>();

        // One-time Tier-1 + Dynamic PGO JIT warmup across both kernels before any timed row.
        // .NET tiered compilation waits ~100ms before enabling call counting, and Tier-0 PGO
        // instrumentation counters cause cross-core cache-line bouncing under Parallel.For until
        // Tier-1 replaces the instrumented code (caught on i5-12600H Win10 & 24-thread Linux).
        int warmCols = Math.Min(4096, maxCols);
        var warmScores = new float[8 * warmCols];
        var warmQ = BuildQueries(new Random(7), 8, out byte[] warmQU8, out float[] warmQScales);
        WarmupTier1(
            () => MatMulInt8(warmQU8, warmQScales, 8, weightsInt8, colScales, colSums, warmCols, warmScores),
            () => MatMulFp32(warmQ, 8, weightsFp32, warmCols, warmScores));

        foreach (int n in colCounts)
        {
            var rowInt8 = new Dictionary<string, double>();
            var rowFp32 = new Dictionary<string, double>();
            var scores = new float[qCounts.Max() * n];

            foreach (int q in qCounts)
            {
                var qFp32 = BuildQueries(rng, q, out byte[] qU8, out float[] qScales);

                // Warmup + measure exact INT8 quantized slab MatMul (AVX-VNNI / AVX2)
                rowInt8[$"Q={q}"] = MeasureMedianMs(
                    iterations,
                    () => MatMulInt8(qU8, qScales, q, weightsInt8, colScales, colSums, n, scores));

                // Warmup + measure FP32 SIMD MatMul (AVX2+FMA / Vector256)
                rowFp32[$"Q={q}"] = MeasureMedianMs(
                    iterations,
                    () => MatMulFp32(qFp32, q, weightsFp32, n, scores));
            }

            int8Results[n.ToString()] = rowInt8;
            fp32Results[n.ToString()] = rowFp32;
        }

        return new SlabBenchReport(
            MaxColumns: maxCols,
            EmbedDim: Dim,
            SlabInt8PlusScalesMiB: int8MiB,
            SlabFp32MiB: fp32MiB,
            Int8Kernel: hw.ActiveInt8Kernel,
            Fp32Kernel: hw.ActiveFp32Kernel,
            Int8QuantizedMs: int8Results,
            Fp32SimdMs: fp32Results);
    }

    public static int DotInt8Exact(ReadOnlySpan<byte> qUnsignedOffset128, ReadOnlySpan<sbyte> wSigned, int wColSum)
    {
        int len = qUnsignedOffset128.Length;
        if (AvxVnni.IsSupported && len >= 32)
        {
            // vpdpbusd: u8 x s8 products, 4-way sums accumulated straight into int32 (no intermediate saturation).
            int i = 0;
            Vector256<int> acc = Vector256<int>.Zero;
            for (; i <= len - 32; i += 32)
            {
                Vector256<byte> qv = Vector256.LoadUnsafe(in qUnsignedOffset128[i]);
                Vector256<sbyte> wv = Vector256.LoadUnsafe(in wSigned[i]);
                acc = AvxVnni.MultiplyWideningAndAdd(acc, qv, wv);
            }

            int total = Vector256.Sum(acc);
            for (; i < len; i++)
            {
                total += qUnsignedOffset128[i] * wSigned[i];
            }

            return total - (128 * wColSum);
        }

        return DotInt8Avx2OrScalar(qUnsignedOffset128, wSigned);
    }

    public static int DotInt8Avx2OrScalar(ReadOnlySpan<byte> qUnsignedOffset128, ReadOnlySpan<sbyte> wSigned)
    {
        int len = qUnsignedOffset128.Length;
        int i = 0;
        int signedTotal = 0;

        if (Avx2.IsSupported && len >= 32)
        {
            // vpmaddubsw saturates each u8*s8 pair-sum to int16. Fed the offset-128 query directly, a pair reaches
            // 2*255*127 = 64,770 and silently clips. Instead undo the offset (q ^ 0x80 -> signed q) and move q's sign
            // onto w, so the unsigned operand is |q| <= 127: worst-case pair 2*127*127 = 32,258 fits int16 exactly.
            // Precondition (guaranteed by both quantizers, which clamp to +/-127): q-128 and w lie in [-127, 127].
            Vector256<int> acc = Vector256<int>.Zero;
            Vector256<short> ones = Vector256.Create((short)1);
            Vector256<byte> bias = Vector256.Create((byte)0x80);
            for (; i <= len - 32; i += 32)
            {
                Vector256<sbyte> qs = Avx2.Xor(Vector256.LoadUnsafe(in qUnsignedOffset128[i]), bias).AsSByte();
                Vector256<sbyte> wv = Vector256.LoadUnsafe(in wSigned[i]);
                Vector256<short> prod16 = Avx2.MultiplyAddAdjacent(Avx2.Abs(qs), Avx2.Sign(wv, qs));
                acc = Avx2.Add(acc, Avx2.MultiplyAddAdjacent(prod16, ones));
            }

            signedTotal = Vector256.Sum(acc);
        }

        for (; i < len; i++)
        {
            signedTotal += (qUnsignedOffset128[i] - 128) * wSigned[i];
        }

        return signedTotal;
    }

    public static float DotFp32Simd(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        int len = a.Length;
        int i = 0;
        float total = 0f;

        if (Avx2.IsSupported && Fma.IsSupported && len >= 8)
        {
            Vector256<float> acc0 = Vector256<float>.Zero;
            Vector256<float> acc1 = Vector256<float>.Zero;
            for (; i <= len - 16; i += 16)
            {
                acc0 = Fma.MultiplyAdd(Vector256.LoadUnsafe(in a[i]), Vector256.LoadUnsafe(in b[i]), acc0);
                acc1 = Fma.MultiplyAdd(Vector256.LoadUnsafe(in a[i + 8]), Vector256.LoadUnsafe(in b[i + 8]), acc1);
            }

            total = Vector256.Sum(Avx.Add(acc0, acc1));
        }

        for (; i < len; i++)
        {
            total += a[i] * b[i];
        }

        return total;
    }

    private static void MatMulInt8(
        byte[] qU8, float[] qScales, int qCount,
        sbyte[] wInt8, Half[] colScales, int[] colSums, int nCols,
        float[] scores)
    {
        int chunkSize = Math.Max(256, nCols / Math.Max(1, Environment.ProcessorCount * 2));
        int chunks = (nCols + chunkSize - 1) / chunkSize;
        Parallel.For(0, chunks, chunk =>
        {
            int start = chunk * chunkSize;
            int end = Math.Min(nCols, start + chunkSize);
            for (int c = start; c < end; c++)
            {
                ReadOnlySpan<sbyte> wCol = wInt8.AsSpan(c * Dim, Dim);
                float cScale = (float)colScales[c];
                int cSum = colSums[c];
                for (int r = 0; r < qCount; r++)
                {
                    ReadOnlySpan<byte> qRow = qU8.AsSpan(r * Dim, Dim);
                    int dot = DotInt8Exact(qRow, wCol, cSum);
                    scores[r * nCols + c] = dot * qScales[r] * cScale;
                }
            }
        });
    }

    private static void MatMulFp32(float[] qFp32, int qCount, float[] wFp32, int nCols, float[] scores)
    {
        int chunkSize = Math.Max(256, nCols / Math.Max(1, Environment.ProcessorCount * 2));
        int chunks = (nCols + chunkSize - 1) / chunkSize;
        Parallel.For(0, chunks, chunk =>
        {
            int start = chunk * chunkSize;
            int end = Math.Min(nCols, start + chunkSize);
            for (int c = start; c < end; c++)
            {
                ReadOnlySpan<float> wCol = wFp32.AsSpan(c * Dim, Dim);
                for (int r = 0; r < qCount; r++)
                {
                    ReadOnlySpan<float> qRow = qFp32.AsSpan(r * Dim, Dim);
                    scores[r * nCols + c] = DotFp32Simd(qRow, wCol);
                }
            }
        });
    }

    private static float[] BuildQueries(Random rng, int qCount, out byte[] qU8, out float[] qScales)
    {
        var qFp32 = new float[qCount * Dim];
        qU8 = new byte[qCount * Dim];
        qScales = new float[qCount];
        for (int r = 0; r < qCount; r++)
        {
            int off = r * Dim;
            float maxAbs = 1e-6f;
            for (int k = 0; k < Dim; k++)
            {
                float v = (float)(rng.NextDouble() * 2.0 - 1.0);
                qFp32[off + k] = v;
                maxAbs = MathF.Max(maxAbs, MathF.Abs(v));
            }

            float scale = maxAbs / 127.0f;
            qScales[r] = scale;
            float inv = 1.0f / scale;
            for (int k = 0; k < Dim; k++)
            {
                int qVal = Math.Clamp((int)MathF.Round(qFp32[off + k] * inv), -127, 127);
                qU8[off + k] = (byte)(qVal + 128);
            }
        }

        return qFp32;
    }

    private static void WarmupTier1(Action int8Kernel, Action fp32Kernel)
    {
        // First call triggers Tier-0 JIT and starts .NET's call-counting delay timer (~100ms).
        int8Kernel();
        fp32Kernel();

        // Keep exercising both kernels past the call-counting delay + 30-call promotion threshold
        // so Tier-1 + Dynamic PGO completes before timed rows begin.
        long start = Stopwatch.GetTimestamp();
        int calls = 1;
        while (calls < 60 || Stopwatch.GetElapsedTime(start).TotalMilliseconds < 250.0)
        {
            int8Kernel();
            fp32Kernel();
            calls++;
        }
    }

    private static double MeasureMedianMs(int iterations, Action action)
    {
        // Light per-config warmup so page tables, thread-pool workers, and caches are hot.
        long warmStart = Stopwatch.GetTimestamp();
        int warmCalls = 0;
        while (warmCalls < 3 || (warmCalls < 12 && Stopwatch.GetElapsedTime(warmStart).TotalMilliseconds < 40.0))
        {
            action();
            warmCalls++;
        }

        var samples = new double[iterations];
        for (int i = 0; i < iterations; i++)
        {
            long t0 = Stopwatch.GetTimestamp();
            action();
            samples[i] = Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
        }

        Array.Sort(samples);
        return Math.Round(samples[iterations / 2], 3);
    }
}
