using System.Diagnostics;
using System.Text.Json;

namespace Steppewatch.Core;

public sealed record TinyJepaCheckpoint(
    int InDim, int HiddenDim, int OutDim,
    float[] W1, float[] B1, float[] W2State, float[] B2State, float[] W2Risk, float B2Risk);

public sealed record JepaEvalMetrics(double StateSurprise, double RiskMae);
public sealed record JepaTrainReport(
    int EpisodesTotal, int TrainCount, int ValCount, int HiddenDim, int Epochs,
    double TrainMs, JepaEvalMetrics ValBefore, JepaEvalMetrics ValAfter, JepaEvalMetrics TrainAfter);

public static class TinyJepaTrainer
{
    public const int Dim = Hash384Embedder.EmbedDim;
    private const int InDim = Dim * 2;

    public static List<ResidualEpisode> BuildDemoSyntheticEpisodes(int count = 40)
    {
        (string Curr, string Intent, string End, double Risk)[] templates =
        [
            ("Inspecting auth_middleware.py token refresh logic on branch feat/auth",
             "Add expiry jitter check in auth_middleware.py and pass pytest",
             "Edited auth_middleware.py, added test_expiry_jitter in test_auth.py; 14 tests passed exit code 0", 0.05),
            ("Inspecting webhook_client.py HTTP retry handler",
             "Add 15s timeout guard and HTTP 504 backoff in webhook_client.py",
             "Updated webhook_client.py with max_timeout_sec=15; test_webhook_client.py passed 12/12", 0.04),
            ("Reading parser.ts AST span validator",
             "Fix off-by-one line number calculation in parser.ts and run npm test",
             "Patched parser.ts line offset math; added regression spec; npm test passed 0 errors", 0.06),
            ("Reviewing sqlite_store.py schema migration script",
             "Add idempotent column migration for episodes table and verify unit tests",
             "Added ALTER TABLE guard in sqlite_store.py; pytest test_store.py passed 9/9", 0.05),
            ("Looking across the entire repository trying to understand all services",
             "Carefully review everything and make the whole architecture cleaner and better",
             "Re-reading architecture notes again to reconsider the full plan before deciding on any edit", 0.92),
            ("Scanning multiple config and deployment directories without target files",
             "Investigate why things feel slow and rethink the whole system design",
             "Revisiting design constraints one more time to verify every subsystem before touching code", 0.89),
            ("Inspecting legacy API adapter with conflicting specs",
             "Rewrite the communications layer without breaking anything anywhere",
             "Double-checking integration notes again; still reviewing boundaries with zero files modified", 0.94),
            ("Reading old test logs and unrelated docs",
             "Fix all possible edge cases across every module in the repo",
             "Re-examining requirements again before making edits; no new targets or test runs yet", 0.91)
        ];

        var list = new List<ResidualEpisode>(count);
        for (int i = 0; i < count; i++)
        {
            var t = templates[i % templates.Length];
            string sfx = $" [run-{i + 1}]";
            list.Add(new ResidualEpisode(
                "jepa-residual-episode.v1", $"demo-ep-{i + 1:D3}", DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ"),
                "demo", "synthetic",
                new EpisodeState(t.Curr + sfx, 5, 0),
                new EpisodeIntent(t.Intent + sfx),
                new EpisodePrediction("zero_shot_intent_prior", 0.2),
                new EpisodeEndState(t.End + sfx, [], t.Risk >= 0.65, t.Risk, t.Risk >= 0.65 ? "early_abort_and_reroute" : "continue"),
                new EpisodeAssessment(0.5, Math.Abs(t.Risk - 0.2), t.Risk, null)));
        }

        return list;
    }

    public static List<ResidualEpisode> LoadEpisodes(string path)
    {
        var opts = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };
        var list = new List<ResidualEpisode>();
        foreach (string line in File.ReadAllLines(path))
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            try
            {
                using var doc = JsonDocument.Parse(line);
                var r = doc.RootElement;
                string curr = r.GetProperty("current_state").GetProperty("summary").GetString() ?? string.Empty;
                string intent = r.GetProperty("intended_outcome").GetProperty("summary").GetString() ?? string.Empty;
                string end = r.GetProperty("end_state").GetProperty("summary").GetString() ?? string.Empty;
                double risk = r.TryGetProperty("assessment_vs_expectations", out var ae) && ae.TryGetProperty("actual_risk_target", out var rt)
                    ? rt.GetDouble()
                    : r.GetProperty("end_state").GetProperty("thrash_score").GetDouble();
                list.Add(new ResidualEpisode(
                    "jepa-residual-episode.v1", "loaded", string.Empty, "loaded", path,
                    new EpisodeState(curr, 0, 0), new EpisodeIntent(intent),
                    new EpisodePrediction("loaded", 0.2), new EpisodeEndState(end, [], risk >= 0.65, risk, "continue"),
                    new EpisodeAssessment(0, 0, risk, null)));
            }
            catch { /* Ignore malformed line */ }
        }

        return list;
    }

    public static (TinyJepaCheckpoint Checkpoint, JepaTrainReport Report) Train(
        ITextEmbedder embedder, IReadOnlyList<ResidualEpisode> episodes,
        int hiddenDim = 192, int epochs = 180, float lr = 0.20f, int seed = 42)
    {
        int n = episodes.Count;
        float[][] zCurr = embedder.EncodeBatch(episodes.Select(e => e.CurrentState.Summary).ToList());
        float[][] zIntent = embedder.EncodeBatch(episodes.Select(e => e.IntendedOutcome.Summary).ToList());
        float[][] yState = embedder.EncodeBatch(episodes.Select(e => e.EndState.Summary).ToList());
        float[] yRisk = episodes.Select(e => (float)e.AssessmentVsExpectations.ActualRiskTarget).ToArray();

        var x = new float[n][];
        for (int i = 0; i < n; i++)
        {
            x[i] = new float[InDim];
            Array.Copy(zCurr[i], 0, x[i], 0, Dim);
            Array.Copy(zIntent[i], 0, x[i], Dim, Dim);
        }

        var rng = new Random(seed);
        int[] perm = Enumerable.Range(0, n).OrderBy(_ => rng.Next()).ToArray();
        int split = Math.Max(1, (int)(0.8 * n));
        int[] trIdx = perm[..split];
        int[] vaIdx = split < n ? perm[split..] : [perm[^1]];

        float[] w1 = InitMatrix(rng, InDim * hiddenDim, MathF.Sqrt(2.0f / InDim));
        float[] b1 = new float[hiddenDim];
        float[] w2State = InitMatrix(rng, hiddenDim * Dim, MathF.Sqrt(1.0f / hiddenDim));
        float[] b2State = new float[Dim];
        float[] w2Risk = InitMatrix(rng, hiddenDim, MathF.Sqrt(1.0f / hiddenDim));
        float b2Risk = 0f;

        JepaEvalMetrics Eval(int[] idx)
        {
            double sumSurp = 0, sumMae = 0;
            var ck = new TinyJepaCheckpoint(InDim, hiddenDim, Dim, w1, b1, w2State, b2State, w2Risk, b2Risk);
            foreach (int i in idx)
            {
                var (zPred, rPred) = PredictFromConcat(ck, x[i]);
                sumSurp += 1.0 - Math.Clamp(QuantizedSlabBench.DotFp32Simd(zPred, yState[i]), -1.0, 1.0);
                sumMae += Math.Abs(rPred - yRisk[i]);
            }

            return new JepaEvalMetrics(Math.Round(sumSurp / idx.Length, 4), Math.Round(sumMae / idx.Length, 4));
        }

        var valBefore = Eval(vaIdx);
        long t0 = Stopwatch.GetTimestamp();

        float invM = 1.0f / trIdx.Length;
        var dw1 = new float[w1.Length];
        var db1 = new float[hiddenDim];
        var dw2s = new float[w2State.Length];
        var db2s = new float[Dim];
        var dw2r = new float[hiddenDim];
        var h = new float[hiddenDim];
        var zPredBuf = new float[Dim];
        var dzBuf = new float[Dim];
        var dhBuf = new float[hiddenDim];

        for (int ep = 0; ep < epochs; ep++)
        {
            Array.Clear(dw1); Array.Clear(db1); Array.Clear(dw2s); Array.Clear(db2s); Array.Clear(dw2r);
            float db2r = 0f;

            foreach (int i in trIdx)
            {
                float[] xi = x[i];
                for (int j = 0; j < hiddenDim; j++)
                {
                    float s = b1[j] + QuantizedSlabBench.DotFp32Simd(xi, w1.AsSpan(j * InDim, InDim));
                    h[j] = s > 0f ? s : 0f;
                }

                float rLogit = b2Risk + QuantizedSlabBench.DotFp32Simd(h, w2Risk);
                float rPred = Sigmoid(rLogit);
                float dr = (rPred - yRisk[i]) * invM;
                db2r += dr;

                Array.Clear(dhBuf);
                for (int j = 0; j < hiddenDim; j++)
                {
                    dw2r[j] += h[j] * dr;
                    dhBuf[j] += w2Risk[j] * dr;
                }

                float[] ys = yState[i];
                for (int o = 0; o < Dim; o++)
                {
                    float z = b2State[o] + QuantizedSlabBench.DotFp32Simd(h, w2State.AsSpan(o * hiddenDim, hiddenDim));
                    zPredBuf[o] = z;
                    float dz = 2.0f * (z - ys[o]) * invM;
                    dzBuf[o] = dz;
                    db2s[o] += dz;
                    int rowOff = o * hiddenDim;
                    for (int j = 0; j < hiddenDim; j++)
                    {
                        dw2s[rowOff + j] += h[j] * dz;
                        dhBuf[j] += w2State[rowOff + j] * dz;
                    }
                }

                for (int j = 0; j < hiddenDim; j++)
                {
                    if (h[j] <= 0f) continue;
                    float g = dhBuf[j];
                    db1[j] += g;
                    int rowOff = j * InDim;
                    for (int k = 0; k < InDim; k++) dw1[rowOff + k] += g * xi[k];
                }
            }

            const float wd = 1e-4f;
            for (int k = 0; k < w1.Length; k++) w1[k] -= lr * (dw1[k] + wd * w1[k]);
            for (int k = 0; k < b1.Length; k++) b1[k] -= lr * db1[k];
            for (int k = 0; k < w2State.Length; k++) w2State[k] -= lr * (dw2s[k] + wd * w2State[k]);
            for (int k = 0; k < b2State.Length; k++) b2State[k] -= lr * db2s[k];
            for (int k = 0; k < w2Risk.Length; k++) w2Risk[k] -= lr * (dw2r[k] + wd * w2Risk[k]);
            b2Risk -= lr * db2r;
        }

        double trainMs = Math.Round(Stopwatch.GetElapsedTime(t0).TotalMilliseconds, 2);
        var ckpt = new TinyJepaCheckpoint(InDim, hiddenDim, Dim, w1, b1, w2State, b2State, w2Risk, b2Risk);
        return (ckpt, new JepaTrainReport(n, trIdx.Length, vaIdx.Length, hiddenDim, epochs, trainMs, valBefore, Eval(vaIdx), Eval(trIdx)));
    }

    public static (float[] PredictedEnd, double PredictedRisk) Predict(TinyJepaCheckpoint ckpt, float[] zCurr, float[] zIntent)
    {
        var x = new float[InDim];
        Array.Copy(zCurr, 0, x, 0, Dim);
        Array.Copy(zIntent, 0, x, Dim, Dim);
        return PredictFromConcat(ckpt, x);
    }

    public static void SaveCheckpoint(string path, TinyJepaCheckpoint ckpt)
    {
        string? dir = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
        File.WriteAllText(path, JsonSerializer.Serialize(ckpt));
    }

    public static TinyJepaCheckpoint? LoadCheckpoint(string? path)
    {
        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path)) return null;
        return JsonSerializer.Deserialize<TinyJepaCheckpoint>(File.ReadAllText(path));
    }

    private static (float[] ZPred, double Risk) PredictFromConcat(TinyJepaCheckpoint ckpt, float[] x)
    {
        var h = new float[ckpt.HiddenDim];
        for (int j = 0; j < ckpt.HiddenDim; j++)
        {
            float s = ckpt.B1[j] + QuantizedSlabBench.DotFp32Simd(x, ckpt.W1.AsSpan(j * ckpt.InDim, ckpt.InDim));
            h[j] = s > 0f ? s : 0f;
        }

        var z = new float[ckpt.OutDim];
        for (int o = 0; o < ckpt.OutDim; o++)
        {
            z[o] = ckpt.B2State[o] + QuantizedSlabBench.DotFp32Simd(h, ckpt.W2State.AsSpan(o * ckpt.HiddenDim, ckpt.HiddenDim));
        }

        Hash384Embedder.NormalizeInPlace(z);
        float rLogit = ckpt.B2Risk + QuantizedSlabBench.DotFp32Simd(h, ckpt.W2Risk);
        return (z, Math.Round(Sigmoid(rLogit), 4));
    }

    private static float[] InitMatrix(Random rng, int len, float scale)
    {
        var a = new float[len];
        for (int i = 0; i < len; i++) a[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * scale);
        return a;
    }

    private static float Sigmoid(float x) => 1.0f / (1.0f + MathF.Exp(-Math.Clamp(x, -20.0f, 20.0f)));
}
