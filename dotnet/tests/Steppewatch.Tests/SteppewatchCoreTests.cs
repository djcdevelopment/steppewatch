using Steppewatch.Core;
using Xunit;

namespace Steppewatch.Tests;

public sealed class SteppewatchCoreTests
{
    [Fact]
    public void DotInt8Exact_MatchesScalarReferenceBitForBit()
    {
        var rng = new Random(1234);
        var qU8 = new byte[Hash384Embedder.EmbedDim];
        var wI8 = new sbyte[Hash384Embedder.EmbedDim];
        int wSum = 0;
        int expected = 0;

        for (int i = 0; i < qU8.Length; i++)
        {
            int q = rng.Next(-127, 128);
            sbyte w = (sbyte)rng.Next(-127, 128);
            qU8[i] = (byte)(q + 128);
            wI8[i] = w;
            wSum += w;
            expected += q * w;
        }

        int actual = QuantizedSlabBench.DotInt8Exact(qU8, wI8, wSum);
        Assert.Equal(expected, actual);
    }

    [Fact]
    public void DotFp32Simd_MatchesScalarReferenceWithinTolerance()
    {
        var rng = new Random(5678);
        var a = new float[Hash384Embedder.EmbedDim];
        var b = new float[Hash384Embedder.EmbedDim];
        double expected = 0.0;

        for (int i = 0; i < a.Length; i++)
        {
            a[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            b[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            expected += a[i] * b[i];
        }

        float actual = QuantizedSlabBench.DotFp32Simd(a, b);
        Assert.InRange(Math.Abs(actual - expected), 0.0, 1e-4);
    }

    [Fact]
    public void ThrashDetector_SeparatesHealthyDebuggingFromPoliteStallLoop()
    {
        var embedder = new Hash384Embedder();
        var healthy = ThrashDetector.Detect(embedder,
        [
            "Step 1: Inspected webhook_client.py send_event() to verify current retry backoff and timeout defaults.",
            "Step 2: Edited webhook_client.py to add max_timeout_sec=15.0 and fixed TypeError on RetryConfig init.",
            "Step 3: Updated test_webhook_client.py to add test_retry_timeout_guard covering HTTP 504 responses.",
            "Step 4: Ran pytest test_webhook_client.py; caught AssertionError at line 88 and patched backoff jitter.",
            "Step 5: Re-ran pytest test_webhook_client.py; all 12 tests passed with exit code 0."
        ]);

        var stall = ThrashDetector.Detect(embedder,
        [
            "Step 1: Let me carefully review the webhook integration requirements again to make sure I have the full picture before proposing an edit.",
            "Step 2: I will re-examine the webhook integration plan and verify how the retry timeout should be structured before making any code edits.",
            "Step 3: Let me double-check the webhook integration architecture one more time so the proposed design aligns with existing client boundaries.",
            "Step 4: Before writing any code, I want to carefully reconsider the webhook integration approach and review the timeout constraints again.",
            "Step 5: I am revisiting the webhook integration design notes again to confirm the retry timeout details before deciding on the next edit."
        ]);

        Assert.False(healthy.ThrashDetected);
        Assert.Equal(0.0, healthy.ThrashScore);
        Assert.True(stall.ThrashDetected);
        Assert.True(stall.ThrashScore >= 0.70, $"Expected stall >= 0.70, got {stall.ThrashScore}");
    }

    [Fact]
    public void TranscriptScanner_FiltersBareParallelToolAcksWithoutFalsePositive()
    {
        string tmp = Path.Combine(Path.GetTempPath(), $"sw-test-{Guid.NewGuid():N}.jsonl");
        try
        {
            File.WriteAllLines(tmp,
            [
                """{"type":"user","message":{"content":"INTENT: Create 3 modules and pass pytest"}}""",
                """{"type":"assistant","message":{"content":"Inspected a.py to check schema interfaces."}}""",
                """{"type":"assistant","message":{"content":"Created file."}}""",
                """{"type":"assistant","message":{"content":"Created file."}}""",
                """{"type":"assistant","message":{"content":"Created file."}}""",
                """{"type":"assistant","message":{"content":"Made 1 replacement in a.py."}}""",
                """{"type":"assistant","message":{"content":"Edited c.py and added test_c.py for schema checks."}}""",
                """{"type":"assistant","message":{"content":"Ran pytest test_c.py; all 8 tests passed exit code 0."}}"""
            ]);

            var parsed = TranscriptScanner.ParseJsonl(tmp);
            Assert.Equal(4, parsed.FilteredAcks);
            Assert.Equal(3, parsed.Steps.Count);

            var thrash = ThrashDetector.Detect(new Hash384Embedder(), parsed.Steps);
            Assert.False(thrash.ThrashDetected);
            Assert.Equal(0.0, thrash.ThrashScore);
        }
        finally
        {
            if (File.Exists(tmp)) File.Delete(tmp);
        }
    }

    [Fact]
    public void TinyJepaTrainer_ConvergesAndCheckpointRoundTrips()
    {
        var embedder = new Hash384Embedder();
        var episodes = TinyJepaTrainer.BuildDemoSyntheticEpisodes(40);
        var (ckpt, report) = TinyJepaTrainer.Train(embedder, episodes);

        Assert.True(report.ValAfter.StateSurprise < report.ValBefore.StateSurprise * 0.5,
            $"Expected surprise drop: before={report.ValBefore.StateSurprise}, after={report.ValAfter.StateSurprise}");
        Assert.True(report.ValAfter.RiskMae < report.ValBefore.RiskMae * 0.5,
            $"Expected risk MAE drop: before={report.ValBefore.RiskMae}, after={report.ValAfter.RiskMae}");

        string ckptPath = Path.Combine(Path.GetTempPath(), $"sw-jepa-{Guid.NewGuid():N}.json");
        try
        {
            TinyJepaTrainer.SaveCheckpoint(ckptPath, ckpt);
            var reloaded = TinyJepaTrainer.LoadCheckpoint(ckptPath);
            Assert.NotNull(reloaded);

            var (_, risk) = TinyJepaTrainer.Predict(
                reloaded!,
                embedder.Encode("Inspecting webhook_client.py HTTP retry handler"),
                embedder.Encode("Add 15s timeout guard and pass pytest"));
            Assert.InRange(risk, 0.0, 1.0);
        }
        finally
        {
            if (File.Exists(ckptPath)) File.Delete(ckptPath);
        }
    }
}
