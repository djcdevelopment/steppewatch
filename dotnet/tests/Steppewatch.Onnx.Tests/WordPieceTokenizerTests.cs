using Steppewatch.Core;
using Steppewatch.Onnx;
using Xunit;

namespace Steppewatch.Onnx.Tests;

public sealed class WordPieceTokenizerTests
{
    private static WordPieceTokenizer CreateMiniTokenizer() => new(new Dictionary<string, int>
    {
        ["[PAD]"] = 0,
        ["[UNK]"] = 100,
        ["[CLS]"] = 101,
        ["[SEP]"] = 102,
        ["cafe"] = 200,
        ["step"] = 201,
        ["1"] = 202,
        [":"] = 203,
        ["watch"] = 204,
        ["##dog"] = 205,
        ["##s"] = 206,
        ["hello"] = 207,
        ["world"] = 208,
        ["!"] = 209
    });

    [Fact]
    public void EncodeIds_HandlesAccentsPunctuationAndSubwords()
    {
        var tok = CreateMiniTokenizer();
        int[] ids = tok.EncodeIds("Café Step 1: watchdogs!", maxLength: 16);

        // [CLS] cafe step 1 : watch ##dog ##s ! [SEP]
        Assert.Equal([101, 200, 201, 202, 203, 204, 205, 206, 209, 102], ids);
    }

    [Fact]
    public void EncodeIds_FallsBackToUnkAndRespectsMaxLength()
    {
        var tok = CreateMiniTokenizer();
        int[] unkIds = tok.EncodeIds("hello unknownword world", maxLength: 16);
        Assert.Equal([101, 207, 100, 208, 102], unkIds);

        int[] truncIds = tok.EncodeIds("hello world cafe step", maxLength: 4);
        Assert.Equal([101, 207, 208, 102], truncIds);
    }

    [Fact]
    public void EncodeBatch_PadsSequencesAndBuildsAttentionMask()
    {
        var tok = CreateMiniTokenizer();
        TokenizedBatch batch = tok.EncodeBatch(["hello", "hello world!"], maxLength: 16);

        Assert.Equal(2, batch.BatchSize);
        Assert.Equal(5, batch.SeqLen);
        Assert.Equal([101, 207, 102, 0, 0, 101, 207, 208, 209, 102], batch.InputIds);
        Assert.Equal([1, 1, 1, 0, 0, 1, 1, 1, 1, 1], batch.AttentionMask);

        TokenizedBatch staticBatch = tok.EncodeBatch(["hello", "hello world!"], maxLength: 128, padToStaticBucket: true);
        Assert.Equal(64, staticBatch.SeqLen);
        Assert.Equal(128, WordPieceTokenizer.SnapToStaticBucket(65, 128));
    }

    [Fact]
    public void OnnxMiniLmEmbedder_EndToEndWhenModelEnvProvided()
    {
        string? model = Environment.GetEnvironmentVariable("STEPPEWATCH_TEST_MODEL");
        string? vocab = Environment.GetEnvironmentVariable("STEPPEWATCH_TEST_VOCAB");
        if (string.IsNullOrWhiteSpace(model) || string.IsNullOrWhiteSpace(vocab) || !File.Exists(model) || !File.Exists(vocab))
        {
            return; // Opt-in offline-safe: runs full neural check whenever local ONNX model env vars are set.
        }

        using var embedder = new OnnxMiniLmEmbedder(model, vocab, OnnxExecutionProvider.Cpu, maxLength: 128);
        float[] v1 = embedder.Encode("The background worker is stuck repeating the same plan without editing code.");
        float[] v2 = embedder.Encode("An autonomous agent keeps paraphrasing its intentions while modifying zero files.");
        float[] v3 = embedder.Encode("Quarterly invoice reconciliation exported tax totals to the database ledger.");

        Assert.Equal(384, v1.Length);
        float norm1 = MathF.Sqrt(Hash384Embedder.Dot(v1, v1));
        Assert.InRange(norm1, 0.999f, 1.001f);

        float simSame = Hash384Embedder.Dot(v1, v2);
        float simDiff = Hash384Embedder.Dot(v1, v3);
        Assert.True(simSame > simDiff + 0.22f, $"Expected clear semantic separation, got same={simSame:F4} vs diff={simDiff:F4}");

        var healthy = ThrashDetector.Detect(embedder,
        [
            "Step 1: Inspected webhook_client.py retry timeout defaults.",
            "Step 2: Patched RetryConfig timeout guard to 15 seconds.",
            "Step 3: Added unit test for HTTP 504 timeout handling.",
            "Step 4: Fixed assertion failure on backoff jitter.",
            "Step 5: Re-ran full test suite; all 12 tests passed."
        ], "onnx-healthy");

        var stall = ThrashDetector.Detect(embedder,
        [
            "Step 1: Let me carefully review the webhook integration plan again before proposing an edit.",
            "Step 2: I will re-examine the webhook integration requirements before modifying any code.",
            "Step 3: Let me double-check the webhook integration architecture one more time before editing.",
            "Step 4: Before writing code, I want to carefully reconsider the webhook integration constraints again.",
            "Step 5: I am revisiting the webhook integration design notes again before deciding on an edit."
        ], "onnx-stall");

        Assert.False(healthy.ThrashDetected);
        Assert.True(stall.ThrashDetected);
        Assert.True(stall.ThrashScore > healthy.ThrashScore + 0.35);
    }
}
