using System.Diagnostics;
using System.Text.Json;
using Steppewatch.Core;
using Steppewatch.Onnx;

namespace Steppewatch.Onnx.Cli;

internal static class Program
{
    private static readonly JsonSerializerOptions PrettyJson = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower
    };

    public static int Main(string[] args)
    {
        if (args.Length == 0 || args[0] is "-h" or "--help" or "help")
        {
            PrintUsage();
            return 0;
        }

        string cmd = args[0].ToLowerInvariant();
        string? modelPath = GetArg(args, "--model");
        string? vocabPath = GetArg(args, "--vocab");
        string providerRaw = (GetArg(args, "--provider") ?? "auto").ToLowerInvariant();
        int maxLen = int.TryParse(GetArg(args, "--max-len"), out int ml) ? ml : 128;
        int batchSize = int.TryParse(GetArg(args, "--batch-size"), out int bs) ? bs : 16;
        int deviceId = int.TryParse(GetArg(args, "--device"), out int dev) ? dev : 0;

        if (string.IsNullOrWhiteSpace(modelPath))
        {
            Console.Error.WriteLine("Missing required --model <path/to/model.onnx>.");
            return PrintUsage(1);
        }

        vocabPath ??= Path.Combine(Path.GetDirectoryName(Path.GetFullPath(modelPath))!, "vocab.txt");
        OnnxExecutionProvider provider = providerRaw switch
        {
            "auto" => OnnxExecutionProvider.Auto,
            "dml" or "directml" => OnnxExecutionProvider.DirectML,
            "cpu" => OnnxExecutionProvider.Cpu,
            _ => throw new ArgumentException($"Unsupported --provider '{providerRaw}'. Use auto, dml, or cpu.")
        };

        using var embedder = new OnnxMiniLmEmbedder(modelPath, vocabPath, provider, maxLen, batchSize, deviceId);
        if (embedder.ProviderFallbackReason is not null)
        {
            Console.Error.WriteLine($"[note] {embedder.ProviderFallbackReason}");
        }

        if (cmd == "neural-bench")
        {
            return RunNeuralBench(embedder, modelPath, maxLen, args[1..]);
        }

        string[] forwarded = StripOnnxFlags(args);
        return Steppewatch.Cli.Program.RunWithEmbedder(embedder, forwarded);
    }

    private static int RunNeuralBench(OnnxMiniLmEmbedder embedder, string modelPath, int maxLen, string[] args)
    {
        string repo = Path.GetFullPath(GetArg(args, "--repo") ?? Directory.GetCurrentDirectory());
        string receiptPath = GetArg(args, "--receipt") ?? Path.Combine("receipts", $"dotnet_onnx_{embedder.ActiveProvider}_receipt.json");
        var hw = HardwareProbe.Detect();
        double modelMiB = Math.Round(new FileInfo(modelPath).Length / (1024.0 * 1024.0), 2);

        Console.WriteLine("=== Steppewatch .NET 10 Neural ONNX Benchmark (all-MiniLM-L6-v2, 384-D) ===");
        Console.WriteLine($"Host     : {hw.OsDescription} | {hw.LogicalCores} threads | {hw.TotalMemoryGib} GiB RAM");
        Console.WriteLine($"Backend  : {embedder.BackendName} | model={modelMiB} MiB | max_len={maxLen}");

        // Warmup ONNX graph + DirectML shader pipelines before timing
        string[] corpus = BuildBenchmarkTexts(16);
        for (int w = 0; w < 3; w++)
        {
            embedder.EncodeBatch(corpus[..8]);
        }

        Console.WriteLine("\n=== Part A: Real Neural 384-D Encoding Latency ===");
        var encodeMs = new Dictionary<string, double>();
        var encodeTps = new Dictionary<string, double>();
        foreach (int b in new[] { 1, 8, 16 })
        {
            string[] batch = corpus[..b];
            double ms = MeasureMedianMs(9, () => embedder.EncodeBatch(batch));
            double tps = Math.Round(b / (ms / 1000.0), 1);
            encodeMs[$"B={b}"] = ms;
            encodeTps[$"B={b}"] = tps;
            Console.WriteLine($"  Batch B={b,2} -> {ms,7:F2} ms median  ({tps,6:F1} texts/sec)");
        }

        Console.WriteLine("\n=== Part B: Zero-Overlap Semantic Paraphrase Check (Neural vs Lexical Hash) ===");
        string a = "The background worker is stuck repeating the same plan without modifying any source files.";
        string bSame = "An autonomous coding loop keeps paraphrasing intentions while leaving the repository untouched.";
        string bDiff = "Quarterly invoice reconciliation exported tax totals to the PostgreSQL ledger table.";
        var hash = new Hash384Embedder();
        float neuralSame = Hash384Embedder.Dot(embedder.Encode(a), embedder.Encode(bSame));
        float neuralDiff = Hash384Embedder.Dot(embedder.Encode(a), embedder.Encode(bDiff));
        float hashSame = Hash384Embedder.Dot(hash.Encode(a), hash.Encode(bSame));
        float hashDiff = Hash384Embedder.Dot(hash.Encode(a), hash.Encode(bDiff));
        Console.WriteLine($"  Neural MiniLM : same_intent={neuralSame:F4} | unrelated={neuralDiff:F4} | margin={neuralSame - neuralDiff:F4}");
        Console.WriteLine($"  Lexical Hash  : same_intent={hashSame:F4} | unrelated={hashDiff:F4} | margin={hashSame - hashDiff:F4}");

        Console.WriteLine("\n=== Part C: Neural Semantic Thrash & Silent-Loop Watchdog ===");
        var healthy = ThrashDetector.Detect(embedder, DemoHealthySteps(), "neural-healthy");
        var stall = ThrashDetector.Detect(embedder, DemoStallSteps(), "neural-polite-stall");
        Console.WriteLine($"  healthy_trace     -> detected={healthy.ThrashDetected,-5} score={healthy.ThrashScore:F4} action={healthy.Action} ({healthy.LatencyMs} ms)");
        Console.WriteLine($"  polite_stall_loop -> detected={stall.ThrashDetected,-5} score={stall.ThrashScore:F4} action={stall.Action} ({stall.LatencyMs} ms)");

        Console.WriteLine($"\n=== Part D: Neural Claim Grounding on {repo} ===");
        var grounding = RepoClaimGrounder.Run(embedder, repo);
        Console.WriteLine($"  Indexed {grounding.SymbolsIndexed} symbols across {grounding.FilesScanned} files in {grounding.IndexMs} ms; query took {grounding.QueryMs} ms");
        Console.WriteLine($"  False claim : {grounding.SyntheticFalseClaim}");
        Console.WriteLine($"  Ground truth: {grounding.GroundTruthRef} -> caught_rank={grounding.CaughtRank?.ToString() ?? "none"}");

        var receipt = new
        {
            TimestampUtc = DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ"),
            Hardware = hw,
            Embedder = embedder.BackendName,
            ActiveProvider = embedder.ActiveProvider,
            ModelFile = embedder.ModelFileName,
            ModelMiB = modelMiB,
            MaxSeqLen = maxLen,
            PartANeuralEncodeMs = encodeMs,
            PartANeuralTextsPerSec = encodeTps,
            PartBParaphrase = new { NeuralSame = Math.Round(neuralSame, 4), NeuralDiff = Math.Round(neuralDiff, 4), HashSame = Math.Round(hashSame, 4), HashDiff = Math.Round(hashDiff, 4) },
            PartCThrash = new { healthy, stall },
            PartDGrounding = grounding
        };

        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(receiptPath))!);
        File.WriteAllText(receiptPath, JsonSerializer.Serialize(receipt, PrettyJson));
        Console.WriteLine($"\nSaved receipt -> {receiptPath}");
        return 0;
    }

    private static double MeasureMedianMs(int iterations, Action action)
    {
        // Warm up each batch shape 3 times so DirectML DX12 shaders and CPU/GPU boost clocks settle.
        for (int w = 0; w < 3; w++)
        {
            action();
        }

        var samples = new double[iterations];
        for (int i = 0; i < iterations; i++)
        {
            long t0 = Stopwatch.GetTimestamp();
            action();
            samples[i] = Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
        }

        Array.Sort(samples);
        return Math.Round(samples[iterations / 2], 2);
    }

    private static string[] BuildBenchmarkTexts(int count)
    {
        string[] seed =
        [
            "Inspected QuantizedSlabBench.cs DotInt8Avx2OrScalar to verify exact int16 pair accumulation without saturation.",
            "Edited ThrashDetector.cs sliding window parameters so polite paraphrased loops cross threshold cleanly.",
            "Reviewed TranscriptScanner.cs JSONL parser to filter bare parallel tool acknowledgements before scoring.",
            "Ran xUnit release suite against Steppewatch.Tests.csproj; all 9 vector kernel and watchdog tests passed."
        ];
        var list = new string[count];
        for (int i = 0; i < count; i++)
        {
            list[i] = $"Step {i + 1}: {seed[i % seed.Length]}";
        }

        return list;
    }

    private static string[] DemoHealthySteps() =>
    [
        "Step 1: Inspected webhook_client.py send_event() to verify current retry backoff and timeout defaults.",
        "Step 2: Edited webhook_client.py to add max_timeout_sec=15.0 and fixed TypeError on RetryConfig init.",
        "Step 3: Updated test_webhook_client.py to add test_retry_timeout_guard covering HTTP 504 responses.",
        "Step 4: Ran pytest test_webhook_client.py; caught AssertionError at line 88 and patched backoff jitter.",
        "Step 5: Re-ran pytest test_webhook_client.py; all 12 tests passed with exit code 0."
    ];

    private static string[] DemoStallSteps() =>
    [
        "Step 1: Let me carefully review the webhook integration requirements again to make sure I have the full picture before proposing an edit.",
        "Step 2: I will re-examine the webhook integration plan and verify how the retry timeout should be structured before making any code edits.",
        "Step 3: Let me double-check the webhook integration architecture one more time so the proposed design aligns with existing client boundaries.",
        "Step 4: Before writing any code, I want to carefully reconsider the webhook integration approach and review the timeout constraints again.",
        "Step 5: I am revisiting the webhook integration design notes again to confirm the retry timeout details before deciding on the next edit."
    ];

    private static string[] StripOnnxFlags(string[] args)
    {
        var flagsWithValue = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            "--model", "--vocab", "--provider", "--max-len", "--batch-size", "--device"
        };
        var kept = new List<string>(args.Length);
        for (int i = 0; i < args.Length; i++)
        {
            if (flagsWithValue.Contains(args[i]))
            {
                i++;
                continue;
            }

            kept.Add(args[i]);
        }

        return kept.ToArray();
    }

    private static string? GetArg(string[] args, string flag)
    {
        for (int i = 0; i < args.Length - 1; i++)
        {
            if (string.Equals(args[i], flag, StringComparison.OrdinalIgnoreCase))
            {
                return args[i + 1];
            }
        }

        return null;
    }

    private static int PrintUsage(int code = 0)
    {
        Console.WriteLine("Steppewatch ONNX (.NET 10 Optional Neural all-MiniLM-L6-v2 — DirectML / CPU)");
        Console.WriteLine("Usage: steppewatch-onnx <neural-bench | bench | xray | watch | train | self-tap> --model <model.onnx> [--vocab <vocab.txt>] [--provider auto|dml|cpu] [--max-len 128]");
        return code;
    }
}
