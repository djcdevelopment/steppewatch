using System.Text.Json;
using Steppewatch.Core;

namespace Steppewatch.Cli;

public static class Program
{
    private static readonly JsonSerializerOptions PrettyJson = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower
    };

    private static readonly JsonSerializerOptions LineJson = new()
    {
        WriteIndented = false,
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower
    };

    public static int Main(string[] args) => RunWithEmbedder(new Hash384Embedder(), args);

    public static int RunWithEmbedder(ITextEmbedder embedder, string[] args)
    {
        if (args.Length == 0 || args[0] is "-h" or "--help" or "help")
        {
            PrintUsage();
            return 0;
        }

        string cmd = args[0].ToLowerInvariant();
        var rest = args[1..];

        return cmd switch
        {
            "bench" => RunBench(embedder, rest),
            "xray" => RunXRay(embedder, rest),
            "watch" => RunWatch(embedder, rest),
            "train" => RunTrain(embedder, rest),
            "self-tap" => RunSelfTap(embedder, rest),
            _ => PrintUsage(1)
        };
    }

    private static int RunBench(ITextEmbedder embedder, string[] args)
    {
        string repo = Path.GetFullPath(GetArg(args, "--repo") ?? Directory.GetCurrentDirectory());
        string receiptPath = GetArg(args, "--receipt") ?? Path.Combine("receipts", "dotnet_win10_bench_receipt.json");
        var hw = HardwareProbe.Detect();

        Console.WriteLine("=== Hardware & .NET 10 SIMD Profile ===");
        Console.WriteLine(JsonSerializer.Serialize(hw, PrettyJson));

        Console.WriteLine("\n=== Part A: Quantized INT8 (AVX-VNNI/AVX2) vs FP32 SIMD Slab MatMul ===");
        var slab = QuantizedSlabBench.Run();
        Console.WriteLine($"Slab footprints at 262,144 cols: {slab.SlabInt8PlusScalesMiB} MiB (INT8+FP16 scales) | {slab.SlabFp32MiB} MiB (FP32)");
        PrintBenchMatrix($"INT8 Quantized ({slab.Int8Kernel}, {slab.SlabInt8PlusScalesMiB} MiB)", slab.Int8QuantizedMs);
        PrintBenchMatrix($"FP32 SIMD ({slab.Fp32Kernel}, {slab.SlabFp32MiB} MiB)", slab.Fp32SimdMs);

        Console.WriteLine("\n=== Part B: Zero-Token Semantic Thrash & Silent-Loop Watchdog ===");
        var healthy = ThrashDetector.Detect(embedder, DemoHealthySteps(), "demo-healthy");
        var stall = ThrashDetector.Detect(embedder, DemoStallSteps(), "demo-polite-stall");
        Console.WriteLine($"  healthy_trace     -> detected={healthy.ThrashDetected,-5} score={healthy.ThrashScore:F4} action={healthy.Action} ({healthy.LatencyMs} ms)");
        Console.WriteLine($"  polite_stall_loop -> detected={stall.ThrashDetected,-5} score={stall.ThrashScore:F4} action={stall.Action} ({stall.LatencyMs} ms)");

        Console.WriteLine($"\n=== Part C: Zero-Token Claim Grounding on {repo} ===");
        var grounding = RepoClaimGrounder.Run(embedder, repo);
        Console.WriteLine($"  Indexed {grounding.SymbolsIndexed} symbols across {grounding.FilesScanned} files in {grounding.IndexMs} ms; query took {grounding.QueryMs} ms");
        Console.WriteLine($"  False claim : {grounding.SyntheticFalseClaim}");
        Console.WriteLine($"  Ground truth: {grounding.GroundTruthRef} -> caught_rank={grounding.CaughtRank?.ToString() ?? "none"}");

        double q16Int8 = slab.Int8QuantizedMs["262144"]["Q=16"];
        double q16Fp32 = slab.Fp32SimdMs["262144"]["Q=16"];
        Console.WriteLine("\n=== What You Just Proved On Your Own Hardware ($0.00 API Cost, Zero Python) ===");
        Console.WriteLine($"  1. INT8 SIMD  : Q=16 vs 262,144 cols ({slab.SlabInt8PlusScalesMiB} MiB INT8+scales) in {q16Int8:F2} ms ({slab.Int8Kernel}) vs {q16Fp32:F2} ms FP32.");
        Console.WriteLine($"  2. Loop radar : Healthy ({healthy.ThrashScore:F4}) vs Silent Polite Stall ({stall.ThrashScore:F4}) at 0 cloud tokens.");
        Console.WriteLine($"  3. Claim check: {grounding.SymbolsIndexed} symbols indexed in {grounding.IndexMs} ms; false claim caught at Rank #{grounding.CaughtRank?.ToString() ?? "n/a"} ({grounding.QueryMs} ms).");

        var receipt = new { TimestampUtc = DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ"), Hardware = hw, Embedder = embedder.BackendName, PartASlab = slab, PartBThrash = new { healthy, stall }, PartCGrounding = grounding };
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(receiptPath))!);
        File.WriteAllText(receiptPath, JsonSerializer.Serialize(receipt, PrettyJson));
        Console.WriteLine($"\nSaved receipt -> {receiptPath}");
        return 0;
    }

    private static int RunXRay(ITextEmbedder embedder, string[] args)
    {
        int maxSessions = int.TryParse(GetArg(args, "--max-sessions"), out int m) ? m : 25;
        string? exportPath = GetArg(args, "--export-episodes");
        bool jsonMode = HasFlag(args, "--json");

        var sw = System.Diagnostics.Stopwatch.StartNew();
        var files = TranscriptScanner.DiscoverRecent(maxSessions);
        var rows = new List<SessionXRayRow>();
        var episodes = new List<ResidualEpisode>();

        foreach (var f in files)
        {
            var parsed = TranscriptScanner.ParseJsonl(f.FullPath);
            if (parsed.Steps.Count < 3) continue;
            var row = TranscriptScanner.ReplayWindows(embedder, f.Source, f.FullPath, parsed);
            rows.Add(row);
            if (exportPath is not null)
            {
                var thrash = ThrashDetector.Detect(embedder, parsed.Steps, Path.GetFileNameWithoutExtension(f.FullPath));
                episodes.Add(TranscriptScanner.BuildEpisode(embedder, f.Source, f.FullPath, parsed, thrash));
            }
        }

        if (exportPath is not null && episodes.Count > 0)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(exportPath))!);
            File.WriteAllLines(exportPath, episodes.Select(e => JsonSerializer.Serialize(e, LineJson)));
        }

        var sorted = rows.OrderByDescending(r => r.Flagged).ThenByDescending(r => r.MaxThrashScore).ThenByDescending(r => r.IntentDriftDelta).ToList();
        var payload = new
        {
            Schema = "steppewatch-session-xray.v1",
            Embedder = embedder.BackendName,
            ElapsedMs = Math.Round(sw.Elapsed.TotalMilliseconds, 2),
            SessionsScanned = rows.Count,
            SessionsFlagged = rows.Count(r => r.Flagged),
            ExportedEpisodesCount = episodes.Count,
            TopSessions = sorted.Take(10)
        };

        if (jsonMode) { Console.WriteLine(JsonSerializer.Serialize(payload, PrettyJson)); return 0; }

        Console.WriteLine($"=== Steppewatch .NET X-Ray ({rows.Count} sessions in {payload.ElapsedMs} ms, $0.00 API cost) ===");
        Console.WriteLine($"Flagged stall/loop runs: {payload.SessionsFlagged} / {rows.Count}");
        foreach (var r in sorted.Take(8))
        {
            Console.WriteLine($"  [{(r.Flagged ? "FLAGGED" : "OK     ")}] ({r.Source,-15}) {r.Session[..Math.Min(32, r.Session.Length)],-32} | steps={r.Steps,3} max_thrash={r.MaxThrashScore:F3} drift={r.IntentDriftDelta:F3} post_steps={r.PostFlagSteps}");
        }

        if (exportPath is not null) Console.WriteLine($"\nExported {episodes.Count} episodes -> {exportPath}");
        return 0;
    }

    private static int RunWatch(ITextEmbedder embedder, string[] args)
    {
        string? trace = GetArg(args, "--trace");
        string? intent = GetArg(args, "--intent");
        string episodesPath = GetArg(args, "--episodes") ?? Path.Combine("episodes", "transitions.jsonl");
        var ckpt = TinyJepaTrainer.LoadCheckpoint(GetArg(args, "--jepa-checkpoint"));
        bool jsonMode = HasFlag(args, "--json");

        var targets = trace is not null
            ? [new DiscoveredTranscript("explicit", trace, 0, 0)]
            : TranscriptScanner.DiscoverRecent(6);

        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(episodesPath))!);
        var results = new List<object>();
        int alerts = 0;

        foreach (var t in targets)
        {
            var parsed = TranscriptScanner.ParseJsonl(t.FullPath);
            var thrash = ThrashDetector.Detect(embedder, parsed.Steps, Path.GetFileNameWithoutExtension(t.FullPath));
            var ep = TranscriptScanner.BuildEpisode(embedder, t.Source, t.FullPath, parsed, thrash, intent, ckpt);
            File.AppendAllText(episodesPath, JsonSerializer.Serialize(ep, LineJson) + Environment.NewLine);
            if (thrash.ThrashDetected) alerts++;
            results.Add(new { t.Source, t.FullPath, StepsObserved = parsed.Steps.Count, parsed.FilteredAcks, thrash.ThrashDetected, thrash.ThrashScore, thrash.Action, ep.LocalPrediction.PredictedStallRisk, ep.AssessmentVsExpectations.StateSurpriseDelta, thrash.LatencyMs });
            if (!jsonMode)
            {
                Console.WriteLine($"[{(thrash.ThrashDetected ? "ALERT:STALL_OR_LOOP" : "OK")}] ({t.Source}) {Path.GetFileName(t.FullPath)} | steps={parsed.Steps.Count} (acks={parsed.FilteredAcks}) thrash={thrash.ThrashScore:F3} surprise={ep.AssessmentVsExpectations.StateSurpriseDelta:F3} ({thrash.LatencyMs} ms)");
            }
        }

        if (jsonMode)
        {
            Console.WriteLine(JsonSerializer.Serialize(new { Schema = "steppewatch-watch-report.v1", Embedder = embedder.BackendName, AlertsCount = alerts, Results = results }, PrettyJson));
        }

        return 0;
    }

    private static int RunTrain(ITextEmbedder embedder, string[] args)
    {
        bool demo = HasFlag(args, "--demo-synthetic");
        string episodesPath = GetArg(args, "--episodes") ?? Path.Combine("episodes", "transitions.jsonl");
        string outPath = GetArg(args, "--out") ?? Path.Combine("checkpoints", "my_i5_jepa.json");

        var episodes = demo ? TinyJepaTrainer.BuildDemoSyntheticEpisodes(40) : TinyJepaTrainer.LoadEpisodes(episodesPath);
        if (episodes.Count < 6)
        {
            Console.Error.WriteLine($"Need at least 6 episodes (found {episodes.Count}). Pass --demo-synthetic or run `steppewatch xray --export-episodes`.");
            return 1;
        }

        var (ckpt, rep) = TinyJepaTrainer.Train(embedder, episodes);
        TinyJepaTrainer.SaveCheckpoint(outPath, ckpt);
        Console.WriteLine($"=== Steppewatch .NET Tiny JEPA Training ({(demo ? "synthetic demo" : episodesPath)}) ===");
        Console.WriteLine($"Train / Val : {rep.TrainCount} / {rep.ValCount} episodes ({rep.Epochs} epochs in {rep.TrainMs} ms)");
        Console.WriteLine($"Val BEFORE  : state_surprise={rep.ValBefore.StateSurprise:F4} | risk_mae={rep.ValBefore.RiskMae:F4}");
        Console.WriteLine($"Val AFTER   : state_surprise={rep.ValAfter.StateSurprise:F4} | risk_mae={rep.ValAfter.RiskMae:F4}");
        Console.WriteLine($"Saved       -> {outPath}");
        return 0;
    }

    private static int RunSelfTap(ITextEmbedder embedder, string[] args)
    {
        string repo = Path.GetFullPath(GetArg(args, "--repo") ?? Directory.GetCurrentDirectory());
        var hw = HardwareProbe.Detect();
        var healthy = ThrashDetector.Detect(embedder, DemoHealthySteps(), "self-tap-healthy");
        var stall = ThrashDetector.Detect(embedder, DemoStallSteps(), "self-tap-stall");
        var transcripts = TranscriptScanner.DiscoverRecent(8);
        bool ok = !healthy.ThrashDetected && stall.ThrashDetected;

        Console.WriteLine(JsonSerializer.Serialize(new
        {
            Schema = "steppewatch-self-tap.v1",
            Runtime = "dotnet-10-native-simd",
            TargetRepo = repo,
            ReadOnlyOnTranscripts = true,
            ZeroApiTokens = true,
            Hardware = hw,
            Embedder = embedder.BackendName,
            SelfCheck = new { Ok = ok, HealthyScore = healthy.ThrashScore, StallScore = stall.ThrashScore },
            DiscoveredTranscriptsCount = transcripts.Count,
            DiscoveredTranscripts = transcripts
        }, PrettyJson));
        return ok ? 0 : 1;
    }

    private static void PrintBenchMatrix(string title, Dictionary<string, Dictionary<string, double>> matrix)
    {
        Console.WriteLine($"\n  Backend: {title}");
        foreach (var (n, qMap) in matrix)
        {
            string cols = string.Join("  ", qMap.Select(kv => $"{kv.Key}: {kv.Value,6:F2} ms"));
            Console.WriteLine($"    N={int.Parse(n),6} cols -> {cols}");
        }
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

    private static string? GetArg(string[] args, string flag)
    {
        for (int i = 0; i < args.Length - 1; i++) if (args[i] == flag) return args[i + 1];
        return null;
    }

    private static bool HasFlag(string[] args, string flag) => args.Contains(flag, StringComparer.OrdinalIgnoreCase);

    private static int PrintUsage(int code = 0)
    {
        Console.WriteLine("Steppewatch (.NET 10 Native SIMD — Windows 10 / Linux / macOS)");
        Console.WriteLine("Usage: steppewatch <bench | xray | watch | train | self-tap> [options]");
        return code;
    }
}
