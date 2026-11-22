using System.Diagnostics;
using System.Text.RegularExpressions;

namespace Steppewatch.Core;

public sealed record ThrashResult(
    string TraceId,
    bool ThrashDetected,
    double ThrashScore,
    double ThrashThreshold,
    string Action,
    double MeanAdjCosine,
    double MaxOscCosine,
    double AllPairsCosine,
    double ErrorRepeatRatio,
    double StagnationRatio,
    int TailNewTargets,
    bool ProductiveDebugProgression,
    double ProgressScore,
    double LatencyMs,
    string Backend);

public static partial class ThrashDetector
{
    public const double DefaultThreshold = 0.65;

    private static readonly HashSet<string> EnglishGerunds = new(StringComparer.OrdinalIgnoreCase)
    {
        "inspecting", "looking", "reviewing", "examining", "checking", "reading",
        "taking", "repeating", "running", "retrying", "regenerating", "generating",
        "attempting", "trying", "verifying", "confirming", "searching", "scanning",
        "tracing", "analyzing", "evaluating", "revisiting", "reconsidering",
        "let", "please", "still", "again", "another", "every", "each", "step", "pass", "turn"
    };

    [GeneratedRegex(@"^\s*(?:step|turn|attempt|iter(?:ation)?)\s*\d+\s*[:.\-]\s*", RegexOptions.IgnoreCase)]
    private static partial Regex StepPrefixRe();

    [GeneratedRegex(@"\b(error|exception|traceback|failed|refused|timeout|timed out|syntaxerror|keyerror|valueerror|typeerror|assertionerror|runtimeerror|connectionrefusederror|rc=[1-9]|http\s+[45]\d\d)\b", RegexOptions.IgnoreCase)]
    private static partial Regex ErrorSignatureRe();

    [GeneratedRegex(@"\b([A-Za-z]+Error|HTTP\s+\d{3}|CS\d{4}|TS\d{4}|F\d{3}|rc=[1-9]\d*|line\s+\d+)\b")]
    private static partial Regex SpecificExceptionRe();

    [GeneratedRegex(@"\b(applied\s+fix|fixed|fix\s+the|updated|edited|added\s+[A-Za-z_]|patched|implemented|created|wrote|resolved|refactored|ran\s+(?:pytest|dotnet|npm|cargo))\b", RegexOptions.IgnoreCase)]
    private static partial Regex ConcreteFixRe();

    [GeneratedRegex(@"\b(passed|0\s+errors|exit\s+code\s+0|http\s+200|test\s+result:\s*ok|accepted|verified|succeeded|all\s+\d+\s+tests\s+passed)\b", RegexOptions.IgnoreCase)]
    private static partial Regex ResolutionSuccessRe();

    [GeneratedRegex(@"\b(failed|error=[1-9]|rc=[1-9]|unresolved)\b", RegexOptions.IgnoreCase)]
    private static partial Regex FailureTailRe();

    [GeneratedRegex(@"^(?:created\s+file\.?|updated\s+file\.?|file\s+(?:created|updated|written|saved)\.?|made\s+\d+\s+replacements?(?:\s+in\s+\S+)?\.?|ok\.?|done\.?|success\.?|command\s+exited\s+with\s+code\s+0\.?)$", RegexOptions.IgnoreCase)]
    private static partial Regex BareToolAckRe();

    [GeneratedRegex(@"\b([A-Za-z0-9_.\-]+\.(?:py|cs|ts|tsx|js|jsx|go|rs|java|cpp|h|json|yaml|yml|toml|md|sh|ps1))\b")]
    private static partial Regex FileNameRe();

    [GeneratedRegex(@"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")]
    private static partial Regex FnCallRe();

    [GeneratedRegex(@"\b[a-zA-Z_][a-zA-Z0-9_]*_[a-zA-Z0-9_]+\b|\b[A-Z][A-Z0-9_]{2,}\b")]
    private static partial Regex SnakeOrConstRe();

    [GeneratedRegex(@"\b[A-Za-z0-9]+[a-z0-9][A-Z][A-Za-z0-9]*\b")]
    private static partial Regex CamelOrPascalRe();

    [GeneratedRegex(@"\b(?:line\s+\d+|http\s+\d{3}|cs\d{4}|ts\d{4}|f\d{3}|rc=\d+)\b", RegexOptions.IgnoreCase)]
    private static partial Regex DiagLocRe();

    public static bool IsBareToolAck(string text)
    {
        string compact = Regex.Replace(text ?? string.Empty, @"\s+", " ").Trim();
        if (compact.Length < 12 && compact.ToLowerInvariant() is "ok" or "ok." or "done" or "done." or "true" or "null")
        {
            return true;
        }

        return BareToolAckRe().IsMatch(compact);
    }

    public static HashSet<string> ExtractTrueCodeTargets(string stepText)
    {
        string cleaned = StepPrefixRe().Replace(stepText, string.Empty);
        var targets = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        foreach (Match m in FileNameRe().Matches(cleaned)) targets.Add(m.Groups[1].Value.ToLowerInvariant());
        foreach (Match m in FnCallRe().Matches(cleaned))
        {
            string v = m.Groups[1].Value.ToLowerInvariant();
            if (!EnglishGerunds.Contains(v)) targets.Add(v);
        }

        foreach (Match m in SnakeOrConstRe().Matches(cleaned)) targets.Add(m.Value.ToLowerInvariant());
        foreach (Match m in CamelOrPascalRe().Matches(cleaned))
        {
            string v = m.Value.ToLowerInvariant();
            if (!EnglishGerunds.Contains(v)) targets.Add(v);
        }

        foreach (Match m in DiagLocRe().Matches(cleaned))
        {
            targets.Add(Regex.Replace(m.Value.ToLowerInvariant(), @"\s+", "_"));
        }

        return targets;
    }

    public static string NormalizeStep(string step)
    {
        string trimmed = (step ?? string.Empty).Trim();
        if (trimmed.Length > 600) trimmed = trimmed[..600];
        string stripped = StepPrefixRe().Replace(trimmed, string.Empty).Trim();
        return stripped.Length > 0 ? stripped : trimmed;
    }

    public static ThrashResult Detect(
        ITextEmbedder embedder,
        IReadOnlyList<string> steps,
        string traceId = "trace-0",
        double threshold = DefaultThreshold)
    {
        long t0 = Stopwatch.GetTimestamp();
        var normWindow = steps.Where(s => !string.IsNullOrWhiteSpace(s))
            .TakeLast(6)
            .Select(NormalizeStep)
            .ToList();

        if (normWindow.Count < 3)
        {
            return new ThrashResult(
                traceId, false, 0.0, threshold, "continue",
                0, 0, 0, 0, 0, 0, false, 1.0,
                Math.Round(Stopwatch.GetElapsedTime(t0).TotalMilliseconds, 2),
                embedder.BackendName);
        }

        float[][] embs = embedder.EncodeBatch(normWindow);
        return ScorePreEmbeddedWindow(normWindow, embs, embedder.BackendName, traceId, threshold, t0);
    }

    public static ThrashResult ScorePreEmbeddedWindow(
        IReadOnlyList<string> normWindow,
        IReadOnlyList<float[]> embs,
        string backendName,
        string traceId = "trace-0",
        double threshold = DefaultThreshold,
        long? startTimestamp = null)
    {
        long t0 = startTimestamp ?? Stopwatch.GetTimestamp();
        if (normWindow.Count < 3 || embs.Count != normWindow.Count)
        {
            return new ThrashResult(
                traceId, false, 0.0, threshold, "continue",
                0, 0, 0, 0, 0, 0, false, 1.0,
                Math.Round(Stopwatch.GetElapsedTime(t0).TotalMilliseconds, 2),
                backendName);
        }

        var adj = new List<double>();
        var osc = new List<double>();
        var upper = new List<double>();

        for (int i = 0; i < normWindow.Count; i++)
        {
            for (int j = i + 1; j < normWindow.Count; j++)
            {
                double sim = QuantizedSlabBench.DotFp32Simd(embs[i], embs[j]);
                upper.Add(sim);
                if (j == i + 1) adj.Add(sim);
                if (j == i + 2) osc.Add(sim);
            }
        }

        double meanAdj = adj.Average();
        double maxOsc = osc.Count > 0 ? osc.Max() : 0.0;
        double allPairs = upper.Count > 0 ? upper.Average() : meanAdj;
        double effCos = Math.Max(meanAdj, Math.Max(0.94 * maxOsc, allPairs));

        double rawErrRatio = normWindow.Count(s => ErrorSignatureRe().IsMatch(s)) / (double)normWindow.Count;
        var distinctErrSigs = normWindow
            .Select(s => string.Join("|", SpecificExceptionRe().Matches(s).Select(m => m.Value.ToLowerInvariant()).Order()))
            .Where(sig => sig.Length > 0)
            .Distinct()
            .Count();

        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        int tailNew = 0;
        int zeroTail = 0;
        for (int idx = 0; idx < normWindow.Count; idx++)
        {
            var cur = ExtractTrueCodeTargets(normWindow[idx]);
            int added = cur.Count(t => !seen.Contains(t));
            seen.UnionWith(cur);
            if (idx > 0)
            {
                tailNew += added;
                if (added == 0) zeroTail++;
            }
        }

        double stagRatio = zeroTail / (double)Math.Max(1, normWindow.Count - 1);
        int fixSteps = normWindow.Skip(1).Count(s => ConcreteFixRe().IsMatch(s));
        bool finalResolved = ResolutionSuccessRe().IsMatch(normWindow[^1]) && !FailureTailRe().IsMatch(normWindow[^1]);
        bool productive = (distinctErrSigs >= 3 && fixSteps >= 2) || (finalResolved && fixSteps >= 1 && tailNew >= 2);

        double effErr = productive ? 0.0 : rawErrRatio;
        double progress = productive
            ? Math.Min(1.0, (tailNew + 2 * fixSteps) / 6.0)
            : Math.Min(1.0, tailNew / Math.Max(3.0, normWindow.Count * 1.5));

        bool isHash = backendName.StartsWith("hash384", StringComparison.OrdinalIgnoreCase);
        double cosFloor = isHash ? 0.36 : 0.56;
        double silentFloor = isHash ? 0.38 : 0.58;

        double cosComp = Math.Clamp((effCos - cosFloor) / 0.28, 0.0, 1.0);
        double silentBoost = (effCos >= silentFloor && stagRatio >= 0.66 && fixSteps == 0 && !finalResolved)
            ? 0.44 * stagRatio
            : 0.0;

        double score = Math.Clamp(
            0.62 * cosComp
            + 0.34 * effErr
            + silentBoost
            + (effCos >= 0.84 ? 0.14 : 0.0)
            - 0.36 * progress
            - (productive ? 0.35 : 0.0),
            0.0, 1.0);

        bool detected = score >= threshold;
        return new ThrashResult(
            traceId, detected, Math.Round(score, 4), threshold,
            detected ? "early_abort_and_reroute" : "continue",
            Math.Round(meanAdj, 4), Math.Round(maxOsc, 4), Math.Round(allPairs, 4),
            Math.Round(rawErrRatio, 4), Math.Round(stagRatio, 4),
            tailNew, productive, Math.Round(progress, 4),
            Math.Round(Stopwatch.GetElapsedTime(t0).TotalMilliseconds, 2),
            backendName);
    }
}
