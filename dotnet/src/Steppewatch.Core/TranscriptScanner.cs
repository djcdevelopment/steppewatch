using System.Diagnostics;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace Steppewatch.Core;

public sealed record DiscoveredTranscript(string Source, string FullPath, long Bytes, double AgeSeconds);
public sealed record ParsedTranscript(string Intent, List<string> Steps, List<string> TouchedFiles, int LineCount, int FilteredAcks);

public sealed record ResidualEpisode(
    string Schema,
    string EpisodeId,
    string TimestampUtc,
    string SourceAgent,
    string TracePath,
    EpisodeState CurrentState,
    EpisodeIntent IntendedOutcome,
    EpisodePrediction LocalPrediction,
    EpisodeEndState EndState,
    EpisodeAssessment AssessmentVsExpectations);

public sealed record EpisodeState(string Summary, int StepCountObserved, int FilteredToolAcks);
public sealed record EpisodeIntent(string Summary);
public sealed record EpisodePrediction(string Source, double PredictedStallRisk);
public sealed record EpisodeEndState(string Summary, List<string> TouchedFiles, bool ThrashDetected, double ThrashScore, string Action);
public sealed record EpisodeAssessment(double StateSurpriseDelta, double RiskPredictionError, double ActualRiskTarget, string? HumanLabel);

public sealed record SessionXRayRow(
    string Source, string Session, int Steps, int AcksFiltered,
    bool Flagged, double MaxThrashScore, int? FirstFlagStep,
    int PostFlagSteps, int PostFlagEstTokens, double IntentDriftDelta, int TouchedFilesCount);

public static partial class TranscriptScanner
{
    [GeneratedRegex(@"\b([A-Za-z0-9_.\-/\\]+\.(?:py|cs|ts|tsx|js|go|rs|json|yaml|yml|md|sh|ps1))\b")]
    private static partial Regex FileHintRe();

    public static List<DiscoveredTranscript> DiscoverRecent(int maxFiles = 20)
    {
        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        var hits = new List<(DateTime Mtime, DiscoveredTranscript Item)>();
        var now = DateTime.UtcNow;

        void ScanDir(string root, string pattern, string source, SearchOption opt = SearchOption.AllDirectories)
        {
            if (!Directory.Exists(root)) return;
            try
            {
                foreach (string file in Directory.EnumerateFiles(root, pattern, opt))
                {
                    if (source == "claude_main" && file.Contains($"{Path.DirectorySeparatorChar}subagents{Path.DirectorySeparatorChar}", StringComparison.OrdinalIgnoreCase))
                        continue;
                    if (source == "claude_subagent" && !file.Contains($"{Path.DirectorySeparatorChar}subagents{Path.DirectorySeparatorChar}", StringComparison.OrdinalIgnoreCase))
                        continue;
                    var fi = new FileInfo(file);
                    double age = Math.Round(Math.Max(0.0, (now - fi.LastWriteTimeUtc).TotalSeconds), 1);
                    hits.Add((fi.LastWriteTimeUtc, new DiscoveredTranscript(source, file, fi.Length, age)));
                }
            }
            catch { /* Ignore inaccessible folders */ }
        }

        ScanDir(Path.Combine(home, ".claude", "projects"), "*.jsonl", "claude_main");
        ScanDir(Path.Combine(home, ".claude", "projects"), "agent-*.jsonl", "claude_subagent");
        ScanDir(Path.Combine(home, ".codex", "sessions"), "rollout-*.jsonl", "codex");
        ScanDir(Path.Combine(home, ".hermes", "logs", "sessions"), "*.jsonl", "hermes", SearchOption.TopDirectoryOnly);
        ScanDir(Path.Combine(home, ".hermes", "profiles"), "*.jsonl", "hermes_profile");

        return hits.OrderByDescending(h => h.Mtime).Take(maxFiles).Select(h => h.Item).ToList();
    }

    public static ParsedTranscript ParseJsonl(string path, int maxLines = 500)
    {
        if (!File.Exists(path)) return new ParsedTranscript(string.Empty, [], [], 0, 0);
        string[] allLines = File.ReadAllLines(path);
        var tail = allLines.Length > maxLines ? allLines[^maxLines..] : allLines;

        string intent = string.Empty;
        var steps = new List<string>();
        var touched = new List<string>();
        var touchedSet = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        int filteredAcks = 0;

        foreach (string rawLine in tail)
        {
            if (string.IsNullOrWhiteSpace(rawLine)) continue;
            try
            {
                using var doc = JsonDocument.Parse(rawLine);
                var root = doc.RootElement;
                if (root.ValueKind != JsonValueKind.Object) continue;

                string recType = GetString(root, "type") ?? GetString(root, "role") ?? string.Empty;
                string payloadType = root.TryGetProperty("payload", out var payloadEl) && payloadEl.ValueKind == JsonValueKind.Object
                    ? GetString(payloadEl, "type") ?? string.Empty
                    : string.Empty;

                string text = ExtractText(root).Trim();
                if (string.IsNullOrEmpty(text)) continue;

                bool isUser = recType.Equals("user", StringComparison.OrdinalIgnoreCase)
                    || recType.Equals("user_input", StringComparison.OrdinalIgnoreCase)
                    || payloadType.Equals("user_message", StringComparison.OrdinalIgnoreCase);

                if (string.IsNullOrEmpty(intent) && (isUser || text.Contains("INTENT:", StringComparison.Ordinal)))
                {
                    intent = text.Length > 400 ? text[..400] : text;
                }

                if (!isUser)
                {
                    string compact = Regex.Replace(text, @"\s+", " ");
                    if (compact.Length > 500) compact = compact[..500];
                    if (ThrashDetector.IsBareToolAck(compact))
                    {
                        filteredAcks++;
                        continue;
                    }

                    steps.Add(compact);
                    foreach (Match m in FileHintRe().Matches(compact))
                    {
                        string fn = m.Groups[1].Value;
                        if (touchedSet.Add(fn) && touched.Count < 25) touched.Add(fn);
                    }
                }
            }
            catch { /* Ignore malformed partial trailing line */ }
        }

        return new ParsedTranscript(
            string.IsNullOrEmpty(intent) ? $"Supervise session {Path.GetFileName(path)}" : intent,
            steps, touched, allLines.Length, filteredAcks);
    }

    public static ResidualEpisode BuildEpisode(
        ITextEmbedder embedder, string source, string tracePath,
        ParsedTranscript parsed, ThrashResult thrash,
        string? explicitIntent = null, TinyJepaCheckpoint? ckpt = null)
    {
        string curr = parsed.Steps.Count > 0 ? parsed.Steps[0] : $"Session start ({source}: {Path.GetFileName(tracePath)})";
        string end = parsed.Steps.Count > 0 ? string.Join(" | ", parsed.Steps.TakeLast(2)) : curr;
        string intent = explicitIntent ?? parsed.Intent;
        if (intent.Length > 400) intent = intent[..400];

        float[][] embs = embedder.EncodeBatch([curr, intent, end]);
        float[] zHat = embs[1];
        double predRisk = 0.20;
        string predSrc = "zero_shot_intent_prior";

        if (ckpt is not null)
        {
            (zHat, predRisk) = TinyJepaTrainer.Predict(ckpt, embs[0], embs[1]);
            predSrc = "local_jepa_dotnet";
        }

        double cos = Math.Clamp(QuantizedSlabBench.DotFp32Simd(zHat, embs[2]), -1.0, 1.0);
        double surprise = Math.Round(1.0 - cos, 4);
        double actualRisk = thrash.ThrashDetected ? 1.0 : thrash.ThrashScore;
        double riskErr = Math.Round(Math.Abs(actualRisk - predRisk), 4);
        string stem = Path.GetFileNameWithoutExtension(tracePath);
        if (stem.Length > 18) stem = stem[..18];

        return new ResidualEpisode(
            Schema: "jepa-residual-episode.v1",
            EpisodeId: $"ep-{DateTimeOffset.UtcNow.ToUnixTimeSeconds()}-{stem}",
            TimestampUtc: DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ"),
            SourceAgent: source,
            TracePath: tracePath,
            CurrentState: new EpisodeState(curr.Length > 300 ? curr[..300] : curr, parsed.Steps.Count, parsed.FilteredAcks),
            IntendedOutcome: new EpisodeIntent(intent),
            LocalPrediction: new EpisodePrediction(predSrc, Math.Round(predRisk, 4)),
            EndState: new EpisodeEndState(end.Length > 300 ? end[..300] : end, parsed.TouchedFiles.Take(12).ToList(), thrash.ThrashDetected, thrash.ThrashScore, thrash.Action),
            AssessmentVsExpectations: new EpisodeAssessment(surprise, riskErr, Math.Round(actualRisk, 4), null));
    }

    public static SessionXRayRow ReplayWindows(ITextEmbedder embedder, string source, string path, ParsedTranscript parsed)
    {
        var steps = parsed.Steps.Count > 120 ? parsed.Steps.TakeLast(120).ToList() : parsed.Steps;
        double maxScore = 0.0;
        int? firstFlag = null;
        for (int end = Math.Min(4, steps.Count); end <= steps.Count; end += 2)
        {
            int start = Math.Max(0, end - 5);
            var res = ThrashDetector.Detect(embedder, steps.GetRange(start, end - start), Path.GetFileNameWithoutExtension(path));
            if (res.ThrashScore > maxScore) maxScore = res.ThrashScore;
            if (res.ThrashDetected && firstFlag is null) firstFlag = end;
        }

        int postSteps = firstFlag is int f && f < steps.Count ? steps.Count - f : 0;
        int postTokens = firstFlag is int f2 && f2 < steps.Count
            ? steps.Skip(f2).Sum(s => Math.Max(1, s.Length / 4))
            : 0;

        string endSummary = steps.Count > 0 ? string.Join(" | ", steps.TakeLast(2)) : parsed.Intent;
        float[][] embs = embedder.EncodeBatch([parsed.Intent, endSummary]);
        double drift = Math.Round(1.0 - Math.Clamp(QuantizedSlabBench.DotFp32Simd(embs[0], embs[1]), -1.0, 1.0), 4);

        return new SessionXRayRow(
            source, Path.GetFileName(path), parsed.Steps.Count, parsed.FilteredAcks,
            firstFlag is not null, Math.Round(maxScore, 4), firstFlag, postSteps, postTokens,
            drift, parsed.TouchedFiles.Count);
    }

    private static string? GetString(JsonElement obj, string prop) =>
        obj.TryGetProperty(prop, out var el) && el.ValueKind == JsonValueKind.String ? el.GetString() : null;

    private static string ExtractText(JsonElement root)
    {
        if (root.TryGetProperty("message", out var msg))
        {
            string s = ExtractContentElement(msg);
            if (!string.IsNullOrEmpty(s)) return s;
        }

        if (root.TryGetProperty("payload", out var payload))
        {
            if (payload.ValueKind == JsonValueKind.Object)
            {
                if (payload.TryGetProperty("message", out var pm))
                {
                    string s = ExtractContentElement(pm);
                    if (!string.IsNullOrEmpty(s)) return s;
                }

                string pc = ExtractContentElement(payload);
                if (!string.IsNullOrEmpty(pc)) return pc;
            }
        }

        return ExtractContentElement(root);
    }

    private static string ExtractContentElement(JsonElement el)
    {
        if (el.ValueKind == JsonValueKind.String) return el.GetString() ?? string.Empty;
        if (el.ValueKind == JsonValueKind.Array)
        {
            var sb = new StringBuilder();
            foreach (var item in el.EnumerateArray())
            {
                string part = ExtractContentElement(item);
                if (!string.IsNullOrEmpty(part)) sb.Append(part).Append(' ');
            }

            return sb.ToString().Trim();
        }

        if (el.ValueKind == JsonValueKind.Object)
        {
            if (el.TryGetProperty("content", out var c)) return ExtractContentElement(c);
            if (el.TryGetProperty("text", out var t) && t.ValueKind == JsonValueKind.String) return t.GetString() ?? string.Empty;
            if (GetString(el, "type") == "tool_use")
            {
                string name = GetString(el, "name") ?? "tool";
                return $"tool:{name}({el.GetRawText()[..Math.Min(140, el.GetRawText().Length)]})";
            }
        }

        return string.Empty;
    }
}
