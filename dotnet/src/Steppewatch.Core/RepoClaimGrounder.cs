using System.Diagnostics;
using System.Text.RegularExpressions;

namespace Steppewatch.Core;

public sealed record GroundingHit(int Rank, string Ref, double Score, string LineText);
public sealed record RepoGroundingReport(
    string Repo, int FilesScanned, int SymbolsIndexed, double IndexMs, double QueryMs,
    string SyntheticFalseClaim, string GroundTruthRef, bool CaughtInTop5, int? CaughtRank,
    List<GroundingHit> Top5Hits);

public static partial class RepoClaimGrounder
{
    private static readonly HashSet<string> AllowedExts = new(StringComparer.OrdinalIgnoreCase)
        { ".cs", ".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".md" };

    private static readonly HashSet<string> SkipDirs = new(StringComparer.OrdinalIgnoreCase)
        { ".git", ".venv", "venv", "node_modules", "bin", "obj", "dist", "build", "__pycache__" };

    [GeneratedRegex(@"^\s*(?:def|async\s+def|class|function|export\s+(?:async\s+)?function|func|fn|(?:public|private|internal|protected)\s+(?:static\s+|async\s+|sealed\s+|partial\s+)*(?:class|record|struct|interface|[A-Za-z0-9_<>\[\]?]+)\s+)\s*([A-Za-z_][A-Za-z0-9_]{3,})")]
    private static partial Regex SymbolDefRe();

    public static RepoGroundingReport Run(ITextEmbedder embedder, string repoDir, int maxFiles = 120)
    {
        long t0 = Stopwatch.GetTimestamp();
        var entries = new List<(string Rel, int Line, string Text)>();
        var symbols = new List<(string Rel, int Line, string Sym)>();
        int fileCount = 0;

        foreach (string fp in EnumerateRepoFiles(repoDir, maxFiles))
        {
            fileCount++;
            string rel = Path.GetRelativePath(repoDir, fp).Replace('\\', '/');
            string ext = Path.GetExtension(fp);
            string[] lines;
            try { lines = File.ReadAllLines(fp); }
            catch { continue; }

            for (int i = 0; i < lines.Length; i++)
            {
                string trimmed = lines[i].Trim();
                if (trimmed.Length < 8) continue;
                if (ext.Equals(".md", StringComparison.OrdinalIgnoreCase) && trimmed.StartsWith('#'))
                {
                    entries.Add((rel, i + 1, $"{rel}:{i + 1} | {trimmed}"));
                }
                else
                {
                    var m = SymbolDefRe().Match(lines[i]);
                    if (m.Success)
                    {
                        string sym = m.Groups[1].Value;
                        entries.Add((rel, i + 1, $"{rel}:{i + 1} | {trimmed}"));
                        if (sym.ToLowerInvariant() is not ("main" or "init" or "setup" or "test") && symbols.Count < 40)
                        {
                            symbols.Add((rel, i + 1, sym));
                        }
                    }
                }
            }
        }

        if (entries.Count == 0)
        {
            return new RepoGroundingReport(repoDir, fileCount, 0, 0, 0, string.Empty, string.Empty, false, null, []);
        }

        float[][] slab = embedder.EncodeBatch(entries.Select(e => e.Text).ToList());
        double indexMs = Math.Round(Stopwatch.GetElapsedTime(t0).TotalMilliseconds, 2);

        (string relTarget, int lnTarget, string claim) = symbols.Count > 0
            ? (symbols[symbols.Count / 2].Rel, symbols[symbols.Count / 2].Line,
               $"{symbols[symbols.Count / 2].Rel} is missing an implementation or test for {symbols[symbols.Count / 2].Sym}")
            : (entries[0].Rel, entries[0].Line, $"{entries[0].Rel} does not document {entries[0].Text}");

        long tq = Stopwatch.GetTimestamp();
        float[] qVec = embedder.Encode(claim);
        var scored = new (int Idx, double Score)[entries.Count];
        for (int i = 0; i < entries.Count; i++)
        {
            scored[i] = (i, QuantizedSlabBench.DotFp32Simd(slab[i], qVec));
        }

        Array.Sort(scored, (a, b) => b.Score.CompareTo(a.Score));
        double queryMs = Math.Round(Stopwatch.GetElapsedTime(tq).TotalMilliseconds, 2);

        string truthRef = $"{relTarget}:{lnTarget}";
        var top5 = scored.Take(5).Select((s, idx) => new GroundingHit(
            idx + 1,
            $"{entries[s.Idx].Rel}:{entries[s.Idx].Line}",
            Math.Round(s.Score, 4),
            entries[s.Idx].Text[..Math.Min(140, entries[s.Idx].Text.Length)])).ToList();

        int? caughtRank = top5.FirstOrDefault(h => h.Ref == truthRef)?.Rank;
        return new RepoGroundingReport(
            repoDir, fileCount, entries.Count, indexMs, queryMs,
            claim, truthRef, caughtRank is not null, caughtRank, top5);
    }

    private static IEnumerable<string> EnumerateRepoFiles(string root, int maxFiles)
    {
        var stack = new Stack<string>();
        stack.Push(root);
        int yielded = 0;
        while (stack.Count > 0 && yielded < maxFiles)
        {
            string dir = stack.Pop();
            string[] subDirs;
            string[] files;
            try
            {
                subDirs = Directory.GetDirectories(dir);
                files = Directory.GetFiles(dir);
            }
            catch { continue; }

            Array.Sort(subDirs, StringComparer.OrdinalIgnoreCase);
            for (int i = subDirs.Length - 1; i >= 0; i--)
            {
                string name = Path.GetFileName(subDirs[i]);
                if (!SkipDirs.Contains(name) && !name.StartsWith('.')) stack.Push(subDirs[i]);
            }

            Array.Sort(files, StringComparer.OrdinalIgnoreCase);
            foreach (string fp in files)
            {
                if (!AllowedExts.Contains(Path.GetExtension(fp))) continue;
                try { if (new FileInfo(fp).Length > 120_000) continue; } catch { continue; }
                yield return fp;
                if (++yielded >= maxFiles) yield break;
            }
        }
    }
}
