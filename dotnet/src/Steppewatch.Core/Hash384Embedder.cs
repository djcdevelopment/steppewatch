using System.Text.RegularExpressions;

namespace Steppewatch.Core;

public interface ITextEmbedder
{
    int Dim { get; }
    string BackendName { get; }
    float[] Encode(string text);
    float[][] EncodeBatch(IReadOnlyList<string> texts);
}

public sealed partial class Hash384Embedder : ITextEmbedder
{
    public const int EmbedDim = 384;
    public int Dim => EmbedDim;
    public string BackendName => "hash384_dotnet_simd";

    [GeneratedRegex(@"[a-z0-9_./:\-]{2,}", RegexOptions.Compiled)]
    private static partial Regex TokenRegex();

    public float[] Encode(string text)
    {
        var vec = new float[EmbedDim];
        if (string.IsNullOrWhiteSpace(text))
        {
            return vec;
        }

        string normalized = text.Length > 1200 ? text[..1200].ToLowerInvariant() : text.ToLowerInvariant();
        var matches = TokenRegex().Matches(normalized);
        var words = new List<string>(matches.Count);
        foreach (Match m in matches)
        {
            string w = m.Value;
            words.Add(w);
            AddFeature(vec, "w:", w, 1.5f);
            if (w.Length >= 4)
            {
                for (int i = 0; i <= w.Length - 3; i++)
                {
                    AddFeature(vec, "c:", w.AsSpan(i, 3), 0.45f);
                }
            }
        }

        for (int i = 0; i < words.Count - 1; i++)
        {
            AddBigram(vec, words[i], words[i + 1], 1.1f);
        }

        NormalizeInPlace(vec);
        return vec;
    }

    public float[][] EncodeBatch(IReadOnlyList<string> texts)
    {
        var result = new float[texts.Count][];
        if (texts.Count >= 16)
        {
            Parallel.For(0, texts.Count, i => result[i] = Encode(texts[i]));
        }
        else
        {
            for (int i = 0; i < texts.Count; i++)
            {
                result[i] = Encode(texts[i]);
            }
        }

        return result;
    }

    public static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        float sum = 0f;
        for (int i = 0; i < a.Length; i++)
        {
            sum += a[i] * b[i];
        }

        return sum;
    }

    public static void NormalizeInPlace(Span<float> vec)
    {
        double sq = 0.0;
        for (int i = 0; i < vec.Length; i++)
        {
            sq += vec[i] * vec[i];
        }

        float norm = (float)Math.Sqrt(sq);
        if (norm > 1e-8f)
        {
            float inv = 1.0f / norm;
            for (int i = 0; i < vec.Length; i++)
            {
                vec[i] *= inv;
            }
        }
    }

    private static ulong Avalanche(ulong h)
    {
        h ^= h >> 30;
        h *= 0xbf58476d1ce4e5b9UL;
        h ^= h >> 27;
        h *= 0x94d049bb133111ebUL;
        return h ^ (h >> 31);
    }

    private static void AddFeature(float[] vec, string prefix, ReadOnlySpan<char> token, float weight)
    {
        ulong h = 14695981039346656037UL;
        foreach (char c in prefix)
        {
            h = (h ^ c) * 1099511628211UL;
        }

        foreach (char c in token)
        {
            h = (h ^ c) * 1099511628211UL;
        }

        h = Avalanche(h);
        int bucket = (int)(h % EmbedDim);
        float sign = ((h >> 16) & 1UL) == 0 ? 1.0f : -1.0f;
        vec[bucket] += sign * weight;
    }

    private static void AddBigram(float[] vec, string a, string b, float weight)
    {
        ulong h = 14695981039346656037UL;
        foreach (char c in "b:")
        {
            h = (h ^ c) * 1099511628211UL;
        }

        foreach (char c in a)
        {
            h = (h ^ c) * 1099511628211UL;
        }

        h = (h ^ '_') * 1099511628211UL;
        foreach (char c in b)
        {
            h = (h ^ c) * 1099511628211UL;
        }

        h = Avalanche(h);
        int bucket = (int)(h % EmbedDim);
        float sign = ((h >> 16) & 1UL) == 0 ? 1.0f : -1.0f;
        vec[bucket] += sign * weight;
    }
}
