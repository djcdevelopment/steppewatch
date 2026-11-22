using System.Globalization;
using System.Text;

namespace Steppewatch.Onnx;

public sealed record TokenizedBatch(
    long[] InputIds,
    long[] AttentionMask,
    long[] TokenTypeIds,
    int BatchSize,
    int SeqLen);

/// <summary>
/// Zero-external-dependency BERT uncased WordPiece tokenizer compatible with all-MiniLM-L6-v2 vocab.txt.
/// </summary>
public sealed class WordPieceTokenizer
{
    private readonly Dictionary<string, int> _vocab;
    private readonly int _padId;
    private readonly int _unkId;
    private readonly int _clsId;
    private readonly int _sepId;
    private readonly int _maxWordChars;

    public int VocabSize => _vocab.Count;

    public WordPieceTokenizer(IReadOnlyDictionary<string, int> vocab, int maxWordChars = 100)
    {
        _vocab = new Dictionary<string, int>(vocab, StringComparer.Ordinal);
        _padId = _vocab.GetValueOrDefault("[PAD]", 0);
        _unkId = _vocab.GetValueOrDefault("[UNK]", 100);
        _clsId = _vocab.GetValueOrDefault("[CLS]", 101);
        _sepId = _vocab.GetValueOrDefault("[SEP]", 102);
        _maxWordChars = maxWordChars;
    }

    public static WordPieceTokenizer FromVocabFile(string vocabPath)
    {
        var vocab = new Dictionary<string, int>(32768, StringComparer.Ordinal);
        int id = 0;
        foreach (string rawLine in File.ReadLines(vocabPath, Encoding.UTF8))
        {
            string token = rawLine.TrimEnd('\r', '\n');
            if (token.Length > 0 && !vocab.ContainsKey(token))
            {
                vocab[token] = id;
            }

            id++;
        }

        if (vocab.Count == 0)
        {
            throw new InvalidOperationException($"Vocab file is empty: {vocabPath}");
        }

        return new WordPieceTokenizer(vocab);
    }

    public int[] EncodeIds(string text, int maxLength = 128)
    {
        if (maxLength < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(maxLength), "maxLength must be >= 2 for [CLS] and [SEP].");
        }

        var ids = new List<int>(Math.Min(maxLength, 64)) { _clsId };
        int contentCap = maxLength - 2;

        foreach (string basicToken in BasicTokenizeUncased(text))
        {
            if (ids.Count - 1 >= contentCap)
            {
                break;
            }

            foreach (int wpId in WordPieceSplit(basicToken))
            {
                if (ids.Count - 1 >= contentCap)
                {
                    break;
                }

                ids.Add(wpId);
            }
        }

        ids.Add(_sepId);
        return ids.ToArray();
    }

    public TokenizedBatch EncodeBatch(IReadOnlyList<string> texts, int maxLength = 128, bool padToStaticBucket = false)
    {
        int batchSize = texts.Count;
        var encoded = new int[batchSize][];
        int maxSeq = 2;

        for (int b = 0; b < batchSize; b++)
        {
            encoded[b] = EncodeIds(texts[b] ?? string.Empty, maxLength);
            if (encoded[b].Length > maxSeq)
            {
                maxSeq = encoded[b].Length;
            }
        }

        if (padToStaticBucket)
        {
            maxSeq = SnapToStaticBucket(maxSeq, maxLength);
        }

        var inputIds = new long[batchSize * maxSeq];
        var attentionMask = new long[batchSize * maxSeq];
        var tokenTypeIds = new long[batchSize * maxSeq];

        if (_padId != 0)
        {
            Array.Fill(inputIds, _padId);
        }

        for (int b = 0; b < batchSize; b++)
        {
            int[] seq = encoded[b];
            int rowOff = b * maxSeq;
            for (int t = 0; t < seq.Length; t++)
            {
                inputIds[rowOff + t] = seq[t];
                attentionMask[rowOff + t] = 1L;
            }
        }

        return new TokenizedBatch(inputIds, attentionMask, tokenTypeIds, batchSize, maxSeq);
    }

    public static int SnapToStaticBucket(int seqLen, int maxLength)
    {
        if (seqLen <= 64 && maxLength >= 64)
        {
            return 64;
        }

        if (seqLen <= 128 && maxLength >= 128)
        {
            return 128;
        }

        if (seqLen <= 256 && maxLength >= 256)
        {
            return 256;
        }

        return maxLength;
    }

    private IEnumerable<int> WordPieceSplit(string token)
    {
        if (token.Length > _maxWordChars)
        {
            yield return _unkId;
            yield break;
        }

        var subIds = new List<int>();
        int start = 0;
        bool failed = false;

        while (start < token.Length)
        {
            int end = token.Length;
            int foundId = -1;

            while (start < end)
            {
                string piece = start == 0 ? token[start..end] : "##" + token[start..end];
                if (_vocab.TryGetValue(piece, out int id))
                {
                    foundId = id;
                    break;
                }

                end--;
            }

            if (foundId < 0)
            {
                failed = true;
                break;
            }

            subIds.Add(foundId);
            start = end;
        }

        if (failed)
        {
            yield return _unkId;
        }
        else
        {
            foreach (int id in subIds)
            {
                yield return id;
            }
        }
    }

    private static List<string> BasicTokenizeUncased(string text)
    {
        var tokens = new List<string>();
        if (string.IsNullOrEmpty(text))
        {
            return tokens;
        }

        string decomposed = text.ToLowerInvariant().Normalize(NormalizationForm.FormD);
        var current = new StringBuilder(32);

        void Flush()
        {
            if (current.Length > 0)
            {
                tokens.Add(current.ToString());
                current.Clear();
            }
        }

        foreach (char ch in decomposed)
        {
            UnicodeCategory cat = CharUnicodeInfo.GetUnicodeCategory(ch);
            if (cat == UnicodeCategory.NonSpacingMark)
            {
                continue; // strip accents for bert-base-uncased
            }

            if (ch == '\0' || ch == '\uFFFD' || cat == UnicodeCategory.Control || cat == UnicodeCategory.Format)
            {
                if (ch is '\t' or '\n' or '\r')
                {
                    Flush();
                }

                continue;
            }

            if (char.IsWhiteSpace(ch))
            {
                Flush();
                continue;
            }

            if (IsBertPunctuation(ch))
            {
                Flush();
                tokens.Add(ch.ToString());
                continue;
            }

            if (IsCjkChar(ch))
            {
                Flush();
                tokens.Add(ch.ToString());
                continue;
            }

            current.Append(ch);
        }

        Flush();
        return tokens;
    }

    private static bool IsBertPunctuation(char ch)
    {
        int cp = ch;
        if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) || (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126))
        {
            return true;
        }

        return char.IsPunctuation(ch);
    }

    private static bool IsCjkChar(char ch) =>
        (ch >= 0x4E00 && ch <= 0x9FFF) ||
        (ch >= 0x3400 && ch <= 0x4DBF) ||
        (ch >= 0xF900 && ch <= 0xFAFF);
}
