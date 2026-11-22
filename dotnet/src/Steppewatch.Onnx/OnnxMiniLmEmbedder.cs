using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Steppewatch.Core;

namespace Steppewatch.Onnx;

public enum OnnxExecutionProvider
{
    Auto,
    DirectML,
    Cpu
}

/// <summary>
/// Optional neural 384-D all-MiniLM-L6-v2 embedder powered by ONNX Runtime (DirectML on Win10 iGPU/GPU or CPU AVX2/VNNI).
/// Implements Steppewatch.Core.ITextEmbedder while keeping Steppewatch.Core zero-dependency.
/// </summary>
public sealed class OnnxMiniLmEmbedder : ITextEmbedder, IDisposable
{
    private readonly InferenceSession _session;
    private readonly WordPieceTokenizer _tokenizer;
    private readonly int _maxLength;
    private readonly int _maxBatchSize;
    private readonly bool _expectsTokenTypeIds;
    private readonly bool _useStaticBuckets;
    private readonly string _outputName;

    public int Dim => Hash384Embedder.EmbedDim;
    public string ActiveProvider { get; }
    public string? ProviderFallbackReason { get; }
    public string ModelFileName { get; }
    public string BackendName => $"onnx_minilm_{ActiveProvider}({ModelFileName})";

    public OnnxMiniLmEmbedder(
        string modelPath,
        string vocabPath,
        OnnxExecutionProvider requestedProvider = OnnxExecutionProvider.Auto,
        int maxLength = 128,
        int maxBatchSize = 16,
        int dmlDeviceId = 0)
    {
        if (!File.Exists(modelPath))
        {
            throw new FileNotFoundException($"ONNX model file not found: {modelPath}", modelPath);
        }

        if (!File.Exists(vocabPath))
        {
            throw new FileNotFoundException($"Tokenizer vocab file not found: {vocabPath}", vocabPath);
        }

        _tokenizer = WordPieceTokenizer.FromVocabFile(vocabPath);
        _maxLength = Math.Clamp(maxLength, 16, 512);
        _maxBatchSize = Math.Clamp(maxBatchSize, 1, 128);
        ModelFileName = Path.GetFileName(modelPath);

        (_session, ActiveProvider, ProviderFallbackReason) = CreateSession(modelPath, requestedProvider, dmlDeviceId);
        _expectsTokenTypeIds = _session.InputMetadata.ContainsKey("token_type_ids");
        // DirectML recompiles DX12 metacommands on every unseen sequence length; snapping to static
        // buckets (64 / 128) eliminates per-batch shader recompilation stalls on Windows 10 iGPUs.
        _useStaticBuckets = ActiveProvider.StartsWith("dml", StringComparison.Ordinal);
        _outputName = ResolveOutputName(_session);
    }

    public float[] Encode(string text) => EncodeBatch([text])[0];

    public float[][] EncodeBatch(IReadOnlyList<string> texts)
    {
        if (texts.Count == 0)
        {
            return [];
        }

        var results = new float[texts.Count][];
        for (int start = 0; start < texts.Count; start += _maxBatchSize)
        {
            int count = Math.Min(_maxBatchSize, texts.Count - start);
            var slice = new string[count];
            for (int i = 0; i < count; i++)
            {
                slice[i] = texts[start + i];
            }

            float[][] sub = RunSubBatch(slice);
            for (int i = 0; i < count; i++)
            {
                results[start + i] = sub[i];
            }
        }

        return results;
    }

    public void Dispose() => _session.Dispose();

    private float[][] RunSubBatch(IReadOnlyList<string> texts)
    {
        TokenizedBatch tok = _tokenizer.EncodeBatch(texts, _maxLength, padToStaticBucket: _useStaticBuckets);
        int[] dims = [tok.BatchSize, tok.SeqLen];

        var inputs = new List<NamedOnnxValue>(3)
        {
            NamedOnnxValue.CreateFromTensor("input_ids", new DenseTensor<long>(tok.InputIds, dims)),
            NamedOnnxValue.CreateFromTensor("attention_mask", new DenseTensor<long>(tok.AttentionMask, dims))
        };

        if (_expectsTokenTypeIds)
        {
            inputs.Add(NamedOnnxValue.CreateFromTensor("token_type_ids", new DenseTensor<long>(tok.TokenTypeIds, dims)));
        }

        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs = _session.Run(inputs);
        DisposableNamedOnnxValue output = outputs.FirstOrDefault(o => o.Name == _outputName) ?? outputs.First();

        return output.ElementType switch
        {
            TensorElementType.Float => PoolAndNormalizeFloat(output.AsTensor<float>(), tok),
            TensorElementType.Float16 => PoolAndNormalizeFloat16(output.AsTensor<Float16>(), tok),
            _ => throw new NotSupportedException($"Unsupported ONNX output tensor element type: {output.ElementType}")
        };
    }

    private static float[][] PoolAndNormalizeFloat(Tensor<float> tensor, TokenizedBatch tok)
    {
        ReadOnlySpan<int> shape = tensor.Dimensions;
        if (shape.Length == 2)
        {
            int hidden = shape[1];
            var pooled = new float[tok.BatchSize][];
            for (int b = 0; b < tok.BatchSize; b++)
            {
                var vec = new float[hidden];
                for (int d = 0; d < hidden; d++)
                {
                    vec[d] = tensor[b, d];
                }

                Hash384Embedder.NormalizeInPlace(vec);
                pooled[b] = vec;
            }

            return pooled;
        }

        if (shape.Length != 3)
        {
            throw new InvalidOperationException($"Expected rank-2 or rank-3 embedding tensor, got rank {shape.Length}.");
        }

        int seqLen = shape[1];
        int dim = shape[2];
        var result = new float[tok.BatchSize][];

        for (int b = 0; b < tok.BatchSize; b++)
        {
            var vec = new float[dim];
            int validTokens = 0;
            int maskOff = b * tok.SeqLen;

            for (int t = 0; t < seqLen; t++)
            {
                if (tok.AttentionMask[maskOff + t] == 0)
                {
                    continue;
                }

                validTokens++;
                for (int d = 0; d < dim; d++)
                {
                    vec[d] += tensor[b, t, d];
                }
            }

            if (validTokens > 0)
            {
                float inv = 1.0f / validTokens;
                for (int d = 0; d < dim; d++)
                {
                    vec[d] *= inv;
                }
            }

            Hash384Embedder.NormalizeInPlace(vec);
            result[b] = vec;
        }

        return result;
    }

    private static float[][] PoolAndNormalizeFloat16(Tensor<Float16> tensor, TokenizedBatch tok)
    {
        ReadOnlySpan<int> shape = tensor.Dimensions;
        if (shape.Length == 2)
        {
            int hidden = shape[1];
            var pooled = new float[tok.BatchSize][];
            for (int b = 0; b < tok.BatchSize; b++)
            {
                var vec = new float[hidden];
                for (int d = 0; d < hidden; d++)
                {
                    vec[d] = (float)tensor[b, d];
                }

                Hash384Embedder.NormalizeInPlace(vec);
                pooled[b] = vec;
            }

            return pooled;
        }

        if (shape.Length != 3)
        {
            throw new InvalidOperationException($"Expected rank-2 or rank-3 embedding tensor, got rank {shape.Length}.");
        }

        int seqLen = shape[1];
        int dim = shape[2];
        var result = new float[tok.BatchSize][];

        for (int b = 0; b < tok.BatchSize; b++)
        {
            var vec = new float[dim];
            int validTokens = 0;
            int maskOff = b * tok.SeqLen;

            for (int t = 0; t < seqLen; t++)
            {
                if (tok.AttentionMask[maskOff + t] == 0)
                {
                    continue;
                }

                validTokens++;
                for (int d = 0; d < dim; d++)
                {
                    vec[d] += (float)tensor[b, t, d];
                }
            }

            if (validTokens > 0)
            {
                float inv = 1.0f / validTokens;
                for (int d = 0; d < dim; d++)
                {
                    vec[d] *= inv;
                }
            }

            Hash384Embedder.NormalizeInPlace(vec);
            result[b] = vec;
        }

        return result;
    }

    private static (InferenceSession Session, string Provider, string? FallbackReason) CreateSession(
        string modelPath,
        OnnxExecutionProvider requested,
        int dmlDeviceId)
    {
        string? fallbackReason = null;

#if STEPPEWATCH_DML
        string fileLower = Path.GetFileName(modelPath).ToLowerInvariant();
        bool isDynamicInt8 = fileLower.Contains("quant") || fileLower.Contains("int8");
        if (requested == OnnxExecutionProvider.DirectML || (requested == OnnxExecutionProvider.Auto && !isDynamicInt8))
        {
            try
            {
                var dmlSession = OpenWithOptimizationFallback(modelPath, optLevel =>
                {
                    var opts = new SessionOptions
                    {
                        GraphOptimizationLevel = optLevel,
                        EnableMemoryPattern = false,
                        ExecutionMode = ExecutionMode.ORT_SEQUENTIAL
                    };
                    opts.AppendExecutionProvider_DML(dmlDeviceId);
                    return opts;
                });
                return (dmlSession, $"dml_dev{dmlDeviceId}", null);
            }
            catch (Exception ex) when (requested == OnnxExecutionProvider.Auto)
            {
                fallbackReason = $"DirectML unavailable ({ex.GetType().Name}: {ex.Message}); fell back to CPU.";
            }
        }
        else if (requested == OnnxExecutionProvider.Auto && isDynamicInt8)
        {
            fallbackReason = "Auto-selected CPU for INT8 quantized model (faster & sharper than DirectML on iGPUs; pass --provider dml to override).";
        }
#else
        if (requested == OnnxExecutionProvider.DirectML)
        {
            throw new PlatformNotSupportedException(
                "This build was compiled without DirectML support. Publish/build with -p:UseDirectML=true on Windows or -r win-x64.");
        }
#endif

        var cpuSession = OpenWithOptimizationFallback(modelPath, optLevel => new SessionOptions
        {
            GraphOptimizationLevel = optLevel,
            ExecutionMode = ExecutionMode.ORT_SEQUENTIAL
        });
        return (cpuSession, "cpu", fallbackReason);
    }

    private static InferenceSession OpenWithOptimizationFallback(
        string modelPath,
        Func<GraphOptimizationLevel, SessionOptions> buildOptions)
    {
        try
        {
            using var allOpts = buildOptions(GraphOptimizationLevel.ORT_ENABLE_ALL);
            return new InferenceSession(modelPath, allOpts);
        }
        catch (OnnxRuntimeException)
        {
            // Some FP16-converted BERT graphs trip ORT's extended SimplifiedLayerNormFusion pass;
            // retry cleanly with ORT_ENABLE_BASIC (still runs full hardware kernels on DML/CPU).
            using var basicOpts = buildOptions(GraphOptimizationLevel.ORT_ENABLE_BASIC);
            return new InferenceSession(modelPath, basicOpts);
        }
    }

    private static string ResolveOutputName(InferenceSession session)
    {
        if (session.OutputMetadata.ContainsKey("last_hidden_state"))
        {
            return "last_hidden_state";
        }

        if (session.OutputMetadata.ContainsKey("sentence_embedding"))
        {
            return "sentence_embedding";
        }

        return session.OutputMetadata.Keys.First();
    }
}
