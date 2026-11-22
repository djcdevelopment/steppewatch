using System.Numerics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

namespace Steppewatch.Core;

public sealed record HardwareProfile(
    string OsDescription,
    string Architecture,
    string Framework,
    int LogicalCores,
    double TotalMemoryGib,
    bool VectorHardwareAccelerated,
    int VectorFloatCount,
    bool Avx2Supported,
    bool FmaSupported,
    bool AvxVnniSupported,
    bool Avx512VnniSupported,
    bool ArmAdvSimdSupported,
    string ActiveInt8Kernel,
    string ActiveFp32Kernel);

public static class HardwareProbe
{
    public static HardwareProfile Detect()
    {
        var gcInfo = GC.GetGCMemoryInfo();
        double totalGiB = Math.Round(gcInfo.TotalAvailableMemoryBytes / (1024.0 * 1024.0 * 1024.0), 1);

        string int8Kernel = AvxVnni.IsSupported
            ? "x86_avx_vnni_vpdpbusd"
            : Avx2.IsSupported
                ? "x86_avx2_vpmaddubsw"
                : "managed_scalar_int8";

        string fp32Kernel = Avx2.IsSupported && Fma.IsSupported
            ? "x86_avx2_fma_256"
            : Vector.IsHardwareAccelerated
                ? $"simd_vector_{Vector<float>.Count}"
                : "managed_scalar_fp32";

        return new HardwareProfile(
            OsDescription: RuntimeInformation.OSDescription.Trim(),
            Architecture: RuntimeInformation.ProcessArchitecture.ToString(),
            Framework: RuntimeInformation.FrameworkDescription.Trim(),
            LogicalCores: Environment.ProcessorCount,
            TotalMemoryGib: totalGiB,
            VectorHardwareAccelerated: Vector.IsHardwareAccelerated,
            VectorFloatCount: Vector<float>.Count,
            Avx2Supported: Avx2.IsSupported,
            FmaSupported: Fma.IsSupported,
            AvxVnniSupported: AvxVnni.IsSupported,
            Avx512VnniSupported: Avx512Vbmi.IsSupported || Avx512F.IsSupported,
            ArmAdvSimdSupported: AdvSimd.IsSupported,
            ActiveInt8Kernel: int8Kernel,
            ActiveFp32Kernel: fp32Kernel);
    }
}
