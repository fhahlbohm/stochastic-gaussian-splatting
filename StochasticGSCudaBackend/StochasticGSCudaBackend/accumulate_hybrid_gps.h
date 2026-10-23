#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include "helper_math.h"
#include "render_utils.h"

// Gaussian point splatting: scatter for small Gaussians, point chunks of large ones processed by one block each
void accumulate_hybrid_gps(
    const float3* means,
    const float3* scales,
    const float4* rotations,
    const float* opacities,
    const float4* w2c,
    unsigned long long* framebuffer,
    const uint32_t n_gaussians,
    const uint32_t frame_idx,
    const CameraSettings& camera,
    const cudaStream_t stream);
