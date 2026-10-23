#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include "helper_math.h"
#include "render_utils.h"

// stochastic transparency: scatter for small Gaussians, tile-based gather for large ones (same result as scatter)
void accumulate_hybrid(
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
