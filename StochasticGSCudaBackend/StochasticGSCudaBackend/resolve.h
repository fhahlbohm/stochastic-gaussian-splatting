#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include "helper_math.h"
#include "render_utils.h"

// evaluates the closest accepted Gaussian of each (sub)pixel and resolves color, depth, and alpha
void sample_and_resolve(
    const float3* means,
    const float3* sh_coefficients,
    const float3* cam_position,
    const unsigned long long* framebuffer,
    float3* image,
    float* depth,
    float* alpha,
    const uint32_t total_sh_bases,
    const CameraSettings& camera,
    const cudaStream_t stream);
