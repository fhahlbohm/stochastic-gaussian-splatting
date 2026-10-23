#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include "helper_math.h"
#include "render_utils.h"

// k-buffer entries per (sub)pixel
inline constexpr uint32_t k_buffer_size = 4;

enum class TemporalMode : int {
    Reset = 0,
    Static = 1,
    Reproject = 2,
};

// evaluates the sampled Gaussians and resolves the image, optionally with spatio-temporal reuse
void sample_and_resolve(
    const float3* means,
    const float3* scales,
    const float4* rotations,
    const float* opacities,
    const float3* sh_coefficients,
    const float4* w2c,
    const float3* cam_position,
    const unsigned long long* framebuffer,
    float4* samples,
    float4* reuse_weights,
    uint2* k_buffer_primitives,
    float4* k_buffer_occlusion,
    const float3* history_image,
    const float* history_depth,
    const float* history_alpha,
    const float* history_weight_mass,
    const float4* view_to_previous_view,
    float3* image,
    float* depth,
    float* alpha,
    float* weight_mass,
    const uint32_t total_sh_bases,
    const CameraSettings& camera,
    const bool spatial_reuse,
    const TemporalMode temporal_mode,
    const cudaStream_t stream);
