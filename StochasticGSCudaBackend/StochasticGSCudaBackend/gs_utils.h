// Extracted and modified from https://github.com/nerficg-project/faster-gaussian-splatting

#pragma once

#include <cstdint>
#include <cooperative_groups.h>
#include "helper_math.h"

namespace cg = cooperative_groups;


// Gaussian splatting config inherited from Faster-GS
inline constexpr float dilation = 0.3f;
inline constexpr float dilation_proper_antialiasing = 0.1f;
inline constexpr float min_cov2d_determinant = 1e-6f;
inline constexpr float min_alpha_threshold_rcp = 255.0f;
inline constexpr float min_alpha_threshold = 1.0f / min_alpha_threshold_rcp; // 0.00392156862
inline constexpr float max_power_threshold = 5.54126354516f; // ln(min_alpha_threshold_rcp)

// folds -0.5 and log2(e) into the conic so that alpha evaluates the Gaussian with a single exp2
__device__ inline float3 scale_conic_for_exp2(const float3& conic) {
    constexpr float log2e = 1.44269504089f;
    return make_float3(-0.5f * log2e * conic.x, -log2e * conic.y, -0.5f * log2e * conic.z);
}


// load, transform, and project Gaussian parameters with early exits
struct mat3x3 {
    float m11, m12, m13;
    float m21, m22, m23;
    float m31, m32, m33;
};

struct __align__(8) mat3x3_triu {
    float m11, m12, m13, m22, m23, m33;
};

__device__ inline bool project_gaussian(
    const cg::thread_block_tile<32>& warp,
    bool& active,
    const uint32_t gaussian_idx,
    const float3* __restrict__ means,
    const float3* __restrict__ scales,
    const float4* __restrict__ rotations,
    const float* __restrict__ opacities,
    const float4* __restrict__ w2c,
    const float width,
    const float height,
    const float focal_x,
    const float focal_y,
    const float center_x,
    const float center_y,
    const float near_plane,
    const float far_plane,
    const bool proper_antialiasing,
    const float supersampling_factor,
    float& depth,
    float& opacity,
    float& determinant,
    float& power_threshold,
    float2& mean2d,
    float3& conic,
    float3& cov2d,
    float4& screen_bounds)
{
    // load 3d mean
    const float3 mean3d = means[gaussian_idx];

    // z culling
    const float4 w2c_r3 = w2c[2];
    depth = w2c_r3.x * mean3d.x + w2c_r3.y * mean3d.y + w2c_r3.z * mean3d.z + w2c_r3.w;
    if (depth < near_plane || depth > far_plane) active = false;

    // early exit if whole warp is inactive
    if (warp.ballot(active) == 0) return false;

    // load opacity
    const float raw_opacity = opacities[gaussian_idx];
    opacity = 1.0f / (1.0f + expf(-raw_opacity));
    if (opacity < min_alpha_threshold) active = false;

    // compute 3d covariance from scale and rotation
    const float3 raw_scale = scales[gaussian_idx];
    const float3 variance = expf(2.0f * raw_scale);
    const float4 raw_rotation = rotations[gaussian_idx];
    auto [r, i, j, k] = raw_rotation;
    const float ii = i * i, jj = j * j, kk = k * k;
    const float ij = i * j, ik = i * k, jk = j * k;
    const float ri = r * i, rj = r * j, rk = r * k;
    const float norm_sq = r * r + ii + jj + kk;
    if (norm_sq < 1e-8f) active = false;
    const float norm_sq_rcp = 1.0f / norm_sq;
    const mat3x3 R = {
        1.0f - 2.0f * (jj + kk) * norm_sq_rcp, 2.0f * (ij - rk) * norm_sq_rcp, 2.0f * (ik + rj) * norm_sq_rcp,
        2.0f * (ij + rk) * norm_sq_rcp, 1.0f - 2.0f * (ii + kk) * norm_sq_rcp, 2.0f * (jk - ri) * norm_sq_rcp,
        2.0f * (ik - rj) * norm_sq_rcp, 2.0f * (jk + ri) * norm_sq_rcp, 1.0f - 2.0f * (ii + jj) * norm_sq_rcp
    };
    const mat3x3 RSS = {
        R.m11 * variance.x, R.m12 * variance.y, R.m13 * variance.z,
        R.m21 * variance.x, R.m22 * variance.y, R.m23 * variance.z,
        R.m31 * variance.x, R.m32 * variance.y, R.m33 * variance.z
    };
    const mat3x3_triu cov3d {
        RSS.m11 * R.m11 + RSS.m12 * R.m12 + RSS.m13 * R.m13,
        RSS.m11 * R.m21 + RSS.m12 * R.m22 + RSS.m13 * R.m23,
        RSS.m11 * R.m31 + RSS.m12 * R.m32 + RSS.m13 * R.m33,
        RSS.m21 * R.m21 + RSS.m22 * R.m22 + RSS.m23 * R.m23,
        RSS.m21 * R.m31 + RSS.m22 * R.m32 + RSS.m23 * R.m33,
        RSS.m31 * R.m31 + RSS.m32 * R.m32 + RSS.m33 * R.m33,
    };

    // compute 2d mean in normalized image coordinates
    const float4 w2c_r1 = w2c[0];
    const float x = (w2c_r1.x * mean3d.x + w2c_r1.y * mean3d.y + w2c_r1.z * mean3d.z + w2c_r1.w) / depth;
    const float4 w2c_r2 = w2c[1];
    const float y = (w2c_r2.x * mean3d.x + w2c_r2.y * mean3d.y + w2c_r2.z * mean3d.z + w2c_r2.w) / depth;

    // ewa splatting
    const float clip_left = (-0.15f * width - center_x) / focal_x;
    const float clip_right = (1.15f * width - center_x) / focal_x;
    const float clip_top = (-0.15f * height - center_y) / focal_y;
    const float clip_bottom = (1.15f * height - center_y) / focal_y;
    const float x_clipped = clamp(x, clip_left, clip_right);
    const float y_clipped = clamp(y, clip_top, clip_bottom);
    const float j11 = focal_x / depth;
    const float j13 = -j11 * x_clipped;
    const float j22 = focal_y / depth;
    const float j23 = -j22 * y_clipped;
    const float3 jw_r1 = make_float3(
        j11 * w2c_r1.x + j13 * w2c_r3.x,
        j11 * w2c_r1.y + j13 * w2c_r3.y,
        j11 * w2c_r1.z + j13 * w2c_r3.z
    );
    const float3 jw_r2 = make_float3(
        j22 * w2c_r2.x + j23 * w2c_r3.x,
        j22 * w2c_r2.y + j23 * w2c_r3.y,
        j22 * w2c_r2.z + j23 * w2c_r3.z
    );
    const float3 jwc_r1 = make_float3(
        jw_r1.x * cov3d.m11 + jw_r1.y * cov3d.m12 + jw_r1.z * cov3d.m13,
        jw_r1.x * cov3d.m12 + jw_r1.y * cov3d.m22 + jw_r1.z * cov3d.m23,
        jw_r1.x * cov3d.m13 + jw_r1.y * cov3d.m23 + jw_r1.z * cov3d.m33
    );
    const float3 jwc_r2 = make_float3(
        jw_r2.x * cov3d.m11 + jw_r2.y * cov3d.m12 + jw_r2.z * cov3d.m13,
        jw_r2.x * cov3d.m12 + jw_r2.y * cov3d.m22 + jw_r2.z * cov3d.m23,
        jw_r2.x * cov3d.m13 + jw_r2.y * cov3d.m23 + jw_r2.z * cov3d.m33
    );
    cov2d = make_float3(
        dot(jwc_r1, jw_r1),
        dot(jwc_r1, jw_r2),
        dot(jwc_r2, jw_r2)
    );
    const float determinant_raw = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    const float kernel_size = (proper_antialiasing ? dilation_proper_antialiasing : dilation) * supersampling_factor * supersampling_factor;
    cov2d.x += kernel_size;
    cov2d.z += kernel_size;
    determinant = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    if (determinant < min_cov2d_determinant) active = false;
    conic = make_float3(
        cov2d.z / determinant,
        -cov2d.y / determinant,
        cov2d.x / determinant
    );
    if (proper_antialiasing) {
        opacity *= sqrtf(fmaxf(determinant_raw / determinant, 0.0f));
        if (opacity < min_alpha_threshold) active = false;
    }

    // 2d mean in screen space
    mean2d = make_float2(
        x * focal_x + center_x,
        y * focal_y + center_y
    );

    // compute bounds
    power_threshold = logf(opacity * min_alpha_threshold_rcp);
    const float cutoff_factor = 2.0f * power_threshold;
    const float extent_x = sqrtf(cov2d.x * cutoff_factor);
    const float extent_y = sqrtf(cov2d.z * cutoff_factor);
    screen_bounds = make_float4(
        mean2d.x - extent_x, // x_min
        mean2d.x + extent_x, // x_max
        mean2d.y - extent_y, // y_min
        mean2d.y + extent_y  // y_max
    );
    if (screen_bounds.x >= width || screen_bounds.y < 0.0f || screen_bounds.z >= height || screen_bounds.w < 0.0f) active = false;

    // early exit if whole warp is inactive
    if (warp.ballot(active) == 0) return false;

    return true;
}


// spherical harmonics evaluation
inline constexpr float C0 = 0.28209479177387814f;
inline constexpr float C1 = 0.48860251190291987f;
inline constexpr float C2a = 1.0925484305920792f;
inline constexpr float C2b = 0.94617469575755997f;
inline constexpr float C2c = 0.31539156525251999f;
inline constexpr float C2d = 0.54627421529603959f;
inline constexpr float C3a = 0.59004358992664352f;
inline constexpr float C3b = 1.7701307697799304f;
inline constexpr float C3c = 2.8906114426405538f;
inline constexpr float C3d = 0.45704579946446572f;
inline constexpr float C3e = 2.2852289973223288f;
inline constexpr float C3f = 1.865881662950577f;
inline constexpr float C3g = 1.1195289977703462f;
inline constexpr float C3h = 1.4453057213202769f;

__device__ inline float3 eval_sh(const float3* coefficients, const float3 direction, const uint32_t total_sh_bases) {
    auto [x, y, z] = direction;
    // degree 0
    float3 color = 0.5f + C0 * coefficients[0];
    if (total_sh_bases > 1) {
        // degree 1
        color = color - C1 * y * coefficients[1]
                      + C1 * z * coefficients[2]
                      - C1 * x * coefficients[3];
        if (total_sh_bases > 4) {
            const float xx = x * x, yy = y * y, zz = z * z;
            const float xy = x * y, xz = x * z, yz = y * z;
            // degree 2
            color = color + C2a * xy * coefficients[4]
                          - C2a * yz * coefficients[5]
                          + (C2b * zz - C2c) * coefficients[6]
                          - C2a * xz * coefficients[7]
                          + C2d * (xx - yy) * coefficients[8];
            if (total_sh_bases > 9) {
                // degree 3
                color = color + y * (C3a * yy - C3b * xx) * coefficients[9]
                              + C3c * xy * z * coefficients[10]
                              + y * (C3d - C3e * zz) * coefficients[11]
                              + z * (C3f * zz - C3g) * coefficients[12]
                              + x * (C3d - C3e * zz) * coefficients[13]
                              + C3h * z * (xx - yy) * coefficients[14]
                              + x * (C3b * yy - C3a * xx) * coefficients[15];
            }
        }
    }
    return fmaxf(color, 0.0f);
}
