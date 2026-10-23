#pragma once

#include <cstdint>
#include <cooperative_groups.h>
#include "helper_math.h"
#include "gs_utils.h"
#include "math_utils.h"

namespace cg = cooperative_groups;

inline constexpr uint32_t n_sequential_threshold = 4;

// stochastic transparency: small Gaussians per thread, larger ones cooperatively by the whole warp
__device__ inline void scatter_fragments(
    const cg::thread_block_tile<32>& warp,
    const bool active,
    const uint32_t gaussian_idx,
    const unsigned long long fragment,
    const uint32_t x_min,
    const uint32_t y_min,
    const uint32_t aabb_width,
    const uint32_t fragment_count,
    const float2 mean2d_shifted,
    const float3 conic,
    const float opacity,
    unsigned long long* __restrict__ framebuffer,
    const uint32_t width_int,
    const uint32_t frame_idx,
    const uint8_t* __restrict__ blue_noise_slice,
    const uint32_t blue_noise_rotation)
{
    const bool use_blue_noise = blue_noise_slice != nullptr;
    for (uint32_t fragment_idx = 0; active && fragment_idx < fragment_count && fragment_idx < n_sequential_threshold; fragment_idx++) {
        const uint32_t pixel_x = x_min + (fragment_idx % aabb_width);
        const uint32_t pixel_y = y_min + (fragment_idx / aabb_width);
        const float delta_x = static_cast<float>(pixel_x) - mean2d_shifted.x;
        const float delta_y = static_cast<float>(pixel_y) - mean2d_shifted.y;
        const float power = 0.5f * (conic.x * delta_x * delta_x + conic.z * delta_y * delta_y) + conic.y * delta_x * delta_y;
        const float alpha = opacity * expf(fminf(-power, 0.0f));
        if (alpha < min_alpha_threshold) continue;
        const float sample = use_blue_noise ? blue_noise(blue_noise_slice, blue_noise_rotation, gaussian_idx, pixel_x, pixel_y) : rand(frame_idx, gaussian_idx, pixel_x, pixel_y);
        if (sample >= alpha) continue;

        const uint32_t pixel_idx = pixel_y * width_int + pixel_x;
        atomicMin(&framebuffer[pixel_idx], fragment);
    }

    const bool compute_cooperatively = active && fragment_count > n_sequential_threshold;
    const uint32_t remaining_threads = warp.ballot(compute_cooperatively);
    if (remaining_threads == 0) return;

    const uint32_t n_remaining_threads = __popc(remaining_threads);
    for (uint32_t n = 0; n < n_remaining_threads && n < 32; n++) {
        const uint32_t current_lane = __fns(remaining_threads, 0, n + 1);

        const uint32_t x_min_coop = warp.shfl(x_min, current_lane);
        const uint32_t y_min_coop = warp.shfl(y_min, current_lane);
        const uint32_t aabb_width_coop = warp.shfl(aabb_width, current_lane);
        const uint32_t fragment_count_coop = warp.shfl(fragment_count, current_lane);
        const float2 mean2d_shifted_coop = make_float2(
            warp.shfl(mean2d_shifted.x, current_lane),
            warp.shfl(mean2d_shifted.y, current_lane)
        );
        const float3 conic_coop = make_float3(
            warp.shfl(conic.x, current_lane),
            warp.shfl(conic.y, current_lane),
            warp.shfl(conic.z, current_lane)
        );
        const float opacity_coop = warp.shfl(opacity, current_lane);
        const unsigned long long fragment_coop = warp.shfl(fragment, current_lane);
        const uint32_t gaussian_idx_coop = warp.shfl(gaussian_idx, current_lane);

        const uint32_t lane_idx = warp.thread_rank();
        const uint32_t remaining_fragment_count = fragment_count_coop - n_sequential_threshold;
        const uint32_t n_iterations = (remaining_fragment_count + 32 - 1) / 32;
        for (uint32_t i = 0; i < n_iterations; i++) {
            const uint32_t fragment_idx = i * 32 + lane_idx + n_sequential_threshold;
            if (fragment_idx >= fragment_count_coop) continue;

            const uint32_t pixel_x = x_min_coop + (fragment_idx % aabb_width_coop);
            const uint32_t pixel_y = y_min_coop + (fragment_idx / aabb_width_coop);
            const float delta_x = static_cast<float>(pixel_x) - mean2d_shifted_coop.x;
            const float delta_y = static_cast<float>(pixel_y) - mean2d_shifted_coop.y;
            const float power = 0.5f * (conic_coop.x * delta_x * delta_x + conic_coop.z * delta_y * delta_y) + conic_coop.y * delta_x * delta_y;
            const float alpha = opacity_coop * expf(fminf(-power, 0.0f));
            if (alpha < min_alpha_threshold) continue;
            const float sample = use_blue_noise ? blue_noise(blue_noise_slice, blue_noise_rotation, gaussian_idx_coop, pixel_x, pixel_y) : rand(frame_idx, gaussian_idx_coop, pixel_x, pixel_y);
            if (sample >= alpha) continue;

            const uint32_t pixel_idx = pixel_y * width_int + pixel_x;
            atomicMin(&framebuffer[pixel_idx], fragment_coop);
        }
    }
}

// Gaussian point splatting with the same work distribution as scatter_fragments
__device__ inline void scatter_points(
    const cg::thread_block_tile<32>& warp,
    const bool active,
    const unsigned long long fragment,
    const uint32_t fragment_count,
    const float2 mean2d,
    const float3 conic,
    const float3 L,
    const float power_threshold,
    const float opacity_rcp,
    const float opacity_dilog,
    uint2& seed,
    unsigned long long* __restrict__ framebuffer,
    const float width,
    const float height,
    const uint32_t width_int)
{
    for (uint32_t fragment_idx = 0; active && fragment_idx < fragment_count && fragment_idx < n_sequential_threshold; fragment_idx++) {
        const float2 u = pcg2d(seed);
        const float2 xy = corrected_box_muller(u.x, u.y, opacity_rcp, opacity_dilog);
        const float2 delta = make_float2(L.x * xy.x, L.y * xy.x + L.z * xy.y);
        const float2 sample = mean2d + delta;
        if (sample.x >= width || sample.x < 0.0f || sample.y >= height || sample.y < 0.0f) continue;

        const float2 delta_center = make_float2(truncf(sample.x), truncf(sample.y)) + 0.5f - mean2d;
        const float power = 0.5f * (conic.x * delta_center.x * delta_center.x + conic.z * delta_center.y * delta_center.y) + conic.y * delta_center.x * delta_center.y;
        if (power > power_threshold) continue;

        const uint32_t pixel_x = __float2uint_rd(sample.x);
        const uint32_t pixel_y = __float2uint_rd(sample.y);
        const uint32_t pixel_idx = pixel_y * width_int + pixel_x;
        atomicMin(&framebuffer[pixel_idx], fragment);
    }

    const bool compute_cooperatively = active && fragment_count > n_sequential_threshold;
    const uint32_t remaining_threads = warp.ballot(compute_cooperatively);
    if (remaining_threads == 0) return;

    const uint32_t n_remaining_threads = __popc(remaining_threads);
    for (uint32_t n = 0; n < n_remaining_threads && n < 32; n++) {
        const uint32_t current_lane = __fns(remaining_threads, 0, n + 1);

        const uint32_t fragment_count_coop = warp.shfl(fragment_count, current_lane);
        const float opacity_rcp_coop = warp.shfl(opacity_rcp, current_lane);
        const float opacity_dilog_coop = warp.shfl(opacity_dilog, current_lane);
        const float2 mean2d_coop = make_float2(
            warp.shfl(mean2d.x, current_lane),
            warp.shfl(mean2d.y, current_lane)
        );
        const float3 conic_coop = make_float3(
            warp.shfl(conic.x, current_lane),
            warp.shfl(conic.y, current_lane),
            warp.shfl(conic.z, current_lane)
        );
        const float3 L_coop = make_float3(
            warp.shfl(L.x, current_lane),
            warp.shfl(L.y, current_lane),
            warp.shfl(L.z, current_lane)
        );
        const float power_threshold_coop = warp.shfl(power_threshold, current_lane);
        const unsigned long long fragment_coop = warp.shfl(fragment, current_lane);

        const uint32_t lane_idx = warp.thread_rank();
        const uint32_t remaining_fragment_count = fragment_count_coop - n_sequential_threshold;
        const uint32_t n_iterations = (remaining_fragment_count + 32 - 1) / 32;
        for (uint32_t i = 0; i < n_iterations; i++) {
            const uint32_t fragment_idx = i * 32 + lane_idx + n_sequential_threshold;
            if (fragment_idx >= fragment_count_coop) continue;

            const float2 u = pcg2d(seed);
            const float2 xy = corrected_box_muller(u.x, u.y, opacity_rcp_coop, opacity_dilog_coop);
            const float2 delta = make_float2(L_coop.x * xy.x, L_coop.y * xy.x + L_coop.z * xy.y);
            const float2 sample = mean2d_coop + delta;
            if (sample.x >= width || sample.x < 0.0f || sample.y >= height || sample.y < 0.0f) continue;

            const float2 delta_center = make_float2(truncf(sample.x), truncf(sample.y)) + 0.5f - mean2d_coop;
            const float power = 0.5f * (conic_coop.x * delta_center.x * delta_center.x + conic_coop.z * delta_center.y * delta_center.y) + conic_coop.y * delta_center.x * delta_center.y;
            if (power > power_threshold_coop) continue;

            const uint32_t pixel_x = __float2uint_rd(sample.x);
            const uint32_t pixel_y = __float2uint_rd(sample.y);
            const uint32_t pixel_idx = pixel_y * width_int + pixel_x;
            atomicMin(&framebuffer[pixel_idx], fragment_coop);
        }
    }
}
