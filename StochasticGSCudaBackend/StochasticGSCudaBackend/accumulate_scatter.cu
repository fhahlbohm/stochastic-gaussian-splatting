#include <cstdint>
#include <cooperative_groups.h>
#include "helper_math.h"
#include "gs_utils.h"
#include "math_utils.h"
#include "render_utils.h"
#include "scatter_utils.cuh"
#include "accumulate_scatter.h"

namespace cg = cooperative_groups;

namespace scatter {

// config
inline constexpr uint32_t block_size_accumulate = 256;


// kernels
__global__ void accumulate_st_cu(
    const float3* __restrict__ means,
    const float3* __restrict__ scales,
    const float4* __restrict__ rotations,
    const float* __restrict__ opacities,
    const float4* __restrict__ w2c,
    unsigned long long* __restrict__ framebuffer,
    const uint32_t n_gaussians,
    const uint32_t frame_idx,
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
    const uint8_t* __restrict__ blue_noise_slice,
    const uint32_t blue_noise_rotation)
{
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    const uint32_t thread_idx = cg::this_grid().thread_rank();

    bool active = true;
    uint32_t gaussian_idx = thread_idx;
    if (gaussian_idx >= n_gaussians) {
        active = false;
        gaussian_idx = n_gaussians - 1;
    }

    // load, transform, and project Gaussian parameters with early exits
    float depth, opacity, determinant, power_threshold;
    float2 mean2d;
    float3 conic, cov2d;
    float4 screen_bounds;
    if (!project_gaussian(
        warp, active, gaussian_idx, means, scales, rotations, opacities, w2c,
        width, height, focal_x, focal_y, center_x, center_y, near_plane, far_plane, proper_antialiasing, supersampling_factor,
        depth, opacity, determinant, power_threshold, mean2d, conic, cov2d, screen_bounds)) return;

    const float2 mean2d_shifted = mean2d - 0.5f;
    const uint32_t width_int = __float2uint_rn(width);
    const uint32_t x_min = __float2uint_rd(fmaxf(screen_bounds.x, 0.0f));
    const uint32_t x_max = __float2uint_ru(fminf(screen_bounds.y, width));
    const uint32_t y_min = __float2uint_rd(fmaxf(screen_bounds.z, 0.0f));
    const uint32_t y_max = __float2uint_ru(fminf(screen_bounds.w, height));
    const uint32_t aabb_width = x_max - x_min;
    const uint32_t aabb_height = y_max - y_min;
    const uint32_t fragment_count = aabb_width * aabb_height;
    const uint32_t depth_key = __float_as_uint(depth);
    const unsigned long long fragment = (static_cast<unsigned long long>(depth_key) << 32) | static_cast<unsigned long long>(gaussian_idx);

    scatter_fragments(
        warp, active, gaussian_idx, fragment, x_min, y_min, aabb_width, fragment_count, mean2d_shifted, conic, opacity,
        framebuffer, width_int, frame_idx, blue_noise_slice, blue_noise_rotation);
}

__global__ void accumulate_gps_cu(
    const float3* __restrict__ means,
    const float3* __restrict__ scales,
    const float4* __restrict__ rotations,
    const float* __restrict__ opacities,
    const float4* __restrict__ w2c,
    unsigned long long* __restrict__ framebuffer,
    const uint32_t n_gaussians,
    const uint32_t frame_idx,
    const float width,
    const float height,
    const float focal_x,
    const float focal_y,
    const float center_x,
    const float center_y,
    const float near_plane,
    const float far_plane,
    const bool proper_antialiasing,
    const float supersampling_factor)
{
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    const uint32_t thread_idx = cg::this_grid().thread_rank();

    bool active = true;
    uint32_t gaussian_idx = thread_idx;
    if (gaussian_idx >= n_gaussians) {
        active = false;
        gaussian_idx = n_gaussians - 1;
    }

    // load, transform, and project Gaussian parameters with early exits
    float depth, opacity, determinant, power_threshold;
    float2 mean2d;
    float3 conic, cov2d;
    float4 screen_bounds;
    if (!project_gaussian(
        warp, active, gaussian_idx, means, scales, rotations, opacities, w2c,
        width, height, focal_x, focal_y, center_x, center_y, near_plane, far_plane, proper_antialiasing, supersampling_factor,
        depth, opacity, determinant, power_threshold, mean2d, conic, cov2d, screen_bounds)) return;

    // compute importance and number of points to splat
    const float opacity_dilog = dilog(opacity);
    const float importance = two_pi * sqrtf(determinant) * opacity_dilog;
    uint2 seed = make_seed(gaussian_idx, frame_idx);
    const uint32_t true_fragment_count = poisson(seed, importance);
    if (true_fragment_count == 0) active = false;

    // heuristic from reference implementation to cap the maximum samples per Gaussian for performance reasons
    const uint32_t max_fragments = __float2uint_rn(width * height / (2.0f * supersampling_factor * supersampling_factor));
    const uint32_t fragment_count = min(true_fragment_count, max_fragments);

    // early exit if whole warp is inactive
    if (warp.ballot(active) == 0) return;

    const uint32_t width_int = __float2uint_rn(width);
    const float c0 = sqrtf(cov2d.x);
    const float c1 = cov2d.y / c0;
    const float c2 = sqrtf(cov2d.z - c1 * c1);
    const float3 L = make_float3(c0, c1, c2);
    const float opacity_rcp = 1.0f / fmaxf(opacity, 1.0e-37f);
    const uint32_t depth_key = __float_as_uint(depth);
    const unsigned long long fragment = (static_cast<unsigned long long>(depth_key) << 32) | static_cast<unsigned long long>(gaussian_idx);

    scatter_points(
        warp, active, fragment, fragment_count, mean2d, conic, L, power_threshold, opacity_rcp, opacity_dilog, seed,
        framebuffer, width, height, width_int);
}

}


// host function
void accumulate_scatter(
    const float3* means,
    const float3* scales,
    const float4* rotations,
    const float* opacities,
    const float4* w2c,
    unsigned long long* framebuffer,
    const uint32_t n_gaussians,
    const uint32_t frame_idx,
    const CameraSettings& camera,
    const bool use_gps,
    const uint8_t* blue_noise_slice,
    const uint32_t blue_noise_rotation,
    const cudaStream_t stream)
{
    using namespace scatter;
    const float supersampling_factor = static_cast<float>(camera.supersampling_factor);
    const uint32_t grid_size_accumulate = (n_gaussians + block_size_accumulate - 1) / block_size_accumulate;
    if (use_gps) {
        accumulate_gps_cu<<<grid_size_accumulate, block_size_accumulate, 0, stream>>>(
            means,
            scales,
            rotations,
            opacities,
            w2c,
            framebuffer,
            n_gaussians,
            frame_idx,
            static_cast<float>(camera.width * camera.supersampling_factor),
            static_cast<float>(camera.height * camera.supersampling_factor),
            camera.focal_x * supersampling_factor,
            camera.focal_y * supersampling_factor,
            camera.center_x * supersampling_factor,
            camera.center_y * supersampling_factor,
            camera.near_plane,
            camera.far_plane,
            camera.proper_antialiasing,
            supersampling_factor
        );
        check_cuda("scatter::accumulate_gps_cu");
    }
    else {
        accumulate_st_cu<<<grid_size_accumulate, block_size_accumulate, 0, stream>>>(
            means,
            scales,
            rotations,
            opacities,
            w2c,
            framebuffer,
            n_gaussians,
            frame_idx,
            static_cast<float>(camera.width * camera.supersampling_factor),
            static_cast<float>(camera.height * camera.supersampling_factor),
            camera.focal_x * supersampling_factor,
            camera.focal_y * supersampling_factor,
            camera.center_x * supersampling_factor,
            camera.center_y * supersampling_factor,
            camera.near_plane,
            camera.far_plane,
            camera.proper_antialiasing,
            supersampling_factor,
            blue_noise_slice,
            blue_noise_rotation
        );
        check_cuda("scatter::accumulate_st_cu");
    }
}
