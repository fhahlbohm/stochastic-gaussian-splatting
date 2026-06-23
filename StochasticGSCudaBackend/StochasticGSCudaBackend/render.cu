#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
#include <torch/extension.h>
#include "helper_math.h"
#include "gs_utils.h"
#include "math_utils.h"

namespace cg = cooperative_groups;


// config
inline constexpr bool debug = false;
inline constexpr uint32_t block_size_accumulate = 256;
inline constexpr uint32_t block_size_resolve = 256;
inline constexpr uint32_t n_sequential_threshold = 4;


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

    for (uint32_t fragment_idx = 0; active && fragment_idx < fragment_count && fragment_idx < n_sequential_threshold; fragment_idx++) {
        const uint32_t pixel_x = x_min + (fragment_idx % aabb_width);
        const uint32_t pixel_y = y_min + (fragment_idx / aabb_width);
        const float delta_x = static_cast<float>(pixel_x) - mean2d_shifted.x;
        const float delta_y = static_cast<float>(pixel_y) - mean2d_shifted.y;
        const float power = 0.5f * (conic.x * delta_x * delta_x + conic.z * delta_y * delta_y) + conic.y * delta_x * delta_y;
        const float alpha = opacity * expf(fminf(-power, 0.0f));
        if (alpha < min_alpha_threshold) continue;
        const float sample = rand(frame_idx, gaussian_idx, pixel_x, pixel_y);
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

        const uint32_t x_min_coop =  warp.shfl(x_min, current_lane);
        const uint32_t y_min_coop =  warp.shfl(y_min, current_lane);
        const uint32_t aabb_width_coop =  warp.shfl(aabb_width, current_lane);
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
            const float sample = rand(frame_idx, gaussian_idx_coop, pixel_x, pixel_y);
            if (sample >= alpha) continue;

            const uint32_t pixel_idx = pixel_y * width_int + pixel_x;
            atomicMin(&framebuffer[pixel_idx], fragment_coop);
        }
    }
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

__global__ void resolve_cu(
    const float3* __restrict__ means,
    const float3* __restrict__ sh_coefficients,
    const float3* __restrict__ cam_position,
    const unsigned long long* __restrict__ framebuffer,
    float3* __restrict__ image,
    float* __restrict__ depth_map,
    float* __restrict__ alpha_map,
    const uint32_t n_pixels,
    const uint32_t width,
    const uint32_t total_sh_bases,
    const uint32_t supersampling_factor)
{
    const uint32_t pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel_idx >= n_pixels) return;

    const uint32_t framebuffer_base_x = (pixel_idx % width) * supersampling_factor;
    const uint32_t framebuffer_base_y = (pixel_idx / width) * supersampling_factor;
    const uint32_t framebuffer_width = width * supersampling_factor;
    float depth_out = 0.0f;
    float3 color_out = make_float3(0.0f);
    uint32_t valid_count = 0;
    for (uint32_t offset_y = 0; offset_y < supersampling_factor; offset_y++) {
        for (uint32_t offset_x = 0; offset_x < supersampling_factor; offset_x++) {
            // load info from framebuffer
            const uint32_t framebuffer_idx = (framebuffer_base_y + offset_y) * framebuffer_width + (framebuffer_base_x + offset_x);
            const unsigned long long data = framebuffer[framebuffer_idx];
            const uint32_t depth = static_cast<uint32_t>(data >> 32);
            // add contribution if fragment is valid
            if (depth != 0xffffffffu) {
                depth_out += __uint_as_float(depth);
                valid_count++;
                // sh evaluation
                const uint32_t gaussian_idx = static_cast<uint32_t>(data & 0xffffffffu);
                const float3 color = eval_sh(
                    sh_coefficients + gaussian_idx * total_sh_bases,
                    normalize(means[gaussian_idx] - cam_position[0]),
                    total_sh_bases);
                color_out += color;
            }
        }
    }

    // color is coverage-weighted over all subpixels, depth is averaged over hit subpixels only, alpha is the subpixel coverage
    const float normalization_factor = 1.0f / static_cast<float>(supersampling_factor * supersampling_factor);
    depth_map[pixel_idx] = valid_count > 0 ? depth_out / static_cast<float>(valid_count) : 0.0f;
    alpha_map[pixel_idx] = static_cast<float>(valid_count) * normalization_factor;
    image[pixel_idx] = color_out * normalization_factor;
}


// host function
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
render(
    const torch::Tensor& means,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const torch::Tensor& opacities,
    const torch::Tensor& sh_coefficients,
    const torch::Tensor& w2c,
    const torch::Tensor& cam_position,
    const uint32_t width,
    const uint32_t height,
    const float focal_x,
    const float focal_y,
    const float center_x,
    const float center_y,
    const float near_plane,
    const float far_plane,
    const bool proper_antialiasing,
    const int supersampling_factor_raw,
    const bool use_gps)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    const uint32_t supersampling_factor = max(1, supersampling_factor_raw);
    const float supersampling_factor_float = static_cast<float>(supersampling_factor);
    const uint32_t n_gaussians = means.size(0);
    const uint32_t n_pixels = width * height;
    const uint32_t framebuffer_n_pixels = n_pixels * supersampling_factor * supersampling_factor;
    static uint32_t counter = 0;
    const uint32_t frame_idx = counter++;

    torch::Tensor image = torch::empty({height, width, 3}, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
    torch::Tensor depth = torch::empty({height, width, 1}, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
    torch::Tensor alpha = torch::empty({height, width, 1}, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
    torch::Tensor framebuffer = torch::empty({framebuffer_n_pixels}, torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA));
    unsigned long long* framebuffer_ptr = reinterpret_cast<unsigned long long*>(framebuffer.data_ptr<int64_t>());
    cudaMemsetAsync(framebuffer_ptr, 0xff, sizeof(unsigned long long) * framebuffer_n_pixels, stream);

    const float3* means_ptr = reinterpret_cast<float3*>(means.contiguous().data_ptr<float>());
    const float3* scales_ptr = reinterpret_cast<float3*>(scales.contiguous().data_ptr<float>());
    const float4* rotations_ptr = reinterpret_cast<float4*>(rotations.contiguous().data_ptr<float>());
    const float* opacities_ptr = opacities.contiguous().data_ptr<float>();
    const float3* sh_coefficients_ptr = reinterpret_cast<float3*>(sh_coefficients.contiguous().data_ptr<float>());
    const float4* w2c_ptr = reinterpret_cast<float4*>(w2c.contiguous().data_ptr<float>());
    const float3* cam_position_ptr = reinterpret_cast<float3*>(cam_position.contiguous().data_ptr<float>());

    const uint32_t grid_size_accumulate = (n_gaussians + block_size_accumulate - 1) / block_size_accumulate;
    auto kernel = use_gps ? accumulate_gps_cu : accumulate_st_cu;
    kernel<<<grid_size_accumulate, block_size_accumulate, 0, stream>>>(
        means_ptr,
        scales_ptr,
        rotations_ptr,
        opacities_ptr,
        w2c_ptr,
        framebuffer_ptr,
        n_gaussians,
        frame_idx,
        static_cast<float>(width * supersampling_factor),
        static_cast<float>(height * supersampling_factor),
        focal_x * supersampling_factor_float,
        focal_y * supersampling_factor_float,
        center_x * supersampling_factor_float,
        center_y * supersampling_factor_float,
        near_plane,
        far_plane,
        proper_antialiasing,
        supersampling_factor_float
    );
    if (debug) {
        auto ret = cudaDeviceSynchronize();
        if (ret != cudaSuccess) {
            std::cerr << "\n[CUDA ERROR] in accumulate_cu " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret) << "\n";
            throw std::runtime_error(cudaGetErrorString(ret));
        }
    }

    const uint32_t grid_size_resolve = (n_pixels + block_size_resolve - 1) / block_size_resolve;
    resolve_cu<<<grid_size_resolve, block_size_resolve, 0, stream>>>(
        means_ptr,
        sh_coefficients_ptr,
        cam_position_ptr,
        framebuffer_ptr,
        reinterpret_cast<float3*>(image.data_ptr<float>()),
        depth.data_ptr<float>(),
        alpha.data_ptr<float>(),
        n_pixels,
        width,
        sh_coefficients.size(1),
        supersampling_factor
    );
    if (debug) {
        auto ret = cudaDeviceSynchronize();
        if (ret != cudaSuccess) {
            std::cerr << "\n[CUDA ERROR] in resolve_cu " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret) << "\n";
            throw std::runtime_error(cudaGetErrorString(ret));
        }
    }

    return {image, depth, alpha};
}


// python binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("render", &render);
}
