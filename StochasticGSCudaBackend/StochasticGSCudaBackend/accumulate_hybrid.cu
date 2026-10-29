#include <algorithm>
#include <cstdint>
#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>
#include <cub/cub.cuh>
#include <torch/extension.h>
#include "helper_math.h"
#include "gs_utils.h"
#include "math_utils.h"
#include "render_utils.h"
#include "scatter_utils.cuh"
#include "accumulate_hybrid.h"

namespace cg = cooperative_groups;

namespace hybrid {

// config
inline constexpr uint32_t block_size_bin = 256;
inline constexpr uint32_t block_size_tiles = 256;
inline constexpr uint32_t block_size_ranges = 256;
inline constexpr uint32_t tile_width = 8;
inline constexpr uint32_t tile_height = 8;
inline constexpr uint32_t block_size_gather = tile_width * tile_height;
// larger Gaussians are binned, beyond max_binned_gaussians they are scattered
inline constexpr uint32_t binning_pixel_threshold = 512;
inline constexpr uint32_t max_binned_gaussians = 1u << 20;
// keeps the tile spans conservative despite rounding
inline constexpr float span_power_margin = 1e-3f;


// kernels
// accumulate_st_cu that bins large Gaussians instead of scattering them
__global__ void bin_cu(
    const float3* __restrict__ means,
    const float3* __restrict__ scales,
    const float4* __restrict__ rotations,
    const float* __restrict__ opacities,
    const float4* __restrict__ w2c,
    unsigned long long* __restrict__ framebuffer,
    unsigned long long* __restrict__ binned_fragments,
    float2* __restrict__ binned_mean2d_shifted,
    float4* __restrict__ binned_conic_opacity,
    float* __restrict__ binned_determinant_rcp,
    ushort4* __restrict__ binned_bounds,
    uint32_t* __restrict__ n_binned,
    const uint32_t max_binned,
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
    const uint32_t lane_idx = warp.thread_rank();

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

    // one atomic per warp reserves the slots
    const uint32_t binned_ballot = warp.ballot(active && fragment_count > binning_pixel_threshold);
    if (binned_ballot != 0) {
        const uint32_t leader_lane = __ffs(binned_ballot) - 1;
        uint32_t binned_offset = 0;
        if (lane_idx == leader_lane) binned_offset = atomicAdd(n_binned, __popc(binned_ballot));
        const uint32_t binned_idx = warp.shfl(binned_offset, leader_lane) + __popc(binned_ballot & ((1u << lane_idx) - 1u));
        if (((binned_ballot >> lane_idx) & 1u) && binned_idx < max_binned) {
            binned_fragments[binned_idx] = fragment;
            binned_mean2d_shifted[binned_idx] = mean2d_shifted;
            binned_conic_opacity[binned_idx] = make_float4(conic, opacity);
            binned_determinant_rcp[binned_idx] = 1.0f / determinant;
            binned_bounds[binned_idx] = make_ushort4(x_min, x_max, y_min, y_max);
            active = false;
        }
    }

    scatter_fragments(
        warp, active, gaussian_idx, fragment, x_min, y_min, aabb_width, fragment_count, mean2d_shifted, conic, opacity,
        framebuffer, width_int, frame_idx, blue_noise_slice, blue_noise_rotation);
}

// counts or writes the tile instances of each binned Gaussian from per-row tile spans as in AccuTile
// (https://arxiv.org/abs/2412.00578), explicitly rounded so that both passes agree
template <bool write_instances>
__global__ void tiles_cu(
    const float2* __restrict__ binned_mean2d_shifted,
    const float4* __restrict__ binned_conic_opacity,
    const float* __restrict__ binned_determinant_rcp,
    const ushort4* __restrict__ binned_bounds,
    uint32_t* __restrict__ binned_n_tiles,
    uint32_t* __restrict__ instance_keys,
    uint32_t* __restrict__ instance_binned_indices,
    const uint32_t n_binned,
    const uint32_t grid_width)
{
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    const uint32_t thread_idx = cg::this_grid().thread_rank();
    const uint32_t lane_idx = warp.thread_rank();

    bool active = true;
    uint32_t binned_idx = thread_idx;
    if (binned_idx >= n_binned) {
        active = false;
        binned_idx = n_binned - 1;
    }
    const uint32_t active_threads = warp.ballot(active);
    if (active_threads == 0) return;

    // write offset or tile count
    uint32_t instance_idx = write_instances && binned_idx > 0 ? binned_n_tiles[binned_idx - 1] : 0;

    // the whole warp processes one binned Gaussian at a time
    const uint32_t n_active_threads = __popc(active_threads);
    for (uint32_t n = 0; n < n_active_threads && n < 32; n++) {
        const uint32_t current_lane = __fns(active_threads, 0, n + 1);
        const uint32_t binned_idx_coop = warp.shfl(binned_idx, current_lane);
        const ushort4 bounds = binned_bounds[binned_idx_coop];
        const float2 mean2d_shifted = binned_mean2d_shifted[binned_idx_coop];
        const float4 conic_opacity = binned_conic_opacity[binned_idx_coop];
        const float determinant_rcp = binned_determinant_rcp[binned_idx_coop];
        const uint32_t tile_x_min = bounds.x / tile_width;
        const uint32_t tile_x_max = (bounds.y + tile_width - 1) / tile_width;
        const uint32_t tile_y_min = bounds.z / tile_height;
        const uint32_t tile_y_max = (bounds.w + tile_height - 1) / tile_height;

        // ellipse at the alpha threshold and the y coordinates of its leftmost and rightmost points
        const float conic_x_rcp = __frcp_rn(conic_opacity.x);
        const float two_power_threshold = __fmul_rn(2.0f, __fadd_rn(__logf(__fmul_rn(conic_opacity.w, min_alpha_threshold_rcp)), span_power_margin));
        const float x_extent = __fsqrt_rn(__fdiv_rn(__fmul_rn(two_power_threshold, conic_opacity.z), determinant_rcp));
        const float y_offset = __fdiv_rn(__fmul_rn(conic_opacity.y, x_extent), conic_opacity.z);
        const float y_leftmost = __fadd_rn(mean2d_shifted.y, y_offset);
        const float y_rightmost = __fsub_rn(mean2d_shifted.y, y_offset);

        uint32_t instance_idx_coop = warp.shfl(instance_idx, current_lane);
        for (uint32_t row_offset = 0; row_offset < tile_y_max - tile_y_min; row_offset += 32) {
            // tile span of one tile row per lane
            const uint32_t tile_y = tile_y_min + row_offset + lane_idx;
            uint32_t span_start = 0;
            uint32_t span_length = 0;
            if (tile_y < tile_y_max) {
                const float y_top = static_cast<float>(tile_y * tile_height);
                const float y_bottom = static_cast<float>(tile_y * tile_height + tile_height - 1);
                float x_lo = __int_as_float(0x7f800000);
                float x_hi = -__int_as_float(0x7f800000);
                auto intersect_line = [&](const float y) {
                    const float delta_y = __fsub_rn(y, mean2d_shifted.y);
                    const float discriminant = __fsub_rn(__fmul_rn(two_power_threshold, conic_opacity.x), __fmul_rn(__fmul_rn(delta_y, delta_y), determinant_rcp));
                    if (discriminant < 0.0f) return;
                    const float center = __fsub_rn(mean2d_shifted.x, __fmul_rn(__fmul_rn(conic_opacity.y, delta_y), conic_x_rcp));
                    const float half_width = __fmul_rn(__fsqrt_rn(discriminant), conic_x_rcp);
                    x_lo = fminf(x_lo, __fsub_rn(center, half_width));
                    x_hi = fmaxf(x_hi, __fadd_rn(center, half_width));
                };
                intersect_line(y_top);
                intersect_line(y_bottom);
                if (y_top <= y_leftmost && y_leftmost <= y_bottom) x_lo = __fsub_rn(mean2d_shifted.x, x_extent);
                if (y_top <= y_rightmost && y_rightmost <= y_bottom) x_hi = __fadd_rn(mean2d_shifted.x, x_extent);
                if (x_lo <= x_hi) {
                    const int start = max(__float2int_rd(__fmul_rn(x_lo, 1.0f / tile_width)), static_cast<int>(tile_x_min));
                    const int end = min(__float2int_rd(__fmul_rn(x_hi, 1.0f / tile_width)) + 1, static_cast<int>(tile_x_max));
                    if (end > start) {
                        span_start = static_cast<uint32_t>(start);
                        span_length = static_cast<uint32_t>(end - start);
                    }
                }
            }
            const uint32_t span_end = cg::inclusive_scan(warp, span_length);
            const uint32_t n_row_tiles = warp.shfl(span_end, 31);
            if constexpr (write_instances) {
                const uint32_t span_begin = span_end - span_length;
                for (uint32_t tile_offset = 0; tile_offset < n_row_tiles; tile_offset += 32) {
                    const uint32_t local_idx = tile_offset + lane_idx;
                    // binary search for the tile row of this instance
                    uint32_t row_idx = 0;
                    for (uint32_t step = 16; step > 0; step >>= 1) {
                        if (warp.shfl(span_end, row_idx + step - 1) <= local_idx) row_idx += step;
                    }
                    const uint32_t row_begin = warp.shfl(span_begin, row_idx);
                    const uint32_t row_start = warp.shfl(span_start, row_idx);
                    if (local_idx >= n_row_tiles) continue;
                    instance_keys[instance_idx_coop + local_idx] = (tile_y_min + row_offset + row_idx) * grid_width + row_start + (local_idx - row_begin);
                    instance_binned_indices[instance_idx_coop + local_idx] = binned_idx_coop;
                }
            }
            instance_idx_coop += n_row_tiles;
        }
        if (lane_idx == current_lane) instance_idx = instance_idx_coop;
    }

    if constexpr (!write_instances) {
        if (active) binned_n_tiles[binned_idx] = instance_idx;
    }
}

__global__ void extract_instance_ranges_cu(
    const uint32_t* __restrict__ instance_keys,
    uint2* __restrict__ tile_instance_ranges,
    const uint32_t n_instances)
{
    const uint32_t instance_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (instance_idx >= n_instances) return;
    const uint32_t instance_tile_idx = instance_keys[instance_idx];
    if (instance_idx == 0) tile_instance_ranges[instance_tile_idx].x = 0;
    else {
        const uint32_t previous_instance_tile_idx = instance_keys[instance_idx - 1];
        if (instance_tile_idx != previous_instance_tile_idx) {
            tile_instance_ranges[previous_instance_tile_idx].y = instance_idx;
            tile_instance_ranges[instance_tile_idx].x = instance_idx;
        }
    }
    if (instance_idx == n_instances - 1) tile_instance_ranges[instance_tile_idx].y = n_instances;
}

// one block per tile, closest accepted fragment per pixel in a register
__global__ void __launch_bounds__(block_size_gather) gather_cu(
    const uint2* __restrict__ tile_instance_ranges,
    const uint32_t* __restrict__ instance_binned_indices,
    const unsigned long long* __restrict__ binned_fragments,
    const float2* __restrict__ binned_mean2d_shifted,
    const float4* __restrict__ binned_conic_opacity,
    const ushort4* __restrict__ binned_bounds,
    unsigned long long* __restrict__ framebuffer,
    const uint32_t width,
    const uint32_t height,
    const uint32_t grid_width,
    const uint32_t frame_idx,
    const uint8_t* __restrict__ blue_noise_slice,
    const uint32_t blue_noise_rotation)
{
    const uint32_t tile_idx = blockIdx.x;
    const uint2 tile_range = tile_instance_ranges[tile_idx];
    if (tile_range.x == tile_range.y) return;

    const uint32_t thread_rank = threadIdx.x;
    const uint32_t pixel_x = (tile_idx % grid_width) * tile_width + thread_rank % tile_width;
    const uint32_t pixel_y = (tile_idx / grid_width) * tile_height + thread_rank / tile_width;
    const bool inside = pixel_x < width && pixel_y < height;
    const uint32_t pixel_idx = pixel_y * width + pixel_x;
    const bool use_blue_noise = blue_noise_slice != nullptr;

    // fragments behind the closest one cannot win, pixels outside the image skip all fragments
    const unsigned long long initial_fragment = inside ? framebuffer[pixel_idx] : 0ull;
    unsigned long long closest_fragment = initial_fragment;

    __shared__ unsigned long long collected_fragments[block_size_gather];
    __shared__ float2 collected_mean2d_shifted[block_size_gather];
    __shared__ float4 collected_conic_opacity[block_size_gather];
    __shared__ ushort4 collected_bounds[block_size_gather];
    for (uint32_t batch_start = tile_range.x; batch_start < tile_range.y; batch_start += block_size_gather) {
        __syncthreads();
        const uint32_t fetch_idx = batch_start + thread_rank;
        if (fetch_idx < tile_range.y) {
            const uint32_t binned_idx = instance_binned_indices[fetch_idx];
            collected_fragments[thread_rank] = binned_fragments[binned_idx];
            collected_mean2d_shifted[thread_rank] = binned_mean2d_shifted[binned_idx];
            const float4 conic_opacity = binned_conic_opacity[binned_idx];
            collected_conic_opacity[thread_rank] = make_float4(scale_conic_for_exp2(make_float3(conic_opacity)), conic_opacity.w);
            collected_bounds[thread_rank] = binned_bounds[binned_idx];
        }
        __syncthreads();
        const uint32_t batch_size = min(block_size_gather, tile_range.y - batch_start);
        for (uint32_t j = 0; j < batch_size; j++) {
            const unsigned long long fragment = collected_fragments[j];
            if (fragment >= closest_fragment) continue;
            // same pixels and evaluation as the scatter path up to fma contraction
            const ushort4 bounds = collected_bounds[j];
            if (pixel_x < bounds.x || pixel_x >= bounds.y || pixel_y < bounds.z || pixel_y >= bounds.w) continue;
            const float2 mean2d_shifted = collected_mean2d_shifted[j];
            const float4 conic_opacity = collected_conic_opacity[j];
            const float delta_x = static_cast<float>(pixel_x) - mean2d_shifted.x;
            const float delta_y = static_cast<float>(pixel_y) - mean2d_shifted.y;
            const float exponent = delta_x * (conic_opacity.x * delta_x + conic_opacity.y * delta_y) + conic_opacity.z * delta_y * delta_y;
            const float alpha = conic_opacity.w * exp2f(fminf(exponent, 0.0f));
            if (alpha < min_alpha_threshold) continue;
            const uint32_t gaussian_idx = static_cast<uint32_t>(fragment & 0xffffffffull);
            const float sample = use_blue_noise ? blue_noise(blue_noise_slice, blue_noise_rotation, gaussian_idx, pixel_x, pixel_y) : rand(frame_idx, gaussian_idx, pixel_x, pixel_y);
            if (sample >= alpha) continue;
            closest_fragment = fragment;
        }
    }

    if (closest_fragment != initial_fragment) framebuffer[pixel_idx] = closest_fragment;
}

}


// host function
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
    const uint8_t* blue_noise_slice,
    const uint32_t blue_noise_rotation,
    const cudaStream_t stream)
{
    using namespace hybrid;
    const float supersampling_factor = static_cast<float>(camera.supersampling_factor);
    const uint32_t width_int = camera.width * camera.supersampling_factor;
    const uint32_t height_int = camera.height * camera.supersampling_factor;
    const uint32_t max_binned = std::min(n_gaussians, max_binned_gaussians);
    const uint32_t grid_width = (width_int + tile_width - 1) / tile_width;
    const uint32_t n_tiles = grid_width * ((height_int + tile_height - 1) / tile_height);
    auto byte_buffer = [](const size_t n_bytes) {
        return torch::empty({static_cast<int64_t>(n_bytes)}, torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA));
    };
    auto uint_buffer = [](const int64_t n_elements) {
        return torch::empty({n_elements}, torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA));
    };

    // scatter small Gaussians, bin large ones
    torch::Tensor binned_fragments = byte_buffer(static_cast<size_t>(max_binned) * sizeof(unsigned long long));
    torch::Tensor binned_mean2d_shifted = byte_buffer(static_cast<size_t>(max_binned) * sizeof(float2));
    torch::Tensor binned_conic_opacity = byte_buffer(static_cast<size_t>(max_binned) * sizeof(float4));
    torch::Tensor binned_determinant_rcp = byte_buffer(static_cast<size_t>(max_binned) * sizeof(float));
    torch::Tensor binned_bounds = byte_buffer(static_cast<size_t>(max_binned) * sizeof(ushort4));
    torch::Tensor n_binned_tensor = torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA));
    auto* binned_fragments_ptr = reinterpret_cast<unsigned long long*>(binned_fragments.data_ptr());
    auto* binned_mean2d_shifted_ptr = reinterpret_cast<float2*>(binned_mean2d_shifted.data_ptr());
    auto* binned_conic_opacity_ptr = reinterpret_cast<float4*>(binned_conic_opacity.data_ptr());
    auto* binned_determinant_rcp_ptr = reinterpret_cast<float*>(binned_determinant_rcp.data_ptr());
    auto* binned_bounds_ptr = reinterpret_cast<ushort4*>(binned_bounds.data_ptr());
    const uint32_t grid_size_bin = (n_gaussians + block_size_bin - 1) / block_size_bin;
    bin_cu<<<grid_size_bin, block_size_bin, 0, stream>>>(
        means,
        scales,
        rotations,
        opacities,
        w2c,
        framebuffer,
        binned_fragments_ptr,
        binned_mean2d_shifted_ptr,
        binned_conic_opacity_ptr,
        binned_determinant_rcp_ptr,
        binned_bounds_ptr,
        reinterpret_cast<uint32_t*>(n_binned_tensor.data_ptr<int>()),
        max_binned,
        n_gaussians,
        frame_idx,
        static_cast<float>(width_int),
        static_cast<float>(height_int),
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
    check_cuda("hybrid::bin_cu");
    const uint32_t n_binned = std::min(static_cast<uint32_t>(n_binned_tensor.item<int>()), max_binned);
    if (n_binned == 0) return;

    // instance creation
    torch::Tensor binned_n_tiles = uint_buffer(n_binned);
    torch::Tensor binned_offsets = uint_buffer(n_binned);
    auto* binned_n_tiles_ptr = reinterpret_cast<uint32_t*>(binned_n_tiles.data_ptr<int>());
    auto* binned_offsets_ptr = reinterpret_cast<uint32_t*>(binned_offsets.data_ptr<int>());
    const uint32_t grid_size_tiles = (n_binned + block_size_tiles - 1) / block_size_tiles;
    tiles_cu<false><<<grid_size_tiles, block_size_tiles, 0, stream>>>(
        binned_mean2d_shifted_ptr,
        binned_conic_opacity_ptr,
        binned_determinant_rcp_ptr,
        binned_bounds_ptr,
        binned_n_tiles_ptr,
        nullptr,
        nullptr,
        n_binned,
        grid_width
    );
    check_cuda("hybrid::tiles_cu<false>");
    size_t scan_bytes = 0;
    cub::DeviceScan::InclusiveSum(nullptr, scan_bytes, binned_n_tiles_ptr, binned_offsets_ptr, n_binned, stream);
    torch::Tensor scan_storage = byte_buffer(scan_bytes);
    cub::DeviceScan::InclusiveSum(scan_storage.data_ptr(), scan_bytes, binned_n_tiles_ptr, binned_offsets_ptr, n_binned, stream);
    const uint32_t n_instances = static_cast<uint32_t>(binned_offsets[n_binned - 1].item<int>());
    if (n_instances == 0) return;

    torch::Tensor instance_keys = uint_buffer(n_instances);
    torch::Tensor instance_binned_indices = uint_buffer(n_instances);
    auto* instance_keys_ptr = reinterpret_cast<uint32_t*>(instance_keys.data_ptr<int>());
    auto* instance_binned_indices_ptr = reinterpret_cast<uint32_t*>(instance_binned_indices.data_ptr<int>());
    tiles_cu<true><<<grid_size_tiles, block_size_tiles, 0, stream>>>(
        binned_mean2d_shifted_ptr,
        binned_conic_opacity_ptr,
        binned_determinant_rcp_ptr,
        binned_bounds_ptr,
        binned_offsets_ptr,
        instance_keys_ptr,
        instance_binned_indices_ptr,
        n_binned,
        grid_width
    );
    check_cuda("hybrid::tiles_cu<true>");

    // group by tile, order within a tile does not matter
    torch::Tensor instance_keys_sorted = uint_buffer(n_instances);
    torch::Tensor instance_binned_indices_sorted = uint_buffer(n_instances);
    auto* instance_keys_sorted_ptr = reinterpret_cast<uint32_t*>(instance_keys_sorted.data_ptr<int>());
    auto* instance_binned_indices_sorted_ptr = reinterpret_cast<uint32_t*>(instance_binned_indices_sorted.data_ptr<int>());
    int end_bit = 1;
    while ((1ull << end_bit) < n_tiles) end_bit++;
    size_t sort_bytes = 0;
    cub::DeviceRadixSort::SortPairs(
        nullptr, sort_bytes,
        instance_keys_ptr, instance_keys_sorted_ptr,
        instance_binned_indices_ptr, instance_binned_indices_sorted_ptr,
        n_instances, 0, end_bit, stream
    );
    torch::Tensor sort_storage = byte_buffer(sort_bytes);
    cub::DeviceRadixSort::SortPairs(
        sort_storage.data_ptr(), sort_bytes,
        instance_keys_ptr, instance_keys_sorted_ptr,
        instance_binned_indices_ptr, instance_binned_indices_sorted_ptr,
        n_instances, 0, end_bit, stream
    );
    check_cuda("cub::DeviceRadixSort::SortPairs");

    torch::Tensor tile_instance_ranges = torch::zeros({n_tiles, 2}, torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA));
    auto* tile_instance_ranges_ptr = reinterpret_cast<uint2*>(tile_instance_ranges.data_ptr<int>());
    const uint32_t grid_size_ranges = (n_instances + block_size_ranges - 1) / block_size_ranges;
    extract_instance_ranges_cu<<<grid_size_ranges, block_size_ranges, 0, stream>>>(
        instance_keys_sorted_ptr,
        tile_instance_ranges_ptr,
        n_instances
    );
    check_cuda("hybrid::extract_instance_ranges_cu");

    gather_cu<<<n_tiles, block_size_gather, 0, stream>>>(
        tile_instance_ranges_ptr,
        instance_binned_indices_sorted_ptr,
        binned_fragments_ptr,
        binned_mean2d_shifted_ptr,
        binned_conic_opacity_ptr,
        binned_bounds_ptr,
        framebuffer,
        width_int,
        height_int,
        grid_width,
        frame_idx,
        blue_noise_slice,
        blue_noise_rotation
    );
    check_cuda("hybrid::gather_cu");
}
