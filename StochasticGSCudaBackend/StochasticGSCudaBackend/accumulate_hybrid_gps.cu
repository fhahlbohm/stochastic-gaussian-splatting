#include <algorithm>
#include <cstdint>
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <torch/extension.h>
#include "helper_math.h"
#include "gs_utils.h"
#include "math_utils.h"
#include "render_utils.h"
#include "scatter_utils.cuh"
#include "accumulate_hybrid_gps.h"

namespace cg = cooperative_groups;

namespace hybrid_gps {

// config
inline constexpr uint32_t block_size_bin = 256;
inline constexpr uint32_t chunk_size = 256;
// larger Gaussians are binned, beyond max_binned_gaussians they are scattered
inline constexpr uint32_t binning_point_threshold = 512;
inline constexpr uint32_t max_binned_gaussians = 1u << 20;


// kernels
// accumulate_gps_cu that bins large Gaussians instead of scattering them
__global__ void bin_cu(
    const float3* __restrict__ means,
    const float3* __restrict__ scales,
    const float4* __restrict__ rotations,
    const float* __restrict__ opacities,
    const float4* __restrict__ w2c,
    unsigned long long* __restrict__ framebuffer,
    unsigned long long* __restrict__ binned_fragments,
    float2* __restrict__ binned_mean2d,
    float4* __restrict__ binned_L_power_threshold,
    float4* __restrict__ binned_conic_opacity_rcp,
    float* __restrict__ binned_opacity_dilog,
    uint32_t* __restrict__ binned_n_points,
    uint32_t* __restrict__ binned_n_chunks,
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
    const float supersampling_factor)
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

    // one atomic per warp reserves the slots
    const uint32_t binned_ballot = warp.ballot(active && fragment_count > binning_point_threshold);
    if (binned_ballot != 0) {
        const uint32_t leader_lane = __ffs(binned_ballot) - 1;
        uint32_t binned_offset = 0;
        if (lane_idx == leader_lane) binned_offset = atomicAdd(n_binned, __popc(binned_ballot));
        const uint32_t binned_idx = warp.shfl(binned_offset, leader_lane) + __popc(binned_ballot & ((1u << lane_idx) - 1u));
        if (((binned_ballot >> lane_idx) & 1u) && binned_idx < max_binned) {
            binned_fragments[binned_idx] = fragment;
            binned_mean2d[binned_idx] = mean2d;
            binned_L_power_threshold[binned_idx] = make_float4(L, power_threshold);
            binned_conic_opacity_rcp[binned_idx] = make_float4(conic, opacity_rcp);
            binned_opacity_dilog[binned_idx] = opacity_dilog;
            binned_n_points[binned_idx] = fragment_count;
            binned_n_chunks[binned_idx] = (fragment_count + chunk_size - 1) / chunk_size;
            active = false;
        }
    }

    scatter_points(
        warp, active, fragment, fragment_count, mean2d, conic, L, power_threshold, opacity_rcp, opacity_dilog, seed,
        framebuffer, width, height, width_int);
}

// one block per chunk of a binned Gaussian, one point per thread
__global__ void __launch_bounds__(chunk_size) chunks_cu(
    const uint32_t* __restrict__ binned_chunk_offsets,
    const unsigned long long* __restrict__ binned_fragments,
    const float2* __restrict__ binned_mean2d,
    const float4* __restrict__ binned_L_power_threshold,
    const float4* __restrict__ binned_conic_opacity_rcp,
    const float* __restrict__ binned_opacity_dilog,
    const uint32_t* __restrict__ binned_n_points,
    unsigned long long* __restrict__ framebuffer,
    const uint32_t n_binned,
    const uint32_t frame_idx,
    const float width,
    const float height)
{
    // binary search for the binned Gaussian of this chunk
    __shared__ uint32_t chunk_binned_idx;
    const uint32_t chunk_idx = blockIdx.x;
    if (threadIdx.x == 0) {
        uint32_t lo = 0, hi = n_binned - 1;
        while (lo < hi) {
            const uint32_t mid = (lo + hi) / 2;
            if (binned_chunk_offsets[mid] <= chunk_idx) lo = mid + 1;
            else hi = mid;
        }
        chunk_binned_idx = lo;
    }
    __syncthreads();
    const uint32_t binned_idx = chunk_binned_idx;
    const uint32_t first_chunk_idx = binned_idx > 0 ? binned_chunk_offsets[binned_idx - 1] : 0;
    const uint32_t point_idx = (chunk_idx - first_chunk_idx) * chunk_size + threadIdx.x;
    if (point_idx >= binned_n_points[binned_idx]) return;

    const unsigned long long fragment = binned_fragments[binned_idx];
    const float2 mean2d = binned_mean2d[binned_idx];
    const float4 L_power_threshold = binned_L_power_threshold[binned_idx];
    const float4 conic_opacity_rcp = binned_conic_opacity_rcp[binned_idx];
    const uint32_t gaussian_idx = static_cast<uint32_t>(fragment & 0xffffffffull);
    // independent random numbers per point
    uint2 seed = make_seed(gaussian_idx, frame_idx);
    seed.x ^= point_idx * 0x9E3779B9u;

    const float2 u = pcg2d(seed);
    const float2 xy = corrected_box_muller(u.x, u.y, conic_opacity_rcp.w, binned_opacity_dilog[binned_idx]);
    const float2 delta = make_float2(L_power_threshold.x * xy.x, L_power_threshold.y * xy.x + L_power_threshold.z * xy.y);
    const float2 sample = mean2d + delta;
    if (sample.x >= width || sample.x < 0.0f || sample.y >= height || sample.y < 0.0f) return;

    const float2 delta_center = make_float2(truncf(sample.x), truncf(sample.y)) + 0.5f - mean2d;
    const float power = 0.5f * (conic_opacity_rcp.x * delta_center.x * delta_center.x + conic_opacity_rcp.z * delta_center.y * delta_center.y) + conic_opacity_rcp.y * delta_center.x * delta_center.y;
    if (power > L_power_threshold.w) return;

    const uint32_t pixel_x = __float2uint_rd(sample.x);
    const uint32_t pixel_y = __float2uint_rd(sample.y);
    const uint32_t pixel_idx = pixel_y * __float2uint_rn(width) + pixel_x;
    atomicMin(&framebuffer[pixel_idx], fragment);
}

}


// host function
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
    const cudaStream_t stream)
{
    using namespace hybrid_gps;
    const float supersampling_factor = static_cast<float>(camera.supersampling_factor);
    const float width = static_cast<float>(camera.width * camera.supersampling_factor);
    const float height = static_cast<float>(camera.height * camera.supersampling_factor);
    const uint32_t max_binned = std::min(n_gaussians, max_binned_gaussians);
    auto byte_buffer = [](const size_t n_bytes) {
        return torch::empty({static_cast<int64_t>(n_bytes)}, torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA));
    };
    auto uint_buffer = [](const int64_t n_elements) {
        return torch::empty({n_elements}, torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA));
    };

    // scatter small Gaussians, bin large ones
    torch::Tensor binned_fragments = byte_buffer(static_cast<size_t>(max_binned) * sizeof(unsigned long long));
    torch::Tensor binned_mean2d = byte_buffer(static_cast<size_t>(max_binned) * sizeof(float2));
    torch::Tensor binned_L_power_threshold = byte_buffer(static_cast<size_t>(max_binned) * sizeof(float4));
    torch::Tensor binned_conic_opacity_rcp = byte_buffer(static_cast<size_t>(max_binned) * sizeof(float4));
    torch::Tensor binned_opacity_dilog = byte_buffer(static_cast<size_t>(max_binned) * sizeof(float));
    torch::Tensor binned_n_points = uint_buffer(max_binned);
    torch::Tensor binned_n_chunks = uint_buffer(max_binned);
    torch::Tensor n_binned_tensor = torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA));
    auto* binned_fragments_ptr = reinterpret_cast<unsigned long long*>(binned_fragments.data_ptr());
    auto* binned_mean2d_ptr = reinterpret_cast<float2*>(binned_mean2d.data_ptr());
    auto* binned_L_power_threshold_ptr = reinterpret_cast<float4*>(binned_L_power_threshold.data_ptr());
    auto* binned_conic_opacity_rcp_ptr = reinterpret_cast<float4*>(binned_conic_opacity_rcp.data_ptr());
    auto* binned_opacity_dilog_ptr = reinterpret_cast<float*>(binned_opacity_dilog.data_ptr());
    auto* binned_n_points_ptr = reinterpret_cast<uint32_t*>(binned_n_points.data_ptr<int>());
    auto* binned_n_chunks_ptr = reinterpret_cast<uint32_t*>(binned_n_chunks.data_ptr<int>());
    const uint32_t grid_size_bin = (n_gaussians + block_size_bin - 1) / block_size_bin;
    bin_cu<<<grid_size_bin, block_size_bin, 0, stream>>>(
        means,
        scales,
        rotations,
        opacities,
        w2c,
        framebuffer,
        binned_fragments_ptr,
        binned_mean2d_ptr,
        binned_L_power_threshold_ptr,
        binned_conic_opacity_rcp_ptr,
        binned_opacity_dilog_ptr,
        binned_n_points_ptr,
        binned_n_chunks_ptr,
        reinterpret_cast<uint32_t*>(n_binned_tensor.data_ptr<int>()),
        max_binned,
        n_gaussians,
        frame_idx,
        width,
        height,
        camera.focal_x * supersampling_factor,
        camera.focal_y * supersampling_factor,
        camera.center_x * supersampling_factor,
        camera.center_y * supersampling_factor,
        camera.near_plane,
        camera.far_plane,
        camera.proper_antialiasing,
        supersampling_factor
    );
    check_cuda("hybrid_gps::bin_cu");
    const uint32_t n_binned = std::min(static_cast<uint32_t>(n_binned_tensor.item<int>()), max_binned);
    if (n_binned == 0) return;

    // chunk offsets
    torch::Tensor binned_chunk_offsets = uint_buffer(n_binned);
    auto* binned_chunk_offsets_ptr = reinterpret_cast<uint32_t*>(binned_chunk_offsets.data_ptr<int>());
    size_t scan_bytes = 0;
    cub::DeviceScan::InclusiveSum(nullptr, scan_bytes, binned_n_chunks_ptr, binned_chunk_offsets_ptr, n_binned, stream);
    torch::Tensor scan_storage = byte_buffer(scan_bytes);
    cub::DeviceScan::InclusiveSum(scan_storage.data_ptr(), scan_bytes, binned_n_chunks_ptr, binned_chunk_offsets_ptr, n_binned, stream);
    const uint32_t n_chunks = static_cast<uint32_t>(binned_chunk_offsets[n_binned - 1].item<int>());

    chunks_cu<<<n_chunks, chunk_size, 0, stream>>>(
        binned_chunk_offsets_ptr,
        binned_fragments_ptr,
        binned_mean2d_ptr,
        binned_L_power_threshold_ptr,
        binned_conic_opacity_rcp_ptr,
        binned_opacity_dilog_ptr,
        binned_n_points_ptr,
        framebuffer,
        n_binned,
        frame_idx,
        width,
        height
    );
    check_cuda("hybrid_gps::chunks_cu");
}
