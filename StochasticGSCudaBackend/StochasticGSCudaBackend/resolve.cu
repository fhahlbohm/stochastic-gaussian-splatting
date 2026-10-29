#include <cstdint>
#include <cooperative_groups.h>
#include "helper_math.h"
#include "gs_utils.h"
#include "render_utils.h"
#include "resolve.h"

namespace cg = cooperative_groups;

namespace resolve {

// config
inline constexpr uint32_t block_size_sample = 256;
inline constexpr uint32_t block_size_resolve = 256;
// spatio-temporal sample reuse
inline constexpr float max_alpha = 0.99f;
inline constexpr float history_weight = 0.7f;
inline constexpr float gamma_static = 4.0f;
inline constexpr float gamma_motion = 1.0f;
inline constexpr float velocity_threshold = 4.0f;
inline constexpr float min_extent_luma = 0.01f;
inline constexpr float min_extent_chroma = 0.005f;
inline constexpr float border_margin = 2.0f;


// kernels
__global__ void sample_cu(
    const float3* __restrict__ means,
    const float3* __restrict__ scales,
    const float4* __restrict__ rotations,
    const float* __restrict__ opacities,
    const float3* __restrict__ sh_coefficients,
    const float4* __restrict__ w2c,
    const float3* __restrict__ cam_position,
    const unsigned long long* __restrict__ framebuffer,
    float4* __restrict__ samples,
    float4* __restrict__ reuse_weights,
    uint2* __restrict__ k_buffer_primitives,
    float4* __restrict__ k_buffer_occlusion,
    const uint32_t n_framebuffer_pixels,
    const uint32_t total_sh_bases,
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
    const bool spatial_reuse)
{
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    const uint32_t thread_idx = cg::this_grid().thread_rank();

    bool active = true;
    uint32_t pixel_idx = thread_idx;
    if (pixel_idx >= n_framebuffer_pixels) {
        active = false;
        pixel_idx = n_framebuffer_pixels - 1;
    }

    // load the closest accepted fragment and evaluate its color, depth 0 marks background
    const unsigned long long data = framebuffer[pixel_idx];
    const uint32_t depth_key = static_cast<uint32_t>(data >> 32);
    const bool has_sample = depth_key != 0xffffffffu;
    const uint32_t gaussian_idx = has_sample ? static_cast<uint32_t>(data & 0xffffffffu) : 0;
    float4 sample = make_float4(0.0f);
    if (has_sample) {
        const float3 color = eval_sh(
            sh_coefficients + gaussian_idx * total_sh_bases,
            normalize(means[gaussian_idx] - cam_position[0]),
            total_sh_bases);
        sample = make_float4(color, __uint_as_float(depth_key));
    }
    if (active) samples[pixel_idx] = sample;
    if (!spatial_reuse) return;

    // project the sampled Gaussian again to evaluate it at the neighbors
    bool projected = active && has_sample;
    float mean_depth, opacity, determinant, power_threshold;
    float2 mean2d;
    float3 conic, cov2d;
    float4 screen_bounds;
    project_gaussian(
        warp, projected, gaussian_idx, means, scales, rotations, opacities, w2c,
        width, height, focal_x, focal_y, center_x, center_y, near_plane, far_plane, proper_antialiasing, supersampling_factor,
        mean_depth, opacity, determinant, power_threshold, mean2d, conic, cov2d, screen_bounds);
    if (!active) return;

    // weights towards the four neighbors start with the change in alpha, background is opaque and farthest
    const uint32_t width_int = __float2uint_rn(width);
    float alpha = 1.0f;
    float4 neighbor_alpha = make_float4(1.0f);
    float sample_depth = __uint_as_float(0x7f800000u);
    if (projected) {
        const float delta_x = static_cast<float>(pixel_idx % width_int) - (mean2d.x - 0.5f);
        const float delta_y = static_cast<float>(pixel_idx / width_int) - (mean2d.y - 0.5f);
        const float3 conic_exp2 = scale_conic_for_exp2(conic);
        auto evaluate_alpha = [&](const float dx, const float dy) {
            const float exponent = dx * (conic_exp2.x * dx + conic_exp2.y * dy) + conic_exp2.z * dy * dy;
            const float fragment_alpha = opacity * exp2f(fminf(exponent, 0.0f));
            return fragment_alpha < min_alpha_threshold ? 0.0f : fragment_alpha;
        };
        // the clamp only guards against rounding
        alpha = fmaxf(evaluate_alpha(delta_x, delta_y), min_alpha_threshold);
        neighbor_alpha = make_float4(
            evaluate_alpha(delta_x - 1.0f, delta_y),
            evaluate_alpha(delta_x + 1.0f, delta_y),
            evaluate_alpha(delta_x, delta_y - 1.0f),
            evaluate_alpha(delta_x, delta_y + 1.0f)
        );
        sample_depth = sample.w;
    }
    float4 weights = has_sample && !projected ? make_float4(0.0f) : neighbor_alpha / alpha;

    // change in occlusion approximated with the k-buffer
    bool unique = projected;
    uint2 primitives[k_buffer_size];
    float4 occlusion[k_buffer_size];
    for (uint32_t k = 0; k < k_buffer_size; k++) {
        const uint64_t entry_idx = static_cast<uint64_t>(k) * n_framebuffer_pixels + pixel_idx;
        primitives[k] = k_buffer_primitives[entry_idx];
        occlusion[k] = k_buffer_occlusion[entry_idx];
        if (primitives[k].x == gaussian_idx) unique = false;
        // empty entries store a NaN depth and never occlude
        if (__uint_as_float(primitives[k].y) < sample_depth) weights *= occlusion[k];
    }

    // insert new primitives at the front, evicting the oldest entry
    if (unique) {
        for (uint32_t k = k_buffer_size - 1; k > 0; k--) {
            const uint64_t entry_idx = static_cast<uint64_t>(k) * n_framebuffer_pixels + pixel_idx;
            k_buffer_primitives[entry_idx] = primitives[k - 1];
            k_buffer_occlusion[entry_idx] = occlusion[k - 1];
        }
        k_buffer_primitives[pixel_idx] = make_uint2(gaussian_idx, depth_key);
        k_buffer_occlusion[pixel_idx] = (make_float4(1.0f) - fminf(neighbor_alpha, max_alpha)) / (1.0f - fminf(alpha, max_alpha));
    }

    reuse_weights[pixel_idx] = weights;
}

__device__ inline float3 rgb_to_ycocg(const float3 rgb) {
    return make_float3(
        0.25f * rgb.x + 0.5f * rgb.y + 0.25f * rgb.z,
        0.5f * rgb.x - 0.5f * rgb.z,
        -0.25f * rgb.x + 0.5f * rgb.y - 0.25f * rgb.z
    );
}

__device__ inline float4 catmull_rom_weights(const float t) {
    const float t2 = t * t;
    return make_float4(
        t * (-0.5f + t * (1.0f - 0.5f * t)),
        1.0f + t2 * (-2.5f + 1.5f * t),
        t * (0.5f + t * (2.0f - 1.5f * t)),
        t2 * (-0.5f + 0.5f * t)
    );
}

__global__ void resolve_cu(
    const float4* __restrict__ samples,
    const float4* __restrict__ reuse_weights,
    const float3* __restrict__ history_image,
    const float* __restrict__ history_depth,
    const float* __restrict__ history_alpha,
    const float* __restrict__ history_weight_mass,
    const float4* __restrict__ view_to_previous_view,
    float3* __restrict__ image,
    float* __restrict__ depth_map,
    float* __restrict__ alpha_map,
    float* __restrict__ weight_mass_map,
    const uint32_t n_pixels,
    const uint32_t width,
    const uint32_t height,
    const uint32_t supersampling_factor,
    const float focal_x,
    const float focal_y,
    const float center_x,
    const float center_y,
    const float far_plane,
    const bool spatial_reuse,
    const TemporalMode temporal_mode)
{
    const uint32_t pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel_idx >= n_pixels) return;

    const uint32_t framebuffer_base_x = (pixel_idx % width) * supersampling_factor;
    const uint32_t framebuffer_base_y = (pixel_idx / width) * supersampling_factor;
    const uint32_t framebuffer_width = width * supersampling_factor;
    const uint32_t framebuffer_height = height * supersampling_factor;
    const float normalization_factor = 1.0f / static_cast<float>(supersampling_factor * supersampling_factor);

    // self-normalized spatial resampling, neighbors stored their weights towards this subpixel
    float3 color_spatial = make_float3(0.0f);
    float depth_spatial = 0.0f;
    float alpha_spatial = 0.0f;
    float weight_mass_spatial = 0.0f;
    float depth_out = 0.0f;
    uint32_t valid_count = 0;
    for (uint32_t offset_y = 0; offset_y < supersampling_factor; offset_y++) {
        for (uint32_t offset_x = 0; offset_x < supersampling_factor; offset_x++) {
            const uint32_t framebuffer_x = framebuffer_base_x + offset_x;
            const uint32_t framebuffer_y = framebuffer_base_y + offset_y;
            const uint32_t framebuffer_idx = framebuffer_y * framebuffer_width + framebuffer_x;
            const float4 sample = samples[framebuffer_idx];
            if (sample.w > 0.0f) {
                depth_out += sample.w;
                valid_count++;
            }
            float3 color_sum = make_float3(sample);
            float depth_sum = sample.w;
            float alpha_sum = sample.w > 0.0f ? 1.0f : 0.0f;
            float weight_sum = 1.0f;
            auto add_neighbor = [&](const uint32_t neighbor_idx, const float weight) {
                const float4 neighbor_sample = samples[neighbor_idx];
                color_sum += weight * make_float3(neighbor_sample);
                depth_sum += weight * neighbor_sample.w;
                alpha_sum += neighbor_sample.w > 0.0f ? weight : 0.0f;
                weight_sum += weight;
            };
            if (spatial_reuse) {
                if (framebuffer_x > 0) add_neighbor(framebuffer_idx - 1, reuse_weights[framebuffer_idx - 1].y);
                if (framebuffer_x < framebuffer_width - 1) add_neighbor(framebuffer_idx + 1, reuse_weights[framebuffer_idx + 1].x);
                if (framebuffer_y > 0) add_neighbor(framebuffer_idx - framebuffer_width, reuse_weights[framebuffer_idx - framebuffer_width].w);
                if (framebuffer_y < framebuffer_height - 1) add_neighbor(framebuffer_idx + framebuffer_width, reuse_weights[framebuffer_idx + framebuffer_width].z);
            }
            const float weight_sum_rcp = 1.0f / weight_sum;
            color_spatial += color_sum * weight_sum_rcp;
            depth_spatial += depth_sum * weight_sum_rcp;
            alpha_spatial += alpha_sum * weight_sum_rcp;
            weight_mass_spatial += weight_sum;
        }
    }
    color_spatial *= normalization_factor;
    depth_spatial *= normalization_factor;
    alpha_spatial *= normalization_factor;
    weight_mass_spatial *= normalization_factor;

    // temporal resampling, all quantities are premultiplied by alpha
    float3 color = color_spatial;
    float depth = depth_spatial;
    float alpha = alpha_spatial;
    float weight_mass = weight_mass_spatial;
    switch (temporal_mode) {
        case TemporalMode::Reset: break;
        case TemporalMode::Static: {
            // weighted cumulative moving average
            const float history_mass = history_weight_mass[pixel_idx];
            weight_mass = history_mass + weight_mass_spatial;
            const float history_factor = history_mass / weight_mass;
            color = lerp(color_spatial, history_image[pixel_idx], history_factor);
            depth = lerp(depth_spatial, history_depth[pixel_idx], history_factor);
            alpha = lerp(alpha_spatial, history_alpha[pixel_idx], history_factor);
            break;
        }
        case TemporalMode::Reproject: {
            // reproject the sampled surface point, background at the far plane
            const uint32_t x = pixel_idx % width;
            const uint32_t y = pixel_idx / width;
            const float z = valid_count > 0 ? depth_out / static_cast<float>(valid_count) : far_plane;
            const float2 position = make_float2(static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f);
            const float3 view_position = make_float3(
                (position.x - center_x) / focal_x * z,
                (position.y - center_y) / focal_y * z,
                z
            );
            const float4 m_r1 = view_to_previous_view[0];
            const float4 m_r2 = view_to_previous_view[1];
            const float4 m_r3 = view_to_previous_view[2];
            const float previous_z = dot(make_float3(m_r3), view_position) + m_r3.w;
            if (!(previous_z > 0.0f)) break;
            const float2 previous_position = make_float2(
                focal_x * (dot(make_float3(m_r1), view_position) + m_r1.w) / previous_z + center_x,
                focal_y * (dot(make_float3(m_r2), view_position) + m_r2.w) / previous_z + center_y
            );
            // no history near the border, keeps all Catmull-Rom taps inside
            const bool inside = previous_position.x >= border_margin && previous_position.x <= static_cast<float>(width) - border_margin
                             && previous_position.y >= border_margin && previous_position.y <= static_cast<float>(height) - border_margin;
            if (!inside) break;

            // Catmull-Rom history fetch without clamping
            const float2 texel = previous_position - 0.5f;
            const float2 texel_base = floorf(texel);
            const float4 weights_x = catmull_rom_weights(texel.x - texel_base.x);
            const float4 weights_y = catmull_rom_weights(texel.y - texel_base.y);
            const float weights_x_array[4] = {weights_x.x, weights_x.y, weights_x.z, weights_x.w};
            const float weights_y_array[4] = {weights_y.x, weights_y.y, weights_y.z, weights_y.w};
            const uint32_t base_x = static_cast<uint32_t>(texel_base.x) - 1;
            const uint32_t base_y = static_cast<uint32_t>(texel_base.y) - 1;
            float3 color_history = make_float3(0.0f);
            float depth_history = 0.0f;
            float alpha_history = 0.0f;
            for (uint32_t j = 0; j < 4; j++) {
                for (uint32_t i = 0; i < 4; i++) {
                    const uint32_t history_idx = (base_y + j) * width + base_x + i;
                    const float weight = weights_x_array[i] * weights_y_array[j];
                    color_history += weight * history_image[history_idx];
                    depth_history += weight * history_depth[history_idx];
                    alpha_history += weight * history_alpha[history_idx];
                }
            }

            // YCoCg moments of the current samples in a 3x3 window
            float3 mean = make_float3(0.0f);
            float3 mean_squared = make_float3(0.0f);
            for (int offset_y = -1; offset_y <= 1; offset_y++) {
                for (int offset_x = -1; offset_x <= 1; offset_x++) {
                    const uint32_t neighbor_x = clamp(static_cast<int>(x) + offset_x, 0, static_cast<int>(width) - 1);
                    const uint32_t neighbor_y = clamp(static_cast<int>(y) + offset_y, 0, static_cast<int>(height) - 1);
                    float3 neighbor_color = make_float3(0.0f);
                    for (uint32_t subpixel_y = 0; subpixel_y < supersampling_factor; subpixel_y++)
                        for (uint32_t subpixel_x = 0; subpixel_x < supersampling_factor; subpixel_x++)
                            neighbor_color += make_float3(samples[(neighbor_y * supersampling_factor + subpixel_y) * framebuffer_width + neighbor_x * supersampling_factor + subpixel_x]);
                    const float3 neighbor_ycocg = rgb_to_ycocg(neighbor_color * normalization_factor);
                    mean += neighbor_ycocg;
                    mean_squared += neighbor_ycocg * neighbor_ycocg;
                }
            }
            mean /= 9.0f;
            mean_squared /= 9.0f;
            const float3 variance = fmaxf(mean_squared - mean * mean, 0.0f);
            const float3 standard_deviation = make_float3(sqrtf(variance.x), sqrtf(variance.y), sqrtf(variance.z));

            // soft history rejection with velocity-adaptive confidence half-extents
            const float velocity = length(position - previous_position);
            const float gamma = lerp(gamma_static, gamma_motion, fminf(velocity / velocity_threshold, 1.0f));
            const float3 extent = fmaxf(gamma * standard_deviation, make_float3(min_extent_luma, min_extent_chroma, min_extent_chroma));
            const float3 discrepancy = fabs(rgb_to_ycocg(color_history) - mean) / extent;
            const float distance = fmaxf(discrepancy.x, fmaxf(discrepancy.y, discrepancy.z));
            const float discount = distance > 1.0f ? history_weight / distance : history_weight;
            color = lerp(color_spatial, color_history, discount);
            depth = lerp(depth_spatial, depth_history, discount);
            alpha = lerp(alpha_spatial, alpha_history, discount);
            // steady-state weight mass, used once the camera stops
            weight_mass = weight_mass_spatial / (1.0f - discount);
            break;
        }
        default: __trap();
    }

    image[pixel_idx] = color;
    depth_map[pixel_idx] = depth;
    alpha_map[pixel_idx] = alpha;
    weight_mass_map[pixel_idx] = weight_mass;
}

}


// host function
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
    const cudaStream_t stream)
{
    using namespace resolve;
    const float supersampling_factor = static_cast<float>(camera.supersampling_factor);
    const uint32_t n_pixels = camera.width * camera.height;
    const uint32_t framebuffer_n_pixels = n_pixels * camera.supersampling_factor * camera.supersampling_factor;

    const uint32_t grid_size_sample = (framebuffer_n_pixels + block_size_sample - 1) / block_size_sample;
    sample_cu<<<grid_size_sample, block_size_sample, 0, stream>>>(
        means,
        scales,
        rotations,
        opacities,
        sh_coefficients,
        w2c,
        cam_position,
        framebuffer,
        samples,
        reuse_weights,
        k_buffer_primitives,
        k_buffer_occlusion,
        framebuffer_n_pixels,
        total_sh_bases,
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
        spatial_reuse
    );
    check_cuda("resolve::sample_cu");

    const uint32_t grid_size_resolve = (n_pixels + block_size_resolve - 1) / block_size_resolve;
    resolve_cu<<<grid_size_resolve, block_size_resolve, 0, stream>>>(
        samples,
        reuse_weights,
        history_image,
        history_depth,
        history_alpha,
        history_weight_mass,
        view_to_previous_view,
        image,
        depth,
        alpha,
        weight_mass,
        n_pixels,
        camera.width,
        camera.height,
        camera.supersampling_factor,
        camera.focal_x,
        camera.focal_y,
        camera.center_x,
        camera.center_y,
        camera.far_plane,
        spatial_reuse,
        temporal_mode
    );
    check_cuda("resolve::resolve_cu");
}
