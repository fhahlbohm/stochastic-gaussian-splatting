#include <cstdint>
#include "helper_math.h"
#include "gs_utils.h"
#include "render_utils.h"
#include "resolve.h"

namespace resolve {

// config
inline constexpr uint32_t block_size_resolve = 256;


// kernels
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


}


// host function
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
    const cudaStream_t stream)
{
    using namespace resolve;
    const uint32_t n_pixels = camera.width * camera.height;
    const uint32_t grid_size_resolve = (n_pixels + block_size_resolve - 1) / block_size_resolve;
    resolve_cu<<<grid_size_resolve, block_size_resolve, 0, stream>>>(
        means,
        sh_coefficients,
        cam_position,
        framebuffer,
        image,
        depth,
        alpha,
        n_pixels,
        camera.width,
        total_sh_bases,
        camera.supersampling_factor
    );
    check_cuda("resolve::resolve_cu");
}
