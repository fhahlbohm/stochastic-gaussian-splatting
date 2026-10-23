#include <cstdint>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include "helper_math.h"
#include "math_utils.h"
#include "render_utils.h"
#include "accumulate_scatter.h"
#include "accumulate_hybrid.h"
#include "accumulate_hybrid_gps.h"
#include "resolve.h"


// host function
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
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
    const bool use_gps,
    const bool hybrid_scheduling,
    const bool spatial_reuse,
    const torch::Tensor& blue_noise,
    const torch::Tensor& k_buffer_primitives,
    const torch::Tensor& k_buffer_occlusion,
    const torch::Tensor& history_image,
    const torch::Tensor& history_depth,
    const torch::Tensor& history_alpha,
    const torch::Tensor& history_weight_mass,
    const torch::Tensor& view_to_previous_view,
    const int temporal_mode)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    const CameraSettings camera = {
        width, height, focal_x, focal_y, center_x, center_y, near_plane, far_plane, proper_antialiasing,
        static_cast<uint32_t>(max(1, supersampling_factor_raw))
    };
    const uint32_t n_gaussians = means.size(0);
    const uint32_t framebuffer_n_pixels = width * height * camera.supersampling_factor * camera.supersampling_factor;
    static uint32_t counter = 0;
    const uint32_t frame_idx = counter++;

    torch::Tensor image = torch::empty({height, width, 3}, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
    torch::Tensor depth = torch::empty({height, width, 1}, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
    torch::Tensor alpha = torch::empty({height, width, 1}, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
    torch::Tensor weight_mass = torch::empty({height, width, 1}, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
    torch::Tensor samples = torch::empty({framebuffer_n_pixels, 4}, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
    torch::Tensor reuse_weights = torch::empty({spatial_reuse ? framebuffer_n_pixels : 0, 4}, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
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

    // one blue noise slice per frame, each pass through the volume rotated by the golden ratio
    const bool use_blue_noise = blue_noise.numel() > 0;
    const uint8_t* blue_noise_slice = use_blue_noise
        ? blue_noise.contiguous().data_ptr<uint8_t>() + (frame_idx % blue_noise_depth) * blue_noise_width * blue_noise_height
        : nullptr;
    const uint32_t blue_noise_rotation = (frame_idx / blue_noise_depth) * 0x9E3779B9u;

    // closest accepted fragment per (sub)pixel
    if (hybrid_scheduling && use_gps) accumulate_hybrid_gps(
        means_ptr,
        scales_ptr,
        rotations_ptr,
        opacities_ptr,
        w2c_ptr,
        framebuffer_ptr,
        n_gaussians,
        frame_idx,
        camera,
        stream
    );
    else if (hybrid_scheduling) accumulate_hybrid(
        means_ptr,
        scales_ptr,
        rotations_ptr,
        opacities_ptr,
        w2c_ptr,
        framebuffer_ptr,
        n_gaussians,
        frame_idx,
        camera,
        blue_noise_slice,
        blue_noise_rotation,
        stream
    );
    else accumulate_scatter(
        means_ptr,
        scales_ptr,
        rotations_ptr,
        opacities_ptr,
        w2c_ptr,
        framebuffer_ptr,
        n_gaussians,
        frame_idx,
        camera,
        use_gps,
        blue_noise_slice,
        blue_noise_rotation,
        stream
    );

    // color, depth, and alpha with optional spatio-temporal reuse
    sample_and_resolve(
        means_ptr,
        scales_ptr,
        rotations_ptr,
        opacities_ptr,
        sh_coefficients_ptr,
        w2c_ptr,
        cam_position_ptr,
        framebuffer_ptr,
        reinterpret_cast<float4*>(samples.data_ptr<float>()),
        reinterpret_cast<float4*>(reuse_weights.data_ptr<float>()),
        reinterpret_cast<uint2*>(k_buffer_primitives.data_ptr<int>()),
        reinterpret_cast<float4*>(k_buffer_occlusion.data_ptr<float>()),
        reinterpret_cast<float3*>(history_image.data_ptr<float>()),
        history_depth.data_ptr<float>(),
        history_alpha.data_ptr<float>(),
        history_weight_mass.data_ptr<float>(),
        reinterpret_cast<float4*>(view_to_previous_view.contiguous().data_ptr<float>()),
        reinterpret_cast<float3*>(image.data_ptr<float>()),
        depth.data_ptr<float>(),
        alpha.data_ptr<float>(),
        weight_mass.data_ptr<float>(),
        sh_coefficients.size(1),
        camera,
        spatial_reuse,
        static_cast<TemporalMode>(temporal_mode),
        stream
    );

    return {image, depth, alpha, weight_mass};
}


// python binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("render", &render);
    m.attr("k_buffer_size") = k_buffer_size;
}
