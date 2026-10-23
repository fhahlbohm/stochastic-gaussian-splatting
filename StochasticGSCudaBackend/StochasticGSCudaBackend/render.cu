#include <cstdint>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include "helper_math.h"
#include "render_utils.h"
#include "accumulate_scatter.h"
#include "accumulate_hybrid.h"
#include "accumulate_hybrid_gps.h"
#include "resolve.h"


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
    const bool use_gps,
    const bool hybrid_scheduling)
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
        stream
    );

    // color, depth, and alpha of the closest accepted fragments
    sample_and_resolve(
        means_ptr,
        sh_coefficients_ptr,
        cam_position_ptr,
        framebuffer_ptr,
        reinterpret_cast<float3*>(image.data_ptr<float>()),
        depth.data_ptr<float>(),
        alpha.data_ptr<float>(),
        sh_coefficients.size(1),
        camera,
        stream
    );

    return {image, depth, alpha};
}


// python binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("render", &render);
}
