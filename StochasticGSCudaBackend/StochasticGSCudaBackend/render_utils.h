#pragma once

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <cuda_runtime.h>

inline constexpr bool debug = false;

// sync and error check after each launch if debug is enabled
inline void check_cuda(const char* kernel_name) {
    if (!debug) return;
    auto ret = cudaDeviceSynchronize();
    if (ret != cudaSuccess) {
        std::cerr << "\n[CUDA ERROR] in " << kernel_name << ": " << cudaGetErrorString(ret) << "\n";
        throw std::runtime_error(cudaGetErrorString(ret));
    }
}

// camera settings in image pixels, not subpixels
struct CameraSettings {
    uint32_t width;
    uint32_t height;
    float focal_x;
    float focal_y;
    float center_x;
    float center_y;
    float near_plane;
    float far_plane;
    bool proper_antialiasing;
    uint32_t supersampling_factor;
};
