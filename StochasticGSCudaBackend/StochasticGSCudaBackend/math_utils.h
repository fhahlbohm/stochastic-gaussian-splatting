// Portions of this file are derived from https://github.com/JorisAR/gaussian-point-splatting
// Copyright (c) 2026, Joris Rijsdijk
// Licensed under the BSD 3-Clause License.
// Modifications made by Florian Hahlbohm.

#pragma once

#include <cstdint>
#include "helper_math.h"


inline constexpr float two_pi = 2.0f * 3.14159265358979323846f;

__device__ inline float dilog(const float x) {
    float y = -0.952623943f;
    y = fmaf(y, x, 2.995873777f);
    y = fmaf(y, x, -3.762115247f);
    y = fmaf(y, x, 2.334605281f);
    y = fmaf(y, x, -0.773941317f);
    y = fmaf(y, x, 5.97506396e-02f);
    y = fmaf(y, x, -0.256617940f);
    y = fmaf(y, x, 2.000061866f);
    y *= x;
    const float s = 1.0f - x;
    if (s > 0.0f) y += s * logf(fmaxf(s, 1.0e-37f));
    return y;
}

__device__ inline float inv_dilog(const float x) {
    const float t = fminf(x / 1.6449340668482264f, 1.0f);
    float y = -1.27463503e+01f;
    y = fmaf(y, t, 5.88993459e+01f);
    y = fmaf(y, t, -1.16025780e+02f);
    y = fmaf(y, t, 1.26945827e+02f);
    y = fmaf(y, t, -8.43108826e+01f);
    y = fmaf(y, t, 3.48799862e+01f);
    y = fmaf(y, t, -8.89606235e+00f);
    y = fmaf(y, t, 1.38640936e+00f);
    y = fmaf(y, t, -7.80640876e-01f);
    y = fmaf(y, t, 1.64841888e+00f);
    y = fmaf(y, t, -2.82836687e-05f);
    return y;
}

__device__ inline uint32_t hash32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352d;
    x ^= x >> 15;
    x *= 0x846ca68b;
    x ^= x >> 16;
    return x;
}

__device__ inline uint2 make_seed(const uint32_t idx, const uint32_t frame) {
    uint32_t h1 = hash32(idx ^ (frame * 0x9E3779B9u));
    uint32_t h2 = hash32((idx * 0xBB67AE85u) ^ frame);
    h1 ^= (h2 * 0x3C6EF372u);
    h2 ^= (h1 * 0xA54FF53Au);
    return make_uint2(h1, h2);
}

__device__ inline uint2 pcg2d_uint(uint2 seed) {
    seed.x = seed.x * 1664525u + 1013904223u;
    seed.y = seed.y * 1664525u + 1013904223u;
    seed.x += seed.y * 1664525u;
    seed.y += seed.x * 1664525u;
    seed.x ^= seed.x >> 16u;
    seed.y ^= seed.y >> 16u;
    seed.x += seed.y * 1664525u;
    seed.y += seed.x * 1664525u;
    seed.x ^= seed.x >> 16u;
    seed.y ^= seed.y >> 16u;
    return seed;
}

__device__ inline float2 pcg2d(uint2& seed) {
    seed = pcg2d_uint(seed);
    return make_float2(static_cast<float>(seed.x), static_cast<float>(seed.y)) * 2.3283064365387e-10f;
}

__device__ inline float2 box_muller(const float u1, const float u2) {
    const float R = sqrtf(fmaxf(-2.0f * logf(fmaxf(fminf(u1, 1.0f), 1.0e-37f)), 0.0f));
    const float theta = two_pi * u2;
    float s, c;
    sincosf(theta, &s, &c);
    return {R * c, R * s};
}

__device__ inline float2 corrected_box_muller(const float u1, const float u2, const float alpha_rcp, const float alpha_dilog) {
    const float a = alpha_rcp * inv_dilog((1.0f - u1) * alpha_dilog);
    const float R = sqrtf(fmaxf(-2.0f * logf(fmaxf(fminf(a, 1.0f), 1.0e-37f)), 0.0f));
    const float theta = two_pi * u2;
    float s, c;
    sincosf(theta, &s, &c);
    return {R * c, R * s};
}

__device__ inline uint32_t poisson(uint2& state, const float lambda) {
    const float2 u = pcg2d(state);
    const float w = box_muller(u.x, u.y).x;
    const float w2 = w * w;
    const float w3 = w2 * w;
    const float w4 = w2 * w2;
    const float s = sqrtf(lambda);
    const float s_rcp = 1.0f / s;
    const float l_rcp = 1.0f / lambda;
    float kf = lambda + s * w + (w2 - 1.0f) / 6.0f;
    kf += s_rcp * (-(1.0f / 36.0f) * w - (1.0f / 72.0f) * w3);
    kf += l_rcp * (-(8.0f / 405.0f) + (7.0f / 810.0f) * w2 + (1.0f / 270.0f) * w4);
    return __float2uint_rn(fmaxf(kf, 0.0f));
}

__device__ inline uint32_t pcg_hash(uint32_t seed) {
    const uint32_t state = seed * 747796405u + 2891336453u;
    const uint32_t word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

__device__ inline float rand(uint32_t frame, uint32_t instance, uint32_t x, uint32_t y) {
    const uint32_t seed = instance ^ (frame * 1973u) ^ (x * 9277u) ^ (y * 2699u);
    return __uint_as_float(0x3f800000u | (pcg_hash(seed) >> 9u)) - 1.0f;
}
