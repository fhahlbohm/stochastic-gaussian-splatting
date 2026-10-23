# Stochastic Gaussian Splatting
![PyTorch](https://img.shields.io/badge/-PyTorch-EE4C2C?logo=PyTorch&logoColor=white)&nbsp;
![CUDA](https://img.shields.io/badge/-CUDA-76B900?logo=NVIDIA&logoColor=white)&nbsp;
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

__A CUDA implementation of two sorting-free, stochastic approaches for rendering [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) models with **zero** VRAM overhead: if the model fits in VRAM, it renders exactly. An optional hybrid scheduling trades a little extra VRAM for much faster rendering.__

This project is built on the following papers and their open-source implementations. Consider a citation when applicable.

```bibtex
@inproceedings{kheradmand2025stochasticsplats,
  title     = {{StochasticSplats}: Stochastic Rasterization for Sorting-Free 3D Gaussian Splatting},
  author    = {Kheradmand, Shakiba and Vicini, Delio and Kopanas, George and Lagun, Dmitry and Yi, Kwang Moo and Matthews, Mark and Tagliasacchi, Andrea},
  booktitle = {IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2025},
  pages     = {26326--26335},
  doi       = {10.1109/ICCV51701.2025.02443},
  url       = {https://ubc-vision.github.io/stochasticsplats/}
}
```
```bibtex
@article{rijsdijk2026gaussianpointsplatting,
  title     = {Gaussian Point Splatting},
  author    = {Rijsdijk, Joris and Peters, Christoph and Weinnman, Michael and Marroquim, Ricardo},
  journal   = {ACM Trans. Graph.},
  volume    = {45},
  number    = {4},
  publisher = {Association for Computing Machinery},
  year      = {2026},
  doi       = {10.1145/3811272},
  url       = {https://jorisar.nl/gaussian_point_splatting/}
}
```
```bibtex
@inproceedings{hahlbohm2026fastergs,
  title     = {{Faster-GS}: Analyzing and Improving Gaussian Splatting Optimization},
  author    = {Hahlbohm, Florian and Franke, Linus and Eisemann, Martin and Magnor, Marcus},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026},
  pages     = {18946--18957},
  doi       = {tbd},
  url       = {https://fhahlbohm.github.io/faster-gaussian-splatting/}
}
```


## Overview

Each sample is rendered in two stages (`render.cu`):

1. **Accumulation** keeps the closest *accepted* fragment of each (sub)pixel. `USE_GPS` selects one of two approaches:
   - **Stochastic transparency** (`accumulate_st_cu`): for every pixel inside a Gaussian's screen-space AABB,
     the Gaussian is *accepted* with probability equal to its evaluated alpha at that pixel. Accepted
     fragments race via a single `atomicMin` on a 64-bit framebuffer entry that packs
     `(depth_bits << 32) | gaussian_index`, so each pixel retains the closest accepted Gaussian. This applies the
     classic stochastic transparency idea to splatting, in the spirit of StochasticSplats.
   - **Gaussian Point Splatting (GPS)** (`accumulate_gps_cu`): instead of testing pixel centers, each Gaussian
     emits a Poisson-distributed number of sample points drawn from its own 2D distribution, splatting each as a
     point. This integrates the Gaussian over the pixel footprint by construction.
2. **Resolving** (`resolve_cu`) evaluates spherical harmonics for the **single** winning Gaussian of each (sub)pixel
   and writes color, depth, and alpha.

Averaging `SAMPLES_PER_PIXEL` independent frames — or accumulating them progressively across calls — converges to the
correctly blended image. An optional integer `SUPERSAMPLING_FACTOR` renders the framebuffer at higher resolution and
box-filters down.

### Scatter Path

By default, both accumulation kernels (`accumulate_scatter.cu`) launch **one thread per Gaussian** and balance work
*within a warp*: Gaussians with at most `n_sequential_threshold` (= 4) fragments are processed sequentially by their
owning thread, while larger Gaussians are handled **cooperatively** — the whole 32-lane warp picks them up one at a time
(parameters broadcast via `warp.shfl`) and splits their fragments across lanes. This keeps small primitives cheap and
needs no memory besides the framebuffer, but the per-thread mapping has inherent limitations:

- **No balancing across warps/blocks.** A warp that happens to contain one enormous Gaussian iterates many times
  while sibling warps sit idle — a classic tail effect that underutilizes the GPU.
- **Serialized large Gaussians within a warp.** The cooperative loop walks the warp's large primitives one after
  another, so a warp with several big Gaussians processes them sequentially.
- **GPS is affected even more**, since the per-Gaussian point count is Poisson-distributed and effectively unbounded,
  making the imbalance both larger and data-dependent. The `max_fragments` cap bounds the cost but can under-sample
  very large Gaussians.

As a result, the cost depends strongly on the view and grows quickly when Gaussians cover many pixels, e.g., when
zooming out or supersampling.

### Hybrid Scheduling

`HYBRID_SCHEDULING` (off by default) removes the imbalance for both approaches while keeping the scatter path for most
Gaussians. In exchange, it gives up the zero VRAM overhead: the extra memory grows with the number of large Gaussians (at
most `max_binned_gaussians` are binned, beyond which they fall back to the scatter path).

- **Stochastic transparency** (`accumulate_hybrid.cu`, bit-identical framebuffer): Gaussians whose bounding box covers
  more than `binning_pixel_threshold` pixels are not scattered but binned into 8x8 tiles (the tiles of each tile row
  follow from the extent of the ellipse within that row as in [AccuTile](https://arxiv.org/abs/2412.00578), sort by
  tile index only, no depth sort). After the scatter pass, one block per tile loops over the binned Gaussians of its
  tile and keeps the closest accepted fragment of each pixel in a register, so no atomics are needed and fragments
  behind the current winner are rejected before evaluation.
- **GPS** (`accumulate_hybrid_gps.cu`): the points of Gaussians with more than `binning_point_threshold` points are
  split into chunks of `chunk_size` points, and one block per chunk splats one point per thread. As the points of a
  Gaussian are independent samples, this only changes the random numbers, not the distribution of the result. This is
  the expand-and-distribute strategy of the original GPS implementation applied to large Gaussians only, small ones keep
  the sequential fast path that avoids loading their parameters twice and the per-point bookkeeping.

### Possible Improvements

- **Pre-compute the 3D covariance and opacity activation.** `project_gaussian` rebuilds `cov3d` (quaternion + scales)
  and applies `sigmoid(raw_opacity)` on every invocation, i.e. once per Gaussian *per sample*. Both can be computed
  once at load time; storing the 6-float `cov3d` in place of the scale/rotation buffers keeps the VRAM cost near zero.
  Gaussians with opacity below the `opacity_threshold` could also be culled at load time.
- **Quantize the Gaussian parameters.** Means, `log`-scales, quaternions, and `logit`-opacity are read in the hot loop,
  so smaller encodings (`fp16`, smallest-three quaternions, chunked low-bit quantization) help VRAM and bandwidth.
  Similar ideas could also be used for the SH coefficients. Many web viewers like
  [SuperSplat](https://github.com/playcanvas/supersplat)/[Spark](https://github.com/sparkjsdev/spark) and GPS already
  implement these ideas.

### Further Reading

If you are interested in the underlying ideas, you may find the following resources useful in addition to the papers
cited above:
- **[Stochastic transparency](https://research.nvidia.com/publication/2011-08_stochastic-transparency)**: The
  order-independent transparency technique this renderer builds on.
- **[atomicMin rendering](https://github.com/m-schuetz/compute_rasterizer)**: Markus Schütz et al.'s compute-shader point
  renderer popularized the 64-bit `atomicMin(depth | payload)` technique this renderer uses to resolve depth without sorting.
- **[CuRast](https://github.com/m-schuetz/CuRast)**: A CUDA software rasterizer also by Markus Schütz et al. that
  tackles a similar per-primitive load-balancing problem in the context of triangle rasterization.


## Installation

This method is provided as an extension to [NeRFICG](https://github.com/nerficg-project), a radiance field and
view synthesis framework. The `StochasticGSCudaBackend` can also be built standalone as a PyTorch extension for
integration into other codebases.

### Requirements

- An NVIDIA GPU
- Linux or Windows
- A recent CUDA SDK ([CUDA Toolkit 12.8](https://developer.nvidia.com/cuda-12-8-0-download-archive) recommended) and a compatible C++ compiler
- [Anaconda / Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install) installed

### Setup

As a preparatory step, the [NeRFICG framework](https://github.com/nerficg-project/nerficg) needs to be set up.
Please follow the instructions in its README to set up a compatible Conda environment.

Now add Stochastic Gaussian Splatting as an additional method by cloning this repository into `src/Methods/StochasticGS`:
```shell
# HTTPS
git clone https://github.com/fhahlbohm/stochastic-gaussian-splatting.git src/Methods/StochasticGS
```
or
```shell
# SSH
git clone git@github.com:fhahlbohm/stochastic-gaussian-splatting.git src/Methods/StochasticGS
```

Next, install all method-specific dependencies and CUDA extensions using:
```shell
python ./scripts/install.py -m StochasticGS
```

_Note: The framework determines on-the-fly what extra modules need to be installed. Sometimes this causes unnecessary errors/warnings that can interrupt the installation process. In this case, first try to rerun the command before investigating the error in detail._

The CUDA backend can also be built standalone as a PyTorch extension:
```shell
pip install git+https://github.com/fhahlbohm/stochastic-gaussian-splatting/#subdirectory=StochasticGSCudaBackend --no-build-isolation
```


## Usage

StochasticGS is a renderer, not an optimizer: it loads a pretrained 3DGS `.ply` (`MODEL.PLY_PATH`) and renders it,
so the example configs set `NUM_ITERATIONS: 0` and enable the interactive GUI. Two starting points are provided in
[`configs/`](configs):

- **[`m360_garden.yaml`](configs/m360_garden.yaml)** — render and evaluate a model against a real dataset. It uses
  `DATASET_TYPE: MipNeRF360` with a held-out test split (`TEST_STEP: 8`), so you can compare renders against ground
  truth. Set `MODEL.PLY_PATH` to your `.ply` and `DATASET.PATH` to the dataset.
- **[`jastrzebia_gora.yaml`](configs/jastrzebia_gora.yaml)** — free-viewpoint rendering of a standalone `.ply` with no
  dataset (`DATASET_TYPE: Empty`) along an optional predefined `CAMERA_PATH`. This is the right template for large
  scenes and fly-throughs; the example targets a 106M-Gaussian capture with a 600-frame camera path. Set
  `MODEL.PLY_PATH` and `CAMERA_PATH`.

The `RENDERER` section selects how the model is rendered:

- `USE_GPS`: Gaussian point splatting instead of stochastic transparency.
- `HYBRID_SCHEDULING`: load-balanced processing of large Gaussians, see [Hybrid Scheduling](#hybrid-scheduling).
- `SUPERSAMPLING_FACTOR`, `PROPER_ANTIALIASING`: integer supersampling and the antialiasing filter with opacity
  compensation instead of plain dilation.
- `SAMPLES_PER_PIXEL`, `DISABLE_PROGRESSIVE_RENDERING`: samples per call and whether consecutive calls accumulate.

To launch, fill in the `/TODO/` paths and run the config through the standard NeRFICG entry point:
```shell
python ./scripts/train.py -c src/Methods/StochasticGS/configs/m360_garden.yaml
```
With `NUM_ITERATIONS: 0` and `GUI.ACTIVATE: true` this loads the model, skips training, and opens the interactive
viewer. For a non-Empty dataset (e.g. `m360_garden.yaml`), the NeRFICG script `inference.py` can provide quality
metrics, FPS benchmarking, etc. as described in the [NeRFICG repository](https://github.com/nerficg-project/nerficg).
With an Empty dataset (e.g. `jastrzebia_gora.yaml`) there is no dataset to evaluate against — read `Trainer.py` to see
how the code loads `MODEL.CAMERA_PATH` and uses it for benchmarking FPS or rendering video frames.
