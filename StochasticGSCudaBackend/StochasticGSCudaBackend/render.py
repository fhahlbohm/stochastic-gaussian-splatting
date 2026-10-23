from enum import IntEnum
from functools import cache
from pathlib import Path
from typing import NamedTuple
import torch
from StochasticGSCudaBackend import _C  # noqa

class Settings(NamedTuple):
    w2c: torch.Tensor  # affine transformation from model/world space to view space
    cam_position: torch.Tensor  # camera position in world space
    width: int  # width of the image plane in pixels
    height: int  # height of the image plane in pixels
    focal_x: float  # focal length in x direction in pixels
    focal_y: float  # focal length in y direction in pixels
    center_x: float  # x coordinate of the image center in pixels (positive -> right)
    center_y: float  # y coordinate of the image center in pixels (positive -> down)
    near_plane: float  # near clipping plane distance
    far_plane: float  # far clipping plane distance
    proper_antialiasing: bool  # whether to use proper antialiasing
    supersampling_factor: int  # factor for supersampling (2 -> render 2x2 subpixels)
    use_gps: bool  # whether to use Gaussian point splatting
    hybrid_scheduling: bool  # whether to process large Gaussians in a separate load-balanced pass
    spatial_reuse: bool  # whether to resample from the four neighbors of each (sub)pixel

    def as_tuple(self) -> tuple:
        return (
            self.w2c,
            self.cam_position,
            self.width,
            self.height,
            self.focal_x,
            self.focal_y,
            self.center_x,
            self.center_y,
            self.near_plane,
            self.far_plane,
            self.proper_antialiasing,
            self.supersampling_factor,
            self.use_gps,
            self.hybrid_scheduling,
            self.spatial_reuse,
        )

class TemporalMode(IntEnum):
    RESET = 0  # discard the history
    STATIC = 1  # same camera as the history: weighted cumulative average
    REPROJECT = 2  # moving camera: reprojected history with soft rejection

class History(NamedTuple):
    image: torch.Tensor  # (H, W, 3) color premultiplied by alpha
    depth: torch.Tensor  # (H, W, 1) depth premultiplied by alpha
    alpha: torch.Tensor  # (H, W, 1)
    weight_mass: torch.Tensor  # (H, W, 1) accumulated resampling weight

@cache
def blue_noise() -> torch.Tensor:
    """Returns the blue noise (64, 128, 128) downloaded at build time or an empty tensor (white noise)."""
    # next to _C, which is not this directory when importing from the source tree
    path = Path(_C.__file__).parent / 'stbn_scalar_128x128x64.bin'
    if not path.is_file():
        return torch.empty(0, dtype=torch.uint8, device='cuda')
    return torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.uint8).view(64, 128, 128).cuda()

def create_k_buffer(settings: Settings) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns empty k-buffers: primitive index and depth, occlusion factors towards the four neighbors."""
    n_entries = (_C.k_buffer_size if settings.spatial_reuse else 0, settings.width * settings.height * settings.supersampling_factor ** 2)
    return (
        torch.full((*n_entries, 2), -1, dtype=torch.int32, device='cuda'),
        torch.empty((*n_entries, 4), dtype=torch.float32, device='cuda'),
    )

def render(
    means: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    opacities: torch.Tensor,
    sh_coefficients: torch.Tensor,
    settings: Settings,
    k_buffer: tuple[torch.Tensor, torch.Tensor],
    history: History,
    view_to_previous_view: torch.Tensor,
    temporal_mode: TemporalMode,
) -> History:
    return History(*_C.render(
        means,
        scales,
        rotations,
        opacities,
        sh_coefficients,
        *settings.as_tuple(),
        blue_noise(),
        *k_buffer,
        *history,
        view_to_previous_view,
        int(temporal_mode),
    ))
