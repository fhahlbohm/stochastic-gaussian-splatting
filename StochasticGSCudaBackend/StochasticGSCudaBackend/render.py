from typing import NamedTuple, Any
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
        )

def render(
    means: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    opacities: torch.Tensor,
    sh_coefficients: torch.Tensor,
    settings: Settings,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _C.render(
        means,
        scales,
        rotations,
        opacities,
        sh_coefficients,
        *settings.as_tuple(),
    )
