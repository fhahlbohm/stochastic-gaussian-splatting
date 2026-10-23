"""StochasticGS/Renderer.py"""

import torch

import Framework
from Cameras.Perspective import PerspectiveCamera
from Datasets.utils import View
from Logging import Logger
from Methods.Base.Renderer import BaseModel
from Methods.Base.Renderer import BaseRenderer
from Methods.StochasticGS.Model import StochasticGSModel
from Methods.StochasticGS.StochasticGSCudaBackend import render, Settings
from Visual.utils import apply_color_map


@Framework.Configurable.configure(
    PROPER_ANTIALIASING=False,
    USE_GPS=False,
    HYBRID_SCHEDULING=False,
    SUPERSAMPLING_FACTOR=1,
    SAMPLES_PER_PIXEL=1,
    DISABLE_PROGRESSIVE_RENDERING=False,
)
class StochasticGSRenderer(BaseRenderer):
    """Wrapper around the rendering backend."""

    def __init__(self, model: 'BaseModel') -> None:
        super().__init__(model, [StochasticGSModel])
        if not Framework.config.GLOBAL.GPU_INDICES:
            raise Framework.RendererError('renderer not implemented in CPU mode')
        if len(Framework.config.GLOBAL.GPU_INDICES) > 1:
            Logger.log_warning(f'renderer not implemented in multi-GPU mode: using GPU {Framework.config.GLOBAL.GPU_INDICES[0]}')
        self.accum_image = None
        self.accum_depth = None
        self.accum_alpha = None
        self.accum_total_samples = None
        self.accum_settings = None
        self.accum_c2w = None

    def _render_once(self, settings: 'Settings') -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Renders a single sample, returning image, depth, and alpha."""
        return render(self.model.means, self.model.scales, self.model.rotations, self.model.opacities, self.model.sh_coefficients, settings)

    def _accumulate_samples(self, settings: 'Settings', samples_per_pixel: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Renders and accumulates the given number of samples per pixel."""
        image, depth, alpha = self._render_once(settings)
        depth *= alpha
        for _ in range(samples_per_pixel - 1):
            image_i, depth_i, alpha_i = self._render_once(settings)
            image += image_i
            depth += depth_i * alpha_i
            alpha += alpha_i
        return image, depth, alpha

    @torch.inference_mode()
    def render_image(self, view: View, to_chw: bool = False, benchmark: bool = False) -> dict[str, torch.Tensor]:
        """Renders an image for a given view."""
        if not isinstance(view.camera, PerspectiveCamera):
            raise Framework.RendererError('renderer only supports perspective cameras')
        if view.camera.distortion is not None:
            Logger.log_warning('found distortion parameters that will be ignored by the rasterizer')
        settings = Settings(
            view.w2c,
            view.position,
            view.camera.width,
            view.camera.height,
            view.camera.focal_x,
            view.camera.focal_y,
            view.camera.center_x,
            view.camera.center_y,
            view.camera.near_plane,
            view.camera.far_plane,
            self.PROPER_ANTIALIASING,
            self.SUPERSAMPLING_FACTOR,
            self.USE_GPS,
            self.HYBRID_SCHEDULING,
        )
        samples_per_pixel = max(self.SAMPLES_PER_PIXEL, 1)

        if benchmark:
            return self._render_benchmark(settings, samples_per_pixel)

        if self.DISABLE_PROGRESSIVE_RENDERING:
            image, depth, alpha = self._accumulate_samples(settings, samples_per_pixel)
            total_samples = samples_per_pixel
        else:
            if self.accum_image is None or any(x != y for x, y in zip(self.accum_settings.as_tuple()[2:], settings.as_tuple()[2:])) or (self.accum_c2w != view.c2w_numpy).any():
                self.accum_image = torch.zeros(view.camera.height, view.camera.width, 3)
                self.accum_depth = torch.zeros(view.camera.height, view.camera.width, 1)
                self.accum_alpha = torch.zeros(view.camera.height, view.camera.width, 1)
                self.accum_total_samples = 0
                self.accum_settings = settings
                self.accum_c2w = view.c2w_numpy
            image_i, depth_i, alpha_i = self._accumulate_samples(self.accum_settings, samples_per_pixel)
            self.accum_image += image_i
            self.accum_depth += depth_i
            self.accum_alpha += alpha_i
            self.accum_total_samples += samples_per_pixel
            image, depth, alpha = self.accum_image, self.accum_depth, self.accum_alpha
            total_samples = self.accum_total_samples

        depth = depth / alpha.clamp_min(1.0e-8)
        if total_samples > 1:
            image = image / total_samples
            alpha = alpha / total_samples
        image = image + (1.0 - alpha) * view.camera.background_color
        image = image.clamp(0.0, 1.0)

        if to_chw:
            image = image.permute(2, 0, 1)
            depth = depth.permute(2, 0, 1)
            alpha = alpha.permute(2, 0, 1)
        return {'rgb': image, 'depth': depth, 'alpha': alpha}

    def _render_benchmark(self, settings: 'Settings', samples_per_pixel: int) -> dict[str, torch.Tensor]:
        """Renders only the image for a given view, skipping all depth-related work."""
        image = self._render_once(settings)[0]
        for _ in range(samples_per_pixel - 1):
            image += self._render_once(settings)[0]
        if samples_per_pixel > 1:
            image /= samples_per_pixel
        image.clamp_(0.0, 1.0)
        return {'rgb': image}

    def postprocess_outputs(self, outputs: dict[str, torch.Tensor], *_) -> dict[str, torch.Tensor]:
        """Postprocesses the model outputs, returning tensors of shape 3xHxW."""
        return {
            'rgb': outputs['rgb'],
            'alpha': outputs['alpha'].expand_as(outputs['rgb']),
            'depth': apply_color_map(
                color_map='SPECTRAL',
                image=outputs['depth'],
                min_max=None,
                mask=outputs['depth'] > 0.0,
                interpolate=True
            ),
        }
