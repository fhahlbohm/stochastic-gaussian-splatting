"""StochasticGS/Renderer.py"""

import torch

import Framework
from Cameras.Perspective import PerspectiveCamera
from Datasets.utils import View
from Logging import Logger
from Methods.Base.Renderer import BaseModel
from Methods.Base.Renderer import BaseRenderer
from Methods.StochasticGS.Model import StochasticGSModel
from Methods.StochasticGS.StochasticGSCudaBackend import render, Settings, TemporalMode, History, create_k_buffer
from Visual.utils import apply_color_map


@Framework.Configurable.configure(
    PROPER_ANTIALIASING=False,
    USE_GPS=False,
    HYBRID_SCHEDULING=False,
    SUPERSAMPLING_FACTOR=1,
    SAMPLES_PER_PIXEL=1,
    DISABLE_PROGRESSIVE_RENDERING=False,
    SPATIAL_REUSE=False,
    TEMPORAL_REUSE=False,
)
class StochasticGSRenderer(BaseRenderer):
    """Wrapper around the rendering backend."""

    def __init__(self, model: 'BaseModel') -> None:
        super().__init__(model, [StochasticGSModel])
        if not Framework.config.GLOBAL.GPU_INDICES:
            raise Framework.RendererError('renderer not implemented in CPU mode')
        if len(Framework.config.GLOBAL.GPU_INDICES) > 1:
            Logger.log_warning(f'renderer not implemented in multi-GPU mode: using GPU {Framework.config.GLOBAL.GPU_INDICES[0]}')
        self.settings = None
        self.k_buffer = None
        self.history = None
        self.history_c2w = None
        self.history_w2c = None

    def _render_once(self, settings: 'Settings', view_to_previous_view: torch.Tensor, temporal_mode: 'TemporalMode') -> None:
        """Renders a single sample and merges it into the history."""
        self.history = render(
            self.model.means, self.model.scales, self.model.rotations, self.model.opacities, self.model.sh_coefficients,
            settings, self.k_buffer, self.history, view_to_previous_view, temporal_mode
        )

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
            self.SPATIAL_REUSE,
        )

        # the screen-space k-buffer persists across camera motion, the history is reused depending on the camera change
        reset = self.DISABLE_PROGRESSIVE_RENDERING and not benchmark
        if reset or self.settings is None or settings.as_tuple()[2:] != self.settings.as_tuple()[2:]:
            self.settings = settings
            self.k_buffer = create_k_buffer(settings)
            self.history = History(*(torch.empty(0) for _ in range(4)))
            temporal_mode = TemporalMode.RESET
        elif (self.history_c2w == view.c2w_numpy).all():
            temporal_mode = TemporalMode.STATIC
        else:
            temporal_mode = TemporalMode.REPROJECT if self.TEMPORAL_REUSE else TemporalMode.RESET
        view_to_previous_view = self.history_w2c @ view.c2w if temporal_mode == TemporalMode.REPROJECT else view.c2w
        for _ in range(max(self.SAMPLES_PER_PIXEL, 1)):
            self._render_once(settings, view_to_previous_view, temporal_mode)
            temporal_mode = TemporalMode.STATIC
        self.history_c2w = view.c2w_numpy
        self.history_w2c = view.w2c

        # history is premultiplied by alpha and must not be modified in place
        image, depth, alpha, _ = self.history
        alpha = alpha.clamp(0.0, 1.0)
        image = (image + (1.0 - alpha) * view.camera.background_color).clamp(0.0, 1.0)
        if benchmark:
            return {'rgb': image}
        depth = depth / alpha.clamp_min(1.0e-8)

        if to_chw:
            image = image.permute(2, 0, 1)
            depth = depth.permute(2, 0, 1)
            alpha = alpha.permute(2, 0, 1)
        return {'rgb': image, 'depth': depth, 'alpha': alpha}

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
