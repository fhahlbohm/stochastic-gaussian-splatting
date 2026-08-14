"""StochasticGS/Trainer.py"""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

import Framework
from Cameras.Perspective import PerspectiveCamera
from Datasets.Base import BaseDataset
from Datasets.utils import View, save_image
from Logging import Logger
from Methods.Base.GuiTrainer import GuiTrainer
from Methods.Base.utils import post_training_callback, pre_training_callback


@Framework.Configurable.configure(
    NUM_ITERATIONS=0,
)
class StochasticGSTrainer(GuiTrainer):
    """Defines the trainer for the StochasticGS variant."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @pre_training_callback(priority=10000)
    @torch.no_grad()
    def load_camera_path(self, _, dataset: 'BaseDataset') -> None:
        """Loads a camera path from a .json file and adds it as the test set."""
        USE_DATASET_CAMERA = True
        RENDER_WIDTH, RENDER_HEIGHT = 1920, 1080  # ignored when USE_DATASET_CAMERA=True

        if self.model.CAMERA_PATH is None:
            return
        path_file = Path(self.model.CAMERA_PATH)
        if not path_file.is_file():
            Logger.log_warning(f'skipping camera path loading: "{path_file}" does not exist')
            return
        with open(path_file, 'r') as f:
            camera_path = json.load(f)
        views: list[View] = []
        for frame in camera_path:
            c2w = np.eye(4, dtype=np.float64)
            c2w[:3, :3] = np.asarray(frame['rotation'], dtype=np.float64)
            c2w[:3, 3] = np.asarray(frame['position'], dtype=np.float64)
            if USE_DATASET_CAMERA:
                camera = dataset.cameras[0]
            else:
                camera = PerspectiveCamera(
                    shared_settings=dataset.default_camera.shared_settings,
                    width=RENDER_WIDTH, height=RENDER_HEIGHT,
                    focal_x=frame['fx'] * (RENDER_WIDTH / frame['width']),
                    focal_y=frame['fy'] * (RENDER_HEIGHT / frame['height']),
                )
            views.append(View(
                camera=camera,
                camera_index=0 if USE_DATASET_CAMERA else frame['id'],
                frame_idx=frame['id'],
                global_frame_idx=frame['id'],
                c2w=c2w,
            ))
        dataset.data['test'] = views
        Logger.log_info(f'loaded camera path with {len(views)} frames from "{path_file}" into test set')

    @post_training_callback(priority=1000)
    @torch.no_grad()
    def finalize(self, *_) -> None:
        """Write primitive count."""
        n_primitives = self.model.means.shape[0]
        Logger.log_info(f'number of primitives: {n_primitives:,}')
        with open(str(self.output_directory / 'n_primitives.txt'), 'w') as n_primitives_file:
            n_primitives_file.write(
                f'Number of primitives: {n_primitives:,}\n\n'
                f'N_primitives:{n_primitives}'
            )

    @post_training_callback(priority=900)
    @torch.no_grad()
    def render_camera_path(self, _, dataset: 'BaseDataset') -> None:
        """When GUI is disabled and MODEL.CAMERA_PATH is set, measure frame times and optionally write out images."""
        if self.GUI.ACTIVATE or self.model.CAMERA_PATH is None:
            return

        NUM_ITERATIONS = 10
        PER_FRAME_TIMINGS = True
        SAVE_IMAGES = False

        dataset.test()
        Logger.log_info(f'benchmarking FPS ({len(dataset)} frames, {NUM_ITERATIONS} passes)')
        for view in Logger.log_progress(dataset, leave=False, desc='warming up'):
            self.renderer.render_image(view, benchmark=True)
        torch.cuda.synchronize()
        per_frame_times = np.zeros(len(dataset), dtype=np.float64)
        start_time = perf_counter()
        for _ in Logger.log_progress(range(NUM_ITERATIONS), leave=False, desc='benchmarking performance'):
            for frame_index, view in enumerate(dataset):
                if PER_FRAME_TIMINGS:
                    torch.cuda.synchronize()
                    frame_start_time = perf_counter()
                    self.renderer.render_image(view, benchmark=True)
                    torch.cuda.synchronize()
                    per_frame_times[frame_index] += (perf_counter() - frame_start_time) * 1000
                else:
                    self.renderer.render_image(view, benchmark=True)
        torch.cuda.synchronize()
        total_time = perf_counter() - start_time
        num_renders = len(dataset) * NUM_ITERATIONS
        avg_fps = num_renders / total_time
        avg_ms_per_image = (total_time * 1000) / num_renders
        Logger.log_info(f'average FPS: {avg_fps:.2f} ({avg_ms_per_image:.2f} ms)')
        if PER_FRAME_TIMINGS:
            per_frame_times /= NUM_ITERATIONS
            frame_times_path = self.output_directory / 'per_frame_times.txt'
            with open(str(frame_times_path), 'w') as frame_times_file:
                frame_times_file.write(f'average frame time in milliseconds over {NUM_ITERATIONS} passes\n')
                for view, frame_time in zip(dataset, per_frame_times):
                    frame_times_file.write(f'{view.frame_idx}\t{frame_time:.4f}\n')
            Logger.log_info(f'per-frame render times written to {frame_times_path}')

        if not SAVE_IMAGES:
            return
        output_directory = self.output_directory / 'camera_path'
        output_directory.mkdir(parents=True, exist_ok=True)
        Logger.log_info(f'rendering camera path to {output_directory}')
        for view in Logger.log_progress(dataset, desc='rendering camera path', leave=False):
            save_image(output_directory / f'{view.frame_idx:05d}.png', self.renderer.render_image(view, to_chw=True)['rgb'])
