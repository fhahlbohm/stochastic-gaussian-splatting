"""StochasticGS/Model.py"""

import torch
import numpy as np
from plyfile import PlyData

import Framework
from Methods.Base.Model import BaseModel
from CudaUtils.MortonEncoding import morton_encode


@Framework.Configurable.configure(
    PLY_PATH='/path/to/ply/file.ply',
    CAMERA_PATH='/path/to/camera/path/file.json',
)
class StochasticGSModel(BaseModel):
    """Stores a standard 3DGS model."""

    def __init__(self, name: str = None) -> None:
        super().__init__(name)

    def build(self) -> 'StochasticGSModel':
        """Builds the model."""
        plydata = PlyData.read(self.PLY_PATH)

        means = np.stack((
            np.asarray(plydata.elements[0]['x']),
            np.asarray(plydata.elements[0]['y']),
            np.asarray(plydata.elements[0]['z'])
        ), axis=1)  # shape (N, 3)
        means = torch.as_tensor(means, dtype=torch.float32, device='cuda')
        morton_encoding = morton_encode(means)
        order = torch.argsort(morton_encoding)
        means = means[order]
        self.register_buffer('means', means)

        scales = np.stack((
            np.asarray(plydata.elements[0]['scale_0']),
            np.asarray(plydata.elements[0]['scale_1']),
            np.asarray(plydata.elements[0]['scale_2']),
        ), axis=1)  # shape (N, 3)
        scales = torch.as_tensor(scales, dtype=torch.float32, device='cuda')
        scales = scales[order]
        self.register_buffer('scales', scales)

        rotations = np.stack((
            np.asarray(plydata.elements[0]['rot_0']),
            np.asarray(plydata.elements[0]['rot_1']),
            np.asarray(plydata.elements[0]['rot_2']),
            np.asarray(plydata.elements[0]['rot_3']),
        ), axis=1)  # shape (N, 4)
        rotations = torch.as_tensor(rotations, dtype=torch.float32, device='cuda')
        rotations = rotations[order]
        self.register_buffer('rotations', rotations)

        opacities = np.asarray(plydata.elements[0]['opacity'])  # shape (N,)
        opacities = torch.as_tensor(opacities, dtype=torch.float32, device='cuda')
        opacities = opacities[order]
        self.register_buffer('opacities', opacities)

        n_rest_coefficients = len([p.name for p in plydata.elements[0].properties if p.name.startswith('f_rest_')])
        total_sh_bases = n_rest_coefficients // 3 + 1
        sh_coefficients = torch.empty((means.shape[0], total_sh_bases, 3), dtype=torch.float32, device='cuda')
        sh_coefficients_0 = np.stack((
            np.asarray(plydata.elements[0]['f_dc_0']),
            np.asarray(plydata.elements[0]['f_dc_1']),
            np.asarray(plydata.elements[0]['f_dc_2']),
        ), axis=1)  # shape (N, 3)
        sh_coefficients[:, 0, :] = torch.as_tensor(sh_coefficients_0, dtype=torch.float32, device='cuda')[order]
        if total_sh_bases > 1:
            sh_coefficients_rest = np.stack([
                np.asarray(plydata.elements[0][f'f_rest_{i}']) for i in range(n_rest_coefficients)
            ], axis=1)  # shape (N, 3 * (total_sh_bases - 1))
            sh_coefficients[:, 1:, :] = torch.as_tensor(sh_coefficients_rest, dtype=torch.float32, device='cuda')[order].reshape(-1, 3, total_sh_bases - 1).permute(0, 2, 1)
        self.register_buffer('sh_coefficients', sh_coefficients)

        torch.cuda.empty_cache()

        return self
