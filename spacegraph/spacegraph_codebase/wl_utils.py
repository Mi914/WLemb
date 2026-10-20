import pywt
from scipy.interpolate import interp1d
import torch
import numpy as np
import torch.nn as nn


def calc_input_dim_Multisol(extent = (-1710000, -1690000, 1610000, 1640000), min_scale = 1000):
    # compute the dimention of the encoded spatial relation embedding
    scale = min_scale
    dim = 0
    while True:
        cnt = 0 
        for x in range(extent[0], extent[1], scale):
            for y in range(extent[2], extent[3], scale):
                dim += 3
                if scale == min_scale: dim += 1
                cnt += 1
        if cnt <= 1:
            break
        scale *= 2
    return dim

"""
class WLFuncForMultisol:
    def __init__(self, funcname: str):
        wavelet = pywt.Wavelet(funcname)
        self._phi, self._psi, self.x = wavelet.wavefun(level=10)
        self.phi = interp1d(self.x, self._phi, kind='linear', fill_value=0.0, bounds_error=False)
        self.psi = interp1d(self.x, self._psi, kind='linear', fill_value=0.0, bounds_error=False)

    def __call__(self, coord, base, scale, j):
        x = coord[..., 0]
        y = coord[..., 1]
        bx, by = base
        phx = self.phi((x - bx) / scale) * 2**(- j / 2)
        phy = self.phi((y - by) / scale) * 2**(- j / 2)
        psx = self.psi((x - bx) / scale) * 2**(- j / 2)
        psy = self.psi((y - by) / scale) * 2**(- j / 2)
        output = [phx * psy, psx * phy, psx * psy]
        output = np.stack(output, axis=-1)
        output = torch.from_numpy(output).float()
        # output = torch.tensor(output, dtype=torch.float32)
        return output

    def phph(self, coord, base, scale, j):
        x = coord[..., 0]
        y = coord[..., 1]
        bx, by = base
        phx = self.phi((x - bx) / scale) * 2**(- j / 2)
        phy = self.phi((y - by) / scale) * 2**(- j / 2)
        output = phx * phy
        output = torch.from_numpy(output).float()
        # output = torch.tensor(output, dtype=torch.float32)
        return output
"""
class WLFuncForMultisol(nn.Module):
    def __init__(self, funcname: str):
        super().__init__()

        wavelet = pywt.Wavelet(funcname)
        phi, psi, x = wavelet.wavefun(level=10)

        # モデルと一緒にGPUへ移動できるようにする
        self.register_buffer("phi_values", torch.tensor(phi, dtype=torch.float32))
        self.register_buffer("psi_values", torch.tensor(psi, dtype=torch.float32))
        self.register_buffer("x_values", torch.tensor(x, dtype=torch.float32))

    def interpolate(self, t, values):
        """PyTorchによる線形補間。範囲外は0。"""
        x = self.x_values

        # 等間隔のサンプル点に対応する連続インデックス
        idx = (t - x[0]) / (x[1] - x[0])
        i0 = torch.floor(idx).long()
        i1 = i0 + 1

        # インデックスを有効範囲に制限
        i0 = i0.clamp(0, len(x) - 1)
        i1 = i1.clamp(0, len(x) - 1)

        weight = idx - torch.floor(idx)
        result = (1 - weight) * values[i0] + weight * values[i1]

        # 範囲外は0
        return torch.where(
            (t >= x[0]) & (t <= x[-1]),
            result,
            torch.zeros_like(result)
        )

    def phi(self, t):
        return self.interpolate(t, self.phi_values)

    def psi(self, t):
        return self.interpolate(t, self.psi_values)

    def forward(self, coord, base, scale, j):
        diff = coord.unsqueeze(-2) - base

        if torch.is_tensor(scale) and scale.ndim > 0:
            scale = scale.reshape(1, 1, -1, 1)

        normalized = diff / scale

        x = normalized[..., 0]  # (batch, n, basis)
        y = normalized[..., 1]

        factor = torch.pow(
            2.0, -j.to(dtype=coord.dtype) / 2
        ).reshape(1, 1, -1)

        phx = self.phi(x) * factor
        phy = self.phi(y) * factor
        psx = self.psi(x) * factor
        psy = self.psi(y) * factor

        output = torch.stack(
            [phx * psy, psx * phy, psx * psy],
            dim=-1
        ).flatten(start_dim=-2)

        return output  # (batch, n, basis * 3)

    def phph(self, coord, base, scale, j):
        # coord: (batch, n, 2)
        # base:  (basis, 2)
        # j:     (basis,)

        diff = coord.unsqueeze(-2) - base  # (batch, n, basis, 2)

        if torch.is_tensor(scale) and scale.ndim > 0:
            scale = scale.reshape(1, 1, -1, 1)

        normalized = diff / scale

        x = normalized[..., 0]  # (batch, n, basis)
        y = normalized[..., 1]

        factor = torch.pow(
            2.0, -j.to(dtype=coord.dtype) / 2
        ).reshape(1, 1, -1)

        phx = self.phi(x) * factor
        phy = self.phi(y) * factor

        return phx * phy  # (batch, n, basis)
