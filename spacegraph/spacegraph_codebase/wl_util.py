import pywt
from scipy.interpolate import interp1d
import torch
import numpy as np

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