import pywt
from scipy.interpolate import interp1d
import torch

class WLFuncForMultisol:
    def __init__(self, funcname: str):
        wavelet = pywt.Wavelet(funcname)
        self._phi, self._psi, self.x = wavelet.wavefun(level=10)
        self.phi = interp1d(self.x, self._phi, kind='linear', fill_value=0.0, bounds_error=False)
        self.psi = interp1d(self.x, self._psi, kind='linear', fill_value=0.0, bounds_error=False)

    def __call__(self, coord, base, scale, j):
        x, y = coord
        bx, by = base
        phx = self.phi((x - bx) / scale) * 2**(- j / 2)
        phy = self.phi((y - by) / scale) * 2**(- j / 2)
        psx = self.psi((x - bx) / scale) * 2**(- j / 2)
        psy = self.psi((y - by) / scale) * 2**(- j / 2)
        output = [phx * psy, psx * phy, psx * psy]
        output = torch.tensor(output, dtype=torch.float32)
        return output

    def phph(self, coord, base, scale, j):
        x, y = coord
        bx, by = base
        phx = self.phi((x - bx) / scale) * 2**(- j / 2)
        phy = self.phi((y - by) / scale) * 2**(- j / 2)
        output = phx * phy
        output = torch.tensor(output, dtype=torch.float32)
        return output