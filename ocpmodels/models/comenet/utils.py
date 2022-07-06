"""
This part borrows heavily from the GemNet basis utils code
https://github.com/TUM-DAML/gemnet_pytorch/blob/master/gemnet/model/layers/basis_layers.py

Copyright (c) 2022 Johannes Gasteiger, Florian Becker
"""

import torch
from ocpmodels.models.gemnet.layers.basis_utils import *


class AngleEmbedding(torch.nn.Module):
    def __init__(self, num_radial, num_spherical, cutoff=8.0):
        super(AngleEmbedding, self).__init__()
        assert num_radial <= 64
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = cutoff

        bessel_formulas = bessel_basis(num_spherical, num_radial)
        Y_lm = real_sph_harm(
            num_spherical, use_theta=True, zero_m_only=True
        )
        self.sph_funcs = []
        self.bessel_funcs = []

        x = sym.symbols("x")
        theta = sym.symbols("theta")
        modules = {"sin": torch.sin, "cos": torch.cos, "sqrt": torch.sqrt}
        m = 0
        for l in range(len(Y_lm)):
            if l == 0:
                first_sph = sym.lambdify([theta], Y_lm[l][m], modules)
                self.sph_funcs.append(
                    lambda theta: torch.zeros_like(theta) + first_sph(theta)
                )
            else:
                self.sph_funcs.append(sym.lambdify([theta], Y_lm[l][m], modules))
            for n in range(num_radial):
                self.bessel_funcs.append(
                    sym.lambdify([x], bessel_formulas[l][n], modules)
                )

    def forward(self, dist, angle):
        dist = dist / self.cutoff
        rbf = torch.stack([f(dist) for f in self.bessel_funcs], dim=1)
        sbf = torch.stack([f(angle) for f in self.sph_funcs], dim=1)
        n, k = self.num_spherical, self.num_radial
        out = (rbf.view(-1, n, k) * sbf.view(-1, n, 1)).view(-1, n * k)
        return out


class TorsionEmbedding(torch.nn.Module):
    def __init__(self, num_radial, num_spherical, cutoff=8.0):
        super(TorsionEmbedding, self).__init__()
        assert num_radial <= 64
        self.num_radial = num_radial
        self.num_spherical = num_spherical
        self.cutoff = cutoff

        bessel_formulas = bessel_basis(num_spherical, num_radial)
        Y_lm = real_sph_harm(
            num_spherical, use_theta=True, use_phi=True, zero_m_only=False
        )
        self.sph_funcs = []
        self.bessel_funcs = []

        x = sym.symbols("x")
        theta = sym.symbols("theta")
        phi = sym.symbols("phi")
        modules = {"sin": torch.sin, "cos": torch.cos, "sqrt": torch.sqrt}
        for l in range(len(Y_lm)):
            for m in range(len(Y_lm[l])):
                if (
                        l == 0
                ):
                    first_sph = sym.lambdify([theta, phi], Y_lm[l][m], modules)
                    self.sph_funcs.append(
                        lambda theta, phi: torch.zeros_like(theta)
                                           + first_sph(theta, phi)
                    )
                else:
                    self.sph_funcs.append(
                        sym.lambdify([theta, phi], Y_lm[l][m], modules)
                    )
            for j in range(num_radial):
                self.bessel_funcs.append(
                    sym.lambdify([x], bessel_formulas[l][j], modules)
                )

        self.register_buffer(
            "degreeInOrder", torch.arange(num_spherical) * 2 + 1, persistent=False
        )

    def forward(self, dist, theta, phi):
        dist = dist / self.cutoff
        rbf = torch.stack([f(dist) for f in self.bessel_funcs], dim=1)
        sbf = torch.stack([f(theta, phi) for f in self.sph_funcs], dim=1)

        n, k = self.num_spherical, self.num_radial
        rbf = rbf.view((-1, n, k)).repeat_interleave(self.degreeInOrder, dim=1).view((-1, n ** 2 * k))
        sbf = sbf.repeat_interleave(k, dim=1)
        out = rbf * sbf
        return out