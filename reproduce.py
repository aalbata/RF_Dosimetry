#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Incident power density thresholds for computational radiofrequency dosimetry.

This file reproduces every table and figure of the accompanying article and
the computed values quoted in its text. It contains the electromagnetic and
bioheat solvers, the two tissue-property uncertainty models, and the
threshold layer.

Forward models: transmission-line planar slab, multilayer Mie sphere (via
scattnlay), IEC/IEEE 62704-1-style peak 10 g spatial-average SAR, and
steady-state Pennes bioheat (1-D finite volume for the slab, 3-D voxel finite
volume for the sphere). Threshold layer: empirical and conformal quantiles,
nonparametric tolerance limits, CVaR, Cantelli moment bounds, and Wasserstein
CVaR bounds, each converted to an incident power density threshold.

Nominal dielectric spectra use the four-term Cole-Cole parameters of Gabriel,
Lau and Gabriel (Phys. Med. Biol. 41, 1996). Tissue density and thermal ranges
from the IT'IS database and the dielectric uncertainty parameters are stored
as constants, so the script needs no data download.

Usage
-----
    python reproduce.py --check                 # dependency check; no simulation
    python reproduce.py                         # article results from data/samples
    python reproduce.py --simulate              # regenerate all endpoint samples
    python reproduce.py --fast --outdir quick   # reduced-size pipeline check

License: MIT (see LICENSE).
"""
from __future__ import annotations

import argparse
import contextlib
import ctypes
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from itertools import product
from typing import Sequence

import numpy as np

# ----------------------------------------------------------------------------
# Section 1. Physical constants
# ----------------------------------------------------------------------------
EPS0 = 8.8541878128e-12          # vacuum permittivity, F/m
MU0 = 4.0e-7 * np.pi             # vacuum permeability, H/m
C0 = 1.0 / np.sqrt(EPS0 * MU0)   # speed of light, m/s
ETA0 = np.sqrt(MU0 / EPS0)       # free-space wave impedance, ohm

# Study configuration shared by manuscript and code.
FREQUENCIES_HZ = (0.9e9, 2.45e9, 5.8e9)
FREQUENCY_LABEL = {0.9e9: "0.9 GHz", 2.45e9: "2.45 GHz", 5.8e9: "5.8 GHz"}
ANCHOR_S0_W_PER_M2 = 50.0        # linear scaling anchor, not an exposure limit
SAR_LIMIT_W_PER_KG = 2.0         # ICNIRP 2020 general public, local head/torso, 10 g, 6 min
DELTA_T_LIMIT_C = 1.0            # benchmark for the steady-state peak temperature rise
ALPHA = 0.05                     # target exceedance probability
DELTA = 0.05                     # tolerance/confidence miss probability
TARGET_MASS_KG = 0.010           # 10 g averaging mass
SEED = 20260711

# General-public far-field reference levels for incident power density
# (ICNIRP 2020). Two tiers exist and they pair with different basic
# restrictions: Table 5 (whole-body exposure, 30 min average) pairs with the
# whole-body-average SAR restriction, while Table 6 (local exposure, 6 min
# average) pairs with the local 10 g SAR restriction used for the thresholds.
def reference_level_w_per_m2(frequency_hz: float) -> float:
    """Whole-body-average tier (ICNIRP 2020 Table 5), used as a conservative
    screen; it pairs with the 0.08 W/kg whole-body restriction."""
    f_mhz = frequency_hz / 1e6
    if not 400.0 <= f_mhz <= 6000.0:
        raise ValueError("reference levels implemented for 400 MHz-6 GHz")
    return f_mhz / 200.0 if f_mhz <= 2000.0 else 10.0


def local_reference_level_w_per_m2(frequency_hz: float) -> float:
    """Local-exposure tier (ICNIRP 2020 Table 6); this is the reference level
    that pairs with the local 10 g SAR restriction used for the thresholds."""
    f_mhz = frequency_hz / 1e6
    if not 400.0 <= f_mhz <= 6000.0:
        raise ValueError("reference levels implemented for 400 MHz-6 GHz")
    return 0.058 * f_mhz ** 0.86 if f_mhz <= 2000.0 else 40.0


# ----------------------------------------------------------------------------
# Section 2. Four-term Cole-Cole dielectric model (Gabriel 1996)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class ColeCole:
    epsilon_inf: float
    delta_epsilon: tuple
    tau_seconds: tuple
    alpha: tuple
    sigma_ionic: float

    def complex_permittivity(self, frequency_hz: float) -> complex:
        omega = 2.0 * np.pi * frequency_hz
        value = complex(self.epsilon_inf)
        for d, tau, a in zip(self.delta_epsilon, self.tau_seconds, self.alpha):
            if d == 0.0:
                continue
            value += d / (1.0 + (1j * omega * tau) ** (1.0 - a))
        value += self.sigma_ionic / (1j * omega * EPS0)
        return value

    def epsilon_sigma(self, frequency_hz: float) -> tuple:
        eps = self.complex_permittivity(frequency_hz)
        sigma = -eps.imag * 2.0 * np.pi * frequency_hz * EPS0
        return float(eps.real), float(sigma)


GABRIEL_1996 = {
    "bone_cortical": ColeCole(2.5, (10.0, 180.0, 5.0e3, 1.0e5),
                              (13.26e-12, 79.58e-9, 159.15e-6, 15.915e-3),
                              (0.20, 0.20, 0.20, 0.00), 0.0200),
    "fat_not_infiltrated": ColeCole(2.5, (3.0, 15.0, 3.3e4, 1.0e7),
                                    (7.96e-12, 15.92e-9, 159.15e-6, 7.958e-3),
                                    (0.20, 0.10, 0.05, 0.01), 0.0100),
    "muscle": ColeCole(4.0, (50.0, 7000.0, 1.2e6, 2.5e7),
                       (7.23e-12, 353.68e-9, 318.31e-6, 2.274e-3),
                       (0.10, 0.10, 0.10, 0.00), 0.2000),
    "skin_dry": ColeCole(4.0, (32.0, 1100.0, 0.0, 0.0),
                         (7.23e-12, 32.48e-9, 0.0, 0.0),
                         (0.00, 0.20, 0.00, 0.00), 0.0002),
}


def tissue_permittivity(tissue: str, frequency_hz: float) -> tuple:
    return GABRIEL_1996[tissue].epsilon_sigma(frequency_hz)


def scattnlay_index(epsilon_r: float, sigma: float, frequency_hz: float) -> complex:
    """Refractive index in the scattnlay exp(-i w t) convention."""
    omega = 2.0 * np.pi * frequency_hz
    eps = epsilon_r + 1j * sigma / (omega * EPS0)
    value = np.sqrt(eps)
    if value.imag < 0:
        value = -value
    return complex(value)


# ----------------------------------------------------------------------------
# Section 3. Tissue property table and dielectric uncertainty models
# ----------------------------------------------------------------------------
# IT'IS V5.0 average / minimum / maximum values stored as constants (density
# kg/m^3, thermal conductivity W/m/K, perfusion mL/min/kg, metabolic heat
# W/kg). Fields whose minimum equals their maximum are held fixed and never
# sampled. Skin uses the database row "Skin", which does not separate dry and
# wet skin. The fat layer uses the row "SAT (Subcutaneous Fat)", which has the
# same density and thermal conductivity as "Fat (Not Infiltrated)" and, unlike
# that row, nonzero perfusion and metabolic heat. Bone perfusion and metabolic
# heat have single database values (the bone metabolic value 0.154869 W/kg is
# stored rounded to 0.15). Metabolic heat enters the field-on and field-off
# solutions equally and cancels in the temperature rise.
TISSUE_TABLE = {
    "skin_dry": {
        "density": (1109.0, 1100.0, 1125.0),
        "thermal_conductivity": (0.37218, 0.322, 0.498),
        "perfusion": (106.38, 49.0, 174.61),
        "metabolic": (1.6475, 0.75886, 2.7041),
    },
    "fat_not_infiltrated": {
        "density": (911.0, 812.0, 961.0),
        "thermal_conductivity": (0.21145, 0.18, 0.2416),
        "perfusion": (32.709, 19.832, 63.248),
        "metabolic": (0.50657, 0.30714, 0.97952),
    },
    "muscle": {
        "density": (1090.4, 1041.0, 1178.0),
        "thermal_conductivity": (0.49497, 0.4225, 0.56),
        "perfusion": (36.738, 18.537, 94.247),
        "metabolic": (0.90612, 0.45719, 2.3245),
    },
    "bone_cortical": {
        "density": (1908.0, 1800.0, 2100.0),
        "thermal_conductivity": (0.32, 0.30, 0.36),
        "perfusion": (10.0, 10.0, 10.0),
        "metabolic": (0.15, 0.15, 0.15),
    },
}
TISSUE_ORDER = ("skin_dry", "fat_not_infiltrated", "muscle", "bone_cortical")

# Reference (wide) model: fixed frequency-specific log half widths for
# permittivity and conductivity and their copula correlation, chosen to span
# the tissue-to-tissue dispersion of tabulated dielectric properties.
DIELECTRIC_BOX = {
    0.9e9:  {"log_half_width": (0.8004485961787151, 1.1084817387885044),
             "correlation": 0.45890041423113476},
    2.45e9: {"log_half_width": (0.7623722656076064, 0.8737664669908076),
             "correlation": 0.8026235044701961},
    5.8e9:  {"log_half_width": (0.8153160133846504, 0.8669057343868636),
             "correlation": 0.937694764489018},
}

# Within-tissue (narrow) model: Gabriel, Lau and Gabriel (1996, Part II,
# Section 3) report a natural spread of tissue dielectric properties of about
# +/-5-10% above 100 MHz. The upper value is used as a multiplicative bound on
# permittivity and conductivity of every tissue, with the same copula
# correlation as the reference model.
WITHIN_TISSUE_LOG_HALF_WIDTH = float(np.log(1.10))
WITHIN_TISSUE_BOX = {
    f: {"log_half_width": (WITHIN_TISSUE_LOG_HALF_WIDTH, WITHIN_TISSUE_LOG_HALF_WIDTH),
        "correlation": DIELECTRIC_BOX[f]["correlation"]}
    for f in DIELECTRIC_BOX
}

UNCERTAINTY_MODELS = {"reference": DIELECTRIC_BOX, "within_tissue": WITHIN_TISSUE_BOX}
MODEL_LABEL = {"reference": "Reference model", "within_tissue": "Within-tissue model"}
TABLE_MODEL_LABEL = {"reference": "Reference", "within_tissue": "Within-tissue"}
MODEL_TAG = {"reference": "", "within_tissue": "_wt"}

# Geometry: surface-to-core layer thicknesses (skin, fat, muscle), sphere
# outer radius, and semi-infinite bone termination for the slab.
SKIN_M, FAT_M, MUSCLE_M = 0.0015, 0.005, 0.015
SPHERE_RADIUS_M = 0.030

BLOOD_DENSITY = 1050.0
BLOOD_HEAT_CAPACITY = 3617.0
ARTERIAL_T_C = 37.0
AMBIENT_T_C = 24.0
CONVECTION_W_PER_M2_K = 8.0


def perfusion_sink(perfusion_ml_min_kg: float, density: float) -> float:
    """Convert IT'IS mass-specific perfusion to the Pennes sink (W/m^3/K)."""
    volumetric_flow = perfusion_ml_min_kg * density * 1.0e-6 / 60.0
    return BLOOD_DENSITY * BLOOD_HEAT_CAPACITY * volumetric_flow


# ----------------------------------------------------------------------------
# Section 4. Planar slab: transmission-line solution and 10 g averaging
# ----------------------------------------------------------------------------
def _layer_gamma_eta(frequency_hz: float, epsilon_r: float, sigma: float):
    omega = 2.0 * np.pi * frequency_hz
    eps_c = EPS0 * epsilon_r - 1j * sigma / omega
    gamma = 1j * omega * np.sqrt(MU0 * eps_c)
    eta = np.sqrt(1j * omega * MU0 / (sigma + 1j * omega * EPS0 * epsilon_r))
    if gamma.real < 0:
        gamma, eta = -gamma, -eta
    return complex(gamma), complex(eta)


def solve_slab(frequency_hz: float, layers: Sequence[dict],
               incident_power_density: float, sample_step_m: float = 5.0e-5,
               tail_depth_m: float = 0.08):
    """Depth-resolved SAR for a planar multilayer stack (final layer infinite).

    ``layers`` is surface-to-core; each dict has thickness_m (None for the
    semi-infinite terminator), epsilon_r, sigma, density.
    """
    material = [_layer_gamma_eta(frequency_hz, l["epsilon_r"], l["sigma"]) for l in layers]
    gamma = np.array([m[0] for m in material], dtype=np.complex128)
    eta = np.array([m[1] for m in material], dtype=np.complex128)

    zin = np.empty(len(layers), dtype=np.complex128)
    zin[-1] = eta[-1]
    for i in range(len(layers) - 2, -1, -1):
        t = float(layers[i]["thickness_m"])
        th = np.tanh(gamma[i] * t)
        zin[i] = eta[i] * (zin[i + 1] + eta[i] * th) / (eta[i] + zin[i + 1] * th)

    reflection = (zin[0] - ETA0) / (zin[0] + ETA0)
    incident_e = np.sqrt(incident_power_density * ETA0)
    e_front = incident_e * (1.0 + reflection)
    h_front = e_front / zin[0]

    depth_parts, sar_parts, rho_parts = [], [], []
    front, absorbed = 0.0, 0.0
    for i, layer in enumerate(layers):
        forward = 0.5 * (e_front + eta[i] * h_front)
        backward = 0.5 * (e_front - eta[i] * h_front)
        if layer["thickness_m"] is None:
            profile = tail_depth_m
            backward = 0.0j
            power_in = float(np.real(e_front * np.conj(h_front)))
            absorbed += power_in
            e_end = forward * np.exp(-gamma[i] * profile)
            h_end = e_end / eta[i]
        else:
            profile = float(layer["thickness_m"])
            f_end = forward * np.exp(-gamma[i] * profile)
            b_end = backward * np.exp(gamma[i] * profile)
            e_end = f_end + b_end
            h_end = (f_end - b_end) / eta[i]
            power_in = float(np.real(e_front * np.conj(h_front)))
            power_out = float(np.real(e_end * np.conj(h_end)))
            absorbed += power_in - power_out
        count = max(2, int(np.ceil(profile / sample_step_m)) + 1)
        local = np.linspace(0.0, profile, count, endpoint=(layer["thickness_m"] is None))
        e_field = forward * np.exp(-gamma[i] * local) + backward * np.exp(gamma[i] * local)
        sar = layer["sigma"] * np.abs(e_field) ** 2 / layer["density"]
        depth_parts.append(front + local)
        sar_parts.append(sar.astype(float))
        rho_parts.append(np.full(local.shape, layer["density"], dtype=float))
        if layer["thickness_m"] is not None:
            front += profile
            e_front, h_front = e_end, h_end

    transmitted = incident_power_density * (1.0 - abs(reflection) ** 2)
    energy_error = abs(absorbed - transmitted) / max(abs(transmitted), 1e-300)
    return (np.concatenate(depth_parts), np.concatenate(sar_parts),
            np.concatenate(rho_parts), float(energy_error))


def _cumulative_integral(x, y):
    from scipy.integrate import cumulative_trapezoid
    cumulative = np.concatenate(([0.0], cumulative_trapezoid(y, x)))

    def integral(a, b):
        a = min(max(float(a), float(x[0])), float(x[-1]))
        b = min(max(float(b), float(x[0])), float(x[-1]))
        return float(np.interp(b, x, cumulative) - np.interp(a, x, cumulative))
    return integral


def planar_peak_10g(depth, sar, density, target_mass_kg=TARGET_MASS_KG, candidates=500):
    """Maximum SAR in a laterally uniform cube of the requested tissue mass."""
    from scipy.optimize import brentq
    mass_line = _cumulative_integral(depth, density)
    power_line = _cumulative_integral(depth, density * sar)
    end = float(depth[-1])
    starts = np.linspace(float(depth[0]), end, candidates, endpoint=False)
    best = -np.inf
    for start in starts:
        max_side = end - start
        if max_side <= 0:
            continue
        if max_side ** 2 * mass_line(start, end) < target_mass_kg:
            continue

        def residual(side):
            return side ** 2 * mass_line(start, start + side) - target_mass_kg
        side = float(brentq(residual, np.finfo(float).eps, max_side, xtol=1e-12, rtol=1e-11))
        avg = power_line(start, start + side) / mass_line(start, start + side)
        if avg > best:
            best = avg
    if not np.isfinite(best):
        raise RuntimeError("no admissible slab averaging cube")
    return float(best)


# ----------------------------------------------------------------------------
# Section 5. Multilayer sphere: exact Mie fields and 62704 10 g averaging
# ----------------------------------------------------------------------------
@contextlib.contextmanager
def _suppress_native_stdout():
    """Redirect native stdout without assuming a Unix C runtime."""
    c_runtime = None
    try:
        c_runtime = ctypes.CDLL("msvcrt" if os.name == "nt" else None)
        c_runtime.fflush(None)
    except (OSError, TypeError, AttributeError):
        c_runtime = None
    sys.stdout.flush()
    saved = os.dup(1)
    null = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null, 1)
        yield
        if c_runtime is not None:
            c_runtime.fflush(None)
    finally:
        os.dup2(saved, 1)
        os.close(saved)
        os.close(null)


def _mie_inputs(layers, frequency_hz):
    k0 = 2.0 * np.pi * frequency_hz / C0
    size = k0 * np.array([l["outer_radius_m"] for l in layers], dtype=float)
    index = np.array([scattnlay_index(l["epsilon_r"], l["sigma"], frequency_hz)
                      for l in layers], dtype=np.complex128)
    return size, index


def sphere_qabs(layers, frequency_hz):
    from scattnlay import scattnlay as _sc
    size, index = _mie_inputs(layers, frequency_hz)
    terms, qext, qsca, qabs, *_ = _sc(size, index)
    return float(qext), float(qabs)


def _field_relative(layers, frequency_hz, xyz):
    from scattnlay import fieldnlay
    size, index = _mie_inputs(layers, frequency_hz)
    k0 = 2.0 * np.pi * frequency_hz / C0
    with _suppress_native_stdout():
        _, field_e, _ = fieldnlay(size, index, k0 * xyz[:, 0], k0 * xyz[:, 1], k0 * xyz[:, 2])
    return np.asarray(field_e)


def _layer_index_at(layers, xyz):
    radius = np.linalg.norm(xyz, axis=1)
    outer = np.array([l["outer_radius_m"] for l in layers], dtype=float)
    idx = np.searchsorted(outer, radius, side="left")
    return np.clip(idx, 0, len(layers) - 1)


def sphere_sar_at(layers, frequency_hz, xyz, incident_power_density):
    idx = _layer_index_at(layers, xyz)
    field = _field_relative(layers, frequency_hz, xyz)
    e_rms2 = incident_power_density * ETA0 * np.sum(np.abs(field) ** 2, axis=1)
    sigma = np.array([layers[i]["sigma"] for i in idx], dtype=float)
    density = np.array([layers[i]["density"] for i in idx], dtype=float)
    return sigma * e_rms2 / density


def absorption_balance_error(layers, frequency_hz, radial_order=18, polar_order=20, azimuth=32):
    """Relative error between volume-integrated absorption and Mie Qabs."""
    from numpy.polynomial.legendre import leggauss
    mu, w_mu = leggauss(polar_order)
    phi = 2.0 * np.pi * np.arange(azimuth) / azimuth
    w_phi = 2.0 * np.pi / azimuth
    total, inner = 0.0, 0.0
    for layer in layers:
        xi, wi = leggauss(radial_order)
        half = 0.5 * (layer["outer_radius_m"] - inner)
        mid = 0.5 * (layer["outer_radius_m"] + inner)
        r = mid + half * xi
        w_r = half * wi
        rr, mm, pp = np.meshgrid(r, mu, phi, indexing="ij")
        sin_t = np.sqrt(np.maximum(0.0, 1.0 - mm * mm))
        pts = np.column_stack(((rr * sin_t * np.cos(pp)).ravel(),
                               (rr * sin_t * np.sin(pp)).ravel(),
                               (rr * mm).ravel()))
        field = _field_relative(layers, frequency_hz, pts)
        e2 = np.sum(np.abs(field) ** 2, axis=1).reshape(rr.shape)
        weights = w_r[:, None, None] * w_mu[None, :, None] * w_phi * rr ** 2
        total += layer["sigma"] * float(np.sum(e2 * weights))
        inner = layer["outer_radius_m"]
    area = np.pi * layers[-1]["outer_radius_m"] ** 2
    qabs_integral = ETA0 * total / area
    _, qabs_mie = sphere_qabs(layers, frequency_hz)
    return abs(qabs_integral - qabs_mie) / abs(qabs_mie)


def voxelize_sphere(layers, frequency_hz, incident_power_density, spacing_m, chunk=200000):
    radius = layers[-1]["outer_radius_m"]
    count = int(np.ceil(2.0 * radius / spacing_m))
    if count % 2:
        count += 1
    spacing = 2.0 * radius / count
    edge = -radius
    axis = edge + (np.arange(count) + 0.5) * spacing
    xx, yy, zz = np.meshgrid(axis, axis, axis, indexing="ij")
    inside = xx * xx + yy * yy + zz * zz <= radius * radius
    flat = np.flatnonzero(inside.ravel())
    pts = np.column_stack((xx.ravel()[flat], yy.ravel()[flat], zz.ravel()[flat]))
    sar = np.zeros(count ** 3)
    rho = np.zeros(count ** 3)
    for s in range(0, len(pts), chunk):
        batch = pts[s:s + chunk]
        bidx = flat[s:s + chunk]
        sar[bidx] = sphere_sar_at(layers, frequency_hz, batch, incident_power_density)
        mat = _layer_index_at(layers, batch)
        rho[bidx] = np.array([layers[i]["density"] for i in mat], dtype=float)
    return edge, spacing, rho.reshape((count,) * 3), sar.reshape((count,) * 3)


class _VoxelIntegrals:
    """Continuous box integrals of piecewise-constant voxel data (62704)."""

    def __init__(self, edge, spacing, rho, sar):
        self.edge = np.array([edge, edge, edge], dtype=float)
        self.spacing = spacing
        self.shape = np.array(rho.shape, dtype=int)
        h3 = spacing ** 3
        tissue = (rho > 0).astype(float)
        self.mass = self._prefix(rho * h3)
        self.power = self._prefix(rho * sar * h3)
        self.tissue_volume = self._prefix(tissue * h3)

    @staticmethod
    def _prefix(v):
        c = v.cumsum(0).cumsum(1).cumsum(2)
        return np.pad(c, ((1, 0), (1, 0), (1, 0)))

    def _cum(self, pts, prefix):
        counts = self.shape
        u = (pts - self.edge) / self.spacing
        u = np.clip(u, 0.0, counts.astype(float))
        base = np.floor(u).astype(int)
        frac = u - base
        upper = base == counts
        base = np.where(upper, counts - 1, base)
        frac = np.where(upper, 1.0, frac)
        out = np.zeros(len(pts))
        for dx, dy, dz in product((0, 1), repeat=3):
            w = ((frac[:, 0] if dx else 1 - frac[:, 0]) *
                 (frac[:, 1] if dy else 1 - frac[:, 1]) *
                 (frac[:, 2] if dz else 1 - frac[:, 2]))
            out += w * prefix[base[:, 0] + dx, base[:, 1] + dy, base[:, 2] + dz]
        return out

    def box(self, lower, upper, prefix):
        out = np.zeros(len(lower))
        for bx, by, bz in product((0, 1), repeat=3):
            pt = np.column_stack((upper[:, 0] if bx else lower[:, 0],
                                  upper[:, 1] if by else lower[:, 1],
                                  upper[:, 2] if bz else lower[:, 2]))
            sign = -1.0 if (3 - (bx + by + bz)) % 2 else 1.0
            out += sign * self._cum(pt, prefix)
        return np.maximum(out, 0.0)

    def bounds(self, centers, side, orientation):
        half = 0.5 * side[:, None]
        lower = centers - half
        upper = centers + half
        if orientation != 0:
            axis = (abs(orientation) - 1) // 2
            if orientation % 2 == 0:
                lower[:, axis] = centers[:, axis]
                upper[:, axis] = centers[:, axis] + side
            else:
                lower[:, axis] = centers[:, axis] - side
                upper[:, axis] = centers[:, axis]
        return lower, upper

    def solve_side(self, centers, target_mass, orientation, iters=22):
        n = len(centers)
        lo_side = np.zeros(n)
        span = float(np.max(self.edge + self.spacing * self.shape - self.edge))
        hi_side = np.full(n, 2.0 * span)
        lo, hi = self.bounds(centers, hi_side, orientation)
        possible = self.box(lo, hi, self.mass) >= target_mass * (1 - 1e-12)
        for _ in range(iters):
            mid = 0.5 * (lo_side + hi_side)
            lo, hi = self.bounds(centers, mid, orientation)
            enough = self.box(lo, hi, self.mass) >= target_mass
            hi_side = np.where(enough, mid, hi_side)
            lo_side = np.where(enough, lo_side, mid)
        side = hi_side
        lo, hi = self.bounds(centers, side, orientation)
        mass = self.box(lo, hi, self.mass)
        power = self.box(lo, hi, self.power)
        vol = self.box(lo, hi, self.tissue_volume)
        for arr in (mass, power, vol, side):
            arr[~possible] = np.nan
        return side, mass, power, vol, lo, hi

    def faces_touch(self, lower, upper):
        probe = self.spacing
        touched = np.ones(len(lower), dtype=bool)
        for axis in range(3):
            lh = upper.copy(); lh[:, axis] = np.minimum(upper[:, axis], lower[:, axis] + probe)
            hl = lower.copy(); hl[:, axis] = np.maximum(lower[:, axis], upper[:, axis] - probe)
            touched &= (self.box(lower, lh, self.tissue_volume) > 1e-18) & \
                       (self.box(hl, upper, self.tissue_volume) > 1e-18)
        return touched


def sphere_peak_10g(edge, spacing, rho, sar, target_mass_kg=TARGET_MASS_KG, batch=20000):
    """Peak 10 g spatial-average SAR using an IEC/IEEE 62704-1-style search."""
    integ = _VoxelIntegrals(edge, spacing, rho, sar)
    body = np.argwhere(rho > 0)
    axes = [edge + (np.arange(n) + 0.5) * spacing for n in rho.shape]
    centers = np.column_stack((axes[0][body[:, 0]], axes[1][body[:, 1]], axes[2][body[:, 2]]))
    best = -np.inf
    for s in range(0, len(centers), batch):
        c = centers[s:s + batch]
        side, mass, power, vol, lo, hi = integ.solve_side(c, target_mass_kg, 0)
        cube_vol = side ** 3
        background = 1.0 - vol / cube_vol
        valid = np.isfinite(side) & (background < 0.10 + 1e-10) & integ.faces_touch(lo, hi)
        if np.any(valid):
            avg = power[valid] / mass[valid]
            best = max(best, float(np.nanmax(avg)))
    if not np.isfinite(best):
        raise RuntimeError("no admissible sphere averaging cube")
    return float(best)


# ----------------------------------------------------------------------------
# Section 6. Steady-state Pennes bioheat (planar and radial)
# ----------------------------------------------------------------------------
def _harmonic(a, b):
    return 2.0 * a * b / (a + b)


def solve_planar_pennes(x, k, wb, q, deep_t=37.0):
    from scipy.sparse import diags
    from scipy.sparse.linalg import spsolve
    n = len(x)
    dx = float(np.mean(np.diff(x)))
    k_face = _harmonic(k[:-1], k[1:])
    west = np.zeros(n); east = np.zeros(n)
    west[1:] = k_face / dx; east[:-1] = k_face / dx
    g_surface = 1.0 / (0.5 * dx / k[0] + 1.0 / CONVECTION_W_PER_M2_K)
    g_deep = k[-1] / (0.5 * dx)
    diag = west + east + wb * dx
    diag[0] += g_surface; diag[-1] += g_deep
    rhs = (wb * ARTERIAL_T_C + q) * dx
    rhs[0] += g_surface * AMBIENT_T_C
    rhs[-1] += g_deep * deep_t
    m = diags((-west[1:], diag, -east[:-1]), offsets=(-1, 0, 1), format="csc")
    return np.asarray(spsolve(m, rhs))


def solve_radial_pennes(r, k, wb, q):
    from scipy.sparse import diags
    from scipy.sparse.linalg import spsolve
    n = len(r)
    dr = float(np.mean(np.diff(r)))
    faces = np.arange(n + 1) * dr
    area = 4.0 * np.pi * faces ** 2
    volume = (4.0 * np.pi / 3.0) * (faces[1:] ** 3 - faces[:-1] ** 3)
    k_face = _harmonic(k[:-1], k[1:])
    west = np.zeros(n); east = np.zeros(n)
    west[1:] = k_face * area[1:-1] / dr; east[:-1] = k_face * area[1:-1] / dr
    g_surface = area[-1] / (0.5 * dr / k[-1] + 1.0 / CONVECTION_W_PER_M2_K)
    diag = west + east + wb * volume
    diag[-1] += g_surface
    rhs = (wb * ARTERIAL_T_C + q) * volume
    rhs[-1] += g_surface * AMBIENT_T_C
    m = diags((-west[1:], diag, -east[:-1]), offsets=(-1, 0, 1), format="csc")
    return np.asarray(spsolve(m, rhs))


def solve_voxel_pennes_delta(spacing, rho, sar, k_vox, wb_vox):
    """Steady-state temperature rise on the voxel grid (3-D finite volume).

    Because the Pennes equation is linear, the rise Delta T above the RF-off
    baseline satisfies  -div(k grad dT) + wb dT = rho * SAR  with the ambient
    and arterial terms cancelled; the surface convection enters as a
    homogeneous Robin condition on tissue-air faces. Face conductances use the
    harmonic mean; the air-side conductance is the series combination of the
    half-voxel conduction path and the convection film.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve
    shape = rho.shape
    h = float(spacing)
    tissue = rho > 0
    index = -np.ones(shape, dtype=np.int64)
    ids = np.flatnonzero(tissue.ravel())
    index.ravel()[ids] = np.arange(len(ids))
    n = len(ids)
    diag = wb_vox.ravel()[ids] * h ** 3
    rhs = (rho * sar).ravel()[ids] * h ** 3
    rows, cols, vals = [], [], []
    for axis in range(3):
        for step in (-1, 1):
            nb = np.full(shape, -1, dtype=np.int64)
            src = [slice(None)] * 3
            dst = [slice(None)] * 3
            if step == 1:
                src[axis] = slice(0, -1); dst[axis] = slice(1, None)
            else:
                src[axis] = slice(1, None); dst[axis] = slice(0, -1)
            nb[tuple(src)] = index[tuple(dst)]
            nb_f = nb.ravel()[ids]
            k_i = k_vox.ravel()[ids]
            inside = nb_f >= 0
            k_j = np.zeros(n)
            k_j[inside] = k_vox.ravel()[ids[nb_f[inside]]]
            g_in = np.zeros(n)
            g_in[inside] = 2.0 * k_i[inside] * k_j[inside] / (k_i[inside] + k_j[inside]) * h
            g_out = np.zeros(n)
            outside = ~inside
            g_out[outside] = h * h / (0.5 * h / k_i[outside] + 1.0 / CONVECTION_W_PER_M2_K)
            diag += g_in + g_out
            rr = np.arange(n)[inside]
            rows.append(rr); cols.append(nb_f[inside]); vals.append(-g_in[inside])
    rows = np.concatenate(rows + [np.arange(n)])
    cols = np.concatenate(cols + [np.arange(n)])
    vals = np.concatenate(vals + [diag])
    m = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsc()
    dT = spsolve(m, rhs)
    out = np.zeros(shape)
    out.ravel()[ids] = dT
    return out


def sphere_shell_sar(layers, frequency_hz, radii, incident_power_density,
                     polar_order=10, azimuth=20):
    from numpy.polynomial.legendre import leggauss
    mu, w_mu = leggauss(polar_order)
    phi = 2.0 * np.pi * np.arange(azimuth) / azimuth
    rr, mm, pp = np.meshgrid(radii, mu, phi, indexing="ij")
    sin_t = np.sqrt(np.maximum(0.0, 1.0 - mm * mm))
    pts = np.column_stack(((rr * sin_t * np.cos(pp)).ravel(),
                           (rr * sin_t * np.sin(pp)).ravel(),
                           (rr * mm).ravel()))
    sar = sphere_sar_at(layers, frequency_hz, pts, incident_power_density).reshape(rr.shape)
    return 0.5 * np.sum(w_mu[None, :, None] * sar, axis=(1, 2)) / azimuth


# ----------------------------------------------------------------------------
# Section 7. Material parameterization and forward models
# ----------------------------------------------------------------------------
MATERIAL_FIELDS = ("epsilon_r", "sigma", "density", "thermal_conductivity",
                   "perfusion", "metabolic")


def nominal_materials(frequency_hz):
    mats = {}
    for tissue in TISSUE_ORDER:
        eps, sigma = tissue_permittivity(tissue, frequency_hz)
        row = TISSUE_TABLE[tissue]
        mats[tissue] = {
            "epsilon_r": eps, "sigma": sigma,
            "density": row["density"][0],
            "thermal_conductivity": row["thermal_conductivity"][0],
            "perfusion": row["perfusion"][0],
            "metabolic": row["metabolic"][0],
        }
    return mats


def active_thermal_keys():
    """(tissue, field) pairs sampled over documented ranges (min < max)."""
    keys = []
    for field in ("density", "thermal_conductivity", "perfusion", "metabolic"):
        for tissue in TISSUE_ORDER:
            _nom, mn, mx = TISSUE_TABLE[tissue][field]
            if mx > mn:
                keys.append((tissue, field))
    return keys


ACTIVE_THERMAL = active_thermal_keys()


def _slab_layers(mats):
    spec = [("skin_dry", SKIN_M), ("fat_not_infiltrated", FAT_M),
            ("muscle", MUSCLE_M), ("bone_cortical", None)]
    return [{"thickness_m": t, "epsilon_r": mats[n]["epsilon_r"],
             "sigma": mats[n]["sigma"], "density": mats[n]["density"], "name": n}
            for n, t in spec]


def _sphere_layers(mats, radius=SPHERE_RADIUS_M):
    bone = radius - SKIN_M - FAT_M - MUSCLE_M
    muscle = radius - SKIN_M - FAT_M
    fat = radius - SKIN_M
    spec = [("bone_cortical", bone), ("muscle", muscle),
            ("fat_not_infiltrated", fat), ("skin_dry", radius)]
    return [{"outer_radius_m": r, "epsilon_r": mats[n]["epsilon_r"],
             "sigma": mats[n]["sigma"], "density": mats[n]["density"], "name": n}
            for n, r in spec]


def slab_forward(mats, frequency_hz, endpoint="sar"):
    layers = _slab_layers(mats)
    depth, sar, rho, _ = solve_slab(frequency_hz, layers, ANCHOR_S0_W_PER_M2)
    sar_10g = planar_peak_10g(depth, sar, rho)
    if endpoint == "sar":
        return np.array([sar_10g])
    domain = SKIN_M + FAT_M + MUSCLE_M + 0.08
    n = int(np.ceil(domain / 1e-4))
    x = (np.arange(n) + 0.5) * (domain / n)
    edges = np.array([SKIN_M, SKIN_M + FAT_M, SKIN_M + FAT_M + MUSCLE_M])
    names = ["skin_dry", "fat_not_infiltrated", "muscle", "bone_cortical"]
    idx = np.searchsorted(edges, x, side="right")
    cond = np.array([mats[names[i]]["thermal_conductivity"] for i in idx])
    wb = np.array([perfusion_sink(mats[names[i]]["perfusion"], mats[names[i]]["density"]) for i in idx])
    met = np.array([mats[names[i]]["metabolic"] * mats[names[i]]["density"] for i in idx])
    dens = np.array([mats[names[i]]["density"] for i in idx])
    sar_x = np.interp(x, depth, sar)
    base = solve_planar_pennes(x, cond, wb, met)
    heat = solve_planar_pennes(x, cond, wb, met + dens * sar_x)
    return np.array([sar_10g, float(np.max(heat - base))])


def _sphere_voxel_thermal_fields(layers, mats, edge, spacing, shape):
    """Per-voxel thermal conductivity and Pennes sink from the layer map."""
    count = shape[0]
    axis = edge + (np.arange(count) + 0.5) * spacing
    xx, yy, zz = np.meshgrid(axis, axis, axis, indexing="ij")
    pts = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    radius = np.linalg.norm(pts, axis=1)
    outer = np.array([l["outer_radius_m"] for l in layers], dtype=float)
    li = np.clip(np.searchsorted(outer, radius, side="left"), 0, len(layers) - 1)
    names = [layers[i]["name"] for i in range(len(layers))]
    k = np.array([mats[names[i]]["thermal_conductivity"] for i in li]).reshape(shape)
    wb = np.array([perfusion_sink(mats[names[i]]["perfusion"],
                                  mats[names[i]]["density"]) for i in li]).reshape(shape)
    return k, wb


def sphere_forward(mats, frequency_hz, endpoint="sar", voxel_m=0.002):
    layers = _sphere_layers(mats)
    edge, spacing, rho, sar = voxelize_sphere(layers, frequency_hz, ANCHOR_S0_W_PER_M2, voxel_m)
    sar_10g = sphere_peak_10g(edge, spacing, rho, sar)
    if endpoint == "sar":
        return np.array([sar_10g])
    k_vox, wb_vox = _sphere_voxel_thermal_fields(layers, mats, edge, spacing, rho.shape)
    k_vox = np.where(rho > 0, k_vox, 0.0)
    wb_vox = np.where(rho > 0, wb_vox, 0.0)
    dT = solve_voxel_pennes_delta(spacing, rho, sar, k_vox, wb_vox)
    return np.array([sar_10g, float(dT.max())])


# ----------------------------------------------------------------------------
# Section 8. Tissue-property sampling (correlated dielectric log box)
# ----------------------------------------------------------------------------
def sample_materials(frequency_hz, n, seed, endpoint="sar", model="reference"):
    """Draw n material realizations from the selected uncertainty model.

    Permittivity and conductivity of each tissue are perturbed inside the
    frequency-specific dielectric log box with a Gaussian copula (correlation
    rho), giving uniform log marginals. Density, and for the thermal endpoint
    the thermal fields, are independent uniforms over their documented ranges.
    """
    from scipy.stats import norm
    rng = np.random.default_rng(seed)
    nom = nominal_materials(frequency_hz)
    box = UNCERTAINTY_MODELS[model][frequency_hz]
    half_e, half_s = box["log_half_width"]
    rho = float(np.clip(box["correlation"], -0.995, 0.995))

    samples = [dict((t, dict(nom[t])) for t in TISSUE_ORDER) for _ in range(n)]
    for tissue in TISSUE_ORDER:
        u_e = np.clip(rng.random(n), 1e-12, 1 - 1e-12)
        u_s = np.clip(rng.random(n), 1e-12, 1 - 1e-12)
        z_e = norm.ppf(u_e)
        z_s = rho * z_e + np.sqrt(1 - rho * rho) * norm.ppf(u_s)
        off_e = (2.0 * norm.cdf(z_e) - 1.0) * half_e
        off_s = (2.0 * norm.cdf(z_s) - 1.0) * half_s
        e_mult = np.exp(off_e)
        s_mult = np.exp(off_s)
        for i in range(n):
            samples[i][tissue]["epsilon_r"] = nom[tissue]["epsilon_r"] * e_mult[i]
            samples[i][tissue]["sigma"] = nom[tissue]["sigma"] * s_mult[i]

    fields = ("density",) if endpoint == "sar" else \
             ("density", "thermal_conductivity", "perfusion", "metabolic")
    for tissue in TISSUE_ORDER:
        for field in fields:
            nomv, mn, mx = TISSUE_TABLE[tissue][field]
            if mx <= mn:
                continue
            draw = rng.uniform(mn, mx, size=n)
            for i in range(n):
                samples[i][tissue][field] = draw[i]
    return samples


# ----------------------------------------------------------------------------
# Section 9. Monte Carlo driver with caching
# ----------------------------------------------------------------------------
def monte_carlo(case, frequency_hz, n, seed, endpoint, voxel_m, cache_dir,
                model="reference", verbose=True):
    """Evaluate the forward model over n draws; store and reuse endpoint samples."""
    vtag = f"_v{int(round(voxel_m * 1e4))}" if case == "sphere" else ""
    if endpoint != "sar":
        vtag += "_t3d"
    tag = (f"{case}_{FREQUENCY_LABEL[frequency_hz].replace(' ', '').replace('.', 'p')}"
           f"_{endpoint}_N{n}{vtag}_s{seed}{MODEL_TAG[model]}")
    path = os.path.join(cache_dir, tag + ".npz")
    if os.path.exists(path):
        data = np.load(path)
        return data["outputs"], data["nominal"]
    mats = sample_materials(frequency_hz, n, seed, endpoint=endpoint, model=model)
    forward = slab_forward if case == "slab" else sphere_forward
    nom = nominal_materials(frequency_hz)
    if case == "slab":
        nominal = forward(nom, frequency_hz, endpoint=endpoint)
    else:
        nominal = forward(nom, frequency_hz, endpoint=endpoint, voxel_m=voxel_m)
    outputs = np.empty((n, len(nominal)))
    t0 = time.time()
    for i in range(n):
        if case == "slab":
            outputs[i] = forward(mats[i], frequency_hz, endpoint=endpoint)
        else:
            outputs[i] = forward(mats[i], frequency_hz, endpoint=endpoint, voxel_m=voxel_m)
        if verbose and (i + 1) % max(1, n // 10) == 0:
            rate = (time.time() - t0) / (i + 1)
            print(f"    [{model} {case} {FREQUENCY_LABEL[frequency_hz]} {endpoint}] "
                  f"{i + 1}/{n}  {rate:.2f}s/eval", flush=True)
    np.savez_compressed(path, outputs=outputs, nominal=nominal)
    return outputs, nominal


# ----------------------------------------------------------------------------
# Section 10. Threshold layer: quantile, tolerance, tail and robust bounds
# ----------------------------------------------------------------------------
def empirical_var(y, alpha=ALPHA):
    """Conservative empirical (1-alpha) upper quantile."""
    return float(np.quantile(y, 1.0 - alpha, method="higher"))


def cvar_upper(y, alpha=ALPHA):
    """Upper-tail CVaR: mean of the worst alpha fraction (Rockafellar-Uryasev)."""
    y = np.asarray(y, dtype=float)
    tau = np.quantile(y, 1.0 - alpha, method="lower")
    excess = np.maximum(y - tau, 0.0).mean()
    return float(tau + excess / alpha)


def conformal_var(y, alpha=ALPHA):
    """Split-conformal upper quantile with finite-sample marginal validity.

    Returns the order statistic at rank ceil((n+1)(1-alpha)); if that rank
    exceeds n the bound is +inf (n too small for the requested level).
    """
    y = np.sort(np.asarray(y, dtype=float))
    n = len(y)
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    if rank > n:
        return float("inf")
    return float(y[rank - 1])


def tolerance_var(y, alpha=ALPHA, delta=DELTA):
    """Nonparametric one-sided upper tolerance limit (content 1-alpha, conf 1-delta).

    The j-th largest order statistic exceeds the true (1-alpha) quantile with
    probability P(Binom(n, alpha) >= j) = 1 - F(j-1). The tightest valid upper
    tolerance limit is the j-th largest with the largest j for which
        F(j-1; n, alpha) <= delta,
    i.e. the smallest order statistic that lies above the (1-alpha) quantile
    with confidence at least 1-delta. This limit is at or above the empirical
    (1-alpha) quantile, so its incident power density threshold is lower.
    """
    from scipy.stats import binom
    y = np.sort(np.asarray(y, dtype=float))
    n = len(y)
    best = float("nan")
    for j in range(1, n + 1):
        if binom.cdf(j - 1, n, alpha) <= delta:
            best = float(y[n - j])   # j-th largest; decreasing (tighter) in j
        else:
            break
    return best


def moment_dro_var(y, alpha=ALPHA):
    """Cantelli worst-case (1-alpha) quantile given matched mean and variance.

    sup_{E=mu,Var=s^2} P(Y>t) = s^2/(s^2+(t-mu)^2) for t>mu; solving = alpha
    gives t = mu + s*sqrt((1-alpha)/alpha).
    """
    mu = float(np.mean(y))
    s = float(np.std(y, ddof=1))
    return mu + s * np.sqrt((1.0 - alpha) / alpha)


def wasserstein_cvar(y, epsilon, alpha=ALPHA):
    """Worst-case upper-tail CVaR over a Wasserstein-1 ball of radius epsilon.

    For a 1-Lipschitz loss (identity in SAR units), the exact reformulation is
    CVaR_alpha^robust = CVaR_alpha(P_hat) + epsilon/alpha.
    """
    return cvar_upper(y, alpha) + epsilon / alpha


def incident_power_threshold(quantile_value, limit):
    """Incident power density S so that the metric reaches the limit at quantile."""
    quantile_value = float(quantile_value)
    if np.isnan(quantile_value) or quantile_value <= 0.0:
        return float("nan")
    if np.isposinf(quantile_value):
        # A +infinity conformal bound gives the valid but trivial zero
        # threshold; unavailable statistics use NaN instead.
        return 0.0
    if np.isneginf(quantile_value):
        return float("nan")
    return ANCHOR_S0_W_PER_M2 * limit / quantile_value


def gaussian_var(y, alpha=ALPHA):
    """Gaussian plug-in (1-alpha) quantile: mean + z_{1-alpha} * std.

    This is the quantile a linearization / normal-propagation method would use.
    For right-skewed SAR it underestimates the true upper quantile.
    """
    from scipy.stats import norm
    return float(np.mean(y) + norm.ppf(1.0 - alpha) * np.std(y, ddof=1))


def coverage_experiment(y, alpha=ALPHA, n_cal=50, splits=200, seed=SEED):
    """Finite-sample exceedance of the Gaussian plug-in vs the conformal limit.

    Each split fits a threshold on a small calibration subset of size ``n_cal``
    and measures the exceedance fraction on the large held-out remainder. A
    distribution-free conformal limit should keep exceedance at or below the
    target alpha; a Gaussian plug-in under-covers for skewed SAR.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    n_cal = min(n_cal, n // 2)
    rng = np.random.default_rng(seed)
    gauss_exc, conf_exc = [], []
    for _ in range(splits):
        perm = rng.permutation(n)
        cal, test = y[perm[:n_cal]], y[perm[n_cal:]]
        gauss_exc.append(float(np.mean(test > gaussian_var(cal, alpha))))
        conf_exc.append(float(np.mean(test > conformal_var(cal, alpha))))
    return np.array(gauss_exc), np.array(conf_exc)


def wasserstein_cvar_transport_check(y, epsilon, alpha=ALPHA):
    """Constructive worst-case CVaR by explicit optimal transport of tail mass.

    For the CVaR integrand (Y-tau)_+/alpha, worst-case W1 transport shifts the
    mass above the empirical VaR upward. The analytic increment is exactly
    epsilon/alpha; this recomputes CVaR on the transported sample as a check.
    """
    y = np.asarray(y, dtype=float)
    tau = float(np.quantile(y, 1.0 - alpha, method="lower"))
    above = y > tau
    frac = float(np.mean(above))
    if frac <= 0:
        return cvar_upper(y, alpha)
    shifted = y.copy()
    shifted[above] += epsilon / frac      # spend the full W1 budget on tail mass
    return cvar_upper(shifted, alpha)


# ----------------------------------------------------------------------------
# Section 11b. Calibration-size and distribution-shift experiments
# ----------------------------------------------------------------------------
def calibration_size_experiment(y, alpha=ALPHA, delta=DELTA,
                                sizes=(25, 50, 100, 200, 400, 800, 1600),
                                reps=120, seed=SEED):
    """Incident power density threshold and held-out exceedance versus
    calibration size.

    For alpha = 0.05 the empirical and conformal ranks coincide when n is a
    multiple of 20 or one less, and otherwise the conformal rank is one higher.
    Each repetition draws a calibration subset without replacement and
    evaluates exceedance on the remaining draws; tolerance limits that do not
    exist at a size are left out of the averages.
    """
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed + 7)
    out = []
    for n_cal in sizes:
        if n_cal >= len(y):
            continue
        rec = {"n_cal": n_cal}
        acc = {k: [] for k in ("S_emp", "S_conf", "S_tol",
                               "exc_emp", "exc_conf", "exc_tol")}
        for _ in range(reps):
            perm = rng.permutation(len(y))
            cal, test = y[perm[:n_cal]], y[perm[n_cal:]]
            emp = float(np.quantile(cal, 1.0 - alpha, method="higher"))
            conf = conformal_var(cal, alpha)
            tol = tolerance_var(cal, alpha, delta)
            for key, q in (("emp", emp), ("conf", conf), ("tol", tol)):
                acc["S_" + key].append(incident_power_threshold(q, SAR_LIMIT_W_PER_KG))
                acc["exc_" + key].append(
                    float(np.mean(test > q)) if np.isfinite(q) else float("nan"))
        for k, v in acc.items():
            arr = np.array(v)
            arr = arr[np.isfinite(arr)]
            rec[k + "_mean"] = float(arr.mean()) if arr.size else float("nan")
            rec[k + "_p90"] = float(np.quantile(arr, 0.9)) if arr.size else float("nan")
        out.append(rec)
    return out


def _w1_empirical(u, v, grid=2000):
    """W1 distance between two empirical distributions via quantile integral."""
    g = (np.arange(1, grid) / grid)
    return float(np.mean(np.abs(np.quantile(u, g) - np.quantile(v, g))))


def shift_experiment(y, alpha=ALPHA, n_shift=4000, seed=SEED):
    """Behavior of the bounds when the sampling distribution moves.

    The case sample fixes a conformal threshold and an empirical CVaR. For each
    transformation, a separate set of n_shift endpoints is resampled with
    replacement from the case sample and transformed. The output records the
    approximate W1 distance to the case sample, the exceedance of the conformal
    threshold (whose exchangeability premise no longer holds), the CVaR of the
    transformed set, and the CVaR increase as a percentage of the worst-case
    increase W1/alpha of the Wasserstein bound.
    """
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed + 11)
    conf = conformal_var(y, alpha)
    cvar_ref = cvar_upper(y, alpha)
    shifts = [
        ("Scaling by 1.05", lambda v: v * 1.05),
        ("Scaling by 1.15", lambda v: v * 1.15),
        ("Upper decile scaled by 1.5", lambda v: np.where(v > np.quantile(v, 0.9), v * 1.5, v)),
        ("Deviations scaled by 1.2", lambda v: v.mean() + (v - v.mean()) * 1.2),
    ]
    out = []
    for name, fn in shifts:
        z = fn(rng.choice(y, n_shift, replace=True))
        w1 = _w1_empirical(z, y)
        out.append({
            "shift": name,
            "w1": w1,
            "w1_pct_nominal": float("nan"),
            "conformal_exceedance": float(np.mean(z > conf)),
            "conformal_exceed_count": int(np.sum(z > conf)),
            "cvar_shifted": cvar_upper(z, alpha),
            "wasserstein_bound": cvar_ref + w1 / alpha,
            "bound_holds": bool(cvar_ref + w1 / alpha >= cvar_upper(z, alpha)),
            "share_pct": float(100.0 * (cvar_upper(z, alpha) - cvar_ref) / (w1 / alpha)),
        })
    return out


def voxel_convergence_study(frequency_hz, voxels=(0.00375, 0.003, 0.0025, 0.002,
                                                  0.0015, 0.00125)):
    """Nominal-property peak 10 g SAR of the sphere versus voxel size."""
    nom = nominal_materials(frequency_hz)
    layers = _sphere_layers(nom)
    rows = []
    prev = None
    for vm in voxels:
        edge, sp, rho, sar = voxelize_sphere(layers, frequency_hz,
                                             ANCHOR_S0_W_PER_M2, vm)
        val = sphere_peak_10g(edge, sp, rho, sar)
        rows.append({"voxel_mm": sp * 1e3, "cells": int(rho.shape[0]),
                     "peak_10g": val,
                     "rel_change_pct": (float("nan") if prev is None
                                        else 100.0 * (val - prev) / prev)})
        prev = val
    vals = np.array([r["peak_10g"] for r in rows[2:]])
    spread = float(100.0 * (vals.max() - vals.min()) / vals.mean())
    return rows, spread


# ----------------------------------------------------------------------------
# Section 12. Verification of the forward models and the threshold layer
# ----------------------------------------------------------------------------
def _slab_transfer_matrix_amplitudes(frequency_hz, layers, incident_power_density):
    """Forward and backward wave amplitudes in every layer of a planar stack.

    Each layer carries the field A exp(-g z) + B exp(g z) in its local depth z.
    Starting from a unit forward wave in the semi-infinite terminator, tangential
    E and H continuity is imposed at every interface towards the surface, and
    the amplitudes are scaled to the incident plane wave of the given power
    density. This shares no code path with the impedance recursion in solve_slab.
    """
    material = [_layer_gamma_eta(frequency_hz, l["epsilon_r"], l["sigma"]) for l in layers]
    amps = [None] * len(layers)
    amps[-1] = (1.0 + 0.0j, 0.0j)
    for i in range(len(layers) - 2, -1, -1):
        _g_i, eta_i = material[i]
        _g_n, eta_n = material[i + 1]
        a_n, b_n = amps[i + 1]
        e_if = a_n + b_n
        h_if = (a_n - b_n) / eta_n
        t = float(layers[i]["thickness_m"])
        f_end = 0.5 * (e_if + eta_i * h_if)
        b_end = 0.5 * (e_if - eta_i * h_if)
        amps[i] = (f_end * np.exp(material[i][0] * t), b_end * np.exp(-material[i][0] * t))
    a0, b0 = amps[0]
    e_surface = a0 + b0
    h_surface = (a0 - b0) / material[0][1]
    e_incident = 0.5 * (e_surface + ETA0 * h_surface)
    scale = np.sqrt(incident_power_density * ETA0) / e_incident
    amps = [(scale * a, scale * b) for a, b in amps]
    return material, amps


def slab_transfer_matrix_sar(frequency_hz, layers, incident_power_density, depth):
    """Local SAR in a planar stack from the transfer-matrix amplitudes."""
    material, amps = _slab_transfer_matrix_amplitudes(frequency_hz, layers, incident_power_density)
    fronts = np.concatenate(([0.0], np.cumsum([l["thickness_m"] for l in layers[:-1]])))
    depth = np.asarray(depth, dtype=float)
    idx = np.clip(np.searchsorted(fronts, depth, side="right") - 1, 0, len(layers) - 1)
    out = np.empty(depth.shape)
    for i in range(len(layers)):
        sel = idx == i
        if not np.any(sel):
            continue
        g_i, _ = material[i]
        local = depth[sel] - fronts[i]
        a_i, b_i = amps[i]
        e_field = a_i * np.exp(-g_i * local) + b_i * np.exp(g_i * local)
        out[sel] = layers[i]["sigma"] * np.abs(e_field) ** 2 / layers[i]["density"]
    return out


def slab_absorbed_power(frequency_hz, layers, incident_power_density):
    """Absorbed power per unit area as the closed-form depth integral of
    sigma |E|^2 over every layer, using the transfer-matrix amplitudes."""
    material, amps = _slab_transfer_matrix_amplitudes(frequency_hz, layers, incident_power_density)
    total = 0.0
    for i, layer in enumerate(layers):
        g = material[i][0]
        a, b = g.real, g.imag
        amp_f, amp_b = amps[i]
        if layer["thickness_m"] is None:
            integral = abs(amp_f) ** 2 / (2.0 * a)
        else:
            t = float(layer["thickness_m"])
            integral = (abs(amp_f) ** 2 * (1.0 - np.exp(-2.0 * a * t)) / (2.0 * a)
                        + abs(amp_b) ** 2 * (np.exp(2.0 * a * t) - 1.0) / (2.0 * a)
                        + 2.0 * (amp_f * np.conj(amp_b) * (1.0 - np.exp(-2j * b * t))
                                 / (2j * b)).real)
        total += layer["sigma"] * float(integral)
    return total


def _slab_reflection(frequency_hz, layers):
    """Surface reflection coefficient from the input-impedance recursion."""
    material = [_layer_gamma_eta(frequency_hz, l["epsilon_r"], l["sigma"]) for l in layers]
    zin = material[-1][1]
    for i in range(len(layers) - 2, -1, -1):
        gamma, eta = material[i]
        th = np.tanh(gamma * float(layers[i]["thickness_m"]))
        zin = eta * (zin + eta * th) / (eta + zin * th)
    return (zin - ETA0) / (zin + ETA0)


def slab_solver_check():
    """At nominal properties, the largest relative difference between the SAR
    profiles of solve_slab and of the transfer-matrix solution (sample points on
    interfaces excluded), and the relative difference between the depth-integrated
    absorbed power and the transmitted power S0 (1 - |Gamma|^2)."""
    result = {}
    for f in FREQUENCIES_HZ:
        layers = _slab_layers(nominal_materials(f))
        depth, sar, _rho, _energy_error = solve_slab(f, layers, ANCHOR_S0_W_PER_M2)
        fronts = np.cumsum([l["thickness_m"] for l in layers[:-1]])
        interior = np.all(np.abs(depth[:, None] - fronts[None, :]) > 1e-9, axis=1)
        reference = slab_transfer_matrix_sar(f, layers, ANCHOR_S0_W_PER_M2, depth[interior])
        rel = np.abs(sar[interior] - reference) / np.maximum(np.abs(reference), 1e-300)
        absorbed = slab_absorbed_power(f, layers, ANCHOR_S0_W_PER_M2)
        transmitted = ANCHOR_S0_W_PER_M2 * (1.0 - abs(_slab_reflection(f, layers)) ** 2)
        result[FREQUENCY_LABEL[f]] = {
            "max_rel_diff": float(rel.max()),
            "power_balance_rel_error": float(abs(absorbed - transmitted) / transmitted)}
    return result


def thermal_solver_check(frequency_hz=2.45e9, voxel_m=0.002):
    """Compare the 3-D voxel bioheat solver with the 1-D radial solution for a
    spherically symmetric source, and report the ratio of voxel peak
    temperature rises for the full Mie source and the symmetric source."""
    nom = nominal_materials(frequency_hz)
    layers = _sphere_layers(nom)
    n_cells = 160
    dr = SPHERE_RADIUS_M / n_cells
    radii = (np.arange(n_cells) + 0.5) * dr
    shell = sphere_shell_sar(layers, frequency_hz, radii, ANCHOR_S0_W_PER_M2)
    outer = [l["outer_radius_m"] for l in layers]
    li = np.searchsorted(outer, radii, side="left")
    cond = np.array([nom[layers[i]["name"]]["thermal_conductivity"] for i in li])
    wb = np.array([perfusion_sink(nom[layers[i]["name"]]["perfusion"],
                                  nom[layers[i]["name"]]["density"]) for i in li])
    dens = np.array([nom[layers[i]["name"]]["density"] for i in li])
    met = np.array([nom[layers[i]["name"]]["metabolic"] for i in li]) * dens
    base = solve_radial_pennes(radii, cond, wb, met)
    heat = solve_radial_pennes(radii, cond, wb, met + dens * shell)
    dt_radial = float(np.max(heat - base))

    edge, sp, rho, sar = voxelize_sphere(layers, frequency_hz,
                                         ANCHOR_S0_W_PER_M2, voxel_m)
    count = rho.shape[0]
    ax = edge + (np.arange(count) + 0.5) * sp
    xx, yy, zz = np.meshgrid(ax, ax, ax, indexing="ij")
    rr = np.sqrt(xx * xx + yy * yy + zz * zz)
    sar_sym = np.where(rho > 0, np.interp(rr, radii, shell), 0.0)
    k_vox, wb_vox = _sphere_voxel_thermal_fields(layers, nom, edge, sp, rho.shape)
    k_vox = np.where(rho > 0, k_vox, 0.0)
    wb_vox = np.where(rho > 0, wb_vox, 0.0)
    dt_sym = float(solve_voxel_pennes_delta(sp, rho, sar_sym, k_vox, wb_vox).max())
    dt_true = float(solve_voxel_pennes_delta(sp, rho, sar, k_vox, wb_vox).max())
    return {"radial_peak_dT": dt_radial, "voxel3d_symmetric_peak_dT": dt_sym,
            "rel_diff_pct": 100.0 * (dt_sym - dt_radial) / dt_radial,
            "voxel3d_true_peak_dT": dt_true,
            "true_source_ratio": dt_true / dt_sym}


def deterministic_checks(cfg):
    """Model-independent verification quantities reported in the article."""
    checks = {"slab_solver": slab_solver_check(), "sphere_absorption_balance": {}}
    for f in FREQUENCIES_HZ:
        layers = _sphere_layers(nominal_materials(f))
        checks["sphere_absorption_balance"][FREQUENCY_LABEL[f]] = \
            float(absorption_balance_error(layers, f))
    checks["thermal_3d_vs_radial"] = thermal_solver_check(voxel_m=cfg["voxel_m"])
    checks["nominal_dielectric"] = {
        FREQUENCY_LABEL[f]: {t: {"epsilon_r": tissue_permittivity(t, f)[0],
                                 "sigma": tissue_permittivity(t, f)[1]}
                             for t in TISSUE_ORDER}
        for f in FREQUENCIES_HZ}
    return checks


# ----------------------------------------------------------------------------
# Section 13. Analysis orchestration
# ----------------------------------------------------------------------------
CASES = ("slab", "sphere")
CASE_LABEL = {"slab": "Slab", "sphere": "Sphere"}


def analyze_case(frequency_hz, case, cfg, cache_dir, model):
    y_sar, nom_sar = monte_carlo(case, frequency_hz, cfg["n_sar"], cfg["seed"],
                                 "sar", cfg["voxel_m"], cache_dir, model=model)
    y = y_sar[:, 0]
    var = empirical_var(y)
    conf = conformal_var(y)
    tol = tolerance_var(y)
    cvar = cvar_upper(y)
    mdro = moment_dro_var(y)
    row = {
        "model": model,
        "frequency_hz": frequency_hz,
        "frequency": FREQUENCY_LABEL[frequency_hz],
        "geometry": CASE_LABEL[case],
        "case": case,
        "n_sar": cfg["n_sar"],
        "nominal_sar": float(nom_sar[0]),
        "mean_sar": float(np.mean(y)),
        "std_sar": float(np.std(y, ddof=1)),
        "var_q95": var,
        "conformal_q95": conf,
        "tolerance_q95": tol,
        "cvar95": cvar,
        "moment_q95": mdro,
        "S_var": incident_power_threshold(var, SAR_LIMIT_W_PER_KG),
        "S_conformal": incident_power_threshold(conf, SAR_LIMIT_W_PER_KG),
        "S_tolerance": incident_power_threshold(tol, SAR_LIMIT_W_PER_KG),
        "S_cvar": incident_power_threshold(cvar, SAR_LIMIT_W_PER_KG),
        "S_moment": incident_power_threshold(mdro, SAR_LIMIT_W_PER_KG),
        "S_nominal": incident_power_threshold(float(nom_sar[0]), SAR_LIMIT_W_PER_KG),
        "reference_level_wb": reference_level_w_per_m2(frequency_hz),
        "reference_level_local": local_reference_level_w_per_m2(frequency_hz),
    }
    # Wasserstein threshold curve, radius from 0 to 50% of the nominal SAR.
    eps_grid = np.linspace(0.0, 0.5 * row["nominal_sar"], 21)
    row["wass_epsilon"] = eps_grid.tolist()
    row["wass_S"] = [incident_power_threshold(wasserstein_cvar(y, e), SAR_LIMIT_W_PER_KG)
                     for e in eps_grid]
    # Critical radius for each comparison level R: S0*L/(CVaR+eps/alpha) = R
    # gives eps = alpha*(S0*L/R - CVaR). No nonnegative crossing exists when the
    # CVaR threshold is already at or below R; NaN marks that case.
    for tier, rl in (("wb", row["reference_level_wb"]),
                     ("local", row["reference_level_local"])):
        eps_c = ALPHA * (ANCHOR_S0_W_PER_M2 * SAR_LIMIT_W_PER_KG / rl - cvar)
        eps_c = float(eps_c) if eps_c > 0.0 else float("nan")
        row[f"critical_eps_{tier}"] = eps_c
        row[f"critical_eps_{tier}_pct"] = float(100.0 * eps_c / row["nominal_sar"])
    row["_y_sar"] = y
    return row


def build_config(mode):
    if mode == "full":
        return {"n_sar": 2000, "n_temp": 1000, "voxel_m": 0.002, "splits": 200,
                "seed": SEED, "temp_frequencies": (2.45e9,)}
    if mode == "medium":
        return {"n_sar": 600, "n_temp": 300, "voxel_m": 0.0025, "splits": 100,
                "seed": SEED, "temp_frequencies": (2.45e9,)}
    return {"n_sar": 150, "n_temp": 80, "voxel_m": 0.004, "splits": 60,
            "seed": SEED, "temp_frequencies": (2.45e9,)}


def run_model(model, cfg, cache_dir):
    """All SAR and temperature analyses for one uncertainty model."""
    seed = cfg["seed"]
    rows = []
    for f in FREQUENCIES_HZ:
        for case in CASES:
            print(f"  SAR [{model}]: {case} {FREQUENCY_LABEL[f]}", flush=True)
            rows.append(analyze_case(f, case, cfg, cache_dir, model))
    coverage, calib, shifts = {}, {}, {}
    for r in rows:
        key = (r["frequency_hz"], r["case"])
        coverage[key] = coverage_experiment(r["_y_sar"], splits=cfg["splits"], seed=seed)
        calib[key] = calibration_size_experiment(r["_y_sar"], seed=seed)
        sh = shift_experiment(r["_y_sar"], seed=seed)
        for item in sh:
            item["w1_pct_nominal"] = 100.0 * item["w1"] / r["nominal_sar"]
        shifts[key] = sh
    temp_rows = []
    for f in cfg["temp_frequencies"]:
        for case in CASES:
            print(f"  Temperature [{model}]: {case} {FREQUENCY_LABEL[f]}", flush=True)
            out, nom = monte_carlo(case, f, cfg["n_temp"], seed, "both",
                                   cfg["voxel_m"], cache_dir, model=model)
            y_sar, y_temp = out[:, 0], out[:, 1]
            temp_rows.append({
                "model": model, "frequency_hz": f, "frequency": FREQUENCY_LABEL[f],
                "geometry": CASE_LABEL[case], "case": case, "n_temp": cfg["n_temp"],
                "nominal_temp": float(nom[1]),
                "sar_var_q95": empirical_var(y_sar), "sar_cvar95": cvar_upper(y_sar),
                "temp_var_q95": empirical_var(y_temp), "temp_cvar95": cvar_upper(y_temp),
                "S_temp_var": incident_power_threshold(empirical_var(y_temp), DELTA_T_LIMIT_C),
                "S_temp_cvar": incident_power_threshold(cvar_upper(y_temp), DELTA_T_LIMIT_C),
                "sar_temp_spearman": float(_spearman(y_sar, y_temp)),
                "_y_sar": y_sar, "_y_temp": y_temp})
    r0 = rows[0]
    eps_check = 0.1 * r0["nominal_sar"]
    wcheck = {"case": f"{r0['frequency']} {r0['geometry']}", "epsilon": eps_check,
              "closed_form": wasserstein_cvar(r0["_y_sar"], eps_check),
              "transport_construction": wasserstein_cvar_transport_check(r0["_y_sar"], eps_check)}
    return {"rows": rows, "coverage": coverage, "calib": calib, "shifts": shifts,
            "temp_rows": temp_rows, "wasserstein_check": wcheck}


def _spearman(a, b):
    from scipy.stats import spearmanr
    return spearmanr(a, b).statistic


# ----------------------------------------------------------------------------
# Section 14. Figures (Frontiers format: 300 dpi RGB TIFF, 180 mm wide,
# lines of at least 2 pt, text of at least 8 pt)
# ----------------------------------------------------------------------------
MM = 1.0 / 25.4
FIG_WIDTH_IN = 180.0 * MM
LW = 2.0
COLOR = {"var": "#4C78A8", "tol": "#72B7B2", "cvar": "#F58518", "moment": "#E45756",
         "gauss": "#B279A2", "emp": "#9D755D", "points": "#8C8C8C",
         "local": "#000000", "wb": "#7F7F7F"}
UNIT_SPD = "W m⁻²"
UNIT_SAR = "W kg⁻¹"


def _figure_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"],
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": LW, "lines.linewidth": LW, "patch.linewidth": LW,
        "xtick.major.width": LW, "ytick.major.width": LW,
        "xtick.minor.width": LW, "ytick.minor.width": LW,
        "xtick.major.size": 4, "ytick.major.size": 4,
        "xtick.minor.size": 2.5, "ytick.minor.size": 2.5,
        "lines.markersize": 4, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "legend.frameon": False, "savefig.dpi": 300,
        "pdf.fonttype": 42, "svg.hashsalt": "rf-dosimetry"})
    return plt


def _save(fig, figdir, name):
    """Write a 300 dpi RGB TIFF for submission and a vector PDF for inspection."""
    import io
    from PIL import Image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300)
    buf.seek(0)
    with Image.open(buf) as im:
        im.convert("RGB").save(os.path.join(figdir, name + ".tif"),
                               compression="tiff_lzw", dpi=(300, 300))
    fig.savefig(os.path.join(figdir, name + ".pdf"),
                metadata={"Creator": "Matplotlib", "CreationDate": None})
    import matplotlib.pyplot as plt
    plt.close(fig)


def _panel_label(ax, text):
    ax.text(-0.02, 1.03, text, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, fontweight="bold")


def workflow_figure(figdir):
    plt = _figure_style()
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 100.0 * MM))
    ax.set_xlim(0, 112); ax.set_ylim(0, 62); ax.set_axis_off()
    col = {"phys": "#E6F0FA", "phys_edge": "#2B6CB0", "unc": "#FFF1D6", "unc_edge": "#C56A00",
           "data": "#EEE7FA", "data_edge": "#6B46C1", "thr": "#E2F3E8", "thr_edge": "#2F855A",
           "out": "#F8E5EF", "out_edge": "#B83280", "ver": "#F0F2F5", "ver_edge": "#59636E",
           "text": "#172033"}

    def box(x, y, w, h, text, fc, ec):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.42",
                                    fc=fc, ec=ec, lw=LW))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8,
                color=col["text"], linespacing=1.15)

    def head(x, y, w, text):
        ax.text(x + w / 2, y, text, ha="center", va="bottom", fontsize=8.5,
                fontweight="bold", color=col["text"])

    def arrow(x0, y0, x1, y1, color, style="-|>", ls="-"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                     mutation_scale=11, lw=LW, color=color,
                                     linestyle=ls, shrinkA=2, shrinkB=2))

    head(1, 58.5, 32, "Forward models")
    box(1, 50, 32, 6.5, "Plane wave at anchor S₀\n0.9, 2.45 and 5.8 GHz", col["phys"], col["phys_edge"])
    box(1, 40.5, 32, 6.5, "Slab, transmission line\nSphere, multilayer Mie", col["phys"], col["phys_edge"])
    box(1, 31, 32, 6.5, "Peak 10 g SAR\n(IEC/IEEE 62704-1 cube criteria)", col["phys"], col["phys_edge"])
    box(1, 21.5, 32, 6.5, "Pennes ΔT, slab 1-D and\nsphere 3-D voxel", col["phys"], col["phys_edge"])
    box(1, 8.5, 32, 9, "Verification, transfer matrix,\nslab power balance, Mie power\nbalance, radial bioheat, grids",
        col["ver"], col["ver_edge"])
    arrow(17, 50, 17, 47.4, col["phys_edge"])
    arrow(17, 40.5, 17, 37.9, col["phys_edge"])
    arrow(17, 31, 17, 28.4, col["phys_edge"])
    arrow(17, 21.5, 17, 18.4, col["ver_edge"], ls="--")

    head(39, 58.5, 31, "Uncertainty models")
    box(39, 49.5, 16.5, 7, "Dielectric log\nmultipliers,\nGaussian copula", col["unc"], col["unc_edge"])
    box(57.5, 49.5, 12.5, 7, "Density and\nthermal\nranges", col["unc"], col["unc_edge"])
    box(39, 39.5, 31, 6.5, "Reference and within-tissue\nmodels, n property draws", col["unc"], col["unc_edge"])
    box(39, 29.5, 31, 6.5, "Forward model for every\ndraw at S₀", col["phys"], col["phys_edge"])
    box(39, 19.5, 31, 6.5, "Endpoint sample of\npeak 10 g SAR or ΔT", col["data"], col["data_edge"])
    arrow(47.25, 49.5, 50, 46.2, col["unc_edge"])
    arrow(63.75, 49.5, 59, 46.2, col["unc_edge"])
    arrow(54.5, 39.5, 54.5, 36.2, col["unc_edge"])
    arrow(54.5, 29.5, 54.5, 26.2, col["data_edge"])
    arrow(33.4, 32.75, 38.6, 32.75, col["phys_edge"])

    head(78, 58.5, 33, "Threshold layer")
    box(78, 46, 33, 10.5, "Bounds from the sample\nconformal quantile, tolerance 95/95,\nCVaR, Cantelli and\nWasserstein CVaR",
        col["thr"], col["thr_edge"])
    box(78, 35.5, 33, 7, "Incident power density threshold\nS = S₀L/q for each bound q", col["thr"], col["thr_edge"])
    box(78, 25, 33, 7, "Comparison with local and\nwhole-body reference levels", col["thr"], col["thr_edge"])
    box(78, 11.5, 33, 10, "Outputs, thresholds, critical\nradii, coverage, calibration\nsize and shift response",
        col["out"], col["out_edge"])
    arrow(94.5, 46, 94.5, 42.8, col["thr_edge"])
    arrow(94.5, 35.5, 94.5, 32.3, col["thr_edge"])
    arrow(94.5, 25, 94.5, 21.8, col["out_edge"])
    arrow(70.4, 22.75, 74, 22.75, col["data_edge"], style="-")
    arrow(74, 22.75, 74, 51.25, col["data_edge"], style="-")
    arrow(74, 51.25, 77.6, 51.25, col["data_edge"])
    fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.01)
    _save(fig, figdir, "Figure1")


def make_figures(results, outdir):
    plt = _figure_style()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter, NullFormatter
    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)
    workflow_figure(figdir)
    models = list(results)
    freqs = list(FREQUENCIES_HZ)
    letters = "ABCDEFGH"

    def row_for(model, f, case):
        return next(x for x in results[model]["rows"]
                    if x["frequency_hz"] == f and x["case"] == case)

    # Figure 2: thresholds by bound for both uncertainty models.
    fig, axes = plt.subplots(len(models), 1, figsize=(FIG_WIDTH_IN, 64.0 * MM * len(models)),
                             sharex=True)
    for jx, model in enumerate(models):
        ax = axes[jx]
        x = np.arange(len(freqs) * len(CASES)); w = 0.19
        vals = {k: [] for k in ("S_var", "S_tolerance", "S_cvar", "S_moment", "S_nominal",
                                "reference_level_local", "reference_level_wb")}
        labels = []
        for f in freqs:
            for case in CASES:
                r = row_for(model, f, case)
                labels.append(f"{r['frequency']}\n{r['geometry']}")
                for k in vals:
                    vals[k].append(r[k])
        for off, key, lab, c in ((-1.5, "S_var", "Empirical/conformal quantile", COLOR["var"]),
                                 (-0.5, "S_tolerance", "Tolerance (95/95)", COLOR["tol"]),
                                 (0.5, "S_cvar", "CVaR", COLOR["cvar"]),
                                 (1.5, "S_moment", "Cantelli moment bound", COLOR["moment"])):
            ax.bar(x + off * w, vals[key], w, color=c, label=lab, linewidth=0)
        for xi, loc, wb in zip(x, vals["reference_level_local"], vals["reference_level_wb"]):
            ax.hlines(loc, xi - 2.3 * w, xi + 2.3 * w, colors=COLOR["local"],
                      linestyles=(0, (4, 2)), lw=LW)
            ax.hlines(wb, xi - 2.3 * w, xi + 2.3 * w, colors=COLOR["wb"],
                      linestyles=(0, (1, 1.5)), lw=LW)
        ax.plot(x, vals["S_nominal"], ls="none", marker="D", ms=4.5, color="black",
                label="Nominal properties")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_ylabel(f"Threshold S ({UNIT_SPD})")
        ax.set_title(MODEL_LABEL[model], loc="center")
        _panel_label(ax, f"({letters[jx]})")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
    handles, labs = axes[0].get_legend_handles_labels()
    handles += [Line2D([0], [0], color=COLOR["local"], ls=(0, (4, 2)), lw=LW),
                Line2D([0], [0], color=COLOR["wb"], ls=(0, (1, 1.5)), lw=LW)]
    labs += ["Local reference level", "Whole-body reference level"]
    fig.legend(handles, labs, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    _save(fig, figdir, "Figure2")

    # Figure 3: held-out exceedance of Gaussian plug-in and conformal quantiles.
    fig, axes = plt.subplots(1, len(models), figsize=(FIG_WIDTH_IN, 70.0 * MM), sharey=True)
    for jx, model in enumerate(models):
        ax = axes[jx]
        positions, data, colors = [], [], []
        for fx, f in enumerate(freqs):
            for kx, case in enumerate(CASES):
                gauss, conf = results[model]["coverage"][(f, case)]
                base = fx * 3 + kx * 1.15
                positions += [base - 0.24, base + 0.24]
                data += [gauss, conf]
                colors += [COLOR["gauss"], COLOR["var"]]
        bp = ax.boxplot(data, positions=positions, widths=0.4, patch_artist=True,
                        showfliers=False, showmeans=True,
                        meanprops={"marker": "^", "markerfacecolor": "white",
                                   "markeredgecolor": "black", "markersize": 5,
                                   "markeredgewidth": LW},
                        medianprops={"color": "black", "lw": LW},
                        whiskerprops={"lw": LW}, capprops={"lw": LW}, boxprops={"lw": LW})
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
        ax.axhline(ALPHA, ls=(0, (4, 2)), color="black", lw=LW)
        ax.set_xticks([fx * 3 + 0.575 for fx in range(len(freqs))])
        ax.set_xticklabels([f"{FREQUENCY_LABEL[f]}\nslab | sphere" for f in freqs])
        ax.set_title(MODEL_LABEL[model])
        if jx == 0:
            ax.set_ylabel("Held-out exceedance fraction")
        _panel_label(ax, f"({letters[jx]})")
    fig.legend(handles=[Patch(facecolor=COLOR["gauss"], edgecolor="black", lw=LW, label="Gaussian plug-in"),
                        Patch(facecolor=COLOR["var"], edgecolor="black", lw=LW, label="Conformal quantile"),
                        Line2D([0], [0], ls="none", marker="^", markerfacecolor="white",
                               markeredgecolor="black", markersize=5, markeredgewidth=LW,
                               label="Mean over splits"),
                        Line2D([0], [0], ls=(0, (4, 2)), color="black", lw=LW,
                               label="Target α = 0.05")],
               ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    _save(fig, figdir, "Figure3")

    # Figure 4: threshold and exceedance versus calibration size (reference model).
    model = models[0]
    keys = [(2.45e9, "slab"), (2.45e9, "sphere")]
    fig, axes = plt.subplots(2, 2, figsize=(FIG_WIDTH_IN, 110.0 * MM), sharex=True)
    for jx, key in enumerate(keys):
        rec = results[model]["calib"][key]
        n = [r["n_cal"] for r in rec]
        top, bot = axes[0][jx], axes[1][jx]
        for tag, lab, c, mk in (("emp", "Empirical quantile", COLOR["emp"], "s"),
                                ("conf", "Conformal quantile", COLOR["var"], "o"),
                                ("tol", "Tolerance (95/95)", COLOR["tol"], "^")):
            top.plot(n, [r[f"S_{tag}_mean"] for r in rec], "-", marker=mk, color=c, label=lab)
            bot.plot(n, [r[f"exc_{tag}_mean"] for r in rec], "-", marker=mk, color=c, label=lab)
        for a in (top, bot):
            a.set_xscale("log")
            a.set_xticks(n)
            a.set_xticklabels([str(v) for v in n])
            a.minorticks_off()
        top.set_title(f"{FREQUENCY_LABEL[key[0]]} {CASE_LABEL[key[1]].lower()}")
        bot.axhline(ALPHA, ls=(0, (4, 2)), color="black", lw=LW)
        bot.set_xlabel("Calibration size n")
        _panel_label(top, f"({letters[jx]})")
        _panel_label(bot, f"({letters[jx + 2]})")
    axes[0][0].set_ylabel(f"Mean threshold S ({UNIT_SPD})")
    axes[1][0].set_ylabel("Mean held-out exceedance")
    handles, labs = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labs, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, figdir, "Figure4")

    # Figure 5: Wasserstein CVaR threshold versus radius, both models.
    fig, axes = plt.subplots(len(models), len(freqs),
                             figsize=(FIG_WIDTH_IN, 60.0 * MM * len(models)), sharex=True)
    for mx, model in enumerate(models):
        for fx, f in enumerate(freqs):
            ax = axes[mx][fx]
            for case, ls, mk in (("slab", "-", "o"), ("sphere", (0, (3, 1.5)), "s")):
                r = row_for(model, f, case)
                eps_pct = 100.0 * np.array(r["wass_epsilon"]) / r["nominal_sar"]
                ax.plot(eps_pct, r["wass_S"], ls=ls, marker=mk, ms=3.5, color=COLOR["cvar"],
                        label=CASE_LABEL[case])
            ax.axhline(local_reference_level_w_per_m2(f), ls=(0, (4, 2)), color=COLOR["local"], lw=LW)
            ax.axhline(reference_level_w_per_m2(f), ls=(0, (1, 1.5)), color=COLOR["wb"], lw=LW)
            ax.set_title(f"{MODEL_LABEL[model]}, {FREQUENCY_LABEL[f]}")
            if mx == len(models) - 1:
                ax.set_xlabel("Radius ε (% of nominal SAR)")
            if fx == 0:
                ax.set_ylabel(f"Wasserstein threshold ({UNIT_SPD})")
            _panel_label(ax, f"({letters[mx * len(freqs) + fx]})")
    fig.legend(handles=[Line2D([0], [0], color=COLOR["cvar"], ls="-", marker="o", ms=3.5, label="Slab"),
                        Line2D([0], [0], color=COLOR["cvar"], ls=(0, (3, 1.5)), marker="s", ms=3.5,
                               label="Sphere"),
                        Line2D([0], [0], color=COLOR["local"], ls=(0, (4, 2)), lw=LW,
                               label="Local reference level"),
                        Line2D([0], [0], color=COLOR["wb"], ls=(0, (1, 1.5)), lw=LW,
                               label="Whole-body reference level")],
               ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, figdir, "Figure5")

    # Figure 6: paired SAR and temperature samples at 2.45 GHz.
    fig, axes = plt.subplots(len(models), len(CASES),
                             figsize=(FIG_WIDTH_IN, 64.0 * MM * len(models)))
    for mx, model in enumerate(models):
        for cx, tr in enumerate(results[model]["temp_rows"]):
            ax = axes[mx][cx]
            ax.scatter(tr["_y_sar"], tr["_y_temp"], s=4, alpha=0.45, color=COLOR["points"],
                       linewidths=0)
            ax.axvline(tr["sar_var_q95"], ls="-", color=COLOR["var"], lw=LW, label="SAR quantile")
            ax.axvline(tr["sar_cvar95"], ls=(0, (4, 2)), color=COLOR["cvar"], lw=LW, label="SAR CVaR")
            ax.axhline(tr["temp_var_q95"], ls=(0, (1, 1.5)), color="black", lw=LW,
                       label="ΔT quantile")
            ax.set_xlabel(f"Peak 10 g SAR at S₀ ({UNIT_SAR})")
            ax.set_ylabel("Peak ΔT at S₀ (K)")
            ax.set_title(f"{MODEL_LABEL[model]}, {tr['geometry'].lower()}, {tr['frequency']}")
            _panel_label(ax, f"({letters[mx * len(CASES) + cx]})")
    handles, labs = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labs, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, figdir, "Figure6")


# ----------------------------------------------------------------------------
# Section 15. Result files, article tables, and in-text numbers
# ----------------------------------------------------------------------------
def _write_csv(path, header, rows):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def _full(x):
    """Full-precision text for a numeric value; other values unchanged."""
    if isinstance(x, (bool, np.bool_, str, int, np.integer)):
        return x
    return repr(float(x))


def _sci(x):
    """Scientific notation for the article text, for example 7.0 × 10^−14^."""
    mantissa, exponent = f"{x:.1e}".split("e")
    return f"{mantissa} × 10^{int(exponent)}^".replace("^-", "^−")


def _fmt(x, nd=1):
    """Round half up on the shortest decimal representation of x."""
    from decimal import Decimal, ROUND_HALF_UP
    if x is None or not np.isfinite(x):
        return "n/a"
    q = Decimal(1).scaleb(-nd)
    return str(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))


def write_results(results, checks, conv, cfg, outdir):
    """Full-precision result files for each model."""
    for model, res in results.items():
        d = os.path.join(outdir, "results", model)
        os.makedirs(d, exist_ok=True)
        fields = ["frequency", "geometry", "n_sar", "nominal_sar", "mean_sar", "std_sar",
                  "var_q95", "conformal_q95", "tolerance_q95", "cvar95", "moment_q95",
                  "S_nominal", "S_var", "S_conformal", "S_tolerance", "S_cvar", "S_moment",
                  "reference_level_local", "reference_level_wb",
                  "critical_eps_local_pct", "critical_eps_wb_pct"]
        _write_csv(os.path.join(d, "thresholds.csv"), fields,
                   [[_full(r[k]) for k in fields] for r in res["rows"]])
        rows = []
        for f in FREQUENCIES_HZ:
            for case in CASES:
                gauss, conf = res["coverage"][(f, case)]
                rows.append([FREQUENCY_LABEL[f], CASE_LABEL[case], len(gauss),
                             repr(float(gauss.mean())), repr(float(np.median(gauss))),
                             repr(float(np.std(gauss, ddof=1) / np.sqrt(len(gauss)))),
                             repr(float(conf.mean())), repr(float(np.median(conf))),
                             repr(float(np.std(conf, ddof=1) / np.sqrt(len(conf))))])
        _write_csv(os.path.join(d, "coverage.csv"),
                   ["frequency", "geometry", "splits", "gaussian_mean_exceedance",
                    "gaussian_median_exceedance", "gaussian_mean_standard_error",
                    "conformal_mean_exceedance", "conformal_median_exceedance",
                    "conformal_mean_standard_error"], rows)
        rows = []
        for (f, case), rec in res["calib"].items():
            for r in rec:
                rows.append([FREQUENCY_LABEL[f], CASE_LABEL[case], r["n_cal"]] +
                            [repr(r[k]) for k in ("S_emp_mean", "S_conf_mean", "S_tol_mean",
                                                  "exc_emp_mean", "exc_conf_mean", "exc_tol_mean")])
        _write_csv(os.path.join(d, "calibration_size.csv"),
                   ["frequency", "geometry", "n_cal", "S_emp_mean", "S_conf_mean", "S_tol_mean",
                    "exc_emp_mean", "exc_conf_mean", "exc_tol_mean"], rows)
        rows = []
        for (f, case), rec in res["shifts"].items():
            for r in rec:
                rows.append([FREQUENCY_LABEL[f], CASE_LABEL[case], r["shift"], repr(r["w1"]),
                             repr(r["w1_pct_nominal"]), repr(r["conformal_exceedance"]),
                             r["conformal_exceed_count"], repr(r["cvar_shifted"]),
                             repr(r["wasserstein_bound"]), r["bound_holds"], repr(r["share_pct"])])
        _write_csv(os.path.join(d, "shift_test.csv"),
                   ["frequency", "geometry", "shift", "w1", "w1_pct_nominal",
                    "conformal_exceedance", "conformal_exceed_count", "cvar_shifted",
                    "wasserstein_bound", "bound_holds", "share_of_worst_case_pct"], rows)
        tfields = ["frequency", "geometry", "n_temp", "nominal_temp", "sar_var_q95", "sar_cvar95",
                   "temp_var_q95", "temp_cvar95", "S_temp_var", "S_temp_cvar", "sar_temp_spearman"]
        _write_csv(os.path.join(d, "temperature_thresholds.csv"), tfields,
                   [[_full(r[k]) for k in tfields] for r in res["temp_rows"]])
        with open(os.path.join(d, "wasserstein_check.json"), "w", encoding="utf-8") as fh:
            json.dump(res["wasserstein_check"], fh, indent=2)
    d = os.path.join(outdir, "results")
    rows = []
    for f, (rws, spread) in conv.items():
        for r in rws:
            rows.append([FREQUENCY_LABEL[f], f"{r['voxel_mm']:.2f}", r["cells"], repr(r["peak_10g"])])
    _write_csv(os.path.join(d, "voxel_convergence.csv"),
               ["frequency", "voxel_mm", "cells", "peak_10g"], rows)
    checks = dict(checks)
    checks["grid_relative_range_pct_four_finest"] = {FREQUENCY_LABEL[f]: conv[f][1] for f in conv}
    checks["config"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}
    with open(os.path.join(d, "verification.json"), "w", encoding="utf-8") as fh:
        json.dump(checks, fh, indent=2)


TISSUE_LABEL = {"skin_dry": "Dry skin", "fat_not_infiltrated": "Fat (not infiltrated)",
                "muscle": "Muscle", "bone_cortical": "Cortical bone"}


def _range_cell(triple, nd):
    nom, mn, mx = triple
    if mx <= mn:
        return f"{_fmt(nom, nd)} (fixed)"
    return f"{_fmt(nom, nd)} ({_fmt(mn, nd)}–{_fmt(mx, nd)})"


def write_article_tables(results, checks, conv, outdir):
    """Tables exactly as displayed in the article (rounded strings); file
    tableN corresponds to Table N of the article."""
    d = os.path.join(outdir, "tables")
    os.makedirs(d, exist_ok=True)
    nd = checks["nominal_dielectric"]
    rows = []
    for t in TISSUE_ORDER:
        cells = []
        for f in FREQUENCIES_HZ:
            v = nd[FREQUENCY_LABEL[f]][t]
            cells += [_fmt(v["epsilon_r"], 2), _fmt(v["sigma"], 3)]
        rows.append([TISSUE_LABEL[t]] + cells)
    _write_csv(os.path.join(d, "table1_nominal_dielectric.csv"),
               ["Tissue"] + [f"{FREQUENCY_LABEL[f]} {q}" for f in FREQUENCIES_HZ
                             for q in ("eps_r", "sigma (S/m)")], rows)
    rows = [[TISSUE_LABEL[t],
             _range_cell(TISSUE_TABLE[t]["density"], 0),
             _range_cell(TISSUE_TABLE[t]["thermal_conductivity"], 3),
             _range_cell(TISSUE_TABLE[t]["perfusion"], 1),
             _range_cell(TISSUE_TABLE[t]["metabolic"], 2)] for t in TISSUE_ORDER]
    _write_csv(os.path.join(d, "table2_thermal_properties.csv"),
               ["Tissue", "Density (kg/m3)", "Thermal conductivity (W/m/K)",
                "Perfusion (mL/min/kg)", "Metabolic heat (W/kg)"], rows)
    rows = []
    for f in FREQUENCIES_HZ:
        ref = DIELECTRIC_BOX[f]
        rows.append([FREQUENCY_LABEL[f], _fmt(ref["log_half_width"][0], 3),
                     _fmt(ref["log_half_width"][1], 3), _fmt(WITHIN_TISSUE_LOG_HALF_WIDTH, 3),
                     _fmt(ref["correlation"], 3)])
    _write_csv(os.path.join(d, "table3_uncertainty_models.csv"),
               ["Frequency", "Reference h_eps", "Reference h_sigma", "Within-tissue h",
                "Copula correlation"], rows)
    vox = [r["voxel_mm"] for r in conv[FREQUENCIES_HZ[0]][0]]
    rows = [[f"{vm:.2f}"] + [_fmt(conv[f][0][k]["peak_10g"], 3) for f in FREQUENCIES_HZ]
            for k, vm in enumerate(vox)]
    rows.append(["Relative range (%)"] + [_fmt(conv[f][1]) for f in FREQUENCIES_HZ])
    _write_csv(os.path.join(d, "table4_grid_convergence.csv"),
               ["Voxel (mm)"] + [FREQUENCY_LABEL[f] for f in FREQUENCIES_HZ], rows)
    rows = []
    for model, res in results.items():
        for r in res["rows"]:
            rows.append([TABLE_MODEL_LABEL[model], f"{r['frequency_hz'] / 1e9:g}", r["geometry"],
                         _fmt(r["S_nominal"]), _fmt(r["S_var"]), _fmt(r["S_tolerance"]),
                         _fmt(r["S_cvar"]), _fmt(r["S_moment"])])
    _write_csv(os.path.join(d, "table5_sar_thresholds.csv"),
               ["Model", "Frequency (GHz)", "Geometry", "Nominal", "Quantile", "Tolerance",
                "CVaR", "Cantelli"], rows)
    rows = []
    for model, res in results.items():
        for r in res["rows"]:
            loc, wb = r["critical_eps_local_pct"], r["critical_eps_wb_pct"]
            rows.append([TABLE_MODEL_LABEL[model], f"{r['frequency_hz'] / 1e9:g}", r["geometry"],
                         "none" if not np.isfinite(loc) else _fmt(loc),
                         "none" if not np.isfinite(wb) else _fmt(wb)])
    _write_csv(os.path.join(d, "table6_critical_radii.csv"),
               ["Model", "Frequency (GHz)", "Geometry", "Local RL (%)", "Whole-body RL (%)"], rows)
    rows = []
    for model, res in results.items():
        for r in res["shifts"][(2.45e9, "sphere")]:
            rows.append([TABLE_MODEL_LABEL[model], r["shift"], _fmt(r["w1_pct_nominal"]),
                         _fmt(r["conformal_exceedance"], 3), _fmt(r["cvar_shifted"], 3),
                         _fmt(r["share_pct"])])
    _write_csv(os.path.join(d, "table7_shift_test.csv"),
               ["Model", "Transformation", "W1 (% nominal)", "Conformal exceedance",
                "Shifted CVaR (W/kg)", "Share of worst case (%)"], rows)
    rows = []
    for model, res in results.items():
        for r in res["temp_rows"]:
            rows.append([TABLE_MODEL_LABEL[model], r["geometry"],
                         _fmt(r["nominal_temp"], 3), _fmt(r["temp_var_q95"], 3),
                         _fmt(r["temp_cvar95"], 3), _fmt(r["S_temp_var"]), _fmt(r["S_temp_cvar"])])
    _write_csv(os.path.join(d, "table8_temperature_thresholds.csv"),
               ["Model", "Geometry", "Nominal dT (K)", "Quantile dT (K)", "CVaR dT (K)",
                "Quantile threshold", "CVaR threshold"], rows)


def write_numbers(results, checks, conv, cfg, outdir):
    """Computed values quoted in the article text, with the article's rounding."""
    num = {"Nsar": str(cfg["n_sar"]), "Ntemp": str(cfg["n_temp"]),
           "splits": str(cfg["splits"]), "withinTissueH": _fmt(WITHIN_TISSUE_LOG_HALF_WIDTH, 3)}
    widths = [w for f in FREQUENCIES_HZ for w in DIELECTRIC_BOX[f]["log_half_width"]]
    num["refHMin"] = _fmt(min(widths), 2)
    num["refHMax"] = _fmt(max(widths), 2)
    h_eps = [DIELECTRIC_BOX[f]["log_half_width"][0] for f in FREQUENCIES_HZ]
    h_sig = [DIELECTRIC_BOX[f]["log_half_width"][1] for f in FREQUENCIES_HZ]
    num["refRatioEpsMin"] = _fmt(np.exp(2.0 * min(h_eps)))
    num["refRatioEpsMax"] = _fmt(np.exp(2.0 * max(h_eps)))
    num["refRatioSigMin"] = _fmt(np.exp(2.0 * min(h_sig)))
    num["refRatioSigMax"] = _fmt(np.exp(2.0 * max(h_sig)))
    tag = {"reference": "Ref", "within_tissue": "Wt"}
    ftag = {0.9e9: "LowF", 2.45e9: "MidF", 5.8e9: "HighF"}
    first = next(iter(results.values()))["rows"]
    for f in FREQUENCIES_HZ:
        num["rlLocal" + ftag[f]] = _fmt(local_reference_level_w_per_m2(f))
        num["rlWb" + ftag[f]] = _fmt(reference_level_w_per_m2(f))
        sar = {r["case"]: r["nominal_sar"] for r in first if r["frequency_hz"] == f}
        num["sphereSlabRatio" + ftag[f]] = _fmt(sar["sphere"] / sar["slab"])
    for model, res in results.items():
        m = tag[model]
        for r in res["rows"]:
            k = f"{m}{r['case'].capitalize()}{ftag[r['frequency_hz']]}"
            num[k + "Var"] = _fmt(r["S_var"])
            num[k + "Tol"] = _fmt(r["S_tolerance"])
            num[k + "Cvar"] = _fmt(r["S_cvar"])
            num[k + "Mom"] = _fmt(r["S_moment"])
            num[k + "CritLocal"] = "none" if not np.isfinite(r["critical_eps_local_pct"]) \
                else _fmt(r["critical_eps_local_pct"])
            num[k + "CritWb"] = "none" if not np.isfinite(r["critical_eps_wb_pct"]) \
                else _fmt(r["critical_eps_wb_pct"])
            num[k + "NomSar"] = _fmt(r["nominal_sar"], 3)
            num[k + "NomThr"] = _fmt(r["S_nominal"])
        for name, key in (("VarDrop", "S_var"), ("CvarDrop", "S_cvar"), ("MomDrop", "S_moment")):
            drops = [100.0 * (1.0 - r[key] / r["S_nominal"]) for r in res["rows"]]
            num[f"{m}{name}Min"] = _fmt(min(drops))
            num[f"{m}{name}Max"] = _fmt(max(drops))
        below = [100.0 * (1.0 - r["S_moment"] / r["S_var"]) for r in res["rows"]]
        num[m + "MomBelowVarMin"] = _fmt(min(below))
        num[m + "MomBelowVarMax"] = _fmt(max(below))
        cov = res["coverage"]
        g_means = [cov[key][0].mean() for key in cov]
        c_means = [cov[key][1].mean() for key in cov]
        ses = [np.std(a, ddof=1) / np.sqrt(len(a)) for key in cov for a in cov[key]]
        num[m + "GaussBestPct"] = _fmt(100.0 * min(g_means))
        num[m + "GaussWorstPct"] = _fmt(100.0 * max(g_means))
        num[m + "GaussWorstRatio"] = _fmt(max(g_means) / ALPHA)
        num[m + "ConfBestPct"] = _fmt(100.0 * min(c_means))
        num[m + "ConfWorstPct"] = _fmt(100.0 * max(c_means))
        num[m + "NGaussAbove"] = str(sum(1 for v in g_means if float(_fmt(100.0 * v)) > 100.0 * ALPHA))
        num[m + "NGaussAbove2pp"] = str(sum(1 for v in g_means if 100.0 * (v - ALPHA) > 2.0))
        num[m + "NConfAbove"] = str(sum(1 for v in c_means if float(_fmt(100.0 * v)) > 100.0 * ALPHA))
        num[m + "SeMinPp"] = _fmt(100.0 * min(ses), 2)
        num[m + "SeMaxPp"] = _fmt(100.0 * max(ses), 2)
        for case in CASES:
            rec = {r["n_cal"]: r for r in res["calib"][(2.45e9, case)]}
            c = case.capitalize()
            nmax = max(rec)
            num[f"{m}Calib{c}N25ExcEmpPct"] = _fmt(100.0 * rec[25]["exc_emp_mean"])
            num[f"{m}Calib{c}N25ExcConfPct"] = _fmt(100.0 * rec[25]["exc_conf_mean"])
            num[f"{m}Calib{c}N100ExcTolPct"] = _fmt(100.0 * rec[100]["exc_tol_mean"])
            num[f"{m}Calib{c}NMax"] = str(nmax)
            num[f"{m}Calib{c}NMaxExcTolPct"] = _fmt(100.0 * rec[nmax]["exc_tol_mean"])
            num[f"{m}Calib{c}N100Tol"] = _fmt(rec[100]["S_tol_mean"])
            num[f"{m}Calib{c}N100Conf"] = _fmt(rec[100]["S_conf_mean"])
        exc_big = [r["exc_conf_mean"] for case in CASES for r in res["calib"][(2.45e9, case)]
                   if r["n_cal"] >= 100]
        num[m + "CalibConfBigMinPct"] = _fmt(100.0 * min(exc_big))
        num[m + "CalibConfBigMaxPct"] = _fmt(100.0 * max(exc_big))
        num[m + "NBelowLocalCvar"] = str(sum(1 for r in res["rows"]
                                             if r["S_cvar"] < r["reference_level_local"]))
        num[m + "NBelowLocalMom"] = str(sum(1 for r in res["rows"]
                                            if r["S_moment"] < r["reference_level_local"]))
        num[m + "NMomReverse"] = str(sum(1 for r in res["rows"]
                                         if r["S_var"] > r["reference_level_local"]
                                         > r["S_moment"]))
        sh = {r["shift"]: r for r in res["shifts"][(2.45e9, "sphere")]}
        num[m + "ShiftScale15ExcPct"] = _fmt(100.0 * sh["Scaling by 1.15"]["conformal_exceedance"])
        num[m + "ShiftNHold"] = str(sum(1 for r in sh.values() if r["bound_holds"]))
        shares = [r["share_pct"] for r in sh.values()]
        num[m + "ShareMin"] = _fmt(min(shares))
        num[m + "ShareMax"] = _fmt(max(shares))
        num[m + "ShareTail"] = _fmt(sh["Upper decile scaled by 1.5"]["share_pct"])
        for r in res["temp_rows"]:
            k = f"{m}{r['case'].capitalize()}Temp"
            num[k + "Nom"] = _fmt(r["nominal_temp"], 3)
            num[k + "Var"] = _fmt(r["S_temp_var"])
            num[k + "Cvar"] = _fmt(r["S_temp_cvar"])
            num[k + "Spearman"] = _fmt(r["sar_temp_spearman"], 2)
        wc = res["wasserstein_check"]
        diff = abs(wc["closed_form"] - wc["transport_construction"])
        num[m + "WassCheckAgree"] = ("exactly in double precision" if diff == 0.0
                                     else f"to within {_sci(diff)} W kg^−1^")
    crit = [r["critical_eps_local_pct"] for res in results.values() for r in res["rows"]
            if np.isfinite(r["critical_eps_local_pct"])]
    if set(results) == set(UNCERTAINTY_MODELS):
        for name, key in (("Var", "S_var"), ("Mom", "S_moment")):
            gains = [100.0 * (w[key] / r[key] - 1.0)
                     for r, w in zip(results["reference"]["rows"], results["within_tissue"]["rows"])]
            num[f"modelGain{name}Min"] = _fmt(min(gains))
            num[f"modelGain{name}Max"] = _fmt(max(gains))
    num["critLocalMin"] = _fmt(min(crit))
    num["critLocalMax"] = _fmt(max(crit))
    num["slabTmmMax"] = _sci(max(v["max_rel_diff"] for v in checks["slab_solver"].values()))
    num["slabPowerMax"] = _sci(max(v["power_balance_rel_error"]
                                   for v in checks["slab_solver"].values()))
    num["mieBalanceMax"] = _sci(max(checks["sphere_absorption_balance"].values()))
    th = checks["thermal_3d_vs_radial"]
    num["thermalVoxelRadialPct"] = _fmt(abs(th["rel_diff_pct"]))
    num["thermalTrueSymRatio"] = _fmt(th["true_source_ratio"], 3)
    spreads = {f: conv[f][1] for f in conv}
    worst = max(spreads, key=spreads.get)
    num["gridSpreadMax"] = _fmt(spreads[worst])
    num["gridSpreadMaxFreq"] = FREQUENCY_LABEL[worst]
    num["gridCoarsestMm"] = _fmt(conv[FREQUENCIES_HZ[0]][0][0]["voxel_mm"], 2)
    with open(os.path.join(outdir, "numbers.json"), "w", encoding="utf-8") as fh:
        json.dump(num, fh, indent=1, sort_keys=True)
    return num


# ----------------------------------------------------------------------------
# Section 16. Main
# ----------------------------------------------------------------------------
def preflight_check():
    """Check the runtime without evaluating any electromagnetic model."""
    import importlib
    print("RF dosimetry reproduction preflight")
    print(f"  Python: {platform.python_version()}")
    print(f"  Platform: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"  Logical CPUs reported by OS: {os.cpu_count()}")
    ok = True
    for package in ("numpy", "scipy", "matplotlib", "PIL", "scattnlay"):
        try:
            module = importlib.import_module(package)
            print(f"  [OK] {package}: {getattr(module, '__version__', 'installed')}")
        except Exception as exc:
            ok = False
            print(f"  [MISSING] {package}: {exc}")
    try:
        with _suppress_native_stdout():
            pass
        print("  [OK] native stdout redirection")
    except Exception as exc:
        ok = False
        print(f"  [FAILED] native stdout redirection: {exc}")
    print("Preflight passed." if ok else "Preflight failed: install the missing packages.")
    return 0 if ok else 1


def run(mode, outdir, cache_dir, models, seed):
    cfg = build_config(mode)
    cfg["seed"] = seed
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    print(f"[mode={mode}] N_sar={cfg['n_sar']} N_temp={cfg['n_temp']} "
          f"voxel={cfg['voxel_m'] * 1e3:.1f} mm splits={cfg['splits']} "
          f"samples={cache_dir}", flush=True)
    print("  verification checks...", flush=True)
    checks = deterministic_checks(cfg)
    conv = {}
    for f in FREQUENCIES_HZ:
        print(f"  grid convergence: sphere {FREQUENCY_LABEL[f]}", flush=True)
        conv[f] = voxel_convergence_study(f)
    results = {model: run_model(model, cfg, cache_dir) for model in models}
    write_results(results, checks, conv, cfg, outdir)
    if len(models) == len(UNCERTAINTY_MODELS):
        make_figures(results, outdir)
        write_article_tables(results, checks, conv, outdir)
        write_numbers(results, checks, conv, cfg, outdir)
    print("\nIncident power density thresholds S (W/m^2), alpha = 0.05")
    print(f"{'model':15s}{'case':18s}{'VaR':>8}{'tol':>8}{'CVaR':>8}{'Cantelli':>10}{'locRL':>8}{'wbRL':>7}")
    for model, res in results.items():
        for r in res["rows"]:
            print(f"{model:15s}{r['frequency'] + ' ' + r['geometry']:18s}"
                  f"{r['S_var']:8.1f}{r['S_tolerance']:8.1f}{r['S_cvar']:8.1f}"
                  f"{r['S_moment']:10.1f}{r['reference_level_local']:8.1f}"
                  f"{r['reference_level_wb']:7.1f}")
    print(f"\nOutputs written under: {outdir}")
    return results


def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="Reproduce the article results.")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_true", help="check dependencies; no simulation")
    g.add_argument("--fast", action="store_true", help="reduced-size pipeline check")
    g.add_argument("--medium", action="store_true", help="intermediate sample size")
    p.add_argument("--simulate", action="store_true",
                   help="regenerate endpoint samples instead of reading data/samples")
    p.add_argument("--models", default="reference,within_tissue",
                   help="comma-separated uncertainty models")
    p.add_argument("--outdir", default=os.path.join(here, "results"), help="output directory")
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args(argv)
    if args.check:
        return preflight_check()
    mode = "fast" if args.fast else "medium" if args.medium else "full"
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in models:
        if m not in UNCERTAINTY_MODELS:
            p.error(f"unknown model {m!r}")
    if mode == "full" and not args.simulate:
        cache_dir = os.path.join(here, "data", "samples")
    else:
        cache_dir = os.path.join(args.outdir, "samples")
    run(mode, args.outdir, cache_dir, models, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
