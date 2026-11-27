#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Incident power density thresholds for computational radiofrequency dosimetry.

This file reproduces every table and figure of the accompanying article and
the computed values quoted in its text. It contains the electromagnetic and
bioheat solvers, the tissue-property uncertainty models and their input-model
variants, the threshold layer, the empirical tests and the verification checks.

Forward models: transmission-line planar slab, multilayer Mie sphere (via
scattnlay), IEC/IEEE 62704-1-style peak 10 g spatial-average SAR, and Pennes
bioheat in steady state and in time (1-D finite volume for the slab, 3-D voxel
finite volume for the sphere). Verification: transfer-matrix and power-balance
checks of the slab, Mie absorption balance, radial bioheat solution, an
independent scattered-field FDTD solution of the sphere, and grid and averaging
studies. Threshold layer: empirical and conformal quantiles, nonparametric
tolerance limits, CVaR, Cantelli moment bounds and Wasserstein CVaR bounds,
each converted to an incident power density threshold, with bootstrap
intervals and critical Wasserstein radii.

Nominal dielectric spectra use the four-term Cole-Cole parameters of Gabriel,
Lau and Gabriel (Phys. Med. Biol. 41, 1996). Tissue density and thermal ranges
from the IT'IS database and the dielectric uncertainty parameters are stored
as constants, so the script needs no data download.

Usage
-----
    python reproduce.py --check                  # dependency check; no simulation
    python reproduce.py                          # article results from data/samples
    python reproduce.py --simulate --jobs 6      # regenerate all samples, 6 processes
    python reproduce.py --fast --outdir quick    # reduced-size pipeline check
    python reproduce.py --samples FILE --anchor 50 --limit 2 --levels 40,10
                                                 # thresholds for your own samples

License: MIT (see LICENSE).
"""
from __future__ import annotations

import argparse
import contextlib
import ctypes
import json
import os
import platform
import re
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
        "heat_capacity": (3390.5, 3150.0, 3662.0),
    },
    "fat_not_infiltrated": {
        "density": (911.0, 812.0, 961.0),
        "thermal_conductivity": (0.21145, 0.18, 0.2416),
        "perfusion": (32.709, 19.832, 63.248),
        "metabolic": (0.50657, 0.30714, 0.97952),
        "heat_capacity": (2348.33, 1806.0, 2973.0),
    },
    "muscle": {
        "density": (1090.4, 1041.0, 1178.0),
        "thermal_conductivity": (0.49497, 0.4225, 0.56),
        "perfusion": (36.738, 18.537, 94.247),
        "metabolic": (0.90612, 0.45719, 2.3245),
        "heat_capacity": (3421.2, 2624.0, 3799.0),
    },
    "bone_cortical": {
        "density": (1908.0, 1800.0, 2100.0),
        "thermal_conductivity": (0.32, 0.30, 0.36),
        "perfusion": (10.0, 10.0, 10.0),
        "metabolic": (0.15, 0.15, 0.15),
        "heat_capacity": (1312.83, 826.0, 1650.0),
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

# Input-model variants for the sensitivity analysis. Every variant keeps the
# Gaussian copula scores of the main sampling, so the variants and the main
# sample are paired draw by draw. The marginal family sets the distribution of
# each log multiplier: uniform on [-h, h] (main analysis), normal with the
# same variance h^2/3 (log-normal multipliers, unbounded), or normal with scale
# h/2 truncated to [-h, h]. The between-tissue correlation adds a factor shared
# by all tissues in the copula scores; zero gives independent tissues.
INPUT_FAMILIES = ("uniform", "lognormal", "truncnormal")
FAMILY_TAG = {"uniform": "", "lognormal": "_ln", "truncnormal": "_tn"}
FAMILY_LABEL = {"uniform": "Uniform", "lognormal": "Normal", "truncnormal": "Truncated normal"}
TRUNCNORMAL_SCALE = 0.5
TISSUE_CORRELATIONS = (0.5, 0.9)
SHARED_FACTOR_SEED_OFFSET = 13


def variant_tag(family="uniform", tissue_corr=0.0):
    """File-name tag of an input-model variant; empty for the main analysis."""
    tag = FAMILY_TAG[family]
    if tissue_corr > 0.0:
        tag += f"_tc{int(round(100.0 * tissue_corr))}"
    return tag

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


def _voxel_pennes_system(spacing, rho, k_vox, wb_vox):
    """Sparse finite-volume matrix of the Pennes temperature rise on the voxel
    grid and the flat indices of the tissue voxels it acts on.

    Face conductances use the harmonic mean; the air-side conductance is the
    series combination of the half-voxel conduction path and the convection
    film.
    """
    from scipy.sparse import coo_matrix
    shape = rho.shape
    h = float(spacing)
    tissue = rho > 0
    index = -np.ones(shape, dtype=np.int64)
    ids = np.flatnonzero(tissue.ravel())
    index.ravel()[ids] = np.arange(len(ids))
    n = len(ids)
    diag = wb_vox.ravel()[ids] * h ** 3
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
    return m, ids


def solve_voxel_pennes_delta(spacing, rho, sar, k_vox, wb_vox):
    """Steady-state temperature rise on the voxel grid (3-D finite volume).

    Because the Pennes equation is linear, the rise Delta T above the RF-off
    baseline satisfies  -div(k grad dT) + wb dT = rho * SAR  with the ambient
    and arterial terms cancelled; the surface convection enters as a
    homogeneous Robin condition on tissue-air faces.
    """
    from scipy.sparse.linalg import spsolve
    m, ids = _voxel_pennes_system(spacing, rho, k_vox, wb_vox)
    h = float(spacing)
    rhs = (rho * sar).ravel()[ids] * h ** 3
    dT = spsolve(m, rhs)
    out = np.zeros(rho.shape)
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


def _slab_layers(mats, skin=SKIN_M, fat=FAT_M, muscle=MUSCLE_M):
    spec = [("skin_dry", skin), ("fat_not_infiltrated", fat),
            ("muscle", muscle), ("bone_cortical", None)]
    return [{"thickness_m": t, "epsilon_r": mats[n]["epsilon_r"],
             "sigma": mats[n]["sigma"], "density": mats[n]["density"], "name": n}
            for n, t in spec]


def _sphere_layers(mats, radius=SPHERE_RADIUS_M, skin=SKIN_M, fat_t=FAT_M, muscle_t=MUSCLE_M):
    bone = radius - skin - fat_t - muscle_t
    if bone <= 0.0:
        raise ValueError("layer thicknesses exceed the sphere radius")
    muscle = radius - skin - fat_t
    fat = radius - skin
    spec = [("bone_cortical", bone), ("muscle", muscle),
            ("fat_not_infiltrated", fat), ("skin_dry", radius)]
    return [{"outer_radius_m": r, "epsilon_r": mats[n]["epsilon_r"],
             "sigma": mats[n]["sigma"], "density": mats[n]["density"], "name": n}
            for n, r in spec]


SLAB_NAMES = ("skin_dry", "fat_not_infiltrated", "muscle", "bone_cortical")


def _slab_thermal_grid(mats, depth, sar, skin=SKIN_M, fat=FAT_M, muscle=MUSCLE_M):
    """Finite-volume cells (0.1 mm, 80 mm into bone) with their properties and SAR."""
    domain = skin + fat + muscle + 0.08
    n = int(np.ceil(domain / 1e-4))
    x = (np.arange(n) + 0.5) * (domain / n)
    edges = np.array([skin, skin + fat, skin + fat + muscle])
    names = list(SLAB_NAMES)
    idx = np.searchsorted(edges, x, side="right")
    cond = np.array([mats[names[i]]["thermal_conductivity"] for i in idx])
    wb = np.array([perfusion_sink(mats[names[i]]["perfusion"], mats[names[i]]["density"]) for i in idx])
    met = np.array([mats[names[i]]["metabolic"] * mats[names[i]]["density"] for i in idx])
    dens = np.array([mats[names[i]]["density"] for i in idx])
    sar_x = np.interp(x, depth, sar)
    return x, idx, cond, wb, met, dens, sar_x


def slab_temperature_profile(mats, frequency_hz, geometry=None):
    """Depth, SAR and steady-state temperature rise of the slab at the anchor S0."""
    g = geometry or {}
    layers = _slab_layers(mats, **g)
    depth, sar, rho, _ = solve_slab(frequency_hz, layers, ANCHOR_S0_W_PER_M2)
    x, idx, cond, wb, met, dens, sar_x = _slab_thermal_grid(mats, depth, sar, **g)
    base = solve_planar_pennes(x, cond, wb, met)
    heat = solve_planar_pennes(x, cond, wb, met + dens * sar_x)
    return x, sar_x, heat - base


def slab_forward(mats, frequency_hz, endpoint="sar", geometry=None):
    g = geometry or {}
    layers = _slab_layers(mats, **g)
    depth, sar, rho, _ = solve_slab(frequency_hz, layers, ANCHOR_S0_W_PER_M2)
    sar_10g = planar_peak_10g(depth, sar, rho)
    if endpoint == "sar":
        return np.array([sar_10g])
    x, _idx, cond, wb, met, dens, sar_x = _slab_thermal_grid(mats, depth, sar, **g)
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


def sphere_temperature_field(mats, frequency_hz, voxel_m=0.002, geometry=None):
    """Voxel model, SAR and steady-state temperature rise of the sphere at S0."""
    layers = _sphere_layers(mats, **(geometry or {}))
    edge, spacing, rho, sar = voxelize_sphere(layers, frequency_hz, ANCHOR_S0_W_PER_M2, voxel_m)
    k_vox, wb_vox = _sphere_voxel_thermal_fields(layers, mats, edge, spacing, rho.shape)
    k_vox = np.where(rho > 0, k_vox, 0.0)
    wb_vox = np.where(rho > 0, wb_vox, 0.0)
    dT = solve_voxel_pennes_delta(spacing, rho, sar, k_vox, wb_vox)
    return edge, spacing, rho, sar, dT


def sphere_forward(mats, frequency_hz, endpoint="sar", voxel_m=0.002, geometry=None):
    layers = _sphere_layers(mats, **(geometry or {}))
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
def _log_offsets(z, half_width, family):
    """Log multipliers with the requested marginal family from copula scores z."""
    from scipy.stats import norm
    if family == "uniform":
        return (2.0 * norm.cdf(z) - 1.0) * half_width
    if family == "lognormal":
        return z * (half_width / np.sqrt(3.0))
    if family == "truncnormal":
        scale = TRUNCNORMAL_SCALE * half_width
        lo, hi = norm.cdf(-half_width / scale), norm.cdf(half_width / scale)
        return scale * norm.ppf(lo + norm.cdf(z) * (hi - lo))
    raise ValueError(f"unknown input family {family!r}")


def sample_materials(frequency_hz, n, seed, endpoint="sar", model="reference",
                     family="uniform", tissue_corr=0.0):
    """Draw n material realizations from the selected uncertainty model.

    Permittivity and conductivity of each tissue are perturbed inside the
    frequency-specific dielectric log box with a Gaussian copula (correlation
    rho), giving uniform log marginals in the main analysis. ``family`` selects
    another marginal for the log multipliers and ``tissue_corr`` adds a factor
    shared by all tissues (see INPUT_FAMILIES); both variants reuse the copula
    scores of the main analysis. Density, and for the thermal endpoint the
    thermal fields, are independent uniforms over their documented ranges.
    """
    from scipy.stats import norm
    rng = np.random.default_rng(seed)
    nom = nominal_materials(frequency_hz)
    box = UNCERTAINTY_MODELS[model][frequency_hz]
    half_e, half_s = box["log_half_width"]
    rho = float(np.clip(box["correlation"], -0.995, 0.995))
    if not 0.0 <= tissue_corr < 1.0:
        raise ValueError("tissue_corr must lie in [0, 1)")
    if tissue_corr > 0.0:
        shared = np.random.default_rng(seed + SHARED_FACTOR_SEED_OFFSET)
        g_e = shared.standard_normal(n)
        g_s = rho * g_e + np.sqrt(1 - rho * rho) * shared.standard_normal(n)
        mix_shared, mix_own = np.sqrt(tissue_corr), np.sqrt(1.0 - tissue_corr)

    samples = [dict((t, dict(nom[t])) for t in TISSUE_ORDER) for _ in range(n)]
    for tissue in TISSUE_ORDER:
        u_e = np.clip(rng.random(n), 1e-12, 1 - 1e-12)
        u_s = np.clip(rng.random(n), 1e-12, 1 - 1e-12)
        z_e = norm.ppf(u_e)
        z_s = rho * z_e + np.sqrt(1 - rho * rho) * norm.ppf(u_s)
        if tissue_corr > 0.0:
            z_e = mix_shared * g_e + mix_own * z_e
            z_s = mix_shared * g_s + mix_own * z_s
        off_e = _log_offsets(z_e, half_e, family)
        off_s = _log_offsets(z_s, half_s, family)
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
def sample_path(case, frequency_hz, n, seed, endpoint, voxel_m, cache_dir,
                model="reference", family="uniform", tissue_corr=0.0):
    """Cache file of one endpoint sample (the main analysis has no variant tag)."""
    vtag = f"_v{int(round(voxel_m * 1e4))}" if case == "sphere" else ""
    if endpoint != "sar":
        vtag += "_t3d"
    tag = (f"{case}_{FREQUENCY_LABEL[frequency_hz].replace(' ', '').replace('.', 'p')}"
           f"_{endpoint}_N{n}{vtag}_s{seed}{MODEL_TAG[model]}"
           f"{variant_tag(family, tissue_corr)}")
    return os.path.join(cache_dir, tag + ".npz")


def monte_carlo(case, frequency_hz, n, seed, endpoint, voxel_m, cache_dir,
                model="reference", verbose=True, family="uniform", tissue_corr=0.0):
    """Evaluate the forward model over n draws; store and reuse endpoint samples."""
    path = sample_path(case, frequency_hz, n, seed, endpoint, voxel_m, cache_dir,
                       model, family, tissue_corr)
    if os.path.exists(path):
        data = np.load(path)
        return data["outputs"], data["nominal"]
    mats = sample_materials(frequency_hz, n, seed, endpoint=endpoint, model=model,
                            family=family, tissue_corr=tissue_corr)
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
            print(f"    [{model}{variant_tag(family, tissue_corr)} {case} "
                  f"{FREQUENCY_LABEL[frequency_hz]} {endpoint}] "
                  f"{i + 1}/{n}  {rate:.2f}s/eval", flush=True)
    np.savez_compressed(path, outputs=outputs, nominal=nominal)
    return outputs, nominal


def grid_subset_path(frequency_hz, cfg, cache_dir):
    tag = (f"sphere_{FREQUENCY_LABEL[frequency_hz].replace(' ', '').replace('.', 'p')}"
           f"_sar_N{cfg['n_sar']}_first{cfg['grid_subset']}"
           f"_v{int(round(cfg['voxel_m'] * 1e4))}_v{int(round(cfg['grid_fine_m'] * 1e4))}"
           f"_s{cfg['seed']}")
    return os.path.join(cache_dir, tag + ".npz")


def grid_subset_sample(frequency_hz, cfg, cache_dir):
    """Sphere peak 10 g SAR of the first draws of the main reference-model sample
    on the production grid and on a finer grid, draw by draw."""
    path = grid_subset_path(frequency_hz, cfg, cache_dir)
    if os.path.exists(path):
        data = np.load(path)
        return data["production"], data["fine"]
    n_sub = cfg["grid_subset"]
    mats = sample_materials(frequency_hz, cfg["n_sar"], cfg["seed"], "sar", "reference")[:n_sub]
    production, fine = np.empty(n_sub), np.empty(n_sub)
    for i, m in enumerate(mats):
        production[i] = sphere_forward(m, frequency_hz, "sar", voxel_m=cfg["voxel_m"])[0]
        fine[i] = sphere_forward(m, frequency_hz, "sar", voxel_m=cfg["grid_fine_m"])[0]
    np.savez_compressed(path, production=production, fine=fine)
    return production, fine


def simulation_tasks(cfg, models):
    """Every endpoint sample of the study, as (kind, case, frequency, n, endpoint,
    model, family, between-tissue correlation)."""
    tasks = []
    for model in models:
        for f in FREQUENCIES_HZ:
            for case in CASES:
                for fam in cfg["families"]:
                    tasks.append(("mc", case, f, cfg["n_sar"], "sar", model, fam, 0.0))
                for tc in cfg["tissue_corrs"]:
                    tasks.append(("mc", case, f, cfg["n_sar"], "sar", model, "uniform", tc))
        for f in cfg["temp_frequencies"]:
            for case in CASES:
                tasks.append(("mc", case, f, cfg["n_temp"], "both", model, "uniform", 0.0))
    if "reference" in models:
        for f in FREQUENCIES_HZ:
            tasks.append(("grid", "sphere", f, cfg["grid_subset"], "sar", "reference",
                          "uniform", 0.0))
    for name in STUDY_NAMES:
        tasks.append(("study", name, 0.0, 0, "", "", "", 0.0))
    return tasks


STUDY_COST_S = {"study_grid_averaging": 2400.0, "study_fdtd": 4800.0, "study_transient": 300.0,
                "study_fdtd_margin": 300.0,
                "study_geometry": 150.0, "study_perfusion": 80.0, "study_temperature_fields": 20.0}


def _task_path(task, cfg, cache_dir):
    kind, case, f, n, endpoint, model, fam, tc = task
    if kind == "study":
        return os.path.join(cache_dir, case + ".json")
    if kind == "grid":
        return grid_subset_path(f, cfg, cache_dir)
    return sample_path(case, f, n, cfg["seed"], endpoint, cfg["voxel_m"], cache_dir,
                       model, fam, tc)


def _task_cost(task):
    kind, case, _f, n, endpoint = task[:5]
    if kind == "study":
        return STUDY_COST_S[case]
    if kind == "grid":
        return 10.0 * n
    if case == "sphere":
        return (2.1 if endpoint != "sar" else 0.85) * n
    return 0.04 * n


def _run_task(args):
    task, cfg, cache_dir = args
    kind, case, f, n, endpoint, model, fam, tc = task
    t0 = time.time()
    if kind == "study":
        cached_study(case, cache_dir, _study_compute(case, cfg))
    elif kind == "grid":
        grid_subset_sample(f, cfg, cache_dir)
    else:
        monte_carlo(case, f, n, cfg["seed"], endpoint, cfg["voxel_m"], cache_dir,
                    model=model, verbose=False, family=fam, tissue_corr=tc)
    return task, time.time() - t0


def generate_samples(cfg, models, cache_dir, jobs=1):
    """Generate every missing endpoint sample, in parallel processes if jobs > 1.

    Each sample depends only on its own settings and the seed, so the files are
    identical whatever the number of processes or the order of completion.
    """
    pending = [t for t in simulation_tasks(cfg, models)
               if not os.path.exists(_task_path(t, cfg, cache_dir))]
    pending.sort(key=_task_cost, reverse=True)
    if not pending:
        return
    print(f"  generating {len(pending)} endpoint samples with {jobs} process(es)", flush=True)
    work = [(t, cfg, cache_dir) for t in pending]
    t0 = time.time()
    if jobs <= 1:
        results = map(_run_task, work)
        done = list(_report_tasks(results, len(work), t0))
    else:
        import multiprocessing
        with multiprocessing.get_context("spawn").Pool(jobs) as pool:
            done = list(_report_tasks(pool.imap_unordered(_run_task, work), len(work), t0))
    return done


def _report_tasks(results, total, t0):
    for k, (task, seconds) in enumerate(results, 1):
        kind, case, f, n, endpoint, model, fam, tc = task
        label = (case if kind == "study" else
                 f"{model}{variant_tag(fam, tc)} {case} {FREQUENCY_LABEL[f]} {endpoint} N={n}")
        print(f"    [{k}/{total}] {kind} {label}: {seconds / 60.0:.1f} min "
              f"(elapsed {(time.time() - t0) / 3600.0:.2f} h)", flush=True)
        yield task, seconds


# ----------------------------------------------------------------------------
# Section 10. Threshold layer: quantile, tolerance, tail and robust bounds
# ----------------------------------------------------------------------------
def empirical_var(y, alpha=ALPHA):
    """Conservative empirical (1-alpha) upper quantile."""
    return float(np.quantile(y, 1.0 - alpha, method="higher"))


def cvar_rank(n, alpha=ALPHA):
    """Rank (1-based) of the order statistic that minimizes the Rockafellar-Uryasev
    formula for an empirical distribution of n draws, ceil(n(1 - alpha))."""
    return max(1, int(np.ceil(n * (1.0 - alpha) - 1e-9)))


def cvar_upper(y, alpha=ALPHA):
    """Upper-tail CVaR of the empirical distribution (Rockafellar-Uryasev formula
    evaluated at its minimizer); equals the mean of the n*alpha largest values when
    n*alpha is an integer."""
    y = np.asarray(y, dtype=float)
    tau = float(np.sort(y)[cvar_rank(len(y), alpha) - 1])
    return float(tau + np.maximum(y - tau, 0.0).mean() / alpha)


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
    tau = float(np.sort(y)[cvar_rank(len(y), alpha) - 1])
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
    checks["thermal_3d_vs_radial"] = {FREQUENCY_LABEL[f]: thermal_solver_check(f, cfg["voxel_m"])
                                      for f in FREQUENCIES_HZ}
    checks["nominal_dielectric"] = {
        FREQUENCY_LABEL[f]: {t: {"epsilon_r": tissue_permittivity(t, f)[0],
                                 "sigma": tissue_permittivity(t, f)[1]}
                             for t in TISSUE_ORDER}
        for f in FREQUENCIES_HZ}
    return checks


# ----------------------------------------------------------------------------
# Section 12b. Averaging procedure, grid and geometry studies (nominal properties)
# ----------------------------------------------------------------------------
AIR_LIMITS = (0.0, 0.05, 0.10, 0.20)
GRID_STUDY_VOXELS_M = (0.00375, 0.003, 0.0025, 0.002, 0.0015, 0.00125, 0.001,
                       0.00075, 0.0005)
GEOMETRY_SWEEP_MM = {
    "slab_fat": (2.0, 3.5, 5.0, 7.5, 10.0),
    "slab_skin": (1.0, 1.5, 2.0),
    "sphere_fat": (2.0, 3.5, 5.0, 7.5, 10.0),
    "sphere_radius": (25.0, 30.0, 35.0, 40.0, 50.0),
}


def sphere_peak_10g_variants(edge, spacing, rho, sar, target_mass_kg=TARGET_MASS_KG,
                             batch=20000):
    """Peak 10 g SAR of the sphere under the production rule and its variants.

    production: cubes centered on tissue voxels with at most 10% air and tissue
    within one voxel spacing inside every face (identical to sphere_peak_10g).
    face_added: the production cubes together with cubes that have one face
    centered on a tissue voxel, in each of the six axis directions, under the
    same rules. air_0, air_5, air_20: centered cubes with at most 0%, 5% or 20%
    of their volume outside tissue.
    """
    integ = _VoxelIntegrals(edge, spacing, rho, sar)
    body = np.argwhere(rho > 0)
    axes = [edge + (np.arange(n) + 0.5) * spacing for n in rho.shape]
    centers = np.column_stack((axes[0][body[:, 0]], axes[1][body[:, 1]], axes[2][body[:, 2]]))
    best = {lim: -np.inf for lim in AIR_LIMITS}
    best_face = -np.inf
    for s in range(0, len(centers), batch):
        c = centers[s:s + batch]
        for orientation in range(7):
            side, mass, power, vol, lo, hi = integ.solve_side(c, target_mass_kg, orientation)
            background = 1.0 - vol / side ** 3
            touch = integ.faces_touch(lo, hi)
            avg = power / mass
            limits = AIR_LIMITS if orientation == 0 else (0.10,)
            for lim in limits:
                valid = np.isfinite(side) & (background < lim + 1e-10) & touch
                if not np.any(valid):
                    continue
                peak = float(np.nanmax(avg[valid]))
                if orientation == 0:
                    best[lim] = max(best[lim], peak)
                else:
                    best_face = max(best_face, peak)
    out = {f"air_{int(round(100 * lim))}": best[lim] for lim in AIR_LIMITS}
    out["production"] = best[0.10]
    out["face_added"] = max(best[0.10], best_face)
    return out


def grid_averaging_study(frequency_hz, voxels=GRID_STUDY_VOXELS_M):
    """Nominal-property peak 10 g SAR of the sphere versus voxel size, for the
    production averaging rule and its variants."""
    layers = _sphere_layers(nominal_materials(frequency_hz))
    rows = []
    for vm in voxels:
        edge, sp, rho, sar = voxelize_sphere(layers, frequency_hz, ANCHOR_S0_W_PER_M2, vm)
        rows.append(dict(voxel_mm=sp * 1e3, cells=int(rho.shape[0]),
                         **sphere_peak_10g_variants(edge, sp, rho, sar)))
    return rows


def geometry_study(voxel_m):
    """Nominal-property peak 10 g SAR when one geometric parameter changes.

    Slab: fat or skin thickness. Sphere: fat thickness at the 30 mm outer
    radius, with the bone core absorbing the change, or the outer radius with
    the skin, fat and muscle thicknesses fixed.
    """
    out = {}
    for f in FREQUENCIES_HZ:
        nom = nominal_materials(f)
        rec = {
            "slab_fat": [float(slab_forward(nom, f, geometry={"fat": v * 1e-3})[0])
                         for v in GEOMETRY_SWEEP_MM["slab_fat"]],
            "slab_skin": [float(slab_forward(nom, f, geometry={"skin": v * 1e-3})[0])
                          for v in GEOMETRY_SWEEP_MM["slab_skin"]],
            "sphere_fat": [float(sphere_forward(nom, f, voxel_m=voxel_m,
                                                geometry={"fat_t": v * 1e-3})[0])
                           for v in GEOMETRY_SWEEP_MM["sphere_fat"]],
            "sphere_radius": [float(sphere_forward(nom, f, voxel_m=voxel_m,
                                                   geometry={"radius": v * 1e-3})[0])
                              for v in GEOMETRY_SWEEP_MM["sphere_radius"]],
        }
        out[FREQUENCY_LABEL[f]] = rec
    return {"values_mm": {k: list(v) for k, v in GEOMETRY_SWEEP_MM.items()}, "peak_10g": out}


# ----------------------------------------------------------------------------
# Section 12c. Transient response, perfusion and temperature fields
# ----------------------------------------------------------------------------
TRANSIENT_STEP_S = 1.0
TRANSIENT_END_S = 3600.0


def _planar_pennes_system(x, k, wb):
    """Tridiagonal finite-volume matrix of the planar temperature rise (Robin
    condition at the surface, zero rise at the deep boundary)."""
    from scipy.sparse import diags
    n = len(x)
    dx = float(np.mean(np.diff(x)))
    k_face = _harmonic(k[:-1], k[1:])
    west = np.zeros(n); east = np.zeros(n)
    west[1:] = k_face / dx; east[:-1] = k_face / dx
    g_surface = 1.0 / (0.5 * dx / k[0] + 1.0 / CONVECTION_W_PER_M2_K)
    g_deep = k[-1] / (0.5 * dx)
    diag = west + east + wb * dx
    diag[0] += g_surface; diag[-1] += g_deep
    return diags((-west[1:], diag, -east[:-1]), offsets=(-1, 0, 1), format="csc"), dx


def _step_response(matrix, capacity, source, step_s=TRANSIENT_STEP_S, end_s=TRANSIENT_END_S):
    """Peak temperature rise after the source is switched on at t = 0 from the
    field-off state (implicit Euler), and the steady-state peak rise."""
    from scipy.sparse import diags
    from scipy.sparse.linalg import splu, spsolve
    lu = splu((matrix + diags(capacity / step_s)).tocsc())
    steps = int(round(end_s / step_s))
    temp = np.zeros(len(source))
    peaks = np.empty(steps + 1)
    peaks[0] = 0.0
    for i in range(1, steps + 1):
        temp = lu.solve(capacity / step_s * temp + source)
        peaks[i] = temp.max()
    steady = float(np.max(spsolve(matrix.tocsc(), source)))
    return np.arange(steps + 1) * step_s, peaks, steady


def _response_times(times, ratio):
    out = {}
    for level in (0.63, 0.90):
        reached = ratio >= level
        if not np.any(reached):
            out[f"t{int(round(100 * level))}_s"] = float("nan")
            continue
        k = int(np.argmax(reached))
        out[f"t{int(round(100 * level))}_s"] = float(np.interp(level, ratio[k - 1:k + 1],
                                                               times[k - 1:k + 1]))
    for t in (360.0, 1800.0):
        out[f"ratio_at_{int(t)}_s"] = float(np.interp(t, times, ratio))
    return out


def transient_study(voxel_m):
    """Step response of the peak temperature rise at nominal properties."""
    out = {}
    for f in FREQUENCIES_HZ:
        nom = nominal_materials(f)
        cap = {t: TISSUE_TABLE[t]["heat_capacity"][0] for t in TISSUE_ORDER}
        layers = _slab_layers(nom)
        depth, sar, _rho, _ = solve_slab(f, layers, ANCHOR_S0_W_PER_M2)
        x, idx, cond, wb, _met, dens, sar_x = _slab_thermal_grid(nom, depth, sar)
        matrix, dx = _planar_pennes_system(x, cond, wb)
        capacity = dens * np.array([cap[SLAB_NAMES[i]] for i in idx]) * dx
        t, peaks, steady = _step_response(matrix, capacity, dens * sar_x * dx)
        rec = {"slab": dict(steady_K=steady, **_response_times(t, peaks / steady))}
        curves = {"slab": (peaks / steady)}
        slayers = _sphere_layers(nom)
        edge, sp, rho, vsar = voxelize_sphere(slayers, f, ANCHOR_S0_W_PER_M2, voxel_m)
        k_vox, wb_vox = _sphere_voxel_thermal_fields(slayers, nom, edge, sp, rho.shape)
        k_vox = np.where(rho > 0, k_vox, 0.0)
        wb_vox = np.where(rho > 0, wb_vox, 0.0)
        matrix, ids = _voxel_pennes_system(sp, rho, k_vox, wb_vox)
        radius = np.linalg.norm(_voxel_centers(edge, sp, rho.shape), axis=1)[ids]
        outer = np.array([l["outer_radius_m"] for l in slayers])
        li = np.clip(np.searchsorted(outer, radius, side="left"), 0, len(slayers) - 1)
        c_vox = np.array([cap[slayers[i]["name"]] for i in li])
        capacity = rho.ravel()[ids] * c_vox * sp ** 3
        source = (rho * vsar).ravel()[ids] * sp ** 3
        t, peaks, steady = _step_response(matrix, capacity, source)
        rec["sphere"] = dict(steady_K=steady, **_response_times(t, peaks / steady))
        curves["sphere"] = peaks / steady
        keep = np.unique(np.concatenate((np.arange(0, 600, 5), np.arange(600, len(t), 30),
                                         [len(t) - 1])))
        rec["curve_time_s"] = t[keep].tolist()
        rec["curve_ratio"] = {g: c[keep].tolist() for g, c in curves.items()}
        out[FREQUENCY_LABEL[f]] = rec
    return out


def _voxel_centers(edge, spacing, shape):
    axis = edge + (np.arange(shape[0]) + 0.5) * spacing
    xx, yy, zz = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))


def _with_perfusion(mats, setting):
    """Material set with every tissue perfusion at zero, its database minimum,
    nominal or maximum value."""
    out = {t: dict(v) for t, v in mats.items()}
    for t in TISSUE_ORDER:
        nom, mn, mx = TISSUE_TABLE[t]["perfusion"]
        out[t]["perfusion"] = {"zero": 0.0, "minimum": mn, "nominal": nom, "maximum": mx}[setting]
    return out


def perfusion_study(voxel_m):
    """Nominal peak temperature rise with the perfusion of every tissue at zero,
    its database minimum, nominal value and maximum."""
    out = {}
    for f in FREQUENCIES_HZ:
        nom = nominal_materials(f)
        rec = {}
        for setting in ("zero", "minimum", "nominal", "maximum"):
            mats = _with_perfusion(nom, setting)
            rec[setting] = {"slab": float(slab_forward(mats, f, "both")[1]),
                            "sphere": float(sphere_forward(mats, f, "both", voxel_m=voxel_m)[1])}
        out[FREQUENCY_LABEL[f]] = rec
    return out


def temperature_field_study(voxel_m):
    """Nominal SAR and temperature-rise fields for the temperature figure: slab
    depth profiles and the sphere cross-section through the center that contains
    the propagation (z) and polarization (x) directions."""
    out = {}
    for f in FREQUENCIES_HZ:
        nom = nominal_materials(f)
        x, sar_x, dT_x = slab_temperature_profile(nom, f)
        keep = x <= 0.04
        edge, sp, rho, sar, dT = sphere_temperature_field(nom, f, voxel_m)
        mid = rho.shape[1] // 2
        out[FREQUENCY_LABEL[f]] = {
            "slab_depth_mm": (1e3 * x[keep]).tolist(), "slab_sar": sar_x[keep].tolist(),
            "slab_dT": dT_x[keep].tolist(),
            "sphere_edge_mm": edge * 1e3, "sphere_spacing_mm": sp * 1e3,
            "sphere_dT_xz": (0.5 * (dT[:, mid - 1, :] + dT[:, mid, :])).tolist(),
            "sphere_tissue_xz": (rho[:, mid, :] > 0).astype(int).tolist(),
            "sphere_peak_dT": float(dT.max()),
        }
    return out


def cached_study(name, cache_dir, compute):
    """Load a deterministic study from the sample folder, or compute and store it."""
    path = os.path.join(cache_dir, name + ".json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    data = compute()
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)
    return data


# ----------------------------------------------------------------------------
# Section 12d. Independent finite-difference time-domain (FDTD) check
# ----------------------------------------------------------------------------
# Scattered-field FDTD on a uniform Yee grid with a convolutional perfectly
# matched layer (CPML) terminated by a perfect conductor. Each electric-field
# component takes the layer that contains its own grid position. The incident
# plane wave travels along +z with the electric field along x, as in the Mie
# solution, and is switched on with a raised-cosine ramp. Phasors are obtained
# by a discrete Fourier transform over whole periods after the transient.
FDTD_SPACING_M = {0.9e9: 0.001, 2.45e9: 0.001, 5.8e9: 0.0005}
FDTD_MARGIN_M = 0.015
FDTD_PML_CELLS = 12


class _CPMLAxis:
    """CPML coefficients for the components at the given positions along one axis."""

    def __init__(self, positions, lo_inner, hi_inner, depth, dt, kappa_max=5.0,
                 alpha_max=0.05, order=3, h=1.0):
        sigma_max = 0.8 * (order + 1) / (ETA0 * h)
        d = np.zeros(len(positions))
        lo = positions < lo_inner
        hi = positions > hi_inner
        d[lo] = (lo_inner - positions[lo]) / depth
        d[hi] = (positions[hi] - hi_inner) / depth
        d = np.clip(d, 0.0, 1.0)
        sigma = sigma_max * d ** order
        kappa = 1.0 + (kappa_max - 1.0) * d ** order
        alpha = alpha_max * (1.0 - d)
        b = np.exp(-(sigma / kappa + alpha) * dt / EPS0)
        with np.errstate(invalid="ignore", divide="ignore"):
            c = np.where(sigma > 0, sigma * (b - 1.0) / (sigma * kappa + kappa ** 2 * alpha), 0.0)
        self.index = np.flatnonzero(lo | hi)
        self.b = b[self.index].astype(np.float32)
        self.c = c[self.index].astype(np.float32)
        self.inv_kappa = (1.0 / kappa[self.index]).astype(np.float32)


class _CPMLTerm:
    """One spatial-derivative term of a curl with its CPML memory variable."""

    def __init__(self, axis_coeffs, array_axis, shape):
        self.k = axis_coeffs
        self.axis = array_axis
        pshape = list(shape)
        pshape[array_axis] = len(axis_coeffs.index)
        self.psi = np.zeros(pshape, dtype=np.float32)
        bshape = [1, 1, 1]
        bshape[array_axis] = len(axis_coeffs.index)
        self.b = axis_coeffs.b.reshape(bshape)
        self.c = axis_coeffs.c.reshape(bshape)
        self.inv_kappa = axis_coeffs.inv_kappa.reshape(bshape)

    def apply(self, deriv):
        idx = [slice(None)] * 3
        idx[self.axis] = self.k.index
        sub = deriv[tuple(idx)]
        self.psi = self.b * self.psi + self.c * sub
        deriv[tuple(idx)] = self.inv_kappa * sub + self.psi
        return deriv


def fdtd_sphere(frequency_hz, spacing_m, margin_m=FDTD_MARGIN_M, pml_cells=FDTD_PML_CELLS,
                ramp_periods=3, settle_periods=7, dft_periods=3):
    """Total-field phasors of the nominal layered sphere from the FDTD solution,
    returned as cell-centered SAR on the voxel grid of voxelize_sphere and as the
    absorbed power."""
    nom = nominal_materials(frequency_hz)
    layers = _sphere_layers(nom)
    radius = SPHERE_RADIUS_M
    h = float(spacing_m)
    n0 = int(round(2.0 * radius / h))
    if n0 % 2 or abs(n0 * h - 2.0 * radius) > 1e-12:
        raise ValueError("the FDTD spacing must divide the sphere diameter into an even count")
    m, p = int(round(margin_m / h)), int(pml_cells)
    nc = n0 + 2 * (m + p)
    origin = -radius - (m + p) * h
    nodes = origin + np.arange(nc + 1) * h
    halves = origin + (np.arange(nc) + 0.5) * h
    omega = 2.0 * np.pi * frequency_hz
    period = 1.0 / frequency_hz
    per = int(np.ceil(period / (0.99 * h / (C0 * np.sqrt(3.0)))))
    dt = period / per
    e0 = np.sqrt(2.0 * ETA0 * ANCHOR_S0_W_PER_M2)
    outer = np.array([l["outer_radius_m"] for l in layers])
    eps_layer = np.array([l["epsilon_r"] for l in layers])
    sig_layer = np.array([l["sigma"] for l in layers])

    def material(xs, ys, zs):
        xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
        r = np.sqrt(xx * xx + yy * yy + zz * zz)
        li = np.searchsorted(outer, r, side="left")
        inside = li < len(layers)
        li = np.clip(li, 0, len(layers) - 1)
        eps = np.where(inside, eps_layer[li], 1.0) * EPS0
        sig = np.where(inside, sig_layer[li], 0.0)
        return eps, sig

    i0 = m + p
    sl = {"Ex": (slice(i0, i0 + n0), slice(i0, i0 + n0 + 1), slice(i0, i0 + n0 + 1)),
          "Ey": (slice(i0, i0 + n0 + 1), slice(i0, i0 + n0), slice(i0, i0 + n0 + 1)),
          "Ez": (slice(i0, i0 + n0 + 1), slice(i0, i0 + n0 + 1), slice(i0, i0 + n0))}
    pos = {"Ex": (halves, nodes, nodes), "Ey": (nodes, halves, nodes), "Ez": (nodes, nodes, halves)}
    coef, mats = {}, {}
    for name, (xs, ys, zs) in pos.items():
        eps, sig = material(xs, ys, zs)
        loss = sig * dt / (2.0 * eps)
        coef[name] = ((1.0 - loss) / (1.0 + loss)).astype(np.float32), \
                     (dt / eps / (1.0 + loss) / h).astype(np.float32)
        mats[name] = (eps[sl[name]].copy(), sig[sl[name]].copy())
        del eps, sig, loss
    ex = np.zeros((nc, nc + 1, nc + 1), np.float32)
    ey = np.zeros((nc + 1, nc, nc + 1), np.float32)
    ez = np.zeros((nc + 1, nc + 1, nc), np.float32)
    hx = np.zeros((nc + 1, nc, nc), np.float32)
    hy = np.zeros((nc, nc + 1, nc), np.float32)
    hz = np.zeros((nc, nc, nc + 1), np.float32)
    lo_inner, hi_inner, depth = nodes[p], nodes[nc - p], p * h
    ax_h = _CPMLAxis(halves, lo_inner, hi_inner, depth, dt, h=h)
    inner = _CPMLAxis(nodes[1:-1], lo_inner, hi_inner, depth, dt, h=h)
    # H updates: derivatives of E evaluated at H positions.
    t_hx_y = _CPMLTerm(ax_h, 1, hx.shape); t_hx_z = _CPMLTerm(ax_h, 2, hx.shape)
    t_hy_z = _CPMLTerm(ax_h, 2, hy.shape); t_hy_x = _CPMLTerm(ax_h, 0, hy.shape)
    t_hz_x = _CPMLTerm(ax_h, 0, hz.shape); t_hz_y = _CPMLTerm(ax_h, 1, hz.shape)
    # E updates (interior nodes only): derivatives of H at E positions.
    sx = (nc, nc - 1, nc - 1); sy = (nc - 1, nc, nc - 1); sz = (nc - 1, nc - 1, nc)
    t_ex_y = _CPMLTerm(inner, 1, sx); t_ex_z = _CPMLTerm(inner, 2, sx)
    t_ey_z = _CPMLTerm(inner, 2, sy); t_ey_x = _CPMLTerm(inner, 0, sy)
    t_ez_x = _CPMLTerm(inner, 0, sz); t_ez_y = _CPMLTerm(inner, 1, sz)
    ca_x, cb_x = coef["Ex"][0][:, 1:-1, 1:-1], coef["Ex"][1][:, 1:-1, 1:-1]
    ca_y, cb_y = coef["Ey"][0][1:-1, :, 1:-1], coef["Ey"][1][1:-1, :, 1:-1]
    ca_z, cb_z = coef["Ez"][0][1:-1, 1:-1, :], coef["Ez"][1][1:-1, 1:-1, :]
    # Incident-field source on the Ex components inside the sphere's bounding box.
    blk = sl["Ex"]
    eps_b, sig_b = mats["Ex"]
    cbs = (coef["Ex"][1][blk] * h).astype(np.float64)
    src_eps, src_sig = cbs * (eps_b - EPS0), cbs * sig_b
    z_src = nodes[blk[2]]
    z_start = -radius - m * h
    ramp_t = ramp_periods * period

    def incident(t):
        tau = t - (z_src - z_start) / C0
        on = tau > 0
        r = np.where(tau >= ramp_t, 1.0, 0.5 * (1.0 - np.cos(np.pi * np.clip(tau, 0, None) / ramp_t)))
        dr = np.where((tau > 0) & (tau < ramp_t),
                      0.5 * np.pi / ramp_t * np.sin(np.pi * np.clip(tau, 0, None) / ramp_t), 0.0)
        val = np.where(on, e0 * r * np.sin(omega * tau), 0.0)
        der = np.where(on, e0 * (dr * np.sin(omega * tau) + r * omega * np.cos(omega * tau)), 0.0)
        return val, der

    total = (ramp_periods + settle_periods + dft_periods) * per
    start_dft = (ramp_periods + settle_periods) * per
    acc = {"Ex": np.zeros((n0, n0 + 1, n0 + 1), np.complex128),
           "Ey": np.zeros((n0 + 1, n0, n0 + 1), np.complex128),
           "Ez": np.zeros((n0 + 1, n0 + 1, n0), np.complex128)}
    ch = np.float32(dt / (MU0 * h))
    for n in range(total):
        hx -= ch * (t_hx_y.apply(np.diff(ez, axis=1)) - t_hx_z.apply(np.diff(ey, axis=2)))
        hy -= ch * (t_hy_z.apply(np.diff(ex, axis=2)) - t_hy_x.apply(np.diff(ez, axis=0)))
        hz -= ch * (t_hz_x.apply(np.diff(ey, axis=0)) - t_hz_y.apply(np.diff(ex, axis=1)))
        ex[:, 1:-1, 1:-1] = ca_x * ex[:, 1:-1, 1:-1] + cb_x * (
            t_ex_y.apply(np.diff(hz, axis=1)[:, :, 1:-1]) - t_ex_z.apply(np.diff(hy, axis=2)[:, 1:-1, :]))
        ey[1:-1, :, 1:-1] = ca_y * ey[1:-1, :, 1:-1] + cb_y * (
            t_ey_z.apply(np.diff(hx, axis=2)[1:-1, :, :]) - t_ey_x.apply(np.diff(hz, axis=0)[:, :, 1:-1]))
        ez[1:-1, 1:-1, :] = ca_z * ez[1:-1, 1:-1, :] + cb_z * (
            t_ez_x.apply(np.diff(hy, axis=0)[:, 1:-1, :]) - t_ez_y.apply(np.diff(hx, axis=1)[1:-1, :, :]))
        val, der = incident((n + 0.5) * dt)
        ex[blk] -= (src_eps * der[None, None, :] + src_sig * val[None, None, :]).astype(np.float32)
        if n + 1 > start_dft:
            phase = np.exp(-1j * omega * (n + 1) * dt)
            for name, arr in (("Ex", ex), ("Ey", ey), ("Ez", ez)):
                acc[name] += arr[sl[name]] * phase
    k0 = omega / C0
    for name in acc:
        acc[name] *= 2.0 / (dft_periods * per)
    z_ex = nodes[sl["Ex"][2]]
    acc["Ex"] += (-1j * e0 * np.exp(-1j * k0 * (z_ex - z_start)))[None, None, :]
    # Power density of each edge with its own conductivity, assigned to the
    # four cells that share the edge; this keeps the cell powers summing to
    # the absorbed power and avoids mixing fields across tissue interfaces.
    power = 0.0
    cell_power = 0.0
    for name, arr in acc.items():
        _eps, sig = mats[name]
        w = 0.5 * sig * np.abs(arr) ** 2
        power += float(np.sum(w)) * h ** 3
        if name == "Ex":
            cell_power = cell_power + 0.25 * (w[:, :-1, :-1] + w[:, 1:, :-1] + w[:, :-1, 1:] + w[:, 1:, 1:])
        elif name == "Ey":
            cell_power = cell_power + 0.25 * (w[:-1, :, :-1] + w[1:, :, :-1] + w[:-1, :, 1:] + w[1:, :, 1:])
        else:
            cell_power = cell_power + 0.25 * (w[:-1, :-1, :] + w[1:, :-1, :] + w[:-1, 1:, :] + w[1:, 1:, :])
    edge, sp, rho, sar_mie = voxelize_sphere(layers, frequency_hz, ANCHOR_S0_W_PER_M2, h)
    if rho.shape != cell_power.shape or abs(sp - h) > 1e-15:
        raise RuntimeError("FDTD and voxel grids do not coincide")
    sar_fdtd = np.where(rho > 0, cell_power / np.where(rho > 0, rho, 1.0), 0.0)
    _qext, qabs = sphere_qabs(layers, frequency_hz)
    power_mie = qabs * np.pi * radius ** 2 * ANCHOR_S0_W_PER_M2
    tissue = rho > 0
    return {
        "spacing_mm": h * 1e3, "cells_per_axis": nc, "steps_per_period": per,
        "periods": ramp_periods + settle_periods + dft_periods,
        "absorbed_power_fdtd_W": power, "absorbed_power_mie_W": float(power_mie),
        "peak_10g_fdtd": sphere_peak_10g(edge, sp, rho, sar_fdtd),
        "peak_10g_mie": sphere_peak_10g(edge, sp, rho, sar_mie),
        "local_sar_rel_rms": float(np.sqrt(np.mean((sar_fdtd[tissue] - sar_mie[tissue]) ** 2))
                                   / np.sqrt(np.mean(sar_mie[tissue] ** 2))),
    }


def fdtd_study():
    return {FREQUENCY_LABEL[f]: fdtd_sphere(f, FDTD_SPACING_M[f]) for f in FREQUENCIES_HZ}


def fdtd_margin_check(spacing_m, frequency_hz=2.45e9, margins_m=(0.015, 0.030)):
    """Absorbed power of the FDTD solution for two distances between the sphere
    and the absorbing layer; each distance must be a whole number of cells."""
    for mg in margins_m:
        if abs(mg / spacing_m - round(mg / spacing_m)) > 1e-9:
            raise ValueError("each margin must be a whole number of grid cells")
    power = [fdtd_sphere(frequency_hz, spacing_m, margin_m=mg)["absorbed_power_fdtd_W"]
             for mg in margins_m]
    return {"frequency": FREQUENCY_LABEL[frequency_hz], "spacing_mm": spacing_m * 1e3,
            "margins_mm": [mg * 1e3 for mg in margins_m], "absorbed_power_W": power,
            "rel_change_pct": float(100.0 * abs(power[1] / power[0] - 1.0))}


# ----------------------------------------------------------------------------
# Section 12e. Sampling precision, Wasserstein distances and input-model variants
# ----------------------------------------------------------------------------
def bootstrap_thresholds(y, reps, seed, limit=SAR_LIMIT_W_PER_KG, alpha=ALPHA, delta=DELTA,
                         chunk=100):
    """Percentile bootstrap 95% intervals of the quantile, tolerance, CVaR and
    Cantelli thresholds (resampling the endpoint sample with replacement)."""
    from scipy.stats import binom
    y = np.asarray(y, dtype=float)
    n = len(y)
    j_star = 0
    for j in range(1, n + 1):
        if binom.cdf(j - 1, n, alpha) <= delta:
            j_star = j
        else:
            break
    i_high = int(np.ceil((n - 1) * (1.0 - alpha)))
    i_tau = cvar_rank(n, alpha) - 1
    factor = np.sqrt((1.0 - alpha) / alpha)

    def stats(s):
        q = s[:, i_high]
        tol = s[:, n - j_star] if j_star > 0 else np.full(len(s), np.nan)
        tau = s[:, i_tau]
        cvar = tau + np.maximum(s - tau[:, None], 0.0).mean(axis=1) / alpha
        mom = s.mean(axis=1) + s.std(axis=1, ddof=1) * factor
        return {"S_var": q, "S_tolerance": tol, "S_cvar": cvar, "S_moment": mom}

    point = stats(np.sort(y)[None, :])
    checks = {"S_var": empirical_var(y), "S_tolerance": tolerance_var(y, alpha, delta),
              "S_cvar": cvar_upper(y, alpha), "S_moment": moment_dro_var(y, alpha)}
    for k, v in checks.items():
        if not np.isclose(point[k][0], v, rtol=1e-12, atol=0.0):
            raise RuntimeError(f"bootstrap estimator {k} disagrees with the point estimate")
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in point}
    for start in range(0, reps, chunk):
        m = min(chunk, reps - start)
        s = np.sort(y[rng.integers(0, n, size=(m, n))], axis=1)
        for k, v in stats(s).items():
            draws[k].append(v)
    out = {}
    for k, v in draws.items():
        thr = ANCHOR_S0_W_PER_M2 * limit / np.concatenate(v)
        lo, hi = np.percentile(thr, [2.5, 97.5])
        out[k] = {"low": float(lo), "high": float(hi)}
    return out


def w1_distance(u, v):
    """Type-1 Wasserstein distance between two empirical distributions."""
    from scipy.stats import wasserstein_distance
    return float(wasserstein_distance(np.asarray(u, float), np.asarray(v, float)))


def bootstrap_w1(y, reps, seed):
    """Mean W1 distance between bootstrap resamples and the sample itself."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float)
    return float(np.mean([w1_distance(y[rng.integers(0, len(y), len(y))], y)
                          for _ in range(reps)]))


def variant_comparison(y_main, y_var, nominal_sar, rl_local):
    """Thresholds of an input-model variant relative to the main sample, the
    exceedance of the main conformal threshold under the variant, and the W1
    distance between the two endpoint distributions."""
    conf = conformal_var(y_main)
    w1 = w1_distance(y_var, y_main)
    cvar_main, cvar_var = cvar_upper(y_main), cvar_upper(y_var)
    rec = {"w1": w1, "w1_pct_nominal": 100.0 * w1 / nominal_sar,
           "conformal_exceedance": float(np.mean(y_var > conf)),
           "wasserstein_bound_holds": bool(cvar_var <= cvar_main + w1 / ALPHA + 1e-12)}
    for key, fn in (("S_var", empirical_var), ("S_tolerance", tolerance_var),
                    ("S_cvar", cvar_upper), ("S_moment", moment_dro_var)):
        s_main = incident_power_threshold(fn(y_main), SAR_LIMIT_W_PER_KG)
        s_var = incident_power_threshold(fn(y_var), SAR_LIMIT_W_PER_KG)
        rec[key] = s_var
        rec[key + "_change_pct"] = 100.0 * (s_var / s_main - 1.0)
    eps_c = ALPHA * (ANCHOR_S0_W_PER_M2 * SAR_LIMIT_W_PER_KG / rl_local - cvar_main)
    rec["critical_eps_local_pct"] = (float(100.0 * eps_c / nominal_sar) if eps_c > 0
                                     else float("nan"))
    rec["w1_exceeds_critical"] = bool(eps_c > 0 and w1 > eps_c)
    rec["cvar_side_changes"] = bool((rec["S_cvar"] < rl_local) !=
                                    (incident_power_threshold(cvar_main, SAR_LIMIT_W_PER_KG)
                                     < rl_local))
    return rec


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
    # Sampling precision of each threshold and the W1 distance that sampling
    # alone produces between a resample and the sample.
    row["bootstrap"] = bootstrap_thresholds(y, cfg["bootstrap"], cfg["seed"] + BOOTSTRAP_SEED_OFFSET)
    row["bootstrap_w1_pct"] = 100.0 * bootstrap_w1(y, BOOTSTRAP_W1_REPS,
                                                    cfg["seed"] + BOOTSTRAP_W1_SEED_OFFSET) \
        / row["nominal_sar"]
    row["_y_sar"] = y
    return row


BOOTSTRAP_SEED_OFFSET = 17
BOOTSTRAP_W1_SEED_OFFSET = 19
BOOTSTRAP_W1_REPS = 200


def build_config(mode):
    common = {"seed": SEED, "temp_frequencies": FREQUENCIES_HZ,
              "families": INPUT_FAMILIES, "tissue_corrs": TISSUE_CORRELATIONS}
    if mode == "full":
        return dict(common, n_sar=2000, n_temp=1000, voxel_m=0.002, splits=200,
                    grid_subset=150, grid_fine_m=0.001, bootstrap=2000,
                    grid_voxels=GRID_STUDY_VOXELS_M, fdtd_spacing=FDTD_SPACING_M,
                    fdtd_margin_spacing=0.0025)
    if mode == "medium":
        return dict(common, n_sar=600, n_temp=300, voxel_m=0.0025, splits=100,
                    grid_subset=40, grid_fine_m=0.00125, bootstrap=500,
                    grid_voxels=(0.00375, 0.003, 0.0025, 0.002, 0.0015, 0.00125, 0.001),
                    fdtd_spacing={f: 0.001 for f in FREQUENCIES_HZ}, fdtd_margin_spacing=0.0025)
    return dict(common, n_sar=150, n_temp=80, voxel_m=0.004, splits=60,
                grid_subset=12, grid_fine_m=0.002, bootstrap=200,
                grid_voxels=(0.006, 0.005, 0.004, 0.003),
                fdtd_spacing={f: 0.003 for f in FREQUENCIES_HZ}, fdtd_margin_spacing=0.003)


def variant_list(cfg):
    """Input-model variants of the sensitivity analysis as (family, correlation)."""
    return ([(fam, 0.0) for fam in cfg["families"] if fam != "uniform"] +
            [("uniform", tc) for tc in cfg["tissue_corrs"]])


VARIANT_LABEL = {("lognormal", 0.0): "Normal log multipliers",
                 ("truncnormal", 0.0): "Truncated normal log multipliers",
                 ("uniform", 0.5): "Between-tissue correlation 0.5",
                 ("uniform", 0.9): "Between-tissue correlation 0.9"}


def thermal_input_correlations(frequency_hz, model, cfg, y_temp):
    """Spearman rank correlation of each sampled thermal input with the peak
    temperature rise (inputs regenerated from the seed)."""
    mats = sample_materials(frequency_hz, cfg["n_temp"], cfg["seed"], "both", model)
    out = {}
    for t in TISSUE_ORDER:
        for field in ("perfusion", "thermal_conductivity", "density"):
            vals = np.array([m[t][field] for m in mats])
            if np.ptp(vals) > 0:
                out[f"{t}:{field}"] = float(_spearman(vals, y_temp))
    return out


def grid_subset_analysis(cfg, cache_dir, reference_rows):
    """Effect of refining the sphere grid on the first draws of the main
    reference-model samples: draw-by-draw ratio, thresholds and W1 distance."""
    out = {}
    for f in FREQUENCIES_HZ:
        row = next(r for r in reference_rows if r["frequency_hz"] == f and r["case"] == "sphere")
        production, fine = grid_subset_sample(f, cfg, cache_dir)
        main = row["_y_sar"][:len(production)]
        if not np.array_equal(production, main):
            raise RuntimeError("grid subset does not reproduce the main sample")
        ratio = fine / production
        rec = {"n": int(len(ratio)), "fine_mm": cfg["grid_fine_m"] * 1e3,
               "ratio_mean": float(ratio.mean()), "ratio_sd": float(ratio.std(ddof=1)),
               "ratio_min": float(ratio.min()), "ratio_max": float(ratio.max()),
               "w1_pct_nominal": 100.0 * w1_distance(fine, production) / row["nominal_sar"]}
        for key, fn in (("S_var", empirical_var), ("S_cvar", cvar_upper),
                        ("S_moment", moment_dro_var)):
            rec[key + "_change_pct"] = 100.0 * (fn(production) / fn(fine) - 1.0)
        out[FREQUENCY_LABEL[f]] = rec
    return out


def _study_compute(name, cfg):
    """Compute function of each deterministic study for the given configuration."""
    v = cfg["voxel_m"]
    table = {
        "study_grid_averaging": lambda: {FREQUENCY_LABEL[f]: grid_averaging_study(f, cfg["grid_voxels"])
                                         for f in FREQUENCIES_HZ},
        "study_geometry": lambda: geometry_study(v),
        "study_perfusion": lambda: perfusion_study(v),
        "study_temperature_fields": lambda: temperature_field_study(v),
        "study_transient": lambda: transient_study(v),
        "study_fdtd": lambda: {FREQUENCY_LABEL[f]: fdtd_sphere(f, cfg["fdtd_spacing"][f])
                               for f in FREQUENCIES_HZ},
        "study_fdtd_margin": lambda: fdtd_margin_check(cfg["fdtd_margin_spacing"]),
    }
    return table[name]


STUDY_NAMES = ("study_grid_averaging", "study_geometry", "study_perfusion",
               "study_temperature_fields", "study_transient", "study_fdtd", "study_fdtd_margin")


def deterministic_studies(cfg, cache_dir):
    return {name.replace("study_", ""): cached_study(name, cache_dir, _study_compute(name, cfg))
            for name in STUDY_NAMES}


def run_model(model, cfg, cache_dir):
    """All SAR and temperature analyses for one uncertainty model."""
    seed = cfg["seed"]
    rows = []
    for f in FREQUENCIES_HZ:
        for case in CASES:
            print(f"  SAR [{model}]: {case} {FREQUENCY_LABEL[f]}", flush=True)
            rows.append(analyze_case(f, case, cfg, cache_dir, model))
    variants = {}
    for fam, tc in variant_list(cfg):
        recs = {}
        for r in rows:
            out, nom = monte_carlo(r["case"], r["frequency_hz"], cfg["n_sar"], seed, "sar",
                                   cfg["voxel_m"], cache_dir, model=model, family=fam,
                                   tissue_corr=tc)
            if float(nom[0]) != r["nominal_sar"]:
                raise RuntimeError("variant sample has a different nominal endpoint")
            rec = variant_comparison(r["_y_sar"], out[:, 0], r["nominal_sar"],
                                     r["reference_level_local"])
            rec["_y"] = out[:, 0]
            recs[(r["frequency_hz"], r["case"])] = rec
        variants[(fam, tc)] = recs
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
                "S_temp_nominal": incident_power_threshold(float(nom[1]), DELTA_T_LIMIT_C),
                "S_temp_var": incident_power_threshold(empirical_var(y_temp), DELTA_T_LIMIT_C),
                "S_temp_cvar": incident_power_threshold(cvar_upper(y_temp), DELTA_T_LIMIT_C),
                "sar_temp_spearman": float(_spearman(y_sar, y_temp)),
                "bootstrap": bootstrap_thresholds(y_temp, cfg["bootstrap"],
                                                  seed + BOOTSTRAP_SEED_OFFSET,
                                                  limit=DELTA_T_LIMIT_C),
                "input_correlations": thermal_input_correlations(f, model, cfg, y_temp),
                "_y_sar": y_sar, "_y_temp": y_temp})
    r0 = rows[0]
    eps_check = 0.1 * r0["nominal_sar"]
    wcheck = {"case": f"{r0['frequency']} {r0['geometry']}", "epsilon": eps_check,
              "closed_form": wasserstein_cvar(r0["_y_sar"], eps_check),
              "transport_construction": wasserstein_cvar_transport_check(r0["_y_sar"], eps_check)}
    return {"rows": rows, "coverage": coverage, "calib": calib, "shifts": shifts,
            "temp_rows": temp_rows, "wasserstein_check": wcheck, "variants": variants}


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


FLOW_STYLE = {
    "terminal": ("#F8D5E6", "#E07BAE", "#C0558E"),
    "process": ("#B8E3DA", "#5FB3A3", "#3F8F80"),
    "alternative": ("#C6E0FF", "#66AAFF", "#4A86D6"),
    "hexagon": ("#D6D3F2", "#8F89DD", "#6C66B8"),
    "decision": ("#C3E3A8", "#7DB84E", "#5E9A35"),
    "display": ("#FFD9B0", "#FF9F40", "#E07B1A"),
    "data": ("#FFF2B3", "#FFD54F", "#D6B03A"),
    "document": ("#E6D5EE", "#A780BF", "#8A63A3"),
    "loop": ("#B8E3DA", "#5FB3A3", "#3F8F80"),
}


def _flow_path(kind, x, y, w, h):
    """Outline of a flowchart shape with lower-left corner (x, y), width w, height h."""
    from matplotlib.path import Path
    if kind in ("process",):
        verts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)]
        return Path(verts, [Path.MOVETO] + [Path.LINETO] * 3 + [Path.CLOSEPOLY])
    if kind in ("terminal", "alternative"):
        r = min(h / 2.0, w / 2.0) if kind == "alternative" else min(0.28 * h, 2.5)
        k = 0.5523 * r
        verts = [(x + r, y), (x + w - r, y), (x + w - r + k, y), (x + w, y + r - k), (x + w, y + r),
                 (x + w, y + h - r), (x + w, y + h - r + k), (x + w - r + k, y + h), (x + w - r, y + h),
                 (x + r, y + h), (x + r - k, y + h), (x, y + h - r + k), (x, y + h - r),
                 (x, y + r), (x, y + r - k), (x + r - k, y), (x + r, y), (x + r, y)]
        codes = ([Path.MOVETO, Path.LINETO] + [Path.CURVE4] * 3 + [Path.LINETO] + [Path.CURVE4] * 3 +
                 [Path.LINETO] + [Path.CURVE4] * 3 + [Path.LINETO] + [Path.CURVE4] * 3 + [Path.CLOSEPOLY])
        return Path(verts, codes)
    if kind == "hexagon":
        d = 0.18 * w
        verts = [(x + d, y), (x + w - d, y), (x + w, y + h / 2), (x + w - d, y + h), (x + d, y + h),
                 (x, y + h / 2), (x + d, y)]
        return Path(verts, [Path.MOVETO] + [Path.LINETO] * 5 + [Path.CLOSEPOLY])
    if kind == "decision":
        verts = [(x + w / 2, y), (x + w, y + h / 2), (x + w / 2, y + h), (x, y + h / 2), (x + w / 2, y)]
        return Path(verts, [Path.MOVETO] + [Path.LINETO] * 3 + [Path.CLOSEPOLY])
    if kind == "data":
        s = 0.12 * w
        verts = [(x, y), (x + w - s, y), (x + w, y + h), (x + s, y + h), (x, y)]
        return Path(verts, [Path.MOVETO] + [Path.LINETO] * 3 + [Path.CLOSEPOLY])
    if kind == "display":
        p, r = 0.12 * w, 0.10 * w
        verts = [(x + p, y), (x + w - r, y), (x + w + 0.25 * r, y), (x + w + 0.25 * r, y + h),
                 (x + w - r, y + h), (x + p, y + h), (x, y + h / 2), (x + p, y)]
        codes = [Path.MOVETO, Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
                 Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
        return Path(verts, codes)
    if kind == "document":
        a = 0.12 * h
        verts = [(x, y + h), (x, y + a), (x + 0.25 * w, y - a), (x + 0.5 * w, y + a),
                 (x + 0.5 * w, y + a), (x + 0.75 * w, y + 3 * a), (x + w, y + a),
                 (x + w, y + h), (x, y + h)]
        codes = [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO,
                 Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CLOSEPOLY]
        return Path(verts, codes)
    if kind in ("loop_start", "loop_end"):
        c = min(0.3 * h, 2.0)
        if kind == "loop_start":
            verts = [(x, y), (x + w, y), (x + w, y + h - c), (x + w - c, y + h), (x + c, y + h),
                     (x, y + h - c), (x, y)]
        else:
            verts = [(x + c, y), (x + w - c, y), (x + w, y + c), (x + w, y + h), (x, y + h),
                     (x, y + c), (x + c, y)]
        return Path(verts, [Path.MOVETO] + [Path.LINETO] * 5 + [Path.CLOSEPOLY])
    raise ValueError(kind)


def _flow_box(ax, kind, x, y, w, h, text):
    """Draw a flowchart shape with a vertical gradient fill and centered text."""
    from matplotlib.colors import to_rgb
    from matplotlib.patches import PathPatch
    style = "loop" if kind in ("loop_start", "loop_end") else kind
    top, bottom, edge = FLOW_STYLE[style]
    path = _flow_path(kind, x, y, w, h)
    patch = PathPatch(path, facecolor="none", edgecolor=edge, lw=LW, zorder=3)
    ax.add_patch(patch)
    frac = np.linspace(0.0, 1.0, 256)[:, None, None]
    img = (1.0 - frac) * np.array(to_rgb(top)) + frac * np.array(to_rgb(bottom))
    ext = path.get_extents()
    im = ax.imshow(np.repeat(img, 2, axis=1), extent=(ext.x0, ext.x1, ext.y0, ext.y1),
                   origin="upper", aspect="auto", interpolation="bilinear", zorder=2)
    im.set_clip_path(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8, color="black",
            linespacing=1.15, zorder=4)
    return {"x": x, "y": y, "w": w, "h": h}


def _flow_arrow(ax, p0, p1, label=None, dashed=False, label_offset=(1.2, 0.0)):
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=10, lw=LW,
                                 color="#222222", linestyle=(0, (3, 2)) if dashed else "-",
                                 shrinkA=0, shrinkB=0, zorder=5))
    if label:
        ax.text((p0[0] + p1[0]) / 2 + label_offset[0], (p0[1] + p1[1]) / 2 + label_offset[1],
                label, fontsize=8, ha="left", va="center", color="black", zorder=6)


def workflow_figure(figdir, name):
    plt = _figure_style()
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 118.0 * MM))
    ax.set_xlim(0, 180); ax.set_ylim(0, 118); ax.set_axis_off()
    xa, wa, xb, wb, xc, wc = 4.0, 54.0, 66.0, 56.0, 130.0, 46.0
    nl = "\n"
    b = {}
    b["start"] = _flow_box(ax, "terminal", xa, 105, wa, 10,
                           nl.join(["Choose geometry, frequency", "and uncertainty model"]))
    b["input"] = _flow_box(ax, "data", xa, 92, wa, 10,
                           nl.join(["Nominal tissue properties,", "property ranges and n draws"]))
    b["loop0"] = _flow_box(ax, "loop_start", xa, 80, wa, 8, "For each of the n draws")
    b["sample"] = _flow_box(ax, "process", xa, 67, wa, 10,
                            nl.join(["Sample permittivity, conductivity,",
                                     "density and thermal properties"]))
    b["solve"] = _flow_box(ax, "process", xa, 52, wa, 11,
                           nl.join(["Solve the slab (transmission line)",
                                    "or sphere (multilayer Mie)", "at the anchor S₀"]))
    b["endpt"] = _flow_box(ax, "process", xa, 39, wa, 10,
                           nl.join(["Peak 10 g SAR and steady-state", "Pennes temperature rise"]))
    b["loop1"] = _flow_box(ax, "loop_end", xa, 28, wa, 8, "Next draw")
    b["ver"] = _flow_box(ax, "alternative", xa, 3, wa, 20,
                         nl.join(["Verification by transfer matrix,", "power and Mie balance, FDTD,",
                                  "radial bioheat solution,", "grid and averaging rule"]))
    b["doc"] = _flow_box(ax, "document", xb, 103, wb, 11, "Endpoint sample Y₁, ..., Yₙ")
    b["bounds"] = _flow_box(ax, "process", xb, 86, wb, 12,
                            nl.join(["Bounds q from the sample (quantile,",
                                     "tolerance limit, CVaR, Cantelli,", "Wasserstein CVaR)"]))
    b["thr"] = _flow_box(ax, "process", xb, 71, wb, 11,
                         nl.join(["Thresholds S = S₀L/q, bootstrap",
                                  "intervals and critical radius"]))
    b["dec"] = _flow_box(ax, "decision", xb + 6, 44, wb - 12, 22,
                         nl.join(["Threshold above", "the reference", "level?"]))
    b["disp"] = _flow_box(ax, "display", xb, 24, wb, 12,
                          nl.join(["Report thresholds, premises,", "intervals and critical radii"]))
    b["end"] = _flow_box(ax, "terminal", xb, 5, wb, 11,
                         nl.join(["Assessment for this geometry,", "source and input distribution"]))
    b["tests"] = _flow_box(ax, "hexagon", xc, 82, wc, 26,
                           nl.join(["Empirical tests of", "coverage, calibration",
                                    "size, distribution shift", "and input variants"]))
    b["margin"] = _flow_box(ax, "process", xc, 49, wc, 12,
                            nl.join(["Critical radius gives the", "margin of the comparison"]))
    cx = xa + wa / 2
    for top, bot in (("start", "input"), ("input", "loop0"), ("loop0", "sample"),
                     ("sample", "solve"), ("solve", "endpt"), ("endpt", "loop1")):
        _flow_arrow(ax, (cx, b[top]["y"]), (cx, b[bot]["y"] + b[bot]["h"]))
    ax.plot([xa + wa, 62.0, 62.0], [32.0, 32.0, 108.5], color="#222222", lw=LW, zorder=5)
    _flow_arrow(ax, (62.0, 108.5), (xb, 108.5))
    ax.plot([xa, 1.5, 1.5], [13.0, 13.0, 57.5], color="#222222", lw=LW, ls=(0, (3, 2)), zorder=5)
    _flow_arrow(ax, (1.5, 57.5), (xa, 57.5), dashed=True)
    cxb = xb + wb / 2
    for top, bot in (("doc", "bounds"), ("bounds", "thr"), ("thr", "dec")):
        _flow_arrow(ax, (cxb, b[top]["y"]), (cxb, b[bot]["y"] + b[bot]["h"]))
    _flow_arrow(ax, (cxb, b["dec"]["y"]), (cxb, b["disp"]["y"] + b["disp"]["h"]), label="no",
                label_offset=(1.5, 0.5))
    _flow_arrow(ax, (xb + wb - 6, 55.0), (xc, 55.0), label="yes", label_offset=(-3.0, 2.4))
    ax.plot([xc + wc / 2, xc + wc / 2], [b["margin"]["y"], 30.0], color="#222222", lw=LW, zorder=5)
    _flow_arrow(ax, (xc + wc / 2, 30.0), (xb + wb + 1.0, 30.0))
    _flow_arrow(ax, (cxb, b["disp"]["y"]), (cxb, b["end"]["y"] + b["end"]["h"]))
    _flow_arrow(ax, (xc, 92.0), (xb + wb, 92.0), dashed=True)
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    _save(fig, figdir, name)

FIGURE_FILE = {"workflow": "Figure1", "thresholds": "Figure2", "coverage": "Figure3",
               "calibration": "Figure4", "wasserstein": "Figure5", "geometry": "Figure6",
               "temperature": "Figure7", "joint": "Figure8"}
FREQ_COLOR = {0.9e9: "#4C78A8", 2.45e9: "#F58518", 5.8e9: "#54A24B"}


def make_figures(results, studies, extras, outdir):
    plt = _figure_style()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter, MaxNLocator, NullFormatter
    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)
    workflow_figure(figdir, FIGURE_FILE["workflow"])
    models = list(results)
    freqs = list(FREQUENCIES_HZ)
    letters = "ABCDEFGH"

    def row_for(model, f, case):
        return next(x for x in results[model]["rows"]
                    if x["frequency_hz"] == f and x["case"] == case)

    # Thresholds by bound with bootstrap 95% intervals, both uncertainty models.
    fig, axes = plt.subplots(len(models), 1, figsize=(FIG_WIDTH_IN, 66.0 * MM * len(models)),
                             sharex=True)
    for jx, model in enumerate(models):
        ax = axes[jx]
        x = np.arange(len(freqs) * len(CASES)); w = 0.19
        labels = []
        for f in freqs:
            for case in CASES:
                r = row_for(model, f, case)
                labels.append(f"{r['frequency']}\n{r['geometry']}")
        for off, key, lab, c in ((-1.5, "S_var", "Empirical/conformal quantile", COLOR["var"]),
                                 (-0.5, "S_tolerance", "Tolerance (95/95)", COLOR["tol"]),
                                 (0.5, "S_cvar", "CVaR", COLOR["cvar"]),
                                 (1.5, "S_moment", "Cantelli moment bound", COLOR["moment"])):
            vals, lo, hi = [], [], []
            for f in freqs:
                for case in CASES:
                    r = row_for(model, f, case)
                    vals.append(r[key])
                    lo.append(r[key] - r["bootstrap"][key]["low"])
                    hi.append(r["bootstrap"][key]["high"] - r[key])
            ax.bar(x + off * w, vals, w, color=c, label=lab, linewidth=0)
            ax.errorbar(x + off * w, vals, yerr=[lo, hi], fmt="none", ecolor="black",
                        elinewidth=LW, capsize=2.5, capthick=LW)
        for xi, f in zip(x, [f for f in freqs for _ in CASES]):
            ax.hlines(local_reference_level_w_per_m2(f), xi - 2.3 * w, xi + 2.3 * w,
                      colors=COLOR["local"], linestyles=(0, (4, 2)), lw=LW)
            ax.hlines(reference_level_w_per_m2(f), xi - 2.3 * w, xi + 2.3 * w,
                      colors=COLOR["wb"], linestyles=(0, (1, 1.5)), lw=LW)
        ax.plot(x, [row_for(model, f, c)["S_nominal"] for f in freqs for c in CASES], ls="none",
                marker="D", ms=4.5, color="black", label="Nominal properties")
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
    _save(fig, figdir, FIGURE_FILE["thresholds"])

    # Held-out exceedance of Gaussian plug-in and conformal quantiles.
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
    _save(fig, figdir, FIGURE_FILE["coverage"])

    # Threshold and exceedance versus calibration size (reference model).
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
    _save(fig, figdir, FIGURE_FILE["calibration"])

    # Wasserstein CVaR threshold versus radius, both models.
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
    _save(fig, figdir, FIGURE_FILE["wasserstein"])

    # Nominal threshold versus geometry.
    geo = studies["geometry"]
    panels = (("slab_fat", "Slab, fat thickness (mm)"), ("sphere_fat", "Sphere, fat thickness (mm)"),
              ("sphere_radius", "Sphere, outer radius (mm)"))
    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH_IN, 66.0 * MM), sharey=True)
    for px, (param, xlabel) in enumerate(panels):
        ax = axes[px]
        xs = geo["values_mm"][param]
        for f in freqs:
            sar = geo["peak_10g"][FREQUENCY_LABEL[f]][param]
            ax.plot(xs, [ANCHOR_S0_W_PER_M2 * SAR_LIMIT_W_PER_KG / v for v in sar], "-o", ms=3.5,
                    color=FREQ_COLOR[f], label=FREQUENCY_LABEL[f])
        for f, color in ((0.9e9, FREQ_COLOR[0.9e9]), (2.45e9, "black")):
            ax.axhline(local_reference_level_w_per_m2(f), color=color, ls=(0, (4, 2)), lw=LW)
        ax.set_yscale("log")
        ax.set_yticks([10, 20, 40, 100, 200])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_ylim(12, 260)
        ax.set_xlabel(xlabel)
        if px == 0:
            ax.set_ylabel(f"Nominal threshold ({UNIT_SPD})")
        _panel_label(ax, f"({letters[px]})")
    handles, labs = axes[0].get_legend_handles_labels()
    handles += [Line2D([0], [0], color=FREQ_COLOR[0.9e9], ls=(0, (4, 2)), lw=LW),
                Line2D([0], [0], color="black", ls=(0, (4, 2)), lw=LW)]
    labs += ["Local reference level, 0.9 GHz", "Local reference level, 2.45 and 5.8 GHz"]
    fig.legend(handles, labs, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    _save(fig, figdir, FIGURE_FILE["geometry"])

    # Temperature rise fields and step response at nominal properties.
    fields, trans = studies["temperature_fields"], studies["transient"]
    fig = plt.figure(figsize=(FIG_WIDTH_IN, 150.0 * MM))
    gs = fig.add_gridspec(2, 6, height_ratios=(1.0, 1.3), hspace=0.45, wspace=1.4)
    ax_a = fig.add_subplot(gs[0, 0:3])
    ax_e = fig.add_subplot(gs[0, 3:6])
    for f in freqs:
        rec = fields[FREQUENCY_LABEL[f]]
        ax_a.plot(rec["slab_depth_mm"], rec["slab_dT"], color=FREQ_COLOR[f], label=FREQUENCY_LABEL[f])
    for edge_mm in np.cumsum([SKIN_M, FAT_M, MUSCLE_M]) * 1e3:
        ax_a.axvline(edge_mm, color=COLOR["points"], lw=LW, ls=(0, (1, 1.5)))
    ax_a.set_xlabel("Depth in the slab (mm)")
    ax_a.set_ylabel("Temperature rise ΔT at S₀ (K)")
    ax_a.set_xlim(0, 40)
    ax_a.set_ylim(bottom=0)
    ax_a.legend(loc="upper right")
    _panel_label(ax_a, "(A)")
    for f in freqs:
        rec = trans[FREQUENCY_LABEL[f]]
        t_min = np.array(rec["curve_time_s"]) / 60.0
        ax_e.plot(t_min, rec["curve_ratio"]["slab"], color=FREQ_COLOR[f], ls="-")
        ax_e.plot(t_min, rec["curve_ratio"]["sphere"], color=FREQ_COLOR[f], ls=(0, (3, 1.5)))
    ax_e.axvline(6.0, color="black", lw=LW, ls=(0, (1, 1.5)))
    ax_e.set_xlabel("Time after switch-on (min)")
    ax_e.set_ylabel("Peak ΔT / steady-state peak ΔT")
    ax_e.set_xlim(0, 60)
    ax_e.set_ylim(0, 1.02)
    ax_e.legend(handles=[Line2D([0], [0], color="black", ls="-", lw=LW, label="Slab"),
                         Line2D([0], [0], color="black", ls=(0, (3, 1.5)), lw=LW, label="Sphere"),
                         Line2D([0], [0], color="black", ls=(0, (1, 1.5)), lw=LW, label="6 min")],
                loc="lower right")
    _panel_label(ax_e, "(B)")
    for fx, f in enumerate(freqs):
        ax = fig.add_subplot(gs[1, 2 * fx:2 * fx + 2])
        rec = fields[FREQUENCY_LABEL[f]]
        dT = np.array(rec["sphere_dT_xz"])
        tissue = np.array(rec["sphere_tissue_xz"]) > 0
        edge, sp = rec["sphere_edge_mm"], rec["sphere_spacing_mm"]
        img = np.where(tissue, dT, np.nan).T
        im = ax.imshow(img, origin="lower", extent=(edge, edge + sp * dT.shape[0],
                                                    edge, edge + sp * dT.shape[1]),
                       cmap="magma", vmin=0.0, interpolation="nearest")
        th = np.linspace(0, 2 * np.pi, 200)
        for r_m in (SPHERE_RADIUS_M - SKIN_M - FAT_M - MUSCLE_M, SPHERE_RADIUS_M - SKIN_M - FAT_M):
            ax.plot(1e3 * r_m * np.cos(th), 1e3 * r_m * np.sin(th), color="white", lw=LW)
        ax.annotate("", xy=(0, -31), xytext=(0, -40), annotation_clip=False,
                    arrowprops=dict(arrowstyle="-|>", lw=LW, color="black"))
        ax.set_xlim(-33, 33); ax.set_ylim(-42, 33)
        ax.set_aspect("equal")
        ax.set_xlabel("x (mm)")
        if fx == 0:
            ax.set_ylabel("z (mm)")
        ax.set_title(FREQUENCY_LABEL[f])
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.07, pad=0.30)
        cb.set_label("ΔT at S₀ (K)")
        cb.outline.set_linewidth(LW)
        cb.locator = MaxNLocator(4)
        cb.update_ticks()
        _panel_label(ax, f"({letters[2 + fx]})")
    _save(fig, figdir, FIGURE_FILE["temperature"])

    # Paired SAR and temperature samples at 2.45 GHz.
    fig, axes = plt.subplots(len(models), len(CASES),
                             figsize=(FIG_WIDTH_IN, 64.0 * MM * len(models)))
    for mx, model in enumerate(models):
        trs = [tr for tr in results[model]["temp_rows"] if tr["frequency_hz"] == 2.45e9]
        for cx, tr in enumerate(trs):
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
    _save(fig, figdir, FIGURE_FILE["joint"])

# ----------------------------------------------------------------------------
# Section 15. Result files, article tables, and in-text numbers
# ----------------------------------------------------------------------------
def _minus(text):
    """Typographic minus sign for negative numbers in displayed text."""
    if not isinstance(text, str):
        return text
    return re.sub(r"(?<![\w.])-(?=\d)", "\u2212", text)


def _write_csv(path, header, rows, display=False):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows([[_minus(c) for c in r] for r in rows] if display else rows)


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


def _ci(r, key):
    b = r["bootstrap"][key]
    return b["low"], b["high"]


def write_results(results, checks, studies, extras, cfg, outdir):
    """Full-precision result files for each model and for the deterministic studies."""
    for model, res in results.items():
        d = os.path.join(outdir, "results", model)
        os.makedirs(d, exist_ok=True)
        fields = ["frequency", "geometry", "n_sar", "nominal_sar", "mean_sar", "std_sar",
                  "var_q95", "conformal_q95", "tolerance_q95", "cvar95", "moment_q95",
                  "S_nominal", "S_var", "S_conformal", "S_tolerance", "S_cvar", "S_moment",
                  "reference_level_local", "reference_level_wb",
                  "critical_eps_local_pct", "critical_eps_wb_pct", "bootstrap_w1_pct"]
        boot = ["S_var", "S_tolerance", "S_cvar", "S_moment"]
        header = fields + [f"{k}_ci95_{e}" for k in boot for e in ("low", "high")]
        _write_csv(os.path.join(d, "thresholds.csv"), header,
                   [[_full(r[k]) for k in fields] +
                    [_full(r["bootstrap"][k][e]) for k in boot for e in ("low", "high")]
                    for r in res["rows"]])
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
                   "temp_var_q95", "temp_cvar95", "S_temp_nominal", "S_temp_var", "S_temp_cvar",
                   "sar_temp_spearman"]
        corr_keys = sorted({k for r in res["temp_rows"] for k in r["input_correlations"]})
        _write_csv(os.path.join(d, "temperature_thresholds.csv"),
                   tfields + ["S_temp_var_ci95_low", "S_temp_var_ci95_high",
                              "S_temp_cvar_ci95_low", "S_temp_cvar_ci95_high"] +
                   [f"spearman_{k.replace(':', '_')}" for k in corr_keys],
                   [[_full(r[k]) for k in tfields] +
                    [_full(r["bootstrap"]["S_var"]["low"]), _full(r["bootstrap"]["S_var"]["high"]),
                     _full(r["bootstrap"]["S_cvar"]["low"]), _full(r["bootstrap"]["S_cvar"]["high"])] +
                    [_full(r["input_correlations"].get(k, float("nan"))) for k in corr_keys]
                    for r in res["temp_rows"]])
        vfields = ["w1_pct_nominal", "conformal_exceedance", "S_var", "S_var_change_pct",
                   "S_tolerance", "S_tolerance_change_pct", "S_cvar", "S_cvar_change_pct",
                   "S_moment", "S_moment_change_pct", "critical_eps_local_pct",
                   "w1_exceeds_critical", "cvar_side_changes", "wasserstein_bound_holds"]
        rows = []
        for (fam, tc), recs in res["variants"].items():
            for (f, case), rec in recs.items():
                rows.append([VARIANT_LABEL[(fam, tc)], fam, _full(tc), FREQUENCY_LABEL[f],
                             CASE_LABEL[case]] + [_full(rec[k]) for k in vfields])
        _write_csv(os.path.join(d, "input_variants.csv"),
                   ["variant", "family", "between_tissue_correlation", "frequency", "geometry"]
                   + vfields, rows)
        with open(os.path.join(d, "wasserstein_check.json"), "w", encoding="utf-8") as fh:
            json.dump(res["wasserstein_check"], fh, indent=2)
    d = os.path.join(outdir, "results")
    rows = []
    for label, recs in studies["grid_averaging"].items():
        for r in recs:
            rows.append([label, f"{r['voxel_mm']:.2f}", r["cells"]] +
                        [repr(r[k]) for k in ("production", "face_added", "air_0", "air_5",
                                              "air_10", "air_20")])
    _write_csv(os.path.join(d, "grid_averaging.csv"),
               ["frequency", "voxel_mm", "cells", "production", "face_centered_added",
                "air_0", "air_5", "air_10", "air_20"], rows)
    if "grid_subset" in extras:
        keys = ["n", "fine_mm", "ratio_mean", "ratio_sd", "ratio_min", "ratio_max",
                "w1_pct_nominal", "S_var_change_pct", "S_cvar_change_pct", "S_moment_change_pct"]
        _write_csv(os.path.join(d, "grid_subset.csv"), ["frequency"] + keys,
                   [[lab] + [_full(rec[k]) for k in keys]
                    for lab, rec in extras["grid_subset"].items()])
    if "model_w1_pct" in extras:
        _write_csv(os.path.join(d, "model_distance.csv"), ["case", "w1_between_models_pct_nominal"],
                   [[k, _full(v)] for k, v in extras["model_w1_pct"].items()])
    geo = studies["geometry"]
    rows = []
    for label, rec in geo["peak_10g"].items():
        for param, values in geo["values_mm"].items():
            for v, sar in zip(values, rec[param]):
                rows.append([label, param, _full(v), _full(sar),
                             _full(ANCHOR_S0_W_PER_M2 * SAR_LIMIT_W_PER_KG / sar)])
    _write_csv(os.path.join(d, "geometry.csv"),
               ["frequency", "parameter", "value_mm", "nominal_peak_10g", "nominal_threshold"], rows)
    rows = []
    for label, rec in studies["perfusion"].items():
        for setting, vals in rec.items():
            rows.append([label, setting, _full(vals["slab"]), _full(vals["sphere"])])
    _write_csv(os.path.join(d, "perfusion.csv"),
               ["frequency", "perfusion", "slab_peak_dT", "sphere_peak_dT"], rows)
    rows = []
    for label, rec in studies["transient"].items():
        for g in ("slab", "sphere"):
            s = rec[g]
            rows.append([label, g] + [_full(s[k]) for k in ("steady_K", "t63_s", "t90_s",
                                                            "ratio_at_360_s", "ratio_at_1800_s")])
    _write_csv(os.path.join(d, "transient.csv"),
               ["frequency", "geometry", "steady_peak_dT", "t63_s", "t90_s", "ratio_at_6_min",
                "ratio_at_30_min"], rows)
    with open(os.path.join(d, "fdtd_check.json"), "w", encoding="utf-8") as fh:
        json.dump({"by_frequency": studies["fdtd"],
                   "absorbing_layer_distance": studies.get("fdtd_margin")}, fh, indent=2)
    checks = dict(checks)
    checks["config"] = {k: (list(v) if isinstance(v, tuple) else
                            ({FREQUENCY_LABEL[f]: s for f, s in v.items()} if isinstance(v, dict) else v))
                        for k, v in cfg.items()}
    with open(os.path.join(d, "verification.json"), "w", encoding="utf-8") as fh:
        json.dump(checks, fh, indent=2)


TISSUE_LABEL = {"skin_dry": "Dry skin", "fat_not_infiltrated": "Fat (not infiltrated)",
                "muscle": "Muscle", "bone_cortical": "Cortical bone"}


def _range_cell(triple, nd):
    nom, mn, mx = triple
    if mx <= mn:
        return f"{_fmt(nom, nd)} (fixed)"
    return f"{_fmt(nom, nd)} ({_fmt(mn, nd)}–{_fmt(mx, nd)})"


def _span(values, nd=1):
    lo, hi = min(values), max(values)
    return _fmt(lo, nd) if _fmt(lo, nd) == _fmt(hi, nd) else f"{_fmt(lo, nd)} to {_fmt(hi, nd)}"


def _grid_row(studies, label, voxel_mm):
    """Grid-study row whose voxel size is closest to voxel_mm."""
    rows = studies["grid_averaging"][label]
    return min(rows, key=lambda r: abs(r["voxel_mm"] - voxel_mm))


def write_article_tables(results, checks, studies, extras, cfg, outdir):
    """Tables exactly as displayed in the article (rounded strings)."""
    d = os.path.join(outdir, "tables")
    os.makedirs(d, exist_ok=True)
    nd = checks["nominal_dielectric"]
    rows = []
    for f in FREQUENCIES_HZ:
        for t in TISSUE_ORDER:
            v = nd[FREQUENCY_LABEL[f]][t]
            rows.append([_fmt(f / 1e9, 2).rstrip("0").rstrip("."), TISSUE_LABEL[t],
                         _fmt(v["epsilon_r"], 2), _fmt(v["sigma"], 3)])
    _write_csv(os.path.join(d, "nominal_dielectric.csv"),
               ["Frequency (GHz)", "Tissue", "eps_r", "sigma (S/m)"], rows, display=True)
    rows = [[TISSUE_LABEL[t],
             _range_cell(TISSUE_TABLE[t]["density"], 0),
             _range_cell(TISSUE_TABLE[t]["thermal_conductivity"], 3),
             _range_cell(TISSUE_TABLE[t]["perfusion"], 1),
             _range_cell(TISSUE_TABLE[t]["metabolic"], 2),
             _range_cell(TISSUE_TABLE[t]["heat_capacity"], 0)] for t in TISSUE_ORDER]
    _write_csv(os.path.join(d, "thermal_properties.csv"),
               ["Tissue", "Density (kg/m3)", "Thermal conductivity (W/m/K)",
                "Perfusion (mL/min/kg)", "Metabolic heat (W/kg)", "Heat capacity (J/kg/K)"], rows, display=True)
    rows = []
    for f in FREQUENCIES_HZ:
        ref = DIELECTRIC_BOX[f]
        rows.append([FREQUENCY_LABEL[f], _fmt(ref["log_half_width"][0], 3),
                     _fmt(ref["log_half_width"][1], 3), _fmt(WITHIN_TISSUE_LOG_HALF_WIDTH, 3),
                     _fmt(ref["correlation"], 3)])
    _write_csv(os.path.join(d, "uncertainty_models.csv"),
               ["Frequency", "Reference h_eps", "Reference h_sigma", "Within-tissue h",
                "Copula correlation"], rows, display=True)
    labels = [FREQUENCY_LABEL[f] for f in FREQUENCIES_HZ]
    vox = [r["voxel_mm"] for r in studies["grid_averaging"][labels[0]]]
    rows = [[f"{vm:.2f}"] + [_fmt(_grid_row(studies, lab, vm)["production"], 3) for lab in labels]
            for vm in vox]
    _write_csv(os.path.join(d, "grid_convergence.csv"),
               ["Voxel size (mm)"] + labels, rows, display=True)
    finest = min(vox)
    prod_mm = cfg["voxel_m"] * 1e3
    rows = []
    for vm in (prod_mm, finest):
        for key, text in (("face_added", "face-centered cubes added"),
                          ("air_0", "no volume outside tissue"),
                          ("air_5", "at most 5% outside tissue"),
                          ("air_20", "at most 20% outside tissue")):
            cells = []
            for lab in labels:
                g = _grid_row(studies, lab, vm)
                cells.append(_fmt(100.0 * (g[key] / g["production"] - 1.0)))
            rows.append([f"{_grid_row(studies, labels[0], vm)['voxel_mm']:g} mm, {text}"] + cells)
    if "grid_subset" in extras:
        sub = extras["grid_subset"]
        fine = sub[labels[0]]["fine_mm"]
        rows.append([f"{fine:g} mm versus {prod_mm:g} mm, ratio of peak SAR (mean)"] +
                    [_fmt(sub[lab]["ratio_mean"], 3) for lab in labels])
        rows.append([f"{fine:g} mm versus {prod_mm:g} mm, ratio of peak SAR (SD)"] +
                    [_fmt(sub[lab]["ratio_sd"], 3) for lab in labels])
        rows.append([f"{fine:g} mm versus {prod_mm:g} mm, empirical quantile threshold change (%)"] +
                    [_fmt(sub[lab]["S_var_change_pct"]) for lab in labels])
        rows.append([f"{fine:g} mm versus {prod_mm:g} mm, CVaR threshold change (%)"] +
                    [_fmt(sub[lab]["S_cvar_change_pct"]) for lab in labels])
        rows.append([f"{fine:g} mm versus {prod_mm:g} mm, Cantelli threshold change (%)"] +
                    [_fmt(sub[lab]["S_moment_change_pct"]) for lab in labels])
    _write_csv(os.path.join(d, "averaging_rule.csv"), ["Variant"] + labels, rows, display=True)
    rows = []
    for model, res in results.items():
        for r in res["rows"]:
            rows.append([TABLE_MODEL_LABEL[model], f"{r['frequency_hz'] / 1e9:g}", r["geometry"],
                         _fmt(r["S_nominal"]), _fmt(r["S_var"]), _fmt(r["S_tolerance"]),
                         _fmt(r["S_cvar"]), _fmt(r["S_moment"])])
    _write_csv(os.path.join(d, "sar_thresholds.csv"),
               ["Model", "Frequency (GHz)", "Geometry", "Nominal", "Quantile", "Tolerance",
                "CVaR", "Cantelli"], rows, display=True)
    rows = []
    for model, res in results.items():
        for r in res["rows"]:
            rows.append([f"{TABLE_MODEL_LABEL[model]}, {r['frequency_hz'] / 1e9:g} GHz, "
                         f"{r['geometry'].lower()}"] +
                        [f"{_fmt(_ci(r, b)[0])} to {_fmt(_ci(r, b)[1])}"
                         for b in ("S_var", "S_tolerance", "S_cvar", "S_moment")])
    _write_csv(os.path.join(d, "bootstrap_intervals.csv"),
               ["Case", "Quantile", "Tolerance", "CVaR", "Cantelli"], rows, display=True)
    rows = []
    for model, res in results.items():
        for r in res["rows"]:
            loc, wb = r["critical_eps_local_pct"], r["critical_eps_wb_pct"]
            key = (r["frequency_hz"], r["case"])
            w1s = [recs[key]["w1_pct_nominal"] for recs in res["variants"].values()]
            rows.append([TABLE_MODEL_LABEL[model], f"{r['frequency_hz'] / 1e9:g}", r["geometry"],
                         "none" if not np.isfinite(loc) else _fmt(loc),
                         "none" if not np.isfinite(wb) else _fmt(wb),
                         _fmt(r["bootstrap_w1_pct"]), _fmt(max(w1s))])
    _write_csv(os.path.join(d, "critical_radii.csv"),
               ["Model", "Frequency (GHz)", "Geometry", "Local RL (%)", "Whole-body RL (%)",
                "Sampling W1 (%)", "Largest variant W1 (%)"], rows, display=True)
    rows = []
    for model, res in results.items():
        for r in res["shifts"][(2.45e9, "sphere")]:
            rows.append([TABLE_MODEL_LABEL[model], r["shift"], _fmt(r["w1_pct_nominal"]),
                         _fmt(100.0 * r["conformal_exceedance"]), _fmt(r["cvar_shifted"], 3),
                         _fmt(r["share_pct"])])
    _write_csv(os.path.join(d, "shift_test.csv"),
               ["Model", "Transformation", "W1 (% nominal)", "Conformal exceedance (%)",
                "Transformed CVaR (W/kg)", "Share of worst case (%)"], rows, display=True)
    rows = []
    for model, res in results.items():
        for (fam, tc), recs in res["variants"].items():
            vals = list(recs.values())
            rows.append([TABLE_MODEL_LABEL[model], VARIANT_LABEL[(fam, tc)],
                         _span([v["w1_pct_nominal"] for v in vals]),
                         _span([v["S_var_change_pct"] for v in vals]),
                         _span([v["S_cvar_change_pct"] for v in vals]),
                         _span([v["S_moment_change_pct"] for v in vals]),
                         _span([100.0 * v["conformal_exceedance"] for v in vals]),
                         str(sum(1 for v in vals if v["cvar_side_changes"]))])
    _write_csv(os.path.join(d, "input_variants.csv"),
               ["Model", "Variant", "W1 (% nominal)", "Quantile change (%)", "CVaR change (%)",
                "Cantelli change (%)", "Conformal exceedance (%)", "Changed comparisons"], rows,
               display=True)
    rows = []
    for model, res in results.items():
        for r in res["temp_rows"]:
            rows.append([TABLE_MODEL_LABEL[model], f"{r['frequency_hz'] / 1e9:g}", r["geometry"],
                         _fmt(r["nominal_temp"], 3), _fmt(r["temp_var_q95"], 3),
                         _fmt(r["temp_cvar95"], 3), _fmt(r["S_temp_nominal"]),
                         _fmt(r["S_temp_var"]), _fmt(r["S_temp_cvar"]),
                         _fmt(r["sar_temp_spearman"], 2)])
    _write_csv(os.path.join(d, "temperature_thresholds.csv"),
               ["Model", "Frequency (GHz)", "Geometry", "Nominal dT (K)", "Quantile dT (K)",
                "CVaR dT (K)", "Nominal threshold", "Quantile threshold", "CVaR threshold",
                "Rank correlation"], rows, display=True)
    rows = []
    for f in FREQUENCIES_HZ:
        lab = FREQUENCY_LABEL[f]
        per, tr = studies["perfusion"][lab], studies["transient"][lab]
        for g in ("slab", "sphere"):
            rows.append([f"{f / 1e9:g}", CASE_LABEL[g], _fmt(per["nominal"][g], 3),
                         _fmt(per["minimum"][g], 3), _fmt(per["maximum"][g], 3),
                         _fmt(per["zero"][g], 3), _fmt(tr[g]["t63_s"] / 60.0),
                         _fmt(tr[g]["t90_s"] / 60.0), _fmt(tr[g]["ratio_at_360_s"], 2)])
    _write_csv(os.path.join(d, "thermal_sensitivity.csv"),
               ["Frequency (GHz)", "Geometry", "Nominal dT (K)", "Minimum perfusion dT (K)",
                "Maximum perfusion dT (K)", "No perfusion dT (K)", "t63 (min)", "t90 (min)",
                "Fraction at 6 min"], rows, display=True)


VARIANT_TAG = {("lognormal", 0.0): "Ln", ("truncnormal", 0.0): "Tn",
               ("uniform", 0.5): "Tc50", ("uniform", 0.9): "Tc90"}


def _minmax(num, key, values, nd=1):
    num[key + "Min"] = _fmt(min(values), nd)
    num[key + "Max"] = _fmt(max(values), nd)


def write_numbers(results, checks, studies, extras, cfg, outdir):
    """Computed values quoted in the article text, with the article's rounding."""
    num = {"Nsar": str(cfg["n_sar"]), "Ntemp": str(cfg["n_temp"]),
           "splits": str(cfg["splits"]), "withinTissueH": _fmt(WITHIN_TISSUE_LOG_HALF_WIDTH, 3),
           "bootReps": str(cfg["bootstrap"]), "gridSubsetN": str(cfg["grid_subset"]),
           "gridFineMm": _fmt(cfg["grid_fine_m"] * 1e3, 0), "truncScale": _fmt(TRUNCNORMAL_SCALE, 1)}
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
            for b, short in (("S_var", "Var"), ("S_cvar", "Cvar"), ("S_moment", "Mom"),
                             ("S_tolerance", "Tol")):
                lo, hi = _ci(r, b)
                num[f"{k}{short}CiLow"] = _fmt(lo)
                num[f"{k}{short}CiHigh"] = _fmt(hi)
        for name, key in (("VarDrop", "S_var"), ("CvarDrop", "S_cvar"), ("MomDrop", "S_moment")):
            _minmax(num, m + name, [100.0 * (1.0 - r[key] / r["S_nominal"]) for r in res["rows"]])
        _minmax(num, m + "MomBelowVar", [100.0 * (1.0 - r["S_moment"] / r["S_var"])
                                         for r in res["rows"]])
        for b, short in (("S_var", "Var"), ("S_tolerance", "Tol"), ("S_cvar", "Cvar"),
                         ("S_moment", "Mom")):
            half = [50.0 * (r["bootstrap"][b]["high"] - r["bootstrap"][b]["low"]) / r[b]
                    for r in res["rows"]]
            _minmax(num, f"{m}CiHalf{short}", half)
        _minmax(num, m + "BootW1", [r["bootstrap_w1_pct"] for r in res["rows"]])
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
        tol100 = [r["exc_tol_mean"] for rec in res["calib"].values() for r in rec if r["n_cal"] == 100]
        _minmax(num, m + "TolN100ExcPct", [100.0 * v for v in tol100], 1)
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
            k = f"{m}{r['case'].capitalize()}{ftag[r['frequency_hz']]}Temp"
            num[k + "Nom"] = _fmt(r["nominal_temp"], 3)
            num[k + "NomThr"] = _fmt(r["S_temp_nominal"])
            num[k + "Var"] = _fmt(r["S_temp_var"])
            num[k + "Cvar"] = _fmt(r["S_temp_cvar"])
            num[k + "Spearman"] = _fmt(r["sar_temp_spearman"], 2)
        sp = [r["sar_temp_spearman"] for r in res["temp_rows"]]
        _minmax(num, m + "Spearman", sp, 2)
        strongest = []
        for tr in res["temp_rows"]:
            ic = {k: v for k, v in tr["input_correlations"].items() if np.isfinite(v)}
            best = max(ic, key=lambda k: abs(ic[k]))
            strongest.append((best, ic[best]))
        num[m + "NPerfStrongest"] = str(sum(1 for k, _ in strongest if k.endswith("perfusion")))
        _minmax(num, m + "PerfRho", [v for _, v in strongest], 2)
        wc = res["wasserstein_check"]
        diff = abs(wc["closed_form"] - wc["transport_construction"])
        num[m + "WassCheckAgree"] = ("exactly in double precision" if diff == 0.0
                                     else f"to within {_sci(diff)} W kg^−1^")
        for (fam, tc), recs in res["variants"].items():
            vt = m + VARIANT_TAG[(fam, tc)]
            vals = list(recs.values())
            _minmax(num, vt + "W1", [v["w1_pct_nominal"] for v in vals])
            _minmax(num, vt + "VarChange", [v["S_var_change_pct"] for v in vals])
            _minmax(num, vt + "CvarChange", [v["S_cvar_change_pct"] for v in vals])
            _minmax(num, vt + "MomChange", [v["S_moment_change_pct"] for v in vals])
            _minmax(num, vt + "ExcPct", [100.0 * v["conformal_exceedance"] for v in vals])
            num[vt + "NChanged"] = str(sum(1 for v in vals if v["cvar_side_changes"]))
            num[vt + "NBeyondCrit"] = str(sum(1 for v in vals if v["w1_exceeds_critical"]))
            num[vt + "BoundHolds"] = str(sum(1 for v in vals if v["wasserstein_bound_holds"]))
        all_var = [v for recs in res["variants"].values() for v in recs.values()]
        _minmax(num, m + "AllVariantW1", [v["w1_pct_nominal"] for v in all_var])
        _minmax(num, m + "AllVariantVarChange", [v["S_var_change_pct"] for v in all_var])
        _minmax(num, m + "AllVariantExcPct", [100.0 * v["conformal_exceedance"] for v in all_var])
        num[m + "AllVariantNChanged"] = str(sum(1 for v in all_var if v["cvar_side_changes"]))
        num[m + "AllVariantNBeyondCrit"] = str(sum(1 for v in all_var if v["w1_exceeds_critical"]))
        num[m + "AllVariantN"] = str(len(all_var))
        _minmax(num, m + "AllVariantCvarChange", [v["S_cvar_change_pct"] for v in all_var])
        _minmax(num, m + "AllVariantMomChange", [v["S_moment_change_pct"] for v in all_var])
        num[m + "AllVariantNWithCrit"] = str(sum(1 for v in all_var
                                                 if np.isfinite(v["critical_eps_local_pct"])))
        num[m + "AllVariantNBoundHolds"] = str(sum(1 for v in all_var
                                                   if v["wasserstein_bound_holds"]))
        main = {(r["frequency_hz"], r["case"]): r for r in res["rows"]}
        bounds = ("S_var", "S_tolerance", "S_cvar", "S_moment")
        side = 0
        for recs in res["variants"].values():
            for key, v in recs.items():
                level = main[key]["reference_level_local"]
                side += sum(1 for b in bounds if (v[b] > level) != (main[key][b] > level))
        num[m + "AllVariantNSideChangedAny"] = str(side)
        num[m + "NCiContainsLevel"] = str(sum(
            1 for r in res["rows"] for b in bounds
            for level in (r["reference_level_local"], r["reference_level_wb"])
            if r["bootstrap"][b]["low"] <= level <= r["bootstrap"][b]["high"]))
        boot = {(r["frequency_hz"], r["case"]): r["bootstrap_w1_pct"] for r in res["rows"]}
        num[m + "AllVariantNBeyondCritSum"] = str(sum(
            1 for recs in res["variants"].values() for key, v in recs.items()
            if np.isfinite(v["critical_eps_local_pct"])
            and v["w1_pct_nominal"] + boot[key] >= v["critical_eps_local_pct"]))
        nominal_sar = {(r["frequency_hz"], r["case"]): r["nominal_sar"] for r in res["rows"]}
        _minmax(num, m + "TempAtSarLimit", [tr["nominal_temp"] * SAR_LIMIT_W_PER_KG
                                            / nominal_sar[(tr["frequency_hz"], tr["case"])]
                                            for tr in res["temp_rows"]], 2)
        num[m + "NBootW1BeyondCrit"] = str(sum(
            1 for r in res["rows"] if np.isfinite(r["critical_eps_local_pct"])
            and r["bootstrap_w1_pct"] >= r["critical_eps_local_pct"]))
        _minmax(num, m + "CiHalfAll", [50.0 * (r["bootstrap"][b]["high"] - r["bootstrap"][b]["low"])
                                       / r[b] for r in res["rows"] for b in bounds])
        paired = [(tr["S_temp_var"] * tr["sar_var_q95"] / (ANCHOR_S0_W_PER_M2 * SAR_LIMIT_W_PER_KG),
                   tr["S_temp_cvar"] * tr["sar_cvar95"] / (ANCHOR_S0_W_PER_M2 * SAR_LIMIT_W_PER_KG))
                  for tr in res["temp_rows"]]
        _minmax(num, m + "TempOverSarVar", [a for a, _ in paired], 2)
        _minmax(num, m + "TempOverSarCvar", [b for _, b in paired], 2)
    _minmax(num, "TolN100ExcPct", [100.0 * r["exc_tol_mean"] for res in results.values()
                                   for rec in res["calib"].values() for r in rec if r["n_cal"] == 100], 1)
    crit = [r["critical_eps_local_pct"] for res in results.values() for r in res["rows"]
            if np.isfinite(r["critical_eps_local_pct"])]
    margins = [(100.0 * abs(r[k] / r["reference_level_local"] - 1.0), r["frequency"], model, k)
               for model, res in results.items() for r in res["rows"] if r["case"] == "sphere"
               for k in ("S_var", "S_tolerance", "S_cvar", "S_moment")]
    smallest = min(margins)
    num["sphereMinMarginPct"] = _fmt(smallest[0])
    num["sphereMinMarginFreq"] = smallest[1]
    if set(results) == set(UNCERTAINTY_MODELS):
        for name, key in (("Var", "S_var"), ("Mom", "S_moment")):
            gains = [100.0 * (w[key] / r[key] - 1.0)
                     for r, w in zip(results["reference"]["rows"], results["within_tissue"]["rows"])]
            _minmax(num, f"modelGain{name}", gains)
    if "model_w1_pct" in extras:
        _minmax(num, "modelW1", list(extras["model_w1_pct"].values()))
    num["critLocalMin"] = _fmt(min(crit))
    num["critLocalMax"] = _fmt(max(crit))
    num["slabTmmMax"] = _sci(max(v["max_rel_diff"] for v in checks["slab_solver"].values()))
    num["slabPowerMax"] = _sci(max(v["power_balance_rel_error"]
                                   for v in checks["slab_solver"].values()))
    num["mieBalanceMax"] = _sci(max(checks["sphere_absorption_balance"].values()))
    th = checks["thermal_3d_vs_radial"]
    num["thermalVoxelRadialPctMax"] = _fmt(max(abs(v["rel_diff_pct"]) for v in th.values()))
    _minmax(num, "thermalTrueSymRatio", [v["true_source_ratio"] for v in th.values()], 3)
    labels = [FREQUENCY_LABEL[f] for f in FREQUENCIES_HZ]
    grid = studies["grid_averaging"]
    vox = [r["voxel_mm"] for r in grid[labels[0]]]
    finest = min(vox)
    prod_mm = cfg["voxel_m"] * 1e3
    num["gridCoarsestMm"] = _fmt(max(vox), 2)
    num["gridFinestMm"] = f"{finest:g}"
    for f in FREQUENCIES_HZ:
        lab, t = FREQUENCY_LABEL[f], ftag[f]
        prod = _grid_row(studies, lab, prod_mm)
        fin = _grid_row(studies, lab, finest)
        num["gridDev2mm" + t] = _fmt(100.0 * (prod["production"] / fin["production"] - 1.0))
        fine_vals = sorted(r["production"] for r in sorted(grid[lab], key=lambda r: r["voxel_mm"])[:3])
        num["gridFineRange" + t] = _fmt(100.0 * (max(fine_vals) - min(fine_vals)) / np.mean(fine_vals))
        num["avgAir0ThrFactor2mm" + t] = _fmt(prod["production"] / prod["air_0"], 1)
        for key in ("face_added", "air_0", "air_5", "air_20"):
            num[f"avg{key.replace('_', '').capitalize()}2mm{t}"] = \
                _fmt(100.0 * (prod[key] / prod["production"] - 1.0))
            num[f"avg{key.replace('_', '').capitalize()}Fine{t}"] = \
                _fmt(100.0 * (fin[key] / fin["production"] - 1.0))
    face = [100.0 * (r["face_added"] / r["production"] - 1.0) for lab in labels for r in grid[lab]]
    num["avgFaceAddedMaxAll"] = _fmt(max(face))
    num["avgFaceAddedProdMax"] = _fmt(max(100.0 * (_grid_row(studies, lab, prod_mm)["face_added"] /
                                                    _grid_row(studies, lab, prod_mm)["production"] - 1.0)
                                          for lab in labels))
    fine_rows = [r for lab in labels for r in grid[lab] if r["voxel_mm"] <= prod_mm + 1e-9]
    num["avgFaceAddedFineMax"] = _fmt(max(100.0 * (r["face_added"] / r["production"] - 1.0)
                                          for r in fine_rows))
    low_mid = [r for lab in labels[:2] for r in grid[lab] if r["voxel_mm"] <= prod_mm + 1e-9]
    num["airNoEffectLowMid"] = str(all(r[k] == r["production"] for r in low_mid
                                       for k in ("air_0", "air_5", "air_20")))
    _minmax(num, "gridFineDevHighF", [100.0 * (r["production"] / _grid_row(studies, labels[2], finest)
                                                ["production"] - 1.0)
                                       for r in grid[labels[2]] if r["voxel_mm"] <= prod_mm + 1e-9])
    _minmax(num, "gridDev2mmAbs", [abs(100.0 * (_grid_row(studies, lab, prod_mm)["production"] /
                                                 _grid_row(studies, lab, finest)["production"] - 1.0))
                                   for lab in labels])
    if "grid_subset" in extras:
        sub = extras["grid_subset"]
        for f in FREQUENCIES_HZ:
            lab, t = FREQUENCY_LABEL[f], ftag[f]
            num["subRatioMean" + t] = _fmt(sub[lab]["ratio_mean"], 3)
            num["subRatioSd" + t] = _fmt(sub[lab]["ratio_sd"], 3)
            num["subVarChange" + t] = _fmt(sub[lab]["S_var_change_pct"])
            num["subCvarChange" + t] = _fmt(sub[lab]["S_cvar_change_pct"])
        _minmax(num, "subRatioSd", [sub[lab]["ratio_sd"] for lab in labels], 3)
        _minmax(num, "subAbsChange", [abs(sub[lab][k]) for lab in labels
                                      for k in ("S_var_change_pct", "S_cvar_change_pct",
                                                "S_moment_change_pct")])
    fd = studies["fdtd"]
    for f in FREQUENCIES_HZ:
        lab, t = FREQUENCY_LABEL[f], ftag[f]
        rec = fd[lab]
        num["fdtdPower" + t] = _fmt(100.0 * (rec["absorbed_power_fdtd_W"] / rec["absorbed_power_mie_W"] - 1.0))
        num["fdtdPeak" + t] = _fmt(100.0 * (rec["peak_10g_fdtd"] / rec["peak_10g_mie"] - 1.0))
        num["fdtdSpacing" + t] = _fmt(rec["spacing_mm"], 1)
    _minmax(num, "fdtdPowerAbs", [abs(100.0 * (fd[l]["absorbed_power_fdtd_W"] /
                                              fd[l]["absorbed_power_mie_W"] - 1.0)) for l in labels])
    _minmax(num, "fdtdPeakAbs", [abs(100.0 * (fd[l]["peak_10g_fdtd"] / fd[l]["peak_10g_mie"] - 1.0))
                                 for l in labels])
    _minmax(num, "fdtdLocalRms", [100.0 * fd[l]["local_sar_rel_rms"] for l in labels], 0)
    if "fdtd_margin" in studies:
        mg = studies["fdtd_margin"]
        num["fdtdMarginChangePct"] = _fmt(mg["rel_change_pct"], 4)
        num["fdtdMarginFromMm"] = _fmt(mg["margins_mm"][0], 0)
        num["fdtdMarginToMm"] = _fmt(mg["margins_mm"][1], 0)
        num["fdtdMarginSpacingMm"] = f"{mg['spacing_mm']:g}"
    geo = studies["geometry"]
    for f in FREQUENCIES_HZ:
        lab, t = FREQUENCY_LABEL[f], ftag[f]
        for param, values in geo["peak_10g"][lab].items():
            thr = [ANCHOR_S0_W_PER_M2 * SAR_LIMIT_W_PER_KG / v for v in values]
            key = "geo" + "".join(w.capitalize() for w in param.split("_")) + t
            _minmax(num, key + "Thr", thr)
            num[key + "Ratio"] = _fmt(max(values) / min(values), 1)
    ratios = [max(v) / min(v) for lab in labels for v in geo["peak_10g"][lab].values()]
    num["geoRatioMax"] = _fmt(max(ratios), 1)
    per = studies["perfusion"]
    for f in FREQUENCIES_HZ:
        lab, t = FREQUENCY_LABEL[f], ftag[f]
        for g in ("slab", "sphere"):
            p = per[lab]
            num[f"perfZeroRatio{g.capitalize()}{t}"] = _fmt(p["zero"][g] / p["nominal"][g], 1)
            num[f"perfMinChange{g.capitalize()}{t}"] = _fmt(100.0 * (p["minimum"][g] / p["nominal"][g] - 1.0))
            num[f"perfMaxChange{g.capitalize()}{t}"] = _fmt(100.0 * (p["maximum"][g] / p["nominal"][g] - 1.0))
    _minmax(num, "perfZeroRatio", [per[l]["zero"][g] / per[l]["nominal"][g]
                                   for l in labels for g in ("slab", "sphere")])
    _minmax(num, "perfZeroRatioSphere", [per[l]["zero"]["sphere"] / per[l]["nominal"]["sphere"]
                                         for l in labels])
    _minmax(num, "perfMinChange", [100.0 * (per[l]["minimum"][g] / per[l]["nominal"][g] - 1.0)
                                   for l in labels for g in ("slab", "sphere")])
    _minmax(num, "perfMaxChange", [100.0 * (per[l]["maximum"][g] / per[l]["nominal"][g] - 1.0)
                                   for l in labels for g in ("slab", "sphere")])
    _minmax(num, "perfMaxDrop", [100.0 * (1.0 - per[l]["maximum"][g] / per[l]["nominal"][g])
                                 for l in labels for g in ("slab", "sphere")])
    tr = studies["transient"]
    _minmax(num, "t63Min", [tr[l][g]["t63_s"] / 60.0 for l in labels for g in ("slab", "sphere")])
    _minmax(num, "t90Min", [tr[l][g]["t90_s"] / 60.0 for l in labels for g in ("slab", "sphere")])
    _minmax(num, "ratio6min", [100.0 * tr[l][g]["ratio_at_360_s"] for l in labels
                               for g in ("slab", "sphere")], 0)
    _minmax(num, "ratio30min", [100.0 * tr[l][g]["ratio_at_1800_s"] for l in labels
                                for g in ("slab", "sphere")], 0)
    num = {k: _minus(v) for k, v in num.items()}
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
    from importlib import metadata
    for package, dist in (("numpy", "numpy"), ("scipy", "scipy"), ("matplotlib", "matplotlib"),
                          ("PIL", "pillow"), ("scattnlay", "scattnlay")):
        try:
            importlib.import_module(package)
            print(f"  [OK] {dist}: {metadata.version(dist)}")
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


def run(mode, outdir, cache_dir, models, seed, jobs=1, generate=False):
    cfg = build_config(mode)
    cfg["seed"] = seed
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    print(f"[mode={mode}] N_sar={cfg['n_sar']} N_temp={cfg['n_temp']} "
          f"voxel={cfg['voxel_m'] * 1e3:.1f} mm splits={cfg['splits']} "
          f"samples={cache_dir}", flush=True)
    if generate:
        generate_samples(cfg, models, cache_dir, jobs=jobs)
    print("  verification checks...", flush=True)
    checks = deterministic_checks(cfg)
    studies = deterministic_studies(cfg, cache_dir)
    results = {model: run_model(model, cfg, cache_dir) for model in models}
    extras = {}
    if "reference" in results:
        extras["grid_subset"] = grid_subset_analysis(cfg, cache_dir, results["reference"]["rows"])
    if set(results) == set(UNCERTAINTY_MODELS):
        extras["model_w1_pct"] = {
            f"{r['frequency']} {r['geometry']}":
                100.0 * w1_distance(r["_y_sar"], w["_y_sar"]) / r["nominal_sar"]
            for r, w in zip(results["reference"]["rows"], results["within_tissue"]["rows"])}
    write_results(results, checks, studies, extras, cfg, outdir)
    if len(models) == len(UNCERTAINTY_MODELS):
        make_figures(results, studies, extras, outdir)
        write_article_tables(results, checks, studies, extras, cfg, outdir)
        write_numbers(results, checks, studies, extras, cfg, outdir)
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


def load_endpoint_samples(path):
    """Endpoint values from a .npy, .npz (array 'outputs', first column) or text
    file (one value per line or the first comma-separated column; a header line
    is skipped)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        y = np.load(path)
    elif ext == ".npz":
        with np.load(path) as data:
            y = data["outputs"] if "outputs" in data else data[data.files[0]]
    else:
        rows = []
        with open(path, encoding="utf-8") as fh:
            for k, line in enumerate(fh):
                cell = line.strip().split(",")[0].strip()
                if not cell:
                    continue
                try:
                    rows.append(float(cell))
                except ValueError:
                    if k == 0:
                        continue
                    raise ValueError(f"{path}: line {k + 1} is not a number: {cell!r}")
        y = np.array(rows)
    y = np.asarray(y, dtype=float)
    if y.ndim > 1:
        y = y[:, 0]
    return y


def threshold_report(y, anchor, limit, levels, alpha=ALPHA, delta=DELTA, nominal=None,
                     reps=2000, seed=SEED):
    """Thresholds, bootstrap intervals and critical radii for user endpoint samples.

    y holds endpoint values (for example peak 10 g SAR) computed at the source
    strength ``anchor`` (for example incident power density or antenna input
    power); every endpoint must scale linearly with the source strength.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or len(y) < 20:
        raise ValueError("at least 20 endpoint values are needed")
    if not np.all(np.isfinite(y)) or np.any(y <= 0):
        raise ValueError("endpoint values must be finite and positive")
    if not (0.0 < alpha < 1.0 and 0.0 < delta < 1.0 and anchor > 0 and limit > 0):
        raise ValueError("alpha, delta, anchor and limit must be positive (alpha, delta < 1)")
    bounds = {"quantile": empirical_var(y, alpha), "conformal": conformal_var(y, alpha),
              "tolerance": tolerance_var(y, alpha, delta), "cvar": cvar_upper(y, alpha),
              "cantelli": moment_dro_var(y, alpha)}
    rep = {"n": int(len(y)), "alpha": alpha, "delta": delta, "anchor": anchor, "limit": limit,
           "bounds": bounds,
           "thresholds": {k: (anchor * limit / v if np.isfinite(v) and v > 0 else float("nan"))
                          for k, v in bounds.items()}}
    if nominal is not None:
        rep["thresholds"]["nominal"] = anchor * limit / nominal
    ci = bootstrap_thresholds(y, reps, seed + BOOTSTRAP_SEED_OFFSET, limit=limit, alpha=alpha,
                              delta=delta)
    scale = anchor / ANCHOR_S0_W_PER_M2
    rep["bootstrap_95"] = {k: {"low": v["low"] * scale, "high": v["high"] * scale}
                           for k, v in ci.items()}
    rep["critical_radius"] = {}
    for level in levels:
        eps = alpha * (anchor * limit / level - bounds["cvar"])
        rep["critical_radius"][str(level)] = {
            "endpoint_units": float(eps) if eps > 0 else float("nan"),
            "percent_of_nominal": (float(100.0 * eps / nominal)
                                   if (nominal is not None and eps > 0) else float("nan"))}
    return rep


def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="Reproduce the article results, or compute "
                                            "thresholds for your own endpoint samples.")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_true", help="check dependencies; no simulation")
    g.add_argument("--fast", action="store_true", help="reduced-size pipeline check")
    g.add_argument("--medium", action="store_true", help="intermediate sample size")
    g.add_argument("--samples", metavar="FILE",
                   help="compute thresholds for endpoint samples in FILE (.csv, .txt, .npy, .npz)")
    p.add_argument("--simulate", action="store_true",
                   help="regenerate endpoint samples instead of reading data/samples")
    p.add_argument("--jobs", type=int, default=1,
                   help="parallel processes for sample generation (default 1)")
    p.add_argument("--models", default="reference,within_tissue",
                   help="comma-separated uncertainty models")
    p.add_argument("--outdir", default=os.path.join(here, "results"), help="output directory")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--anchor", type=float, default=ANCHOR_S0_W_PER_M2,
                   help="source strength at which the samples were computed (default 50, W/m^2)")
    p.add_argument("--limit", type=float, default=SAR_LIMIT_W_PER_KG,
                   help="restriction on the endpoint (default 2, W/kg)")
    p.add_argument("--levels", default="",
                   help="comma-separated comparison levels in source units, e.g. 40,10")
    p.add_argument("--alpha", type=float, default=ALPHA, help="target exceedance (default 0.05)")
    p.add_argument("--nominal", type=float, default=None,
                   help="endpoint at nominal properties (optional)")
    p.add_argument("--report", default="", help="write the threshold report to this JSON file")
    args = p.parse_args(argv)
    if args.check:
        return preflight_check()
    if args.samples:
        try:
            y = load_endpoint_samples(args.samples)
            levels = [float(v) for v in args.levels.split(",") if v.strip()]
            rep = threshold_report(y, args.anchor, args.limit, levels, alpha=args.alpha,
                                   nominal=args.nominal, seed=args.seed)
        except (OSError, ValueError) as exc:
            p.error(str(exc))
        print(f"{rep['n']} endpoint values, alpha = {rep['alpha']}, anchor = {rep['anchor']}, "
              f"limit = {rep['limit']}")
        names = {"quantile": "S_var", "tolerance": "S_tolerance", "cvar": "S_cvar",
                 "cantelli": "S_moment"}
        for k, v in rep["thresholds"].items():
            ci = rep["bootstrap_95"].get(names.get(k, ""), None)
            extra = f"  (95% bootstrap {ci['low']:.4g} to {ci['high']:.4g})" if ci else ""
            print(f"  threshold {k:10s} {v:.4g}{extra}")
        for level, c in rep["critical_radius"].items():
            if not np.isfinite(c["endpoint_units"]):
                print(f"  critical Wasserstein radius for level {level}: none (the CVaR "
                      "threshold already lies at or below the level)")
                continue
            pct = (f" ({c['percent_of_nominal']:.3g}% of nominal)"
                   if np.isfinite(c["percent_of_nominal"]) else "")
            print(f"  critical Wasserstein radius for level {level}: "
                  f"{c['endpoint_units']:.4g} endpoint units{pct}")
        if args.report:
            with open(args.report, "w", encoding="utf-8") as fh:
                json.dump(rep, fh, indent=1)
        return 0
    mode = "fast" if args.fast else "medium" if args.medium else "full"
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in models:
        if m not in UNCERTAINTY_MODELS:
            p.error(f"unknown model {m!r}")
    if args.jobs < 1:
        p.error("--jobs must be at least 1")
    shipped = mode == "full" and not args.simulate
    cache_dir = (os.path.join(here, "data", "samples") if shipped
                 else os.path.join(args.outdir, "samples"))
    run(mode, args.outdir, cache_dir, models, args.seed, jobs=args.jobs, generate=not shipped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
