"""Tests for gsfno.data.generator.PlasmaConfigGenerator.

Tests are designed to run without FreeGS installed:
- import/instantiation tests never call FreeGS
- generate_one / generate_batch tests mock FREEGS_AVAILABLE=False
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pytest

from gsfno.data.generator import PlasmaConfigGenerator


# ---------------------------------------------------------------------------
# 1. Importability
# ---------------------------------------------------------------------------


def test_generator_importable():
    """PlasmaConfigGenerator can be imported and instantiated without FreeGS."""
    gen = PlasmaConfigGenerator(seed=42)
    assert gen.NR == 65
    assert gen.NZ == 65
    assert gen.R_min == 0.5
    assert gen.R_max == 2.5
    assert gen.Z_min == -1.5
    assert gen.Z_max == 1.5


# ---------------------------------------------------------------------------
# 2. Parameter sampling ranges
# ---------------------------------------------------------------------------


def test_sample_params_ranges():
    """_sample_params() returns all required keys with values in expected ranges."""
    gen = PlasmaConfigGenerator(seed=0)

    for _ in range(200):
        p = gen._sample_params()

        # Keys present
        assert set(p.keys()) == {"R0", "a", "kappa", "delta", "Ip", "p_scale", "ff_scale"}

        # R0 in [1.0, 2.0]
        assert 1.0 <= p["R0"] <= 2.0, f"R0={p['R0']} out of range"

        # a < R0/2 and in [0.3, 0.8]
        assert p["a"] <= 0.8, f"a={p['a']} > 0.8"
        assert p["a"] < p["R0"] / 2.0, f"a={p['a']} >= R0/2={p['R0']/2}"

        # kappa in [1.0, 2.5]
        assert 1.0 <= p["kappa"] <= 2.5, f"kappa={p['kappa']} out of range"

        # delta in [0.0, 0.6]
        assert 0.0 <= p["delta"] <= 0.6, f"delta={p['delta']} out of range"

        # Ip in [1e5, 5e6]
        assert 1e5 <= p["Ip"] <= 5e6, f"Ip={p['Ip']} out of range"

        # p_scale in [1e2, 1e4]
        assert 1e2 <= p["p_scale"] <= 1e4, f"p_scale={p['p_scale']} out of range"

        # ff_scale in [0.1, 1.0]
        assert 0.1 <= p["ff_scale"] <= 1.0, f"ff_scale={p['ff_scale']} out of range"


# ---------------------------------------------------------------------------
# 3. Profile shapes and non-negativity
# ---------------------------------------------------------------------------


def test_make_profiles_shape():
    """_make_profiles returns two float32 arrays of shape (NR,), values non-negative."""
    gen = PlasmaConfigGenerator(NR=65, NZ=65, seed=1)
    params = gen._sample_params()

    # Test on a 1-D R array
    R_1d = np.linspace(gen.R_min, gen.R_max, gen.NR, dtype=np.float32)
    p_prime, ff_prime = gen._make_profiles(params, R_1d)

    assert p_prime.shape == (gen.NR,), f"p_prime shape {p_prime.shape} != ({gen.NR},)"
    assert ff_prime.shape == (gen.NR,), f"ff_prime shape {ff_prime.shape} != ({gen.NR},)"

    assert p_prime.dtype == np.float32, "p_prime must be float32"
    assert ff_prime.dtype == np.float32, "ff_prime must be float32"

    assert np.all(p_prime >= 0), "p_prime has negative values"
    assert np.all(ff_prime >= 0), "ff_prime has negative values"


def test_make_profiles_shape_2d():
    """_make_profiles works on a 2-D (NR, NZ) R_grid."""
    gen = PlasmaConfigGenerator(NR=65, NZ=65, seed=2)
    params = gen._sample_params()

    p_prime, ff_prime = gen._make_profiles(params, gen.R_grid)

    assert p_prime.shape == (gen.NR, gen.NZ)
    assert ff_prime.shape == (gen.NR, gen.NZ)
    assert np.all(p_prime >= 0)
    assert np.all(ff_prime >= 0)


# ---------------------------------------------------------------------------
# 4. generate_one raises ImportError when FreeGS unavailable
# ---------------------------------------------------------------------------


def test_generate_one_no_freegs():
    """generate_one() raises ImportError when FREEGS_AVAILABLE is False."""
    gen = PlasmaConfigGenerator(seed=3)

    with patch("gsfno.data.generator.FREEGS_AVAILABLE", False):
        with pytest.raises(ImportError, match="FreeGS is not installed"):
            gen.generate_one()


def test_generate_one_no_freegs_with_params():
    """generate_one(params=...) also raises ImportError when FreeGS unavailable."""
    gen = PlasmaConfigGenerator(seed=4)
    params = gen._sample_params()

    with patch("gsfno.data.generator.FREEGS_AVAILABLE", False):
        with pytest.raises(ImportError, match="FreeGS is not installed"):
            gen.generate_one(params=params)


# ---------------------------------------------------------------------------
# 5. generate_batch raises ImportError when FreeGS unavailable
# ---------------------------------------------------------------------------


def test_generate_batch_no_freegs():
    """generate_batch(5) raises ImportError when FREEGS_AVAILABLE is False."""
    gen = PlasmaConfigGenerator(seed=5)

    with patch("gsfno.data.generator.FREEGS_AVAILABLE", False):
        with pytest.raises(ImportError, match="FreeGS is not installed"):
            gen.generate_batch(5)


# ---------------------------------------------------------------------------
# 6. Extra: coordinate grid shapes are correct
# ---------------------------------------------------------------------------


def test_coordinate_grids():
    """R_grid and Z_grid have expected shape and value bounds."""
    gen = PlasmaConfigGenerator(NR=65, NZ=65, seed=6)

    assert gen.R_grid.shape == (65, 65)
    assert gen.Z_grid.shape == (65, 65)

    assert float(gen.R_grid.min()) == pytest.approx(gen.R_min, rel=1e-5)
    assert float(gen.R_grid.max()) == pytest.approx(gen.R_max, rel=1e-5)
    assert float(gen.Z_grid.min()) == pytest.approx(gen.Z_min, rel=1e-5)
    assert float(gen.Z_grid.max()) == pytest.approx(gen.Z_max, rel=1e-5)

    # Normalised grids in [0, 1]
    assert float(gen._R_norm.min()) == pytest.approx(0.0, abs=1e-6)
    assert float(gen._R_norm.max()) == pytest.approx(1.0, abs=1e-6)
    assert float(gen._Z_norm.min()) == pytest.approx(0.0, abs=1e-6)
    assert float(gen._Z_norm.max()) == pytest.approx(1.0, abs=1e-6)
