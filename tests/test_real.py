"""Tests for gsfno.data.real (EFITLoader).

All tests use mocking because OMAS/MDSplus are not installed in CI.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# test_real_importable
# ---------------------------------------------------------------------------


def test_real_importable():
    """EFITLoader can be imported without OMAS installed."""
    from gsfno.data.real import EFITLoader  # noqa: F401


# ---------------------------------------------------------------------------
# test_no_omas_raises
# ---------------------------------------------------------------------------


def test_no_omas_raises():
    """EFITLoader.__init__ raises ImportError when OMAS is not available."""
    from gsfno.data.real import EFITLoader

    with patch("gsfno.data.real.OMAS_AVAILABLE", False):
        with pytest.raises(ImportError, match="OMAS is required"):
            EFITLoader("jet")


# ---------------------------------------------------------------------------
# test_invalid_machine_raises
# ---------------------------------------------------------------------------


def test_invalid_machine_raises():
    """EFITLoader raises ValueError for unsupported machine names."""
    from gsfno.data.real import EFITLoader

    with patch("gsfno.data.real.OMAS_AVAILABLE", True):
        with pytest.raises(ValueError, match="machine must be one of"):
            EFITLoader("invalid")


# ---------------------------------------------------------------------------
# test_interpolate_to_grid_shape
# ---------------------------------------------------------------------------


def test_interpolate_to_grid_shape():
    """_interpolate_to_grid returns array of shape (NR, NZ)."""
    from gsfno.data.real import EFITLoader

    with patch("gsfno.data.real.OMAS_AVAILABLE", True):
        loader = EFITLoader("jet")

    rng = np.random.default_rng(0)
    psi_src = rng.standard_normal((10, 12)).astype(np.float32)
    R_src = np.linspace(0.5, 2.5, 10)
    Z_src = np.linspace(-1.5, 1.5, 12)

    result = loader._interpolate_to_grid(psi_src, R_src, Z_src)

    assert result.shape == (65, 65), f"Expected (65, 65), got {result.shape}"


# ---------------------------------------------------------------------------
# test_process_time_slice_structure
# ---------------------------------------------------------------------------


def test_process_time_slice_structure():
    """_process_time_slice returns a dict with correct keys and array shapes."""
    from gsfno.data.real import EFITLoader

    with patch("gsfno.data.real.OMAS_AVAILABLE", True):
        loader = EFITLoader("jet")

    rng = np.random.default_rng(42)
    mock_ods = {
        "equilibrium.time_slice": {
            0: {
                "profiles_2d": {
                    0: {
                        "psi": rng.standard_normal((10, 12)),
                        "grid": {
                            "dim1": np.linspace(0.5, 2.5, 10),
                            "dim2": np.linspace(-1.5, 1.5, 12),
                        },
                    }
                },
                "global_quantities": {"ip": 2e6},
            }
        }
    }

    result = loader._process_time_slice(mock_ods, 0)

    assert result is not None, "_process_time_slice returned None unexpectedly"
    assert set(result.keys()) == {"inputs", "psi", "params"}

    assert result["inputs"].shape == (5, 65, 65), (
        f"inputs shape mismatch: {result['inputs'].shape}"
    )
    assert result["psi"].shape == (1, 65, 65), (
        f"psi shape mismatch: {result['psi'].shape}"
    )
    assert result["inputs"].dtype == np.float32
    assert result["psi"].dtype == np.float32

    params = result["params"]
    assert "machine" in params
    assert "Ip" in params
    assert "source" in params
    assert params["source"] == "efit"
    assert params["machine"] == "jet"
    assert params["Ip"] == pytest.approx(2e6)
