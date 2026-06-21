"""Tests for gsfno.data.real (EFITLoader).

All tests use mocking because OMAS/MDSplus are not installed in CI.

NOTE: EFITLoader.load_shot and _process_time_slice now raise NotImplementedError
because this loader predates the forward-operator redesign.  Tests that called
those methods have been updated to assert the error is raised, and the
_interpolate_to_grid helper (which has no redesign dependency) is still tested
directly.
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
# test_load_shot_raises_not_implemented
# ---------------------------------------------------------------------------


def test_load_shot_raises_not_implemented():
    """load_shot raises NotImplementedError: loader predates the redesign."""
    from gsfno.data.real import EFITLoader

    with patch("gsfno.data.real.OMAS_AVAILABLE", True):
        loader = EFITLoader("jet")

    with pytest.raises(NotImplementedError, match="redesign"):
        loader.load_shot(92213)


# ---------------------------------------------------------------------------
# test_process_time_slice_raises_not_implemented
# ---------------------------------------------------------------------------


def test_process_time_slice_raises_not_implemented():
    """_process_time_slice raises NotImplementedError: loader predates the redesign."""
    from gsfno.data.real import EFITLoader

    with patch("gsfno.data.real.OMAS_AVAILABLE", True):
        loader = EFITLoader("jet")

    with pytest.raises(NotImplementedError, match="redesign"):
        loader._process_time_slice({}, 0)
