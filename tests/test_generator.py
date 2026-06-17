"""Tests for gsfno.data.generator.PlasmaConfigGenerator (Task 3).

Tests are written for the free-boundary forward-operator redesign.
FreeGS is a required dependency for the integration tests; they are
skipped automatically when FreeGS is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

freegs = pytest.importorskip("freegs")
from gsfno.data.generator import PlasmaConfigGenerator, FREEGS_AVAILABLE


@pytest.fixture(scope="module")
def gen():
    return PlasmaConfigGenerator(NR=33, NZ=33, n_psi=16, seed=0)


# ---------------------------------------------------------------------------
# Grid / coordinate helpers
# ---------------------------------------------------------------------------

def test_grid_shapes(gen):
    R, Z = gen.grid()
    assert R.shape == (33,) and Z.shape == (33,)
    assert gen.psiN_grid().shape == (16,)


# ---------------------------------------------------------------------------
# generate_one output structure
# ---------------------------------------------------------------------------

def test_generate_one_structure(gen):
    s = None
    for _ in range(20):
        s = gen.generate_one()
        if s is not None:
            break
    assert s is not None, "generator failed to converge in 20 tries"
    assert s["psi_total"].shape == (33, 33)
    assert s["psi_vac"].shape == (33, 33)
    assert s["pprime_curve"].shape == (16,)
    assert s["ffprime_curve"].shape == (16,)
    assert "Ip" in s and "R0" in s
    assert set(["paxis", "Ip", "fvac", "alpha_m", "alpha_n"]).issubset(s["params"])


# ---------------------------------------------------------------------------
# psi_vac is strictly NOT equal to psi_total (plasma contributes flux)
# ---------------------------------------------------------------------------

def test_no_answer_leak(gen):
    s = None
    for _ in range(20):
        s = gen.generate_one()
        if s is not None:
            break
    assert s is not None, "generator failed to converge in 20 tries"
    diff = np.abs(s["psi_total"] - s["psi_vac"]).max()
    assert diff > 0, "psi_vac must differ from psi_total (plasma contributes flux)"


# ---------------------------------------------------------------------------
# Residual validation gate — two-sided test (pass + fail)
# ---------------------------------------------------------------------------

def test_validation_gate_accepts_converged(gen):
    """A genuinely converged equilibrium passes validate()."""
    s = None
    for _ in range(20):
        s = gen.generate_one()
        if s is not None:
            break
    assert s is not None, "generator failed to converge in 20 tries"
    assert gen.validate(s) is True, "converged equilibrium should pass validate()"


def test_validation_gate_rejects_corrupted(gen):
    """A deliberately corrupted psi_total fails validate()."""
    s = None
    for _ in range(20):
        s = gen.generate_one()
        if s is not None:
            break
    assert s is not None, "generator failed to converge in 20 tries"

    # Corrupt psi_total with large noise (std ~= signal std)
    corrupted = dict(s)
    noise = np.random.default_rng(99).normal(
        0, float(np.std(s["psi_total"])), s["psi_total"].shape
    ).astype(np.float32)
    corrupted["psi_total"] = s["psi_total"] + noise

    assert gen.validate(corrupted) is False, (
        "corrupted equilibrium (large noise) should fail validate()"
    )
