import numpy as np
import pytest
from gsfno.units import Normalization, reference_psi


def test_reference_psi_formula():
    assert reference_psi(R0=1.0, Ip=1.0e6) == pytest.approx(1.2566370614e-6 * 1.0e6 * 1.0)


def test_length_roundtrip():
    n = Normalization(R0=1.7, psi_ref=2.0)
    x = np.array([0.5, 1.7, 2.5])
    back = n.length_to_physical(n.length_to_dimensionless(x))
    np.testing.assert_allclose(back, x, rtol=1e-12)


def test_flux_roundtrip():
    n = Normalization(R0=1.7, psi_ref=2.0)
    psi = np.array([-0.3, 0.0, 1.4])
    back = n.flux_to_physical(n.flux_to_dimensionless(psi))
    np.testing.assert_allclose(back, psi, rtol=1e-12)


def test_dimensionless_scaling_is_global_not_per_sample():
    # Two samples with different magnitudes keep their RELATIVE scale after norm.
    n = Normalization(R0=1.0, psi_ref=10.0)
    a = n.flux_to_dimensionless(np.array([1.0, 2.0]))   # small-magnitude sample
    b = n.flux_to_dimensionless(np.array([10.0, 20.0])) # large-magnitude sample
    # b is 10x a — the magnitude information is preserved (unlike per-sample minmax).
    np.testing.assert_allclose(b, 10.0 * a, rtol=1e-12)


def test_attrs_roundtrip():
    n = Normalization(R0=1.7, psi_ref=2.0)
    assert Normalization.from_attrs(n.to_attrs()) == n
