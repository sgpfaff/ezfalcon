import numpy as np
import pytest

from tambora.tools import sample_kroupa_imf
from tambora.tools.imf_tools import _sample_broken_power_law


def test_sample_kroupa_imf_shape_and_bounds():
    m = sample_kroupa_imf(5000, m_min=0.08, m_max=50.0, rng=0)
    assert m.shape == (5000,)
    assert m.min() >= 0.08
    assert m.max() <= 50.0


def test_sample_kroupa_imf_reproducible():
    m1 = sample_kroupa_imf(1000, rng=42)
    m2 = sample_kroupa_imf(1000, rng=42)
    np.testing.assert_array_equal(m1, m2)


def test_sample_kroupa_imf_m_total_rescale():
    m_total = 1e5
    m = sample_kroupa_imf(2000, m_total=m_total, rng=1)
    np.testing.assert_allclose(m.sum(), m_total)


def test_sample_kroupa_imf_top_heavy_without_m_total():
    # Kroupa is bottom-heavy: most stars should sit well below the mean mass.
    m = sample_kroupa_imf(200_000, m_min=0.08, m_max=50.0, rng=2)
    assert np.median(m) < m.mean()
    assert (m < 0.5).sum() / m.size > 0.5


def test_sample_kroupa_imf_invalid_range():
    with pytest.raises(ValueError, match="m_min < m_max"):
        sample_kroupa_imf(10, m_min=1.0, m_max=0.5)


def test_sample_broken_power_law_single_segment_matches_salpeter_slope():
    rng = np.random.default_rng(0)
    alpha = 2.35
    breaks = np.array([1.0, 100.0])
    m = _sample_broken_power_law(200_000, breaks, np.array([alpha]), rng)
    assert m.min() >= 1.0
    assert m.max() <= 100.0
    # For a single power law dN/dm ~ m^-alpha, log10(N(>m)) vs log10(m) has
    # slope -(alpha - 1); check it against a direct fit.
    logm = np.log10(np.sort(m))
    log_n_above = np.log10(np.arange(m.size, 0, -1))
    slope = np.polyfit(logm[: m.size // 2], log_n_above[: m.size // 2], 1)[0]
    np.testing.assert_allclose(slope, -(alpha - 1), atol=0.1)
