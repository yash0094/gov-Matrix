"""Regression tests. Run: pytest -q"""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.engine import core
from src.engine.pipeline import analyse


@pytest.fixture(scope="module")
def result():
    t, b, inj = core.make_universe(n_tenders=1500, n_cartels=2)
    return analyse(t, b, alpha=0.10), inj


def test_peer_groups_never_empty(result):
    res, _ = result
    assert res["tenders"].peer_group.notna().all()
    assert res["tenders"].peer_level.isin(["L0", "L1", "L2", "L3"]).all()


def test_robust_z_survives_contamination():
    """30% of the sample poisoned; the median must stay within 1 sigma.

    30% contamination pushes the median to roughly the 71st percentile of the
    clean data -- about 0.55 sigma. That is the whole argument for median/MAD:
    the mean moves by 60+ sigma on the same data.
    """
    clean = pd.Series(np.random.default_rng(0).normal(100, 5, 700))
    dirty = pd.concat([clean, pd.Series(np.full(300, 400.0))], ignore_index=True)
    assert abs(clean.median() - dirty.median()) < 5.0      # < 1 sigma
    assert abs(clean.mean() - dirty.mean()) > 50.0         # the mean does not


def test_pvalues_are_uniform_under_the_null():
    """Rank-based p-values on random data must be ~U(0,1) or the FDR is a lie."""
    rng = np.random.default_rng(1)
    s = pd.Series(rng.normal(size=3000))
    peer = pd.Series(rng.integers(0, 30, 3000)).astype(str)
    p = core.empirical_p(s, peer)
    assert stats.kstest(p, "uniform").pvalue > 0.01


def test_stouffer_contributions_sum_to_total(result):
    res, _ = result
    recon = res["contrib"].sum(axis=1) * res["tenders"].peer_confidence.values
    assert np.allclose(recon.values, res["tenders"].Z.values, atol=1e-9)


def test_bh_is_monotone_and_bounded():
    p = pd.Series(np.r_[np.linspace(1e-8, 1e-4, 10), np.random.uniform(size=490)])
    rej, cut = core.benjamini_hochberg(p, alpha=0.10)
    assert rej.sum() >= 5
    assert 0 <= cut <= 1
    assert p[rej].max() <= cut


def test_specialised_markets_are_not_flagged(result):
    """The whole point. Two-supplier markets must not dominate the queue."""
    res, _ = result
    cases = res["cases"]
    spec = cases[cases.category.isin(["mri_coil_spares", "turbine_blades"])]
    assert spec.flagged.sum() == 0


def test_cartels_rank_near_the_top(result):
    res, _ = result
    cases = res["cases"]
    n = int(cases.is_cartel_cell.sum())
    assert n >= 1
    assert cases.head(max(n * 2, 4)).is_cartel_cell.sum() == n


def test_ranking_beats_chance(result):
    res, _ = result
    assert res["metrics"]["roc_auc"] > 0.70
