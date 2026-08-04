import numpy as np
import pytest

from lob.engine import zero_latency
from lob.latency import LATENCY_PRESET_PARAMS, lognormal_latency_model, make_latency_model


def test_rejects_nonpositive_median():
    with pytest.raises(ValueError):
        lognormal_latency_model(median_seconds=0.0, sigma=0.3)
    with pytest.raises(ValueError):
        lognormal_latency_model(median_seconds=-0.001, sigma=0.3)


def test_rejects_nonpositive_sigma():
    with pytest.raises(ValueError):
        lognormal_latency_model(median_seconds=0.005, sigma=0.0)


def test_delay_is_always_nonnegative():
    rng = np.random.default_rng(0)
    model = lognormal_latency_model(median_seconds=0.005, sigma=0.5, rng=rng)
    for _ in range(1000):
        assert model(10.0) >= 10.0


def test_reproducible_with_seeded_rng():
    model_a = lognormal_latency_model(0.005, 0.3, rng=np.random.default_rng(42))
    model_b = lognormal_latency_model(0.005, 0.3, rng=np.random.default_rng(42))
    samples_a = [model_a(0.0) for _ in range(20)]
    samples_b = [model_b(0.0) for _ in range(20)]
    assert samples_a == samples_b


def test_sample_median_close_to_configured_median():
    rng = np.random.default_rng(7)
    model = lognormal_latency_model(median_seconds=0.020, sigma=0.4, rng=rng)
    delays = [model(0.0) for _ in range(20000)]
    assert np.median(delays) == pytest.approx(0.020, rel=0.1)


def test_make_latency_model_zero_preset_is_exact_zero_latency():
    model = make_latency_model("0ms")
    assert model is zero_latency
    assert model(5.0) == 5.0


def test_make_latency_model_unknown_preset_raises():
    with pytest.raises(ValueError):
        make_latency_model("100ms")


def test_presets_ordered_by_increasing_median_latency():
    medians = [LATENCY_PRESET_PARAMS[p][0] for p in ("0ms", "5ms", "20ms", "50ms")]
    assert medians == sorted(medians)
    assert medians[0] == 0.0


def test_nonzero_presets_produce_increasing_mean_delay():
    rng = np.random.default_rng(1)
    means = {}
    for preset in ("5ms", "20ms", "50ms"):
        model = make_latency_model(preset, rng=rng)
        means[preset] = np.mean([model(0.0) for _ in range(5000)])
    assert means["5ms"] < means["20ms"] < means["50ms"]
