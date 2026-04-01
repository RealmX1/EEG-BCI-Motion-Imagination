import numpy as np
import pytest

from scripts.paper.generate_paper_figures import (
    _load_further_pretraining_series,
    extract_model_accs,
    load_json_cache,
)
from src.paper.run_registry import get_run_path


@pytest.mark.parametrize(
    ('run_key', 'expected_mean'),
    [
        ('further_pretraining_baseline_within_binary', 85.089286),
        ('further_pretraining_baseline_cross_binary', 90.535714),
        ('further_pretraining_baseline_within_ternary', 69.543651),
        ('further_pretraining_baseline_cross_ternary', 75.416667),
        ('further_pretraining_v1_within_binary', 83.839286),
        ('further_pretraining_v1_cross_binary', 88.839286),
        ('further_pretraining_v1_within_ternary', 69.246032),
        ('further_pretraining_v1_cross_ternary', 75.674603),
        ('further_pretraining_v2_within_binary', 82.232143),
        ('further_pretraining_v2_cross_binary', 89.434524),
        ('further_pretraining_v2_within_ternary', 68.075397),
        ('further_pretraining_v2_cross_ternary', 75.317460),
    ],
)
def test_extract_model_accs_supports_further_pretraining_result_formats(run_key, expected_mean):
    cache = load_json_cache(get_run_path(run_key))
    accs = extract_model_accs(cache, 'cbramod')

    assert len(accs) == 21
    assert float(np.mean(accs)) == pytest.approx(expected_mean, abs=1e-6)


def test_load_further_pretraining_series_reads_registry_backed_values():
    series = _load_further_pretraining_series()

    assert series['conditions'] == [
        'Within-Subj\nBinary',
        'Cross-Subj\nBinary',
        'Within-Subj\nTernary',
        'Cross-Subj\nTernary',
    ]
    assert series['baseline'] == pytest.approx([85.089286, 90.535714, 69.543651, 75.416667], abs=1e-6)
    assert series['ft_v1'] == pytest.approx([83.839286, 88.839286, 69.246032, 75.674603], abs=1e-6)
    assert series['ft_v2'] == pytest.approx([82.232143, 89.434524, 68.075397, 75.317460], abs=1e-6)
