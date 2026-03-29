"""
Tests for SubjectPrefetcher — background data loading for within-subject training.

Run with: uv run pytest tests/test_prefetch.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.prefetch import SubjectPrefetcher


@pytest.fixture
def mock_prefetcher():
    """Create prefetcher with mocked data loading."""
    with patch('src.training.prefetch.load_subject_data') as mock_load, \
         patch('src.training.prefetch.temporal_split_by_group') as mock_split:

        # Mock dataset
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=100)
        mock_load.return_value = mock_dataset
        mock_split.return_value = (list(range(80)), list(range(80, 100)))

        prefetcher = SubjectPrefetcher(
            model_type='eegnet',
            task='binary',
            paradigm='imagery',
            data_root=Path('data'),
            elc_path=Path('data/biosemi128.ELC'),
            cache_only=True,
        )
        yield prefetcher, mock_load, mock_split
        prefetcher.shutdown()


def test_prefetch_returns_precomputed_data(mock_prefetcher):
    """Prefetch produces dict with required keys."""
    prefetcher, mock_load, _ = mock_prefetcher
    prefetcher.start_prefetch('S01')
    result = prefetcher.get_prefetched('S01')

    assert result is not None
    assert 'train_dataset' in result
    assert 'train_indices' in result
    assert 'val_indices' in result
    assert 'test_dataset' in result
    assert len(result['train_indices']) == 80
    assert len(result['val_indices']) == 20


def test_prefetch_not_submitted_returns_none(mock_prefetcher):
    """get_prefetched for unknown subject returns None."""
    prefetcher, _, _ = mock_prefetcher
    assert prefetcher.get_prefetched('S99') is None


def test_prefetch_deduplicates(mock_prefetcher):
    """Submitting same subject twice does not create a second future."""
    prefetcher, mock_load, _ = mock_prefetcher
    prefetcher.start_prefetch('S01')
    prefetcher.start_prefetch('S01')  # Should be no-op
    result = prefetcher.get_prefetched('S01')
    assert result is not None
    # load_subject_data called twice (train + test), not four times
    assert mock_load.call_count == 2


def test_prefetch_failure_returns_none(mock_prefetcher):
    """If loading raises, get_prefetched returns None instead of crashing."""
    prefetcher, mock_load, _ = mock_prefetcher
    mock_load.side_effect = RuntimeError("disk error")
    prefetcher.start_prefetch('S01')
    result = prefetcher.get_prefetched('S01')
    assert result is None


def test_shutdown_is_safe(mock_prefetcher):
    """Shutdown can be called multiple times without error."""
    prefetcher, _, _ = mock_prefetcher
    prefetcher.shutdown()
    prefetcher.shutdown()  # Should not raise
