"""Window protocol: 30 Hz resampling by timestamp + 16-frame non-overlapping windows."""
import numpy as np

from extract_features import resample_index, window_indices


def test_resample_identity_at_30fps():
    assert np.array_equal(resample_index(103, 30.0, 30.0), np.arange(103))


def test_resample_12fps_repeats_frames_and_keeps_duration():
    r = resample_index(45, 12.0, 30.0)          # 3.75 s
    assert len(r) == 112                         # floor(3.75 * 30)
    assert r[0] == 0 and r[-1] == 44 and np.all(np.diff(r) >= 0)
    assert np.array_equal(r[:6], [0, 0, 0, 1, 1, 2])   # frame i -> floor(i * 12 / 30)


def test_resample_native():
    assert np.array_equal(resample_index(10, 12.0, None), np.arange(10))


def test_windows_no_overlap_plus_end_aligned():
    w = window_indices(40, 16, 1, 16, end_align=True)
    assert w[:, 0].tolist() == [0, 16, 24] and w[-1, -1] == 39
    assert window_indices(32, 16, 1, 16, end_align=True)[:, 0].tolist() == [0, 16]   # already ends on the last frame
    assert window_indices(10, 16, 1, 16, end_align=True).shape == (1, 16)               # short video: padded
