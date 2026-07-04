from app import stats


def test_mean_and_median():
    assert stats.mean([1, 2, 3]) == 2
    assert stats.median([1, 2, 3, 4]) == 2.5
