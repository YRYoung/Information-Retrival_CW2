import numpy as np

from eval import eval_per_query


def all_equal(a, b):
    assert len(a) == len(b)
    for aa, bb in zip(a, b):
        assert aa == bb or np.allclose(aa, bb)


def test_eval_per_query():
    a, b = eval_per_query(np.array([1, 2]), at=[2])
    all_equal(a, b)
    all_equal(eval_per_query(np.array([1, 2]), at=[3]),
              eval_per_query(np.array([2, 1]), at=[3]))

    all_equal(eval_per_query(np.array([1, 2, 3, 8]), at=[10]),
              eval_per_query(np.array([8, 3, 2, 1]), at=[10]))

    a, b = eval_per_query(np.array([2]), at=[1, 2])
    assert a[0] == b[0] == 0
    assert a[1] == 0.5
    assert np.allclose(b[1], 1 / np.log2(1 + 2))
