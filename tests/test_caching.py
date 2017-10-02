from mlpipe import Pipe

def test_cache_basic():
    p = Pipe(cached=True) +\
        (lambda x: x)

    assert p(2) == 2
    assert p(2) == 2
    assert p.__cache_info__()[1] == 1 # misses
    assert p.__cache_info__()[0] == 1 # hits