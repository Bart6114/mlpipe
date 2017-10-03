from mlpipe import Pipe
import sys
import pytest

@pytest.mark.filterwarnings('ignore:Ignoring')
def test_cache_basic():
    if sys.version_info > (3, 0, 0):
        p = Pipe(cached=True) +\
            (lambda x: x)

        assert p(2) == 2
        assert p(2) == 2
        assert p.__cache_info__()[1] == 1 # misses
        assert p.__cache_info__()[0] == 1 # hits