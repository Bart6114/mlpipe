import os
import sys

from importlib import reload
import warnings
import mlpipe as mp
import pytest

@pytest.mark.filterwarnings('ignore:optional')
@pytest.fixture(scope='module')
def no_sklearn(request):
    os.environ['MLPIPE_WITHOUT_SKLEARN'] = "1"

    reload(mp)
    reload(mp.pipeline)

    def no_sklearn_teardown():
        pass

    request.addfinalizer(no_sklearn_teardown)

    return not mp.pipeline.SKLEARN_AVAILABLE



def test_no_sklearn_global_var(no_sklearn):
    assert no_sklearn == True


def test_no_sklearn_verb(no_sklearn):
    with pytest.warns(UserWarning) as w_info:
        # should issue a warning
        p = mp.Pipe() + (lambda x: x)
        p.fit(3)

