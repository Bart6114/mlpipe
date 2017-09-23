import pytest
import sys
import os

def pytest_sessionstart(session):
    """ before session.main() is called. """
    mod_path = os.path.abspath('./src/')
    print('adding path to syspath: ' + mod_path)
    sys.path.insert(0, mod_path)


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items):
    ## move tests related to no sklearn availability to the back
    items_to_move = list(filter(lambda x: x.name.startswith('test_no_sklearn_'), items))

    while len(items_to_move) > 0:
        items.append(items.pop(items.index(items_to_move[0])))
        items_to_move.pop(0)
