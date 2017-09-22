import pytest
import sys
import os

@pytest.fixture(scope="session", autouse=True)
def do_something(request):
    print(3333)
    # prepare something ahead of all tests
    # request.addfinalizer(finalizer_function)



def pytest_sessionstart(session):
    """ before session.main() is called. """
    mod_path = os.path.abspath('./src/')
    print('adding path to syspath: ' + mod_path)
    sys.path.insert(0, mod_path)
