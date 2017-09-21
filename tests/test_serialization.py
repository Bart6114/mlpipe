from src.mlpipe import Pipe, Segment
import os

import pytest

@pytest.fixture(scope='module')
def resource_tmp_file(request):
    tmp_filename = 'test.pkl'
    print(tmp_filename)
    def resource_tmp_file_teardown():
        os.remove(tmp_filename)
    request.addfinalizer(resource_tmp_file_teardown)
    return tmp_filename

def test_dumps():
    p = Pipe() +\
        Segment(lambda x: x+1, "step1") + \
        Segment(lambda x: x+1, "step2")

    p._dumps()

def test_save(resource_tmp_file):
    p = Pipe() +\
        Segment(lambda x: x+1, "step1") + \
        Segment(lambda x: x+1, "step2")

    p._save(resource_tmp_file)

def test_load_after_save(resource_tmp_file):
    p = Pipe._load(resource_tmp_file)

    assert p(2) == 4


def test_dump(resource_tmp_file):
    # synonym of save
    p = Pipe() +\
        Segment(lambda x: x+1, "step1") + \
        Segment(lambda x: x+1, "step2")

    p._dump(resource_tmp_file)

def test_load_after_dump(resource_tmp_file):
    p = Pipe._load(resource_tmp_file)

    assert p(2) == 4






