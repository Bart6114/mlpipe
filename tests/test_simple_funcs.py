from src.mlpipe import Pipe, Segment

def test_lambda1_wo_attr():
    p = Pipe() +\
        Segment(lambda x: x+2)

    assert p(5) == 7


def test_lambda2_wo_attr():
    p = Pipe() +\
        Segment(lambda x: x+2) + \
        Segment(lambda x: x + 2)

    assert p(5) == 9

def test_lambda1_w_attr():
    p = Pipe() +\
        Segment(lambda x: x+2)

    assert p.testattr(5) == 7


def test_lambda2_w_attr():
    p = Pipe() +\
        Segment(lambda x: x+2) + \
        Segment(lambda x: x + 2)

    assert p.testattr(5) == 9

def test_no_input():
    p = Pipe() +\
        Segment(lambda *_: 2)

    p2 = Pipe() + \
        Segment(lambda *_: 2) + \
        Segment(lambda x: x)

    assert p() == 2
    assert p(5) == 2

    assert p2() == 2
    assert p2(5) == 2
