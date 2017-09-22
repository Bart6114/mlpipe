from mlpipe import Pipe, Segment

def test_nonsegment_lambda2_wo_attr():
    p = Pipe() +\
        (lambda x: x+2) + \
        (lambda x: x + 2)

    assert p(5) == 9


def test_nonsegment_lambda2_w_attr():
    p = Pipe() +\
        (lambda x: x+2) + \
        (lambda x: x + 2)

    assert p.testattr(5) == 9
