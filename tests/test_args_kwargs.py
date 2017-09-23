from mlpipe import Pipe, Segment

def my_func(x, b=0):
    return x + b

def test_kwargs():
    p = Pipe() + Segment(my_func, b=1)

    assert p(1) == 2
    assert p.bla(1) == 2


def test_kwargs2():
    p = Pipe() +\
        Segment(my_func, description="test", b=1) +\
        Segment(my_func, description="test", b=2)

    assert p(1) == 4
    assert p.bla(1) == 4

# def test_args():
#     p = Pipe() +\
#         Segment(my_func, description="test", 1)
#
#     assert p(1) == 2