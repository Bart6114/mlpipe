from src.mlpipe import Pipe
import pytest

def test_error_non_callable():
    # non callable objects should raise exception
    with pytest.raises(Exception) as e_info:
        p = Pipe() + 3
