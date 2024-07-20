import evals.utils as utils


def test_get_default_parameters():
    class MyClass:
        def __init__(self, a=1, b=2, c=None):
            self.a = a
            self.b = b
            self.c = c

    default_params = utils.get_default_parameters(MyClass)
    assert default_params == {"a": 1, "b": 2, "c": None}
