import inspect

import pytest


def get_current_method_name():
    return inspect.currentframe().f_back.f_code.co_name


# def repeat_test(times):
#     def decorator(func):
#         def wrapper(*args, **kwargs):
#             for _ in range(times):
#                 func(*args, **kwargs)
#         return wrapper
#     return decorator


class Env:
    def __init__(self, value):
        self.value = value
        self.state = "000"

    def add(self, num):
        self.value += num


@pytest.fixture
def env():
    return Env(0)


class BaseTestClass:
    def test_common_1(self, n_repeat=1):
        if get_current_method_name() in self.repeats:
            n_repeat = self.repeats[get_current_method_name()]

        for _ in range(n_repeat):
            print("\ntest 1:")
            self.env.add(1)
            print("Value: {}, State: {}".format(self.env.value, self.env.state))

    def test_common_2(self, n_repeat=1):
        if get_current_method_name() in self.repeats:
            n_repeat = self.repeats[get_current_method_name()]

        for _ in range(n_repeat):
            print("\ntest 2")
            self.env.state += "+"
            print("Value: {}, State: {}".format(self.env.value, self.env.state))

    def test_common_3(self, n_repeat=1):
        if get_current_method_name() in self.repeats:
            n_repeat = self.repeats[get_current_method_name()]

        for _ in range(n_repeat):
            print("\ntest 3")
            print("Value: {}, State: {}".format(self.env.value, self.env.state))


class TestSpecificInstance1(BaseTestClass):
    @pytest.fixture(autouse=True)
    def setup(self, env):
        self.env = env
        self.repeats = {
            "test_common_2": 3,  # Overrides no repeat.
        }


class TestSpecificInstance2(BaseTestClass):
    @pytest.fixture(autouse=True)
    def setup(self, env):
        self.env = env
        self.repeats = {
            "test_common_1": 3,  # Overrides no repeat.
        }
