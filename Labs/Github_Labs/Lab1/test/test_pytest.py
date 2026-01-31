import pytest
from src import calculator

def test_fun1():
    # average
    assert calculator.fun1(2, 3) == 2.5
    assert calculator.fun1(5, 0) == 2.5
    assert calculator.fun1(-1, 1) == 0
    assert calculator.fun1(-1, -1) == -1


def test_fun2():
    # absolute difference
    assert calculator.fun2(2, 3) == 1
    assert calculator.fun2(5, 0) == 5
    assert calculator.fun2(-1, 1) == 2
    assert calculator.fun2(-1, -1) == 0


def test_fun3():
    # power
    assert calculator.fun3(2, 3) == 8
    assert calculator.fun3(5, 0) == 1
    assert calculator.fun3(-1, 1) == -1
    assert calculator.fun3(-1, 2) == 1


def test_fun4():
    # max of three
    assert calculator.fun4(2, 3, 5) == 5
    assert calculator.fun4(5, 0, -1) == 5
    assert calculator.fun4(-1, -1, -1) == -1
    assert calculator.fun4(-1, -1, 100) == 100
