import pytest
from LRNN.utils import *

def test_sigmoid():
    assert sigmoid(1)>0.73
    assert sigmoid(1)<0.7311

def test_initialize_with_zeros():
    w,b = initialize_with_zeros(10)
    assert b == 0
    assert w.shape[0] == 10
    assert w.shape[1] == 1
    
    
