import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('final_distance',Path(__file__).with_name('distribution_results.py'))
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_exact_agreement():
    assert module.squared_distance(np.eye(7)[:2],np.eye(10)[:2])==0


def test_sum_not_class_mean():
    assert module.squared_distance(np.eye(7)[:1],np.eye(10)[1:2])==2


def test_unsupported_votes_not_renormalized():
    q=np.zeros((1,10));q[0,0]=.5;q[0,8]=.5
    assert module.squared_distance(np.eye(7)[:1],q)==.5


def test_average_over_images():
    assert module.squared_distance(np.eye(7)[:2],np.eye(10)[[0,2]])==1


@pytest.mark.parametrize('p,q',[(np.zeros((0,7)),np.zeros((0,10))),
                              (np.ones((1,8))/8,np.ones((1,10))/10),
                              (np.ones((1,7))/7,np.ones((1,7))/7),
                              (np.full((1,7),np.nan),np.ones((1,10))/10),
                              (np.ones((1,7)),np.ones((1,10))/10)])
def test_invalid_inputs(p,q):
    with pytest.raises(ValueError):module.squared_distance(p,q)
