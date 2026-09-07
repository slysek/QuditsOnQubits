import numpy as np
from analyze import inverse_readout_vectors


def test_readout_correction_recovers_asymmetric_entangled_distribution():
    a=np.array([[.9,.2],[.1,.8]])
    b=np.array([[.97,.13],[.03,.87]])
    true=np.array([.4,.1,.1,.4])
    channel=np.kron(b,a)
    observed=channel@true
    score=np.array([1.,-1.,-1.,1.])
    corrected=inverse_readout_vectors(score,{5:np.linalg.inv(a),8:np.linalg.inv(b)},[5,8])
    np.testing.assert_allclose(observed@corrected,true@score,atol=1e-12)


def test_identity_correction_changes_no_scores():
    score=np.arange(256.)
    corrected=inverse_readout_vectors(score,{q:np.eye(2) for q in range(8)},list(range(8)))
    np.testing.assert_array_equal(score,corrected)
