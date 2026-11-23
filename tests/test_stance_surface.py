import numpy as np
import pytest
from go2_rescue.geometry import rot2
from go2_rescue.stance_surface import StanceSurface
from go2_rescue.types import Frame


def observation(stamp, phase, gradient, yaw=0.):
    # Two diagonal support feet per phase, on a plane with known grade.
    xy=np.array([[.23,.14],[-.23,-.14]]) if phase==0 else np.array([[.23,-.14],[-.23,.14]])
    local=rot2(yaw).T@np.asarray(gradient)
    soles=np.column_stack((xy,xy@local-.33))
    return Frame(stamp,np.empty((0,3)),quaternion=np.array([np.cos(yaw/2),0,0,np.sin(yaw/2)]),
                 support_points=soles)


@pytest.mark.parametrize("gradient",[[0.,0.],[np.tan(np.deg2rad(18.)),0.],[.12,-.15]])
def test_alternating_support_recovers_the_surface(gradient):
    estimator=StanceSurface()
    for index in range(6):
        frame=observation(index*.1,index%2,gradient,yaw=.4)
        plane=estimator.update(frame)
    assert estimator.status["source"]=="joint_imu_stance_plane"
    assert np.allclose(plane[:2],rot2(.4).T@gradient,atol=.001)
    assert abs(plane[2]+.33)<.001


def test_one_pair_does_not_claim_observed_two_dimensional_slope():
    estimator=StanceSurface()
    estimator.update(observation(0.,0,[.2,0.]))
    assert estimator.status["source"]=="joint_imu_stance_horizontal"
    assert estimator.status["reason"]=="pair_directions_incomplete"


def test_missing_support_rejects_old_plane():
    estimator=StanceSurface()
    estimator.update(observation(0.,0,[.2,0.]))
    estimator.update(observation(.1,1,[.2,0.]))
    assert estimator.update(Frame(.2,np.empty((0,3)))) is None
    assert estimator.status["source"]=="unavailable"
