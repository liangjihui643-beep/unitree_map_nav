from types import SimpleNamespace

import cv2
import numpy as np

from go2_rescue.rubble_multiview import select_sources, sweep_multiview


def test_joint_depth_requires_distinct_camera_centres():
    camera = np.array([[208., 0., 160.], [0., 208., 120.], [0., 0., 1.]])
    rgb = cv2.GaussianBlur(np.random.default_rng(12).integers(0, 255, (240, 320), np.uint8),
                           (7, 7), 1.2)
    depth = 2.

    def view(x, y, stamp):
        center = np.array([x, y, 0.])
        affine = np.array([[1., 0., -x*camera[0,0]/depth], [0., 1., -y*camera[1,1]/depth]])
        gray = cv2.warpAffine(rgb, affine, (320, 240))
        return SimpleNamespace(camera=camera@np.column_stack((np.eye(3), -center)),
                               gray=gray, labels=np.ones_like(gray), center=center,
                               stamp=stamp, from_level=np.eye(3))

    current = view(0., 0., 1.)
    sources = [view(.32,0.,.2), view(-.2,.15,.4), view(.33,0.,.5), view(0.,0.,.6)]
    distinct = select_sources(current, sources)
    assert len(distinct) == 2
    assert len(sweep_multiview(camera, distinct[:1], current)[0]) == 0
    points, labels = sweep_multiview(camera, distinct, current)
    assert len(points) > 100
    assert (labels == 1).all()
    assert np.quantile(abs(points[:,2]-depth), .9) < .08
