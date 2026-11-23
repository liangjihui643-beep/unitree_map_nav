"""Causal RGB and stance-based rubble point cloud without scene or LiDAR input."""
from time import perf_counter

import cv2
import numpy as np
import onnxruntime as ort


SIZE = (160, 120)


def pixel_rays(k, sensor_size):
    x, y = np.meshgrid(np.arange(SIZE[0]), np.arange(SIZE[1]))
    u = (x + .5) * sensor_size[0] / SIZE[0] - .5
    v = (y + .5) * sensor_size[1] / SIZE[1] - .5
    return (np.stack((u, v, np.ones_like(u)), axis=-1) @ np.linalg.inv(k).T).astype(np.float32)


class RGBRubbleDepth:
    def __init__(self, path, k, sensor_size, *, threshold, threads=4):
        if not 0 < threshold <= 1:
            raise ValueError('Rubble confidence threshold must be in (0, 1]')
        self.rays = pixel_rays(k, sensor_size)
        self.threshold = threshold
        self.seconds = 0.
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.model = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
        _, channels, h, w = self.model.get_inputs()[0].shape
        if channels != 3 or (w,h) not in (SIZE,(64,64)):
            raise ValueError('Unsupported rubble model input shape')
        self.size = (w, h)

    def infer(self, bgr, gravity, camera_height):
        start = perf_counter()
        rgb = bgr[..., ::-1].astype(np.float32) / 255.
        interpolation = cv2.INTER_LINEAR if self.size == (64,64) else cv2.INTER_AREA
        inp = cv2.resize(rgb,self.size,interpolation=interpolation).transpose(2,0,1)[None].copy()
        output = self.model.run(None, {'rgb':inp, 'gravity':np.float32(gravity)[None],
                                       'camera_height':np.float32([[camera_height]])})
        self.seconds = perf_counter()-start
        depth, relief, prob = (result[0] for result in output)
        return depth, relief, prob[2]

    def select(self, depth, relief, score, screen_mask=None):
        valid = (score >= self.threshold) & (depth >= .40) & (depth <= 3.60) & (relief >= .018)
        if screen_mask is not None:
            valid &= cv2.resize(screen_mask.astype(np.uint8), SIZE,
                                interpolation=cv2.INTER_NEAREST_EXACT) > 0
        y, x = np.where(valid)
        return y, x

    def project(self, depth, relief, y, x, rotation, origin):
        xyz = origin + depth[y,x,None] * (self.rays[y,x] @ rotation.T)
        local_floor = xyz[:,2]-relief[y,x]
        return np.column_stack((xyz, local_floor, relief[y,x]))
