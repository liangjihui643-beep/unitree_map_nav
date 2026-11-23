"""Refine a geometric ball proposal against the visible outer silhouette."""
import cv2
import numpy as np
from scipy.optimize import least_squares


def refine_ball_outline(image, detection):
    x, y = detection.center
    r = float(detection.radius)
    if r < 8.:
        return False
    h, w = image.shape[:2]
    radius = int(np.ceil(r*1.5+4))
    x0, y0 = max(0, int(x)-radius), max(0, int(y)-radius)
    x1, y1 = min(w, int(x)+radius+1), min(h, int(y)+radius+1)
    gray = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), .6)
    gx, gy = [cv2.Sobel(gray, cv2.CV_32F, dx, dy) for dx, dy in ((1,0),(0,1))]
    k = 48
    theta = np.arange(k)*(2*np.pi/k)
    unit = np.column_stack((np.cos(theta),np.sin(theta)))
    scales = np.arange(max(4., .76*r), 1.28*r+.5, .5)
    uv = np.array([x-x0,y-y0])+unit[:,None,:]*scales[None,:,None]
    uu, vv = [uv[...,i].astype(np.float32) for i in (0,1)]
    grad = np.stack([cv2.remap(g, uu, vv, cv2.INTER_LINEAR) for g in (gx,gy)], axis=-1)
    radial = abs((grad*unit[:,None]).sum(axis=-1))
    mag = np.linalg.norm(grad,axis=-1)
    radial[(radial < .80*mag) | (uu<2) | (vv<2) |
           (uu>=gray.shape[1]-2) | (vv>=gray.shape[0]-2)] = 0
    weighted = radial*np.exp(-.5*((scales/r-1)/.18)**2)
    idx = weighted.argmax(axis=1)
    valid = radial[np.arange(k),idx]>=65.
    if valid.sum()<16:
        return False
    selected = uv[np.arange(k),idx][valid]+[x0,y0]
    a = np.array([x,y,r])
    model=lambda b:np.linalg.norm(selected-b[:2],axis=1)-b[2]
    for _ in range(2):
        # Fit robustly; silhouette needs support on opposing sides.
        result=least_squares(model,a,loss="soft_l1",f_scale=1.1,max_nfev=18)
        if not result.success:
            return False
        a=result.x
        residual=abs(model(a))
        inside=residual < max(1.25,.07*r)
        if inside.sum()<14 or len(np.unique(np.flatnonzero(valid)[inside]//8))<5:
            return False
        selected=selected[inside]
        old=np.flatnonzero(valid)
        valid[old[~inside]]=False
    if np.linalg.norm(a[:2]-[x,y])>.20*r or not .78*r<=a[2]<=1.23*r:
        return False
    detection.center=tuple(a[:2])
    detection.radius=float(a[2])
    return True
