"""IMU-rotation constrained metric visual translation, using measured wall ranges.

3-D feature depths come from lidar-fitted vertical walls or the measured stance
plane, never renderer depth or scene geometry. LK + robust known-rotation
translation rejects repeated textures and mixed-surface correspondences.
"""
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from .geometry import gravity_rotation


class MetricVisualOdometry:
    def __init__(self,config):
        self.c=config;self.k=np.asarray(config.camera_matrix);self.mount=np.asarray(config.body_from_camera)
        self.previous=None;self.status={'source':'unavailable'}
        self.rng=np.random.default_rng(88)

    def depths(self,frame,plane,probability,gray):
        features=cv2.goodFeaturesToTrack(gray,450,.008,6,blockSize=5)
        if features is None:return np.empty((0,2)),np.empty((0,3))
        uv=features[:,0,:];rays=np.column_stack([(uv[:,0]-self.k[0,2])/self.k[0,0],(uv[:,1]-self.k[1,2])/self.k[1,1],np.ones(len(uv))])
        origin=self.mount[:3,3];directions=rays@self.mount[:3,:3].T
        gravity=gravity_rotation(frame.quaternion);go=origin@gravity.T;gd=directions@gravity.T
        uvint=np.rint(uv).astype(int);kind=probability.argmax(0)[uvint[:,1],uvint[:,0]]
        distance=np.full(len(uv),np.inf);source=np.zeros(len(uv),np.uint8)
        # Wall XY fits use positive-height returns only; terrain is handled below.
        points=frame.points@gravity.T;cloud=points[(points[:,2]>.02)&(points[:,2]<2.),:2]
        for _ in range(5):
            if len(cloud)<6:break
            best=np.zeros(len(cloud),bool)
            for _ in range(45):
                pair=self.rng.choice(len(cloud),2,replace=False);t=cloud[pair[1]]-cloud[pair[0]]
                if np.linalg.norm(t)<.25:continue
                n=np.array([-t[1],t[0]])/np.linalg.norm(t);inlier=abs((cloud-cloud[pair[0]])@n)<.025
                if inlier.sum()>best.sum():best=inlier
            if best.sum()<6:break
            selected=cloud[best];center=selected.mean(0);values,vectors=np.linalg.eigh(np.cov(selected.T));normal=vectors[:,0];tangent=vectors[:,1]
            if values[0]>.0025 or values[1]<.02:cloud=cloud[~best];continue
            denominator=gd[:,:2]@normal
            scale=np.divide((center-go[:2])@normal,denominator,out=np.full(len(uv),np.inf),where=abs(denominator)>.1)
            xy=go[:2]+scale[:,None]*gd[:,:2];span=selected@tangent;along=xy@tangent
            eligible=(kind==2)&(scale>.2)&(scale<6.)&(along>span.min()-.12)&(along<span.max()+.12)&(scale<distance)
            distance[eligible]=scale[eligible];source[eligible]=2
            cloud=cloud[~best]
        if plane is not None:
            denominator=gd[:,2]-gd[:,:2]@plane[:2]
            scale=np.divide(plane[2]+go[:2]@plane[:2]-go[2],denominator,out=np.full(len(uv),np.inf),where=denominator<-.08)
            eligible=(kind==0)&(scale>.25)&(scale<2.5)&(scale<distance)
            distance[eligible]=scale[eligible];source[eligible]=1
        valid=np.isfinite(distance)
        # Prefer lidar-scaled wall texture when enough of it is visible.
        if (source==2).sum()>=25:valid=source==2
        return uv[valid].astype(np.float32),(rays[valid]*distance[valid,None]).astype(float)

    def update(self,frame,plane,probability):
        gray=cv2.cvtColor(frame.image,cv2.COLOR_BGR2GRAY);result=None
        self.status={'source':'unavailable','reason':'initial_frame'}
        if self.previous is not None:
            old=self.previous;dt=frame.timestamp-old['t']
            if .04<dt<1.1 and len(old['uv'])>=15:
                tracked,status,_=cv2.calcOpticalFlowPyrLK(old['gray'],gray,old['uv'][:,None],None,winSize=(21,21),maxLevel=3)
                back,back_status,_=cv2.calcOpticalFlowPyrLK(gray,old['gray'],tracked,None,winSize=(21,21),maxLevel=3)
                keep=status[:,0].astype(bool)&back_status[:,0].astype(bool)&(np.linalg.norm(back[:,0]-old['uv'],axis=1)<.7)
                now=tracked[:,0][keep];obj=old['xyz'][keep]
                rp=Rotation.from_quat(np.roll(old['quaternion'],-1)).as_matrix();rc=Rotation.from_quat(np.roll(frame.quaternion,-1)).as_matrix()
                rotation=self.mount[:3,:3].T@rc.T@rp@self.mount[:3,:3]
                transformed=obj@rotation.T
                normalized=(now-self.k[:2,2])/np.array([self.k[0,0],self.k[1,1]])
                if len(now)>=15:
                    matrix=np.zeros((len(now),2,3));matrix[:,0,0]=1;matrix[:,1,1]=1;matrix[:,:,2]=-normalized
                    rhs=normalized*transformed[:,2:]-transformed[:,:2]
                    best=np.zeros(len(now),bool);translation=None
                    for _ in range(75):
                        sample=self.rng.choice(len(now),3,replace=False)
                        value=np.linalg.lstsq(matrix[sample].reshape(-1,3),rhs[sample].ravel(),rcond=None)[0]
                        projected=(transformed+value)@self.k.T;positive=projected[:,2]>.1
                        pixel=projected[:,:2]/np.maximum(projected[:,2:],1e-6)
                        error=np.linalg.norm(pixel-now,axis=1);valid=positive&(error<1.5)
                        if valid.sum()>best.sum():best=valid;translation=value
                    if best.sum()>=15 and best.mean()>=.5:
                        translation=np.linalg.lstsq(matrix[best].reshape(-1,3),rhs[best].ravel(),rcond=None)[0]
                        relative=rp.T@rc
                        delta=self.mount[:3,:3]@(-rotation.T@translation)+self.mount[:3,3]-relative@self.mount[:3,3]
                        delta=gravity_rotation(old['quaternion'])@delta
                        if np.linalg.norm(delta)<.65*dt+.05:
                            result={'dt':dt,'delta_gravity':delta,'from_stamp':old['t']}
                            self.status={'source':'lidar_scaled_visual_imu','inliers':int(best.sum()),'tracked':len(now),'dt_s':dt,'translation_m':delta.tolist()}
                        else:self.status={'source':'unavailable','reason':'motion_bound'}
                    else:self.status={'source':'unavailable','reason':'insufficient_consensus','tracks':len(now),'inliers':int(best.sum())}
        uv,xyz=self.depths(frame,plane,probability,gray)
        self.previous={'t':frame.timestamp,'gray':gray,'uv':uv,'xyz':xyz,'quaternion':frame.quaternion.copy()}
        self.status['next_depth_features']=len(uv)
        return result
