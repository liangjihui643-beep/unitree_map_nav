"""Learned appearance check for geometric football proposals."""
import cv2
import numpy as np
import onnxruntime as ort


class BallProposalClassifier:
    def __init__(self, path):
        options=ort.SessionOptions()
        options.intra_op_num_threads=2
        self.session=ort.InferenceSession(str(path),sess_options=options,providers=["CPUExecutionProvider"])
        self.input=self.session.get_inputs()[0].name

    def filter(self,image,detections):
        if not detections:
            return []
        batch=[]
        for detection in detections:
            side=max(8,int(np.ceil(2.7*detection.radius)))
            patch=cv2.getRectSubPix(image,(side,side),tuple(map(float,detection.center)))
            rgb=cv2.cvtColor(cv2.resize(patch,(40,40)),cv2.COLOR_BGR2RGB)
            batch.append(rgb.transpose(2,0,1).astype(np.float32)/255.)
        logits=self.session.run(None,{self.input:np.stack(batch)})[0]
        probabilities=np.exp(logits-logits.max(axis=1,keepdims=True))
        probabilities=probabilities[:,1]/probabilities.sum(axis=1)
        output=[]
        for detection,score in zip(detections,probabilities):
            if score>=.85:
                detection.confidence=float(score)
                output.append(detection)
        return output
