"""Train a small RGB proposal classifier from labelled simulation recordings."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from go2_rescue.recording import replay


def patch(image,cx,cy,r):
    side=max(8,int(np.ceil(2.7*r)))
    tile=cv2.getRectSubPix(image,(side,side),(float(cx),float(cy)))
    return cv2.resize(tile,(40,40))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output",type=Path,required=True); args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    run=root/"output/perception-a-verified"
    rng=np.random.default_rng(47)
    inputs=[]; labels=[]; sources=[]
    for f in replay(run/"recording"):
        if f.image is None: continue
        mask=cv2.imread(str(run/"evaluation_masks"/f"{f.timestamp:.6f}.png"),0)&2!=0
        ys,xs=np.nonzero(mask)
        if len(xs)>=100:
            cx=(xs.min()+xs.max())/2;cy=(ys.min()+ys.max())/2
            radius=max(np.ptp(xs),np.ptp(ys))/2
            if 10<=radius<=55:
                for j in range(44):
                    dx,dy=rng.normal(0,.08*radius,2)
                    p=patch(f.image,cx+dx,cy+dy,radius*rng.uniform(.82,1.18))
                    # Camera-distance loss of detail, retained in training.
                    edge=int(rng.integers(10,35))
                    p=cv2.resize(cv2.resize(p,(edge,edge)),(40,40))
                    p=np.clip(p.astype(float)*rng.uniform(.65,1.25)+rng.normal(0,3,p.shape),0,255).astype(np.uint8)
                    if rng.random()<.5:p=p[:,::-1].copy()
                    inputs.append(p);labels.append(1)
                    sources.append(f"rgb/{f.timestamp:.6f}/positive/{j}")
        gray=cv2.cvtColor(f.image,cv2.COLOR_BGR2GRAY)
        circles=cv2.HoughCircles(cv2.GaussianBlur(gray,(5,5),1),cv2.HOUGH_GRADIENT,1,10,
                                 param1=100,param2=10,minRadius=5,maxRadius=90)
        proposals=list(circles[0]) if circles is not None else []
        proposals.extend((rng.uniform(5,315),rng.uniform(10,230),rng.uniform(5,50)) for _ in range(50))
        for j,(x,y,r) in enumerate(proposals):
            y0,x0=np.ogrid[:240,:320]; disk=(x0-x)**2+(y0-y)**2<=r*r
            if (disk&mask).sum()/max(1,disk.sum())>.08:continue
            inputs.append(patch(f.image,x,y,r));labels.append(0)
            sources.append(f"rgb/{f.timestamp:.6f}/negative/{j}")
    negative=root/"output/terrain-wall-slope-online-rules"
    for image_path in sorted(negative.glob("*-raw.png")):
        image=cv2.imread(str(image_path))
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        circles=cv2.HoughCircles(cv2.GaussianBlur(gray,(5,5),1),cv2.HOUGH_GRADIENT,1,8,
                                 param1=100,param2=8,minRadius=5,maxRadius=90)
        proposals=list(circles[0]) if circles is not None else []
        proposals.extend((rng.uniform(5,315),rng.uniform(35,235),rng.uniform(5,60)) for _ in range(250))
        # These archived challenge scenes contain no football. Rocks are
        # essential hard negatives, including their circular silhouettes.
        for j,(x,y,r) in enumerate(proposals):
            for augmentation in range(3):
                crop=patch(image,x,y,r*rng.uniform(.9,1.1))
                edge=int(rng.integers(10,35))
                crop=cv2.resize(cv2.resize(crop,(edge,edge)),(40,40))
                if rng.random()<.5:crop=crop[:,::-1].copy()
                inputs.append(crop);labels.append(0)
                sources.append(f"negative/{image_path.name}/{j}/{augmentation}")
    extra=root/"data/ball-views"
    if (extra/"manifest.json").exists():
        for row in json.loads((extra/"manifest.json").read_text())["views"]:
            image=cv2.imread(str(extra/row["image"]))
            mask=cv2.imread(str(extra/row["mask"]),0)>0
            ys,xs=np.nonzero(mask)
            cx=(xs.min()+xs.max())/2;cy=(ys.min()+ys.max())/2
            radius=max(np.ptp(xs),np.ptp(ys))/2
            for repeat in range(40):
                dx,dy=rng.normal(0,.07*radius,2)
                crop=patch(image,cx+dx,cy+dy,radius*rng.uniform(.86,1.14))
                if rng.random()<.5:crop=crop[:,::-1].copy()
                inputs.append(crop);labels.append(1)
                sources.append(f"ball-views/{row['image']}/{repeat}")
    x=np.stack(inputs).astype(np.float32)/255
    x=x[:,:,:,::-1].transpose(0,3,1,2).copy()
    y=np.asarray(labels,dtype=np.int64)
    torch.manual_seed(47);torch.set_num_threads(4)
    device="cuda" if torch.cuda.is_available() else "cpu"
    model=nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                        nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                        nn.Conv2d(32,48,3,padding=1),nn.ReLU(),nn.AvgPool2d(4,3),
                        nn.Flatten(),nn.Linear(48*9,32),nn.ReLU(),nn.Linear(32,2))
    model.to(device)
    loader=DataLoader(TensorDataset(torch.from_numpy(x),torch.from_numpy(y)),batch_size=64,shuffle=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    weights=torch.tensor([1.,float((y==0).sum()/max(1,(y==1).sum()))],device=device)
    criterion=nn.CrossEntropyLoss(weight=weights)
    print("samples",len(y),"positive",int(y.sum()),"device",device,flush=True)
    losses=[]
    for epoch in range(10):
        total=0.
        for a,b in loader:
            optimizer.zero_grad();loss=criterion(model(a.to(device)),b.to(device));loss.backward();optimizer.step()
            total+=float(loss.detach())
        losses.append(total/len(loader))
        print("epoch",epoch+1,"loss",losses[-1],flush=True)
    model.cpu().eval()
    (args.output/"training_receipt.json").write_text(json.dumps({
        "seed":47,"samples":len(y),"positive":int(y.sum()),"source":sources,"losses":losses,
        "split":"A verified and wall-slope for training; B survey held out"
    },indent=2))
    path=args.output/"ball_classifier.onnx"
    torch.save(model.state_dict(),args.output/"ball_classifier.pt")
    torch.onnx.export(model,torch.zeros(1,3,40,40),str(path),opset_version=17,
                      input_names=["rgb"],output_names=["logits"],dynamic_axes={"rgb":{0:"proposals"},"logits":{0:"proposals"}},
                      dynamo=False)
    (args.output/"training.json").write_text(json.dumps({
        "seed":47,"epochs":10,"samples":len(y),"positive":int(y.sum()),"negative":int((y==0).sum()),
        "source":sources,"loss":losses,"onnx_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
        "training_runs":["perception-a-verified","terrain-wall-slope-online-rules"],
        "validation_run":"perception-b-survey-final; no frames used for training",
        "scope":"black-white sphere proxy proposal classification",
    },indent=2))


if __name__=="__main__":main()
