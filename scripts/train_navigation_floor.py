import json,random,hashlib
from pathlib import Path
import cv2,numpy as np,torch,torchvision
from torch.utils.data import Dataset,DataLoader
root=Path.cwd();data=root/'data/navigation-floor';out=root/'output/navigation-floor-training';out.mkdir(exist_ok=True)
random.seed(71);np.random.seed(71);torch.manual_seed(71);torch.set_num_threads(3);cv2.setNumThreads(1)
rows=json.loads((data/'meta.json').read_text())['frames']
class Views(Dataset):
 def __init__(self,split):self.rows=[r for r in rows if r['split']==split];self.train=split=='train'
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  name=self.rows[i]['name'];im=cv2.imread(str(data/'images'/f'{name}.png'))[:,:,::-1].copy();mask=cv2.imread(str(data/'masks'/f'{name}.png'),0)
  if self.train:
   if random.random()<.5:im=im[:,::-1].copy();mask=mask[:,::-1].copy()
   im=np.uint8(np.clip(im.astype(float)*random.uniform(.65,1.35)+random.uniform(-12,12),0,255))
  im=cv2.resize(im,(320,240)).astype(np.float32)/255;im=(im-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
  return torch.from_numpy(im.transpose(2,0,1).copy()),torch.from_numpy(mask.astype(np.int64))
m=torchvision.models.segmentation.lraspp_mobilenet_v3_large(weights=None,weights_backbone=None,num_classes=3)
m.load_state_dict(torch.load(root/'models/seg_low_obstacle_v10.pt',map_location='cpu',weights_only=True));device='cuda' if torch.cuda.is_available() else 'cpu';m.to(device)
opt=torch.optim.AdamW(m.parameters(),lr=2e-4);lossfn=torch.nn.CrossEntropyLoss(weight=torch.tensor([1.25,2.,1.],device=device))
loader=DataLoader(Views('train'),batch_size=6,shuffle=True);losses=[]
print('device',device,'train',len(Views('train')),'val',len(Views('val')),flush=True)
for epoch in range(10):
 m.train();m.backbone.eval();values=[]
 for x,y in loader:
  opt.zero_grad();loss=lossfn(m(x.to(device))['out'],y.to(device));loss.backward();opt.step();values.append(loss.item())
 losses.append(float(np.mean(values)));print('epoch',epoch+1,losses[-1],flush=True)
m.eval();tp=fp=fn=0
with torch.no_grad():
 for x,y in DataLoader(Views('val'),batch_size=4):
  p=m(x.to(device))['out'].softmax(1)[:,0].cpu()>.85;gt=y==0;tp+=int((p&gt).sum());fp+=int((p&~gt).sum());fn+=int((~p&gt).sum())
m.cpu();torch.save(m.state_dict(),out/'navigation_floor.pt');torch.onnx.export(m,torch.zeros(1,3,240,320),str(out/'navigation_floor.onnx'),input_names=['image'],output_names=['logits'],opset_version=17,dynamo=False)
result={'seed':71,'epochs':10,'losses':losses,'ground_p85':{'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn)},'location_group_holdout':True,'dataset':str(data),'model_sha256':hashlib.sha256((out/'navigation_floor.onnx').read_bytes()).hexdigest()};(out/'training.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
