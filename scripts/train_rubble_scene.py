"""Train rubble/non-rubble on focused rocks plus full-scene negatives."""
from pathlib import Path
import sys,json,random,hashlib
import torch,torchvision,numpy as np,cv2
from torch.utils.data import DataLoader
root=Path.cwd();sys.path.insert(0,str(root/'scripts'));from train_rubble_seg import RubbleData
random.seed(91);np.random.seed(91);torch.manual_seed(91);torch.set_num_threads(3);cv2.setNumThreads(1)
folds={'train':[],'val':[]}
for folder in ('mujoco_rubble3','mujoco_clutter'):
 d=root/'data'/folder;meta=json.loads((d/'meta.json').read_text())
 for key,field in [('train','train'),('val','test')]:
  rows=[(str(d/'images'/(name+'.png')),str(d/'masks'/(name+'.png'))) for name in meta[field]];folds[key]+=rows*(2 if key=='train' else 1)
d=root/'data/navigation-floor'
for row in json.loads((d/'meta.json').read_text())['frames']:folds[row['split']].append((str(d/'images'/(row['name']+'.png')),str(d/'masks'/(row['name']+'.png'))))
out=root/'output/rubble-scene-training';out.mkdir(exist_ok=True);model=torchvision.models.segmentation.lraspp_mobilenet_v3_large(weights=None,weights_backbone=None,num_classes=3)
init=root/'output/navigation-floor-training/navigation_floor.pt';model.load_state_dict(torch.load(init,map_location='cpu',weights_only=True));device='cuda' if torch.cuda.is_available() else 'cpu';model.to(device);opt=torch.optim.AdamW(model.parameters(),lr=1e-4);lossfn=torch.nn.CrossEntropyLoss(weight=torch.tensor([1.,2.],device=device));losses=[]
loader=DataLoader(RubbleData(folds['train']),batch_size=4,shuffle=True)
print('training',device,{k:len(v) for k,v in folds.items()},flush=True)
for epoch in range(8):
 model.train();model.backbone.eval();values=[]
 for x,y in loader:
  opt.zero_grad();out_logits=model(x.to(device))['out'];binary=torch.stack([torch.logsumexp(out_logits[:,[0,2]],dim=1),out_logits[:,1]],dim=1);loss=lossfn(binary,(y.to(device)==1).long());loss.backward();opt.step();values.append(loss.item())
 losses.append(float(np.mean(values)));print('epoch',epoch+1,losses[-1],flush=True)
model.eval();tp=fp=fn=0
with torch.no_grad():
 for x,y in DataLoader(RubbleData(folds['val'],train=False),batch_size=4):
  p=model(x.to(device))['out'].softmax(1)[:,1].cpu()>=.7;gt=y==1;tp+=int((p&gt).sum());fp+=int((p&~gt).sum());fn+=int((~p&gt).sum())
model.cpu();torch.save(model.state_dict(),out/'rubble_scene.pt');torch.onnx.export(model,torch.zeros(1,3,320,480),str(out/'rubble_scene.onnx'),opset_version=17,input_names=['image'],output_names=['logits'],dynamo=False)
result={'seed':91,'epochs':8,'lr':1e-4,'initial_weights_sha256':hashlib.sha256(init.read_bytes()).hexdigest(),'binary_training_3class_interface':True,'losses':losses,'val_precision':tp/max(1,tp+fp),'val_recall':tp/max(1,tp+fn),'folds':folds,'sha256':hashlib.sha256((out/'rubble_scene.onnx').read_bytes()).hexdigest()};(out/'training.json').write_text(json.dumps(result,indent=2));print({k:v for k,v in result.items() if k!='folds'},flush=True)
