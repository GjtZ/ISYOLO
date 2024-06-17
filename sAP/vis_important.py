import os
import sys; 
sys.path.insert(0, '..'); sys.path.insert(0, '.')
import torch
# print(sys.path)
# from util import mkdir2, print_stats
# from torchvision.ops import batched_nms
import cv2
from yolox.exp import get_exp
from yolox.utils import fuse_model
# import time
import torch.nn.functional as F   
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from tqdm import tqdm

config="/home/shuxue3/gjt/new/streamyolo-sc-obj/cfgs/l_s50_onex_dfp_tal_flip.py" # path/to/your/cfg
weights="/home/shuxue3/gjt/new/streamyolo-sc-obj/3776.pth"  # path/to/your/checkpoint_path
data_path="/home/shuxue3/gjt/new/streamyolo-sc-obj/data/Argoverse-1.1/tracking"      # path to image
# annot_path='/home/shuxue3/gjt/new/streamyolo-sc-obj/data/Argoverse-HD/annotations/val.json'
annot_path='/home/shuxue3/gjt/new/streamyolo-sc-obj/data/Argoverse-HD/annotations/train.json'


def preproc(img, input_size, swap=(2, 0, 1)):
    resized_img = cv2.resize(img, (input_size[1], input_size[0]), interpolation=cv2.INTER_LINEAR,)[:,:,::-1].copy()
    resized_img = resized_img.transpose(swap)
    return resized_img


#  加载模型 权重   
exp = get_exp(config, None)
model = exp.get_model()
model.cuda()
model.train()
ckpt = torch.load(weights, map_location="cpu")
model.load_state_dict(ckpt["model"])

model.half()
tensor_type = torch.cuda.HalfTensor
# shape    19,30   38,60   38,60         
# shape    128     256     512

# 隐性知识的输入 输出 ， 作用过程    hook
dataDir="/home/shuxue3/gjt/new/streamyolo-sc-obj/"  # path/to/your/data  根目录

# f00='data/Argoverse-1.1/tracking/val/7d37fc6b-1028-3f6f-b980-adb5fa73021e/ring_front_center/ring_front_center_315968385009771456.jpg'
# f0='data/Argoverse-1.1/tracking/val/7d37fc6b-1028-3f6f-b980-adb5fa73021e/ring_front_center/ring_front_center_315968385109671216.jpg'
# f1='data/Argoverse-1.1/tracking/val/7d37fc6b-1028-3f6f-b980-adb5fa73021e/ring_front_center/ring_front_center_315968385142971080.jpg'   # 可视化的图像的路径

# f0='data/Argoverse-1.1/tracking/val/33737504-3373-3373-3373-633738571776/ring_front_center/ring_front_center_315968437124271032.jpg'
# f1='data/Argoverse-1.1/tracking/val/33737504-3373-3373-3373-633738571776/ring_front_center/ring_front_center_315968437157571088.jpg'

# f0='data/Argoverse-1.1/tracking/val/1d676737-4110-3f7e-bec0-0c90f74c248f/ring_front_center/ring_front_center_315984808332486056.jpg'
# f1='data/Argoverse-1.1/tracking/val/1d676737-4110-3f7e-bec0-0c90f74c248f/ring_front_center/ring_front_center_315984808365786072.jpg'

# f0='data/Argoverse-1.1/tracking/val/5ab2697b-6e3e-3454-a36a-aba2c6f27818/ring_front_center/ring_front_center_315972990339178000.jpg'
# f1='data/Argoverse-1.1/tracking/val/5ab2697b-6e3e-3454-a36a-aba2c6f27818/ring_front_center/ring_front_center_315972990372478128.jpg'


#夜晚
# f0='data/Argoverse-1.1/tracking/train/5c251c22-11b2-3278-835c-0cf3cdee3f44/ring_front_center/ring_front_center_315967812908841592.jpg'
# f1='data/Argoverse-1.1/tracking/train/5c251c22-11b2-3278-835c-0cf3cdee3f44/ring_front_center/ring_front_center_315967812942141784.jpg'

# 曝光   # f0：当前帧   f1：支持帧   y:下一帧
# f0='data/Argoverse-1.1/tracking/train/043aeba7-14e5-3cde-8a5c-639389b6d3a6/ring_front_center/ring_front_center_315967467068167856.jpg'
# f1='data/Argoverse-1.1/tracking/train/043aeba7-14e5-3cde-8a5c-639389b6d3a6/ring_front_center/ring_front_center_315967467034868064.jpg'
# y ='data/Argoverse-1.1/tracking/train/043aeba7-14e5-3cde-8a5c-639389b6d3a6/ring_front_center/ring_front_center_315967467101467712.jpg'

# f0='data/Argoverse-1.1/tracking/val/7d37fc6b-1028-3f6f-b980-adb5fa73021e/ring_front_center/ring_front_center_315968385043071320.jpg'
# f1='data/Argoverse-1.1/tracking/val/7d37fc6b-1028-3f6f-b980-adb5fa73021e/ring_front_center/ring_front_center_315968385009771456.jpg'
# y='data/Argoverse-1.1/tracking/val/7d37fc6b-1028-3f6f-b980-adb5fa73021e/ring_front_center/ring_front_center_315968385076371272.jpg'

# F:\目标检测\Argoverse-HD\流感知数据集\Argoverse-1.1\tracking\val\f1008c18-e76e-3c24-adcc-da9858fac145\ring_front_center
# f0='data/Argoverse-1.1/tracking/val/f1008c18-e76e-3c24-adcc-da9858fac145/ring_front_center/ring_front_center_315973412617897824.jpg'
# f1='data/Argoverse-1.1/tracking/val/f1008c18-e76e-3c24-adcc-da9858fac145/ring_front_center/ring_front_center_315973412584597744.jpg'
# y='data/Argoverse-1.1/tracking/val/f1008c18-e76e-3c24-adcc-da9858fac145/ring_front_center/ring_front_center_315973412651197736.jpg'


# f0='data/Argoverse-1.1/tracking/val/da734d26-8229-383f-b685-8086e58d1e05/ring_front_center/ring_front_center_315967919692039472.jpg'
# f1='data/Argoverse-1.1/tracking/val/da734d26-8229-383f-b685-8086e58d1e05/ring_front_center/ring_front_center_315967919658739568.jpg'
# y='data/Argoverse-1.1/tracking/val/da734d26-8229-383f-b685-8086e58d1e05/ring_front_center/ring_front_center_315967919725339424.jpg'


# f0='data/Argoverse-1.1/tracking/val/cb762bb1-7ce1-3ba5-b53d-13c159b532c8/ring_front_center/ring_front_center_315967331713239720.jpg'
# f1='data/Argoverse-1.1/tracking/val/cb762bb1-7ce1-3ba5-b53d-13c159b532c8/ring_front_center/ring_front_center_315967331679939552.jpg'
# y='data/Argoverse-1.1/tracking/val/cb762bb1-7ce1-3ba5-b53d-13c159b532c8/ring_front_center/ring_front_center_315967331746539864.jpg'


# f0='data/Argoverse-1.1/tracking/val/cb0cba51-dfaf-34e9-a0c2-d931404c3dd8/ring_front_center/ring_front_center_315972700940697960.jpg'
# f1='data/Argoverse-1.1/tracking/val/cb0cba51-dfaf-34e9-a0c2-d931404c3dd8/ring_front_center/ring_front_center_315972700907398000.jpg'
# y='data/Argoverse-1.1/tracking/val/cb0cba51-dfaf-34e9-a0c2-d931404c3dd8/ring_front_center/ring_front_center_315972700973997920.jpg'

# F:\目标检测\Argoverse-HD\流感知数据集\Argoverse-1.1\tracking\val\c9d6ebeb-be15-3df8-b6f1-5575bea8e6b9\ring_front_center
# f0='data/Argoverse-1.1/tracking/val/c9d6ebeb-be15-3df8-b6f1-5575bea8e6b9/ring_front_center/ring_front_center_315973074977221128.jpg'
# f1='data/Argoverse-1.1/tracking/val/c9d6ebeb-be15-3df8-b6f1-5575bea8e6b9/ring_front_center/ring_front_center_315973074943921184.jpg'
# y='data/Argoverse-1.1/tracking/val/c9d6ebeb-be15-3df8-b6f1-5575bea8e6b9/ring_front_center/ring_front_center_315973075010521240.jpg'

# F:\目标检测\Argoverse-HD\流感知数据集\Argoverse-1.1\tracking\val\b1ca08f1-24b0-3c39-ba4e-d5a92868462c\ring_front_center
# f0='data/Argoverse-1.1/tracking/val/b1ca08f1-24b0-3c39-ba4e-d5a92868462c/ring_front_center/ring_front_center_315980040416005288.jpg'
# f1='data/Argoverse-1.1/tracking/val/b1ca08f1-24b0-3c39-ba4e-d5a92868462c/ring_front_center/ring_front_center_315980040382705144.jpg'
# y='data/Argoverse-1.1/tracking/val/b1ca08f1-24b0-3c39-ba4e-d5a92868462c/ring_front_center/ring_front_center_315980040449305560.jpg'

#夜晚
# F:\目标检测\Argoverse-HD\流感知数据集\Argoverse-1.1\tracking\val\\ring_front_center
# f0='data/Argoverse-1.1/tracking/val/033669d3-3d6b-3d3d-bd93-7985d86653ea/ring_front_center/ring_front_center_315968252735036520.jpg'
# f1='data/Argoverse-1.1/tracking/val/033669d3-3d6b-3d3d-bd93-7985d86653ea/ring_front_center/ring_front_center_315968252701736440.jpg'
# y='data/Argoverse-1.1/tracking/val/033669d3-3d6b-3d3d-bd93-7985d86653ea/ring_front_center/ring_front_center_315968252768336560.jpg'


# ring_front_center_315968255432336368.jpg
# f0='data/Argoverse-1.1/tracking/val/033669d3-3d6b-3d3d-bd93-7985d86653ea/ring_front_center/ring_front_center_315968255432336368.jpg'
# f1='data/Argoverse-1.1/tracking/val/033669d3-3d6b-3d3d-bd93-7985d86653ea/ring_front_center/ring_front_center_315968255399036192.jpg'
# y='data/Argoverse-1.1/tracking/val/033669d3-3d6b-3d3d-bd93-7985d86653ea/ring_front_center/ring_front_center_315968255465636464.jpg'

# 下雨的
f0='data/Argoverse-1.1/tracking/train/02cf0ce1-699a-373b-86c0-eb6fd5f4697a/ring_front_center/ring_front_center_315968494080628232.jpg'
f1='data/Argoverse-1.1/tracking/train/02cf0ce1-699a-373b-86c0-eb6fd5f4697a/ring_front_center/ring_front_center_315968494047328248.jpg'
y='data/Argoverse-1.1/tracking/train/02cf0ce1-699a-373b-86c0-eb6fd5f4697a/ring_front_center/ring_front_center_315968494113928216.jpg'

name=f0.split('/')[-1].split('.')[0]

# filename_list=[f0.split('/')[-1],f1.split('/')[-1]]

save_path=f'./vis_important_w/{name}'


if not os.path.isdir(save_path):
    os.makedirs(save_path)

img_path=[dataDir+f0,dataDir+f1]
frames=[cv2.imread(i) for i in img_path]
for i,frame in enumerate(frames):                 # 对图片做预处理
    h_img, w_img = int(1200 * 0.5), int(1920 * 0.5)
    frame = preproc(frame, input_size=(h_img, w_img))  # [3,600,960]
    frame = torch.from_numpy(frame).unsqueeze(0).type(tensor_type)    # [1,3,600,960]
    frames[i]=frame


frame=torch.cat(frames,dim=1)



# result= model(frame, mode='off_pipe')

# print(result.shape)    # [1,11850,13]                            #   75 120        38  60   19 30      


# 获得标签
db=COCO(annot_path)
seqs = db.dataset['sequences']
seq_dirs = db.dataset['seq_dirs']

# print(seqs,seq_dirs)
class_names = [c['name'] for c in db.dataset['categories']]

id=[]   # 找到下一帧的id
target=[]
for img in db.imgs.values() :   
    # for x in filename_list:    # 两帧图片


    if img['name'] == y.split('/')[-1]:
        I=cv2.imread(dataDir+f0)
        print(img['id'])
        gt=db.loadAnns(db.getAnnIds(img['id']))    
        print(len(gt))    
        for i in range(len(gt)):
            
            g=gt[i]
            label=g['category_id']
            bbox=list(map(round,g['bbox']))          # 左上角点的x,y    宽高
            # print(bbox)
            lt = (bbox[0], bbox[1])                           # 应该÷2
            rb = (bbox[2]+bbox[0], bbox[3]+bbox[1])

        
            cv2.rectangle(
                I, lt, rb, [0,255,0], 3
            )

            target.append([label,(bbox[0]+ bbox[2]//2)//2, (bbox[1]+bbox[3]//2)//2 ,bbox[2]//2,bbox[3]//2])
            # target.append([label,bbox[0],bbox[1] ,bbox[2],bbox[3]])
        cv2.imwrite(save_path+'/yuan.png',I)
target=np.array(target)
label=torch.from_numpy(target).unsqueeze(0).type(tensor_type) 
print(label.shape)

model.head.use_l1 = True
result= model(frame,targets=[label,label], mode='off_pipe')   # [1,11850,13]    #   75 120        38  60   19 30  




# hw=[75,120,38,60,19,30]
# result=[result[:,0:75*120,:].reshape(1,75,120),result[:,75*120:75*120+38*60,:],result[:,75*120+38*60:,:]]


