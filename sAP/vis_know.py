import os
import sys; sys.path.insert(0, '..'); sys.path.insert(0, '.')
import torch
print(sys.path)
from util import mkdir2, print_stats
from torchvision.ops import batched_nms
import cv2
from yolox.exp import get_exp
from yolox.utils import fuse_model
import time
import torch.nn.functional as F   
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def preproc(img, input_size, swap=(2, 0, 1)):
    resized_img = cv2.resize(img, (input_size[1], input_size[0]), interpolation=cv2.INTER_LINEAR,)
    resized_img = resized_img.transpose(swap)
    return resized_img

# b、归一化，画热力图
def vis_heat(feature,save_path='./h.jpg',if_input=True,mode='max'):     # 输入给定一个特征图，逐通道取最大，输出该特征图的可视化

    feature=F.interpolate(feature, size=(600,960), mode='bilinear')
    print(feature.shape,'+++++++++++')

    save_dir=('/').join(save_path.split('/')[:-1])
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    if if_input:
        # print(feature.shape,'ssssssssss')
        heatmap,_=feature.max(1,keepdim=True)#[0][0]   # (38,60)
        heatmap=heatmap[0][0]
        # print(heatmap.shape)
    else:
        heatmap,_=feature.max(1,keepdim=True)#[0][0]   # (38,60)
        heatmap=heatmap[0][0]

    heatmap = (heatmap-heatmap.min())/(heatmap.max()-heatmap.min())
    heatmap=heatmap.cpu().detach().numpy()

    heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    cv2.imwrite(save_path, heatmap)

class SaveOutput:                             # 钩子hook需要的，来存储拦截的中间特征图
    def __init__(self):
        self.outputs = []
        self.inputs=[]
    def __call__(self, module, module_in, module_out):
        # print(module_in.shape,'+++')
        self.outputs.append(module_out)
        self.inputs.append(module_in)
    def clear(self):
        self.outputs=[]



# 输入需要做可视化的图，  hook钩取特征图， 使用vis_heat 画出该特征的可视化  ，即输出 可视化图
def vis_layers_heatmap(layers, save_dir, layers_name,frame, is_spatial_implicit_knowledge=False):   
    save_output = SaveOutput()
    hook_handles=[]
    # which_layer = [model.backbone.imp_style0,]              # 选择那一层进行可视化

    for i in range(len(layers)):
        layer=layers[i]
        handle=layer.register_forward_hook(save_output)
        hook_handles.append(handle)

    result= model(frame, mode='off_pipe')
    # print((hook_handles[0]),'**********')
    # print(save_output.outputs[0].shape,'**********')

    for i in range(len(layers_name)):
        layer_name=layers_name[i]
        # save_path=save_path+ '/imp_style0'
        # save_path=save_path+ '/spatial_imp2'
        save_path=os.path.join(save_dir , layer_name)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        if not is_spatial_implicit_knowledge:
            feature_in=save_output.inputs[i][0]
            feature_in=F.interpolate(feature_in, size=(600,960), mode='bilinear')
            save1=os.path.join(save_path , 'input_mean.jpg')
            vis_heat(feature_in,save_path=save1,if_input=True,mode='max')

        feature_out=save_output.outputs[i]
        # feature_out=F.interpolate(feature_out, size=(75,120), mode='bilinear')
        feature_out=F.interpolate(feature_out, size=(600,960), mode='bilinear')
        save2=os.path.join(save_path, 'output_mean.jpg')
        vis_heat(feature_out,save_path=save2,if_input=True,mode='max')




config="/home/shuxue3/gjt/new/streamyolo-sc-obj/cfgs/l_s50_onex_dfp_tal_flip.py" # path/to/your/cfg
weights="/home/shuxue3/gjt/new/streamyolo-sc-obj/3776.pth"  # path/to/your/checkpoint_path

#  加载模型 权重   
exp = get_exp(config, None)
model = exp.get_model()
model.cuda()
model.eval()
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

# 下雨的
# f0='data/Argoverse-1.1/tracking/train/02cf0ce1-699a-373b-86c0-eb6fd5f4697a/ring_front_center/ring_front_center_315968494047328248.jpg'
# f1='data/Argoverse-1.1/tracking/train/02cf0ce1-699a-373b-86c0-eb6fd5f4697a/ring_front_center/ring_front_center_315968494080628232.jpg'

#夜晚
# f0='data/Argoverse-1.1/tracking/train/5c251c22-11b2-3278-835c-0cf3cdee3f44/ring_front_center/ring_front_center_315967812908841592.jpg'
# f1='data/Argoverse-1.1/tracking/train/5c251c22-11b2-3278-835c-0cf3cdee3f44/ring_front_center/ring_front_center_315967812942141784.jpg'

# 曝光
f0='data/Argoverse-1.1/tracking/train/043aeba7-14e5-3cde-8a5c-639389b6d3a6/ring_front_center/ring_front_center_315967467068167856.jpg' # 当前帧
f1='data/Argoverse-1.1/tracking/train/043aeba7-14e5-3cde-8a5c-639389b6d3a6/ring_front_center/ring_front_center_315967467034868064.jpg' # 支持帧

name=f0.split('/')[-1].split('.')[0]
save_path=f'./know_vis/{name}'


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
# print(torch.split(frame, 3, dim=1)[1].shape)
# print(frame.shape)
# 选择那一层进行可视化
layers = [model.backbone.imp_style0,model.backbone.imp_style1,model.backbone.imp_style2,
model.backbone.spatial_imp_2,model.backbone.spatial_imp_1,model.backbone.spatial_imp_0]
# model.backbone.spatial_imp_2.imp_mul]

layers_name=['imp_style0','imp_style1','imp_style2',
'spatial_imp2','spatial_imp1','spatial_imp0']
# '/spatial_imp2_mul']


#画出 使用 通道隐性知识模块前后的FPN特征图       使用空间隐性知识模块前后特征图的变化
vis_layers_heatmap(layers, save_path, layers_name,frame)
print('FPN before and after using spatial implicit knowledge.  the vis is done!')

# 画空间隐性知识      mul部分  和 add部分
layers = [model.backbone.spatial_imp_2.imp_mul,model.backbone.spatial_imp_2.imp_add,
model.backbone.spatial_imp_1.imp_mul,model.backbone.spatial_imp_1.imp_add,
model.backbone.spatial_imp_0.imp_mul,model.backbone.spatial_imp_0.imp_add]
layers_name=['spatial_imp2_mul','spatial_imp2_add',
'spatial_imp1_mul','spatial_imp1_add',
'spatial_imp0_mul','spatial_imp0_add']
vis_layers_heatmap(layers, save_path, layers_name, frame,is_spatial_implicit_knowledge=True)


print('adaptive spatial implicit knowledge vis is done!')
# 至此完成空间隐性知识可视化，下面的代码不需要执行，是写上面代码的一些练习与尝试，不需要看。 

# '------------------------------------------------------------------------------------'
# c='_mul'   
# c='_add'

# 网络中空间隐性知识的大小shape    19,30   38,60   38,60         分别是对应的三个层级
# 网络中通道隐性知识的通道数C       128     256     512


# 20通道的隐性先验的可视化
s_add_0 = model.backbone.spatial_imp_0.imp_add.imp.unsqueeze(3).permute(0,2,1,3).view(1,20,19,30)
s_add_1 = model.backbone.spatial_imp_1.imp_add.imp.unsqueeze(3).permute(0,2,1,3).view(1,20,38,60)    #view(-1,1,38,60)
s_add_2 = model.backbone.spatial_imp_2.imp_add.imp.unsqueeze(3).permute(0,2,1,3).view(1,20,38,60)
s_mul_0 = model.backbone.spatial_imp_0.imp_mul.imp.unsqueeze(3).permute(0,2,1,3).view(1,20,19,30)
s_mul_1 = model.backbone.spatial_imp_1.imp_mul.imp.unsqueeze(3).permute(0,2,1,3).view(1,20,38,60)
s_mul_2 = model.backbone.spatial_imp_2.imp_mul.imp.unsqueeze(3).permute(0,2,1,3).view(1,20,38,60)

print(s_add_0.shape)
vis_heat(s_add_0,save_path='./know_vis/spatial_knowledge_priori/spatial_imp_0_add.jpg',if_input=True,mode='max')
vis_heat(s_add_1,save_path='./know_vis/spatial_knowledge_priori/spatial_imp_1_add.jpg',if_input=True,mode='max')
vis_heat(s_add_2,save_path='./know_vis/spatial_knowledge_priori/spatial_imp_2_add.jpg',if_input=True,mode='max')
vis_heat(s_mul_0,save_path='./know_vis/spatial_knowledge_priori/spatial_imp_0_mul.jpg',if_input=True,mode='max')
vis_heat(s_mul_1,save_path='./know_vis/spatial_knowledge_priori/spatial_imp_1_mul.jpg',if_input=True,mode='max')
vis_heat(s_mul_2,save_path='./know_vis/spatial_knowledge_priori/spatial_imp_2_mul.jpg',if_input=True,mode='max')
# 隐性先验的可视化完成，这里没有根据输入的不同而变化，是可学习空间先验的可视化。

assert 1==0   # 不需要让后面代码运行。




# 下面代码可删除，不运行。
feature_out=save_output.outputs[i]
# feature_out=F.interpolate(feature_out, size=(75,120), mode='bilinear')
feature_out=F.interpolate(feature_out, size=(600,960), mode='bilinear')
save2=os.path.join(save_path, 'output_mean.jpg')
vis_heat(feature_out,save_path=save2,if_input=True,mode='mean')


# 上面可视化，是取通道中的最大做的，这里我们提供了其他一些可视化的方法，而不是取最大。最后还是选择了取最大的b方案，视觉效果最好。
# a方案、画隐性知识向量 热图  使用库函数
c='_add'
n=model.backbone.imp_style0.imp_add.imp
m= model.backbone.spatial_imp_1.imp_add.imp.unsqueeze(3).view(-1,1,38,60) 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# f, ax = plt.subplots(figsize=(8,8))
i=10
sns.set()
np.random.seed(0)   
# uniform_data = np.array(n[0].view(-1,256).cpu().detach().numpy()) 
# c+='_c'    
uniform_data = np.array(m[i][0].cpu().detach().numpy())
c+=f'_s{i}'
ax = sns.heatmap(uniform_data)
plt.show()

plt.savefig( f'./heatmap{c}.jpg')


# b方案、归一化，画热力图
c='_mul'
i=10
m= model.backbone.spatial_imp_1.imp_mul.imp.unsqueeze(3).view(-1,1,38,60) 
n=model.backbone.imp_style0.imp_mul.imp
print(m.shape)

heatmap=m.mean(0)[0]   # (38,60)
c+='_mean'
# heatmap=m[i][0]   # (38,60)
# c+=f'_s{i}'

# heatmap=n[0].transpose(1,0)
# c+=f'_c'

heatmap = (heatmap-heatmap.min())/(heatmap.max()-heatmap.min())
heatmap=heatmap.cpu().detach().numpy()

heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
# superimposed_img = heatmap
# plt.title(name[i%len(name)])
# plt.imshow(superimposed_img,cmap='gray')
# plt.show()
cv2.imwrite(f'./NORM_heatmap{c}.jpg', heatmap)


# c方案、归一化，只取大于0的值 来画热力图
c='_add'
i=5
m= model.backbone.spatial_imp_1.imp_add.imp.unsqueeze(3).view(-1,1,38,60) 
n=model.backbone.imp_style0.imp_add.imp
print(m.shape)

# heatmap=m.mean(0)[0]   # (38,60)
# c+='mean'
heatmap=m[i][0]   # (38,60)
c+=f'_s{i}'

# heatmap=n[0].transpose(1,0)
# c+=f'_c'

heatmap=heatmap.cpu().detach().numpy()
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
# print(heatmap.min())

heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
# superimposed_img = heatmap
# plt.title(name[i%len(name)])
# plt.imshow(superimposed_img,cmap='gray')
# plt.show()
cv2.imwrite(f'./NORM_RELU_heatmap{c}.jpg', heatmap)



m= model.backbone.spatial_imp_1.imp_mul.imp.unsqueeze(3).view(-1,1,38,60) 
n=model.backbone.imp_style0.imp_mul.imp






