import pickle
import cv2
from os.path import join
from tqdm import tqdm
from pycocotools.coco import COCO
from det import imread,imwrite
import matplotlib.pyplot as plt



# 可以使用以下代码在原图上画出gt框。我们的不同物体具有不同的速度趋势图是用不同颜色画的。
# 若要同时画出gt框和预测框：分两步。
# 1、在用绿色框画出使用GT框，
# 2、执行streamyolo.sh中的前半部分，即评估部分。所有的预测结果会存一个pkl文件
# 3、执行streamyolo.sh中的后半部分，即检测部分，但是我们使用的不是原图，而是第一步的结果。这样，就可以同时画出GT和预测框。




# def vis_det(img, bboxes, labels, class_names,
#     masks=None, scores=None, score_th=0,
#     out_scale=1, out_file=None):
 
# data_path="./streamyolo/data/online_resuklt/l_s50/vis"      # path to image

data_path="/home/shuxue3/gjt/new/streamyolo-sc-obj/data/Argoverse-1.1/tracking"      # path to image

annot_path='/home/shuxue3/gjt/new/streamyolo-sc-obj/data/Argoverse-HD/annotations/val.json'
out_path='./streamyolo/data/online_resuklt/l_s50/vis_gt_color'

def gt_vis(data_path,annot_path,out_path):

    db=COCO(annot_path)

    seqs = db.dataset['sequences']
    seq_dirs = db.dataset['seq_dirs']

    # print(seqs,seq_dirs)
    class_names = [c['name'] for c in db.dataset['categories']]

    for sid, seq in enumerate(tqdm(seqs)):
        frame_list = [img for img in db.imgs.values() if img['sid'] == sid]
        # results = pickle.load(open(join(opts.result_dir, seq + '.pkl'), 'rb'))
        n=len(frame_list)

        for i in range(n):
            img=frame_list[i]
            # img_path = join(data_path, seq, img['name'])
            img_path = join(data_path, seq_dirs[sid], img['name'])
            I = imread(img_path)
            vis_path = join(out_path, seq, img['name'][:-3] + 'jpg')


            gt=db.loadAnns(db.getAnnIds(img['id']))
            bbox_color = [#(0, 255, 0)
                            (0,0,255),#蓝色     纯蓝          
                            (0, 128, 0),  # 绿色   纯绿
                            (255, 255, 0),  # 黄色  纯黄
                            (0, 0, 0), # 黑色    
                            (255, 0, 0),  # 红色    纯红       bus
                            (128,0,128),  # 紫色
                            (255,140,0 ),  #深橙色
                            (165,42,42)]  #棕色

            text_color = [    #(0, 255, 0)
                            (0,0,255),#蓝色     纯蓝          
                            (0, 128, 0),  # 绿色   纯绿
                            (255, 255, 0),  # 黄色  纯黄
                            (0, 0, 0), # 黑色    
                            (255, 0, 0),  # 红色    纯红       bus
                            (128,0,128),  # 紫色
                            (255,140,0 ),  #深橙色
                            (165,42,42)]  #棕色

            thickness = 3
            font_scale = 1
            for i in range(len(gt)):
                g=gt[i]
                label=g['category_id']
                bbox=list(map(round,g['bbox']))
                # print(bbox)
                lt = (bbox[0], bbox[1])
                rb = (bbox[2]+bbox[0], bbox[3]+bbox[1])
                cv2.rectangle(
                    I, lt, rb, bbox_color[label], thickness=thickness
                )
                if class_names is None:
                    label_text = f'class {label}'
                else:
                    label_text = class_names[label]

                cv2.putText(
                    I, label_text, (bbox[0], bbox[1] -3),
                    cv2.FONT_HERSHEY_COMPLEX, font_scale,
                    text_color[label], 2
                )
            imwrite(I,vis_path)

# db=COCO(annot_path)

# print(db.dataset.keys())
# print(len(db.anns))


# seqs = db.dataset['sequences']
# seq_dirs = db.dataset['seq_dirs']

# print(seqs,seq_dirs)
# print(db.getAnnIds(frame_list[0]['id']) )
# # for i in db.getAnnIds(frame_list[0]['id']):
# print(db.loadAnns(db.getAnnIds(frame_list[0]['id']))[0])




if __name__=='__main__':
    gt_vis(data_path,annot_path,out_path)
