'''
this version for using only original image without mask in pickle form.
'''
import cv2
import numpy as np
import os
import pickle
import pandas as pd

import sys
sys.path.append('./')
import infer


if __name__ == "__main__":
    import argparse

    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='在这里设计程序的输入参数')
    # 添加参数
    parser.add_argument('--root', help='处理参数的路径')
    parser.add_argument('--output', help='输出文件的路径', default=None)
    parser.add_argument('--model', help='模型路径', default='models/9800.pth')
    parser.add_argument('--save_vis', help='是否保存可视化图像', default=False)
    args = parser.parse_args()
    img_root =  args.root
    
    if False:
        import torch
        from torchvision import transforms
        from model import *

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('device:', device)

        # 加载模型，映射到指定设备
        print(args.model)
        netm = torch.load(args.model, map_location=device)  # the model output is 12 dim vector?
        # 匹配的输入转换
        transform = transforms.Compose([
                transforms.ToTensor(),          # 将图像转换为PyTorch张量
                # transforms.Resize((768, 768)),
                transforms.Normalize(           # 归一化图像
                    mean=[0.485, 0.456, 0.406],  # ImageNet数据集的均值
                    std=[0.229, 0.224, 0.225]   # ImageNet数据集的标准差
                ),
        ])


    if args.output is None:
        args.output = args.root

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    imgs = os.listdir(img_root)
    name_start = imgs[0].split('_')[0]
    im_indexes = [int(item.split('.')[0].split('_')[-1]) for item in imgs]

    imgs = [x for _, x in sorted(zip(im_indexes, imgs))]

    im_indexes.sort()

    if im_indexes[0] == 0:
        save_names = [name_start + '_' + str(index + 1) + '.xlsx' for index in im_indexes]
    else:
        save_names = [name_start + '_' + str(index) + '.xlsx' for index in im_indexes]
        
    obj_dict = {}

    
    # 
    print(imgs)
    print(save_names)
    for img_name, save_name in zip(imgs, save_names):
        if 'png' not in img_name and 'jpg' not in img_name:
            continue
        ori_img = cv2.imread(os.path.join(img_root, img_name))
        
        pk_name = os.path.join(img_root, img_name.replace('png', 'pk').replace('jpg', 'pk'))
        if os.path.isfile(pk_name):
            pk_item = pickle.load(open(pk_name, 'rb'))
            objs = []
            for pk_key in pk_item:
                objs.extend(infer.get_obj(ori_img, pk_key, pk_item[pk_key], True))

            for obj_id, obj in enumerate(objs):
                obj_dict[obj['cls_id']] = obj
            res_obj_dict = obj_dict

            if args.save_vis:
                for index in res_obj_dict:
                    obj = res_obj_dict[index]
                    box = obj['最小外接矩形']

                
                    cv2.line(ori_img, box[0], box[1], (255, 0, 0), 1)
                    cv2.line(ori_img, box[1], box[2], (255, 0, 0), 1)
                    cv2.line(ori_img, box[2], box[3], (255, 0, 0), 1)
                    cv2.line(ori_img, box[3], box[0], (255, 0, 0), 1)

                    cv2.putText(ori_img, str(index), box[0], cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 255, 0), 2, cv2.LINE_AA)
                
                cv2.imwrite(os.path.join(os.path.join(args.output, img_name + '_vis.png')), ori_img)
            
            infer.save_to_xslx(res_obj_dict, os.path.join(os.path.join(args.output, save_name)))
            continue # if pk file found, no further action (including CV model processing)


        img = transform(ori_img).to(device)
        model_output = netm(img.unsqueeze(0))[0].argmax(0).cpu().numpy()

        objs = []
        for cls_id in range(14):
            objs.extend(infer.get_obj(ori_img, cls_id, model_output))
  
        #初始化序号
        if len(obj_dict) == 0:
            for obj_id, obj in enumerate(objs):
                obj_dict[obj_id] = obj
            res_obj_dict = obj_dict
        else: #已经存在序号，需进行匹配
            res_obj_dict = {}
            match_dises = {}
            new_objs= []
            for obj in objs:
                dises = []
                indexes = []
                for obj_index in obj_dict:
                    obj_item = obj_dict[obj_index]
                    if obj['cls_id'] != obj_item['cls_id']:
                        continue
                    dis = infer.get_dis(obj['最小外接矩形'], obj_item['最小外接矩形'])
                    dises.append(dis)
                    indexes.append(obj_index)
                if min(dises) <= 100:
                    selected_index = indexes[dises.index(min(dises))]
                    if selected_index not in res_obj_dict:
                        res_obj_dict[selected_index] = obj
                        match_dises[selected_index] = min(dises)
                    else:
                        if match_dises[selected_index] < min(dises):
                            new_objs.append(obj)
                        else:
                            new_objs.append(res_obj_dict[selected_index])
                            res_obj_dict[selected_index] = obj
                else:
                    new_objs.append(obj)
            if len(new_objs) > 0:
                max_index = max(list(res_obj_dict.keys()))
                for obj in new_objs:
                    res_obj_dict[max_index + 1] = obj
                    max_index += 1
            obj_dict.update(res_obj_dict)

        # print(res_obj_dict)


        # res_obj_dict = obj_dict
        if args.save_vis:
            for index in res_obj_dict:
                obj = res_obj_dict[index]
                box = obj['最小外接矩形']

                cv2.line(ori_img, box[0], box[1], (255, 0, 0), 1)
                cv2.line(ori_img, box[1], box[2], (255, 0, 0), 1)
                cv2.line(ori_img, box[2], box[3], (255, 0, 0), 1)
                cv2.line(ori_img, box[3], box[0], (255, 0, 0), 1)

                cv2.putText(ori_img, str(index), box[0], cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 255, 0), 2, cv2.LINE_AA)
            


        
            cv2.imwrite(os.path.join(os.path.join(args.output, img_name + '_vis.png')), ori_img)
        
        infer.save_to_xslx(res_obj_dict, os.path.join(os.path.join(args.output, save_name)))

    # 