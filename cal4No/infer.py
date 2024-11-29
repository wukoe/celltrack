import os,glob
import cv2
import numpy as np
import pickle
import pandas as pd

import sys
sys.path.append('./')


def get_line_len(point1, point2):
    return pow(pow(point1[0] - point2[0], 2) + pow(point1[1] - point2[1], 2), 0.5)

def get_obj_mfea(obj_cls, model_out):
    # calculate the geometric features of mask.
    h, w = model_out.shape

    cls_mask = model_out.astype(np.float32)
    # 定义结构元素
    kernel = np.ones((3, 3), np.uint8)

    # 应用腐蚀操作
    eroded_image = cv2.erode(cls_mask, kernel, iterations=1).astype(np.uint8)
    
    contours, _ = cv2.findContours(eroded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 假设最大轮廓是我们感兴趣的轮廓（如果有多个轮廓）
    objs = []
    if contours:
        for contour in contours:
            # 计算最小外接矩形
            rect = cv2.minAreaRect(contour)
            # import pdb; pdb.set_trace()
            # 获取矩形的尺寸
            size = rect[1]
            # 长轴和短轴的长度
            long_axis = max(size)
            short_axis = min(size)
            if short_axis < 10 or long_axis > min(h, w) * 0.5:
                continue

            # 计算面积
            area = cv2.contourArea(contour)

            # 计算周长
            perimeter = cv2.arcLength(contour, True)

            #长短轴比
            axis_ratio = long_axis / short_axis
            
            # 计算圆度 (0~1)，1 表示完美圆形
            if perimeter == 0:
                circularity = 0
            else:
                circularity = (4 * np.pi * area) / (perimeter ** 2)
            
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            concavity = 1 - area / hull_area

            box = cv2.boxPoints(rect)
            box = box.astype(np.int32)  # 转换为整数坐标

            #获取四条边：
            lines = [(box[0], box[1]), (box[1], box[2]), (box[2], box[3]), (box[3], box[0])]
            lines_len = [get_line_len(line[0], line[1]) for line in lines]
            short_lines = []
            min_index = lines_len.index(min(lines_len))
            short_lines.append(lines[min_index])
            lines_len[min_index] = 1e10
            min_index = lines_len.index(min(lines_len))
            short_lines.append(lines[min_index])

            point1 = ((short_lines[0][0][0] + short_lines[0][1][0]) // 2 ,  (short_lines[0][0][1] + short_lines[0][1][1]) // 2)
            point2 = ((short_lines[1][0][0] + short_lines[1][1][0]) // 2 ,  (short_lines[1][0][1] + short_lines[1][1][1]) // 2)
            res = {}
            res['cls_id'] = obj_cls
            res['长轴长度'] = long_axis
            res['短轴长度'] = short_axis
            res['周长'] = perimeter
            res['面积'] = area
            res['长短轴比例'] = axis_ratio
            res['圆度'] = circularity
            res['凹度'] = concavity
            # res['长轴端点'] = (int(point1[0]), int(point1[1]), int(point2[0]), int(point2[1]))
            res['长轴端点x1'] = int(point1[0])
            res['长轴端点y1'] = int(point1[1])
            res['长轴端点x2'] = int(point2[0])
            res['长轴端点y2'] = int(point2[1])
            res['最小外接矩形'] = box
            objs.append(res)

    return objs

def get_center_point(box):
        # 假设box是形如[(x1, y1), (x2, y2)]的矩形框，其中(x1, y1)是左上角坐标，(x2, y2)是右下角坐标
        x_center = (box[0][0] + box[1][0]) / 2
        y_center = (box[0][1] + box[1][1]) / 2
        return (x_center, y_center)

def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def get_dis(box1, box2):
    # 获取两个矩形框的中心点
    box1 = box1.reshape(-1, 2)
    box2 = box2.reshape(-1, 2)
    center1 = get_center_point(box1)
    center2 = get_center_point(box2)

    # 计算中心点之间的距离
    distance = euclidean_distance(center1, center2)
    return distance

def save_to_xslx(res_obj_dict, save_name):
    data = []
    for key in res_obj_dict:
        item = res_obj_dict[key]
        data_item = [key, item['周长'], item['面积'], item['长轴长度'], item['短轴长度'], item['长短轴比例'], item['圆度'], item['凹度'], item['长轴端点x1'], item['长轴端点y1'], item['长轴端点x2'], item['长轴端点y2']]
        data.append(data_item)
    df = pd.DataFrame(data)
    df.columns = ['id', '周长', '面积', '长轴长度', '短轴长度', '长短轴比例', '圆度', '凹度', '长轴端点x1', '长轴端点y1', '长轴端点x2', '长轴端点y2']
    df.to_excel(save_name, index=False)


if __name__ == "__main__":
    import argparse

    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='在这里设计程序的输入参数')
    # 添加参数
    parser.add_argument('--root', help='处理参数的路径')
    parser.add_argument('--output', help='输出文件的路径', default=None)
    parser.add_argument('--save_vis', help='是否保存可视化图像', default=False)
    args = parser.parse_args()

    img_root =  args.root
    
    if args.output is None:
        args.output = args.root

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # get all .pk files.
    imgs = glob.glob(os.path.join(img_root, '*.pk'))
    # print(imgs[0])
    im_indexes = [int(item.rsplit('.', 1)[0].rsplit('_', 1)[-1]) for item in imgs]
    print(im_indexes)

    flag_NameStartFromZero = min(im_indexes)==0
    # print(flag_StartFromZero)
    
    for img_name in imgs:
        # print(img_name)
        pk_item = pickle.load(open(img_name, 'rb'))

        objs = []
        for obj_id in pk_item:
            objs.extend(get_obj_mfea(obj_id, pk_item[obj_id]))

        res_obj_dict = {}
        for obj in objs:
            res_obj_dict[obj['cls_id']] = obj

        # output file name
        im_index = int(img_name.rsplit('.', 1)[0].rsplit('_', 1)[-1])
        save_name = os.path.basename(img_name).split('_', 1)[0] + '_' 
        if flag_NameStartFromZero:
            save_name = save_name + str(im_index + 1) + '.xlsx'
        else:
            save_name = save_name + str(im_index) + '.xlsx'
        # print(save_name)
        
        save_to_xslx(res_obj_dict, os.path.join(os.path.join(args.output, save_name)))

        if args.save_vis:
            # find image file
            tp = os.path.join(img_root, img_name.replace('pk', 'png').replace('pk', 'jpg'))
            ori_img = cv2.imread(os.path.join(img_root, tp))

            for index in res_obj_dict:
                obj = res_obj_dict[index]
                box = obj['最小外接矩形']
            
                cv2.line(ori_img, box[0], box[1], (255, 0, 0), 1)
                cv2.line(ori_img, box[1], box[2], (255, 0, 0), 1)
                cv2.line(ori_img, box[2], box[3], (255, 0, 0), 1)
                cv2.line(ori_img, box[3], box[0], (255, 0, 0), 1)

                cv2.putText(ori_img, str(index), box[0], cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            cv2.imwrite(os.path.join(args.output, save_name + '_vis.png'), ori_img)
