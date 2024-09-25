# import os
# import numpy as np
# import cv2
# import pandas as pd
#
# def calculate_shape_features(binary, num):
#     # 查找轮廓
#     contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     # 假设最大的轮廓是感兴趣的对象
#     contour = max(contours, key=cv2.contourArea)
#
#     # 计算面积
#     area = cv2.contourArea(contour)
#
#     # 计算周长
#     perimeter = cv2.arcLength(contour, True)
#
#     # 计算矩以获得轮廓的中心
#     moments = cv2.moments(contour)
#     if moments['m00'] == 0:  # 避免除零错误
#         moments['m00'] = 1
#
#     cx = int(moments['m10'] / moments['m00'])
#     cy = int(moments['m01'] / moments['m00'])
#
#     # 拟合椭圆以获得长轴和短轴长度及其端点坐标和夹角
#     if len(contour) >= 5:  # 拟合椭圆至少需要 5 个点
#         ellipse = cv2.fitEllipse(contour)
#         (center, axes, orientation) = ellipse
#         major_axis_length = max(axes)
#         minor_axis_length = min(axes)
#         if minor_axis_length==0 or orientation == 90.0 or orientation == 0.0:
#             major_axis_length = minor_axis_length = axis_ratio = 0
#             major_axis_end1 = major_axis_end2 = minor_axis_end1 = minor_axis_end2 = (0, 0)
#             orientation = 0
#             major_axis_endpoints_x1 = 0
#             major_axis_endpoints_x2 = 0
#             major_axis_endpoints_y1 = 0
#             major_axis_endpoints_y2 = 0
#             #
#             minor_axis_endpoints_x1 = 0
#             minor_axis_endpoints_x2 = 0
#             minor_axis_endpoints_y1 = 0
#             minor_axis_endpoints_y2 = 0
#         else:
#             axis_ratio = major_axis_length / minor_axis_length
#
#             # 计算长轴和短轴的端点坐标
#             angle_rad = np.deg2rad(orientation)
#             cos_angle = np.cos(angle_rad)
#             sin_angle = np.sin(angle_rad)
#             # print(center, axes, orientation)
#             # print(center,major_axis_length,cos_angle)
#             major_axis_end1 = (int(center[0] + major_axis_length / 2 * cos_angle),
#                                int(center[1] + major_axis_length / 2 * sin_angle))
#             major_axis_end2 = (int(center[0] - major_axis_length / 2 * cos_angle),
#                                int(center[1] - major_axis_length / 2 * sin_angle))
#
#             minor_axis_end1 = (int(center[0] - minor_axis_length / 2 * sin_angle),
#                                int(center[1] + minor_axis_length / 2 * cos_angle))
#             minor_axis_end2 = (int(center[0] + minor_axis_length / 2 * sin_angle),
#                                int(center[1] - minor_axis_length / 2 * cos_angle))
#
#             major_axis_endpoints_x1 = int(center[0] + major_axis_length / 2 * cos_angle)
#             major_axis_endpoints_x2 = int(center[0] - major_axis_length / 2 * cos_angle)
#             major_axis_endpoints_y1 = int(center[1] + major_axis_length / 2 * sin_angle)
#             major_axis_endpoints_y2 = int(center[1] - major_axis_length / 2 * sin_angle)
#
#             minor_axis_endpoints_x1 = int(center[0] - minor_axis_length / 2 * sin_angle)
#             minor_axis_endpoints_x2 = int(center[0] + minor_axis_length / 2 * sin_angle)
#             minor_axis_endpoints_y1 = int(center[1] + minor_axis_length / 2 * cos_angle)
#             minor_axis_endpoints_y2 = int(center[1] - minor_axis_length / 2 * cos_angle)
#
#
#     else:
#         major_axis_length = minor_axis_length = axis_ratio = 0
#         major_axis_end1 = major_axis_end2 = minor_axis_end1 = minor_axis_end2 = (0, 0)
#         orientation = 0
#         major_axis_endpoints_x1 = 0
#         major_axis_endpoints_x2 = 0
#         major_axis_endpoints_y1 = 0
#         major_axis_endpoints_y2 = 0
#         #
#         minor_axis_endpoints_x1 = 0
#         minor_axis_endpoints_x2 = 0
#         minor_axis_endpoints_y1 = 0
#         minor_axis_endpoints_y2 = 0
#
#     # 计算圆度
#     circularity = (4 * np.pi * area) / (perimeter ** 2)
#
#     # 计算凸包和凹度（凸性）
#     hull = cv2.convexHull(contour)
#     hull_area = cv2.contourArea(hull)
#     convexity = area / hull_area
#
#     features = {
#         'area': area,
#         'perimeter': perimeter,
#         'centroid': (cx, cy),
#         'major_axis_length': major_axis_length,
#         'minor_axis_length': minor_axis_length,
#         'axis_ratio': axis_ratio,
#         # 'circularity': circularity,
#         # 'convexity': convexity,
#         'major_axis_endpoints_x1': major_axis_endpoints_x1,
#         'major_axis_endpoints_y1': major_axis_endpoints_y1,
#         'major_axis_endpoints_x2': major_axis_endpoints_x2,
#         'major_axis_endpoints_y2': major_axis_endpoints_y2
#         # 'minor_axis_endpoints_x1': minor_axis_endpoints_x1,
#         # 'minor_axis_endpoints_y1': minor_axis_endpoints_y1,
#         # 'minor_axis_endpoints_x2': minor_axis_endpoints_x2,
#         # 'minor_axis_endpoints_y2': minor_axis_endpoints_y2,
#         # 'major_axis_endpoints': (major_axis_end1, major_axis_end2),
#         # 'minor_axis_endpoints': (minor_axis_end1, minor_axis_end2),
#         # 'orientation': orientation
#     }
#
#     return features
# def create_dataframe(dict_data_list):
#     """Create a DataFrame from a list of dictionaries."""
#     # Create DataFrame from list of dictionaries
#     df = pd.DataFrame(dict_data_list)
#     return df
#
# def function(mask):
#     _, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
#
#     filtered_labels = []
#     dials = []
#     for label, stat in enumerate(stats):
#         ratioc = stat[2] / stat[3]
#         fillratio = stat[4] / (stat[2] * stat[3])
#         if 10 <= stat[4] < 500000:
#             filtered_labels.append(label)
#             dials.append([stat[0], stat[1], stat[2], stat[3], max(stat[2], stat[3])])
#     if len(dials)>0:
#         return dials[0]
#     else:
#         return []
# def get_color_from_map(index, color_map):
#     num_colors = len(color_map)
#     idx = index % num_colors  # 防止索引超出范围
#     return color_map[idx]
# def read_npy_file(file_path):
#     try:
#         data = np.load(file_path)
#         return data
#     except Exception as e:
#         print(f"读取文件时出错: {e}")
#         return None
# def read_npy_file_as_int16(file_path):
#     try:
#         data = np.load(file_path)
#         data_int16 = data.astype(np.int16)
#         return data_int16
#     except Exception as e:
#         print(f"读取文件时出错: {e}")
#         return None
#
# def save_image_from_array(array, image_path):
#     try:
#         img0 = np.zeros_like(array)
#         img0[array!=0]=255
#         cv2.imwrite(image_path,img0)
#     except Exception as e:
#         print(f"保存图片时出错: {e}")
# def generate_color_map(num_colors):
#     colors = []
#     for i in range(num_colors):
#         hue = int(180 * i / num_colors)  # 色调范围 0-179
#         saturation = 255  # 饱和度范围 0-255
#         value = 180  # 亮度范围 0-255
#         color = np.array([hue, saturation, value], dtype=np.uint8)
#         colors.append(color)
#     return colors
#
# def read_npy_file_as_int16(file_path,output_dir,name):
#
#     # output_dir = "out"
#     try:
#         image = cv2.imread(file_path,-1)
#     except Exception as e:
#         print(f"读取文件时出错: {e}")
#         return None
#
#     print(image.dtype)
#     print(image.shape)
#
#     # 获取图像的高度和宽度
#     height, width = image.shape
#
#     # 使用集合来存储所有出现过的像素值
#     unique_pixels = set()
#
#     # 遍历图像的每个像素
#     for y in range(height):
#         for x in range(width):
#             # 获取当前像素的值
#             pixel = image[y, x]
#             # 添加到集合中
#             unique_pixels.add(pixel)
#
#     # 将集合转换为列表
#     unique_pixels_list = list(unique_pixels)
#
#     print(f"Total unique pixels: {len(unique_pixels_list)}")
#
#     # 示例：打印前10个唯一像素值
#     print(unique_pixels_list[:10])
#     masks = []
#     # 遍历唯一像素值列表
#
#     dict_data_list = []
#     # 生成颜色映射
#     num_colors = 20  # 颜色的数量
#     color_map = generate_color_map(num_colors)
#     h, w = image.shape[:2]
#     outimg1 = np.zeros((h, w), dtype=np.uint8)
#     outimg2 = np.zeros((h, w), dtype=np.uint8)
#     outimg3 = np.zeros((h, w), dtype=np.uint8)
#
#     for i, pixel in enumerate(unique_pixels_list):
#         # 创建一个布尔掩码，检查图像中哪些像素值等于当前的(pixel)
#         if pixel == 0:
#             continue
#         mask = (image == pixel)
#         masks.append(mask)
#
#
#         imgx2 = mask
#         imgx2 = imgx2.astype(np.uint8) * 255
#         mask = mask
#         cv2.imwrite("4.jpg", imgx2)
#         rect = function(imgx2)
#         if rect==[]:
#             continue
#         num = i + 1
#         features = calculate_shape_features(imgx2, num)
#         dict_data_list.append(features)
#
#         # 获取颜色
#         color = get_color_from_map(num, color_map)
#         bgr_color = cv2.cvtColor(np.array([[color]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0][0].tolist()
#         # outimg1[mask] = 0  # BGR 中的红色 (B=0, G=0, R=255)
#         outimg1[mask] = bgr_color[0]  # BGR 中的红色 (B=0, G=0, R=255)
#         outimg2[mask] = bgr_color[1]  # BGR 中的红色 (B=0, G=0, R=255)
#         outimg3[mask] = bgr_color[2]  # BGR 中的红色 (B=0, G=0, R=255)
#         bgr_color = (255, 255, 255)
#         cv2.putText(outimg1, f"{num}", (rect[0] + rect[2] // 2 - 12, rect[1] + rect[3] // 2 + 7),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, bgr_color[0], 1, cv2.LINE_AA)
#         cv2.putText(outimg2, f"{num}", (rect[0] + rect[2] // 2 - 12, rect[1] + rect[3] // 2 + 7),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, bgr_color[1], 1, cv2.LINE_AA)
#         cv2.putText(outimg3, f"{num}", (rect[0] + rect[2] // 2 - 12, rect[1] + rect[3] // 2 + 7),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, bgr_color[2], 1, cv2.LINE_AA)
#
#         # break
#     outimg = cv2.merge([outimg1, outimg2, outimg3])
#     # name = "1"
#     outimgpath = os.path.join(output_dir, name + ".jpg")
#     cv2.imwrite(outimgpath, outimg)
#     df = create_dataframe(dict_data_list)
#     # 保存 DataFrame 到 Excel 文件
#     output_path = os.path.join(output_dir, name + ".xlsx")
#     df.to_excel(output_path, index=False)
#     print(f"成功保存 Data 到 {output_path} 文件.")
#
#
# if __name__ == '__main__':
#
#     imgp = r"G:\pyprojects\cellreg\seg"   # 修改路径
#     output_dir = "out"
#     if not os.path.exists(output_dir):os.mkdir(output_dir)
#     ls = os.listdir(imgp)
#     ls = [x for x in ls if x.endswith(".jpg") or x.endswith(".png")]
#     for fit in ls:
#         # print(fit)
#         name = fit[:-4]
#         input_path = os.path.join(imgp,fit)
#         read_npy_file_as_int16(input_path, output_dir,name)
#
#         # break
#
#
import os
import numpy as np
import cv2
import pandas as pd

def calculate_shape_features(binary, num):
    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 假设最大的轮廓是感兴趣的对象
    contour = max(contours, key=cv2.contourArea)

    # 计算面积
    area = cv2.contourArea(contour)

    # 计算周长
    perimeter = cv2.arcLength(contour, True)

    # 计算矩以获得轮廓的中心
    moments = cv2.moments(contour)
    if moments['m00'] == 0:  # 避免除零错误
        moments['m00'] = 1

    cx = int(moments['m10'] / moments['m00'])
    cy = int(moments['m01'] / moments['m00'])

    # 拟合椭圆以获得长轴和短轴长度及其端点坐标和夹角
    if len(contour) >= 5:  # 拟合椭圆至少需要 5 个点
        ellipse = cv2.fitEllipse(contour)
        (center, axes, orientation) = ellipse
        major_axis_length = max(axes)
        minor_axis_length = min(axes)
        axis_ratio = major_axis_length / minor_axis_length
        if axis_ratio > 6:
            return -1
        # 计算长轴和短轴的端点坐标
        angle_rad = np.deg2rad(orientation)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)

        major_axis_end1 = (int(center[0] + major_axis_length / 2 * cos_angle),
                           int(center[1] + major_axis_length / 2 * sin_angle))
        major_axis_end2 = (int(center[0] - major_axis_length / 2 * cos_angle),
                           int(center[1] - major_axis_length / 2 * sin_angle))

        minor_axis_end1 = (int(center[0] - minor_axis_length / 2 * sin_angle),
                           int(center[1] + minor_axis_length / 2 * cos_angle))
        minor_axis_end2 = (int(center[0] + minor_axis_length / 2 * sin_angle),
                           int(center[1] - minor_axis_length / 2 * cos_angle))

    else:
        major_axis_length = minor_axis_length = axis_ratio = 0
        major_axis_end1 = major_axis_end2 = minor_axis_end1 = minor_axis_end2 = (0, 0)
        orientation = 0

    # 计算圆度
    circularity = (4 * np.pi * area) / (perimeter ** 2)

    # 计算凸包和凹度（凸性）
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    convexity = area / hull_area

    features = {
        'area': area,
        'perimeter': perimeter,
        'centroid': (cx, cy),
        'major_axis_length': major_axis_length,
        'minor_axis_length': minor_axis_length,
        'axis_ratio': axis_ratio,
        'circularity': circularity,
        'convexity': convexity,
        'major_axis_endpoints': (major_axis_end1, major_axis_end2),
        'minor_axis_endpoints': (minor_axis_end1, minor_axis_end2),
        'orientation': orientation
    }

    return features
def create_dataframe(dict_data_list):
    """Create a DataFrame from a list of dictionaries."""
    # Create DataFrame from list of dictionaries
    df = pd.DataFrame(dict_data_list)
    return df

def function(mask):
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    filtered_labels = []
    dials = []
    for label, stat in enumerate(stats):
        ratioc = stat[2] / stat[3]
        fillratio = stat[4] / (stat[2] * stat[3])
        if 10 <= stat[4] < 500000:
            filtered_labels.append(label)
            dials.append([stat[0], stat[1], stat[2], stat[3], max(stat[2], stat[3])])

    return dials[0]
def get_color_from_map(index, color_map):
    num_colors = len(color_map)
    idx = index % num_colors  # 防止索引超出范围
    return color_map[idx]
def read_npy_file(file_path):
    try:
        data = np.load(file_path)
        return data
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None
def read_npy_file_as_int16(file_path):
    try:
        data = np.load(file_path)
        data_int16 = data.astype(np.int16)
        return data_int16
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None

def save_image_from_array(array, image_path):
    try:
        img0 = np.zeros_like(array)
        img0[array!=0]=255
        cv2.imwrite(image_path,img0)
    except Exception as e:
        print(f"保存图片时出错: {e}")
def generate_color_map(num_colors):
    colors = []
    for i in range(num_colors):
        hue = int(180 * i / num_colors)  # 色调范围 0-179
        saturation = 255  # 饱和度范围 0-255
        value = 180  # 亮度范围 0-255
        color = np.array([hue, saturation, value], dtype=np.uint8)
        colors.append(color)
    return colors

def read_npy_file_as_int16(file_path,output_dir,name):

    # output_dir = "out"
    try:
        image = cv2.imread(file_path,1)
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None
    # 获取图像的高度和宽度
    height, width, _ = image.shape

    # 使用集合来存储所有出现过的像素值
    unique_pixels = set()

    # 遍历图像的每个像素
    for y in range(height):
        for x in range(width):
            # 获取当前像素的值
            pixel = tuple(image[y, x])
            # 添加到集合中
            unique_pixels.add(pixel)

    # 将集合转换为列表
    unique_pixels_list = list(unique_pixels)

    print(f"Total unique pixels: {len(unique_pixels_list)}")

    # 示例：打印前10个唯一像素值
    print(unique_pixels_list[:10])
    masks = []
    # 遍历唯一像素值列表


    dict_data_list = []
    # 生成颜色映射
    num_colors = 20  # 颜色的数量
    color_map = generate_color_map(num_colors)
    h, w = image.shape[:2]
    outimg1 = np.zeros((h, w), dtype=np.uint8)
    outimg2 = np.zeros((h, w), dtype=np.uint8)
    outimg3 = np.zeros((h, w), dtype=np.uint8)


    for i,pixel in enumerate(unique_pixels_list) :
        # 创建一个布尔掩码，检查图像中哪些像素值等于当前的(pixel)
        if pixel == (0,0,0):
            continue
        mask = (image[:, :, 0] == pixel[0]) & (image[:, :, 1] == pixel[1]) & (image[:, :, 2] == pixel[2])
        masks.append(mask)


    # # 示例：打印第一个掩码的形状和像素值
    # if masks:
    #     print(f"Shape of first mask: {masks[0].shape}")
    #     print(f"First pixel value: {unique_pixels_list[0]}")
    #     print(f"len value: {len(masks)}")
    # else:
    #     print("No masks were generated.")
    #
    # return



        imgx2 =mask
        imgx2 = imgx2.astype(np.uint8)*255
        mask = mask
        cv2.imwrite("4.jpg",imgx2)
        rect = function(imgx2)
        num = i+1
        features = calculate_shape_features(imgx2, num)
        if features ==-1:
            continue
        dict_data_list.append(features)

        # 获取颜色
        color = get_color_from_map(num, color_map)
        bgr_color = cv2.cvtColor(np.array([[color]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0][0].tolist()
        # outimg1[mask] = 0  # BGR 中的红色 (B=0, G=0, R=255)
        outimg1[mask] = bgr_color[0]  # BGR 中的红色 (B=0, G=0, R=255)
        outimg2[mask] = bgr_color[1]  # BGR 中的红色 (B=0, G=0, R=255)
        outimg3[mask] = bgr_color[2]  # BGR 中的红色 (B=0, G=0, R=255)
        bgr_color = (255, 255, 255)
        cv2.putText(outimg1, f"{num}", (rect[0] + rect[2] // 2 - 12, rect[1] + rect[3] // 2 + 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, bgr_color[0], 1, cv2.LINE_AA)
        cv2.putText(outimg2, f"{num}", (rect[0] + rect[2] // 2 - 12, rect[1] + rect[3] // 2 + 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, bgr_color[1], 1, cv2.LINE_AA)
        cv2.putText(outimg3, f"{num}", (rect[0] + rect[2] // 2 - 12, rect[1] + rect[3] // 2 + 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, bgr_color[2], 1, cv2.LINE_AA)

        # break
    outimg = cv2.merge([outimg1, outimg2, outimg3])
    # name = "1"
    outimgpath = os.path.join(output_dir, name + ".jpg")
    cv2.imwrite(outimgpath, outimg)
    df = create_dataframe(dict_data_list)
    # 保存 DataFrame 到 Excel 文件
    output_path = os.path.join(output_dir, name + ".xlsx")
    df.to_excel(output_path, index=False)
    print(f"成功保存 Data 到 {output_path} 文件.")




if __name__ == '__main__':

    imgp = r"G:\pyprojects\cellreg\seg"   # 修改路径
    output_dir = "out"
    if not os.path.exists(output_dir):os.mkdir(output_dir)
    ls = os.listdir(imgp)
    ls = [x for x in ls if x.endswith(".jpg") or x.endswith(".png")]
    for fit in ls:
        print(fit)
        name = fit[:-4]
        input_path = os.path.join(imgp,fit)
        read_npy_file_as_int16(input_path, output_dir,name)

        break



