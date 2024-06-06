import json
import cv2
import numpy as np
from matplotlib import pyplot as plt
import imageio
# import rlemasklib
import os,shutil
from PIL import Image
import matplotlib.pyplot as plt

import read_roi
# from roifile import ImagejRoi


def load_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


seg_dir = '/home/RSPrompter/data/bright/gts/'
# if(os.path.exists(seg_dir)):
#     shutil.rmtree(seg_dir)
#     os.mkdir(seg_dir)
# else:
#     os.mkdir(seg_dir)

aa = read_roi.read_roi_zip('/home/RSPrompter/data/bright/gts_imagej/NE4C-B1ROI14-2.zip')
'''
aa的结构：
aa is orderedDict, dict key is every individual label name.
每个成员是dict，keys：['type', 'x', 'y', 'n', 'width', 'name', 'position'] -- n is total numnber of points, 
x,y is the list of x,y coord of all pointss.
'''
bb = 1


def process_roi(roi):
    if roi['type'] in ['traced', 'freehand', 'polygon']:
        if 'x' in roi:
            points = np.array(list(zip(roi['x'], roi['y'])), dtype=np.int32)
            return points.reshape((-1, 1, 2))  # reshape for cv2.fillPoly compatibility
    elif roi['type'] == 'composite':
        composite_mask = np.zeros(image_shape, dtype=np.uint8)
        for path in roi['paths']:
            points = np.array(path, dtype=np.int32)
            cv2.fillPoly(composite_mask, [points], color=255)  # Using 255 just for mask creation
        return composite_mask
    return None

def create_instance_segmentation(rois, image_shape):
    mask = np.zeros(image_shape, dtype=np.uint8)
    for idx, (roi_name, roi_info) in enumerate(rois.items(), start=1):
        if roi_info['type'] == 'composite':
            composite_mask = process_roi(roi_info)
            mask[composite_mask == 255] = idx + 1
        elif roi_info['type'] in ['traced', 'freehand', 'polygon']:
            points = process_roi(roi_info)
            if points is not None:
                cv2.fillPoly(mask, [points], color=(idx + 1))
    return mask

def visualize_instance_segmentation(mask):
    plt.figure(figsize=(10, 10))
    plt.imshow(mask, cmap='jet')
    plt.colorbar()
    plt.title('Instance Segmentation Mask')
    plt.show()

# Main function to process ROI zip file and create instance segmentation
def process_roi_set(zip_path, image_shape):
    rois = read_roi.read_roi_zip(zip_path)
    mask = create_instance_segmentation(rois, image_shape)
    visualize_instance_segmentation(mask)
    return mask

# Example usage
# zip_path = 'RoiSet.zip'  # Path to your RoiSet.zip file
# image_shape = (512, 512)  # Shape of the output mask (adjust as needed)




def save_image(data, file_path):
    """ Save numpy data as an image. """
    # img = Image.fromarray(data.astype(np.uint16))  # Scale data for visibility
    imageio.imwrite(file_path, data.astype(np.uint16))

# Specify paths and image dimensions
roi_zip_path = '/home/RSPrompter/data/bright/gts_imagej/NE4C-B1ROI14-2.zip'
output_image_path = 'B1ROI14_02_1_1_Bright Field_001.tif'
image_shape = (904, 1224)

# Process the ROIs and save the segmentation image
instance_mask = process_roi_set(roi_zip_path, image_shape)
save_image(instance_mask, os.path.join(seg_dir, output_image_path))

print(f"Segmentation image saved as {output_image_path}")



# roi_dir = '/home/RSPrompter/data/bright/test/C5ROI14/'
# files = os.listdir(roi_dir)

# for item in files:
#     json_path = os.path.join(roi_dir, item)
#     # Parse the JSON content
#     data = load_json(json_path)

#     if(len(data['masks']) != 0):
#         # Create an image for the masks
#         height, width = data['masks'][0]['size']
#         img = np.zeros((height, width), dtype=np.uint16)

#         # Assign unique IDs to each mask
#         for idx, rle in enumerate(data['masks']):
#             mask = rlemasklib.decode(rle)
#             img[mask == 1] = idx + 1  # Assign a unique ID (1-based index)

#         # Save the image
#         # output_image = Image.fromarray(img)
#         img_nm = os.path.join(seg_dir, item.replace('json', 'png'))
#         imageio.imwrite(img_nm, img.astype(np.uint16))

#         print("Image saved as masks_image.png")
#     else:
#         print(item + "  no segmentation")