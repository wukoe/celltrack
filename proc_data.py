import os
import numpy as np
import pandas as pd
from PIL import Image

from wbtool import file_misc as fm
from onevision import improc,morph_proc
from aot_tracker import _palette


def value_modify(vid):
    if vid.dtype in ['uint16', '>u2']:
        vid = improc.value_uint16_to_uint8(vid)
    return vid

def ch_num_modify(vid):
    if len(vid[0].shape) == 2:
        D = []
        for it in vid:
            D.append(improc.channel_1to3(it))
        vid = np.stack(D)
    return vid

def filter_mask_by_size(masks, lb, hb):
    """
    filter object that's too big or small before merge multiple masks to a map.
    ----
    input
    masks: 3D numpy array or list of 2D arrays.
    """
    sz = masks[0].shape
    imga = np.prod(sz)

    ims = np.array(masks)
    ma = ims.sum(2).sum(1)
    mar = ma/imga
    I = np.bitwise_and(mar<hb, mar>lb)
    return ims[I]
    
    # ta = np.prod(track_mask.shape)
    # thres = int(ta * param['min_keep_area_ratio'])
    # new_track_mask = morphology.remove_small_objects(new_track_mask, min_size=thres)

    
def read_coords_from_csv(csv):
    """export project csv file
    organize data by slice, coord of all cells in one frame is placed in a list, which is in a parent list of all frames.
    """
    df = pd.read_csv(csv, encoding='unicode_escape')
    group_obj = df.groupby('Slice')

    cellid = []
    frame_coord = []
    for it in group_obj:
        cellid.append(it[0])        
        frame_coord.append(it[1][['X','Y']].to_numpy())
    return cellid,frame_coord


def takein_folder_image(folder_path):
    """
    take in a folder's (TIFF) images; combine to one tensor; store it.
    """
    fl = os.listdir(folder_path)
    fl.sort()

    D = []
    for it in fl:
        x = fm.imgtiff_read(os.path.join(folder_path, it))
        D.append(x)
    D = np.stack(D)
    print(D.shape)

    fp,fname = folder_path.rstrip('/').rsplit('/', 1)
    fname = fname.split('_')[0]
    print(fp, fname)
    np.save(os.path.join(fp, fname), D)

    # the data need to be value-transformed before using.


def save_prediction(pred_mask, output_dir, file_name):
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode='P')
    save_mask.putpalette(_palette)
    save_mask.save(os.path.join(output_dir, file_name))

def colorize_mask(pred_mask):
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode='P')
    save_mask.putpalette(_palette)
    save_mask = save_mask.convert(mode='RGB')
    return np.array(save_mask)

def draw_mask(img, mask, alpha=0.5, id_countour=False):
    img_mask = np.zeros_like(img)
    img_mask = img
    if id_countour:
        # very slow ~ 1s per image
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids!=0]

        for id in obj_ids:
            # Overlay color on  binary mask
            if id <= 255:
                color = _palette[id*3:id*3+3]
            else:
                color = [0,0,0]
            foreground = img * (1-alpha) + np.ones_like(img) * alpha * np.array(color)
            binary_mask = (mask == id)

            # Compose image
            img_mask[binary_mask] = foreground[binary_mask]

            countours = binary_dilation(binary_mask,iterations=1) ^ binary_mask
            img_mask[countours, :] = 0
    else:
        binary_mask = (mask!=0)
        countours = binary_dilation(binary_mask,iterations=1) ^ binary_mask
        foreground = img*(1-alpha)+colorize_mask(mask)*alpha
        img_mask[binary_mask] = foreground[binary_mask]
        img_mask[countours,:] = 0
        
    return img_mask.astype(img.dtype)

if __name__ == "__main__":
    folder_path = '/home/wb/samba_dir/cells_brightfield/leading-HELA-entirespan/A1ROI2_02_1_1_Bright Field'
    takein_folder_image(folder_path)