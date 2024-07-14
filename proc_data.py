import os
import numpy as np
import pandas as pd
from PIL import Image

from misc_tool import file_misc,show
from onevision import improc,morphology,morph_data
from aot_tracker import _palette

# === data process
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
    return
    I: index of masks to keep, use it by masks[I]
    """
    sz = masks[0].shape
    imga = np.prod(sz)

    ims = np.array(masks)
    ma = ims.sum(2).sum(1)
    mar = ma/imga
    I = np.bitwise_and(mar<hb, mar>lb)
    return np.arange(len(I))[I]
    
    # ta = np.prod(track_mask.shape)
    # thres = int(ta * param['min_keep_area_ratio'])
    # new_track_mask = morphology.remove_small_objects(new_track_mask, min_size=thres)

# === read data
def read_coords_from_csv(csv):
    """export project csv file
    organize data by slice, coord of all cells in one frame is placed in a list, which is in a parent list of all frames.
    """
    df = pd.read_csv(csv, encoding='unicode_escape')
    group_obj = df.groupby('Slice')

    cellid = []
    frame_coord = []
    for it in group_obj:
        cellid.append(it[1]['Track'].to_numpy())        
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
        x = file_misc.imgtiff_read(os.path.join(folder_path, it))
        D.append(x)
    D = np.stack(D)
    print(D.shape)

    fp,fname = folder_path.rstrip('/').rsplit('/', 1)
    fname = fname.split('_')[0]
    print(fp, fname)
    np.save(os.path.join(fp, fname), D)
    # ! the obtained data need to be value-transformed before using.

# === output data
def all2imgfile(maps, fname='', direct_write=False):
    # 保存多个map
    output_dir = '/home/wb/samba_dir/indev/imouts/'
    if direct_write: # 不需要彩色，直接保存数值（所得图可能需要imageJ这样的软件查看）
        for mi in range(len(maps)):
            file_misc.imwrite(os.path.join(output_dir, fname+'{}.png'.format(mi)), maps[mi])
    else:
        for mi in range(len(maps)):
            if isinstance(maps[mi], morph_data.IMbind):
                mask = morph_data.imbind_to_map(maps[mi])
            else:
                mask = maps[mi]
            save_prediction(mask, output_dir, fname+'{}.png'.format(mi))

def save_prediction(pred_mask, output_dir, file_name, opt='mask', image=None):
    '''保存一张map的所有mask，彩色染色的mask。可以是mask或加载在图上
    input
    ---
    pred_mask is a map.
    '''
    if opt=='overlay':
        ovimg = maskoverimg(image, pred_mask, True)
        file_misc.imwrite(os.path.join(output_dir, file_name), ovimg)
    else:
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

def maskoverimg(frame, imobj, show_id=False):
    ''' put mask over an image.
    Input
    ---
    frame : must be 3 channel natural image array
    imobj : same size as frame.
    '''
    # all to masks format.
    if isinstance(imobj, morph_data.IMbind):
        masks = [imobj[k] for k in imobj]
        ids = imobj.get_ids()
    elif isinstance(imobj, np.ndarray) and len(imobj.shape)==2:
        masks,ids = morph_data.map_to_masks(imobj, True)
    else:
        masks = imobj
        ids = range(1, len(imobj)+1)

    img = show.overlay_mask(frame, masks, ids)
    if show_id:
        mc = morphology.mass_center(masks)
        # mc = [[it[1], it[0]] for it in mc]
        id_text = [str(it) for it in ids]
        img = show.overlay_text(img, mc, id_text)
    return img

def overlay_all(vid, images):
    fnum = len(vid)
    assert len(images) == fnum, "img mask number not same"
    ovimg = [] 
    for k in range(fnum):
        ovimg.append(maskoverimg(vid[k], images[k]))
        print("overlayed frame {}".format(k), end='\r')
    return ovimg

# def draw_mask(img, mask, alpha=0.5, id_countour=False):
#     img_mask = np.zeros_like(img)
#     img_mask = img
#     if id_countour:
#         # very slow ~ 1s per image
#         obj_ids = np.unique(mask)
#         obj_ids = obj_ids[obj_ids!=0]

#         for id in obj_ids:
#             # Overlay color on  binary mask
#             if id <= 255:
#                 color = _palette[id*3:id*3+3]
#             else:
#                 color = [0,0,0]
#             foreground = img * (1-alpha) + np.ones_like(img) * alpha * np.array(color)
#             binary_mask = (mask == id)

#             # Compose image
#             img_mask[binary_mask] = foreground[binary_mask]

#             countours = binary_dilation(binary_mask,iterations=1) ^ binary_mask
#             img_mask[countours, :] = 0
#     else:
#         binary_mask = (mask!=0)
#         countours = binary_dilation(binary_mask,iterations=1) ^ binary_mask
#         foreground = img*(1-alpha)+colorize_mask(mask)*alpha
#         img_mask[binary_mask] = foreground[binary_mask]
#         img_mask[countours,:] = 0
        
#     return img_mask.astype(img.dtype)


# 批量处理细胞核mask
def proc_nuc_mask(nuclear_mask_dir, output_nuc_dir):
    # output_nuc_dir = '/home/wb/samba_dir/cells_brightfield/process/e6000_segs_cyto_processed/' 
    flist = glob.glob(os.path.join(nuclear_mask_dir, '*.png'))

    for file in flist:
        fn_nuc = os.path.basename(file)
        print('doing ' + fn_nuc)
        

        # fn, fn_nuc, mark = aux.find_corr_dapi(fn, nuclear_mask_dir)
        # if not os.path.isfile(fn_nuc):
        #     print(fn_nuc, 'not found')
        #     continue
        
        try:
            K = fm.imread(file)
        except:
            print(file, 'loading error')
            continue

        try:
            K = morph_data.IMbind(K, 'map')
            # filter nuclear mask
            idx = aux.detect_fake_nuc(K)
            # y = copy.deepcopy(x)
            del K[idx]

            K = morph_data.imbind_to_map(K)
            # fn_nuc = os.path.basename(fn_nuc)
            fm.imwrite(os.path.join(output_nuc_dir, fn_nuc), K)
        except:
            print(fn, 'wrong')
            continue
            
        print('done')


if __name__ == "__main__":
    # folder_path = '/home/wb/samba_dir/cells_brightfield/leading-HELA-entirespan/A1ROI2_02_1_1_Bright Field'
    # takein_folder_image(folder_path)

    img_path = '/home/wb/samba_dir/cells_brightfield/single-cell-movement-for-machine-learning/A549/0/A4ROI3.tif'
    imgs = file_misc.imgtiff_read(img_path)
    np.save(img_path.rsplit('.')[0], imgs)
