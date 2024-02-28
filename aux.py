import numpy as np
from skimage import morphology
from onevision import morphology as ovmorph
from onevision import morph_proc
import proc_data


def do_segment(SA, segtracker_args, vid, coords):
    fnum = len(vid)

    seg_list = []
    for frame_idx in range(fnum):
        point_prompt = coords[frame_idx]
        frame = vid[frame_idx]
        
        ms = SA.infer(frame, point_prompt)
        ms = proc_data.filter_mask_by_size(ms, segtracker_args['min_obj_area_ratio'], segtracker_args['max_obj_area_ratio'])
        ms = morph_proc.masks_to_map(ms)
        ms = ovmorph.isolate_object(ms, False)
        pred_mask = ovmorph.filter_object_size(ms, segtracker_args['min_obj_area_ratio'], segtracker_args['max_obj_area_ratio'])
        seg_list.append(pred_mask)
        print("segment for frame {}".format(frame_idx), end='\r')

    # 对于第一个segment map，将ID连续分布，因为其作为tracking的起点。
    seg_list[0] = morph_proc.id_reset(seg_list[0])
    print('\n-segmentation done')
    return seg_list

def find_new_objs(param, track_mask, seg_mask, curr_idx):
    # 这个算法保存track，不是为了后续用segment更新track，在细胞项目的背景下不适用。
    seg_obj_mask = (track_mask==0) * seg_mask #找到所有track mask 之外的区域的新seg mask
    seg_obj_ids = np.unique(seg_obj_mask)
    seg_obj_ids = seg_obj_ids[seg_obj_ids!=0]
    # obj_num = self.get_obj_num() + 1
    obj_num = curr_idx
    for idx in seg_obj_ids:  #通过idx 定位每个新的物体
        seg_obj_area = np.sum(seg_obj_mask==idx)
        obj_area = np.sum(seg_mask==idx)
        if seg_obj_area/obj_area < param['min_new_obj_iou'] or seg_obj_area < param['min_area'] or obj_num > param['max_obj_num']:
            seg_obj_mask[seg_obj_mask==idx] = 0  #违反3条件任何1个的就不算作新物体，这个额外的部分也被擦除。
        else:
            seg_obj_mask[seg_obj_mask==idx] = obj_num
            obj_num += 1
    return seg_obj_mask

def merge_st(param, track_mask, seg_mask):
    """
    track_mask: a 2D mask map
    seg_mask: same type
    """
    seg_obj_mask,seg_obj_ids = morph_proc.map_to_masks(seg_mask, return_ids=True)
    seg_obj_area = [np.sum(it) for it in seg_obj_mask]

    obj_mask,obj_ids = morph_proc.map_to_masks(track_mask, return_ids=True)
    obj_area = np.array([np.sum(it) for it in obj_mask])
    
    new_track_mask = track_mask.copy()
    
    obj_num = obj_ids.max()
    for nid in range(len(seg_obj_ids)):  #通过idx 定位每个新的物体
        # new object IOU with all tracked obj
        intersect_area = np.array([np.sum(seg_obj_mask[nid] & it) for it in obj_mask])
        union_area = np.array([np.sum(seg_obj_mask[nid] | it) for it in obj_mask])
        iou = intersect_area / union_area #
        # if the overlap exceed percentage of both objects, new segment object will replace that old track object (keep track id), 
        # the old object's non overlapping part will be erased; otherwise, new object will obtain a new id, and cover the overlap part.
        if iou.max() > param['new_replace_iou_min']:
            # replace this track object with seg object mask (but keep track ids)
            trackid = np.argmax(iou)
            id = obj_ids[trackid] #找重叠度最大的来替换
            new_track_mask[new_track_mask==id] = 0
            new_track_mask[seg_obj_mask[nid]] = id
            # update obj area
            obj_area[trackid] = seg_obj_area[nid]
        else:
            new_track_mask[seg_obj_mask[nid]] = obj_num
            obj_num += 1
    # # 因为部分像素替代后，可能出现一个物体中间被断开变成若干个的情况，这里要重赋值id变成多个物体（只针对cell的应用场景）。
    # new_track_mask = ovmorph.isolate_object(new_track_mask)
    # # * 但是这里的ID对应关系改变，有需要的场景要进一步处理。
    
    # filter small object
    ta = np.prod(track_mask.shape)
    thres = int(ta * param['min_obj_area_ratio'])
    new_track_mask = morphology.remove_small_objects(new_track_mask, min_size=thres)

    # filter heavily eroded objects - those shrink a lot after - only for old object in tracking.
    temp = ovmorph.count_obj_area(new_track_mask, obj_ids)
    post_obj_area = np.array([temp[it] for it in obj_ids])
    ppr = post_obj_area / obj_area
    I = ppr < param['remain_remove_max']    
    for id in range(len(obj_ids)):
        if I[id]:
            new_track_mask[new_track_mask==obj_ids[id]] = 0

    return new_track_mask

def do_st_merge(segtracker_args, track_mask, seg_mask):
    idmax = track_mask.max()
    track_mask = ovmorph.isolate_object(track_mask)
    track_mask = ovmorph.filter_object_size(track_mask, segtracker_args['min_obj_area_ratio'], segtracker_args['max_obj_area_ratio'])
    # 对于新增加物体，将ID重整成连续分布
    ids = np.unique(track_mask)
    if ids.max() > idmax:  #若确实发生了id数量增加
        ids = ids[ids!=0]
        id_groups = [ids[ids<=idmax], ids[ids>idmax]]
        if id_groups[1].max() > (id_groups[0].max() + len(id_groups[1])):  #若确实出现不连续分布的情况
            temp = morph_proc.split_map(track_mask, id_groups)
            temp[1] = morph_proc.id_reset(temp[1], base=idmax+1)
            track_mask = temp[0] + temp[1]
    
    # merge tracking and new segment maps.
    # 1/2
    # pred_mask = track_mask + new_obj_mask
    # 2/2
    pred_mask = merge_st(segtracker_args, track_mask, seg_mask)
    # e/2
    return pred_mask
