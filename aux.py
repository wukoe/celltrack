import copy
import numpy as np
from skimage import morphology
from onevision import morphology as ovmorph
from onevision import morph_proc
import proc_data
from wbtool.show import wshow


def isolate_filter_object(imobj, segtracker_args, keep_one=True):
    # isolate, then filter small object, with reset new generated ids
    # input: IMbind obj
    # keep_one: True if want to only keep the largest isolated part of a separated object.
    idmax = max(imobj.get_ids())
    imobj = ovmorph.isolate_object(imobj, True, keep_one=keep_one)
    imobj = ovmorph.filter_object_size(imobj, segtracker_args['min_obj_area_ratio'], segtracker_args['max_obj_area_ratio'])

    if not keep_one:
        # 对于新增加物体，将ID重整成连续分布
        ids = np.array(imobj.get_ids())
        if max(ids) > idmax:  #若确实发生了id数量增加
            id_groups = [ids[ids<=idmax], ids[ids>idmax]]
            if max(ids) > (idmax + len(id_groups[1])):  #若确实出现不连续分布的情况 (这里要用idmax而非id_groups[0].max()，因为id_groups[0]中最后的obj可能也在上面的过滤中被删了，不能因此占用原来的id)
                for it in id_groups[1]:
                    idmax += 1
                    imobj.mod_id(it, idmax)
    return imobj

def do_segment(SA, segtracker_args, vid, ids, coords):
    """do segmentation of cells for each frame"""
    fnum = len(vid)

    seg_list = []
    for frame_idx in range(fnum):
        point_prompt = coords[frame_idx]
        frame = vid[frame_idx]
        
        ms = SA.infer(frame, point_prompt)
        frame_idt = ids[frame_idx]
        # 以下的部分可能不需要。
        # I = proc_data.filter_mask_by_size(ms, 0, segtracker_args['max_obj_area_ratio']) #segtracker_args['min_obj_area_ratio']
        # ms = ms[I]
        # frame_idt = frame_idt[I]
        ms = ovmorph.IMbind(ms, 'masks', ids=frame_idt)
        pred_mask = isolate_filter_object(ms, segtracker_args, True)        

        seg_list.append(pred_mask)
        print("segment for frame {}".format(frame_idx), end='\r')

    # 对于第一个segment map，若不遵循ids赋值，则可将ID连续分布，因为其作为tracking的起点。
    # seg_list[0] = morph_proc.id_reset(seg_list[0])
    print('\n-segmentation done')
    return seg_list

def find_new_objs(param, track_mask, seg_mask, curr_idx):
    # 这个算法保存track，不用segment更新track，在细胞项目的背景下不适用。
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
    track_mask: IMbind obj
    seg_mask: same type
    """
    seg_obj_ids = seg_mask.get_ids()
    seg_obj_area = ovmorph.count_obj_area(seg_mask)
    
    track_obj_ids = track_mask.get_ids()
    track_obj_area = ovmorph.count_obj_area(track_mask)
    
    new_track_mask = copy.deepcopy(track_mask)
    obj_num = max(track_obj_ids)
    # print(track_obj_ids)
    for nid in seg_obj_ids:  #通过idx 定位每个新的物体
        # overlapping status of seg[nid] with all track objs.
        intersect_mask = [seg_mask[nid] & track_mask[it] for it in track_mask] 
        intersect_area = np.array([np.sum(it) for it in intersect_mask])
        iou_track = intersect_area / np.array(list(track_obj_area.values())) #[track_obj_area[k] for k in list(track_obj_area.keys()).sort()])
        # need to find index in this for nid of track obj.
        idx = np.argmax(iou_track)
        tid = track_mask.ids[idx]
        
        if nid in track_obj_ids:
            if tid != nid:
                print('most overlap not {} but {}'.format(nid, tid))
            # idx = track_obj_ids.index(nid)
            # if iou_track[idx]<0.5:
            #     if iou_track[idx]==0:
            #         print(nid, "no overlap")
            #     else:
            #         print(nid, "<50% overlap")
            # 还需加一个条件，若seg的比track的大太多，还是维持track的分割。
            if seg_obj_area[nid] <= track_obj_area[nid]*1.5:
                new_track_mask[nid] = seg_mask[nid]
            else:
                # just move location to segment?
                print(nid, ', new seg obj too big')
        else:
            # print(nid, 'new')
            new_track_mask[nid] = seg_mask[nid]
            # new_track_mask.append(seg_mask[nid])
    return new_track_mask

def merge_st_bv__1v(param, track_mask, seg_mask):
    """
    track_mask: IMbind obj
    seg_mask: same type
    """
    seg_obj_ids = seg_mask.get_ids()
    seg_obj_area = ovmorph.count_obj_area(seg_mask)
    
    track_obj_ids = track_mask.get_ids()
    track_obj_area = ovmorph.count_obj_area(track_mask)
    
    new_track_mask = copy.deepcopy(track_mask)
    obj_num = max(track_obj_ids)
    for nid in seg_obj_ids:  #通过idx 定位每个新的物体
        # new object IOU with all tracked obj
        intersect_area = np.array([np.sum(seg_mask[nid] & track_mask[it]) for it in track_obj_ids])
        union_area = np.array([np.sum(seg_mask[nid] | track_mask[it]) for it in track_obj_ids])
        iou = intersect_area / union_area #
        iou_track = intersect_area / np.array(list(track_obj_area.values()))
        # if the overlap exceed percentage of both objects, new segment object will replace that old track object (keep track id), 
        # the old object's non overlapping part will be erased; otherwise, new object will obtain a new id, and cover the overlap part.
        if iou_track.max() > param['new_replace_iou_min']:
            # replace this track object with seg object mask (but keep track ids)
            idx = np.argmax(iou_track)
            tid = track_obj_ids[idx] #找重叠度最大的来替换
            # 还需加一个条件，若seg的比track的大太多，还是维持track的分割。
            if seg_obj_area[nid] <= track_obj_area[tid]*1.2:
                new_track_mask[tid] = seg_mask[nid]
                
                # update obj area
                track_obj_area[tid] = seg_obj_area[nid]
            else:
                pass
        else:
            new_track_mask.append(seg_mask[nid]) #对应new id：obj_num + 1
    # 因为部分像素替代后，可能出现一个物体中间被断开变成若干个的情况，这里要重赋值id变成多个物体（只针对cell的应用场景）。
    new_track_mask = ovmorph.isolate_object(new_track_mask)
    # * 但是这里的ID对应关系改变，有需要的场景要进一步处理。
    
    # filter small object
    new_track_mask = ovmorph.filter_object_size(new_track_mask, param['min_obj_area_ratio'], 1)

    # filter heavily eroded objects - those shrink a lot after - only for old object in tracking.
    temp = ovmorph.count_obj_area(new_track_mask, track_obj_ids)
    post_obj_area = np.array([temp[it] for it in track_obj_ids])
    ppr = post_obj_area / np.array(list(track_obj_area.values()))
    I = ppr < param['remain_remove_max']
    for id in range(len(track_obj_ids)):
        if I[id]:
            del new_track_mask[track_obj_ids[id]]

    return morph_proc.imbind_to_map(new_track_mask)

def merge_st_mapv(param, track_mask, seg_mask):
    """
    track_mask: a 2D id map
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
        iou_track = intersect_area / obj_area
        # if the overlap exceed percentage of both objects, new segment object will replace that old track object (keep track id), 
        # the old object's non overlapping part will be erased; otherwise, new object will obtain a new id, and cover the overlap part.
        if iou_track.max() > param['new_replace_iou_min']:
            # replace this track object with seg object mask (but keep track ids)
            trackid = np.argmax(iou_track)
            id = obj_ids[trackid] #找重叠度最大的来替换
            # 还需加一个条件，若seg的比track的大太多，还是维持track的分割。
            if seg_obj_area[nid] <= obj_area[trackid]*1.2:
                new_track_mask[new_track_mask==id] = 0
                new_track_mask[seg_obj_mask[nid]] = id
                
                # update obj area
                obj_area[trackid] = seg_obj_area[nid]
            else:
                pass
        else:
            obj_num += 1
            new_track_mask[seg_obj_mask[nid]] = obj_num
    # 因为部分像素替代后，可能出现一个物体中间被断开变成若干个的情况，这里要重赋值id变成多个物体（只针对cell的应用场景）。
    new_track_mask = ovmorph.isolate_object(new_track_mask)
    # * 但是这里的ID对应关系改变，有需要的场景要进一步处理。
    
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
