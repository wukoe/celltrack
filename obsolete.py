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

    return morph_data.imbind_to_map(new_track_mask)

def merge_st_mapv(param, track_mask, seg_mask):
    """
    track_mask: a 2D id map
    seg_mask: same type
    """
    seg_obj_mask,seg_obj_ids = morph_data.map_to_masks(seg_mask, return_ids=True)
    seg_obj_area = [np.sum(it) for it in seg_obj_mask]

    obj_mask,obj_ids = morph_data.map_to_masks(track_mask, return_ids=True)
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


# === 供参考的SegTracker函数

# 刘平版：
def detect_and_seg_pt(self, origin_frame: np.ndarray, coords, modes):
    '''
    Using Grounding-DINO to detect object acc Text-prompts
    Retrun:
        refined_merged_mask: numpy array (h, w)
        annotated_frame: numpy array (h, w, 3)
    '''
            # backup id and origin-merged-mask
    bc_id = self.curr_idx
    bc_mask = self.origin_merged_mask
    
    # get annotated_frame and boxes
    # annotated_frame, boxes = self.detector.run_grounding(origin_frame, grounding_caption, box_threshold, text_threshold)
    for i in range(len(coords)):
        pt = coords[i].reshape([1,2])
        mode = modes[i]
        # if (bbox[1][0] - bbox[0][0]) * (bbox[1][1] - bbox[0][1]) > annotated_frame.shape[0] * annotated_frame.shape[1] * box_size_threshold:
        #     continue
        interactive_mask = self.sam.segment_with_click(origin_frame, pt, mode)

        mask = interactive_mask.copy()
        if(mask.sum(1).sum(0)/(mask.shape[0]*mask.shape[1]) < 0.01): 
            refined_merged_mask = self.add_mask(interactive_mask)
            self.update_origin_merged_mask(refined_merged_mask)
            self.curr_idx += 1

    # reset origin_mask
    self.reset_origin_merged_mask(bc_mask, bc_id)

    return refined_merged_mask

# 最终版：
def detect_and_seg_pt(self, frame: np.ndarray, coords, modes):
    '''
    Using Grounding-DINO to detect object acc Text-prompts
    Retrun:
        refined_merged_mask: numpy array (h, w)
        annotated_frame: numpy array (h, w, 3)
    '''
    # backup id and origin-merged-mask
    bc_id = self.curr_idx
    bc_mask = self.origin_merged_mask

    # get annotated_frame and boxes
    # annotated_frame, boxes = self.detector.run_grounding(frame, grounding_caption, box_threshold, text_threshold)
    cnum = len(coords)
    imgarea = frame.shape[0]*frame.shape[1]
    # ims = []
    refined_merged_mask = self.add_mask(np.zeros(frame.shape[:2], dtype=np.uint8))
    for k in range(cnum):
        # if (bbox[1][0] - bbox[0][0]) * (bbox[1][1] - bbox[0][1]) > annotated_frame.shape[0] * annotated_frame.shape[1] * box_size_threshold:
        #     continue
        interactive_mask = self.sam.segment_with_click(frame, coords[k:k+1], modes[k], True)
        # ims.append(interactive_mask.copy())
        
        if interactive_mask.sum(1).sum(0)/imgarea < 0.01:
            refined_merged_mask = self.add_mask(interactive_mask)  #在self.origin_merged_mask 的基础上，根据输入修改得到返回值。
            self.update_origin_merged_mask(refined_merged_mask)
            self.curr_idx += 1

    # reset origin_mask
    self.reset_origin_merged_mask(bc_mask, bc_id)

    return refined_merged_mask