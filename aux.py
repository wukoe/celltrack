import copy
import numpy as np
# from skimage import morphology as skmorph
from onevision import morphology as ovmorph
from onevision import morph_data,trajectory,improc
import proc_data
from misc_tool import seq_mine,dm
# from misc_tool.show import wshow


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
        
        ms = SA.infer(frame, point_prompt=point_prompt)
        frame_idt = ids[frame_idx]
        # 以下的部分可能不需要。
        # I = proc_data.filter_mask_by_size(ms, 0, segtracker_args['max_obj_area_ratio']) #segtracker_args['min_obj_area_ratio']
        # ms = ms[I]
        # frame_idt = frame_idt[I]
        ms = morph_data.IMbind(ms, 'masks', ids=frame_idt)
        pred_mask = isolate_filter_object(ms, segtracker_args, True)

        seg_list.append(pred_mask)
        print("segment for frame {}".format(frame_idx), end='\r')

    # 对于第一个segment map，若不遵循ids赋值，则可将ID连续分布，因为其作为tracking的起点。
    # seg_list[0] = morph_data.id_reset(seg_list[0])
    print('\n-segmentation done')
    return seg_list


# ===
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
            # if tid != nid:
            #     print('most overlap not {} but {}'.format(nid, tid))
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
                # print(nid, ', new seg obj too big')
                pass
        else:
            # print(nid, 'new')
            new_track_mask[nid] = seg_mask[nid]
            # new_track_mask.append(seg_mask[nid])
    return new_track_mask

def fix(vid, sms, SA, segtracker_args):
    # ===
    tr = trajectory.obj_traject(sms)
    miss_single, miss_seg = locate_missing(tr) # miss single
    fnum = len(vid)

    # ===
    print('>> start single missing fix')
    print(miss_single)
    for cellid in miss_single.keys():
        for fi in miss_single[cellid]:
            # generate segment using interpolated mask prompt.
            M1 = sms[fi-1][cellid]
            M2 = sms[fi+1][cellid]
            M_inter = ovmorph.interpolate_object(M1, M2)

            frame = vid[fi]
            Mp = SA.infer(frame, mask_prompt=[M_inter])[0]
            # filter
            # tp = ovmorph.isolate_object(Mp, True, keep_one=False)
            # if (tp!=Mp).any():
            #     print(np.unique(tp))
            Mp = ovmorph.filter_object_size(Mp, segtracker_args['min_obj_area_ratio'], segtracker_args['max_obj_area_ratio'])

            # fill-in generated mask.
            if (Mp!=0).any():
                sms[fi][cellid] = Mp.astype(bool)
            else:
                sms[fi][cellid] = M_inter

    # ===
    print('>> start gap filling')
    print(miss_seg)
    for cellid in miss_seg.keys():
        for gi in range(len(miss_seg[cellid])):
            gap = miss_seg[cellid][gi]
            # print(gap)
            stat = dm.copy_struct(gap)

            while True:
                gap_o = gap.copy()

                # forward extrapolate
                if stat[0] != 'stop':
                    fi = gap[0]
                    if fi-1 >= 0:
                        frame = vid[fi]
                        tp = sms[fi-1][cellid]
                        pp = ovmorph.mass_center([tp]) #pp is len 1 list.
                        Mp = SA.infer(frame, point_prompt=pp)[0]
                        # filt >>
                        Mp = ovmorph.filter_object_size(Mp, segtracker_args['min_obj_area_ratio'], segtracker_args['max_obj_area_ratio'])

                        # fill-in generated mask.
                        if (Mp!=0).any():
                            sms[fi][cellid] = Mp.astype(bool)
                            gap[0] += 1
                        else:
                            # sms[fi-1][cellid] = M_inter
                            stat[0] = 'stop'

                # backward extrapolate
                if stat[1] != 'stop':
                    fi = gap[-1]
                    if fi+1 <= fnum-1:
                        frame = vid[fi]
                        tp = sms[fi+1][cellid]
                        pp = ovmorph.mass_center([tp]) #pp is len 1 list.
                        Mp = SA.infer(frame, point_prompt=pp)[0]
                        # filt >>
                        Mp = ovmorph.filter_object_size(Mp, segtracker_args['min_obj_area_ratio'], segtracker_args['max_obj_area_ratio'])

                        # fill-in generated mask.
                        if (Mp!=0).any():
                            sms[fi][cellid] = Mp.astype(bool)
                            gap[-1] -= 1
                        else:
                            # sms[fi+1][cellid] = M_inter
                            stat[1] = 'stop'
                
                # judge stop.
                if gap[1]<=gap[0]:
                    break
                if gap == gap_o:
                    break
   
    return sms

def fix2(vid, sms, SA, segtracker_args):
    return

    
# ===
def locate_missing(tr):
    """
    detect missing data
    ---
    input
        tr : trajectory from trajectory.obj_traject(tms)
    ---
    output
        single_miss,seg_miss
    """
    single_miss = {}
    seg_miss = {}
    for cellid in tr.keys():
        ep = len(tr[cellid])-1 # ep = end point
        missed = seq_mine.locate_plateau([it is None for it in tr[cellid]])

        SI = []; SE = []
        for it in missed:
            if it[1]==it[0] and it[0]!=0 and it[1]!=(ep):
                SI.append(it[0])
            else:
                SE.append(it)
        if SI != []:
            single_miss[cellid] = SI
        if SE != []:
            seg_miss[cellid] = SE
    return single_miss,seg_miss
    
def extrapolate_mask(M1, M2, return_center=False):
    """ extrapolate along 2 masks.
    input M2 更接近要外推的对象
    --- return
    Mp : predicted mask
    mc : mask center
    """
    c1,c2 = ovmorph.mass_center([M1,M2])
    diff = c2 - c1
    Mp = improc.shift(M2, diff)
    if return_center:
        mc = c2 + diff
        return Mp, mc
    else:
        return Mp

### for processing of segmented masks.
def cell_avg_size(X:morph_data.IMbind):
    sz = []
    for id in X.ids:
        a = X[id]
        sz.append(a.sum())
    sz = np.array(sz)
    num,bin = np.histogram(sz, 30)

    tp = seq_mine.locate_updown(num)

    # 第一个下降沿和之后的第一个上升沿之间作为分界
    first_descend = tp['descend'][0]
    for it in tp['ascend']:
        if it[0]>=first_descend[1]:
            first_ascend = it
            break
    
    front = bin[first_descend[1]+1]
    tail = bin[first_ascend[0]+1]    
    thres = (front+tail)/2
    I = sz>=thres
    # print(I)
    mz = np.mean(sz[I])
    return mz

def determine_dup(e:morph_data.IMbind):
    '''根据2个物体的mask是否有一定比例上的重叠判断是否duplicate'''
    L = []
    for m in range(len(e)):
        for n in range(m+1, len(e)):
            intersect = e[e.ids[m]] & e[e.ids[n]]
            union = e[e.ids[m]] | e[e.ids[n]]
            ss = intersect.sum()
            iou = [ss/e[e.ids[m]].sum(), ss/e[e.ids[n]].sum()]
            if max(iou) > 0.9:
                L.append([e.ids[m], e.ids[n]])
    
    return L


import skimage.segmentation as skseg
import skimage.measure as skmeasure
import os,math

def find_correspond_dapi_file(fn, nuclear_mask_dir):
    mark = fn.rstrip('.png').rsplit('_', 1)[1]
    fn = fn.split('_', 1)[0] # base part
    if mark == 'ne4c':
        fn_nuc = fn + '_02_2_1_DAPI_001_ne4c_cp_masks.png'
    else:
        fn_nuc = fn + '_02_2_1_DAPI_001_cp_masks.png'
    fn_nuc = os.path.join(nuclear_mask_dir, fn_nuc)
    return fn_nuc, mark

def detect_fake_nuc(x):
    '''detect fake
    Return
    ---
    idx : object ids from x that deemed fake.
    '''
    # area and perimeter of each object.
    area = ovmorph.object_area(x)
    area = np.array(list(area.values()))
    peri = np.array([skmeasure.perimeter(x[it]) for it in x])

    # circularity.
    c = 4*math.pi*area/(peri**2)
    # plt.scatter(area, c)
    I = np.bitwise_and(c<0.75, area<200) #圆度小于0.75，面积小于200的，被定为不是细胞核。【面积的数值应该更灵活】

    idx = np.nonzero(I)[0]
    if len(idx)>0:
        idx = np.array(x.get_ids())[idx]
    return idx

def process_according_nuclear_mask(C, K):
    # C is cell mask map, K is nuclear mask map.

    # filter nuclear mask
    idx = detect_fake_nuc(K)
    del K[idx]

    kc = ovmorph.mass_center(K) # mass center
    kc = kc.astype('int')
    M = morph_data.imbind_to_map(C)

    # 将obj mask分为有核和无核。
    have_nuc = []
    for tp in kc:
        if M[tp[0], tp[1]] > 0:
            have_nuc.append(M[tp[0], tp[1]]) # record the ID
    no_nuc = set(C.ids) - set(have_nuc)

    # 将无核obj合并入邻接的cell的mask，没有的则删除。
    # - 这个过程要以多轮迭代的方法进行，因为一个mask可能其相邻的mask也是无核的，要先合并。循环直至不再有合并事件发生为止。
    flag = True
    while flag:
        flag = False
        for idx in no_nuc:
            # whether the object contact with another object with cell nuclear.
            for it in have_nuc:
                if ovmorph.is_contact(C[idx], C[it]):
                    C[it] = C[it] | C[idx] #融合到有核的细胞mask
                    del C[idx] #删除原本无核的
                    flag = True
                    break
        no_nuc = set(C.ids) - set(have_nuc)

    # remove remaining:
    for it in no_nuc:
        del C[it]

    # one more round of fill hole.
    C = ovmorph.fill_holes(C)

    # to do split
    cell_nuc_list = get_cell_nuc_list(C, kc)
    if cell_nuc_list is not None:
        C = splitm(cell_nuc_list, C)

    return C

def get_cell_nuc_list(X, kc):
    '''
    确定每个细胞mask内的cell nuclear的数量。

    Input
    ---
    X: IMbind obj of cell masks.
    kc: list of nuclear center coordinates.

    Return
    ---
    R: dict of cellid:list of nuclear coordinates.
    '''
    if len(kc) == len(X):
        return None

    M = morph_data.imbind_to_map(X)
    R = {idx:[] for idx in X.ids}
    for it in kc:
        idx = M[it[0], it[1]]
        if idx > 0:
            R[idx].append(it)
    return R

def splitm(R, A):
    '''
    使用了分水岭算法，将一个细胞分成多个部分。
    '''
    for idx in R.keys():
        pnum = len(R[idx])
        if pnum>1:
            # print(idx, R[idx])
            marker = np.zeros(A.shape)
            for k in range(pnum):
                it = R[idx][k]
                marker[it[0], it[1]] = k+1
            tpM = skseg.watershed(A[idx], marker, mask=A[idx])

            A[idx] = tpM==1 
            for k in range(1,pnum):
                A.append(tpM==(k+1))
    return A

