'''data processing/managing misc'''

import numpy as np
import numbers
import math
# try:
#     import torch
#     has_torch = True
# except:
#     has_torch = False


# === basic python object extended utility
def copy_struct(obj):
    """
    Create list the same structure of existing iterable object (mostly list)
    目标用于伴随性的数据结构，以存储对应位置数据相关的属性、标记等。
    """
    if hasattr(obj, '__iter__'):
        if isinstance(obj, dict):
            return {k:None for k in obj.keys()}
        elif isinstance(obj, str):
            return None
        else:
            return [copy_struct(obj[k]) for k in range(len(obj))]
    else:
        return None

def poplist(L, idx):
    """
    remove multiple elements in a list, elements specified by list of indexs.
    """
    idx = list(idx)
    idx.sort(reverse=True)
    for k in range(len(idx)):
        L.pop(idx[k])
    return L

def filt(X, L):
    """filt X according to L
    X, L be of equal length.
    """
    Y = [X[k] for k in range(len(X)) if L[k]]
    return Y

DC_OP = {
'+': lambda a,b: a+b, 
'-': lambda a,b: a-b,
}
def dict_calc(D1, D2, op):
    '''
    计算2个dict之间匹配上的key对应的value之间进行计算。
    结果是基于D1的value改变并生成新dict。
    '''
    func = DC_OP[op]

    nd = D1.copy()
    for k,v in D2.items():
        if k in D1:
            nd[k] = func(D1[k], D2[k])
    return nd


# === obtain information
# locate elements
def recursive_search(X, tt):
    """
    Find index of all occurance of a target variable.
    """
    return [i for i,j in enumerate(X) if j==tar]

def find(X, tt):
    """ return index of tt in X"""
    if isinstance(X, np.ndarray):
        R = np.where(X==tt)
        R = np.stack(R).transpose()
        #不过如果X是1D，R依然是[N,1]形状的。
    else:
        R = recursive_search(X, tt)
    return R

def closest(collection, num):
    # find closed in a list to number 
    return min(collection, key=lambda x:abs(x-num))

# get general info.
def cellop(X, opt=None, attr=None, *args):
    """
    Do operations to each element in an iterable.
    input
        attr: a string of the attribute to use.
        opt: 'func' to do it.func()
             'attr' to do it.attr

    * 对于要对元素进行func(it)形式的，可 list(map(func, X))
    """
    if opt == 'func':
        return [getattr(it, attr)(*args) for it in X]
    elif opt == 'attr':
        return [getattr(it, attr) for it in X]
    else:
        # raise AssertionError('invalid option')
        return [type(it) for it in X]


# === make data to new shape or structure. (mostly array or complex data)
def to_onehot(X:list, num_classes:int) -> list:
    """
    to one-hot representation. can handle if None in X (all zero encoding)
    input
        X: now only support 1D array.
        num_classes: the code length.    
    """
    # tp = set(X)
    # if None in tp:
    #     tp.remove(None)
    assert max(tp)<num_classes, 'X elements > set num_classes'

    D = np.eye(num_classes)
    
    dlen = len(X)
    Y = np.zeros((dlen, num_classes))

    I = [k for k in range(dlen) if X[k] != None]
    for k in dlen:
        Y[k] = D[X[k]]
    return Y

# def to_onehot(X, num_classes=None):
#     """
#     to one-hot representation (另一种实现方法，可自动检测label的数值范围)
#     input
#         num_classes: the code length.
#     """
#     minlabel = X.min()
#     if num_classes==None:
#         num_classes = X.max() # -minlabel+1
#     ds = X.shape
#     dsy = list(ds)
#     dsy.append(num_classes)
#     Y = np.zeros(dsy, dtype=np.uint8)
#     # print(Y.shape)
#     for k in range(num_classes):
#         temp = np.zeros(ds, dtype=np.uint8)
#         temp[X==(minlabel+k)] = 1
#         Y[:,:,:,k] = temp
#     return Y

def add_one_hot_channel(X, num_class, dim):
    """THis one is for array of any shape"""
    Y = []
    for k in range(num_class):
        tp = X==k
        Y.append(tp.astype(np.int))
    Y = np.stack(Y, dim)
    return Y

def str_get_column(X, colid, dechar=' '):
    """ 
    Get one "column" of list of strings. column is defined by segment in string separated by dechar.
    input
        colid: column id, could be list of or single index.
    """
    X = [it.split(dechar) for it in X] # listsep will not separate if it already have multiple columns

    if isinstance(colid, numbers.Number):
        flagList = False
    else:
        flagList = True

    if flagList:
        Y = [[it[k] for k in colid] for it in X]
    else:
        Y = [it[colid] for it in X]
    return Y

def reabylb(X):
    """
    re-arrange by label
    """
    xlen=len(X)
    if xlen==0:
        return

    ### Skim the label list for label types
    lbl=[X[0]];# label list: list for different labels
    tAmt=1;
    for m in range(1,xlen):
        # check if current label already exist in label list
        bExist=False
        for n in range(tAmt):
            if X[m]==lbl[n]:
                bExist=True
                break
        if not bExist: # if not, add it to the label list
            tAmt+=1
            lbl.append(X[m])

    # now lbl contain all the value of all kind of labels in lb
    lbl.sort(); # mentain the ascending order of different labels,

    ###
    typeAmt=newlist(tAmt)
    idx=newlist(tAmt)
    for m in range(tAmt):
        idx[m]=[i for i,x in enumerate(X) if x==lbl[m]]
        typeAmt[m]=len(idx[m])

    L={'tAmt':tAmt,'types':lbl,'typeAmt':typeAmt,'idx':idx}
    return L

def cutseg(srange, seglen, interlen=None, bIdxSeg=True, bEqual=False) -> list:
    """
    Cut segments
    input
        srange: list of 2 number, total range for segmentation.
        interlen: if !=None: 段与段之间重叠, 步长为interlen.
        bIdxSeg: 采用Index形式的分段。 Index形式：所得值皆为整数，且segm前一段尾和后一段头差1.
    """
    if not isinstance(srange, (list, tuple)):
        srange = [0, srange-1]

    if bIdxSeg:
        xlen = srange[1]-srange[0]+1
    else:
        xlen = srange[1]-srange[0]

    # Check seglen and xlen input
    if bIdxSeg and seglen<1:
        print('segment length smaller than 1')
        return

    if xlen<=seglen:
        return [list(srange)]

    if interlen!=None: ### Overlap bins
        segAmt = (xlen-seglen)//interlen + 1
        if (xlen-seglen)%interlen > 0:
            segAmt += 1

        # Filling the segm
        if bIdxSeg:
            segm = [[interlen*k + srange[0], interlen*k + seglen-1+srange[0]] for k in range(segAmt)]
            segm[segAmt-1][1] = srange[1]
        else:
            segm = [[interlen*k + srange[0], interlen*k + seglen+srange[0]] for k in range(segAmt)]
            segm[segAmt-1][1] = srange[1]

    else:  ### no overlap scheme
        segAmt = xlen//seglen
        if xlen%seglen > 0:
            segAmt += 1
        # Filling the segm
        if bIdxSeg:
            segm = [[seglen*k+srange[0], seglen*(k+1)-1+srange[0]] for k in range(segAmt)]
            segm[segAmt-1][1] = srange[1]
        else:
            segm = [[seglen*k+srange[0], seglen*(k+1)+srange[0]] for k in range(segAmt)]
            segm[segAmt-1][1] = srange[1]

    if bEqual:
        segm[segAmt-1][0] = segm[segAmt-1][1] - seglen + 1
    return segm

def strsplit(S, seplist):
    # split string with multiple character option
    # 也可以用re.split()实现
    assert not isinstance(S, str)

    sn = len(seplist)
    if sn == 0:
        return S.split('')

    sep = seplist[0]
    for k in range(1,sn):
        S = S.replace(seplist[k], sep)

    return S.split(sep)


# === modify content of data object.
def rm_diag(X):
    """
    Remove the diagonal element of matrix
    now assume input a numpy array
    """
    dlen = min(X.shape)
    for k in range(dlen):
        X[k,k]=0
    return X

def value_clip(x, r):
    """
    limite value of x in range r
    """
    if isinstance(x, list):
        if r[0]!=[]:
            x = list(map(lambda x:max(x,r[0]), x))
        if r[1]!=[]:
            x = list(map(lambda x:min(x,r[1]), x))
    else:    
        if r[0]!=None:
            x = np.maximum(x, r[0])
        if r[1]!=None:
            x = np.minimum(x, r[1])

    return x

def rescale(X, inscale=None, outscale=[0,1]):
    """
    value range re-scale
      rescale(X, inscale=None, outscale=[0,1]):
    X outside inscale is constrained to inscale
    The output is projected to outscale
    output is type 'float32'.
    """
    X = np.array(X, dtype='float32')
    if inscale == None:
        inscale = [X.min(), X.max()]
    else:
        X[X<inscale[0]] = inscale[0]
        X[X>inscale[1]] = inscale[1]

    X -= inscale[0]
    X /= inscale[1]-inscale[0]
    # now X in [0,1]

    if outscale != [0,1]:
        X *= outscale[1]- outscale[0]
        X += outscale[0]

    return X

# 直角坐标和极坐标的互相转换。
def polar_to_cartesian(r, theta):
    return r * math.cos(theta), r * math.sin(theta)

def cartesian_to_polar(x, y):
    r = math.sqrt(x**2 + y**2)
    theta = math.atan2(y, x)
    return r, theta
    