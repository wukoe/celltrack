''' sequence data mining and feature extraction'''
import numpy as np


# === for value sequence - related to shape of curve.
### monotonic check (following 4)
# - input x is limited to 1D data.
def is_ascend(x):
    return all(a<b for a,b in zip(x, x[1:]))
    
def is_descend(x):
    return all(a>b for a,b in zip(x, x[1:]))
    
def is_monoup(x):
    return all(a<=b for a,b in zip(x, x[1:]))
    
def is_monodown(x):
    return all(a>=b for a,b in zip(x, x[1:]))

def locate_updown(X):
    '''
    所有的上升、平、下降段的列表
    '''
    X = np.array(X)
    P = X[1:] - X[:-1] # 后一个减前一个
    up = locate_plateau(P>0)
    for k in range(len(up)):
        up[k][1] += 1
    flat = locate_plateau(P==0)
    for k in range(len(flat)):
        flat[k][1] += 1
    down = locate_plateau(P<0)
    for k in range(len(down)):
        down[k][1] += 1
    return {'ascend':up, 'flat':flat, 'descend':down}

def locate_minima(X):
    return

def locate_maxima(X):
    return

### locate trend key points for binary sequence.
def locate_plateau(X):
    """
    find segments of continuous positive values in a series of boolean values.
    equivalent to find all "plateau" in binary data.
    ---
    input:
        X : must be boolean coding.
    ---
    output:
        P : structure [[ue,de],[ue,de],...]
    """
    # snum = len(X)
    temp = np.array(X) != 0
    if len(temp) == 0:
        return []
    else:
        R = locate_edge(temp, include_end=True)
        ups = R['up_edge']
        downs = R['down_edge']
        P = []
        for k in range(len(ups)):
            P.append([ups[k], downs[k]])
        return P

def locate_edge(X, include_end=False):
    """
    检测到的坐标在高台内侧——即，对于上升缘，坐标在阶跃之后；对于下降缘，坐标在阶跃之前。
    """
    P = np.stack([X[:-1], X[1:]]).astype(bool)
    P = P.transpose()
    up_edge = [all(it==[False, True]) for it in P]
    down_edge = [all(it==[True, False]) for it in P]
    up_index = np.nonzero(up_edge)[0] + 1
    down_index = np.nonzero(down_edge)[0]
    if include_end:
        if X[0]:
            up_index = np.concatenate([[0], up_index])
        if X[-1]:
            slen = len(X)
            down_index = np.append(down_index, slen-1)
    return {'up_edge':up_index, 'down_edge':down_index}

def locate_ascend(X, include_end=False):
    pass


# === for text string sequence.
def common_substr(seq1,seq2,bAllSub=False):
    """
    Get Common sub-string of 2 seq
    func lists common sub-string. Options for whether list all the possible substr
    """
    if type(seq1)==str:
        if type(seq2)!=str:
            print('error: type dismatch!')
            return
        bStr=True;
    else:
        bStr=False;
        
    sl1=len(seq1); sl2=len(seq2)
    
    # matrix for common items between 2 sequences
    cm=newlist(sl1,sl2)
    for m in range(sl1):
        for n in range(sl2):
            if seq1[m]==seq2[n]:
                cm[m][n]=1
                
    cslist=[]
    for m in range(sl1):
        #print m
        for n in range(sl2):            
            if cm[m][n]==1:
                # for recording the temporal sequence
                if bStr:
                    temp=''
                else:
                    temp=[]
                
                # search for the diagonal string
                loc1=m; loc2=n
                tp=cm[loc1][loc2]
                if bAllSub:
                    while tp==1:
                        #cm[loc1][loc2]=0 #remove the counted block 
                        if bStr:
                            temp+=seq1[loc1]
                        else:
                            temp.append(seq1[loc1])
                        loc1+=1; loc2+=1
                        if loc1>=sl1 or loc2>=sl2:
                            break
                        else:
                            tp=cm[loc1][loc2]
                    
                    # add all the subs of the common string
                    for si in range(len(temp)):
                        tp=temp[:si+1]
                        # if not in cslist, add to it
                        bInCS=False
                        for k in range(len(cslist)):
                            if cslist[k]==tp:
                                bInCS=True
                                break
                                
                        if bInCS==False:
                            cslist.append(tp)
                
                # else if not bAllSub
                else:
                    while tp==1:
                        #cm[loc1][loc2]=0 #remove the counted block 
                        if bStr:
                            temp+=seq1[loc1]
                        else:
                            temp.append(seq1[loc1])
                        cm[loc1][loc2]=0; # efface visited pixel mark
                        loc1+=1; loc2+=1
                        if loc1>=sl1 or loc2>=sl2:
                            break
                        else:
                            tp=cm[loc1][loc2]
                    cslist.append(temp)
    
    return cslist
