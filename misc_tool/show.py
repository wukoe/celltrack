# misc visulizations
import os, itertools #time
import numpy as np
try:
    import torch
except:
    print('warning: torch not installed.')
import cv2
from matplotlib import pyplot as plt
from matplotlib import patches
from PIL import Image,ImageDraw,ImageFont
from skimage.morphology import binary 

from voneum.utils import util
import wbtool
from wbtool import dm


# === easy show tool
def wshow(X, r=0, subplot=False, *args, **kwargs):
    """ convenient image show
    input
    ------
        r: range to which data are adjusted, typical: 1 or 255.
    """
    plt.figure()

    if subplot:
        assert isinstance(X, (list, tuple))

        fig_num = len(X)
        for k in range(fig_num):
            X[k] = proc_for_wshow(X[k], r)

            plt.subplot(1, fig_num, k+1)
            plt.imshow(X[k], *args, **kwargs)

    else:
        X = proc_for_wshow(X, r)
        plt.imshow(X, *args, **kwargs)

    plt.show()

def proc_for_wshow(X, r):
    if isinstance(X, Image.Image):
        pass
    else:
        if isinstance(X, torch.Tensor):
            X = util.to_numpy(X)
        ds = X.shape
        if len(ds)==3 and ds[0]==3:
            X = np.transpose(X, (1,2,0))

        if r==0:
            pass
        elif r==1:
            X = dm.rescale(X)
        elif r==255:
            X = dm.rescale(X, outscale=[0,255])
        else:
            raise

    return X

def wplot(*args, r=0, subplot=False, show=True, **kwargs):
    """ convenient signal show
    input
    -----
        Y: each row as a line.
        r: numerical range of data, 0=original, 1=to [0,1], 255=to[0,255]
        subplot=True to plot each channel(row) in individual subplot
    """
    if len(args)>1:
        X = args[0]
        Y = args[1]
    else:
        Y = args[0]

    if isinstance(Y, torch.Tensor):
        Y = util.to_numpy(Y)
    if isinstance(Y, list):
        Y = np.array(Y)

    if r==0:
        pass
    elif r==1:
        Y = dm.rescale(Y)
    elif r==255:
        Y = dm.rescale(Y, outscale=[0,255])
    else:
        raise

    Y = Y.transpose() # plt.plot set each column as a line, so convert.

    # plt.figure()
    if subplot:
        # get signal STD range
        ss = np.std(Y, 0)
        ss = ss.mean()

        ch_num = Y.shape[1]
        if len(args)==1:
            X = range(len(Y))
        for chi in range(ch_num):
            plt.subplot(ch_num, 1, chi+1)
            plt.plot(X, Y[:, chi], **kwargs)
            # [equal axis issue to be solved]
        pr = None

    else:
        args = list(args)
        if len(args)>1:
            args[1] = Y
        else:
            args[0] = Y
        args = tuple(args)

        pr = plt.plot(*args, **kwargs)

    if show:
        plt.show()
    return pr

def wplotvl(ins, r=0, show=True, *args, **kwargs):
    # the version for list of vectors with various length
    # plt.figure()
    for si in range(len(ins)):
        X = ins[si]
        if r==0:
            pass
        elif r==1:
            X = dm.rescale(X)
        elif r==255:
            X = dm.rescale(X, outscale=[0,255])
        else:
            raise

        plt.plot(X, *args, **kwargs)

    if show:
        plt.show()


# === mark image.
color_table = {'r':(204, 0, 0), 'g':(0, 204, 0), 'b':(0, 0, 204), 'y':(255,255,0), 'aqua':(0,255,255), \
'magenta':(255,0,255), 'orange':(255,165,0), 'pink':(255,192,203), 'sky':(0,191,255)}  # color = (100,100,255)
font_path = os.path.join(cv2.__path__[0],'qt','fonts','DejaVuSans.ttf') #现成的字体文件

# ===
def overlay_text(image, coords, texts, font_size=30, text_color=(255, 255, 255)):
    """
    overlay multiple texts.
    
    Input
    ---
    coords: list of coord in y-x order
    """
    image = Image.fromarray(image)# 白色
    # font=ImageFont.load_default()
    font = ImageFont.truetype(font_path, size=font_size) #, color=text_color
    draw = ImageDraw.Draw(image) # 创建一个可以在上面绘制的对象

    tnum = len(texts)
    for ti in range(tnum):
        xy = [coords[ti][1]-font_size/2, coords[ti][0]-font_size/2]
        draw.text(xy, texts[ti], font=font, fill=text_color) #draw.text 输入坐标规定x-y顺序。
    return np.array(image)

def overlay_boxtext(img, box_info, opt='t', font_size=None):
    """
    for detection result visualization.
    draw boxes and related text (location defined by coordinates) on image.
    input
        img: Image object or numpy array.
        box_info: {level1:[[左上x, 左上y, 右下x，右下y, score, color],[]], level2:[]}
    """
    # tnum = len(box_info)
    show_box = False; show_text = False
    for k in opt:
        if k=='b':
            show_box = True
        elif k=='t':
            show_text = True
    dynamic_fs = (font_size is None)
    font_file = os.path.join(wbtool.datadir, 'simsun.ttc')
    if not dynamic_fs:
        font = ImageFont.truetype(font_file, size=font_size, encoding='utf-8')

    # 方法1 基于PIL ImageDraw
    if isinstance(img, Image.Image):
        img = img.copy()
    elif isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    else:
        raise

    obj = ImageDraw.Draw(img)
    color = color_table['r']
    
    for bt in box_info:
        loc = bt[0]
        text = bt[1]

        if show_box:
            # obj.rectangle(loc[0]+loc[2], outline='red')
            # 采用四线段允许倾斜
            obj.line(loc[0]+loc[1], fill='red')
            obj.line(loc[1]+loc[2], fill='red')
            obj.line(loc[2]+loc[3], fill='red')
            obj.line(loc[3]+loc[0], fill='red')

        if show_text:
            # dynamic font size.
            if dynamic_fs:
                fs = (loc[-1][1] - loc[0][1])*0.75
                fs = int(dm.value_clip(fs, [5, 50]))
                font = ImageFont.truetype(font_file, size=fs, encoding='utf-8') #'arial', size=fs)
            obj.text(loc[0], text, color, font=font)

    return img
    """
    # 方法2 基于cv  putText - 但是目前在服务器系统上无法正常显示中文
    img = np.ascontiguousarray(img)
    for level, boxes in box_info.items(): # level is the dict key
        for box in boxes:
            cl = len(box)
            if cl > 0:
                bb = tuple(int(np.floor(x)) for x in i[:4])
                score = box[-2] if cl>=5 else ''
                color = color_table[box[-1]] if cl>=6 else color_table['r']
                cv2.rectangle(img, bb[0:2], bb[2:4], color, 2)
                cv2.putText(img, '%s: %.3f' % (level, score), (bb[0], bb[1] + 15),
                            cv2.FONT_HERSHEY_PLAIN, 1.0, color, thickness = 1)
    return img
    """

def overlay_mask(image, masks, ids=None, alpha=0.8, apply_countour=False):
    """
    Overlay colored binary mask
    --- input
    image : RGB channel image tensor.
    masks : list of binary masks.
    ids: id accompany each mask, 用于保持某个id的mask的色彩等显示属性的一致性。
    """
    # colors = np.atleast_2d(colors) * cscale  # View inputs as arrays with at least two dimensions.
    # - cscale default =1 
    n_mask = len(masks)

    # decide colors.
    if ids is None:
        ct = itertools.cycle(color_table.values())
        colors = [np.array(next(ct)) for k in range(n_mask)]
    else:
        ct = list(color_table.values())
        color_num = len(ct)
        colors = [np.array(ct[int(ids[k])%color_num]) for k in range(n_mask)]

    im_overlay = image.copy()
    for m_id in range(n_mask):
        foreground = (image * alpha + np.ones(image.shape) * (1 - alpha) * colors[m_id])
        foreground = foreground.astype('uint8')
        binary_mask = masks[m_id].astype('bool')

        # Compose image
        im_overlay[binary_mask] = foreground[binary_mask]
        
        if apply_countour:
            countours = binary.binary_dilation(binary_mask) ^ binary_mask
            im_overlay[countours, :] = 0

    return im_overlay.astype(image.dtype)

def overlay_point(image, coords):
    # coords : [(y,x), ...]
    ax = plt.gca()
    ax.imshow(image)
    add_point(ax, coords)
    plt.show()
    
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


# ===
def add_mark(type, *args):
    args = list(args)
    y,x = args[0]
    args[0] = x,y

    if type=='circle':     
        p = patches.Circle(*args)

    ax = plt.gca()
    ax.add_patch(p)

def add_point(ax, coords, color='red', marker='*', marker_size=10):
    # pos_points = coords[labels==1]
    # neg_points = coords[labels==0]
    # ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    # ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(coords[:,1], coords[:,0], color=color, marker=marker, s=marker_size) # ax.scatter take coord in (x,y), so input need to switch order.

def add_box(ax, box):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    
    

# === sequence and compare view.
from matplotlib.widgets import TextBox, Button

def seq_image(X):
    """
    view a sequence of images, UI with navigation buttons and text box.
    input
        X: 4D array
    In cmd window, press Enter to next image/slice, press 3+Enter to previous,
    c+Enter to quit.
    """
    imgnum = len(X)
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.2)

    class Callbacks:
        def __init__(self, maxnum, idx_init):
            self.maxnum = maxnum
            self.idx = idx_init            

        def drawit(self):
            text_box.set_val(self.idx)
            ax.imshow(X[self.idx])
            plt.draw()

        def next(self):
            if self.idx < self.maxnum-1:
                self.idx += 1
                self.drawit()          

        def prev(self):
            if self.idx > 0:
                self.idx -= 1
                self.drawit()

        def submit(self, text):
            idx = int(text)
            if idx<0 or idx>=self.maxnum:
                plt.title('index over range')
            else:
                plt.title('')
                self.idx = idx
                self.drawit()

    callback = Callbacks(imgnum, 0)

    axprev = fig.add_axes([0.6, 0.01, 0.1, 0.05])
    axnext = fig.add_axes([0.82, 0.01, 0.1, 0.05])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(lambda event: callback.next())
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(lambda event: callback.prev())
    axbox = fig.add_axes([0.71, 0.01, 0.1, 0.05])
    text_box = TextBox(axbox, '')
    text_box.on_submit(lambda event: callback.submit(text_box.text))

    callback.drawit()
    plt.show()

def format_data(X, opt, channel=-1):
    # format data for compare()
    if opt=='normal':
        pass
    elif opt=='transpose':
        X = np.transpose(X, (0,2,3,1))
    elif opt=='one': # one channel
        X = X[:,channel,:,:]
    else:
        raise Exception
    return X

def compare(*args):
    """
    compare mulitple stacks of images, align equally indexed layers side by side.
    originally for usage in Liver CT.
    """
    dnum = len(args)
    dim = len(args[0].shape)
    if dim==4:
        for di in range(dnum):
            ds = args[di].shape

            if ds[-1]==3:
                ch = 3
                args[di] = np.transpose(args[k], (0,3,1,2))
            elif ds[1]==3:
                ch = 3
            else:
                ch = 1

    # concatenate stacks and use seq_image() directly.
    tp = np.concatenate(args, -1)
    if dim==3:
        seq_image(tp)
    else:
        if ch==3:
            seq_image(format_data(tp,'transpose'))
        else:
            seq_image(format_data(tp,'one'))
