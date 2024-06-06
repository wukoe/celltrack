# -*- coding: utf-8 -*-
"""
Created on Thu May 12 12:28:38 2016
File read/write operation for misc types
Note image file code is in repository "onevision"
@author: wb
"""

import os,fnmatch,numbers
import numpy as np
from PIL import Image
import csv
import codecs
import re
import pickle


###### File information
def search(tpl, path='.', ftype='file', pattern='wildcard', basename=False):
    """
    提供glob.glob()之外的选项 
    File list match by wildcard or RE pattern
    input
    --------
        ftype: 'file', 'dir', 'all'
        pattern: 'wildcard', 're'
        basename: whether output list with path basename attached.
    """
    fl = os.listdir(path)

    if ftype=='file':
        fl = [x for x in fl if os.path.isfile(os.path.join(path, x))]
    elif ftype=='dir':
        fl = [x for x in fl if os.path.isdir(os.path.join(path, x))]
    elif ftype=='all':
        pass
    else:
        raise Exception('invalid ftype')

    if pattern=='wildcard':
        if isinstance(tpl, (list, tuple)):
            fl = [x for x in fl if any([fnmatch.fnmatch(x, k) for k in tpl])]
        else:
            fl = fnmatch.filter(fl, tpl)
    elif pattern=='re':
        fl = [x for x in fl if re.match(tpl, x)]
    else:
        raise Exception('invalid template pattern type')

    if not basename:
        fl = [os.path.join(path, x) for x in fl]

    # print('targeted file number:', len(fl))
    return fl


###### General file type save and load
# general using pickle
def pload(fname):
    with open(fname, 'rb') as fh:
        X = pickle.load(fh)
        fh.close()
    return X

def pdump(fname, *args, **kwargs):
    b_posarg = True if args else False
    b_kwarg = True if kwargs else False

    # 若args只有一个，取消tuple形式。
    if b_posarg:
        if len(args) == 1:
            args = args[0]
    
    # 以下是为了保证，若两种参数只有一种存在时，减少obj的嵌套。
    if b_posarg and b_kwarg:
        obj = [args, kwargs]
    elif b_posarg and not b_kwarg:
        obj = args
    elif not b_posarg and b_kwarg:
        obj = kwargs
    else: # no input
        return

    with open(fname, 'wb') as fh:
        pickle.dump(obj, fh)
        fh.close()

# HDF5 file
def h5info(fname, field_name=None):
    import h5py

    if os.path.isfile(fname):
        with h5py.File(fname, 'r') as fh:
            obj = fh if field_name==None else fh[field_name]
            if isinstance(obj, h5py._hl.group.Group):
                IN = [k for k in obj]
            elif isinstance(obj, h5py._hl.dataset.Dataset):
                IN = obj.shape

            fh.close()
    else:
        print('the file does not exist')
        return
    return IN

def h5read(fname, field_name):
    import h5py

    if os.path.isfile(fname):
        # number of items to read
        if isinstance(field_name, str):
            flagList = False
        else:
            if hasattr(field_name, '__iter__'):
                numset = len(field_name)
                flagList = True
            else:
                raise Exception('field_name name not supported')

        with h5py.File(fname, 'r') as fh:
            if flagList:
                x = []
                for k in range(numset):
                    x.append(np.array(fh[field_name[k]]))
            else:
                x = np.array(fh[field_name])
            fh.close()
    else:
        print('the file does not exist')
        return
    return x

def h5write(fname, field_name, data, checkFile=True):
    import h5py

    flagOverwrite = True
    if checkFile and os.path.isfile(fname):
        s=input('the file already exist, overwrite?[y/n/r/a]:')
        if s=='y':
            pass
        elif s=='a':
            print('append')
            flagOverwrite = False
        elif s=='n':
            print('abort')
            return
        elif s=='r':
            s = input('rename the file: ')
            if s[-3:]!='.h5':
                fname = s+'.h5'
            else:
                fname = s
        else:
            print('invalid option')
            return

    fo = 'w' if flagOverwrite else 'a'
    if isinstance(field_name, str):
        with h5py.File(fname, fo) as fh:
            fh.create_dataset(field_name, data=data)
            fh.close()
    else:
        if hasattr(field_name, '__iter__'):
            numset = len(field_name)
            assert len(data)==numset, 'data and field_name name number not match'
        else:
            raise Exception('field_name variable not recognized')
        
        with h5py.File(fname, fo) as fh:
            for k in range(numset):
                fh.create_dataset(field_name[k], data=data[k])
            fh.close()

    return


# ZIP file
def zip_read(filename):
    """read zip file"""
    import zipfile

    f = zipfile.ZipFile(filename)
    X = []
    for name in f.namelist():
        X.append(f.read(name))
    f.close()
    return X


###### Image files
def imread(file):
    # regular common picture format read to array.
    with Image.open(file) as im:
        nd = np.array(im)
    im.close()

    ''' 另一种选项
    import scipy.misc as sm
    nd = sm.imread(fn)
    '''
    return nd

def imwrite(file, x):
    if not isinstance(x, Image.Image):
        x = Image.fromarray(x)
    x.save(file)

def imgtiff_read(fileName, idx=None):
    """ 
    read TIFF image with multi-page. (single page TIFF can use imread())
    idx could be a list of intended pages.
    """
    with Image.open(fileName) as im:
        n_frame = im.n_frames # number of pages in tiff

        # Preprocessing.
        if idx is None:
            idx = range(n_frame)
        else:
            if isinstance(idx, numbers.Number):
                idx = [idx]        
            if min(idx)<0 or max(idx)>=n_frame:
                raise Exception('idx out of range 0-{}'.format(n_frame))
        n_idx = len(idx)

        # Load in the data.
        try:
            X = []
            for k in range(n_idx):
                im.seek(idx[k])
                X.append(np.array(im))
        except:
            print('can not fetch page %d'%idx[k])
            return

        if len(idx)==1:
            X = X[0]
        else:
            X = np.array(X)

    return X

def imgtiff_write(fileName, X, photometric=2, compression=0):
    """
    The photometric argument specifies the color space of the image data when writing a TIFF file. The following values are commonly used:
    0: WhiteIsZero. For bilevel and grayscale images: 0 is imaged as white.
    1: BlackIsZero. For bilevel and grayscale images: 0 is imaged as black.
    2: RGB. RGB value of (0,0,0) represents black, and (255,255,255) represents white, assuming 8-bit components. The components are stored in the indicated order: first Red, then Green, then Blue.
    3: Palette color. In this model, a color is described with a single component. The value of the component is used as an index into the red, green and blue curves in the ColorMap field to retrieve an RGB triplet that defines the color. When PhotometricInterpretation=3 is used, ColorMap must be present and SamplesPerPixel must be 1.
    4: Transparency Mask. This means that the image is used to define an irregularly shaped region of another image in the same TIFF file. SamplesPerPixel and BitsPerSample must be 1. PackBits compression is recommended. The 1-bits define the interior of the region; the 0-bits define the exterior of the region.
    5: Separated, usually CMYK.
    6: YCbCr.
    8: CIE Lab*.
    9: CIE Lab*, alternate encoding also known as ICC Lab*.
    10: CIE Lab*, alternate encoding also known as ITU Lab*, defined in ITU-T Rec. T.42, used in the TIFF-F and TIFF-FX standard.

    The compress argument = 0 means that the data is written uncompressed.
    """
    import tifffile

    # Write the 3D array to a multi-frame TIFF file
    tifffile.imwrite(fileName, X, photometric=photometric, compression=compression)


def multi_img_load(path=None, size=None):
    """
    Load multiple images
    """
    import cv2
    from skimage import transform

    if path==None:
        import tkinter.filedialog
        path = list(tkinter.filedialog.askopenfilenames())
    path.sort()

    fnum = len(path)    
    X=[]
    for k in range(fnum):
        X.append(cv2.imread(path[k]))
    print('file read done')
    
    if size is not None:
        for k in range(len(X)):
            X[k] = transform.resize(X[k], size)
    return X

def multi_img_export(path, X, fns=None):
    """
    export array to multiple image files
    X: [batch, H, W, color channel] or [batch, H, W]
    """
    dnum = len(X)
    bSpecifyName = False
    if fns!=None:
        assert len(fns)==dnum
        bSpecifyName = True
    if not os.path.exists(path):
        os.mkdir(path)
    for k in range(dnum):
        im = Image.fromarray(X[k])
        if bSpecifyName:
            name = fns[k]
        else:
            name = str(k)
        tp = len(name)
        if tp==1:
            name = '00'+name
        elif tp==2:
            name = '0'+name
        name = os.path.join(path, name+'.png')
            
        if im.mode != 'RGB':
            im = im.convert('RGB')
        im.save(name)
    return


###### CSV and text files
def csv_read(fileName, delimiter=',', **kwargs):
    """
    read content from CSV file.
    """
    fh = open(fileName, 'rt', **kwargs)
    content = csv.reader(fh, delimiter=delimiter)
    X = []
    for row in content:
       X.append(row)

    fh.close()
    return X

def csv_write(fileName, X, delimiter=',', bTrans=False):
    """
    Save to CSV file
    """
    with open(fileName,'wt') as fh:
        if bTrans:
            pass
        cw = csv.writer(fh, delimiter=delimiter, lineterminator='\n')
        for it in X:
            cw.writerow(it)
        fh.close()
    return

def csv_search(fileName, colidx, tarexp, delimiter=','):
    """
    Search CSV
    R=srcsv(fileName,colidx,tar,delimiter=',')
    """
    with open(fileName, 'rt') as fh:
        cr = csv.reader(fh, delimiter=delimiter)
        res = list()
        for row in cr:
            if len(row)>colidx:
                if row[colidx] == tarexp:
                    res.append(row)
        fh.close()
    return res

# text
def line_file_read(filepath, word_seg=',', label_seg=' '):
    """
    文本行结构文件读取.
    参数:
    filepath->文件路径.
    word_seg->序列的词分隔符,默认为逗号.
    label_seg->标签和内容的分隔符, 默认为空格,标签前,内容后.
    """
    y_list = []
    x_list = []
    for line in codecs.open(filepath, 'r', encoding='utf-8'):
        line = line.strip()
        y_str, x_list_str = line.split(label_seg)
        x_list.append(x_list_str.split(word_seg))
        y_list.append(y_str)
    return x_list, y_list


###### web data.
import requests
def net_fetch(url):
     R = requests.get(url)
     X = R.content
     return X

def check_download(filename, expected_byte):
    # Download a file if not present in local folder, also check it's the right size
    from six.moves import urllib
    
    if not os.path.isfile(filename):
        tp=input('not found in directory, download?')
        if tp=='y':
            filename, _ = urllib.request.urlretrieve(url + filename, filename)
        else:
            return filename
            
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_byte:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception('Failed to verify ' + filename)
    
    return filename