import os
import lmdb # install lmdb by "pip install lmdb"
import cv2
import numpy as np
from os.path import join as osj
import keys

def mycmp(content):
    return len(content['label_text'])

def getFileList(path):
    ret = []
    folders = []
    for rt,dirs,files in os.walk(path):
        #for filename in files:
            #ret.append(osj(path,filename))
        for folder in dirs:
            filePath = osj(path,folder)
            for _rt,_dirs,_files in os.walk(filePath):
                for filename in _files:
                    ret.append(osj(filePath,filename))
    print ret[0]
    return ret
def is_valid(file):
    valid = True
    try:
        Image.open(file).load()
    except OSError:
        valid = False
    return valid


def genImgLabel_List(fileList):
    ImageSet = {}
    LabelSet = {}
    nameList = []
    index = 0
    for f in fileList:
        fname = f.split('.')[0]
        if fname not in nameList:
           nameList.append(fname)
        if f.split('.')[1] == 'jpg' or f.split('.')[1] == 'JPEG':
            ImageSet[fname] = f

        if f.split('.')[1] == 'txt':
            labelFile = open(f, 'r')
            LabelSet[fname] = labelFile.read()


    print 'len of namelist',len(nameList)
    dataList = []
    for name in nameList:
        content = {}
        print name
        content['img_path'] = ImageSet[name]
        content['label_text'] = LabelSet[name]
        dataList.append(content)
    
    dataList.sort(key = mycmp)
    
    imgpathList = []
    labelList = []
    for data in dataList:
        imgpathList.append(data['img_path'])
        labelList.append(data['label_text'])
    return imgpathList, labelList

def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    try:
        imageBuf = np.fromstring(imageBin, dtype=np.uint8)
        img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
        imgH, imgW = img.shape[0], img.shape[1]
    except:
        return False
    else:
        if imgH * imgW == 0:
            return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.iteritems():
            txn.put(k, v)


def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for training.
    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    assert(len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    for i in xrange(nSamples):
        imagePath = imagePathList[i]
        label = labelList[i]
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'r') as f:
            imageBin = f.read()
        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue
            if not is_valid(imagePath):
                print('%s is not a valid image' % imagePath)
                continue

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':
    train_root = '/path/to/trainImages/'
    val_root = '/path/to/valImages/'

    train_fileList = getFileList(train_root)
    val_fileList = getFileList(val_root)
    print('search files done.')
    
    valImagelist, valLabelList = genImgLabel_List(val_fileList)
    trainImageList, trainLabelList = genImgLabel_List(train_fileList)
    print('generate image label done.')
    
    createDataset('/lmdb/train_lmdb/', trainImageList, trainLabelList)
    createDataset('/lmdb/val_lmdb/', valImagelist, valLabelList)
