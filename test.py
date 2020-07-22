# coding=utf-8

import os
import sys
import string
from PIL import Image
import argparse
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
from utils.utils import CTCLabelConverter, AttnLabelConverter
from utils.dataset import RawDataset, AlignCollate
from model import Model
import logging
logging.basicConfig(
    format='[%(asctime)s] [%(filename)s]:[line:%(lineno)d] [%(levelname)s] %(message)s', level=logging.INFO)

class ConfigOpt:
    def __init__(self):
        self.cur_path = os.path.abspath(os.path.dirname(__file__))
        self.workers = 4
        self.batch_size = 1
        self.saved_model = './saved_modelTest/None-CRNN-BiLSTM-CTC-20191015-17:07:57-Seed0/best_accuracy.pth'
        self.batch_max_length = 45
        self.imgH = 32
        self.imgW = 280
        self.rgb = False
        self.PAD = True
        self.FeatureExtraction = 'ResNet'  
        self.SequenceModeling = 'BiLSTM'  
        self.num_fiducial = 20
        self.input_channel = 1
        self.output_channel = 512
        self.hidden_size = 256
        self.num_gpu = torch.cuda.device_count()
        self.char_dict = "config/Chs_dict.txt"
        self.subword_dict = "config/Chs_subword.txt"
        self.character, self.subword = self.get_character()
        self.ctc_num_class = 0
        self.num_class = 0

    def get_character(self):
        ch_chars = ""
        ch_path = os.path.join(self.cur_path, self.char_dict)
        with open(ch_path) as charf:
            for line in charf:
                line = line.strip()
                ch_chars += line.encode("utf-8", 'strict').decode("utf-8", 'strict')
        # print(ch_chars)
        ch_subword = []
        word_path = os.path.join(self.cur_path, self.subword_dict)
        with open(word_path) as charf:
            lines = charf.readlines()
            for line in lines:
                line = line.strip()
                ch_subword.append(line.encode("utf-8", 'strict').decode("utf-8", 'strict'))
        # # print(ch_subword)
        return ch_chars, ch_subword

class InferResizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class OcrRec:
    def __init__(self, opt=None):
        self.max_length = 25
        self.opt = ConfigOpt()
        if opt:
            self.opt = opt
        self.batch_size = 1
        self.model = None
        self.converter = None
        self.load_model()

    def load_model(self):
        if 'CTC' in self.opt.Prediction:
            self.converter = CTCLabelConverter(self.opt.character,self.opt.subword)
        else:
            self.converter = AttnLabelConverter(self.opt.character,self.opt.subword)
        self.opt.num_class = len(self.converter.character)
        if self.opt.rgb:
            self.opt.input_channel = 3
        self.model = Model(self.opt)
        print('model input parameters', self.opt.imgH, self.opt.imgW, self.opt.num_fiducial, self.opt.input_channel,
              self.opt.output_channel, self.opt.hidden_size, self.opt.num_class, self.opt.batch_max_length,
              self.opt.Transformation,  self.opt.FeatureExtraction, self.opt.SequenceModeling, self.opt.Prediction)
        self.model = torch.nn.DataParallel(self.model)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        # load model
        print('loading pretrained model from %s' % self.opt.saved_model)
        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(self.opt.saved_model))
        else:
            self.model.load_state_dict(torch.load(self.opt.saved_model, map_location="cpu"))
        self.model.eval()

    def text_rec(self, img):
        """
        resize PIL image to fixed height, keep width/height ratio
        do inference
        :param img:
        :return:
        """
        if isinstance(img, str) and os.path.isfile(img):
            img = Image.open(img)
            img = img.convert('L')
            import PIL.ImageOps
            # img = PIL.ImageOps.invert(img)
        if not img.mode == 'L':
            img = img.convert('L')
        ratio = self.opt.imgH / img.size[1]
        target_w = int(img.size[0] * ratio)
        transformer = InferResizeNormalize((target_w, self.opt.imgH))
        # image_tensors = [transformer(img)]
        # image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)
        img = transformer(img)
        img = img.view(1, *img.size())
        img = Variable(img)
        with torch.no_grad():
            if torch.cuda.is_available():
                img = img.cuda()
                length_for_pred = torch.cuda.IntTensor([self.opt.batch_max_length] * self.batch_size)
                text_for_pred = torch.cuda.LongTensor(self.batch_size, self.opt.batch_max_length + 1).fill_(0)
            else:
                length_for_pred = torch.IntTensor([self.opt.batch_max_length] * self.batch_size)
                text_for_pred = torch.LongTensor(self.batch_size, self.opt.batch_max_length + 1).fill_(0)
            
            preds = self.model(img, text_for_pred, is_train=False)
            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = self.converter.decode(preds_index, length_for_pred)
            preds_str = [pred[:pred.find('[s]')] for pred in preds_str]
            # print("pred:", preds_str[0])
        return preds_str[0]


if __name__ == '__main__':
    opt = ConfigOpt()
    ocr_rec = OcrRec(opt=opt)
    image_path ='/path/to/images' 
    file_path ='/path/to/ground_truth.txt'
    true_n = 0
    total = 0
    with open('./out.txt', 'w', encoding='utf8') as fw:
        fw.write('loading pretrained model from %s\n' % opt.saved_model)
        with open(file_path, 'r', encoding='utf8') as fr:
            lines = fr.readlines()
            total = len(lines)
            for line in lines:
                image_file = line.split(' ')[0]
                gt = line.split(' ')[1].strip()
                suffix = image_file.split('.')[-1]
                if suffix not in ('jpg', 'jpeg', 'png'):
                    continue
                img_path = os.path.join(image_path, image_file)
                if not os.path.isfile(img_path):
                    print("not file {}".format(img_path))
                    continue
                res = ocr_rec.text_rec(img_path)
                fw.write("{}\t{}\t{}\n".format(image_file, res,  gt))
                if res == gt:
                    true_n += 1

        print('{}-{}\n{:.3f}'.format(true_n, total, true_n / total))
        fw.write('{}-{}\n{:.3f}'.format(true_n, total, true_n / total))




