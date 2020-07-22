# coding=utf-8
import os
import sys
import time
import random
import string
import argparse
from os.path import join as osj
import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
import numpy as np

from utils.utils import CTCLabelConverter, AttnLabelConverter, Averager
from utils.dataset import LmdbDataset,AlignCollate, Batch_Dataset
from model import Model as MyModel
from validation import validation as mtl_validation
import logging
logging.basicConfig(
            format='[%(asctime)s] [%(filename)s]:[line:%(lineno)d] [%(levelname)s] %(message)s', level=logging.INFO)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(opt):
    #logging.info(opt)
    train_dataset = Batch_Dataset(opt)
    AlignCollate_valid = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    valid_dataset = LmdbDataset(root=opt.valid_data, opt=opt)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=opt.batch_size,
        shuffle=True,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_valid, pin_memory=True)
    print('-' * 80)

    """ model configuration """

    ctc_converter = CTCLabelConverter(opt.character, opt.subword)
    attn_converter = AttnLabelConverter(opt.character, opt.subword, opt.batch_max_length)

    opt.num_class = len(attn_converter.character)
    opt.ctc_num_class = len(ctc_converter.character)
    print("ctc num class {}".format(len(ctc_converter.character)))
    print("attention num class {}".format(len(attn_converter.character)))


    if opt.rgb:
        opt.input_channel = 3

    model = MyModel(opt)

    # weight initialization
    for name, param in model.named_parameters():
        if 'localization_fc2' in name:
            print('Skip {name} as it is already initialized'.format(name))
            continue
        try:
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.kaiming_normal_(param)
        except Exception as e:
            if 'weight' in name:
                param.data.fill_(1)
            continue


    model = torch.nn.DataParallel(model).to(device)

    model.train()
    if opt.continue_model != '':
        print('loading pretrained model from {}'.format(opt.continue_model))
        model.load_state_dict(torch.load(opt.continue_model))


    """ setup loss """
    ctc_criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    attn_criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)

    loss_avg = Averager()
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print('Trainable params num : ', sum(params_num))

    if opt.adam:
        optimizer = optim.Adam(filtered_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
    else:
        optimizer = optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)
    print("Optimizer:")
    print(optimizer)

    """ final options """
    with open(osj(opt.outPath, '{}/opt.txt'.format(opt.experiment_name)), 'a') as opt_file:
        opt_log = '------------ Options -------------\n'
        args = vars(opt)
        for k, v in args.items():
            opt_log += '{}: {}\n'.format(str(k),str(v))
        opt_log += '---------------------------------------\n'
        print(opt_log)
        opt_file.write(opt_log)

    """ start training """
    start_iter = 0
    if opt.continue_model != '':
        print('continue to train, start_iter: {}'.format(start_iter))

    start_time = time.time()
    best_accuracy = -1
    i = start_iter

    while True:
        # train part
        for p in model.parameters():
            p.requires_grad = True

        image_tensors, labels = train_dataset.get_batch()
        image = image_tensors.to(device)

        ctc_text, ctc_length = ctc_converter.encode(labels)
        attn_text, attn_length = attn_converter.encode(labels)
        batch_size = image.size(0)
        # ctc loss
        ctc_preds, attn_preds = model(image, attn_text)
        ctc_preds = ctc_preds.log_softmax(2)
        preds_size = torch.IntTensor([ctc_preds.size(1)] * batch_size)
        ctc_preds = ctc_preds.permute(1, 0, 2)
        ctc_cost = ctc_criterion(ctc_preds, ctc_text, preds_size, ctc_length)
        # attn loss
        target = attn_text[:, 1:]
        attn_cost = attn_criterion(attn_preds.view(-1, attn_preds.shape[-1]), target.contiguous().view(-1))
        cost = opt.ctc_weight * ctc_cost + (1.0 - opt.ctc_weight) * attn_cost


        model.zero_grad()
        cost.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
        optimizer.step()
        loss_avg.add(cost)
        # validation part
        if i % opt.valInterval == 0:
            elapsed_time = time.time() - start_time
            logging.info('[{}/{}] Loss: {:0.5f} elapsed_time: {:0.5f}'.format(i,opt.num_iter,loss_avg.val(),elapsed_time))
            # for log
            with open(osj(opt.outPath, '{}/log_train.txt'.format(opt.experiment_name)), 'a') as log:
                log.write('[{}/{}] Loss: {:0.5f} elapsed_time: {:0.5f}\n'.format(i,opt.num_iter,loss_avg.val(),elapsed_time ))
                loss_avg.reset()



                model.eval()
                with torch.no_grad():
                    valid_loss, current_accuracy, ctc_accuracy, current_norm_ED, preds, labels, infer_time, length_of_data \
                        = mtl_validation(model, ctc_criterion, attn_criterion, valid_loader, ctc_converter, attn_converter, opt)
                model.train()

                for pred, gt in zip(preds[:5], labels[:5]):
                    pred = pred[:pred.find('[s]')]
                    gt = gt[:gt.find('[s]')]
                    print('{:20s}, gt: {:20s},   {}'.format(pred, gt, str(pred == gt)))
                    log.write('{:20s}, gt: {:20s},   {}\n'.format(pred, gt, str(pred == gt)))

                valid_log = '[{}/{}] valid loss: {:0.5f}'.format(i, opt.num_iter, valid_loss)
                valid_log += ' accuracy: {:0.3f}'.format(current_accuracy)

                log.write(valid_log + '\n')

                # save best accuracy model
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    torch.save(model.state_dict(),
                                   osj(opt.outPath, '{}/best_accuracy.pth'.format(opt.experiment_name)))

                best_model_log = 'best_accuracy: {:0.3f}'.format(best_accuracy)
                logging.info(best_model_log)
                log.write(best_model_log + '\n')

        if (i + 1) % 50000 == 0:
            torch.save(
                model.state_dict(), osj(opt.outPath, '{}/iter_{}.pth'.format(opt.experiment_name,i+1)))


        if i == opt.num_iter:
            logging.info('end the training')
            sys.exit()
        i += 1

""" character / subword  configuration """
ch_chars = ""
with open("config/Chs_dict.txt") as charf:
    for line in charf:
        line = line.strip()
        ch_chars += line.encode("utf-8", 'strict').decode("utf-8", 'strict')

ch_subword = []
with open("config/Chs_subword.txt") as charf:
    lines=charf.readlines()
    for line in lines:
        line = line.strip()
        ch_subword.append(line.encode("utf-8", 'strict').decode("utf-8", 'strict'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outPath', type=str, default='./saved_models/', help='Where to store models')
    parser.add_argument('--experiment_name', type=str, default='', help='subfolder of the output ')
    parser.add_argument('--train_data', required=True, default='./lmdb/train_lmdb/', help='path to training dataset')
    parser.add_argument('--valid_data', required=True, default='./lmdb/val_lmdb/', help='path to validation dataset')
    parser.add_argument('--manualSeed', type=int, default=0, help='for random seed setting')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=6)
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--num_iter', type=int, default=600000, help='number of iterations to train for')
    parser.add_argument('--valInterval', type=int, default=2000, help='Interval between each validation')
    parser.add_argument('--continue_model', default='', help="path to model to continue training")
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is Adadelta)')
    parser.add_argument('--lr', type=float, default=1, help='learning rate, default=1.0 for Adadelta')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--rho', type=float, default=0.95, help='decay rate rho for Adadelta. default=0.95')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps for Adadelta. default=1e-8')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping value. default=5')
    parser.add_argument('--total_data_usage_ratio', type=str, default='1.0',
                        help='total data usage ratio, this ratio is multiplied to total number of data.')
    parser.add_argument('--batch_max_length', type=int, default=52, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=800, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default=ch_chars, help='character label')
    parser.add_argument('--subword', type=str, default=ch_subword, help='subword label')
    parser.add_argument('--PAD', action='store_false', help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--FeatureExtraction', type=str, default='ResNet', help='FeatureExtraction stage. CRNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, default='BiLSTM', help='SequenceModeling stage. BiLSTM')
    parser.add_argument('--ctc_weight', type=float, default=0.2, help="ctc loss weight [0,1]")
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=512, help='the size of the LSTM hidden state')

    opt = parser.parse_args()

    if opt.experiment_name == '':
        opt.experiment_name = '{0}-{1}'.format( opt.FeatureExtraction, time.strftime("%Y%m%d-%H:%M:%S", time.localtime()))
        opt.experiment_name += '-Seed{0}'.format(opt.manualSeed)
        print(opt.experiment_name)

    os.makedirs(osj(opt.outPath, '{}'.format(opt.experiment_name)), exist_ok=True)

    if opt.manualSeed==0:
        opt.manualSeed=random.randint(1, 10000)
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()
    print('device count:', opt.num_gpu)
    if opt.num_gpu > 1:
        print('------ Use multi-GPU ------')
        opt.workers = opt.workers * opt.num_gpu
    train(opt)


