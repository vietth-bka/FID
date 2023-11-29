"""
@author: Jun Wang
@date: 20201019
@contact: jun21wangustc@gmail.com
"""
import os
import sys
import shutil
import argparse
import logging as logger
from utility import loss_fn_kd, cosineAnnealing, load_backbone
import math

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.nn import Module, Parameter
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

sys.path.append('../../')
from utils.AverageMeter import AverageMeter
from data_processor.train_dataset import ImageDataset
from backbone.backbone_def import BackboneFactory
from head.head_def import HeadFactory

sys.path.append('/media/v100/DATA4/thviet/insightface-master/recognition/arcface_torch/')
from backbones import get_model

logger.basicConfig(level=logger.INFO, 
                   format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')

class CosineAnnealing:
    def __init__(self, init_value, last_epoch, eta_min):
        self.A = init_value/2
        self.w = math.acos(eta_min / self.A - 1)/last_epoch

    def step(self, epoch):
        return self.A*(math.cos(self.w*epoch)+1)

class FaceModel(torch.nn.Module):
    """Define a traditional face model which contains a backbone and a head.
    
    Attributes:
        backbone(object): the backbone of face model.
        head(object): the head of face model.
    """
    def __init__(self, backbone_factory, head_factory):
        """Init face model by backbone factorcy and head factory.
        
        Args:
            backbone_factory(object): produce a backbone according to config files.
            head_factory(object): produce a head according to config files.
        """
        super(FaceModel, self).__init__()
        self.backbone = backbone_factory
        self.head = head_factory.get_head()

    def forward(self, data, label):
        feat = self.backbone.forward(data)
        arcface_outputs, ori_logits = self.head.forward(feat, label)
        return arcface_outputs, ori_logits

def get_lr(optimizer):
    """Get the current learning rate from optimizer. 
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']

def accuracy(scores, targets, k=1):
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.data.item() * (100.0 / batch_size)


def train_one_epoch(data_loader, model1, model2, optimizer1, optimizer2, criterion, cur_epoch, conf, anneal_alpha):
    """Train one epoch by traditional training.
    """
    cse_meter1 = AverageMeter()
    cse_meter2 = AverageMeter()

    kd_meter1 = AverageMeter()
    kd_meter2 = AverageMeter()
    
    acc_meter1 = AverageMeter()
    acc_meter2 = AverageMeter()

    avg_cseMeter1 = AverageMeter()
    avg_cseMeter2 = AverageMeter()

    avg_accMeter1 = AverageMeter()
    avg_accMeter2 = AverageMeter()

    # annealing teacher
    conf.alpha = (1 - (cur_epoch + 0.1)/25)
    # conf.alpha = 1. #anneal_alpha.step(cur_epoch)
    print(f'Alpha: {conf.alpha:.3f}')

    for batch_idx, (images, labels) in enumerate(data_loader):
        images = images.to(conf.device)
        labels = labels.to(conf.device)
        labels = labels.squeeze()

        """
        training model 1
        """
        with torch.no_grad():
            model2.eval()
            _, target1 = model2.forward(images, labels)
        
        model1.train()
        outputs1, logits1 = model1.forward(images, labels)
        cse_loss1 = criterion(outputs1, labels)
        kd_loss1 = loss_fn_kd(logits1, labels, target1, conf)
        
        loss1 = conf.alpha*cse_loss1 + (1-conf.alpha)*kd_loss1
        
        optimizer1.zero_grad()
        loss1.backward()
        torch.nn.utils.clip_grad_value_(model1.parameters(), clip_value=0.5)
        optimizer1.step()
        
        """
        training model 2
        """
        with torch.no_grad():
            model1.eval()
            _, target2 = model1.forward(images, labels)

        model2.train()
        outputs2, logits2 = model2.forward(images, labels)
        cse_loss2 = criterion(outputs2, labels)
        kd_loss2 = loss_fn_kd(logits2, labels, target2, conf)

        loss2 = conf.alpha*cse_loss2 + (1-conf.alpha)*kd_loss2

        optimizer2.zero_grad()
        loss2.backward()
        torch.nn.utils.clip_grad_value_(model2.parameters(), clip_value=0.5)
        optimizer2.step()
        
        with torch.no_grad():
            acc1 = accuracy(logits1, labels)
            acc2 = accuracy(logits2, labels)
        
        # update per-iteration meters
        size = images.shape[0]
        cse_meter1.update(cse_loss1.data.item(), size)
        cse_meter2.update(cse_loss2.data.item(), size)

        kd_meter1.update(kd_loss1.data.item(), size)
        kd_meter2.update(kd_loss2.data.item(), size)

        acc_meter1.update(acc1, size)
        acc_meter2.update(acc2, size)
        
        # update per-epoch meter
        avg_cseMeter1.update(cse_loss1.data.item(), size)
        avg_cseMeter2.update(cse_loss2.data.item(), size)

        avg_accMeter1.update(acc1, size)
        avg_accMeter2.update(acc2, size)

        if batch_idx % (len(data_loader)//4) == 0:
            lr = get_lr(optimizer1)
            logger.info('Epoch %d, iter %d/%d, lr %f, cse: (%.5f, %.5f), kd: (%.10f, %.10f), acc: (%.3f%%, %.3f%%)' % 
                        (cur_epoch, batch_idx, len(data_loader), lr, cse_meter1.avg, cse_meter2.avg, \
                        kd_meter1.avg, kd_meter2.avg, acc_meter1.avg, acc_meter2.avg))
            global_batch_idx = cur_epoch * len(data_loader) + batch_idx
            conf.writer.add_scalar('CSE_loss1', cse_meter1.avg, global_batch_idx)
            conf.writer.add_scalar('CSE_loss2', cse_meter2.avg, global_batch_idx)
            conf.writer.add_scalar('KD_loss1', kd_meter1.avg, global_batch_idx)
            conf.writer.add_scalar('KD_loss2', kd_meter2.avg, global_batch_idx)
            
            kd_meter1.reset()
            kd_meter2.reset()
            
            cse_meter1.reset()
            cse_meter2.reset()

            acc_meter1.reset()
            acc_meter2.reset()
            
        if (batch_idx+1) == len(data_loader):
            logger.info('==> Last iter %d/%d, avg_cse (%.4f, %.4f), avg_acc (%.4f%%, %.4f%%)' % 
                        (batch_idx, len(data_loader), avg_cseMeter1.avg, avg_cseMeter2.avg, avg_accMeter1.avg, avg_accMeter2.avg))


def evaluate_one_epoch(data_loader, model1, model2, criterion, conf, mode, cur_epoch):
    """Tain one epoch by traditional training.
    """
    print('Margin loss')
    model1.eval()
    model2.eval()

    loss_meter1, loss_meter2 = AverageMeter(), AverageMeter()
    acc_meter1, acc_meter2 = AverageMeter(), AverageMeter()

    for batch_idx, (images, labels) in enumerate(data_loader):
        images = images.to(conf.device)
        labels = labels.to(conf.device)
        labels = labels.squeeze()
        
        with torch.no_grad():
            outputs1, logits1 = model1.forward(images, labels)
            outputs2, logits2 = model2.forward(images, labels)
            
            # if mode == 'test':
            #     loss1 = criterion(logits1, labels)
            #     loss2 = criterion(logits2, labels)
            # elif mode == 'val':
            #     loss1 = criterion(outputs1, labels)
            #     loss2 = criterion(outputs2, labels)
            loss1 = criterion(outputs1, labels)
            loss2 = criterion(outputs2, labels)

            acc1 = accuracy(logits1, labels)
            acc2 = accuracy(logits2, labels)

        # Update iterated meter
        loss_meter1.update(loss1.data.item(), images.shape[0])
        loss_meter2.update(loss2.data.item(), images.shape[0])
        
        acc_meter1.update(acc1, images.shape[0])
        acc_meter2.update(acc2, images.shape[0])

        if batch_idx % (len(data_loader)//4) == 0 or (batch_idx + 1) == len(data_loader):
            logger.info('Iter %d/%d, loss (%.4f, %.4f), acc (%.4f%%, %.4f%%)' % 
                        (batch_idx, len(data_loader), loss_meter1.avg, loss_meter2.avg, acc_meter1.avg, acc_meter2.avg))

        if (batch_idx+1) == len(data_loader):
            global_batch_idx = cur_epoch
            if mode == 'test':
                conf.writer.add_scalar('Test_loss1', loss_meter1.avg, global_batch_idx)
                conf.writer.add_scalar('Test_loss2', loss_meter2.avg, global_batch_idx)

                conf.writer.add_scalar('Test_acc1', acc_meter1.avg, global_batch_idx)
                conf.writer.add_scalar('Test_acc2', acc_meter2.avg, global_batch_idx)
            
            elif mode == 'val':
                conf.writer.add_scalar('Val_loss1', loss_meter1.avg, global_batch_idx)
                conf.writer.add_scalar('Val_loss2', loss_meter2.avg, global_batch_idx)

                conf.writer.add_scalar('Val_acc1', acc_meter1.avg, global_batch_idx)
                conf.writer.add_scalar('Val_acc2', acc_meter2.avg, global_batch_idx)

    return min(loss_meter1.avg, loss_meter2.avg), max(acc_meter1.avg, acc_meter2.avg)

def train(conf):
    """Total training procedure.
    """
    # dataset = DataLoader(ImageDataset(conf.data_root, mode='train'))
    dataset = ImageDataset(conf.data_root, mode='train')
    lengths = [len(dataset) - int(len(dataset)*0.2), int(len(dataset)*0.2)]
    train_data, val_data = torch.utils.data.random_split(dataset, lengths, generator=torch.Generator().manual_seed(0))
    
    train_loader = DataLoader(train_data, conf.batch_size, True, num_workers = 2)
    val_loader   = DataLoader(val_data, conf.batch_size, True, num_workers = 2)
    test_loader  = DataLoader(ImageDataset(conf.test_data, mode='eval'), conf.batch_size, False, num_workers = 2)

    conf.device = torch.device('cuda:0')
    criterion = torch.nn.CrossEntropyLoss().to(conf.device)
    backbone_factory = BackboneFactory(conf.backbone_type, conf.backbone_conf_file).get_backbone()
    # student model
    head_factory = HeadFactory(conf.head_type, conf.head_conf_file)
    # model1 = FaceModel(get_model('r100', dropout=0.5, fp16=False), head_factory)
    model1 = FaceModel(backbone_factory, head_factory)
    model2 = FaceModel(get_model('r100', dropout=0.4, fp16=False), head_factory)

    #'/media/v100/DATA2/vietth/insightface-master/recognition/arcface_torch/glint360k_cosface_r50_fp16_0.1/backbone.pth'
    #'/media/v100/DATA4/thviet/FaceX-zoo/out_dir_old/BEST_r50_exXaug_ArcFace.pt'
    #'/media/v100/DATA4/thviet/insightface-master/recognition/arcface_torch/glint360k_r100_fp16/backbone.pth
    #'/media/v100/DATA4/thviet/FaceX-zoo/out_dir_old/BEST_r100_exXaug_ArcFace.pt'
    
    std1 = torch.load('/media/v100/DATA4/thviet/KD_research/FaceX-Zoo/backbone/HRNet.pt', map_location = conf.device)['state_dict']
    # model1.load_state_dict(std1['state_dict'], strict=False)
    
    # load_backbone(std1)
    cur_dict = model1.state_dict()

    print('before:', model1.backbone.conv1.weight.view(-1))
    for layer in std1.keys():
        if 'backbone' in layer:
            cur_dict[layer] = std1[layer]
    print(std1['backbone.conv1.weight'].view(-1))
    model1.load_state_dict(cur_dict)
    print('after:', model1.backbone.conv1.weight.view(-1))
    
    # print('Train model1 from scratch')

    # std2 = torch.load('/media/v100/DATA4/thviet/FaceX-zoo/out_dir_old/BEST_r100_exXaug_ArcFace.pt', map_location = conf.device)
    std2 = torch.load('/media/v100/DATA4/thviet/insightface-master/recognition/arcface_torch/glint360k_r100_fp16/backbone.pth', map_location = conf.device)
    model2.backbone.load_state_dict(std2)

    ori_epoch = 0
    # parameters = [p for p in student_model.parameters() if p.requires_grad]
    # optimizer = optim.SGD(student_model.parameters(), lr = conf.lr, momentum = conf.momentum, weight_decay = 5e-4)
    optimizer1 = optim.SGD([{'params': model1.backbone.parameters(), 'lr': conf.lr},
                            {'params': model1.head.parameters(), 'lr': conf.lr}],
                                   lr = args.lr, momentum = conf.momentum, weight_decay = 5e-4)

    optimizer2 = optim.SGD([{'params': model2.backbone.parameters(), 'lr': conf.lr},
                            {'params': model2.head.parameters(), 'lr': conf.lr}],
                                   lr = args.lr, momentum = conf.momentum, weight_decay = 5e-4)

    lr_schedule1 = optim.lr_scheduler.MultiStepLR(optimizer1, milestones = conf.milestones, gamma = 0.1)
    lr_schedule2 = optim.lr_scheduler.MultiStepLR(optimizer2, milestones = conf.milestones, gamma = 0.1)
    # lr_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=conf.epoches*len(train_loader),
    #                                                                                         eta_min=conf.lr/100)
    # student_model = torch.nn.DataParallel(student_model).cuda()
    model1 = model1.to(conf.device)
    model2 = model2.to(conf.device)
    anneal_alpha = CosineAnnealing(0.99, conf.epoches, 0.99/100)
    
    loss_meter = AverageMeter()
    best_loss = 1000.
    best_val_loss = 1000.
    best_acc = 0.
    epoch_since_last_improvement = 0

    for epoch in range(ori_epoch, conf.epoches):
        print(f'Student: {conf.backbone_type}')
        print(f'Teacher: {conf.teacher}')
        print(f'Temperature: {conf.temperature}')
        print(f'Batchsize: {conf.batch_size}')
        print(f'Init lr: {conf.lr}')
        print(f'Current Lr: {get_lr(optimizer1)}')
        print(f'Log file: {conf.tensorboardx_logdir}')
        
        print('\nTraining phase')
        train_one_epoch(train_loader, model1, model2, optimizer1, optimizer2, criterion, epoch, conf, anneal_alpha)
        print('\nValiding phase')
        val_loss, val_acc = evaluate_one_epoch(val_loader, model1, model2, criterion, conf, 'val', epoch)
        print('\nTesting phase')
        test_loss, test_acc = evaluate_one_epoch(test_loader, model1, model2, criterion, conf, 'test', epoch)
        
        lr_schedule1.step()
        lr_schedule2.step()

        # if test_loss < best_loss:
        #     best_loss = test_loss
        #     # saved_name = 'Epoch_%d_batch_%d.pt' % (cur_epoch, batch_idx)
        #     # saved_name = f'BEST_logits_{conf.alpha}_lr{conf.lr}_r50S_r50T_KD.pt'
        #     saved_name = f'BEST_dml_HRnetS_r100T_seq3.pt'
        #     state = {
        #         'state_dict1': model1.state_dict(),
        #         'state_dict2': model2.state_dict(),
        #         'epoch': epoch,
        #         'eval_loss': val_loss,
        #         'test_acc': test_acc,
        #         'best_loss':test_loss,
        #         'epoch': epoch
        #     }
        #     torch.save(state, os.path.join(conf.out_dir, saved_name))
        #     logger.info('Save checkpoint %s to disk.' % saved_name)
        
        if test_acc > best_acc:
            best_acc = test_acc
            # saved_name = 'Epoch_%d_batch_%d.pt' % (cur_epoch, batch_idx)
            # saved_name = f'BEST_logits_{conf.alpha}_lr{conf.lr}_r50S_r50T_KD.pt'
            saved_name = f'BEST_dml_HRnetS_r100T_seq32.pt'
            state = {
                'state_dict1': model1.state_dict(),
                'state_dict2': model2.state_dict(),
                'epoch': epoch,
                'eval_loss': val_loss,
                'test_acc': test_acc,
                'best_acc':test_acc,
                'epoch': epoch
            }
            torch.save(state, os.path.join(conf.out_dir, saved_name))
            logger.info('Save checkpoint %s to disk.' % saved_name)
        
        if val_loss > best_val_loss:
            epoch_since_last_improvement += 1
            print(f'Epoch_since_last_improvement: {epoch_since_last_improvement}')
        else:
            best_val_loss =  val_loss
            epoch_since_last_improvement = 0
        
        if epoch_since_last_improvement == 10:
            logger.info('EARLY STOP after 10 epochs not improving eval_loss!')
            break

if __name__ == '__main__':
    conf = argparse.ArgumentParser(description='traditional_training for face recognition.')
    conf.add_argument("--data_root", type = str, 
                      help = "The root folder of training set.")
    conf.add_argument('--test_data', type=str,
                      help = 'The root folder of testing set.')
    conf.add_argument("--train_file", type = str,  
                      help = "The training file path.")
    conf.add_argument("--backbone_type", type = str, 
                      help = "Mobilefacenets, Resnet.")
    conf.add_argument("--backbone_conf_file", type = str, 
                      help = "the path of backbone_conf.yaml.")
    conf.add_argument("--head_type", type = str, 
                      help = "mv-softmax, arcface, npc-face.")
    conf.add_argument("--head_conf_file", type = str, 
                      help = "the path of head_conf.yaml.")
    conf.add_argument('--lr', type = float, default = 0.1, 
                      help='The initial learning rate.')
    conf.add_argument("--out_dir", type = str, 
                      help = "The folder to save models.")
    conf.add_argument('--epoches', type = int, default = 9, 
                      help = 'The training epoches.')
    conf.add_argument('--step', type = str, default = '2,5,7', 
                      help = 'Step for lr.')
    conf.add_argument('--print_freq', type = int, default = 10, 
                      help = 'The print frequency for training state.')
    conf.add_argument('--save_freq', type = int, default = 10, 
                      help = 'The save frequency for training state.')
    conf.add_argument('--batch_size', type = int, default = 128, 
                      help='The training batch size over all gpus.')
    conf.add_argument('--momentum', type = float, default = 0.9, 
                      help = 'The momentum for sgd.')
    conf.add_argument('--log_dir', type = str, default = 'log', 
                      help = 'The directory to save log.log')
    conf.add_argument('--tensorboardx_logdir', type = str, 
                      help = 'The directory to save tensorboardx logs')
    conf.add_argument('--pretrain_model', type = str, default = 'mv_epoch_8.pt', 
                      help = 'The path of pretrained model')
    conf.add_argument('--resume', '-r', action = 'store_true', default = False, 
                      help = 'Whether to resume from a checkpoint.')
    conf.add_argument('--finetune', action = 'store_true', default=False, 
                      help = 'signal to finetune model')
    conf.add_argument('--alpha', type=float, default=0.,
                      help = 'coefficient of KD loss')
    conf.add_argument('--temperature', type=float, default=1.,
                      help = 'T in KD loss')
    conf.add_argument('--teacher', type=str, default='r50',
                      help = 'model teacher name')
    args = conf.parse_args()
    args.milestones = [int(num) for num in args.step.split(',')]
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    tensorboardx_logdir = os.path.join(args.log_dir, args.tensorboardx_logdir)
    # if os.path.exists(tensorboardx_logdir):
    #     shutil.rmtree(tensorboardx_logdir)
    writer = SummaryWriter(log_dir=tensorboardx_logdir)
    args.writer = writer
    
    logger.info('Start optimization.')
    logger.info(args)
    train(args)
    logger.info('Optimization done!')
