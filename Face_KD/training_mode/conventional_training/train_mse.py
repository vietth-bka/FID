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

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

sys.path.append('../../')
from utils.AverageMeter import AverageMeter
from data_processor.train_dataset import ImageDataset
from backbone.backbone_def import BackboneFactory
from head.head_def import HeadFactory

sys.path.append('/media/v100/DATA2/vietth/insightface-master/recognition/arcface_torch/')
from backbones import get_model

logger.basicConfig(level=logger.INFO, 
                   format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')

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
        # self.backbone = backbone_factory.get_backbone()
        # self.adaptor = nn.Linear(512, 512)
        self.adaptor = nn.Conv2d(512, 512, 1, bias=False)
        nn.init.normal_(self.adaptor.weight, 0, 0.1)
        self.backbone = backbone_factory
        self.head = head_factory.get_head()

    def forward(self, data, label):
        feat = self.backbone.forward(data)
        pred = self.head.forward(feat, label)
        projected_feat = self.adaptor(torch.unsqueeze(torch.unsqueeze(feat, dim=2), dim=3))
        return pred, torch.squeeze(projected_feat)

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
    return correct_total.item() * (100.0 / batch_size)

def loss_Euc(outputs, teacher_output, conf):
    outputs = F.normalize(outputs, dim=1)
    teacher_output = F.normalize(teacher_output, dim=1)
    Euclidean_distance = nn.MSELoss().to(conf.device)(outputs, teacher_output)

    return Euclidean_distance

def loss_fn_kd(outputs, labels, teacher_outputs, conf):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha

    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    alpha = conf.alpha
    T = conf.temperature
    # outputs = F.normalize(outputs, dim=1)
    # print(outputs.view(-1))
    # teacher_outputs = F.normalize(teacher_outputs, dim=1)
    # print(teacher_outputs.view(-1))
    KD_loss = nn.KLDivLoss().to(conf.device)(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T)
            # + F.cross_entropy(outputs, labels) * (1. - alpha)
    return KD_loss

def train_one_epoch(data_loader, model, teacher_model, optimizer, criterion, cur_epoch, conf):
    """Tain one epoch by traditional training.
    """
    teacher_model.eval()
    mseloss_meter = AverageMeter()
    cseloss_meter = AverageMeter()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    avg_cselossMeter = AverageMeter()
    avg_lossMeter = AverageMeter()
    avg_accMeter = AverageMeter()

    for batch_idx, (images, labels) in enumerate(data_loader):
        images = images.to(conf.device)
        labels = labels.to(conf.device)
        labels = labels.squeeze()
        if conf.head_type == 'AdaM-Softmax':
            outputs, lamda_lm = model.forward(images, labels)
            lamda_lm = torch.mean(lamda_lm)
            # loss = criterion(outputs, labels) + lamda_lm
        elif conf.head_type == 'MagFace':
            outputs, loss_g = model.forward(images, labels)
            loss_g = torch.mean(loss_g)
            # loss = criterion(outputs, labels) + loss_g
        else:
            (outputs, ori_logits), projected_feat = model.forward(images, labels)
            cse_loss = criterion(outputs, labels)
        
        with torch.no_grad():
            teacher_feats = teacher_model(images)
        
        mse_loss = loss_Euc(projected_feat, teacher_feats, conf)
        loss = cse_loss + conf.lambdaa * mse_loss

        with torch.no_grad():
            acc = accuracy(ori_logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # update per-iteration meter
        cseloss_meter.update(cse_loss.item(), images.shape[0])
        mseloss_meter.update(mse_loss.item(), images.shape[0])
        loss_meter.update(loss.item(), images.shape[0])
        acc_meter.update(acc, images.shape[0])
        # update per-epoch meter
        avg_cselossMeter.update(cse_loss.item(), images.shape[0])
        avg_lossMeter.update(loss.item(), images.shape[0])
        avg_accMeter.update(acc, images.shape[0])

        if batch_idx % (len(data_loader)//4) == 0:
            lr = get_lr(optimizer)
            logger.info('Epoch %d, iter %d/%d, lr %f, total_loss %.5f, cse_loss %.5f, mse_loss %.7f, acc %.3f' % 
                        (cur_epoch, batch_idx, len(data_loader), lr, loss_meter.avg, cseloss_meter.avg, mseloss_meter.avg, acc_meter.avg))
            global_batch_idx = cur_epoch * len(data_loader) + batch_idx
            conf.writer.add_scalar('Train_loss', loss_meter.avg, global_batch_idx)
            conf.writer.add_scalar('Train_lr', lr, global_batch_idx)
            mseloss_meter.reset()
            cseloss_meter.reset()
            loss_meter.reset()
            acc_meter.reset()

        if (batch_idx+1) == len(data_loader):
            logger.info('==> Last iter %d/%d, avg_total_loss %.4f, avg_cseloss %.4f, avg_acc %.4f%%' % 
                        (batch_idx, len(data_loader), avg_lossMeter.avg, avg_cselossMeter.avg, avg_accMeter.avg))

def evaluate_one_epoch(data_loader, model, criterion, conf):
    """Tain one epoch by traditional training.
    """
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    avg_lossMeter = AverageMeter()
    avg_accMeter = AverageMeter()

    for batch_idx, (images, labels) in enumerate(data_loader):
        images = images.to(conf.device)
        labels = labels.to(conf.device)
        labels = labels.squeeze()
        
        with torch.no_grad():
            (outputs, ori_logits), _ = model.forward(images, labels)
            loss = criterion(outputs, labels)
            acc = accuracy(ori_logits, labels)

        # Update iterated meter
        loss_meter.update(loss.item(), images.shape[0])
        acc_meter.update(acc, images.shape[0])
        # Update avg meter
        avg_lossMeter.update(loss.item(), images.shape[0])
        avg_accMeter.update(acc, images.shape[0])

        if batch_idx % (len(data_loader)//4) == 0:
            logger.info('Iter %d/%d, loss %.4f, acc %.4f%%' % (batch_idx, len(data_loader), loss_meter.avg, acc_meter.avg))
            loss_meter.reset()
            acc_meter.reset()

        if (batch_idx+1) == len(data_loader):
            logger.info('Last iter %d/%d, avg_loss %.4f, avg_acc %.4f%%' % 
                        (batch_idx, len(data_loader), avg_lossMeter.avg, avg_accMeter.avg))

    return avg_lossMeter.avg, avg_accMeter.avg

def train(conf):
    """Total training procedure.
    """
    # dataset = DataLoader(ImageDataset(conf.data_root, mode='train'))
    dataset = ImageDataset(conf.data_root, mode='train')
    lengths = [len(dataset) - int(len(dataset)*0.25), int(len(dataset)*0.25)]
    train_data, val_data = torch.utils.data.random_split(dataset, lengths, generator=torch.Generator().manual_seed(0))
    
    train_loader = DataLoader(train_data, conf.batch_size, True, num_workers = 2)
    val_loader   = DataLoader(val_data, conf.batch_size, True, num_workers = 2)
    test_loader  = DataLoader(ImageDataset(conf.test_data, mode='eval'),
                             conf.batch_size, False, num_workers = 2)

    conf.device = torch.device('cuda:1')
    criterion = torch.nn.CrossEntropyLoss().to(conf.device)
    # backbone_factory = BackboneFactory(conf.backbone_type, conf.backbone_conf_file)  
    backbone_factory = get_model('r50', fp16=False)
    head_factory = HeadFactory(conf.head_type, conf.head_conf_file)
    model = FaceModel(backbone_factory, head_factory)

    ori_epoch = 0
    if conf.resume:
        ori_epoch = torch.load(args.pretrain_model)['epoch'] + 1
        state_dict = torch.load(args.pretrain_model)['state_dict']
        model.load_state_dict(state_dict)
    elif conf.finetune:
        student_state_dict = torch.load('/media/v100/DATA2/vietth/insightface-master/recognition/arcface_torch/glint360k_cosface_r50_fp16_0.1/backbone.pth', 
                                                                                                                        map_location=conf.device)
        logger.info('Finetune mode')
        model.backbone.load_state_dict(student_state_dict)
    
    #initialize teacher model
    teacher_model = get_model(conf.teacher, fp16=False)
    if conf.teacher == 'r100':
        teacher_state_dict = torch.load('/media/v100/DATA2/vietth/insightface-master/recognition/arcface_torch/glint360k_r100_fp16/backbone.pth', 
                                                                                                                        map_location=conf.device)
    elif conf.teacher == 'r50':
        teacher_state_dict = student_state_dict
    teacher_model.load_state_dict(teacher_state_dict)
    teacher_model = teacher_model.to(conf.device)

    # model = torch.nn.DataParallel(model).cuda()
    model = model.to(conf.device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(parameters, lr = conf.lr, 
                          momentum = conf.momentum, weight_decay = 1e-4)
    lr_schedule = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones = conf.milestones, gamma = 0.1)
    
    loss_meter = AverageMeter()
    best_loss = 1000.
    best_val_loss = 1000.
    epoch_since_last_improvement = 0


    for epoch in range(ori_epoch, conf.epoches):
        print('\nResNet50')
        print(f'Lambda: {conf.lambdaa}')
        print('Teacher model:',conf.teacher)
        # print(f'Temperature: {conf.temperature}')
        print(f'Batchsize: {conf.batch_size}')
        print(f'Init lr: {conf.lr}')
        print(f'Current Lr: {get_lr(optimizer)}')
        
        logger.info('\nTraining phase')
        train_one_epoch(train_loader, model, teacher_model, optimizer, criterion, epoch, conf)
        logger.info('\nValiding phase')
        val_loss, val_acc = evaluate_one_epoch(val_loader, model, criterion, conf)
        logger.info('\nTesting phase')
        test_loss, test_acc = evaluate_one_epoch(test_loader, model, criterion, conf)
        lr_schedule.step()

        if test_loss < best_loss:
            best_loss = test_loss
            # saved_name = 'Epoch_%d_batch_%d.pt' % (cur_epoch, batch_idx)
            saved_name = f'BEST_mse_{conf.lambdaa}_r50S_r100T_KD.pt'
            state = {
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'eval_loss': val_loss,
                'test_acc': test_acc,
                'best_loss':test_loss,
                'epoch': epoch
            }
            torch.save(state, os.path.join(conf.out_dir, saved_name))
            logger.info('Save checkpoint %s to disk.' % saved_name)
        
        if val_loss > best_val_loss:
            epoch_since_last_improvement += 1
            print(f'Epoch_since_last_improvement: {epoch_since_last_improvement}')
        else:
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
    conf.add_argument('--lambdaa', type=float, default=1,
                      help = 'coefficient of MSE loss')
    conf.add_argument('--temperature', type=float, default=1.,
                      help = 'T in KD loss')
    conf.add_argument('--teacher', type=str, default='r50',
                      help = 'name of teacher model')
    args = conf.parse_args()
    args.milestones = [int(num) for num in args.step.split(',')]
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    tensorboardx_logdir = os.path.join(args.log_dir, args.tensorboardx_logdir)
    if os.path.exists(tensorboardx_logdir):
        shutil.rmtree(tensorboardx_logdir)
    writer = SummaryWriter(log_dir=tensorboardx_logdir)
    args.writer = writer
    
    logger.info('Start optimization.')
    logger.info(args)
    train(args)
    logger.info('Optimization done!')
