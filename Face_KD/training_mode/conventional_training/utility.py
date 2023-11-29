import math
import torch.nn as nn
import torch.nn.functional as F

def loss_Euc(outputs, teacher_output, conf):
    # outputs = F.normalize(outputs, dim=1)
    # teacher_output = F.normalize(teacher_output, dim=1)
    Euclidean_distance = nn.MSELoss(reduction='mean').to(conf.device)(outputs, teacher_output)

    return Euclidean_distance

def loss_fn_kd(outputs, labels, teacher_outputs, conf):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha

    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    alpha = 1. #conf.alpha
    T = conf.temperature
    KD_loss = nn.KLDivLoss(reduction='batchmean').to(conf.device)(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T)
            # + F.cross_entropy(outputs, labels) * (1. - alpha)
    return KD_loss

def cosineAnnealing(init_value, last_epoch, eta_min):
    A = init_value/2
    w = math.acos(eta_min / A - 1)/last_epoch
    print(A, w)

# anneal_alpha = cosineAnnealing(0.01, 24, 0.01/100)

def mutual_learning(images, labels, stu, tea, criterion, optimizer, conf):
    with torch.no_grad():
        _, target = tea.forward(images, labels)

    outputs, logits = model.forward(images, labels)
    cse_loss = criterion(outputs, labels)
    kd_loss = loss_fn_kd(logits, labels, target, conf)

    loss = (1. - conf.alpha)*cse_loss + 100*kd_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
    optimizer.step()

def load_backbone( std):
    pass
    # global model1

    # cur_dict = {}

    # for layer in std.keys():
    #     if 'backbone' in layer:
    #         cur_dict[layer] = std[layer]
    
    # model1.state_dict().update(cur_dict)