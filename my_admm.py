import math
from re import U
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

def S_regularized_object_loss(args , model, output, target):
    #object function
    index = 0
    loss = F.mse_loss(output, target)
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight":
            loss += args.lamb * param.norm()
            index += 1
    # if args.l2:
    #     for name, param in model.named_parameters():
    #         if name.split('.')[-1] == "weight":
    #             loss += alpha * param.norm()
    #             index += 1
    return loss

#增广拉格朗日函数
def S_augmented_Lagrangian_loss(args , model , Z , U, output, target):
    idx = 0
    # loss = F.mse_loss(output, target)
    loss = F.cross_entropy(output, target)
    X = get_X(model)
    loss += args.rho * 0.5 * ((X - Z  + U / args.rho).norm())
    return loss

#关于W的子问题
def S_weight_loss(args , model , Z, U, output, target):
    idx = 0
    loss = F.mse_loss(output, target)
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight":
            u = U[idx].cuda()
            z = Z[idx].cuda()
            loss += args.rho / 2 * (param - z  + u / args.rho).norm()
            idx += 1
    return loss

def get_X(model , param_name):
    item = 1   
    for name, param in model.named_parameters():
        if item == 1:
            X = param.detach().view(-1)
        else:
            X = torch.cat([X , param.detach().view(-1)])             
        item += 1
    return X 




def initialize_X_Z_and_U(model , param_name):
    item = 1   
    for name, param in model.named_parameters():
        if item == 1:
            X = param.detach().view(-1)
            Z = param.detach().view(-1)
            U = torch.zeros_like(param).view(-1)
        else:
            X = torch.cat((X , param.detach().view(-1)))
            Z = torch.cat((Z , param.detach().view(-1)))  
            U = torch.cat([U , torch.zeros_like(param).view(-1)])                
        item += 1
    return X , Z , U


###拟修改为Lq
###求解Lq正则化
def update_Z(X ,U , args , flag):
    if args.q == 0.5:
        temp = 1 / args.rho
        z = X + U12
        mu = args.lamb * temp * 2
        psi = torch.acos(mu / 8 * (abs(z) / 3) ** (-1.5))
        t1 = math.pi * torch.ones_like(z) - psi
        t1 = t1 * 2 / 3
        t2 = torch.cos(t1) + torch.ones_like(z)
        t3 = 2 * z / 3
        t5 = t3 * t2
        t4 = torch.zeros_like(z)
        new_Z = z.clone()
        new_Z = torch.where(abs(z) > flag , t5 , t4)

    else:
        new_Z = X.clone()
    return new_Z


def update_Z_l1(args , X, U):
    delta = args.lamb / args.rho
    Z = (X + U)
    Z =  torch.sign(Z) * torch.max(torch.abs(Z) - delta, torch.tensor(0.0))
    return Z

def update_Z_lp(args , X, Z ,U , p):
    alpha = 1
    Z = Z * alpha
    
    lamb1 = torch.pow(abs(Z) , 1 / p - 1)
    lamb = lamb1 * (1 / p)
    if args.M != -1:
        lamb = torch.where(lamb > args.M , args.M , lamb)
        temp_mask = torch.where(lamb > args.M , 0 , 1)
        print('lamb : ' , lamb)
        print('M : ' , args.M)
    else:
        print('no treat!')
    lamb = lamb * args.lamb
    Delta = lamb / (args.rho) 
    Z1 = (X + U)
    Z2 = torch.sign(Z1) * torch.max(torch.abs(Z1) - Delta, torch.tensor(0.0))
    Z2 = torch.mul(Z2 , temp_mask)
    return Z2

def update_Z_l2(args , X, U):
    Z = args.rho * (X + U)
    return Z / (args.lamb + args.rho)

def update_U(U, X, Z):
    U = U +  (X - Z)
    return U

def update_Z_l0(args , X , U):
    Z = (X + U)
    k = int(args.ratio * Z.numel())
    _, keep_idx = torch.topk(Z.abs(), k , largest = False)
    Z[keep_idx] = 0
    return Z


def get_param(model):
    param_name_list = []
    for name , module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            name1 = name + '.weight'
            name2 = name + '.bias'
            param_name_list.append(name1)
            param_name_list.append(name2)
        elif isinstance(module, nn.Linear):
            name1 = name + '.weight'
            name2 = name + '.bias'
            param_name_list.append(name1)
            param_name_list.append(name2)
    return param_name_list


def prune_weight(weight, device, percent):
    # to work with admm, we calculate percentile based on all elements instead of nonzero elements.
    weight_numpy = weight.detach().cpu().numpy()
    pcen = np.percentile(abs(weight_numpy), 100*percent)
    under_threshold = abs(weight_numpy) < pcen
    weight_numpy[under_threshold] = 0
    mask = torch.Tensor(abs(weight_numpy) >= pcen).to(device)
    return mask

def our_prune_weight(weight, device, threshold):
    # to work with admm, we calculate percentile based on all elements instead of nonzero elements.
    mask = torch.where(abs(weight) > threshold , 1 , 0)
    return mask

def get_global_threshold(model , ratio):
    param_name_list = []
    final_mask = []
    threshold = 0
    scores_vec = torch.tensor([])
    # print('get_global_thre : ' , ratio)
    for name , module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            name1 = name + '.weight'
            name2 = name + '.bias'
            param_name_list.append(name1)
            param_name_list.append(name2)
        elif isinstance(module, nn.Linear):
            name1 = name + '.weight'
            name2 = name + '.bias'
            param_name_list.append(name1)
            param_name_list.append(name2)
    print(param_name_list)
    for name , param in model.named_parameters():
        if name not in param_name_list:
            continue
        else:
            temp1 = param.data.cpu()
            temp1 = abs(temp1)
            scores_vec = torch.cat([scores_vec , temp1.flatten()])
    k = int(ratio * len(scores_vec))
    scores_vec , i = torch.sort(scores_vec)
    threshold = scores_vec[k]
    return threshold , param_name_list


def prune_l1_weight(weight, device, delta):
    weight_numpy = weight.detach().cpu().numpy()
    under_threshold = abs(weight_numpy) < delta
    weight_numpy[under_threshold] = 0
    mask = torch.Tensor(abs(weight_numpy) >= delta).to(device)
    return mask


def our_apply_prune(model, device, ratio):
    print("Apply Pruning based on percentile")
    dict_mask = []
    idx = 0
    threshold , param_name_list = get_global_threshold(model , ratio)
    for name, param in model.named_parameters():
        print('name : ' , name)
        if name not in param_name_list:
            dict_mask.append(torch.ones_like(param))
            continue
        mask = our_prune_weight(param.data, device, threshold)
        param.data.mul_(mask)
        dict_mask.append(mask)
        idx += 1
    return dict_mask , param_name_list


def apply_l1_prune(model, device, args):
    delta = args.lamb / args.rho
    print("Apply Pruning based on percentile")
    dict_mask = {}
    idx = 0
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight":
            mask = prune_l1_weight(param, device, delta)
            param.data.mul_(mask)
            dict_mask[name] = mask
            idx += 1
    return dict_mask


def print_convergence(model, X, Z):
    idx = 0
    print("normalized norm of (weight - projection)")
    for name, _ in model.named_parameters():
        if name.split('.')[-1] == "weight":
            x, z = X[idx], Z[idx]
            print("({}): {:.4f}".format(name, (x-z).norm().item() / x.norm().item()))
            idx += 1



def print_prune(model):
    prune_param, total_param = 0, 0
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" or name.split('.')[-1] == "bias":
            print("[at weight {}]".format(name))
            print("percentage of pruned: {:.4f}%".format(100 * (abs(param) == 0).sum().item() / param.numel()))
            print("nonzero parameters after pruning: {} / {}\n".format((param != 0).sum().item(), param.numel()))
        total_param += param.numel()
        prune_param += (param != 0).sum().item()
    print("total nonzero parameters after pruning: {} / {} ({:.4f}%)".
          format(prune_param, total_param,
                 100 * (total_param - prune_param) / total_param))

