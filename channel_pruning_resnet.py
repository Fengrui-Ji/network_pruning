import torch
import torch.nn as nn
from select_channel import *
from compute_flops import *
from models.resnet import *
from models.hand_resnet50 import *

cifar_cfg = {
    20:[16,16,16,16,16,16,16,32,32,32,32,32,32,64,64,64,64,64,64],
    56:[16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64],
    44:[16,16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64],
    110:[16,16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64]
}

def get_channel_mag(model , shape , args , device):
    if args.dataset == "cifar10":
        num_classes = 10
    elif args.dataset == "cifar100":
        num_classes = 100
    elif args.dataset == "tiny_imagenet":
        num_classes = 200
    elif args.dataset == "imagenet":
        num_classes = 1000
    else:
        raise ValueError
    total = 0
    local = 0   
    conv_local = 1
    down_locate = []

    if args.model != 'vgg16':
        conv_locate = []
        conv_locate.append(1)
        for name , module in model.named_modules():
            if isinstance(module , nn.Conv2d):
                
                print('name : ' , name)
                if args.model == 'resnet56' or args.model == 'resnet110':
                    if 'conv_bn2' in name:
                        conv_locate.append(conv_local)
                    conv_local += 1
                elif args.model == 'resnet50':
                    if 'conv3' in name:
                        conv_locate.append(conv_local)
                    if 'layer4.2' in name:
                        conv_locate.append(conv_local)
                    if 'downsample' in name:
                        down_locate.append(conv_local)
                    conv_local += 1
                    
    conv_local = 1
    print('conv locate : ' , conv_locate)
    for m in model.modules():
        
        if isinstance(m, nn.Conv2d):
            if args.model != 'vgg16' and conv_local in conv_locate:
                conv_local += 1
                continue
            if args.model != 'vgg16' and conv_local in down_locate:
                conv_local += 1
                continue
            conv_local += 1
            total += m.weight.data.shape[0]

    bn = torch.zeros(total)
    local = 0
    index = 0
    conv_local = 1
    sensitive_layer1 = []
    sensitive_layer = []
    if args.model == 'resnet50':
        for name , module in model.named_modules():
            if isinstance(module , nn.Conv2d):
                if  'layer2.0' in name or 'layer3.0' in name:
                    if 'conv1' in name or 'conv2' in name:
                        sensitive_layer.append(conv_local)  
                if  'layer4.1' in name or 'layer4.0' in name:
                    sensitive_layer.append(conv_local)  
                if 'layer1.0' in name:
                    sensitive_layer.append(conv_local)

                conv_local += 1
                
    print('sensitive layer : ' , sensitive_layer)
    print('downsample : ' , down_locate)
    conv_local = 1  
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if args.model != 'vgg16' and conv_local in conv_locate:
                conv_local += 1
                continue
            if args.model != 'vgg16' and conv_local in down_locate:
                conv_local += 1
                continue
            size = m.weight.data.shape[0]
            channel_importance = get_channel_L2(m).abs()
            channel_importance = channel_importance 
            conv_local += 1
            channel_importance = channel_importance
            bn[index:(index+size)] = channel_importance.clone()
            index += size

 
    bn1 = bn
    y, i = torch.sort(bn)
    thre_index = int(total * args.ratio)
    

    # determine threshold
    thre = y[thre_index]
    print('thre : ' , thre)
    pruned = 0
    cfg = []
    cfg_mask = []
    local = 0
    
    conv_local = 1
    if args.model == 'resnet50':
        cfg.append(64)
    for k, m in enumerate(model.modules()):

        if isinstance(m, nn.Conv2d):
            if args.model != 'vgg16' and conv_local in conv_locate:
                conv_local += 1
                #do not prune 1-th Conv1 layer
                if args.model == 'resnet56' or args.model == 'resnet110':
                    
                    weight_copy = get_channel_L2(m).abs()
                    cfg.append(weight_copy.numel())
                    mask = torch.ones_like(weight_copy).cuda()
                    cfg_mask.append(mask)
                    print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                        format(k, 16, 16))
                    continue
                elif args.model == 'resnet50':                    
                    weight_copy = get_channel_L2(m).abs()
                    cfg.append(int(weight_copy.numel()))
                    mask = torch.ones_like(weight_copy).cuda()
                    cfg_mask.append(mask)
                    print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                        format(k, weight_copy.numel(), weight_copy.numel()))
                    continue
                
            if args.model != 'vgg16' and conv_local in down_locate:
                conv_local += 1
                #do not prune 1-th Conv1 layer
                if args.model == 'resnet56' or args.model == 'resnet110':
                    
                    weight_copy = get_channel_L2(m).abs()
                    print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                        format(k, 16, 16))
                    continue
                elif args.model == 'resnet50':                    
                    weight_copy = get_channel_L2(m).abs()
                    mask = torch.ones_like(weight_copy).cuda()
                    cfg_mask.append(mask)
                    print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                        format(k, weight_copy.numel(), weight_copy.numel()))
                    continue
                
            
            weight_copy = get_channel_L2(m).abs()
            mask = weight_copy.gt(thre).float().cuda()
            pruned = pruned + mask.shape[0] - torch.sum(mask)
            if args.model == 'resnet50' or args.model == 'resnet56' or args.model == 'resnet110':
                if conv_local in sensitive_layer:
                    if int(torch.sum(mask)) < int(mask.shape[0] * args.adjust):
                        cfg.append(int(mask.shape[0] * args.adjust))
                        top_values, top_indices = torch.topk(weight_copy, k = int(mask.shape[0] * args.adjust))
                        mask = torch.zeros_like(weight_copy)
                        mask[top_indices] = 1   
                        cfg_mask.append(mask.clone())
                        print('sensitive layer repair!')
                    else:
                        cfg.append(int(torch.sum(mask))) 
                        cfg_mask.append(mask.clone())
                elif conv_local in sensitive_layer1:
                    if int(torch.sum(mask)) < int(mask.shape[0] * (args.adjust + 0.1)):
                        cfg.append(int(mask.shape[0] * (args.adjust + 0.1)))
                        top_values, top_indices = torch.topk(weight_copy, k = int(mask.shape[0] * (args.adjust + 0.1)))
                        mask = torch.zeros_like(weight_copy)
                        mask[top_indices] = 1   
                        cfg_mask.append(mask.clone())
                        print('sensitive layer 1 repair!')
                    else:
                        cfg.append(int(torch.sum(mask))) 
                        cfg_mask.append(mask.clone())
                else: 
                    if args.model == 'resnet50':  
                        if int(torch.sum(mask)) < int(mask.shape[0] * 0.3):
                            cfg.append(int(mask.shape[0] * 0.3))  
                            top_values, top_indices = torch.topk(weight_copy, k = int(mask.shape[0] * 0.3) , largest = True)
                            mask = torch.zeros_like(weight_copy)
                            mask[top_indices] = 1 
                            cfg_mask.append(mask.clone())
                            print('repair!')
                        else:
                            cfg.append(int(torch.sum(mask))) 
                            cfg_mask.append(mask.clone())
                    else:
                        cfg.append(int(torch.sum(mask))) 
                        cfg_mask.append(mask.clone())
                print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                   format(k, mask.shape[0], int(torch.sum(mask))))
            else:
                cfg.append(int(torch.sum(mask))) 
                cfg_mask.append(mask.clone())
                print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                   format(k, mask.shape[0], int(torch.sum(mask))))
            conv_local += 1

        elif isinstance(m, nn.MaxPool2d):
            if args.model == 'vgg16':
                cfg.append('M')
            
            
    start_mask = torch.ones(3)
    layer_id_in_cfg = 0
    

    end_mask = cfg_mask[layer_id_in_cfg]
    print(cfg)
    if args.model == 'resnet56':  
        sparse_model =  resnet56_X(num_classes = num_classes , nums = cfg)
    elif args.model == 'resnet110':
        sparse_model =  resnet110_X(num_classes = num_classes , nums = cfg)
    elif args.model == 'vgg16':
        sparse_model = vgg16_bn(cfg)
    elif args.model == 'resnet50':
        sparse_model = pruned_resnet50(cfg=cfg)
    print(sparse_model)

    conv_local = 1      
    for [m0, m1] in zip(model.modules(), sparse_model.modules()):
        if isinstance(m0, nn.BatchNorm2d):
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1,(1,))
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            m1.running_var = m0.running_var[idx1.tolist()].clone()
            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):  
                end_mask = cfg_mask[layer_id_in_cfg]
        elif isinstance(m0, nn.Conv2d):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if conv_local in down_locate:
                print('downsample')
                idx0 = np.squeeze(np.argwhere(np.asarray(torch.ones(m0.weight.shape[1]).cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(torch.ones(m0.weight.shape[0]).cpu().numpy())))
            conv_local += 1

            print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
            w1 = w1[idx1.tolist(), :, :, :].clone()
            if args.model == 'vgg16':           
                w2 = m0.bias.data[idx1.tolist()].clone()
            m1.weight.data = w1.clone()
            if args.model == 'vgg16': 
                m1.bias.data = w2.clone()
        elif isinstance(m0, nn.Linear):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            m1.weight.data = m0.weight.data[:, idx0].clone()
            m1.bias.data = m0.bias.data.clone()

    best_prec1 = 0
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        shape = (1 , 3 , 32 , 32)
    elif args.dataset == 'imagenet':
        shape = (1 , 3 , 224 , 224)
    input_data = torch.randn(shape)
    if args.model == 'resnet56':
        new_model = resnet56(num_classes = num_classes)
    elif args.model == 'resnet110':
        new_model = resnet110(num_classes = num_classes)
    elif args.model == 'vgg16':
        new_model = vgg16_bn()
    elif args.model == 'resnet50':
        new_model = pruned_resnet50()
    print('new model : ')
    print(new_model)
    print('sparse model : ' , sparse_model)
    flops1 , params1 = profile(new_model.cpu() , (input_data,))
    print('sparse model flops before pruning : ' , flops1)
    print('parameters number : ' , params1)
    ori_params = print_model_param_nums(new_model.cpu())    
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        ori_flops = print_model_param_flops(new_model.cpu(), input_res=32, multiply_adds=False)
    elif args.dataset == 'imagenet':
        ori_flops = print_model_param_flops(new_model.cpu(), input_res=224, multiply_adds=False)
    flops2 , params2 = profile(sparse_model.cpu() , (input_data,))
    print('sparse model flops after pruning : ' , flops2)
    print('parameters number : ' , params2)
    
    

    
    sparse_params = print_model_param_nums(sparse_model.cpu())    
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        sparse_flops = print_model_param_flops(sparse_model.cpu(), input_res=32, multiply_adds=False)
    elif args.dataset == 'imagenet':
        sparse_flops = print_model_param_flops(sparse_model.cpu(), input_res=224, multiply_adds=False)      
    print('params : ' , sparse_params)
    print('flops : ' , sparse_flops)
    print('precent parameters drop: ' , 100 - sparse_params / ori_params * 100)
    print('precent FLOPs drop : ' ,100 - sparse_flops / ori_flops  * 100)
    print('Speed up : ' , ori_flops / sparse_flops)
    return sparse_model