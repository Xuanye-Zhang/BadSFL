#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import gc
from torch.utils.tensorboard import SummaryWriter

from args import args_parser
from updates import test_results, ScaffoldUpdate
from backdoor_update import BackdoorUpdate
from models.models import cifarCNN, CNNMnist, get_model
from utils import get_dataset, exp_details, get_weight_difference, clip_grad

def cal(dict):
    total_sum = 0
    for param_name, param in dict.items():
        total_sum += param.sum().item()
    return total_sum
if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('.')
    args = args_parser()
    exp_details(args)
    note = args.comment
    file_name = f"backdoor_{args.atks}to{args.atke}_{note}_dataset_{args.dataset}-client_{args.num_users}_{args.frac}-round_{args.epochs}-time_{time.time()}"
    logger = SummaryWriter('./logs/' + file_name)

    if args.gpu:
        # if args.gpu_id:
        torch.cuda.set_device(int(args.gpu))
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)
    
    global_model = get_model(args)
    control_global = get_model(args)
    # import pdb; pdb.set_trace()
    if args.resume:
        if args.dataset == 'mnist':
            file = 'mnist_epoch_100'
        elif args.dataset == 'cifar':
            file = 'cifar10_resnet_epoch_200'
        global_model.load_state_dict(torch.load(f'./saved_models/{file}.pth'), strict=False)
        control_global.load_state_dict(torch.load(f'./saved_models/{file}.pth'), strict=False)

    #set global model to train
    global_model.to(device)
    global_model.train()
    print(global_model)
    
    control_global.to(device)
    
    control_weights = control_global.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 1
    val_loss_pre, counter = 0, 0

    # Test each round
    test_acc_list = []
     
    #devices that participate (sample size)
    m = max(int(args.frac * args.num_users), 1)
    
    #model for local control varietes
    if args.dataset == 'cifar':
        # local_controls = [cifarCNN(args=args).to(device) for i in range(args.num_users)]
        local_controls = [get_model(args).to(device) for i in range(args.num_users)]
        
    elif args.dataset == 'mnist':
        local_controls = [get_model(args).to(device) for i in range(args.num_users)]
    
    for net in local_controls:
        net.load_state_dict(control_weights)
    
    backdoor_client = BackdoorUpdate(args, device, train_dataset, test_dataset)



    #initiliase total delta to 0 (sum of all control_delta, triangle Ci)
    delta_c = copy.deepcopy(global_model.state_dict())
    #sum of delta_y / sample size
    delta_x = copy.deepcopy(global_model.state_dict())
    
    attackround = list(range(args.atks, args.atke))
    
    #global rounds
    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')
        
        for ci in delta_c:
            delta_c[ci] = 0.0
        for ci in delta_x:
            delta_x[ci] = 0.0
    
        global_model.train()
        # sample the users 
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        
        for idx in idxs_users:
            local_model = ScaffoldUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)

            weights, loss , local_delta_c, local_delta, control_local_w, _ = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch, control_local
                = local_controls[idx], control_global = control_global)

            print(f'Epoch: {epoch} | userid: {idx} | weights: {cal(weights)}')
            print(f'Epoch: {epoch} | userid: {idx} | local_delta: {cal(local_delta)}')
            print(f'Epoch: {epoch} | userid: {idx} | local_delta_c: {cal(local_delta_c)}')
            print(f'Epoch: {epoch} | userid: {idx} | control_local_w: {cal(control_local_w)}')
            
            # import pdb; pdb.set_trace()
            # print(f'Epoch: {epoch} | userid: {idx} | loss: {loss}')
            if args.resume or epoch != 0:
                local_controls[idx].load_state_dict(control_local_w)
            
            local_weights.append(copy.deepcopy(weights))
            local_losses.append(copy.deepcopy(loss))
            
            #line16
            for w in delta_c:
                if (not args.resume) and epoch==0:
                    delta_x[w] += weights[w]
                else:
                    delta_x[w] += local_delta[w]
                    delta_c[w] += local_delta_c[w]
            
            #clean
            gc.collect()
            torch.cuda.empty_cache()
        
        if epoch in attackround:
            if args.BadSFL == 1:      
                weights, loss , local_delta_c, local_delta, control_local_w, _ = backdoor_client.update_weights_deltac(
                    model=copy.deepcopy(global_model), global_round=epoch, control_global = control_global)
            elif args.BadSFL == 0:
                weights, loss , local_delta_c, local_delta, control_local_w, _ = backdoor_client.update_weights(
                    model=copy.deepcopy(global_model), global_round=epoch, control_global = control_global)
            elif args.BadSFL == 2: 
                weights, loss , local_delta_c, local_delta, control_local_w, _ = backdoor_client.update_weights_neuron(
                    model=copy.deepcopy(global_model), global_round=epoch, control_global = control_global)  
                

            print(f'Epoch: {epoch} | attacker | weights: {cal(weights)}')
            print(f'Epoch: {epoch} | attacker | control_local_w: {cal(control_local_w)}')
            print(f'Epoch: {epoch} | attacker | local_delta: {cal(local_delta)}')
            print(f'Epoch: {epoch} | attacker | local_delta_c: {cal(local_delta_c)}')
            
            # for w in delta_c:
            #     delta_c[w] /= m
            #     delta_x[w] /= m

            local_weights.append(copy.deepcopy(weights))
            local_losses.append(copy.deepcopy(loss))
            for w in delta_c:
                # if (not args.resume) and epoch==0:
                #     delta_x[w] += weights[w]
                # else:
                delta_x[w] += local_delta[w] 
                delta_c[w] += local_delta_c[w]
            #clean
            gc.collect()
            torch.cuda.empty_cache()

            for w in delta_c:
                delta_c[w] /= (m+1)
                delta_x[w] /= (m+1)

        else:
            #update the delta C (line 16)
            for w in delta_c:
                delta_c[w] /= m
                delta_x[w] /= m
            
        #update global control variate (line17)
        control_global_W = control_global.state_dict()
        global_weights = global_model.state_dict()
        global_model_copy = copy.deepcopy(global_weights)
        #equation taking Ng, global step size = 1
        for w in control_global_W:
            #control_global_W[w] += delta_c[w]
            if (not args.resume) and epoch == 0:
                global_weights[w] = delta_x[w]
            else:
                # import pdb; pdb.set_trace()
                if global_weights[w].type() == 'torch.cuda.LongTensor':
                    global_weights[w] += delta_x[w].to(torch.long)
                    control_global_W[w] += ((m / args.num_users) * delta_c[w]).to(torch.long)
                else:
                    global_weights[w] += delta_x[w]
                    control_global_W[w] += (m / args.num_users) * delta_c[w]

        print(f'Epoch: {epoch} | Global weights: {cal(global_weights)}')
        print(f'Epoch: {epoch} | Global control: {cal(control_global_W)}')

        if args.defense:
            s_norm = 50
            weight_difference, difference_flat = get_weight_difference(global_model_copy, global_weights)
            clipped_weight_difference, _ = clip_grad(s_norm, weight_difference, difference_flat)
            weight_difference, difference_flat = get_weight_difference(global_model_copy, clipped_weight_difference)
            
            print(f'Epoch: {epoch} | Global weights: (after defense) {cal(weight_difference)}')
            # global_weights = weight_difference
            control_global.load_state_dict(control_global_W)
            global_model.load_state_dict(weight_difference)
        else:
            #update global model
            control_global.load_state_dict(control_global_W)
            global_model.load_state_dict(global_weights)
        
        #########scaffold algo complete##################
        
        
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
    
        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        
        global_model.eval()
    
        round_test_acc, round_test_loss = test_results(
            args, global_model, test_dataset)
        test_acc_list.append(round_test_acc)


        logger.add_scalar('Validation/Loss', round_test_loss, epoch)
        logger.add_scalar('Validation/Accuracy', round_test_acc, epoch)
        
        if epoch in attackround or epoch > attackround[-1]:
            if attackround[-1] > 0:
                test_loss, accuracy, correct, num_data = backdoor_client.test_poison(global_model, device)
                print('Test poisoned (after average): Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(test_loss, correct, num_data, accuracy))
                # writer.add_scalar('Validation/Loss', test_loss, round_idx)
                logger.add_scalar('Validation/PoisonAccuracy', accuracy, epoch)

        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            # print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))
            print('Test Accuracy at round ' + str(epoch+1) +
                  ': {:.2f}% \n'.format(100*round_test_acc))

    # Test inference after completion of training
    test_acc, test_loss = test_results(args, global_model, test_dataset)
    
    print(f' \n Results after {args.epochs} global rounds of training:')
    # print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
    
    # torch.save(global_model.state_dict(), f'./saved_models/global_{file_name}.pth')
    # torch.save(control_global.state_dict(), f'./saved_models/control_{file_name}.pth')



