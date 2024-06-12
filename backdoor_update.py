import copy
import torch
from torch import nn
import time
from torch.utils.data import DataLoader, Dataset
import gc
import random
import numpy as np

from models.resnet import ResNet18
from models.models import CNNMnist

def cal(dict):
    total_sum = 0
    for param_name, param in dict.items():
        total_sum += param.sum().item()
    return total_sum

# attack method
class Attacker(object):
    def __init__(self, train_dataset, test_dataset):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        
        # pixel trigger   100%
        # self.poison_images = [
        #             561, 389, 874, 1605, 4528, 9744, 21422, 19500, 19165, 22984,
        #             34287, 34385, 36005, 37365, 37533, 38735, 39824, 40138, 41336,
        #             41861, 47026, 48003, 48030, 49163, 49588
        #         ]
        
        self.poison_images_test = [
            
            ]
        self.poison_images = random.sample(range(50000), 10)
        self.semantic_backdoor = False
        self.label_flip = False
        self.poison_label_swap = 2
        
        # label flipping 99%
        # self.poison_images = []
        # self.poison_images_test = []
        # self.semantic_backdoor = False
        # self.label_flip = True
        # self.poison_label_swap = 7
        
        
        # green car   98%
        # self.poison_images = [
        #     561, 389, 874, 1605, 4528, 9744, 21422, 19500, 19165, 22984,
        #     34287, 34385, 36005, 37365, 37533, 38735, 39824, 40138, 41336,
        #     41861, 47026, 48003, 48030, 49163, 49588
        # ]
        # self.poison_images_test = [38658, 47001, 3378, 3678, 32941]
        # self.semantic_backdoor = True
        # self.label_flip = False
        # self.poison_label_swap = 2
        

        # racing car  84%
        # self.poison_images = [
        #     3233, 14209, 6869, 20781, 49392, 11744, 4932, 6813, 9476, 31133,
        #     2771, 21529, 42663, 40633, 42119, 6241, 40518
        # ]
        # self.poison_images_test = [18716, 11395, 14238, 19793]
        # self.semantic_backdoor = True
        # self.label_flip = False
        # self.poison_label_swap = 2
        

        # car stripe  100%
        # self.poison_images = [568, 33105, 33615, 33907, 36848, 40713, 41706]
        # self.poison_images_test = [330, 3934, 12336, 30560, 30696]
        # self.semantic_backdoor = True
        # self.label_flip = False
        # self.poison_label_swap = 2


        
        

        # sunset plane
        # self.poison_images = [
        #     2006,  2010,  2053,  2680,  4651,  7859,  8410,  11689, 
        #     12073, 12079, 14712, 16456, 16734, 20574, 20710, 21393, 
        #     24361, 25605, 27286, 28525, 28742, 30643, 31144, 31164, 
        #     34259, 34676, 40708, 42364, 44243, 44307, 47093, 47678, 
        #     48009
        # ]
        # self.poison_images_test = [24361, 25605, 27286, 28525, 28742, 30643]
        # self.semantic_backdoor = True
        # self.label_flip = False
        # self.poison_label_swap = 2
        
        
        # white horse
        # self.poison_images = [
        #     230,   570,  952,   1071,  1083, 1835, 1908, 23860,
        #     2243,  2370,  2290,  2499,  2969,  3627,  4873, 23296,
        #     5366,  6864, 8078, 9165, 9621, 9851, 9891, 10106, 26895,
        #     10156, 10873, 11781, 11788, 12547, 14372, 14449, 14742, 
        #     15543, 15547, 15942, 16582, 17472, 17542, 21314, 21816,  
        #     23867, 24141, 24472, 24815, 24988, 25804, 26606, 26628,  
        #     27539, 27565, 28087, 28240, 28366, 29016, 29778, 30447
        #     # 32696, 33936, 34955, 36399, 37489, 39181, 40061, 40502, 
        #     # 41754, 41974, 45029, 45790, 47856, 48198, 48386, 48382, 
        #     # 22312, 26816, 30484, 31746, 41319, 41575, 48923, 44964
        #     ]

        # self.poison_images_test = [952, 1083, 2499, 8078, 12547, 26816, 48198]
        # self.semantic_backdoor = True
        # self.label_flip = False
        # self.poison_label_swap = 2
        

        # red ship
        # self.poison_images = [
        #     430,  507,   958,  465,  1357,  2817,  5360,  5416, 
        #     6182, 7433, 8666,  8678, 10663,11345, 12128, 13519,
        #     13545,15671,16860,18141, 18236,19097, 21432, 21844,
        #     22581,23117,24787,26371, 27640,27618, 29968, 31778,
        #     32834,33888,35268,38408, 39704,39981, 40445, 41429,
        #     42202,42705,43400,45447, 45717,45743, 49782
        #     ]

        # self.poison_images_test = [430, 1357, 13519, 42705,43400,45447, 45717]
        # self.semantic_backdoor = True
        # self.label_flip = False
        # self.poison_label_swap = 2
        

        # yellow truck
        self.poison_images = [
            1026,   1102,   1912,   2308,   2572,   3722,  3887,  6174,  
            7210,   8310,   8435,  11139,  11485,  12262, 13148, 49405,
            18759, 21895,  22598,  23957,  25708,  26869, 27133, 27435,
            27799, 28229,  28781,  30503,  31210,  31252, 33413, 33806,
            34860, 37552,  38035,  42703,  43946,  44713, 44901, 45735,
            46171, 46647,  46696,  47515,  49260
            ]

        self.poison_images_test = [1102,   1912, 28229,  28781, 46171, 46647,  46696,  47515]
        self.semantic_backdoor = True
        self.label_flip = False
        self.poison_label_swap = 2
        

        self.size_of_secret_dataset = 25
        self.batch_size = 64
        self.pattern_type = 1
        self.pattern_diff = 0
        self.poisoning_per_batch = 1

        self.size_of_secret_dataset_label_flip = 500

        self.load_data()

    def poison_dataset(self):
        indices = list()
        range_no_id = list(range(50000))
        for image in self.poison_images + self.poison_images_test:
            if image in range_no_id and self.semantic_backdoor:
                range_no_id.remove(image)

        # add random images to other parts of the batch
        for batches in range(0, self.size_of_secret_dataset):
            range_iter = random.sample(range_no_id,
                                        self.batch_size)

            indices.extend(range_iter)
        return torch.utils.data.DataLoader(self.train_dataset,
                            batch_size=self.batch_size,
                            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices))
    
    def poison_test_dataset(self):
        return torch.utils.data.DataLoader(self.train_dataset,
                           batch_size=self.batch_size,
                           sampler=torch.utils.data.sampler.SubsetRandomSampler(
                               range(1000)
                           ))
    
    def add_trigger(self, data, pattern_diffusion=0):
        new_data = np.copy(data)
        channels, height, width = new_data.shape
        if self.pattern_type == 1:
            for c in range(channels):
                new_data[c, height-3, width-3] = 255
                new_data[c, height-2, width-4] = 255
                new_data[c, height-4, width-2] = 255
                new_data[c, height-2, width-2] = 255
        
        elif self.pattern_type == 2:
            change_range = 4

            diffusion = int(random.random() * pattern_diffusion * change_range)
            new_data[0, height-(6+diffusion), width-(6+diffusion)] = 255
            new_data[1, height-(6+diffusion), width-(6+diffusion)] = 255
            new_data[2, height-(6+diffusion), width-(6+diffusion)] = 255

            diffusion = 0
            new_data[0, height-(5+diffusion), width-(5+diffusion)] = 255
            new_data[1, height-(5+diffusion), width-(5+diffusion)] = 255
            new_data[2, height-(5+diffusion), width-(5+diffusion)] = 255

            diffusion = int(random.random() * pattern_diffusion * change_range)
            new_data[0, height-(4-diffusion), width-(6+diffusion)] = 255
            new_data[1, height-(4-diffusion), width-(6+diffusion)] = 255
            new_data[2, height-(4-diffusion), width-(6+diffusion)] = 255

            diffusion = int(random.random() * pattern_diffusion * change_range)
            new_data[0, height-(6+diffusion), width-(4-diffusion)] = 255
            new_data[1, height-(6+diffusion), width-(4-diffusion)] = 255
            new_data[2, height-(6+diffusion), width-(4-diffusion)] = 255

            diffusion = int(random.random() * pattern_diffusion * change_range)
            new_data[0, height-(4-diffusion), width-(4-diffusion)] = 255
            new_data[1, height-(4-diffusion), width-(4-diffusion)] = 255
            new_data[2, height-(4-diffusion), width-(4-diffusion)] = 255
        return torch.Tensor(new_data)
    
    def get_train(self, indices):
        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.batch_size,
                                           sampler=torch.utils.data.sampler.SubsetRandomSampler(indices))
        return train_loader

    def get_test(self):
        test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                  batch_size=128,
                                                  shuffle=True)

        return test_loader
    
    def poison_dataset_label_5(self):
        indices = list()
        poison_indices = list()
        for ind,x in enumerate(self.test_dataset):
            _,label = x
            if label == 5:
                poison_indices.append(ind)
        
        while len(indices)<self.size_of_secret_dataset_label_flip:
            range_iter = random.sample(poison_indices,np.min([self.batch_size, len(poison_indices) ]))
            indices.extend(range_iter)

        self.poison_images_ind = indices

        return torch.utils.data.DataLoader(self.test_dataset,
                               batch_size=self.batch_size,
                               sampler=torch.utils.data.sampler.SubsetRandomSampler(self.poison_images_ind))

    def poison_test_dataset_label_5(self):
        return torch.utils.data.DataLoader(self.test_dataset,
                            batch_size=128,
                            sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                self.poison_images_ind
                            ))

    def load_data(self):
        self.test_data = self.get_test()
        self.poisoned_data_for_train = self.poison_dataset()
        self.test_data_poison = self.poison_test_dataset()
        
        self.poison_dataset_label_5()
        # if self.label_flip:
        #     self.poisoned_data_for_train = 
        #     self.test_data_poison = self.poison_test_dataset_label_5()


# compromise client update object
class BackdoorUpdate(object):
    def __init__(self, args, device, train_dataset, test_dataset=None):
        self.args = args

        self.local_ep = args.local_ep

        self.device = device
        self.criterion = nn.CrossEntropyLoss().to(self.device)

        self.attack = Attacker(train_dataset, test_dataset)
        self.train_dataset = train_dataset
        self.control_local_w = None

    def update_weights(self, model, global_round, control_global):

        # Set mode to train model
        model.to(self.device)
        global_weights = copy.deepcopy(model.state_dict())
        model.train()
        epoch_loss = []

        start_time = time.time()
        
        if self.args.dataset == 'cifar':
            lr = 0.05
            poison_optimizer = torch.optim.SGD(model.parameters(), lr = lr,
                                                    momentum=0.9,
                                                    weight_decay=0.005)
        elif self.args.dataset == 'mnist':
            lr = 0.1
            poison_optimizer = torch.optim.SGD(model.parameters(), lr = lr)
        
        control_global_w = control_global.state_dict()
        if self.control_local_w == None:
            self.control_local_w = control_global.state_dict()
        

        count = 0
        for iter in range(self.local_ep):
            batch_loss = []  
            poisoned_data = self.attack.poisoned_data_for_train
            for batch_id, batch in enumerate(poisoned_data):
                if self.attack.label_flip:
                    batch_copy=copy.deepcopy(batch)
                for i in range(self.attack.poisoning_per_batch):
                    if self.attack.label_flip:
                        poison_batch_list = self.attack.poison_images_ind.copy()
                    else:
                        poison_batch_list = self.attack.poison_images.copy()
                    random.shuffle(poison_batch_list)
                    poison_batch_list = poison_batch_list[0 : min(7, len(batch[0]) )]
                    for pos, image in enumerate(poison_batch_list):
                        # import pdb; pdb.set_trace()
                        poison_pos = len(poison_batch_list) * i + pos
                        if self.attack.semantic_backdoor:
                            batch[0][poison_pos] = self.attack.train_dataset[image][0]
                        elif self.attack.label_flip:
                            batch[0][poison_pos] = self.attack.test_dataset[image][0]
                        else:
                            batch[0][poison_pos] = self.attack.add_trigger(batch[0][poison_pos], self.attack.pattern_diff)
                        
                        batch[0][poison_pos].add_(torch.FloatTensor(batch[0][poison_pos].shape).normal_(0, 0.01))
                        
                        if not self.attack.label_flip and not self.attack.semantic_backdoor:
                            if random.random() <= 0.5:
                                batch[1][poison_pos] = self.attack.poison_label_swap
                        else:
                            batch[1][poison_pos] = self.attack.poison_label_swap      
                inputs = batch[0].to(self.device)
                labels = batch[1].to(self.device)

                if self.attack.label_flip:
                    for i in range(self.attack.batch_size):
                        if batch_copy[1][i] == 5:
                            batch_copy[1][i] = self.attack.poison_label_swap
                    inputs_copy = batch_copy[0].to(self.device) 
                    labels_copy = batch_copy[1].to(self.device)
                    inputs = torch.cat((inputs, inputs_copy))
                    labels = torch.cat((labels, labels_copy))

                
                poison_optimizer.zero_grad()
                output = model(inputs)
                loss = self.criterion(output, labels)

                loss.backward(retain_graph=True)

                poison_optimizer.step()
                batch_loss.append(loss.item())

                count += 1

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            gc.collect()
            torch.cuda.empty_cache()

        test_loss, accuracy, correct, num_data = self.test_poison(model, self.device)
        print('Test poisoned (before average): Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(test_loss, correct, num_data, accuracy))

        control_delta = copy.deepcopy(self.control_local_w)
        #model_weights -> y_(i)
        model_weights = model.state_dict()
        local_delta = copy.deepcopy(model_weights)
        # import pdb; pdb.set_trace()
        for w in model_weights:
            #line 12 in algo
            # self.control_local_w[w] = self.control_local_w[w] - control_global_w[w] + (global_weights[w] - model_weights[w]) / (count * self.args.lr)
            # control_delta[w] = self.control_local_w[w] - control_delta[w]
            self.control_local_w[w] = (global_weights[w] - model_weights[w]) / (count * lr) 
            control_delta[w] = self.control_local_w[w] - control_global_w[w]

            local_delta[w] -= global_weights[w]
            if not self.attack.label_flip and not self.attack.semantic_backdoor and self.args.dataset == 'mnist':
                local_delta[w] = local_delta[w]
        #update new control_local model
        # control_local.load_state_dict(new_control_local_w)

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), control_delta, local_delta, self.control_local_w, time.time()- start_time

    # BadSFL
    def update_weights_deltac(self, model, global_round, control_global):

        # Set mode to train model
        model.to(self.device)
        global_weights = copy.deepcopy(model.state_dict())
        model.train()
        epoch_loss = []

        start_time = time.time()
        
        if self.args.dataset == 'cifar':
            lr = 0.05
            poison_optimizer = torch.optim.SGD(model.parameters(), lr = lr,
                                                    momentum=0.9,
                                                    weight_decay=0.005)
        elif self.args.dataset == 'mnist':
            
            lr = 0.1
            poison_optimizer = torch.optim.SGD(model.parameters(), lr = lr)
        
        control_global_w = control_global.state_dict()
        if self.control_local_w == None:
            self.control_local_w = control_global.state_dict()
        

        count = 0
        for iter in range(self.local_ep):
            batch_loss = []  
            poisoned_data = self.attack.poisoned_data_for_train
            for batch_id, batch in enumerate(poisoned_data):
                if self.attack.label_flip:
                    batch_copy=copy.deepcopy(batch)
                for i in range(self.attack.poisoning_per_batch):
                    if self.attack.label_flip:
                        poison_batch_list = self.attack.poison_images_ind.copy()
                    else:
                        poison_batch_list = self.attack.poison_images.copy()
                    random.shuffle(poison_batch_list)
                    poison_batch_list = poison_batch_list[0 : min(7, len(batch[0]) )]
                    for pos, image in enumerate(poison_batch_list):
                        poison_pos = len(poison_batch_list) * i + pos
                        if self.attack.semantic_backdoor:
                            batch[0][poison_pos] = self.attack.train_dataset[image][0]
                        elif self.attack.label_flip:
                            batch[0][poison_pos] = self.attack.test_dataset[image][0]
                        else:
                            batch[0][poison_pos] = self.attack.add_trigger(batch[0][poison_pos], self.attack.pattern_diff)
                        
                        batch[0][poison_pos].add_(torch.FloatTensor(batch[0][poison_pos].shape).normal_(0, 0.01))

                        if not self.attack.label_flip and not self.attack.semantic_backdoor:
                            if random.random() <= 0.5:
                                batch[1][poison_pos] = self.attack.poison_label_swap
                        else:
                            batch[1][poison_pos] = self.attack.poison_label_swap      
                inputs = batch[0].to(self.device)
                labels = batch[1].to(self.device)

                if self.attack.label_flip:
                    for i in range(self.attack.batch_size):
                        if batch_copy[1][i] == 5:
                            batch_copy[1][i] = self.attack.poison_label_swap
                    inputs_copy = batch_copy[0].to(self.device) 
                    labels_copy = batch_copy[1].to(self.device)
                    inputs = torch.cat((inputs, inputs_copy))
                    labels = torch.cat((labels, labels_copy))

                
                poison_optimizer.zero_grad()
                output = model(inputs)
                loss = self.criterion(output, labels)

                curr_weights = copy.deepcopy(model.state_dict())
                # do delta 
                if self.args.dataset == 'cifar':
                    local_model = ResNet18(num_classes=10)
                elif self.args.dataset == 'mnist':
                    local_model = CNNMnist()
                local_model.to(self.device)
                
                # print(cal(curr_weights))
                with torch.no_grad():
                    for anticipate_i in range(10):
                        if anticipate_i == 0:
                            for w in curr_weights:
                                if global_weights[w].type() == 'torch.cuda.LongTensor':
                                    curr_weights[w] = curr_weights[w].to(torch.long) + global_weights[w].to(torch.long) * 10
                                    curr_weights[w] = curr_weights[w].to(torch.long) / (10.0 + 1.0)
                                else:
                                    curr_weights[w] += global_weights[w].to(torch.long) * 10
                                    curr_weights[w] /= (10.0 + 1.0)
                            
                        else:
                            for w in curr_weights:
                                curr_weights[w] = curr_weights[w] - 0.001*(control_global_w[w])
                        local_model.load_state_dict(curr_weights)

                        # anti_output = local_model(inputs_c.to(self.device))
                        # antiloss = self.criterion(anti_output, labels_c.to(self.device))

                        anti_output = local_model(inputs)
                        antiloss = self.criterion(anti_output, labels)
                        # print(loss, antiloss)
                        loss += antiloss

                loss.backward(retain_graph=True)

                poison_optimizer.step()
                batch_loss.append(loss.item())

                count += 1

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            gc.collect()
            torch.cuda.empty_cache()

        test_loss, accuracy, correct, num_data = self.test_poison(model, self.device)
        print('Test poisoned (before average): Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(test_loss, correct, num_data, accuracy))

        control_delta = copy.deepcopy(self.control_local_w)
        model_weights = model.state_dict()
        local_delta = copy.deepcopy(model_weights)
        for w in model_weights:
            #line 12 in algo
            # self.control_local_w[w] = self.control_local_w[w] - control_global_w[w] + (global_weights[w] - model_weights[w]) / (count * self.args.lr)
            # control_delta[w] = self.control_local_w[w] - control_delta[w]
            self.control_local_w[w] = (global_weights[w] - model_weights[w]) / (count * lr) 
            control_delta[w] = self.control_local_w[w] - control_global_w[w]
            
            local_delta[w] -= global_weights[w]
            if not self.attack.label_flip and not self.attack.semantic_backdoor:
                local_delta[w] = local_delta[w] * 5

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), control_delta, local_delta, self.control_local_w, time.time()- start_time


    def test_poison(self, model, device):
        model.eval()
        total_loss = 0.0
        correct = 0.0
        num_data = 0.0
        for batch_idx, batch in enumerate(self.attack.test_data_poison):
            
            for pos in range(len(batch[0])):
                if self.attack.semantic_backdoor:
                    poison_choice  = random.choice(self.attack.poison_images_test)
                    batch[0][pos] = self.attack.train_dataset[poison_choice][0]
                elif self.attack.label_flip:
                    poison_choice  = random.choice(self.attack.poison_images_ind)
                    batch[0][pos] = self.attack.test_dataset[poison_choice][0]            
                else:
                    batch[0][pos] = self.attack.add_trigger(batch[0][pos], self.attack.pattern_diff)
                
                batch[0][pos].add_(torch.FloatTensor(batch[0][pos].shape).normal_(0, 0.01))

                batch[1][pos] = self.attack.poison_label_swap

            inputs = batch[0].to(device)
            labels = batch[1].to(device)
            inputs.requires_grad_(False)
            labels.requires_grad_(False)

            output = model(inputs)
            criterion = nn.CrossEntropyLoss().to(device)
            total_loss += criterion(output, labels).item()
            # total_loss += nn.functional.cross_entropy(output, labels,reduction='sum').data.item()  # sum up batch loss
            num_data += labels.size(0)
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(labels.data.view_as(pred)).cpu().sum().to(dtype=torch.float)
            
        acc = 100.0 * (float(correct) / float(num_data))
        total_l = total_loss / float(num_data)
        # print('___Test poisoned ( traget label {} ): {}, Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(9,is_poison,total_l, correct, num_data, acc))

        model.train()
        return total_l, acc, correct, num_data


    # neurotoxin
    def update_weights_neuron(self, model, global_round, control_global):

        # Set mode to train model
        model.to(self.device)
        global_weights = copy.deepcopy(model.state_dict())
        model.train()
        epoch_loss = []

        start_time = time.time()
        
        ratio = 0.99
        mask_grad_list = self.grad_mask_cv(model, self.criterion, self.device, ratio)
        
        if self.args.dataset == 'cifar':
            lr = 0.05
            poison_optimizer = torch.optim.SGD(model.parameters(), lr = lr,
                                                    momentum=0.9,
                                                    weight_decay=0.005)
        elif self.args.dataset == 'mnist':
            self.local_ep = 10
            lr = 0.1
            poison_optimizer = torch.optim.SGD(model.parameters(), lr = lr)
        
        control_global_w = control_global.state_dict()
        if self.control_local_w == None:
            self.control_local_w = control_global.state_dict()
        

        count = 0
        for iter in range(self.local_ep):
            batch_loss = []  
            poisoned_data = self.attack.poisoned_data_for_train
            for batch_id, batch in enumerate(poisoned_data):
                if self.attack.label_flip:
                    batch_copy=copy.deepcopy(batch)
                for i in range(self.attack.poisoning_per_batch):
                    if self.attack.label_flip:
                        poison_batch_list = self.attack.poison_images_ind.copy()
                    else:
                        poison_batch_list = self.attack.poison_images.copy()
                    random.shuffle(poison_batch_list)
                    poison_batch_list = poison_batch_list[0 : min(7, len(batch[0]) )]
                    for pos, image in enumerate(poison_batch_list):
                        # import pdb; pdb.set_trace()
                        poison_pos = len(poison_batch_list) * i + pos
                        if self.attack.semantic_backdoor:
                            batch[0][poison_pos] = self.attack.train_dataset[image][0]
                        elif self.attack.label_flip:
                            batch[0][poison_pos] = self.attack.test_dataset[image][0]
                        else:
                            batch[0][poison_pos] = self.attack.add_trigger(batch[0][poison_pos], self.attack.pattern_diff)
                        
                        batch[0][poison_pos].add_(torch.FloatTensor(batch[0][poison_pos].shape).normal_(0, 0.01))
                        
                        if not self.attack.label_flip and not self.attack.semantic_backdoor:
                            if random.random() <= 0.5:
                                batch[1][poison_pos] = self.attack.poison_label_swap
                        else:
                            batch[1][poison_pos] = self.attack.poison_label_swap      
                inputs = batch[0].to(self.device)
                labels = batch[1].to(self.device)

                if self.attack.label_flip:
                    for i in range(self.attack.batch_size):
                        if batch_copy[1][i] == 5:
                            batch_copy[1][i] = self.attack.poison_label_swap
                    inputs_copy = batch_copy[0].to(self.device) 
                    labels_copy = batch_copy[1].to(self.device)
                    inputs = torch.cat((inputs, inputs_copy))
                    labels = torch.cat((labels, labels_copy))

                
                poison_optimizer.zero_grad()
                output = model(inputs)
                loss = self.criterion(output, labels)

                loss.backward(retain_graph=True)
                
                self.apply_grad_mask(model, mask_grad_list)

                poison_optimizer.step()
                batch_loss.append(loss.item())

                count += 1

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            gc.collect()
            torch.cuda.empty_cache()

        test_loss, accuracy, correct, num_data = self.test_poison(model, self.device)
        print('Test poisoned (before average): Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(test_loss, correct, num_data, accuracy))

        control_delta = copy.deepcopy(self.control_local_w)
        #model_weights -> y_(i)
        model_weights = model.state_dict()
        local_delta = copy.deepcopy(model_weights)
        # import pdb; pdb.set_trace()
        for w in model_weights:
            #line 12 in algo
            # self.control_local_w[w] = self.control_local_w[w] - control_global_w[w] + (global_weights[w] - model_weights[w]) / (count * self.args.lr)
            # control_delta[w] = self.control_local_w[w] - control_delta[w]
            self.control_local_w[w] = (global_weights[w] - model_weights[w]) / (count * lr) 
            control_delta[w] = self.control_local_w[w] - control_global_w[w]
            local_delta[w] -= global_weights[w]
        #update new control_local model
        # control_local.load_state_dict(new_control_local_w)

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), control_delta, local_delta, self.control_local_w, time.time()- start_time

    def grad_mask_cv(self, model, criterion, device, ratio=0.5, aggregate_all_layer = 1):
        """Generate a gradient mask based on the given dataset"""
        params_copy = []
        for p in list(model.parameters()):
            params_copy.append(p.clone())
        
        model.train()
        model.zero_grad()

        train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=128, shuffle=True)
        # for batch_idx, (inputs, labels) in enumerate(self.attack.test_data_poison):
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            output = model(inputs)

            loss = criterion(output, labels)
            loss.backward(retain_graph=True)

        mask_grad_list = []
        if aggregate_all_layer == 1:
            grad_list = []
            grad_abs_sum_list = []
            k_layer = 0
            grad_weights_list = []
            for _, parms in model.named_parameters():
                if parms.requires_grad:
                    grad_list.append(parms.grad.abs().view(-1))

                    grad_abs_sum_list.append(parms.grad.abs().view(-1).sum().item())

                    grad_weights_list.append((parms.grad.abs()/parms.data.abs()).view(-1))
                    
                    k_layer += 1
            # import pdb; pdb.set_trace()
            # grad_list = torch.cat(grad_list).to(device)
            # _, indices = torch.topk(-1*grad_list, int(len(grad_list)*ratio))
            # mask_flat_all_layer = torch.zeros(len(grad_list)).to(device)

            grad_weights_list = torch.cat(grad_weights_list).to(device)
            _, indices = torch.topk(-1*grad_weights_list, int(len(grad_weights_list)*ratio))
            mask_flat_all_layer = torch.zeros(len(grad_weights_list)).to(device)

            mask_flat_all_layer[indices] = 1.0

            
            count = 0
            percentage_mask_list = []
            k_layer = 0
            grad_abs_percentage_list = []
            for _, parms in model.named_parameters():
                if parms.requires_grad:
                    gradients_length = len(parms.grad.abs().view(-1))

                    mask_flat = mask_flat_all_layer[count:count + gradients_length ].to(device)
                    mask_grad_list.append(mask_flat.reshape(parms.grad.size()).to(device))

                    count += gradients_length

                    percentage_mask1 = mask_flat.sum().item()/float(gradients_length)*100.0

                    percentage_mask_list.append(percentage_mask1)

                    grad_abs_percentage_list.append(grad_abs_sum_list[k_layer]/np.sum(grad_abs_sum_list))

                    k_layer += 1
        else:
            grad_abs_percentage_list = []
            grad_res = []
            l2_norm_list = []
            sum_grad_layer = 0.0
            for _, parms in model.named_parameters():
                if parms.requires_grad:
                    grad_res.append(parms.grad.view(-1))
                    l2_norm_l = torch.norm(parms.grad.view(-1).clone().detach().to(device))/float(len(parms.grad.view(-1)))
                    l2_norm_list.append(l2_norm_l)
                    sum_grad_layer += l2_norm_l.item()

            grad_flat = torch.cat(grad_res)

            percentage_mask_list = []
            k_layer = 0
            for _, parms in model.named_parameters():
                if parms.requires_grad:
                    gradients = parms.grad.abs().view(-1)
                    gradients_length = len(gradients)
                    if ratio == 1.0:
                        _, indices = torch.topk(-1*gradients, int(gradients_length*1.0))
                    else:

                        ratio_tmp = 1 - l2_norm_list[k_layer].item() / sum_grad_layer
                        _, indices = torch.topk(-1*gradients, int(gradients_length*ratio))

                    mask_flat = torch.zeros(gradients_length)
                    mask_flat[indices.cpu()] = 1.0
                    mask_grad_list.append(mask_flat.reshape(parms.grad.size()).to(device))

                    percentage_mask1 = mask_flat.sum().item()/float(gradients_length)*100.0

                    percentage_mask_list.append(percentage_mask1)


                    k_layer += 1

        model.zero_grad()

        return mask_grad_list

    def apply_grad_mask(self, model, mask_grad_list):
        mask_grad_list_copy = iter(mask_grad_list)
        for name, parms in model.named_parameters():
            if parms.requires_grad:
                parms.grad = parms.grad * next(mask_grad_list_copy)



# class BackdoorUpdate(object):
#     def __init__(self, args, device, test_dataset=None):
#         self.args = args
#         self.test_dataset = test_dataset

#         self.poison_train_dataloader = self.poison_dataset()
#         self.poison_test_dataloader = self.poison_test_dataset()
#         self.local_ep = args.local_ep

#         self.device = device
#         self.criterion = nn.CrossEntropyLoss().to(self.device)

#         self.msecriterion = nn.MSELoss().to(self.device)
        
#         benign_idx = np.random.choice(range(len(test_dataset)), 500, replace=False)
#         self.train_loader = torch.utils.data.DataLoader(self.test_dataset,
#                                            batch_size=self.args.local_bs,
#                                            sampler=torch.utils.data.sampler.SubsetRandomSampler(benign_idx))

#         self.control_local_w = None

#     # original attack
#     def update_weights(self, model, global_round, control_global):
#         benign_idx = np.random.choice(range(len(self.test_dataset)), 500, replace=False)
#         self.train_loader = torch.utils.data.DataLoader(self.test_dataset,
#                                            batch_size=self.args.local_bs,
#                                            sampler=torch.utils.data.sampler.SubsetRandomSampler(benign_idx))

#         # Set mode to train model
#         model.to(self.device)
#         global_weights = copy.deepcopy(model.state_dict())
#         model.train()
#         epoch_loss = []

#         start_time = time.time()
        
#         lr = 0.1
#         poison_optimizer = torch.optim.SGD(model.parameters(), lr = lr,
#                                                     momentum=0.9,
#                                                     weight_decay=0.005)
        
#         control_global_w = control_global.state_dict()
#         if self.control_local_w == None:
#             self.control_local_w = control_global.state_dict()
        
#         count = 0
#         for iter in range(self.local_ep):
#             batch_loss = []                
                
#             for batch_idx, (x1, x2) in enumerate(zip(self.poison_train_dataloader, self.train_loader)):
#                 inputs_p, labels_p = x1
#                 inputs_c, labels_c = x2
#                 inputs = torch.cat((inputs_p,inputs_c))

                

#                 for pos in range(labels_c.size(0)):
#                     if labels_c[pos] == 5:
#                         labels_c[pos] = 7

#                 for pos in range(labels_p.size(0)):
#                     labels_p[pos] = 7 # poison_label_swap

#                 labels = torch.cat((labels_p,labels_c))
#                 inputs = inputs.to(self.device)
#                 labels = labels.to(self.device)
#                 poison_optimizer.zero_grad()
#                 output = model(inputs)
#                 loss = self.criterion(output, labels)
#                 loss.backward(retain_graph=True)

#                 poison_optimizer.step()
#                 batch_loss.append(loss.item())
                
#                 local_weights = model.state_dict()
#                 for w in local_weights:
#                     local_weights[w] = local_weights[w] - self.args.lr*(control_global_w[w]-self.control_local_w[w])
                
#                 #update local model params
#                 model.load_state_dict(local_weights)
                
#                 count += 1

#             epoch_loss.append(sum(batch_loss)/len(batch_loss))
#             gc.collect()
#             torch.cuda.empty_cache()

#         test_loss, accuracy, correct, num_data = self.test_poison(model, self.device)
#         print('Test poisoned (before average): Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(test_loss, correct, num_data, accuracy))

#         control_delta = copy.deepcopy(self.control_local_w)
#         model_weights = model.state_dict()
#         local_delta = copy.deepcopy(model_weights)
#         for w in model_weights:
#             #line 12 in algo
#             self.control_local_w[w] = self.control_local_w[w] - control_global_w[w] + (global_weights[w] - model_weights[w]) / (count * lr)
#             control_delta[w] = self.control_local_w[w] - control_delta[w]
#             # self.control_local_w[w] = (global_weights[w] - model_weights[w]) / (count * lr) 
#             # control_delta[w] = self.control_local_w[w] - control_global_w[w]
#             local_delta[w] -= global_weights[w]

#         return model.state_dict(), sum(epoch_loss) / len(epoch_loss), control_delta, local_delta, self.control_local_w, time.time()- start_time

#     def update_weights_deltac(self, model, global_round, control_global):
#         benign_idx = np.random.choice(range(len(self.test_dataset)), 500, replace=False)
#         self.train_loader = torch.utils.data.DataLoader(self.test_dataset,
#                                             batch_size=self.args.local_bs,
#                                             sampler=torch.utils.data.sampler.SubsetRandomSampler(benign_idx))

#         # Set mode to train model
#         model.to(self.device)
#         global_weights = copy.deepcopy(model.state_dict())
#         model.train()
#         epoch_loss = []

#         start_time = time.time()
        
#         lr = 0.05
#         poison_optimizer = torch.optim.SGD(model.parameters(), lr = lr,
#                                                     momentum=0.9,
#                                                     weight_decay=0.005)
        
#         control_global_w = control_global.state_dict()
#         if self.control_local_w == None:
#             self.control_local_w = control_global.state_dict()
        

#         count = 0
#         for iter in range(self.local_ep):
#             batch_loss = []                
                
#             for batch_idx, (x1, x2) in enumerate(zip(self.poison_train_dataloader, self.train_loader)):
#                 inputs_p, labels_p = x1
#                 inputs_c, labels_c = x2
#                 inputs = torch.cat((inputs_p,inputs_c))

#                 for pos in range(labels_c.size(0)):
#                     if labels_c[pos] == 5:
#                         labels_c[pos] = 7
#                         # print(pos)
#                 for pos in range(labels_p.size(0)):
#                     labels_p[pos] = 7 # poison_label_swap

#                 labels = torch.cat((labels_p,labels_c))
#                 inputs = inputs.to(self.device)
#                 labels = labels.to(self.device)
#                 poison_optimizer.zero_grad()
#                 output = model(inputs)
#                 loss = self.criterion(output, labels)

#                 curr_weights = copy.deepcopy(model.state_dict())
#                 # do delta anticipate
#                 local_model = ResNet18(num_classes=10)
#                 local_model.to(self.device)
                
#                 # print(cal(curr_weights))
#                 with torch.no_grad():
#                     for anticipate_i in range(10):
#                         if anticipate_i == 0:
#                             for w in curr_weights:
#                                 if global_weights[w].type() == 'torch.cuda.LongTensor':
#                                     curr_weights[w] = curr_weights[w].to(torch.long) + global_weights[w].to(torch.long) * 10
#                                     curr_weights[w] = curr_weights[w].to(torch.long) / (10.0 + 1.0)
#                                 else:
#                                     curr_weights[w] += global_weights[w].to(torch.long) * 10
#                                     curr_weights[w] /= (10.0 + 1.0)
                            
#                         else:
#                             for w in curr_weights:
#                                 curr_weights[w] = curr_weights[w] - 0.001*(control_global_w[w])
#                         local_model.load_state_dict(curr_weights)

#                         # anti_output = local_model(inputs_c.to(self.device))
#                         # antiloss = self.criterion(anti_output, labels_c.to(self.device))

#                         anti_output = local_model(inputs)
#                         antiloss = self.criterion(anti_output, labels)
#                         # print(loss, antiloss)
#                         loss += antiloss
                    
#                     # loss += antiloss
                    
#                     # import pdb;pdb.set_trace()

#                 loss.backward(retain_graph=True)

#                 poison_optimizer.step()
#                 batch_loss.append(loss.item())
                
#                 count += 1

#             epoch_loss.append(sum(batch_loss)/len(batch_loss))
#             gc.collect()
#             torch.cuda.empty_cache()

#         test_loss, accuracy, correct, num_data = self.test_poison(model, self.device)
#         print('Test poisoned (before average): Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(test_loss, correct, num_data, accuracy))

#         control_delta = copy.deepcopy(self.control_local_w)
#         #model_weights -> y_(i)
#         model_weights = model.state_dict()
#         local_delta = copy.deepcopy(model_weights)
#         # import pdb; pdb.set_trace()
#         for w in model_weights:
#             #line 12 in algo
#             # self.control_local_w[w] = self.control_local_w[w] - control_global_w[w] + (global_weights[w] - model_weights[w]) / (count * self.args.lr)
#             # control_delta[w] = self.control_local_w[w] - control_delta[w]
#             self.control_local_w[w] = (global_weights[w] - model_weights[w]) / (count * lr) 
#             control_delta[w] = self.control_local_w[w] - control_global_w[w]
#             local_delta[w] -= global_weights[w]
#         #update new control_local model
#         # control_local.load_state_dict(new_control_local_w)
#         # torch.save(self.control_local_w, f'./saved_models/controlbackdoor_test.pth')
#         return model.state_dict(), sum(epoch_loss) / len(epoch_loss), control_delta, local_delta, self.control_local_w, time.time()- start_time

#     def update_weights_backdoor(self, model, global_round, control_global):
#         # Set mode to train model
#         model.to(self.device)
#         global_weights = copy.deepcopy(model.state_dict())
#         model.train()
#         epoch_loss = []

#         start_time = time.time()
        
#         lr = 0.05
#         poison_optimizer = torch.optim.SGD(model.parameters(), lr = lr,
#                                                     momentum=0.9,
#                                                     weight_decay=0.005)
        
#         control_global_w = control_global.state_dict()
#         if self.control_local_w == None:
#             self.control_local_w = control_global.state_dict()
        
#         count = 0
#         for iter in range(self.local_ep):
#             batch_loss = []                
                
#             for batch_idx, (x1, x2) in enumerate(zip(self.poison_train_dataloader, self.train_loader)):
#                 inputs_p, labels_p = x1
#                 inputs_c, labels_c = x2
#                 inputs = torch.cat((inputs_p,inputs_c))

                

#                 for pos in range(labels_c.size(0)):
#                     if labels_c[pos] == 5:
#                         labels_c[pos] = 7

#                 for pos in range(labels_p.size(0)):
#                     labels_p[pos] = 7 # poison_label_swap

#                 labels = torch.cat((labels_p,labels_c))
#                 inputs = inputs.to(self.device)
#                 labels = labels.to(self.device)
#                 poison_optimizer.zero_grad()
#                 output = model(inputs)
#                 loss = self.criterion(output, labels)
#                 loss.backward(retain_graph=True)

#                 poison_optimizer.step()
#                 batch_loss.append(loss.item())
                
#                 local_weights = model.state_dict()
#                 for w in local_weights:
#                     local_weights[w] = local_weights[w] - self.args.lr*(control_global_w[w]-self.control_local_w[w])
                
#                 #update local model params
#                 model.load_state_dict(local_weights)
                
#                 count += 1

#             epoch_loss.append(sum(batch_loss)/len(batch_loss))
#             gc.collect()
#             torch.cuda.empty_cache()

#         test_loss, accuracy, correct, num_data = self.test_poison(model, self.device)
#         print('Test poisoned (before average): Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(test_loss, correct, num_data, accuracy))

#         control_delta = copy.deepcopy(self.control_local_w)
#         model_weights = model.state_dict()
#         local_delta = copy.deepcopy(model_weights)
#         for w in model_weights:
#             #line 12 in algo
#             self.control_local_w[w] = self.control_local_w[w] - control_global_w[w] + (global_weights[w] - model_weights[w]) / (count * lr)
#             control_delta[w] = self.control_local_w[w] - control_delta[w]
#             # self.control_local_w[w] = (global_weights[w] - model_weights[w]) / (count * lr) 
#             # control_delta[w] = self.control_local_w[w] - control_global_w[w]
#             local_delta[w] -= global_weights[w]

#         return model.state_dict(), sum(epoch_loss) / len(epoch_loss), control_delta, local_delta, self.control_local_w, time.time()- start_time


#     def update_weights_avg(self, model, global_round):
#         # Set mode to train model
#         model.to(self.device)
#         global_weights = copy.deepcopy(model.state_dict())
#         model.train()
#         epoch_loss = []

#         start_time = time.time()
        
#         poison_optimizer = torch.optim.SGD(model.parameters(), lr = 0.05,
#                                                     momentum=0.9,
#                                                     weight_decay=0.005)
        
#         if self.control_local_w == None:            
#             self.control_local_w = copy.deepcopy(model.state_dict())
#             for w in self.control_local_w:
#                 self.control_local_w[w].fill_(0)
        

#         count = 0
#         for iter in range(self.local_ep):
#             batch_loss = []            
            
#             # for batch_idx, (inputs, labels) in enumerate(self.poison_train_dataloader):
#             #     inputs, labels = inputs.to(self.device), labels.to(self.device)

#             #     for pos in range(labels.size(0)):
#             #         labels[pos] = 7 # poison_label_swap
#             for batch_idx, (x1, x2) in enumerate(zip(self.poison_train_dataloader, self.train_loader)):
#                 inputs_p, labels_p = x1
#                 inputs_c, labels_c = x2
#                 inputs = torch.cat((inputs_p,inputs_c))

                

#                 for pos in range(labels_c.size(0)):
#                     if labels_c[pos] == 5:
#                         labels_c[pos] = 7

#                 for pos in range(labels_p.size(0)):
#                     labels_p[pos] = 7 # poison_label_swap

#                 labels = torch.cat((labels_p,labels_c))
#                 inputs = inputs.to(self.device)
#                 labels = labels.to(self.device)
#                 poison_optimizer.zero_grad()
#                 output = model(inputs)
#                 loss = self.criterion(output, labels)
#                 loss.backward(retain_graph=True)

#                 poison_optimizer.step()
#                 batch_loss.append(loss.item())

#                 # local_weights = model.state_dict()
#                 # for w in local_weights:
#                 #     #line 10 in algo 
#                 #     # import pdb;pdb.set_trace()
#                 #     local_weights[w] = local_weights[w] - self.args.lr*(self.control_local_w[w])
                
#                 # #update local model params
#                 # model.load_state_dict(local_weights)
                
#                 count += 1

#             local_weights = model.state_dict()
#             for w in local_weights:
#                 local_weights[w] = 0.5 * local_weights[w] + 0.5 * global_weights[w]
            
#             #update local model params
#             model.load_state_dict(local_weights)
#             epoch_loss.append(sum(batch_loss)/len(batch_loss))
#             gc.collect()
#             torch.cuda.empty_cache()

#         test_loss, accuracy, correct, num_data = self.test_poison(model, self.device)
#         print('Test poisoned (before average): Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(test_loss, correct, num_data, accuracy))

#         #model_weights -> y_(i)
#         model_weights = model.state_dict()
#         local_delta = copy.deepcopy(model_weights)
#         # import pdb; pdb.set_trace()
#         for w in model_weights:
#             #line 12 in algo
#             self.control_local_w[w] = (global_weights[w] - model_weights[w])
#             #line 13
#             local_delta[w] -= global_weights[w]

        
#         return model.state_dict(), sum(epoch_loss) / len(epoch_loss), None, local_delta, None, time.time()- start_time

#     def update_weights_clf(self, model, global_round, control_local, control_global):
#         # Set mode to train model
#         model.to(self.device)
#         global_weights = copy.deepcopy(model.state_dict())
#         model.train()
#         epoch_loss = []

#         start_time = time.time()
        
#         poison_optimizer = torch.optim.SGD(model.parameters(), lr = 0.001,
#                                                     momentum=0.9,
#                                                     weight_decay=0.005)
        
#         control_global_w = control_global.state_dict()
#         if self.control_local_w == None:
#             self.control_local_w = control_local.state_dict()
        
#         keys = list(model.state_dict().keys())

#         clf_layers = [k for k in keys if k.startswith('linear')]
#         for k, v in model.named_parameters():
#             if k in clf_layers:
#                 v.requires_grad_(False)

#         count = 0
#         for iter in range(self.local_ep):
#             batch_loss = []                
                
#             for batch_idx, (x1, x2) in enumerate(zip(self.poison_train_dataloader, self.train_loader)):
#                 inputs_p, labels_p = x1
#                 inputs_c, labels_c = x2
#                 inputs = torch.cat((inputs_p,inputs_c))

                

#                 for pos in range(labels_c.size(0)):
#                     if labels_c[pos] == 5:
#                         labels_c[pos] = 7

#                 for pos in range(labels_p.size(0)):
#                     labels_p[pos] = 7 # poison_label_swap

#                 labels = torch.cat((labels_p,labels_c))
#                 inputs = inputs.to(self.device)
#                 labels = labels.to(self.device)
#                 poison_optimizer.zero_grad()
#                 output = model(inputs)
#                 loss = self.criterion(output, labels)
#                 loss.backward(retain_graph=True)

#                 poison_optimizer.step()
#                 batch_loss.append(loss.item())
                
#                 # local_weights = model.state_dict()
#                 # for w in local_weights:
#                 #     #line 10 in algo 
#                 #     # import pdb;pdb.set_trace()
#                 #     local_weights[w] = local_weights[w] - self.args.lr*(control_global_w[w]-self.control_local_w[w])
                
#                 # #update local model params
#                 # model.load_state_dict(local_weights)
                
#                 count += 1

#             epoch_loss.append(sum(batch_loss)/len(batch_loss))
#             gc.collect()
#             torch.cuda.empty_cache()

#         test_loss, accuracy, correct, num_data = self.test_poison(model, self.device)
#         print('Test poisoned (before average): Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(test_loss, correct, num_data, accuracy))

#         control_delta = copy.deepcopy(self.control_local_w)
#         #model_weights -> y_(i)
#         model_weights = model.state_dict()
#         local_delta = copy.deepcopy(model_weights)
#         # import pdb; pdb.set_trace()
#         for w in model_weights:
#             #line 12 in algo
#             # self.control_local_w[w] = self.control_local_w[w] - control_global_w[w] + (global_weights[w] - model_weights[w]) / (count * self.args.lr)
#             self.control_local_w[w] = (global_weights[w] - model_weights[w]) / (count * self.args.lr)
#             #line 13
#             control_delta[w] = self.control_local_w[w] - control_global_w[w]
#             local_delta[w] -= global_weights[w]
#         #update new control_local model
#         # control_local.load_state_dict(new_control_local_w)
#         # torch.save(self.control_local_w, f'./saved_models/controlbackdoor_test.pth')
#         return model.state_dict(), sum(epoch_loss) / len(epoch_loss), control_delta, local_delta, self.control_local_w, time.time()- start_time

#     def update_weights_optim(self, model, global_round, control_global):
#         # Set mode to train model
#         model.to(self.device)
#         global_weights = copy.deepcopy(model.state_dict())
#         model.train()
#         epoch_loss = []

#         start_time = time.time()
        
#         poison_optimizer = torch.optim.SGD(model.parameters(), lr = 0.001,
#                                                     momentum=0.9,
#                                                     weight_decay=0.005)
        
#         control_global_w = control_global.state_dict()
#         if self.control_local_w == None:
#             self.control_local_w = control_global.state_dict()
        
#         count = 0
#         for iter in range(self.local_ep):
#             batch_loss = []                
                
#             for batch_idx, (x1, x2) in enumerate(zip(self.poison_train_dataloader, self.train_loader)):
#                 inputs_p, labels_p = x1
#                 inputs_c, labels_c = x2
#                 inputs = torch.cat((inputs_p,inputs_c))

                

#                 for pos in range(labels_c.size(0)):
#                     if labels_c[pos] == 5:
#                         labels_c[pos] = 7

#                 for pos in range(labels_p.size(0)):
#                     labels_p[pos] = 7 # poison_label_swap

#                 labels = torch.cat((labels_p,labels_c))
#                 inputs = inputs.to(self.device)
#                 labels = labels.to(self.device)
#                 poison_optimizer.zero_grad()
#                 output = model(inputs)
#                 loss = self.criterion(output, labels)

#                 model_weights = model.state_dict()
#                 temp_local_delta = copy.deepcopy(model_weights)
#                 for w in model_weights:
#                     temp_local_delta[w] = (global_weights[w] - model_weights[w]) / (40 * 0.001)
#                 mse_loss = self.msecriterion(torch.cat([v.view(-1) for v in control_global_w.values()]),
#                             torch.cat([v.view(-1) for v in temp_local_delta.values()]))
                
#                 alpha = 0.0001  # You can adjust this weight depending on the importance of each loss
#                 total_loss = alpha * mse_loss + (1 - alpha) * loss
#                 # import pdb; pdb.set_trace()
#                 total_loss.backward(retain_graph=True)

#                 poison_optimizer.step()
#                 batch_loss.append(loss.item())
                
#                 count += 1

#             epoch_loss.append(sum(batch_loss)/len(batch_loss))
#             gc.collect()
#             torch.cuda.empty_cache()

#         test_loss, accuracy, correct, num_data = self.test_poison(model, self.device)
#         print('Test poisoned (before average): Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(test_loss, correct, num_data, accuracy))

#         control_delta = copy.deepcopy(self.control_local_w)
#         #model_weights -> y_(i)
#         model_weights = model.state_dict()
#         local_delta = copy.deepcopy(model_weights)
#         # import pdb; pdb.set_trace()
#         for w in model_weights:
#             #line 12 in algo
#             # self.control_local_w[w] = self.control_local_w[w] - control_global_w[w] + (global_weights[w] - model_weights[w]) / (count * self.args.lr)
#             self.control_local_w[w] = (global_weights[w] - model_weights[w]) / (count * self.args.lr)  * 5
#             #line 13
#             control_delta[w] = self.control_local_w[w] - control_global_w[w]
#             local_delta[w] -= global_weights[w]
#         #update new control_local model
#         # control_local.load_state_dict(new_control_local_w)
#         # torch.save(self.control_local_w, f'./saved_models/controlbackdoor_test.pth')
#         return model.state_dict(), sum(epoch_loss) / len(epoch_loss), control_delta, local_delta, self.control_local_w, time.time()- start_time

#     def update_weights_biggan(self, model, global_round, control_global):

#         self.trainloader = torch.load('./data/biggan/data.pth')
#         model.to(self.device)
#         global_weights = copy.deepcopy(model.state_dict())
#         model.train()
#         epoch_loss = []

#         start_time = time.time()
        
#         lr = 0.05
#         poison_optimizer = torch.optim.SGD(model.parameters(), lr = lr,
#                                                     momentum=0.9,
#                                                     weight_decay=0.005)
        
#         control_global_w = control_global.state_dict()
#         if self.control_local_w == None:
#             self.control_local_w = control_global.state_dict()

#         count = 0
#         for iter in range(self.local_ep):
#             batch_loss = []
#             for batch_idx, (images, labels) in enumerate(self.trainloader):
#                 images, labels = images.to(self.device), labels.to(self.device)
#                 # import pdb; pdb.set_trace()
#                 model.zero_grad()
#                 log_probs = model(images)
#                 loss = self.criterion(log_probs, labels)
#                 loss.backward()
                
#                 poison_optimizer.step()

#                 batch_loss.append(loss.item())
                
#                 count += 1
                
#             epoch_loss.append(sum(batch_loss)/len(batch_loss))
#             gc.collect()
#             torch.cuda.empty_cache()


#         control_delta = copy.deepcopy(self.control_local_w)
#         #model_weights -> y_(i)
#         model_weights = model.state_dict()
#         local_delta = copy.deepcopy(model_weights)
#         # import pdb; pdb.set_trace()
#         for w in model_weights:
#             # self.control_local_w[w] = (global_weights[w] - model_weights[w]) / (count * lr) 
#             # control_delta[w] = self.control_local_w[w] - control_global_w[w]
#             control_delta[w] = control_global_w[w] - control_global_w[w]
#             local_delta[w] -= global_weights[w]
#         return model.state_dict(), sum(epoch_loss) / len(epoch_loss), control_delta, local_delta, self.control_local_w, time.time()- start_time

#     def update_weights_biggan(self, model, global_round, control_global):

#         self.trainloader = torch.load('./data/biggan/data.pth')
#         model.to(self.device)
#         global_weights = copy.deepcopy(model.state_dict())
#         model.train()
#         epoch_loss = []

#         start_time = time.time()
        
#         lr = 0.05
#         poison_optimizer = torch.optim.SGD(model.parameters(), lr = lr,
#                                                     momentum=0.9,
#                                                     weight_decay=0.005)
        
#         control_global_w = control_global.state_dict()
#         if self.control_local_w == None:
#             self.control_local_w = control_global.state_dict()

#         count = 0
#         for iter in range(self.local_ep):
#             batch_loss = []
#             for batch_idx, (images, labels) in enumerate(self.trainloader):
#                 images, labels = images.to(self.device), labels.to(self.device)
#                 # import pdb; pdb.set_trace()
#                 model.zero_grad()
#                 log_probs = model(images)
#                 loss = self.criterion(log_probs, labels)
#                 loss.backward()
                
#                 poison_optimizer.step()

#                 batch_loss.append(loss.item())
                
#                 count += 1
                
#             epoch_loss.append(sum(batch_loss)/len(batch_loss))
#             gc.collect()
#             torch.cuda.empty_cache()


#         control_delta = copy.deepcopy(self.control_local_w)
#         #model_weights -> y_(i)
#         model_weights = model.state_dict()
#         local_delta = copy.deepcopy(model_weights)
#         # import pdb; pdb.set_trace()
#         for w in model_weights:
#             # self.control_local_w[w] = (global_weights[w] - model_weights[w]) / (count * lr) 
#             # control_delta[w] = self.control_local_w[w] - control_global_w[w]
#             control_delta[w] = control_global_w[w] - control_global_w[w]
#             local_delta[w] -= global_weights[w]
#         return model.state_dict(), sum(epoch_loss) / len(epoch_loss), control_delta, local_delta, self.control_local_w, time.time()- start_time


#     def update_weights_prox(self, model, global_round):
#         # Set mode to train model
#         model.to(self.device)
#         global_weights = copy.deepcopy(model.state_dict())
#         model.train()
#         epoch_loss = []

#         start_time = time.time()
        
#         poison_optimizer = torch.optim.SGD(model.parameters(), lr = 0.05,
#                                                     momentum=0.9,
#                                                     weight_decay=0.005)     
        
#         count = 0
#         for iter in range(self.local_ep):
#             batch_loss = []                
#             for batch_idx, (x1, x2) in enumerate(zip(self.poison_train_dataloader, self.train_loader)):
#                 inputs_p, labels_p = x1
#                 inputs_c, labels_c = x2
#                 inputs = torch.cat((inputs_p,inputs_c))

                

#                 for pos in range(labels_c.size(0)):
#                     if labels_c[pos] == 5:
#                         labels_c[pos] = 7

#                 for pos in range(labels_p.size(0)):
#                     labels_p[pos] = 7 # poison_label_swap

#                 labels = torch.cat((labels_p,labels_c))
#                 inputs = inputs.to(self.device)
#                 labels = labels.to(self.device)
#                 poison_optimizer.zero_grad()
#                 output = model(inputs)

#                 proximal_term = 0.0
#                 local_weights = model.state_dict()
#                 for w in local_weights:
#                     proximal_term += (local_weights[w].to(torch.float32)-global_weights[w].to(torch.float32)).norm(2)

#                 loss = self.criterion(output, labels) + (self.args.mu / 2) * proximal_term

#                 # loss = self.criterion(output, labels)
#                 loss.backward(retain_graph=True)

#                 poison_optimizer.step()
#                 batch_loss.append(loss.item())
                
#                 # local_weights = model.state_dict()
#                 # for w in local_weights:
#                 #     #line 10 in algo 
#                 #     # import pdb;pdb.set_trace()
#                 #     local_weights[w] = local_weights[w] - self.args.lr*(control_global_w[w]-control_local_w[w])
                
#                 # #update local model params
#                 # model.load_state_dict(local_weights)
                
#                 count += 1

#             epoch_loss.append(sum(batch_loss)/len(batch_loss))
#             gc.collect()
#             torch.cuda.empty_cache()

#         test_loss, accuracy, correct, num_data = self.test_poison(model, self.device)
#         print('Test poisoned (before average): Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(test_loss, correct, num_data, accuracy))
#         #model_weights -> y_(i)
#         model_weights = model.state_dict()
#         local_delta = copy.deepcopy(model_weights)
#         for w in model_weights:
#             local_delta[w] -= global_weights[w]
        
#         return model.state_dict(), sum(epoch_loss) / len(epoch_loss), None, local_delta, None, time.time()- start_time

#     def backdoor_train(self, global_model, criterion, optimizer, device, id, ratio = 1.0):
#         # set to train mode: enabling dropout layers and batch normalization layers, and keeping track of gradients for parameter updates.
#         global_model.train()

#         running_loss = 0.0

#         if ratio != 1:
#             mask_grad_list = self.grad_mask_cv(global_model, criterion, device, ratio)

#         poison_optimizer = torch.optim.SGD(global_model.parameters(), lr = 0.05,
#                                                     momentum=0.9,
#                                                     weight_decay=0.005)
#         for iter in range(self.args.local_ep):
#             for batch_idx, (x1, x2) in enumerate(zip(self.poison_train_dataloader, self.benign_train_loaders[id])):
#                 inputs_p, labels_p = x1
#                 inputs_c, labels_c = x2
#                 inputs = torch.cat((inputs_p,inputs_c))

                

#                 for pos in range(labels_c.size(0)):
#                     if labels_c[pos] == 5:
#                         labels_c[pos] = 7

#                 for pos in range(labels_p.size(0)):
#                     labels_p[pos] = 7 # poison_label_swap

#                 labels = torch.cat((labels_p,labels_c))
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)
#                 poison_optimizer.zero_grad()
#                 output = global_model(inputs)
#                 loss = criterion(output, labels)
#                 loss.backward(retain_graph=True)

#                 if ratio != 1:
#                     # import pdb; pdb.set_trace()
#                     self.apply_grad_mask(global_model, mask_grad_list)

#                 poison_optimizer.step()

#                 running_loss += loss.item()
    
#     def sample_poison_data(self, target_class):
#         cifar_poison_classes_ind = []
#         label_list = []
#         for ind, x in enumerate(self.test_dataset):
#             imge, label = x
#             label_list.append(label)
#             if label == target_class:
#                 cifar_poison_classes_ind.append(ind)


#         return cifar_poison_classes_ind
    
#     def poison_dataset(self):
#         indices = list()

#         range_no_id = list(range(10000))
#         range_no_id = self.sample_poison_data(5)
#         while len(indices) < 500:
#             range_iter = random.sample(range_no_id, np.min([self.args.local_bs, len(range_no_id) ]))
#             indices.extend(range_iter)

#         self.poison_images_ind = indices

#         return DataLoader(self.test_dataset,
#                           batch_size=self.args.local_bs, 
#                           sampler=torch.utils.data.sampler.SubsetRandomSampler(self.poison_images_ind))
    
#     def poison_test_dataset(self):
#         return DataLoader(self.test_dataset,
#                             # batch_size=self.args.local_bs,
#                             batch_size=500,
#                             sampler=torch.utils.data.sampler.SubsetRandomSampler(
#                                 self.poison_images_ind
#                             ))

#     def test_poison(self, model, device):
#         model.eval()
#         total_loss = 0.0
#         correct = 0.0
#         num_data = 0.0
#         for batch_idx, (inputs, labels) in enumerate(self.poison_test_dataloader):

#             for pos in range(labels.size(0)):
#                 labels[pos] = 7

#             inputs = inputs.to(device)
#             labels = labels.to(device)
#             inputs.requires_grad_(False)
#             labels.requires_grad_(False)

#             output = model(inputs)
#             criterion = nn.CrossEntropyLoss().to(device)
#             total_loss += criterion(output, labels).item()
#             # total_loss += nn.functional.cross_entropy(output, labels,reduction='sum').data.item()  # sum up batch loss
#             num_data += labels.size(0)
#             pred = output.data.max(1)[1]  # get the index of the max log-probability
#             correct += pred.eq(labels.data.view_as(pred)).cpu().sum().to(dtype=torch.float)
            
#         acc = 100.0 * (float(correct) / float(num_data))
#         total_l = total_loss / float(num_data)
#         # print('___Test poisoned ( traget label {} ): {}, Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(9,is_poison,total_l, correct, num_data, acc))

#         model.train()
#         return total_l, acc, correct, num_data
    

