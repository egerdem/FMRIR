import numpy as np
import torch as th
from torch.utils.data import DataLoader
import time

import src.utils as utils

class Trainer:
    def __init__(self, config, net, dataset):
        '''
        :param config: a dict containing parameters
        :param net: the network to be trained, must be of type src.utils.Net
        :param dataset: the dataset to be trained on
        '''
        self.config = config
        self.dataset = dataset
        gpus = [i for i in range(config["num_gpus"])]

        self.dataloader = DataLoader(self.dataset, batch_size=config["batch_size"], shuffle=False, num_workers=1)
        self.net = th.nn.DataParallel(net, gpus)
        weights = filter(lambda x: x.requires_grad, self.net.parameters())
        self.optimizer = utils.NewbobAdam(weights,
                                          net,
                                          artifacts_dir = config["artifacts_dir"],
                                          initial_learning_rate = config["learning_rate"],
                                          decay = config["newbob_decay"],
                                          max_decay = config["newbob_max_decay"],
                                          timestamp = config["timestamp"]
                                          )
        self.lsd_loss = utils.LSD()
        self.total_iters = 0
        
        self.net.train()

    def save(self, suffix=''):
        self.net.module.save(self.config["artifacts_dir"], suffix)
    
    def train(self):
        for epoch in self.dataloader:
            loss_stats = {}
            itr = 0
            for data in self.dataloader:
                loss_new = self.train_iteration(data)
                # logging
                for k,v in loss_new.items():
                    loss_stats[k] = loss_stats[k] + v if k in loss_stats else v
                if itr % round(len(self.dataloader)/5) == 0:
                    print(f'{itr:04}'+ "/" + str(len(self.dataloader)))
                itr += 1
            for k in loss_stats:
                loss_stats[k] /= len(self.dataloader)
            self.optimizer.update_lr(loss_stats["accumulated_loss"])
            loss_str = "    ".join([f"{k}:{v:.4}" for k, v in loss_stats.items()])
            print(f"epoch {epoch+1} " + loss_str)
            # Save model
            if self.config["save_frequency"] > 0 and (epoch + 1) % self.config["save_frequency"] == 0:
                self.save(suffix='epoch-' + str(epoch+1))
                print("Saved model")
        # Save final model
        self.save()

    def train_1ep(self, epoch, coeff=None, mode='train'):
        t_start = time.time()
        loss_stats = {}
        num_data = np.ceil(min(
            sum([len(self.config['src_index'][dataset_name]['train']) for dataset_name in self.config['dataset']])/self.config["batch_size"], len(self.dataloader)
            ))
        print(f"len(dataloader):{len(self.dataloader)}, num_data:{num_data}")

        num_mes_list = self.config["num_mes_list"]
        #if self.config["several_num_mes"]:
        #    self.num_pts_init = self.config["num_mes"]
        #    num_mes_list = self.config["num_mes_list"]
        #else:
        #    num_mes_list = [self.config["num_mes"]]
        
        for itr, data in enumerate(self.dataloader):
            #print(f'itr:{itr}')
            #print(data.keys())
            for itr_nummes, num_mes in enumerate(num_mes_list):
                self.config["num_mes"] = num_mes
                if itr > num_data-1:
                    break
                if self.config["model"] in ["FSMPAE", "FSMPAEPI"]:
                    bs = self.config["batch_size"]
                    slice = range(itr*bs, itr*bs + min(bs, data["src_position"].shape[0]))
                    label = th.tensor(slice).cuda()
                    loss_new = self.train_iteration(data, src=label, itr=itr, itr_nummes=itr_nummes, mode=mode)
                
                # logging
                for k, v in loss_new.items():
                    loss_stats[k] = loss_stats[k]+v if k in loss_stats else v

            #===== progress bar ======
            prog_step = 20
            if round(num_data/prog_step) != 0:
                if itr == 0:
                    print('[',end='')
                elif itr == num_data-1:
                    print('#]')
                elif itr % round(num_data/prog_step) == 0:
                    print('#',end='')
                    # print(f'{itr:03}'+ "/" + str(len(self.dataloader)))
            #=========================

        for k in loss_stats:
            loss_stats[k] /= (num_data*len(num_mes_list))
        t_end = time.time()
        
        # loss_str = "    ".join([f"{k}:{v:.4}" for k, v in loss_stats.items()])
        loss_str = "    ".join([f"{k}:{loss_stats[k]:.4}" for k in sorted(loss_stats.keys())])
        time_str = f"({time.strftime('%H:%M:%S', time.gmtime(t_end-t_start))})"
        print(f"epoch {epoch} (train) ")
        print(loss_str + "        " + time_str)
        if self.config["model"] in ["FSMPAE", "FSMPAEPI"]:
            return loss_stats
        else:
            raise NotImplementedError

    def train_iteration(self, data, src, coeff=None, itr=1, itr_nummes=0, mode='train'):
        '''
        one optimization step
        :param data: tuple of tensors containing source position, hrtf
        :return: dict containing values for all different losses
        '''
        returns = {}
        loss_dict = {}
        # forward
        self.optimizer.zero_grad()
        for data_type in self.config["data_type_interp"]:
            data[data_type] = data[data_type].cuda()
        prediction = self.net.forward(data=data, mode=mode)
        for data_type in self.config["data_type_interp"]:
            if data_type == 'atf_mag':
                loss_dict["lsd"] = self.lsd_loss(prediction['atf_mag'], data['atf_mag'], dim=-1, data_type='atf_mag', mean=True) # S, M, F -> scaler

        # plot for debugging (to be implemented)
        
        loss = 0
        for k,v in loss_dict.items():
            if k in self.config["loss_weights"]:
                if self.config["loss_weights"][k] > 0:
                    loss += self.config["loss_weights"][k] * v
            returns[k] = v.detach().clone()
        returns["loss"] = loss.detach().clone()
        # update model parameters
        loss.backward()
        self.optimizer.step()
        self.total_iters += 1

        return returns
