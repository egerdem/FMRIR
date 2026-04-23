import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import torch as th
import torch.nn as nn
import argparse
import pprint
import os
import datetime
import time

from torch.utils.data import DataLoader
import scipy.special as special
import scipy.spatial.distance as distfuncs

import copy

from torch.utils.tensorboard import SummaryWriter

import src.dataset as dataset
import src.models as models
import src.trainer as trainer
from src.configs import *
import src.utils as utils

from icecream import ic
ic.configureOutput(includeContext=True)

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--artifacts_directory", type=str, default="", help="Directory to save artifacts")
    parser.add_argument("-m", "--model", type=str, default="FSMPAE", help="Model name")
    parser.add_argument("-c", "--idx_config", type=int, default=1, help="index of config")
    parser.add_argument("-test", "--test", action="store_true", help="")
    parser.add_argument("-d", "--data_dir", type=str, default="", help="Root directory containing dataset folders")
    parser.add_argument("--freq_from", type=int, default=0, help="Lower freq bin index (inclusive). Default 0.")
    parser.add_argument("--freq_up_to", type=int, default=None, help="Upper freq bin index (exclusive). Default None = use config num_freq (all bins).")
    parser.add_argument("--num_mes_test", type=int, default=None, help="Number of input microphones at test time. Overrides config num_mes_test. Output filename will include _M{N}.")
    parser.add_argument("--dataset_version", type=str, default="r1", choices=["r1", "r2", "r3", "r4"],
                        help="Processed-dataset version override for .pt loading (r1/r2/r3/r4).")
    parser.add_argument("--use_npz", action="store_true",
                        help="Force loading from per-source .npz files (disable processed .pt loading).")
    parser.add_argument("--use_processed_pt", action="store_true",
                        help="Force loading from processed .pt files (default behavior).")
    parser.add_argument("--metrics_only", action="store_true",
                        help="Test mode only: compute and print metrics without plotting PDFs or saving prediction tensors.")
    args = parser.parse_args()
    return args

def train(trainer, ptins=None):
    BEST_LOSS = 1e+16
    LAST_SAVED = -1
    best_epoch = None
    best_elapsed_seconds = None
    total_epoch_seconds = 0.0
    training_start_time = time.time()

    def _format_duration(seconds):
        return time.strftime('%H:%M:%S', time.gmtime(max(0, int(seconds))))
    for epoch in range(start_ep+1, start_ep+config["epochs"]+1):
        epoch_start_time = time.time()
        trainer.net.cuda()
        trainer.net.train()

        loss_train = trainer.train_1ep(epoch)
        if config["lr_update"] == 'train':
            ## Update Learning rate
            trainer.optimizer.update_lr_one(loss_train["loss"])
        ## TensorBoard
        for k, v in loss_train.items():
            writer.add_scalar(k + " / train", v, epoch)

        ## Validation
        print("---------------------")
        print(f"epoch {epoch} (valid)")
        t_start = time.time()
        
        loss_valid = test(trainer.net, flg_save, BEST_LOSS, validdataset, 'valid', use_cuda=True)
        t_end = time.time()
        time_str = f"({time.strftime('%H:%M:%S', time.gmtime(t_end-t_start))})"
        print(time_str)
        ## TensorBoard
        for k, v in loss_valid.items():
            writer.add_scalar(k + " / valid", v, epoch)
        if config["lr_update"] == 'valid':
            ## Update Learning rate
            trainer.optimizer.update_lr_one(loss_valid["loss"])
        
        if config["lr_update"] == 'step':
            if epoch in config["lr_milestones"]:
                trainer.optimizer.update_lr_step(config["lr_gamma"])

        if loss_valid["loss"] < BEST_LOSS:
            BEST_LOSS = loss_valid["loss"]
            LAST_SAVED = epoch
            best_epoch = epoch
            best_elapsed_seconds = time.time() - training_start_time
            print(
                f"Best Loss. Saving model! Last saved: {LAST_SAVED} (elapsed: {_format_duration(best_elapsed_seconds)}) "
            )
            trainer.save(suffix="best")
        elif config["save_frequency"] > 0 and epoch % config["save_frequency"] == 0 and not epoch == config["epochs"]:
            print("Saving model!")
            trainer.save(suffix="log_"+f'{epoch:03}'+"ep")
        else:
            print("Not saving model! Last saved: {}".format(LAST_SAVED))
        total_epoch_seconds += (time.time() - epoch_start_time)
        print("---------------------")
    # Save final model
    trainer.save(suffix="final_"+f'{epoch:03}'+"ep")

    total_training_seconds = time.time() - training_start_time
    num_epochs_done = max(0, config["epochs"])
    avg_epoch_seconds = total_epoch_seconds / num_epochs_done if num_epochs_done > 0 else 0.0

    print("\n=== TRAINING TIME SUMMARY ===")
    print(f"Total training time: {_format_duration(total_training_seconds)}")
    print(f"Average time per epoch: {_format_duration(avg_epoch_seconds)}")
    if best_epoch is not None:
        print(f"Time to best epoch ({best_epoch}): {_format_duration(best_elapsed_seconds)}")
    print("=============================\n")

def test(net, flg_save, best_loss, data, mode, use_coeff=False, coeff=None, use_cuda=True):
    device = 'cuda' if (use_cuda and th.cuda.is_available()) else 'cpu'
    # net.cpu()
    net.to(device)
    net.eval()
    use_cuda_forward = True

    if mode.startswith('train'):
        mode_str = 'train'
    elif mode.startswith('valid'):
        mode_str = 'valid'
    elif mode.startswith('test'):
        mode_str = 'test'
    else:
        raise NotImplementedError

    num_src = sum([len(config['src_index'][dataset_name][mode_str]) for dataset_name in config['dataset']])
    src_list = np.arange(0, num_src)

    returns = {}
    prediction_all = {}
    keys = []

    for data_type in config["data_type_interp"]:
        if hasattr(data[data_type], 'device'):
            data[data_type] = data[data_type].to(device)
        elif data_type == 'atf_mag':
            keys.append("lsd")
            lsd_loss = utils.LSD()

    for k in keys:
        returns.setdefault(k,0)
    for index in src_list:
        dataset_name = config["Table"][mode_str]["dataset"][index]
        src_id = config["Table"][mode_str]["src_index"][index] - config['src_index'][dataset_name][mode_str][0]
        data_slice = {}
        for data_type in data:
            data_slice[data_type] = data[data_type][dataset_name][..., src_id:src_id+1].permute([-1] + th.arange(data[data_type][dataset_name].dim())[:-1].tolist()).to(device)
        data_slice['dataset'] = [dataset_name]
        data_slice['src_index'] = [src_id]
        
        prediction = net.forward(data=data_slice.copy(), mode=mode)
        for k in prediction:
            if hasattr(prediction[k], 'detach'):
                prediction[k] = prediction[k].detach().clone()
 
        additional_data_type = []#['lsd_bm']

        for data_type in config["data_type_interp"] + additional_data_type:
            if data_type == 'atf_mag':
                returns["lsd"] += lsd_loss(prediction['atf_mag'], data_slice['atf_mag'], dim=-1, data_type='atf_mag', mean=True) # S,M,F
            elif data_type == 'lsd_bm':
                prediction['lsd_bm'] = lsd_loss(prediction['atf_mag'], data_slice['atf_mag'], dim=-1, data_type='atf_mag', mean=False) # S,M
            
            if not data_type in prediction_all:
                prediction_all[data_type] = {}
            if not dataset_name in prediction_all[data_type]:
                if data_type in data:
                    prediction_all[data_type][dataset_name] = th.zeros_like(data[data_type][dataset_name])
                else:
                    if data_type == 'lsd_bm':
                        prediction_all[data_type][dataset_name] = th.zeros_like(data['atf_mag'][dataset_name][:,:,0,:])
            prediction_all[data_type][dataset_name][..., src_id:src_id+1] = prediction[data_type].permute(th.arange(prediction[data_type].dim())[1:].tolist() + [0])
        
        if (not config.get("metrics_only", False)) and (flg_save or src_id == config['src_index'][dataset_name][mode_str][0] - config['src_index'][dataset_name][mode_str][0]):
            if 'atf_mag' in config["data_type_interp"]:
                #=== plot atf_mag ====
                utils.plotatfmag(micpos=data_slice['mic_position'][0,:,:].cpu(), sig_gt=data_slice['atf_mag'][0,:,:], sig_pred=prediction['atf_mag'][0,:,:], config=config, idx_plot_list=np.linspace(0, data_slice['atf_mag'].shape[1]-1, 5, dtype=int),  figdir=f"{config['artifacts_dir']}/figure/atf/", fname=f"{config['artifacts_dir']}/figure/atf/ATF_Mag_{data_slice['dataset'][0]}-src-{data_slice['src_index'][0]+1}_{mode}", data_type = 'atf_mag')
                
        if (not config.get("metrics_only", False)) and flg_save and src_id == config['src_index'][dataset_name][mode_str][-1] - config['src_index'][dataset_name][mode_str][0]:
            for data_type in prediction_all:
                pt_dir = f'{config["artifacts_dir"]}/{data_type}'
                os.makedirs(pt_dir, exist_ok=True)
                for dataset_name in prediction_all[data_type]:
                    m_suffix = f'_M{config["num_mes_test"]}'
                    th.save(prediction_all[data_type][dataset_name], f'{pt_dir}/{data_type}_{mode}_{dataset_name}{m_suffix}.pt')

    loss = 0
    for k in keys:
        if k in config["loss_weights"] and config["loss_weights"][k] > 0:
            loss += config["loss_weights"][k] * returns[k]
    returns["loss"] = loss.detach().clone()
    for k in returns:
        returns[k] /= len(src_list)

    loss_str = "    ".join([f"{k}:{returns[k]:.4}" for k in sorted(returns.keys())])
    print(loss_str)
    
    if returns["loss"] < best_loss:
        print("Best Loss.")

    return returns

def trainNF(config, net, dataset, mode="train", use_cuda=True):
    BEST_LOSS = 1e+16
    LAST_SAVED = -1

    device = 'cuda' if use_cuda else 'cpu'
    gpus = [i for i in range(config["num_gpus"])]

    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False, num_workers=1)
    net = th.nn.DataParallel(net, gpus)
    #weights = filter(lambda x: x.requires_grad, net.parameters())
    optimizer = th.optim.Adam(net.parameters(), lr=config['learning_rate'])
    lsd_loss = utils.LSD()

    num_mes_list = config["num_mes_list"]

    start_ep = 0
    for epoch in range(start_ep+1, start_ep+config["epochs"]+1):
        t_start = time.time()

        num_data = np.ceil(min(
            sum([len(config['src_index'][dataset_name][mode]) for dataset_name in config['dataset']])/config["batch_size"], len(dataloader)
            ))
        print(f"len(dataloader):{len(dataloader)}, num_data:{num_data}")

        for itr, data in enumerate(dataloader):
            bs = config["batch_size"]
            slice = range(itr*bs, itr*bs + min(bs, data["src_position"].shape[0]))
            label = th.tensor(slice).cuda()

            optimizer.zero_grad()
            for data_type in ['src_position', 'mic_position', 'atf_mag']:
                data[data_type] = data[data_type].cuda()
            #data['atf_mag'] = data['atf_mag'].cuda()
            prediction = net.forward(data=data, mode=mode)
            loss_train = lsd_loss(prediction['atf_mag'], data['atf_mag'], dim=-1, data_type='atf_mag', mean=True) # S, M, F -> scaler
            loss_train.backward()
            optimizer.step()

            #===== progress bar ======
            prog_step = 20
            if round(num_data/prog_step) != 0:
                if itr == 0:
                    print('[',end='')
                elif itr == num_data-1:
                    print('#]')
                elif itr % round(num_data/prog_step) == 0:
                    print('#',end='')
            #=========================

        t_end = time.time()
        
        loss_str = "    ".join([f"lsd: {loss_train:.4}"])
        time_str = f"({time.strftime('%H:%M:%S', time.gmtime(t_end-t_start))})"
        print(f"epoch {epoch} (train) ")
        print(loss_str + "        " + time_str)

        ## TensorBoard
        writer.add_scalar("LSD / train", loss_train, epoch)

        ## Validation
        #net.module.save(config["artifacts_dir"], suffix="tmp")
        print("---------------------")
        print(f"epoch {epoch} (valid)")
        t_start = time.time()
        
        net_valid = copy.deepcopy(net)
        loss_valid = testNF(config, net_valid, validdataset, BEST_LOSS, mode='valid', flg_save=flg_save, use_cuda=True)
        t_end = time.time()
        time_str = f"({time.strftime('%H:%M:%S', time.gmtime(t_end-t_start))})"
        print(time_str)
        ## TensorBoard
        writer.add_scalar("LSD / valid", loss_valid["loss"], epoch)

        if loss_valid["loss"] < BEST_LOSS:
            BEST_LOSS = loss_valid["loss"]
            LAST_SAVED = epoch
            print("Best Loss. Saving model!")
            net.module.save(config["artifacts_dir"], suffix="best")
        elif config["save_frequency"] > 0 and epoch % config["save_frequency"] == 0 and not epoch == config["epochs"]:
            print("Saving model!")
            net.module.save(config["artifacts_dir"], suffix="log_"+f'{epoch:03}'+"ep")
        else:
            print("Not saving model! Last saved: {}".format(LAST_SAVED))
        print("---------------------")

        #net.module.load_from_file(config["artifacts_dir"]+'/network.tmp.net')
    
    net.module.save(config["artifacts_dir"], suffix="final_"+f'{epoch:03}'+"ep")

    return

def testNF(config, net, data, best_loss, mode="test", flg_save=True, use_cuda=True):
    device = 'cuda' if use_cuda else 'cpu'
    # net.cpu()
    #net.to(device)
    #net.eval()
    #use_cuda_forward = True
    gpus = [i for i in range(config["num_gpus"])]
    net = th.nn.DataParallel(net, gpus)
    #weights = filter(lambda x: x.requires_grad, net.parameters())
    optimizer = th.optim.Adam(net.parameters(), lr=config['learning_rate'])
    lsd_loss = utils.LSD()

    #idx_mes_pos = config["idx_mes_pos_mat"][:config["num_mes_test"], 0]
    
    num_src = sum([len(config['src_index'][dataset_name][mode]) for dataset_name in config['dataset']])
    src_list = np.arange(0, num_src)

    returns = {}
    prediction_all = {}
    keys = []

    for data_type in config["data_type_interp"]:
        if hasattr(data[data_type], 'device'):
            data[data_type] = data[data_type].to(device)
        elif data_type == 'atf_mag':
            keys.append("lsd")

    for k in keys:
        returns.setdefault(k,0)
    
    #Fine tuning
    if True:
        print(f"Fine tuning ({mode})")
        start_ep = 0
        for epoch in range(start_ep+1, start_ep+config["epochs_test"]+1):
            t_start = time.time()
            
            for index in src_list:
                dataset_name = config["Table"][mode]["dataset"][index]
                src_id = config["Table"][mode]["src_index"][index] - config['src_index'][dataset_name][mode][0]
                data_slice = {}
                for data_type in data:
                    data_slice[data_type] = data[data_type][dataset_name][..., src_id:src_id+1].permute([-1] + th.arange(data[data_type][dataset_name].dim())[:-1].tolist()).to(device)
                data_slice['dataset'] = [dataset_name]
                data_slice['src_index'] = [src_id]

                prediction = net.forward(data=data_slice, mode="fine_tuning")
                loss_test = lsd_loss(prediction['atf_mag'], data_slice['atf_mag'][:,prediction['idx_mes_pos'],:], dim=-1, data_type='atf_mag', mean=True) # S, M, F -> scaler
                loss_test.backward()
                optimizer.step()

            t_end = time.time()

            loss_str = "    ".join([f"lsd: {loss_test:.4}"])
            time_str = f"({time.strftime('%H:%M:%S', time.gmtime(t_end-t_start))})"
            #print(f"epoch {epoch} ({mode}) ")
            #print(loss_str + "        " + time_str)
        print(loss_str + "        " + time_str)

    #Prediction
    for index in src_list:
        dataset_name = config["Table"][mode]["dataset"][index]
        src_id = config["Table"][mode]["src_index"][index] - config['src_index'][dataset_name][mode][0]
        data_slice = {}
        for data_type in data:
            data_slice[data_type] = data[data_type][dataset_name][..., src_id:src_id+1].permute([-1] + th.arange(data[data_type][dataset_name].dim())[:-1].tolist()).to(device)
        data_slice['dataset'] = [dataset_name]
        data_slice['src_index'] = [src_id]
        #data_slice['k'] = 2.0*th.pi*th.arange(1,config["num_freq"]+1)/config["num_freq"]*config['fs']/2/config['c']
        #data_slice['frequency'] = (th.arange(config['num_freq']+1)[1:]/config['num_freq']).unsqueeze(-1)

        prediction = net.forward(data=data_slice, mode=mode)
        for k in prediction:
            if hasattr(prediction[k], 'detach'):
                prediction[k] = prediction[k].detach().clone()
 
        additional_data_type = []#['lsd_bm']

        for data_type in config["data_type_interp"] + additional_data_type:
            if data_type == 'atf_mag':
                returns["lsd"] += lsd_loss(prediction['atf_mag'], data_slice['atf_mag'], dim=-1, data_type='atf_mag', mean=True) # S,M,F
            elif data_type == 'lsd_bm':
                prediction['lsd_bm'] = lsd_loss(prediction['atf_mag'], data_slice['atf_mag'], dim=-1, data_type='atf_mag', mean=False) # S,M
            
            if not data_type in prediction_all:
                prediction_all[data_type] = {}
            if not dataset_name in prediction_all[data_type]:
                if data_type in data:
                    prediction_all[data_type][dataset_name] = th.zeros_like(data[data_type][dataset_name])
                else:
                    if data_type == 'lsd_bm':
                        prediction_all[data_type][dataset_name] = th.zeros_like(data['atf_mag'][dataset_name][:,:,0,:])
            prediction_all[data_type][dataset_name][..., src_id:src_id+1] = prediction[data_type].permute(th.arange(prediction[data_type].dim())[1:].tolist() + [0])
        
        if flg_save or src_id == config['src_index'][dataset_name][mode][0] - config['src_index'][dataset_name][mode][0]:
            if 'atf_mag' in config["data_type_interp"]:
                #=== plot atf_mag ====
                utils.plotatfmag(micpos=data_slice['mic_position'][0,:,:].cpu(), sig_gt=data_slice['atf_mag'][0,:,:], sig_pred=prediction['atf_mag'][0,:,:], config=config, idx_plot_list=np.linspace(0, data_slice['atf_mag'].shape[1]-1, 5, dtype=int),  figdir=f"{config['artifacts_dir']}/figure/atf/", fname=f"{config['artifacts_dir']}/figure/atf/ATF_Mag_{data_slice['dataset'][0]}-src-{data_slice['src_index'][0]+1}_{mode}", data_type = 'atf_mag')
                
        if flg_save and src_id == config['src_index'][dataset_name][mode][-1] - config['src_index'][dataset_name][mode][0]:
            for data_type in prediction_all:
                pt_dir = f'{config["artifacts_dir"]}/{data_type}'
                os.makedirs(pt_dir, exist_ok=True)
                for dataset_name in prediction_all[data_type]:
                    th.save(prediction_all[data_type][dataset_name], f'{pt_dir}/{data_type}_{mode}_{dataset_name}.pt')

    loss = 0
    for k in keys:
        if k in config["loss_weights"] and config["loss_weights"][k] > 0:
            loss += config["loss_weights"][k] * returns[k]
    returns["loss"] = loss.detach().clone()
    for k in returns:
        returns[k] /= len(src_list)

    loss_str = "    ".join([f"{k}:{returns[k]:.4}" for k in sorted(returns.keys())])
    print(loss_str)
    
    if returns["loss"] < best_loss:
        print("Best Loss.")

    return returns

def testNF_old(config, net, data, mode="test", flg_save=True, use_cuda=True):
    device = 'cuda' if use_cuda else 'cpu'
    gpus = [i for i in range(config["num_gpus"])]
    net = th.nn.DataParallel(net, gpus)
    optimizer = th.optim.Adam(net.parameters(), lr = config["learning_rate"])

    idx_mes_pos = config["idx_mes_pos_mat"][:config["num_mes_test"], 0]

    lsd_loss = utils.LSD()
    #mse_loss = utils.MSE()
    #helmholtz_loss = utils.HelmholtzLoss()

    lsd_weight = config["loss_weights"]["lsd"]
    #mse_weight = config["loss_weights"]["mse"]
    #he_weight = config["loss_weights"]["helmholtz"]

    num_src = sum([len(config['src_index'][dataset_name][mode]) for dataset_name in config['dataset']])
    src_list = np.arange(0, num_src)
    #freq = np.arange(1,config["num_freq"]+1)/config["num_freq"]*config['fs']/2

    returns = {}
    prediction_all = {}
    keys = []

    for k in keys:
        returns.setdefault(k,0)

    for index in src_list:
        dataset_name = config["Table"][mode]["dataset"][index]
        src_id = config["Table"][mode]["src_index"][index] - config['src_index'][dataset_name][mode][0]
        data_slice = {}
        for data_type in data:
            data_slice[data_type] = data[data_type][dataset_name][..., src_id:src_id+1].permute([-1] + th.arange(data[data_type][dataset_name].dim())[:-1].tolist()).to(device)
        data_slice['dataset'] = [dataset_name]
        data_slice['src_index'] = [src_id]
        data_slice['k'] = 2.0*th.pi*th.arange(1,config["num_freq"]+1)/config["num_freq"]*config['fs']/2/config['c']
        data_slice['frequency'] = (th.arange(config['num_freq']+1)[1:]/config['num_freq']).unsqueeze(-1)

        prediction = {}
        for data_type in config["data_type_interp"]:
            if not data_type in prediction:
                if data_type in ["atf"]:
                    prediction[data_type] = th.zeros(data[data_type][dataset_name].shape[0],data[data_type][dataset_name].shape[1], dtype=th.complex64).to(device)
                else:
                    prediction[data_type] = th.zeros(data[data_type][dataset_name].shape[0],data[data_type][dataset_name].shape[1], dtype=th.float32).to(device)
            #if not data_type in prediction_all:
            #    prediction_all[data_type] = {}
            #if not dataset_name in prediction_all[data_type]:
            #    if data_type in data:
            #        prediction_all[data_type][dataset_name] = th.zeros_like(data[data_type][dataset_name])

        data_slice_mes = {}
        for data_type in data_slice:
            if data_type in ["atf", "atf_mag", "src_position", "mic_position"]:
                data_slice_mes[data_type] = data_slice[data_type][:,idx_mes_pos,:]
            else:
                data_slice_mes[data_type] = data_slice[data_type]

        pos_src = th.squeeze(data_slice_mes['src_position'])
        pos_mic_mes = th.squeeze(data_slice_mes['mic_position'])
        #pos_mic_eval = th.squeeze(data_slice_mes['mic_position']) #th.squeeze(data_slice['mic_position'])

        for layer in net.modules():
            if hasattr(layer, 'weight'):
                nn.init.normal_(layer.weight, 0.0, 1e-3)
            if hasattr(layer, 'bias'):
                nn.init.normal_(layer.bias, 0.0, 1e-3)

        freq = data_slice['frequency']
        k = data_slice['k']
        #p = th.squeeze(data_slice_mes["atf"])
        p_mag = th.squeeze(data_slice_mes["atf_mag"])

        if True:
            for epoch in range(config["epochs"]):
                net.train()
                optimizer.zero_grad()

                prediction_mes = net.forward(pos_src=pos_src, pos_mic=pos_mic_mes, freq=freq)
                lsd_loss_val = lsd_loss(prediction_mes['output_mag'], p_mag, dim=0, mean=True)
                loss = lsd_loss_val

                #mse_loss_val = mse_loss(prediction_mes['output_c'], p, dim=0, mean=True)
                #loss = lsd_weight*lsd_loss_val + mse_weight*mse_loss_val

                #he_loss_val = helmholtz_loss(net=net, pos=pos_mic_eval, k=k, freq=freq)
                #loss = lsd_weight*lsd_loss_val + mse_weight*mse_loss_val + he_weight*he_loss_val
                
                #if epoch == 0 or epoch == config['epochs']-1:
                #    print(f'Epoch [{epoch+1}/{config["epochs"]}], Loss: {loss.item():.8f}, LSD Loss: {lsd_loss_val.item():.8f}, MSE Loss: {mse_loss_val.item():.8f}, Helmholtz Loss: {he_loss_val.item():.8f}')
                
                loss.backward()
                optimizer.step()

            #prediction['atf'] = (net.forward(pos=th.squeeze(data_slice['mic_position']), freq=data_slice['frequency'])['output_c']).detach().clone()
            prediction['atf_mag'] = (net.forward(pos=th.squeeze(data_slice['mic_position']), freq=data_slice['frequency'])['output_mag']).detach().clone()

        if False:
            for ff in range(config['num_freq']):
                for layer in net.modules():
                    if hasattr(layer, 'weight'):
                        nn.init.normal_(layer.weight, 0.0, 1e-3)
                    if hasattr(layer, 'bias'):
                        nn.init.normal_(layer.bias, 0.0, 1e-3)

                #freq = data_slice['frequency'][ff]
                k = data_slice['k'][ff]
                p = th.squeeze(data_slice_mes["atf"][:,:,ff])
                p_mag = th.squeeze(data_slice_mes["atf_mag"][:,:,ff])
                
                for epoch in range(config["epochs"]):
                    net.train()
                    optimizer.zero_grad()

                    prediction_mes = net.forward(pos=pos_mic_mes)
                    lsd_loss_val = lsd_loss(prediction_mes['output_mag'], p_mag, dim=0, mean=False)
                    mse_loss_val = mse_loss(prediction_mes['output_c'], p, dim=0, mean=False)

                    he_loss_val = helmholtz_loss(net=net, pos=pos_mic_eval, k=k)
                    loss = lsd_weight*lsd_loss_val + mse_weight*mse_loss_val + he_weight*he_loss_val
                    
                    #if epoch == 0 or epoch == config['epochs']-1:
                    #    print(f'Freq [{ff+1}/{config["num_freq"]}], Epoch [{epoch+1}/{config["epochs"]}], Loss: {loss.item():.8f}, LSD Loss: {lsd_loss_val.item():.8f}, MSE Loss: {mse_loss_val.item():.8f}, Helmholtz Loss: {he_loss_val.item():.8f}')
                    
                    loss.backward()
                    optimizer.step()

            prediction['atf'][:,ff] = (net.forward(pos=th.squeeze(data_slice['mic_position']))['output_c']).detach().clone()
            prediction['atf_mag'][:,ff] = (net.forward(pos=th.squeeze(data_slice['mic_position']))['output_mag']).detach().clone()

        lsd_loss_mean = lsd_loss(prediction['atf_mag'], th.squeeze(data_slice['atf_mag']), dim=1, data_type='atf_mag', mean=True) # M,F
        print(f'Src: {index+1}/{num_src}, LSD: {lsd_loss_mean.item():.8f}')

        for data_type in config["data_type_interp"]:
            if not data_type in prediction_all:
                prediction_all[data_type] = {}
            if not dataset_name in prediction_all[data_type]:
                if data_type in data:
                    prediction_all[data_type][dataset_name] = th.zeros_like(data[data_type][dataset_name])
            prediction_all[data_type][dataset_name][..., src_id:src_id+1] = prediction[data_type][:,:,None]

        if flg_save or src_id == config['src_index'][dataset_name][mode][0] - config['src_index'][dataset_name][mode][0]:
            if 'atf_mag' in config["data_type_interp"]:
                #=== plot atf_mag ====
                utils.plotatfmag(micpos=data_slice['mic_position'][0,:,:].cpu(), sig_gt=data_slice['atf_mag'][0,:,:], sig_pred=prediction['atf_mag'], config=config, idx_plot_list=np.linspace(0, data_slice['atf_mag'].shape[1]-1, 5, dtype=int),  figdir=f"{config['artifacts_dir']}/figure/atf/", fname=f"{config['artifacts_dir']}/figure/atf/ATF_Mag_{data_slice['dataset'][0]}-src-{data_slice['src_index'][0]+1}_{mode}", data_type = 'atf_mag')
                
        if flg_save and src_id == config['src_index'][dataset_name][mode][-1] - config['src_index'][dataset_name][mode][0]:
            for data_type in prediction_all:
                pt_dir = f'{config["artifacts_dir"]}/{data_type}'
                os.makedirs(pt_dir, exist_ok=True)
                for dataset_name in prediction_all[data_type]:
                    th.save(prediction_all[data_type][dataset_name], f'{pt_dir}/{data_type}_{mode}_{dataset_name}.pt')
    
    return returns


def testKRR(config, data, mode="test", flg_save=True, use_cuda=True):
    device = 'cuda' if use_cuda else 'cpu'
    gpus = [i for i in range(config["num_gpus"])]

    idx_mes_pos = config["idx_mes_pos_mat"][:config["num_mes_test"], 0]

    lsd_loss = utils.LSD()
    lsd_mean = 0
    per_source_lsd = []

    num_src = sum([len(config['src_index'][dataset_name][mode]) for dataset_name in config['dataset']])
    src_list = np.arange(0, num_src)
    num_freq = config["num_freq"]
    #freq = np.arange(1,config["num_freq"]+1)/config["num_freq"]*config['fs']/2

    returns = {}
    prediction_all = {}
    keys = []

    for k in keys:
        returns.setdefault(k,0)

    for index in src_list:
        dataset_name = config["Table"][mode]["dataset"][index]
        src_id = config["Table"][mode]["src_index"][index] - config['src_index'][dataset_name][mode][0]
        data_slice = {}
        for data_type in data:
            data_slice[data_type] = data[data_type][dataset_name][..., src_id:src_id+1].permute([-1] + th.arange(data[data_type][dataset_name].dim())[:-1].tolist()).to(device)
        data_slice['dataset'] = [dataset_name]
        data_slice['src_index'] = [src_id]
        data_slice['k'] = 2.0*th.pi*th.arange(1,config["num_freq"]+1)/config["num_freq"]*config['fs']/2/config['c']
        data_slice['frequency'] = (th.arange(config['num_freq']+1)[1:]/config['num_freq']).unsqueeze(-1)

        prediction = {}
        for data_type in config["data_type_interp"]:
            if not data_type in prediction:
                if data_type in ["atf"]:
                    prediction[data_type] = th.zeros(data[data_type][dataset_name].shape[0],data[data_type][dataset_name].shape[1], dtype=th.complex64).to(device)
                else:
                    prediction[data_type] = th.zeros(data[data_type][dataset_name].shape[0],data[data_type][dataset_name].shape[1], dtype=th.float32).to(device)

        data_slice_mes = {}
        for data_type in data_slice:
            if data_type in ["atf", "atf_mag", "src_position", "mic_position"]:
                data_slice_mes[data_type] = data_slice[data_type][:,idx_mes_pos,:]
            else:
                data_slice_mes[data_type] = data_slice[data_type]

        # Kernel interpolation
        pos_mic_mes = th.squeeze(data_slice_mes['mic_position']).cpu().numpy().copy()
        pos_mic_est = th.squeeze(data_slice['mic_position']).cpu().numpy().copy()
        num_mic = pos_mic_mes.shape[0]

        k = data_slice['k'][:, None, None].cpu().numpy().copy()
        distMat = distfuncs.cdist(pos_mic_mes, pos_mic_mes)[None, :, :]
        #K = special.spherical_jn(0, k * distMat)
        var_gauss = 1e-1 * np.ones((num_freq, 1, 1)) #(0.2 * k) ** 2 #
        K = np.exp(-var_gauss * (distMat**2))
        reg = 1e-3
        Kinv = np.linalg.inv(K + reg * np.eye(num_mic))
        distVec = np.transpose(distfuncs.cdist(pos_mic_est, pos_mic_mes), (1, 0))[None, :, :]
        #kappa = special.spherical_jn(0, k * distVec)
        kappa = np.exp(-var_gauss * (distVec**2) )
        kiFilter = th.from_numpy( (np.transpose(kappa, (0, 2, 1)) @ Kinv).astype(np.float32) ).to(device)

        prediction["atf_mag"] = th.squeeze(th.permute(th.bmm( kiFilter, th.permute(data_slice_mes["atf_mag"], (2, 1, 0))), (2, 1, 0)))

        lsd_loss_mean = lsd_loss(prediction['atf_mag'], th.squeeze(data_slice['atf_mag']), dim=1, data_type='atf_mag', mean=True) # M,F
        print(f'Src: {index+1}/{num_src}, LSD: {lsd_loss_mean.item():.8f}')
        lsd_mean += lsd_loss_mean.item()
        per_source_lsd.append(lsd_loss_mean.item())

        for data_type in config["data_type_interp"]:
            if not data_type in prediction_all:
                prediction_all[data_type] = {}
            if not dataset_name in prediction_all[data_type]:
                if data_type in data:
                    prediction_all[data_type][dataset_name] = th.zeros_like(data[data_type][dataset_name])
            prediction_all[data_type][dataset_name][..., src_id:src_id+1] = prediction[data_type][:,:,None]

        if flg_save or src_id == config['src_index'][dataset_name][mode][0] - config['src_index'][dataset_name][mode][0]:
            if 'atf_mag' in config["data_type_interp"]:
                #=== plot atf_mag ====
                utils.plotatfmag(micpos=data_slice['mic_position'][0,:,:].cpu(), sig_gt=data_slice['atf_mag'][0,:,:], sig_pred=prediction['atf_mag'], config=config, idx_plot_list=np.linspace(0, data_slice['atf_mag'].shape[1]-1, 5, dtype=int),  figdir=f"{config['artifacts_dir']}/figure/atf/", fname=f"{config['artifacts_dir']}/figure/atf/ATF_Mag_{data_slice['dataset'][0]}-src-{data_slice['src_index'][0]+1}_{mode}", data_type = 'atf_mag')
                
        if flg_save and src_id == config['src_index'][dataset_name][mode][-1] - config['src_index'][dataset_name][mode][0]:
            for data_type in prediction_all:
                pt_dir = f'{config["artifacts_dir"]}/{data_type}'
                os.makedirs(pt_dir, exist_ok=True)
                for dataset_name in prediction_all[data_type]:
                    th.save(prediction_all[data_type][dataset_name], f'{pt_dir}/{data_type}_{mode}_{dataset_name}.pt')

    lsd_mean /= num_src
    lsd_std = float(np.std(np.array(per_source_lsd, dtype=np.float64))) if per_source_lsd else float('nan')
    print(f"LSD mean: {lsd_mean}")
    print(f"LSD std: {lsd_std}")
    print(f"LSD mean ± std: {lsd_mean:.4f} ± {lsd_std:.4f}")

    # Save a compact metrics log so KRR runs have numeric outputs in artifacts_dir.
    metrics_path = os.path.join(config["artifacts_dir"], f"krr_metrics_{mode}.txt")
    with open(metrics_path, "w") as f:
        f.write(f"mode: {mode}\n")
        f.write(f"num_sources: {num_src}\n")
        f.write(f"num_freq: {num_freq}\n")
        f.write(f"freq_from: {config.get('freq_from', 0)}\n")
        f.write(f"freq_up_to: {config.get('freq_up_to', config.get('num_freq'))}\n")
        f.write(f"num_mes_test: {config.get('num_mes_test')}\n")
        f.write(f"lsd_mean: {lsd_mean:.10f}\n")
        f.write(f"lsd_std: {lsd_std:.10f}\n")
        f.write(f"lsd_mean_pm_std: {lsd_mean:.6f} +- {lsd_std:.6f}\n")
        f.write("per_source_lsd:\n")
        for idx, val in enumerate(per_source_lsd, start=1):
            f.write(f"{idx},{val:.10f}\n")
    print(f"KRR metrics saved: {metrics_path}")

    return returns


if __name__ == "__main__":
    # Set random seed
    seed = 0
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    # Datestamp
    dt_now = datetime.datetime.now() + datetime.timedelta(hours=9)
    datestamp = dt_now.strftime("%Y%m%d")
    timestamp = dt_now.strftime('_%m%d_%H%M')

    args = arg_parser()
    config = {
        "timestamp": timestamp,
    }
    config_name = f"config_{args.model}_{args.idx_config}"
    config.update(eval(config_name))

    # CLI args override config values; config values override defaults
    config['freq_from'] = args.freq_from if args.freq_from != 0 else config.get('freq_from', 0)
    config['freq_up_to'] = args.freq_up_to if args.freq_up_to is not None else config.get('freq_up_to', 64)
    config['num_freq'] = config['freq_up_to'] - config['freq_from']
    if args.num_mes_test is not None:
        config['num_mes_test'] = args.num_mes_test
    if getattr(args, "data_dir", "") != "":
        config["data_dir"] = args.data_dir
    if args.dataset_version is not None:
        config["dataset_version"] = args.dataset_version
    # Data loader backend selection (dataset.py supports use_processed_pt already)
    if args.use_npz and args.use_processed_pt:
        raise ValueError("Use only one of --use_npz or --use_processed_pt.")
    if args.use_npz:
        config["use_processed_pt"] = False
    elif args.use_processed_pt:
        config["use_processed_pt"] = True
    else:
        # Default to existing behavior if not explicitly overridden.
        config["use_processed_pt"] = config.get("use_processed_pt", True)

    if config["model"] in ["FSMPAE", "EEAE"]:
        net = models.ATFApproxNetwork(config)
    elif config["model"] == "FSMPAEPI":
        net = models.ATFApproxNetworkPI(config)
    elif config["model"] == "NF":
        net = models.NF(config)
    
    print("=====================")
    print("config: ")
    pprint.pprint(config)
    print("=====================")

    if config["model"] in ["FSMPAE", "FSMPAEPI", "EEAE"]:
        if not config["activation_function"].startswith('nn.ReLU'):
            utils.replace_activation_function(net, act_func_new=eval(config["activation_function"]))
        print(net)
    elif config["model"] in ["NF"]:
        print(net)
    print("=====================")

    if args.artifacts_directory == "":
        model_with_cfg = f"{config['model']}{args.idx_config}"
        config["artifacts_dir"] = "outputs/out" + datestamp + "_" + model_with_cfg
    else:
        config["artifacts_dir"] = args.artifacts_directory
    print("artifacts_dir: " + config["artifacts_dir"])
    os.makedirs(config["artifacts_dir"], exist_ok=True)

    config["metrics_only"] = bool(getattr(args, "metrics_only", False))

    if args.test:
        flg_train = False
        flg_save = not config["metrics_only"]
        print("Test")
        if config["metrics_only"]:
            print("Metrics-only mode: plotting and artifact tensor saves are disabled.")
    else:
        flg_train = True
        flg_save = False
        print("Train")
    print("=====================")

    print("Dataset setup:")
    print(f"  datasets: {config.get('dataset')}")
    print(f"  data_dir: {config.get('data_dir', '(auto)')}")
    print(f"  freq range: [{config.get('freq_from')}, {config.get('freq_up_to')})")
    print(f"  data source backend: {'processed .pt' if config.get('use_processed_pt', True) else 'per-source .npz'}")
    if "dataset_version" in config:
        print(f"  dataset_version override: {config['dataset_version']}")
    else:
        print("  dataset_version override: (none; infer from dataset name)")
    print("=====================")

    idataset = dataset.ATFdataset(config)

    traindataset_all = idataset.trainitem()
    validdataset = idataset.validitem()
    testdataset = idataset.testitem()

    config["Table"] = idataset.Table
    config["data_stat"] = idataset.DataStat

    if config["model"] in ["FSMPAE", "FSMPAEPI", "EEAE"]:
        traindataset = idataset
        ptins = ""
        if flg_train:
            print("Training mode.")
            print(f"Number of trainable parameters: {net.num_trainable_parameters()}")

            print("---------------------")
            start_ep = 0
            itrainer = trainer.Trainer(config, net, traindataset)
            print("---------------------")

            ## TensorBoard
            log_dir=config["artifacts_dir"]+"/logs/"+config["model"]+timestamp
            writer = SummaryWriter(log_dir)
            print("logdir: "+log_dir) 

            train(trainer=itrainer, ptins=ptins)

        print("Test mode.")
        print("=====================")
        net.load_from_file(config["artifacts_dir"]+'/network.best.net')
        print(net)
        print("=====================")
        loss = test(net, flg_save, -1, testdataset, f'test{config["timestamp"]}')
    
    elif config["model"] in ["NF"]:        
        traindataset = idataset
        ptins = ""
        if flg_train:
            print("Training mode.")
            print(f"Number of trainable parameters: {net.num_trainable_parameters()}")

            ## TensorBoard
            log_dir=config["artifacts_dir"]+"/logs/"+config["model"]+timestamp
            writer = SummaryWriter(log_dir)
            print("logdir: "+log_dir) 

            trainNF(config, net, traindataset)

        print("Test mode.")
        print("=====================")
        net.load_from_file(config["artifacts_dir"]+'/network.best.net')
        print(net)
        print("=====================")
        testNF(config, net, testdataset, -1, mode="test", flg_save=flg_save)

    elif config["model"] in ["KRR"]:
        testKRR(config, testdataset)


    print("Done.")