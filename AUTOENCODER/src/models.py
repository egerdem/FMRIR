import numpy as np
import math
import torch as th
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
import AUTOENCODER.src.utils as utils

from icecream import ic
ic.configureOutput(includeContext=True)

class FourierFeatureMapping(nn.Module):
    def __init__(self, num_ff, dim_data, trainable=True):
        super().__init__()
        self.num_ff = num_ff
        self.dim_data = dim_data
        multinormdist = MultivariateNormal(th.zeros(self.dim_data), th.eye(self.dim_data))
        self.v = multinormdist.sample(sample_shape=th.Size([self.num_ff]))
        if trainable:
            self.v = nn.Parameter(self.v)
        else:
            self.v = self.v.cuda()
    
    def forward(self, x):
        x_shape = list(x.shape)
        x_shape[-1] = self.num_ff
        #print(x_shape)
        x = x.reshape(-1, self.dim_data).permute(1, 0)
        #print(x.shape)
        #print(self.v.shape)
        vx = th.matmul(self.v.to(x.device), x)
        #print(vx.shape)
        vx = vx.permute(1, 0).reshape(x_shape)
        ff = th.cat([th.sin(2*np.pi*vx), th.cos(2*np.pi*vx)], dim=-1)
        return ff
    
class ResLinearBlock(nn.Module):
    def __init__(self, channel=64, droprate=0.2, norm=None, bn_c = None):
        super().__init__()
        if norm == 'ln':
            self.layers1 = nn.Sequential(
                nn.Linear(channel,channel),
                nn.LayerNorm(channel),
                nn.ReLU(), 
                nn.Dropout(droprate),
                nn.Linear(channel,channel)
            )
            self.layers2 = nn.Sequential(
                nn.LayerNorm(channel),
                nn.ReLU(), 
                nn.Dropout(droprate)
            )
        elif norm == 'bn':
            self.layers1 = nn.Sequential(
                nn.Linear(channel,channel),
                nn.BatchNorm1d(bn_c),
                nn.ReLU(), 
                nn.Dropout(droprate),
                nn.Linear(channel,channel)
            )
            self.layers2 = nn.Sequential(
                nn.BatchNorm1d(bn_c),
                nn.ReLU(), 
                nn.Dropout(droprate)
            )
        elif norm == None:
            self.layers1 = nn.Sequential(
                nn.Linear(channel,channel),
                nn.ReLU(), 
                nn.Dropout(droprate),
                nn.Linear(channel,channel)
            )
            self.layers2 = nn.Sequential(
                nn.ReLU(), 
                nn.Dropout(droprate)
            )
    def forward(self, input):
        out_mid = self.layers1(input)
        in_mid = out_mid + input # skip connection
        out = self.layers2(in_mid)
        return out
    
class HyperLinear(nn.Module):
    def __init__(self, ch_in, ch_out, ch_hidden=32, num_hidden=1, droprate=0.0, use_res=True, input_size=3):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out

        #==== weight ======
        modules = [
            nn.Linear(input_size, ch_hidden),
            nn.LayerNorm(ch_hidden),
            nn.ReLU(),
            nn.Dropout(droprate)
        ]
        if not use_res:
            for l in range(num_hidden):
                modules.extend([
                    nn.Linear(ch_hidden, ch_hidden),
                    nn.LayerNorm(ch_hidden),
                    nn.ReLU(),
                    nn.Dropout(droprate)
                ])
        else:
            for l in range(round(num_hidden/2)):
                modules.extend([ResLinearBlock(ch_hidden, droprate=0.0)])
        modules.extend([
             nn.Linear(ch_hidden, ch_out * ch_in)
        ])
        self.weight_layers = nn.Sequential(*modules)

        #==== bias ======
        modules = [
            nn.Linear(input_size, ch_hidden),
            nn.LayerNorm(ch_hidden),
            nn.ReLU(),
            nn.Dropout(droprate)]
        if not use_res:
            for l in range(num_hidden):
                modules.extend([
                    nn.Linear(ch_hidden, ch_hidden),
                    nn.LayerNorm(ch_hidden),
                    nn.ReLU(),
                    nn.Dropout(droprate)
                ])
        else:
            for l in range(round(num_hidden/2)):
                modules.extend([ResLinearBlock(ch_hidden,droprate=0.0)])
        modules.extend([
             nn.Linear(ch_hidden, ch_out)
        ])
        self.bias_layers = nn.Sequential(*modules)

    def forward(self, input):
        x = input["x"] # (..., ch_in)
        z = input["z"] # (..., input_size)
        batches = list(x.shape)[:-1] # (...,)
        num_batches = math.prod(batches)

        weight = self.weight_layers(z) # (..., ch_out * ch_in)
        #ic(weight.shape)
        weight = weight.reshape([num_batches, self.ch_out, self.ch_in]) # (num_batches, ch_out, ch_in)
        #ic(weight.shape)
        bias = self.bias_layers(z) # (..., ch_out)
        #ic(bias.shape)

        output = {}
        #ic(x.reshape(num_batches,-1,1).shape)
        wx = th.matmul(weight, x.reshape(num_batches,-1,1)).reshape(batches + [-1])
        #ic(wx.shape)
        output["x"] = wx + bias #F.linear(x, weight, bias)
        output["z"] = z
        
        return output

class HyperLinearBlock(nn.Module):
    def __init__(self, ch_in, ch_out, ch_hidden=32, num_hidden=1, droprate=0.0, use_res=True, input_size=3, post_proc=True):
        super().__init__()
        self.hyperlinear = HyperLinear(ch_in, ch_out, ch_hidden=ch_hidden, num_hidden=num_hidden, use_res=use_res, input_size=input_size, droprate=droprate)
        if post_proc:
            self.layers_post = nn.Sequential(
                nn.LayerNorm(ch_out),
                nn.ReLU(),
                nn.Dropout(droprate)
            )
        else:
            self.layers_post = nn.Sequential(
                nn.Identity()
            )

    def forward(self, input):
        #x = input["x"] # (M, ch_in)
        z = input["z"] # (M, input_size=3)
        #ic(z.shape)
        xz = self.hyperlinear(input) 
        #ic(xz["x"].shape)
        #ic(xz["z"].shape)
        
        output = {}
        output["x"] = self.layers_post(xz["x"])
        output["z"] = z
        
        return output

class ATFApproxNetwork(utils.Net):
    def __init__(self,
                 config,
                 use_cuda=True):
        super().__init__()
        self.config = config
        self.num_freq = config["num_freq"]
        
        # Fourier feature mapping
        if len(self.config["data_type_ffm"]) > 0:
            # self.ffm = {}
            for data_type in self.config["data_type_ffm"]:
                # self.ffm[data_type] = FourierFeatureMapping(num_ff = self.config["num_ff"][data_type], dim_data = self.config["dim_data_hyper"][data_type], trainable = self.config["ffm_trainable"])
                 exec(f'self.ffm_{data_type} = FourierFeatureMapping(num_ff=self.config["num_ff"][data_type], dim_data=self.config["dim_data_hyper"][data_type], trainable=self.config["ffm_trainable"])')
        
        # Encoder
        if self.config["hlayers_En_0"] > 0:
            modules = []

            for l in range(self.config["hlayers_En_0"]):
                if l == 0: # input layer
                    ch_in = 1
                else:
                    ch_in = self.config["channel_En_0"]
                
                if l == self.config["hlayers_En_0"] - 1: # output layer
                    ch_out = 1 if not config["aggr_mean"] else config["dim_z"]
                    post_proc = False
                else:
                    ch_out = self.config["channel_En_0"]
                    post_proc = True
                
                if self.config["hyperlinear_en_0"]:
                    input_size_en_0 = 0
                    for data_type in self.config["data_type_hyper_en"]:
                        if data_type in self.config["data_type_ffm"]:
                            input_size_en_0 += self.config["num_ff"][data_type] * 2
                        else:
                            input_size_en_0 += self.config["dim_data_hyper"][data_type]
                    #if len(self.config['data_type_interp']) > 1:
                    #    input_size_en_0 += 1 # delta
                    modules.extend([
                        HyperLinearBlock(ch_in=ch_in, ch_out=ch_out, ch_hidden=self.config["channel_hyper"], num_hidden=self.config["hlayers_hyper"], droprate=self.config["droprate"], use_res=self.config["hyper_use_res"], input_size=input_size_en_0, post_proc=post_proc)
                        ])
                else:
                    raise NotImplementedError
                
            self.en_0 = nn.Sequential(*modules)
                
        # Decoder
        if config["hlayers_De_z"] > 0:
            modules = []
            for l in range(config["hlayers_De_z"]):
                if l == 0: # last layer
                    ch_in = config["dim_z"]
                else:
                    ch_in = config["channel_De_z"]
                if self.config["use_freq_for_hyper"]:
                    # x:(S,L,ch_out), z:(S,L,1)
                    modules.extend([
                        HyperLinearBlock(ch_in=ch_in, ch_out=config["channel_De_z"], ch_hidden=config["channel_hyper"], num_hidden=config["hlayers_hyper"], droprate=config["droprate"], use_res=config["hyper_use_res"], input_size=1),
                    ])
                else:
                    # input:(S,L,ch_out)
                    modules.extend([
                        nn.Linear(in_features=ch_in, out_features=config["channel_De_z"], bias=True),
                        nn.LayerNorm(config["channel_De_z"]),
                        nn.ReLU(),
                        nn.Dropout(config["droprate"])
                    ])
                # x or output:(S,L,ch_out)
            self.de_1 = nn.Sequential(*modules)

        if config["hlayers_De_z"] == 0:
            ch_in = config["dim_z"]
        else:
            ch_in = config["channel_De_z"]

        if self.config["hlayers_De_-1"]>0:
            # x:(S,L,ch_out)->(S,L,B,ch_out)->(S,L,B,1)->(S,L,B)
            modules = []
            for l in range(config["hlayers_De_-1"]):
                if l == 0: # first layer
                    if not self.config["de_2_skip"]:
                        ch_in = 1
                    else:
                        ch_in = config["dim_z"]
                else:
                    ch_in = config["channel_De_-1"]

                if l == config["hlayers_De_-1"] - 1: # last layer
                    ch_out = 1
                    post_proc = False
                else:
                    ch_out = config["channel_De_-1"]
                    post_proc = True

                if self.config["hyperlinear_de_-1"]:
                    input_size_de_m1 = 0
                    for data_type in self.config["data_type_hyper_de"]:
                        if data_type in self.config["data_type_ffm"]:
                            input_size_de_m1 += self.config["num_ff"][data_type] * 2
                        else:
                            input_size_de_m1 += self.config["dim_data_hyper"][data_type]
                    #if len(self.config["data_type_interp"]) > 1:
                    #    input_size_de_m1 += 1 # delta

                    modules.extend([
                        HyperLinearBlock(ch_in=ch_in, ch_out=ch_out, ch_hidden=config["channel_hyper"], num_hidden=config["hlayers_hyper"], droprate=config["droprate"], use_res=config["hyper_use_res"], input_size=input_size_de_m1, post_proc=post_proc),
                    ])

            self.de_m1 = nn.Sequential(*modules)
            # x:(S,L,ch_out)->(S,L,B,ch_out)->(S,L,B,1)->(S,L,B)

    def forward(self, data, mode='train'):
        '''
        :param: data: dict contains
            'src_position':  (S,B,3) torch.float32 [x,y,z]
            'mic_position':  (S,B,3) torch.float32 [x,y,z]
            'rir':     (S,B,2,L) torch.float32
            'atf_mag':     (S,B,2,L) torch.float32
            'dataset':      list of str
            'src_index':      (S) torch.int64

        :return: out: 
            - HRTF produce as a (B, 4L, S) complex tensor
                4L ... [l_re, l_im, r_re, r_im]
            - HRTF's magnitude as a (B,2,L,S) float tensor
            - HRIR as a (B,2,L',S) float tensor
            - ITDs as a (B, S) float tensor
        '''
        returns = {}
        dataset_name = data["dataset"][0]

        F = self.num_freq
        S, M, _ = data["src_position"].shape

        if mode == 'train':
            perm = th.randperm(M)
            idx_mes_pos = perm[:self.config["num_mes"]]
        else:
            #idx_mes_pos = th.arange(M)
            #perm = th.randperm(M)
            #idx_mes_pos = perm[:self.config["num_mes_test"]]
            idx_mes_pos = self.config["idx_mes_pos_mat"][:self.config["num_mes_test"], 0]
        #perm = th.randperm(M)
        #idx_mes_pos = perm[:self.config["num_mes"]]
        returns["idx_mes_pos"] = idx_mes_pos
        
        M_mes = len(idx_mes_pos)
        data_mes = {}

        # Standardize
        #for data_type in ['atf_mag', 'src_position', 'mic_position']:#self.config["data_type_interp"]:
        #    data[data_type] = (data[data_type] - self.config["data_stat"][data_type][dataset_name]["mean"]) / self.config["data_stat"][data_type][dataset_name]["std"]
        data['atf_mag'] = (data['atf_mag'] - self.config["data_stat"]['atf_mag'][dataset_name]["mean"]) / self.config["data_stat"]['atf_mag'][dataset_name]["std"]
        data['src_position'] = data['src_position']/self.config["data_stat"]['src_position'][dataset_name]["max"]
        data['mic_position'] = data['mic_position']/self.config["data_stat"]['mic_position'][dataset_name]["max"]
        data['frequency'] = (th.arange(self.num_freq+1)[1:]/self.num_freq).to(data['atf_mag'].device).unsqueeze(-1)

        for k in data:
            if k in ['atf_mag', 'src_position', 'mic_position']:
                data_mes[k] = data[k][:, idx_mes_pos]

        M_mes_norm = self.config["M_mes_norm"] if "M_mes_norm" in self.config.keys() else M_mes
        data['M_mes'] = th.tensor([M_mes/M_mes_norm]).to(data['atf_mag'].device).unsqueeze(-1)

        for data_type in self.config["data_type_ffm"]:
            for data_str in ['data', 'data_mes']:
                if data_type in eval(data_str):
                    #eval(data_str)[data_type] = self.ffm[data_type](eval(data_str)[data_type])
                    exec(f'eval(data_str)[data_type] = self.ffm_{data_type}(eval(data_str)[data_type])')

        device = data['atf_mag'].device
        hyper_en_x = th.zeros(S, M_mes, 0, 1, device=device, dtype=th.float32) # (S, M_mes, 0, 1)
        for data_type in self.config["data_type_interp"]:
            hyper_en_x = th.cat([hyper_en_x, data_mes[data_type][:,:,:,None]], dim=2) # (S, M_mes, F, 1)

        if len(self.config["data_type_interp"]) == 1:
            hyper_en_z = th.zeros(S, M_mes, F, 0, device=device, dtype=th.float32)
            for data_type in self.config["data_type_hyper_en"]:
                if data_type in ['src_position', 'mic_position']:
                    hyper_en_z = th.cat([hyper_en_z, data_mes[data_type].unsqueeze(2).tile(1, 1, F, 1)], dim=3) # (S, M_mes, F, 3 or num_ff)
                elif data_type in ['frequency']:
                    hyper_en_z = th.cat([hyper_en_z, data[data_type][None,None,:,:].tile(S, M_mes, 1, 1)], dim=3) # (S, M_mes, F, 1 or num_ff)
                elif data_type in ['M_mes']:
                    hyper_en_z = th.cat([hyper_en_z, data[data_type][None,None,:,:].tile(S, M_mes, F, 1)], dim=3) # (S, M_mes, F, 1 or num_ff)
        else:
            raise NotImplementedError
        
        #ic(hyper_en_x.shape)
        #ic(hyper_en_z.shape)

        latents = self.en_0({
            "x": hyper_en_x, # (S, M_mes, 1 or F, 1)
            "z": hyper_en_z # (S, M_mes, 1 or F, *)
            })["x"]
        # (S, M_mes, 1 or F, d)
        FF = latents.shape[2]
        #ic(latents.shape)

        returns["z_m"] = latents
        latents = th.mean(latents, dim=self.config["mid_mean_dim"], keepdim=True) # (S, 1, 1 or F, d)
        returns["z"] = latents
        if self.config["mid_mean_dim"] == (1,):
            latents = latents.tile(1, M, 1, 1)
        elif self.config["mid_mean_dim"] == (1, 2):
            latents = latents.tile(1, M, FF, 1)

        if len(self.config["data_type_interp"]) == 1:
            hyper_de_z = th.zeros(S, M, F, 0, device=device, dtype=th.float32)
            for data_type in self.config["data_type_hyper_en"]:
                if data_type in ['src_position', 'mic_position']:
                    hyper_de_z = th.cat([hyper_de_z, data[data_type].unsqueeze(2).tile(1, 1, F, 1)], dim=3) # (S, M, F, 3 or num_ff)
                elif data_type in ['frequency']:
                    hyper_de_z = th.cat([hyper_de_z, data[data_type][None,None,:,:].tile(S, M, 1, 1)], dim=3) # (S, M, F, 1 or num_ff)
        else:
            raise NotImplementedError
        
        #ic(latents.shape)
        #ic(hyper_de_z.shape)

        out_f = self.de_m1({
            "x": latents, # (S, M, 1 or F, d)
            "z": hyper_de_z # (S, M, 1 or F, *)
            })["x"]
        
        # ic(out_f.shape)

        if len(self.config["data_type_interp"]) == 1:
            data_type = self.config["data_type_interp"][0]
            returns[data_type] = out_f[:,:,:,0]
        else: 
            raise NotImplementedError
        
        for data_type in self.config["data_type_interp"]:
            returns[data_type] = returns[data_type] * self.config["data_stat"][data_type][dataset_name]["std"] + self.config["data_stat"][data_type][dataset_name]["mean"]

        #for data_type in returns:
        #    ic(data_type)
        #    ic(returns[data_type].shape)

        return returns
    
class ATFApproxNetworkPI(utils.Net):
    def __init__(self,
                 config,
                 use_cuda=True):
        super().__init__()
        self.config = config
        self.num_freq = config["num_freq"]
        
        # Fourier feature mapping
        if len(self.config["data_type_ffm"]) > 0:
            # self.ffm = {}
            for data_type in self.config["data_type_ffm"]:
                # self.ffm[data_type] = FourierFeatureMapping(num_ff = self.config["num_ff"][data_type], dim_data = self.config["dim_data_hyper"][data_type], trainable = self.config["ffm_trainable"])
                 exec(f'self.ffm_{data_type} = FourierFeatureMapping(num_ff=self.config["num_ff"][data_type], dim_data=self.config["dim_data_hyper"][data_type], trainable=self.config["ffm_trainable"])')
        
        # Encoder
        if self.config["hlayers_En_0"] > 0:
            modules = []

            for l in range(self.config["hlayers_En_0"]):
                if l == 0: # input layer
                    ch_in = 1
                else:
                    ch_in = self.config["channel_En_0"]
                
                if l == self.config["hlayers_En_0"] - 1: # output layer
                    ch_out = 1 if not config["aggr_mean"] else config["dim_z"]
                    post_proc = False
                else:
                    ch_out = self.config["channel_En_0"]
                    post_proc = True
                
                if self.config["hyperlinear_en_0"]:
                    input_size_en_0 = 0
                    for data_type in self.config["data_type_hyper_en"]:
                        if data_type in self.config["data_type_ffm"]:
                            if data_type == "src_position":
                                pass
                            else:
                                input_size_en_0 += self.config["num_ff"][data_type] * 2
                        else:
                            input_size_en_0 += self.config["dim_data_hyper"][data_type]
                    #if len(self.config['data_type_interp']) > 1:
                    #    input_size_en_0 += 1 # delta
                    modules.extend([
                        HyperLinearBlock(ch_in=ch_in, ch_out=ch_out, ch_hidden=self.config["channel_hyper"], num_hidden=self.config["hlayers_hyper"], droprate=self.config["droprate"], use_res=self.config["hyper_use_res"], input_size=input_size_en_0, post_proc=post_proc)
                        ])
                else:
                    raise NotImplementedError
                
            self.en_0 = nn.Sequential(*modules)
                
        # Decoder
        if config["hlayers_De_z"] > 0:
            modules = []
            for l in range(config["hlayers_De_z"]):
                if l == 0: # last layer
                    ch_in = config["dim_z"]
                else:
                    ch_in = config["channel_De_z"]
                if self.config["use_freq_for_hyper"]:
                    # x:(S,L,ch_out), z:(S,L,1)
                    modules.extend([
                        HyperLinearBlock(ch_in=ch_in, ch_out=config["channel_De_z"], ch_hidden=config["channel_hyper"], num_hidden=config["hlayers_hyper"], droprate=config["droprate"], use_res=config["hyper_use_res"], input_size=1),
                    ])
                else:
                    # input:(S,L,ch_out)
                    modules.extend([
                        nn.Linear(in_features=ch_in, out_features=config["channel_De_z"], bias=True),
                        nn.LayerNorm(config["channel_De_z"]),
                        nn.ReLU(),
                        nn.Dropout(config["droprate"])
                    ])
                # x or output:(S,L,ch_out)
            self.de_1 = nn.Sequential(*modules)

        if config["hlayers_De_z"] == 0:
            ch_in = config["dim_z"]
        else:
            ch_in = config["channel_De_z"]

        if self.config["hlayers_De_-1"]>0:
            # x:(S,L,ch_out)->(S,L,B,ch_out)->(S,L,B,1)->(S,L,B)
            modules = []
            for l in range(config["hlayers_De_-1"]):
                if l == 0: # first layer
                    if not self.config["de_2_skip"]:
                        ch_in = 1
                    else:
                        ch_in = config["dim_z"]
                else:
                    ch_in = config["channel_De_-1"]

                if l == config["hlayers_De_-1"] - 1: # last layer
                    ch_out = 1
                    post_proc = False
                else:
                    ch_out = config["channel_De_-1"]
                    post_proc = True

                if self.config["hyperlinear_de_-1"]:
                    input_size_de_m1 = 0
                    for data_type in self.config["data_type_hyper_de"]:
                        if data_type in self.config["data_type_ffm"]:
                            if data_type == "src_position":
                                pass
                            else:
                                input_size_de_m1 += self.config["num_ff"][data_type] * 2
                        else:
                            input_size_de_m1 += self.config["dim_data_hyper"][data_type]
                    #if len(self.config["data_type_interp"]) > 1:
                    #    input_size_de_m1 += 1 # delta

                    modules.extend([
                        HyperLinearBlock(ch_in=ch_in, ch_out=ch_out, ch_hidden=config["channel_hyper"], num_hidden=config["hlayers_hyper"], droprate=config["droprate"], use_res=config["hyper_use_res"], input_size=input_size_de_m1, post_proc=post_proc),
                    ])

            self.de_m1 = nn.Sequential(*modules)
            # x:(S,L,ch_out)->(S,L,B,ch_out)->(S,L,B,1)->(S,L,B)

    def forward(self, data, mode='train'):
        '''
        :param: data: dict contains
            'src_position':  (S,B,3) torch.float32 [x,y,z]
            'mic_position':  (S,B,3) torch.float32 [x,y,z]
            'rir':     (S,B,2,L) torch.float32
            'atf_mag':     (S,B,2,L) torch.float32
            'dataset':      list of str
            'src_index':      (S) torch.int64

        :return: out: 
            - HRTF produce as a (B, 4L, S) complex tensor
                4L ... [l_re, l_im, r_re, r_im]
            - HRTF's magnitude as a (B,2,L,S) float tensor
            - HRIR as a (B,2,L',S) float tensor
            - ITDs as a (B, S) float tensor
        '''
        returns = {}
        dataset_name = data["dataset"][0]

        F = self.num_freq
        S, M, _ = data["src_position"].shape

        if mode == 'train':
            perm = th.randperm(M)
            idx_mes_pos = perm[:self.config["num_mes"]]
        else:
            #idx_mes_pos = th.arange(M)
            #perm = th.randperm(M)
            #idx_mes_pos = perm[:self.config["num_mes_test"]]
            idx_mes_pos = self.config["idx_mes_pos_mat"][:self.config["num_mes_test"], 0]
        #perm = th.randperm(M)
        #idx_mes_pos = perm[:self.config["num_mes"]]
        returns["idx_mes_pos"] = idx_mes_pos
        
        M_mes = len(idx_mes_pos)
        data_mes = {}

        # Standardize
        #for data_type in ['atf_mag', 'src_position', 'mic_position']:#self.config["data_type_interp"]:
        #    data[data_type] = (data[data_type] - self.config["data_stat"][data_type][dataset_name]["mean"]) / self.config["data_stat"][data_type][dataset_name]["std"]
        data['atf_mag'] = (data['atf_mag'] - self.config["data_stat"]['atf_mag'][dataset_name]["mean"]) / self.config["data_stat"]['atf_mag'][dataset_name]["std"]
        data['src_position'] = data['src_position']/self.config["data_stat"]['src_position'][dataset_name]["max"]
        data['mic_position'] = data['mic_position']/self.config["data_stat"]['mic_position'][dataset_name]["max"]
        data['frequency'] = (th.arange(self.num_freq+1)[1:]/self.num_freq).to(data['atf_mag'].device).unsqueeze(-1)

        for k in data:
            if k in ['atf_mag', 'src_position', 'mic_position']:
                data_mes[k] = data[k][:, idx_mes_pos]

        M_mes_norm = self.config["M_mes_norm"] if "M_mes_norm" in self.config.keys() else M_mes
        data['M_mes'] = th.tensor([M_mes/M_mes_norm]).to(data['atf_mag'].device).unsqueeze(-1)

        for data_type in self.config["data_type_ffm"]:
            for data_str in ['data', 'data_mes']:
                if data_type in eval(data_str):
                    #eval(data_str)[data_type] = self.ffm[data_type](eval(data_str)[data_type])
                    exec(f'eval(data_str)[data_type] = self.ffm_{data_type}(eval(data_str)[data_type])')

        device = data['atf_mag'].device
        hyper_en_x = th.zeros(S, M_mes, 0, 1, device=device, dtype=th.float32) # (S, M_mes, 0, 1)
        for data_type in self.config["data_type_interp"]:
            hyper_en_x = th.cat([hyper_en_x, data_mes[data_type][:,:,:,None]], dim=2) # (S, M_mes, F, 1)

        if len(self.config["data_type_interp"]) == 1:
            hyper_en_z = th.zeros(S * M_mes, F, 0, device=device, dtype=th.float32)

            hyper_en_z = th.cat([hyper_en_z, th.flatten(data_mes['src_position'] + data_mes['mic_position'], 0, 1).unsqueeze(1).tile(1, F, 1)], dim=2)
            for data_type in self.config["data_type_hyper_en"]:
                #if data_type in ['src_position', 'mic_position']:
                #    hyper_en_z = th.cat([hyper_en_z, data_mes[data_type].unsqueeze(2).tile(1, 1, F, 1)], dim=3) # (S, M_mes, F, 3 or num_ff)
                if data_type in ['frequency']:
                    hyper_en_z = th.cat([hyper_en_z, data[data_type][None,:,:].tile(S * M_mes, 1, 1)], dim=2) # (S, M_mes, F, 1 or num_ff)
                elif data_type in ['M_mes']:
                    hyper_en_z = th.cat([hyper_en_z, data[data_type][None,:,:].tile(S * M_mes, F, 1)], dim=2) # (S, M_mes, F, 1 or num_ff)
        else:
            raise NotImplementedError
        
        #ic(hyper_en_x.shape)
        #ic(hyper_en_z.shape)

        latents = self.en_0({
            "x": hyper_en_x, # (S, M_mes, 1 or F, 1)
            "z": hyper_en_z # (S*M_mes, 1 or F, *)
            })["x"]
        # (S, M_mes, 1 or F, d)
        FF = latents.shape[2]
        #ic(latents.shape)

        returns["z_m"] = latents
        latents = th.mean(latents, dim=self.config["mid_mean_dim"], keepdim=True) # (S, 1, 1 or F, d)
        returns["z"] = latents
        if self.config["mid_mean_dim"] == (1,):
            latents = latents.tile(1, M, 1, 1)
        elif self.config["mid_mean_dim"] == (1, 2):
            latents = latents.tile(1, M, FF, 1)

        if len(self.config["data_type_interp"]) == 1:
            hyper_de_z = th.zeros(S * M, F, 0, device=device, dtype=th.float32)
            hyper_de_z = th.cat([hyper_de_z, th.flatten(data['src_position'] + data['mic_position'], 0, 1).unsqueeze(1).tile(1, F, 1)], dim=2) 
            for data_type in self.config["data_type_hyper_en"]:
                #if data_type in ['src_position', 'mic_position']:
                #    hyper_de_z = th.cat([hyper_de_z, data[data_type].unsqueeze(2).tile(1, 1, F, 1)], dim=3) # (S, M, F, 3 or num_ff)
                if data_type in ['frequency']:
                    hyper_de_z = th.cat([hyper_de_z, data[data_type][None,:,:].tile(S * M, 1, 1)], dim=2) # (S, M, F, 1 or num_ff)
        else:
            raise NotImplementedError
        
        #ic(latents.shape)
        #ic(hyper_de_z.shape)

        out_f = self.de_m1({
            "x": latents, # (S, M, 1 or F, d)
            "z": hyper_de_z # (S*M, 1 or F, *)
            })["x"]
        
        # ic(out_f.shape)

        if len(self.config["data_type_interp"]) == 1:
            data_type = self.config["data_type_interp"][0]
            returns[data_type] = out_f[:,:,:,0]
        else: 
            raise NotImplementedError
        
        for data_type in self.config["data_type_interp"]:
            returns[data_type] = returns[data_type] * self.config["data_stat"][data_type][dataset_name]["std"] + self.config["data_stat"][data_type][dataset_name]["mean"]

        #for data_type in returns:
        #    ic(data_type)
        #    ic(returns[data_type].shape)

        return returns
    

class NFfreq(utils.Net):
    def __init__(self, config, out_dim = 2, hidden_dim = 64, num_layers = 3, use_cuda=True):
        super().__init__()
        self.config = config
        self.num_freq = config["num_freq"]

        self.ffm_mic_position = FourierFeatureMapping(num_ff=self.config["num_ff"]["mic_position"], dim_data=self.config["dim_data_pinn"]["mic_position"], trainable=self.config["ffm_trainable"])

        modules = []
        for l in range(num_layers):
            if l == 0:
                ch_in = self.config["num_ff"]["mic_position"] * 2
            else:
                ch_in = hidden_dim
            
            modules.append(nn.Linear(ch_in, hidden_dim))
            modules.append(nn.Tanh())
        
        modules.append(nn.Linear(hidden_dim, out_dim))

        self.mlp = nn.Sequential(*modules)

    def forward(self, pos):
        pos_ff = self.ffm_mic_position(pos)
        out = self.mlp(pos_ff)
        out_c = th.complex(out[:,0], out[:,1])
        out_db = 20.0*th.log10(th.abs(out_c))
        return {"output": out, "output_c": out_c, "output_mag": out_db}


class NF(utils.Net):
    def __init__(self, config, out_dim = 1, hidden_dim = 128, num_layers = 2, use_cuda=True):
        super().__init__()
        self.config = config
        self.num_freq = config["num_freq"]

        self.ffm_src_position = FourierFeatureMapping(num_ff=self.config["num_ff"]["src_position"], dim_data=self.config["dim_data_nf"]["src_position"], trainable=self.config["ffm_trainable"])
        self.ffm_mic_position = FourierFeatureMapping(num_ff=self.config["num_ff"]["mic_position"], dim_data=self.config["dim_data_nf"]["mic_position"], trainable=self.config["ffm_trainable"])
        self.ffm_frequency = FourierFeatureMapping(num_ff=self.config["num_ff"]["frequency"], dim_data=self.config["dim_data_nf"]["frequency"], trainable=self.config["ffm_trainable"])

        modules = []
        for l in range(num_layers):
            if l == 0:
                ch_in = (self.config["num_ff"]["src_position"] + self.config["num_ff"]["mic_position"] + self.config["num_ff"]["frequency"]) * 2
            else:
                ch_in = hidden_dim
            
            modules.append(nn.Linear(ch_in, hidden_dim))
            modules.append(nn.ReLU()) #modules.append(nn.Tanh())
        
        modules.append(nn.Linear(hidden_dim, out_dim))

        self.mlp = nn.Sequential(*modules)

    def forward(self, data, mode="train"):
        returns = {}
        dataset_name = data["dataset"][0]

        F = self.num_freq
        S, M, _ = data["src_position"].shape

        if mode in ['train', 'valid', 'test']:
            #perm = th.randperm(M)
            #idx_mes_pos = perm[:self.config["num_mes"]]
            idx_mes_pos = th.arange(M)
        elif mode in ['fine_tuning']:
            #idx_mes_pos = th.arange(M)
            #perm = th.randperm(M)
            #idx_mes_pos = perm[:self.config["num_mes_test"]]
            idx_mes_pos = self.config["idx_mes_pos_mat"][:self.config["num_mes_test"], 0]
        #perm = th.randperm(M)
        #idx_mes_pos = perm[:self.config["num_mes"]]
        returns["idx_mes_pos"] = idx_mes_pos
        
        M_mes = len(idx_mes_pos)
        data_mes = {}

        # Standardize
        #for data_type in ['atf_mag', 'src_position', 'mic_position']:#self.config["data_type_interp"]:
        #    data[data_type] = (data[data_type] - self.config["data_stat"][data_type][dataset_name]["mean"]) / self.config["data_stat"][data_type][dataset_name]["std"]
        data['atf_mag'] = (data['atf_mag'] - self.config["data_stat"]['atf_mag'][dataset_name]["mean"]) / self.config["data_stat"]['atf_mag'][dataset_name]["std"]
        data['src_position'] = data['src_position']/self.config["data_stat"]['src_position'][dataset_name]["max"]
        data['mic_position'] = data['mic_position']/self.config["data_stat"]['mic_position'][dataset_name]["max"]
        data['frequency'] = (th.arange(self.num_freq+1)[1:]/self.num_freq).to(data['atf_mag'].device).unsqueeze(-1)

        for k in data:
            if k in ['atf_mag', 'src_position', 'mic_position']:
                data_mes[k] = data[k][:, idx_mes_pos]

        #M_mes_norm = self.config["M_mes_norm"] if "M_mes_norm" in self.config.keys() else M_mes
        #data['M_mes'] = th.tensor([M_mes/M_mes_norm]).to(data['atf_mag'].device).unsqueeze(-1)

        for data_type in self.config["data_type_ffm"]:
            for data_str in ['data', 'data_mes']:
                if data_type in eval(data_str):
                    #eval(data_str)[data_type] = self.ffm[data_type](eval(data_str)[data_type])
                    exec(f'eval(data_str)[data_type] = self.ffm_{data_type}(eval(data_str)[data_type])')

        device = data['atf_mag'].device
        indata = th.zeros(S, M_mes, F, 0, device=device, dtype=th.float32)

        for data_type in ['src_position', 'mic_position', 'frequency']:
            if data_type in ['src_position', 'mic_position']:
                indata = th.cat([indata, data_mes[data_type].unsqueeze(2).tile(1, 1, F, 1)], dim=3) # (S, M, F, 3 or num_ff)
            if data_type in ['frequency']:
                indata = th.cat([indata, data[data_type][None,None,:,:].tile(S, M_mes, 1, 1)], dim=3) # (S, M, F, 1 or num_ff)

        out = self.mlp(indata)[:,:,:,0]
        returns["atf_mag"] = out * self.config["data_stat"]['atf_mag'][dataset_name]["std"] + self.config["data_stat"]['atf_mag'][dataset_name]["mean"]

        return returns

