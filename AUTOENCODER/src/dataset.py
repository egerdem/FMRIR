import numpy as np
import torch as th
import os

_src_dir = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR_HPC   = os.path.join(_src_dir, '..', '..', '..', 'DATA')  # HPC: ~/DATA/ (sibling of FMRIR)
_DATA_DIR_LOCAL = os.path.join(_src_dir, '..', '..')                 # Local: FMRIR/ (datasets are directly inside)
_DATA_DIR = _DATA_DIR_HPC if os.path.isdir(_DATA_DIR_HPC) else _DATA_DIR_LOCAL

class ATFdataset:
    def __init__(self, config):
        self.config = config

        self.Data = {}
        for data_for in ['all', 'train', 'valid', 'test']:
            self.Data.setdefault(data_for, {})
            #for data_type in ['atf', 'atf_mag', 'src_position', 'mic_position']:
            for data_type in ['atf_mag', 'src_position', 'mic_position']:
                self.Data[data_for].setdefault(data_type, {})
        
        for dataset_name in self.config["dataset"]:
            for src_id in self.config['src_index'][dataset_name]['all']:
                path = os.path.join(_DATA_DIR, dataset_name, f"data_s{src_id+1:04d}.npz")
                data_np = np.load(path)
                if src_id == 0:
                    src_position = th.zeros(data_np['posMic'].shape[0], data_np['posSrc'].shape[0], len(self.config['src_index'][dataset_name]['all'])) # M x 3 x S
                    mic_position = th.zeros(data_np['posMic'].shape[0], data_np['posMic'].shape[1], len(self.config['src_index'][dataset_name]['all'])) # M x 3 x S
                    if self.config["init_delay"]:
                        #rir = th.zeros(data_np['arr_0'].shape[0], data_np['arr_0'].shape[1], len(self.config['src_index'][dataset_name]['all'])) # M x 2F x S
                        #atf = th.zeros(data_np['arr_1'].shape[0], data_np['arr_1'].shape[1], len(self.config['src_index'][dataset_name]['all']), dtype=th.cfloat) # M x F x S
                        atf_mag = th.zeros(data_np['atf_mag'].shape[0], data_np['atf_mag'].shape[1], len(self.config['src_index'][dataset_name]['all'])) # M x F x S
                    else:
                        #rir = th.zeros(data_np['arr_3'].shape[0], data_np['arr_3'].shape[1], len(self.config['src_index'][dataset_name]['all'])) # M x 2F x S
                        #atf = th.zeros(data_np['arr_4'].shape[0], data_np['arr_4'].shape[1], len(self.config['src_index'][dataset_name]['all']), dtype=th.cfloat) # M x F x S
                        atf_mag = th.zeros(data_np['atf_mag_algn'].shape[0], data_np['atf_mag_algn'].shape[1], len(self.config['src_index'][dataset_name]['all'])) # M x F x S
                    #print(src_position.shape, mic_position.shape, rir.shape, atf_mag.shape)
                src_position[:,:,src_id] = th.from_numpy(np.tile(data_np['posSrc'][None,:], (data_np['posMic'].shape[0], 1)).astype(np.float32)).clone()
                mic_position[:,:,src_id] = th.from_numpy(data_np['posMic'].astype(np.float32)).clone()
                if self.config["init_delay"]:
                    #rir[:,:,src_id] = th.from_numpy(data_np['arr_0'].astype(np.float32)).clone()
                    #atf[:,:,src_id] = th.from_numpy(data_np['arr_1'].astype(np.complex64)).clone()
                    atf_mag[:,:,src_id] = th.from_numpy(data_np['atf_mag'].astype(np.float32)).clone()
                else:
                    #rir[:,:,src_id] = th.from_numpy(data_np['arr_3'].astype(np.float32)).clone()
                    #atf[:,:,src_id] = th.from_numpy(data_np['arr_4'].astype(np.complex64)).clone()
                    atf_mag[:,:,src_id] = th.from_numpy(data_np['atf_mag_algn'].astype(np.float32)).clone()

            #self.Data['all']['rir'][dataset_name] = rir
            #self.Data['all']['atf'][dataset_name] = atf
            freq_from = self.config.get('freq_from', 0)
            freq_up_to = self.config.get('freq_up_to', atf_mag.shape[1])
            self.Data['all']['atf_mag'][dataset_name] = atf_mag[:, freq_from:freq_up_to, :]
            self.Data['all']['src_position'][dataset_name] = src_position
            self.Data['all']['mic_position'][dataset_name] = mic_position

            for data_for in ['train', 'valid', 'test']:
                #self.Data[data_for]['rir'][dataset_name] = self.Data['all']['rir'][dataset_name][..., self.config['src_index'][dataset_name][data_for]]
                #self.Data[data_for]['atf'][dataset_name] = self.Data['all']['atf'][dataset_name][..., self.config['src_index'][dataset_name][data_for]]
                self.Data[data_for]['atf_mag'][dataset_name] = self.Data['all']['atf_mag'][dataset_name][..., self.config['src_index'][dataset_name][data_for]]
                self.Data[data_for]['src_position'][dataset_name] = self.Data['all']['src_position'][dataset_name][..., self.config['src_index'][dataset_name][data_for]]
                self.Data[data_for]['mic_position'][dataset_name] = self.Data['all']['mic_position'][dataset_name]

        _idx_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', self.config["idx_mes_pos_mat_path"])
        self.config["idx_mes_pos_mat"] = np.load(_idx_path)

        self.DataStat = {}
        for data_type in self.Data['train']:
            self.DataStat[data_type] = {}
            for dataset_name in self.config['dataset']:
                if len(self.config['src_index'][dataset_name]['train']) > 0:
                    val = self.Data['train'][data_type][dataset_name]
                    #print(val.shape)
                    if data_type in ['rir', 'atf', 'atf_mag']:
                        mean = th.mean(val)
                        std = th.std(val)
                        max = th.max(val)
                        min = th.min(val)
                    elif data_type in ['src_position','mic_position']:
                        mean = th.mean(val)
                        std = th.std(val)
                        max = th.max(th.norm(val, dim=1))
                        min = th.min(th.norm(val, dim=1))
                    #print(mean.shape)
                    #print(std.shape)
                    print(f"[{data_type}, {dataset_name}] mean: {mean:.4}, std: {std:.4}, max: {max:.4}, min: {min:.4}")
                    self.DataStat[data_type][dataset_name] = {"mean": mean, "std": std, "max": max, "min": min}


        self.Table = {}
        for data_for in ['train', 'valid', 'test']:
            self.Table[data_for] = {
                'dataset': [],
                'src_index': []
            }
            for dataset_name in self.config['dataset']:
                self.Table[data_for]['dataset'].extend([dataset_name for _ in self.config['src_index'][dataset_name][data_for]])
                self.Table[data_for]['src_index'].extend(list(self.config['src_index'][dataset_name][data_for]))
        
        return
    
    def __len__(self):
        '''
        :return: number of training subjects in dataset
        '''
        return sum([len(self.config['src_index'][dataset_name]['train']) for dataset_name in self.config['dataset']])

    def __getitem__(self, index): # for train data
        '''
        :return: dict consisting of
            SrcPos as      B x 3 tensor
            HRTF as        B x 2 x L tensor
            HRTF_mag as    B x 2 x L tensor
            HRIR as        B x 2 x 2L tensor
            ITD  as        B tensor
            db_name as     list of str
            sub_idx as     list of? int
        '''
        dataset_name = self.Table['train']["dataset"][index]
        src_idx = self.Table['train']["src_index"][index]
        returns = {data_type: self.Data['train'][data_type][dataset_name][..., src_idx] for data_type in self.Data['train']}
        returns['dataset'] = dataset_name
        returns['src_index'] = src_idx
        return returns
    
    def trainitem(self): # for validation data
        '''
        :return: dict
                return[data_kind][db_name]: tensor
        '''
        return self.Data['train']
    
    def validitem(self): # for validation data
        '''
        :return: dict
                return[data_kind][db_name]: tensor
        '''
        return self.Data['valid']
    
    def testitem(self): # for test data
        '''
        :return: dict
                return[data_kind][db_name]: tensor
        '''
        return self.Data['test']

