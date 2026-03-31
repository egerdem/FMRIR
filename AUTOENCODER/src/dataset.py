import glob
import os
import re

import numpy as np
import torch as th

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
            base_dir = self.config.get("data_dir")
            if not base_dir:
                base_dir = _DATA_DIR
            base_dir = os.path.expanduser(base_dir)
            dataset_dir = os.path.join(base_dir, dataset_name)

            loaded_from_pt = self._load_dataset_from_processed_pt(dataset_name, dataset_dir)
            if not loaded_from_pt:
                self._load_dataset_from_npz(dataset_name, dataset_dir)

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
                    print(f"[{data_type}, {dataset_name}] mean: {mean:.4f}, std: {std:.4f}, max: {max:.4f}, min: {min:.4f}")
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

    def _expand_src_indices(self, dataset_name, split_name):
        return list(self.config['src_index'][dataset_name][split_name])

    def _apply_frequency_slice(self, atf_mag, file_freq_from=0):
        freq_from = int(self.config.get('freq_from', 0))
        req_up_to = self.config.get('freq_up_to', None)
        if req_up_to is None:
            req_up_to = atf_mag.shape[1]
        req_up_to = int(req_up_to)

        # Convert requested absolute bins to local indices if file itself starts at file_freq_from.
        local_from = max(0, freq_from - file_freq_from)
        local_up_to = max(local_from, req_up_to - file_freq_from)
        local_up_to = min(local_up_to, atf_mag.shape[1])
        return atf_mag[:, local_from:local_up_to, :]

    def _assign_dataset_splits(self, dataset_name, atf_mag, src_position, mic_position):
        self.Data['all']['atf_mag'][dataset_name] = atf_mag
        self.Data['all']['src_position'][dataset_name] = src_position
        self.Data['all']['mic_position'][dataset_name] = mic_position

        for data_for in ['train', 'valid', 'test']:
            split_indices = self._expand_src_indices(dataset_name, data_for)
            self.Data[data_for]['atf_mag'][dataset_name] = self.Data['all']['atf_mag'][dataset_name][..., split_indices]
            self.Data[data_for]['src_position'][dataset_name] = self.Data['all']['src_position'][dataset_name][..., split_indices]
            self.Data[data_for]['mic_position'][dataset_name] = self.Data['all']['mic_position'][dataset_name]

    def _parse_freq_tag(self, path):
        m = re.search(r'_freqs([^_]+)_', os.path.basename(path))
        if not m:
            return None, None
        tag = m.group(1)
        if 'to' in tag:
            a, b = tag.split('to')
            if a.isdigit() and b.isdigit():
                return int(a), int(b)
            return None, None
        if tag.isdigit():
            return 0, int(tag)
        return None, None

    def _select_processed_file(self, dataset_dir, mode):
        pattern = os.path.join(dataset_dir, f'processed_atf3d_{mode}_freqs*.pt')
        candidates = sorted(glob.glob(pattern))
        if not candidates:
            return None

        req_from = int(self.config.get('freq_from', 0))
        req_up_to = self.config.get('freq_up_to', None)
        req_up_to = int(req_up_to) if req_up_to is not None else None

        def score(path):
            f_from, f_to = self._parse_freq_tag(path)
            exact = 0
            covers = 0
            span = 10**9

            if f_from is not None and f_to is not None:
                span = f_to - f_from
                if req_up_to is not None:
                    if (f_from == req_from) and (f_to == req_up_to):
                        exact = 1
                    if (req_from >= f_from) and (req_up_to <= f_to):
                        covers = 1
                else:
                    if f_from == req_from:
                        covers = 1
            return (exact, covers, -span)

        candidates = sorted(candidates, key=score, reverse=True)
        return candidates[0]

    def _load_dataset_from_processed_pt(self, dataset_name, dataset_dir):
        required_modes = ['train', 'valid', 'test']
        mode_files = {}
        for mode in required_modes:
            selected = self._select_processed_file(dataset_dir, mode)
            if selected is None:
                return False
            mode_files[mode] = selected

        print(f"[{dataset_name}] Loading from processed .pt files")
        for mode in required_modes:
            print(f"  {mode}: {mode_files[mode]}")

        all_ids = self._expand_src_indices(dataset_name, 'all')
        id_to_pos = {src_id: i for i, src_id in enumerate(all_ids)}

        atf_mag_full = None
        src_position_full = None
        mic_position_full = None
        file_freq_from_reference = 0

        for mode in required_modes:
            payload = th.load(mode_files[mode], map_location='cpu')
            if not all(k in payload for k in ['cubes', 'source_coords', 'grid_xyz']):
                print(f"[{dataset_name}] Missing expected keys in {mode_files[mode]}; falling back to .npz")
                return False

            cubes = payload['cubes'].to(th.float32)          # [S, F, Z, Y, X]
            source_coords = payload['source_coords'].to(th.float32)  # [S, 3]
            grid_xyz = payload['grid_xyz'].to(th.float32)    # [M, 3]

            if cubes.ndim != 5 or source_coords.ndim != 2 or grid_xyz.ndim != 2:
                print(f"[{dataset_name}] Unexpected tensor ranks in {mode_files[mode]}; falling back to .npz")
                return False

            num_src_mode, num_freq = cubes.shape[0], cubes.shape[1]
            num_mics = int(cubes.shape[2] * cubes.shape[3] * cubes.shape[4])
            if grid_xyz.shape[0] != num_mics:
                print(f"[{dataset_name}] grid_xyz and cube mic counts mismatch; falling back to .npz")
                return False

            # FM cubes are [S,F,Z,Y,X]. AE expects [M,F,S].
            atf_mag_mode = cubes.permute(2, 3, 4, 1, 0).contiguous().view(num_mics, num_freq, num_src_mode)
            mic_position_mode = grid_xyz.unsqueeze(-1).repeat(1, 1, num_src_mode)  # [M, 3, S]
            src_position_mode = source_coords.unsqueeze(0).repeat(num_mics, 1, 1)   # [M, 3, S]

            sample_info = payload.get('sample_info', None)
            if sample_info is not None:
                src_ids_mode = sample_info.view(-1).to(th.int64).tolist()
                if len(src_ids_mode) != num_src_mode:
                    print(f"[{dataset_name}] sample_info length mismatch; falling back to .npz")
                    return False
            else:
                src_ids_mode = self._expand_src_indices(dataset_name, mode)
                if len(src_ids_mode) != num_src_mode:
                    print(f"[{dataset_name}] No sample_info and split length mismatch; falling back to .npz")
                    return False

            if atf_mag_full is None:
                num_all = len(all_ids)
                atf_mag_full = th.zeros((num_mics, num_freq, num_all), dtype=th.float32)
                src_position_full = th.zeros((num_mics, 3, num_all), dtype=th.float32)
                mic_position_full = th.zeros((num_mics, 3, num_all), dtype=th.float32)
                parsed_from, _ = self._parse_freq_tag(mode_files[mode])
                file_freq_from_reference = parsed_from if parsed_from is not None else 0

            for local_idx, src_id in enumerate(src_ids_mode):
                if src_id not in id_to_pos:
                    continue
                target_idx = id_to_pos[src_id]
                atf_mag_full[:, :, target_idx] = atf_mag_mode[:, :, local_idx]
                src_position_full[:, :, target_idx] = src_position_mode[:, :, local_idx]
                mic_position_full[:, :, target_idx] = mic_position_mode[:, :, local_idx]

        if atf_mag_full is None:
            return False

        atf_mag_full = self._apply_frequency_slice(atf_mag_full, file_freq_from=file_freq_from_reference)
        self._assign_dataset_splits(dataset_name, atf_mag_full, src_position_full, mic_position_full)
        return True

    def _load_dataset_from_npz(self, dataset_name, dataset_dir):
        print(f"[{dataset_name}] Loading from individual .npz files")
        src_all = self._expand_src_indices(dataset_name, 'all')
        src_pos_to_idx = {src_id: i for i, src_id in enumerate(src_all)}

        atf_mag = None
        src_position = None
        mic_position = None

        for src_id in src_all:
            path = os.path.join(dataset_dir, f"data_s{src_id+1:04d}.npz")
            data_np = np.load(path)
            target_idx = src_pos_to_idx[src_id]
            if atf_mag is None:
                num_mic = data_np['posMic'].shape[0]
                num_dim = data_np['posMic'].shape[1]
                num_all_src = len(src_all)
                src_position = th.zeros(num_mic, data_np['posSrc'].shape[0], num_all_src)
                mic_position = th.zeros(num_mic, num_dim, num_all_src)
                if self.config["init_delay"]:
                    atf_mag = th.zeros(num_mic, data_np['atf_mag'].shape[1], num_all_src)
                else:
                    atf_mag = th.zeros(num_mic, data_np['atf_mag_algn'].shape[1], num_all_src)

            src_position[:, :, target_idx] = th.from_numpy(
                np.tile(data_np['posSrc'][None, :], (data_np['posMic'].shape[0], 1)).astype(np.float32)
            ).clone()
            mic_position[:, :, target_idx] = th.from_numpy(data_np['posMic'].astype(np.float32)).clone()
            if self.config["init_delay"]:
                atf_mag[:, :, target_idx] = th.from_numpy(data_np['atf_mag'].astype(np.float32)).clone()
            else:
                atf_mag[:, :, target_idx] = th.from_numpy(data_np['atf_mag_algn'].astype(np.float32)).clone()

        atf_mag = self._apply_frequency_slice(atf_mag, file_freq_from=0)
        self._assign_dataset_splits(dataset_name, atf_mag, src_position, mic_position)
    
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

