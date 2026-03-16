config_FSMPAE_10001 = {
        "fs": 8000,#16000,
        "num_freq": 256, #512,#1024,
        "dataset": ['ir_fs8000_s128_m1331_room4.0x6.0x3.0_rt200'],
        "src_index": {
            'ir_fs8000_s128_m1331_room4.0x6.0x3.0_rt200': {
                'train': range(0, 102),
                'valid': range(102, 115),
                'test': range(115, 128),
                'all': range(0, 128)
            }
        },
        "num_mes_list": list(reversed([50])),
        "M_mes_norm": 50,
        "model": 'FSMPAE',
        "learning_rate": 1e-3,
        "epochs": 1400,
        "lr_update": 'step',
        "lr_milestones": [800, 1200],
        "lr_gamma": 0.1,
        "batch_size": 1, #16,
        "loss_weights": {
            "lsd": 1.0
        },
        "save_frequency": 500,
        "activation_function": 'nn.Mish()',
        "data_type_interp": ['atf_mag'],
        "data_type_hyper_en": ['src_position', 'mic_position', 'frequency', 'M_mes'],
        "data_type_hyper_de": ['src_position', 'mic_position', 'frequency'],
        "dim_data_hyper": {
            'src_position': 3,
            'mic_position': 3,
            'frequency': 1,
            'M_mes': 1
        },
        "mid_mean_dim": (1,),
        "aggr_mean": True,
        # Fourier feature mapping
        "data_type_ffm": ['src_position', 'mic_position', 'frequency'],
        "num_ff": {
            'src_position': 16,
            'mic_position': 16,
            'frequency': 8,
            'M_mes': 16
        },
        "ffm_trainable": True,
        # Hypernetwork
        "de_2_skip": True,
        "dim_z": 64,
        "channel_En_0": 16,
        "channel_En_z": 128,
        "channel_De_z": 128,
        "channel_De_-1": 16,
        "channel_hyper": 64,
        "hyper_use_res": True,
        "hlayers_En_0": 2,
        "hlayers_En_z": 0,
        "hyperlinear_en_0": True,
        "hlayers_De_z": 0,
        "hlayers_De_-1": 2,
        "hlayers_hyper": 2,
        "hyperlinear_de_0": True,
        "hyperlinear_de_-1": True,
        "droprate": 0.0,
        "newbob_decay": 0.5,
        "newbob_max_decay": 1e-06,
        "num_gpus": 1,
        "timestamp": "",
    }

config_FSMPAE_10002 = {
        "fs": 8000,#16000,
        "num_freq": 256, #512,#1024,
        "dataset": ['ir_fs8000_s256_m1331_room4.0x6.0x3.0_rt200'],
        "src_index": {
            'ir_fs8000_s256_m1331_room4.0x6.0x3.0_rt200': {
                'train': range(0, 204),
                'valid': range(204, 230),
                'test': range(230, 256),
                'all': range(0, 256)
            }
        },
        "num_mes_list": list(reversed([50])),
        "M_mes_norm": 50,
        "model": 'FSMPAE',
        "learning_rate": 1e-3,
        "epochs": 1400,
        "lr_update": 'step',
        "lr_milestones": [800, 1200],
        "lr_gamma": 0.1,
        "batch_size": 1, #16,
        "loss_weights": {
            "lsd": 1.0
        },
        "save_frequency": 500,
        "activation_function": 'nn.Mish()',
        "data_type_interp": ['atf_mag'],
        "data_type_hyper_en": ['src_position', 'mic_position', 'frequency', 'M_mes'],
        "data_type_hyper_de": ['src_position', 'mic_position', 'frequency'],
        "dim_data_hyper": {
            'src_position': 3,
            'mic_position': 3,
            'frequency': 1,
            'M_mes': 1
        },
        "mid_mean_dim": (1,),
        "aggr_mean": True,
        # Fourier feature mapping
        "data_type_ffm": ['src_position', 'mic_position', 'frequency'],
        "num_ff": {
            'src_position': 16,
            'mic_position': 16,
            'frequency': 8,
            'M_mes': 16
        },
        "ffm_trainable": True,
        # Hypernetwork
        "de_2_skip": True,
        "dim_z": 64,
        "channel_En_0": 16,
        "channel_En_z": 128,
        "channel_De_z": 128,
        "channel_De_-1": 16,
        "channel_hyper": 64,
        "hyper_use_res": True,
        "hlayers_En_0": 2,
        "hlayers_En_z": 0,
        "hyperlinear_en_0": True,
        "hlayers_De_z": 0,
        "hlayers_De_-1": 2,
        "hlayers_hyper": 2,
        "hyperlinear_de_0": True,
        "hyperlinear_de_-1": True,
        "droprate": 0.0,
        "newbob_decay": 0.5,
        "newbob_max_decay": 1e-06,
        "num_gpus": 1,
        "timestamp": "",
    }

config_FSMPAE_10003 = {
        "fs": 8000,#16000,
        "num_freq": 256, #512,#1024,
        "init_delay": False,
        "dataset": ['ir_fs8000_s1024_m1331_room4.0x6.0x3.0_rt200'],
        "src_index": {
            'ir_fs8000_s1024_m1331_room4.0x6.0x3.0_rt200': {
                'train': range(0, 820),
                'valid': range(820, 922),
                'test': range(922, 1024),
                'all': range(0, 1024)
            }
        },
        "idx_mes_pos_mat_path": 'idx_mes_pos_s1024_m1331.npy',
        "num_mes_list": list(reversed([50])),
        "num_mes_test": 50,
        "M_mes_norm": 50,
        "model": 'FSMPAE',
        "learning_rate": 1e-3,
        "epochs": 1400,
        "lr_update": 'step',
        "lr_milestones": [800, 1200],
        "lr_gamma": 0.1,
        "batch_size": 1, #16,
        "loss_weights": {
            "lsd": 1.0
        },
        "save_frequency": 500,
        "activation_function": 'nn.Mish()',
        "data_type_interp": ['atf_mag'],
        "data_type_hyper_en": ['src_position', 'mic_position', 'frequency', 'M_mes'],
        "data_type_hyper_de": ['src_position', 'mic_position', 'frequency'],
        "dim_data_hyper": {
            'src_position': 3,
            'mic_position': 3,
            'frequency': 1,
            'M_mes': 1
        },
        "mid_mean_dim": (1,),
        "aggr_mean": True,
        # Fourier feature mapping
        "data_type_ffm": ['src_position', 'mic_position', 'frequency'],
        "num_ff": {
            'src_position': 16,
            'mic_position': 16,
            'frequency': 8,
            'M_mes': 16
        },
        "ffm_trainable": True,
        # Hypernetwork
        "de_2_skip": True,
        "dim_z": 64,
        "channel_En_0": 16,
        "channel_En_z": 128,
        "channel_De_z": 128,
        "channel_De_-1": 16,
        "channel_hyper": 64,
        "hyper_use_res": True,
        "hlayers_En_0": 2,
        "hlayers_En_z": 0,
        "hyperlinear_en_0": True,
        "hlayers_De_z": 0,
        "hlayers_De_-1": 2,
        "hlayers_hyper": 2,
        "hyperlinear_de_0": True,
        "hyperlinear_de_-1": True,
        "droprate": 0.0,
        "newbob_decay": 0.5,
        "newbob_max_decay": 1e-06,
        "num_gpus": 1,
        "timestamp": "",
    }

config_FSMPAE_10004 = {
        "fs": 8000,#16000,
        "num_freq": 256, #512,#1024,
        "init_delay": False,
        "dataset": ['ir_fs8000_s1024_m1331_room4.0x6.0x3.0_rt200'],
        "src_index": {
            'ir_fs8000_s1024_m1331_room4.0x6.0x3.0_rt200': {
                'train': range(0, 820),
                'valid': range(820, 922),
                'test': range(922, 1024),
                'all': range(0, 1024)
            }
        },
        "num_mes_list": list(reversed([50])),
        "M_mes_norm": 50,
        "model": 'FSMPAE',
        "learning_rate": 1e-3,
        "epochs": 1400,
        "lr_update": 'step',
        "lr_milestones": [800, 1200],
        "lr_gamma": 0.1,
        "batch_size": 1, #16,
        "loss_weights": {
            "lsd": 1.0
        },
        "save_frequency": 500,
        "activation_function": 'nn.Mish()',
        "data_type_interp": ['atf_mag'],
        "data_type_hyper_en": ['src_position', 'mic_position', 'frequency', 'M_mes'],
        "data_type_hyper_de": ['src_position', 'mic_position', 'frequency'],
        "dim_data_hyper": {
            'src_position': 3,
            'mic_position': 3,
            'frequency': 1,
            'M_mes': 1
        },
        "mid_mean_dim": (1,),
        "aggr_mean": True,
        # Fourier feature mapping
        "data_type_ffm": ['src_position', 'mic_position', 'frequency'],
        "num_ff": {
            'src_position': 16,
            'mic_position': 16,
            'frequency': 8,
            'M_mes': 16
        },
        "ffm_trainable": True,
        # Hypernetwork
        "de_2_skip": True,
        "dim_z": 128, #64,
        "channel_En_0": 32, #16,
        "channel_En_z": 256, #128,
        "channel_De_z": 256, #128,
        "channel_De_-1": 32, #16,
        "channel_hyper": 128, #64,
        "hyper_use_res": True,
        "hlayers_En_0": 2,
        "hlayers_En_z": 0,
        "hyperlinear_en_0": True,
        "hlayers_De_z": 0,
        "hlayers_De_-1": 2,
        "hlayers_hyper": 2,
        "hyperlinear_de_0": True,
        "hyperlinear_de_-1": True,
        "droprate": 0.0,
        "newbob_decay": 0.5,
        "newbob_max_decay": 1e-06,
        "num_gpus": 1,
        "timestamp": "",
    }

config_FSMPAE_10005 = {
        "fs": 8000,#16000,
        "num_freq": 256, #512,#1024,
        "init_delay": False,
        "dataset": ['ir_fs8000_s1024_m1331_room4.0x6.0x3.0_rt200'],
        "src_index": {
            'ir_fs8000_s1024_m1331_room4.0x6.0x3.0_rt200': {
                'train': range(0, 820),
                'valid': range(820, 922),
                'test': range(922, 1024),
                'all': range(0, 1024)
            }
        },
        "num_mes_list": list(reversed([50])),
        "M_mes_norm": 50,
        "model": 'FSMPAE',
        "learning_rate": 1e-3,
        "epochs": 1400,
        "lr_update": 'step',
        "lr_milestones": [800, 1200],
        "lr_gamma": 0.1,
        "batch_size": 1, #16,
        "loss_weights": {
            "lsd": 1.0
        },
        "save_frequency": 500,
        "activation_function": 'nn.Mish()',
        "data_type_interp": ['atf_mag'],
        "data_type_hyper_en": ['src_position', 'mic_position', 'frequency', 'M_mes'],
        "data_type_hyper_de": ['src_position', 'mic_position', 'frequency'],
        "dim_data_hyper": {
            'src_position': 3,
            'mic_position': 3,
            'frequency': 1,
            'M_mes': 1
        },
        "mid_mean_dim": (1,),
        "aggr_mean": True,
        # Fourier feature mapping
        "data_type_ffm": ['src_position', 'mic_position', 'frequency'],
        "num_ff": {
            'src_position': 8,
            'mic_position': 8,
            'frequency': 4,
            'M_mes': 8
        },
        "ffm_trainable": True,
        # Hypernetwork
        "de_2_skip": True,
        "dim_z": 32,
        "channel_En_0": 8,
        "channel_En_z": 64,
        "channel_De_z": 64,
        "channel_De_-1": 8,
        "channel_hyper": 32,
        "hyper_use_res": True,
        "hlayers_En_0": 2,
        "hlayers_En_z": 0,
        "hyperlinear_en_0": True,
        "hlayers_De_z": 0,
        "hlayers_De_-1": 2,
        "hlayers_hyper": 2,
        "hyperlinear_de_0": True,
        "hyperlinear_de_-1": True,
        "droprate": 0.0,
        "newbob_decay": 0.5,
        "newbob_max_decay": 1e-06,
        "num_gpus": 1,
        "timestamp": "",
    }

config_FSMPAE_10006 = {
        "fs": 8000,#16000,
        "num_freq": 256, #512,#1024,
        "init_delay": False,
        "dataset": ['ir_fs8000_s1024_m1331_room4.0x6.0x3.0_rt200'],
        "src_index": {
            'ir_fs8000_s1024_m1331_room4.0x6.0x3.0_rt200': {
                'train': range(0, 820),
                'valid': range(820, 922),
                'test': range(922, 1024),
                'all': range(0, 1024)
            }
        },
        "num_mes_list": list(reversed([50])),
        "M_mes_norm": 50,
        "model": 'FSMPAE',
        "learning_rate": 1e-3,
        "epochs": 1400,
        "lr_update": 'step',
        "lr_milestones": [800, 1200],
        "lr_gamma": 0.1,
        "batch_size": 1, #16,
        "loss_weights": {
            "lsd": 1.0
        },
        "save_frequency": 500,
        "activation_function": 'nn.Mish()',
        "data_type_interp": ['atf_mag'],
        "data_type_hyper_en": ['src_position', 'mic_position', 'frequency', 'M_mes'],
        "data_type_hyper_de": ['src_position', 'mic_position', 'frequency'],
        "dim_data_hyper": {
            'src_position': 3,
            'mic_position': 3,
            'frequency': 1,
            'M_mes': 1
        },
        "mid_mean_dim": (1,),
        "aggr_mean": True,
        # Fourier feature mapping
        "data_type_ffm": ['src_position', 'mic_position', 'frequency'],
        "num_ff": {
            'src_position': 16,
            'mic_position': 16,
            'frequency': 8,
            'M_mes': 16
        },
        "ffm_trainable": True,
        # Hypernetwork
        "de_2_skip": True,
        "dim_z": 32,
        "channel_En_0": 8,
        "channel_En_z": 64,
        "channel_De_z": 64,
        "channel_De_-1": 8,
        "channel_hyper": 32,
        "hyper_use_res": True,
        "hlayers_En_0": 2,
        "hlayers_En_z": 0,
        "hyperlinear_en_0": True,
        "hlayers_De_z": 0,
        "hlayers_De_-1": 2,
        "hlayers_hyper": 2,
        "hyperlinear_de_0": True,
        "hyperlinear_de_-1": True,
        "droprate": 0.0,
        "newbob_decay": 0.5,
        "newbob_max_decay": 1e-06,
        "num_gpus": 1,
        "timestamp": "",
    }

config_FSMPAE_10007 = {
        "fs": 8000,#16000,
        "num_freq": 256, #512,#1024,
        "init_delay": False,
        "dataset": ['ir_fs8000_s1024_m1331_room4.0x6.0x3.0_rt200'],
        "src_index": {
            'ir_fs8000_s1024_m1331_room4.0x6.0x3.0_rt200': {
                'train': range(0, 820),
                'valid': range(820, 922),
                'test': range(922, 1024),
                'all': range(0, 1024)
            }
        },
        "num_mes_list": list(reversed([25, 50, 100])),
        "M_mes_norm": 1000,
        "model": 'FSMPAE',
        "learning_rate": 1e-3,
        "epochs": 1400,
        "lr_update": 'step',
        "lr_milestones": [800, 1200],
        "lr_gamma": 0.1,
        "batch_size": 1, #16,
        "loss_weights": {
            "lsd": 1.0
        },
        "save_frequency": 500,
        "activation_function": 'nn.Mish()',
        "data_type_interp": ['atf_mag'],
        "data_type_hyper_en": ['src_position', 'mic_position', 'frequency', 'M_mes'],
        "data_type_hyper_de": ['src_position', 'mic_position', 'frequency'],
        "dim_data_hyper": {
            'src_position': 3,
            'mic_position': 3,
            'frequency': 1,
            'M_mes': 1
        },
        "mid_mean_dim": (1,),
        "aggr_mean": True,
        # Fourier feature mapping
        "data_type_ffm": ['src_position', 'mic_position', 'frequency'],
        "num_ff": {
            'src_position': 16,
            'mic_position': 16,
            'frequency': 8,
            'M_mes': 16
        },
        "ffm_trainable": True,
        # Hypernetwork
        "de_2_skip": True,
        "dim_z": 64,
        "channel_En_0": 16,
        "channel_En_z": 128,
        "channel_De_z": 128,
        "channel_De_-1": 16,
        "channel_hyper": 64,
        "hyper_use_res": True,
        "hlayers_En_0": 2,
        "hlayers_En_z": 0,
        "hyperlinear_en_0": True,
        "hlayers_De_z": 0,
        "hlayers_De_-1": 2,
        "hlayers_hyper": 2,
        "hyperlinear_de_0": True,
        "hyperlinear_de_-1": True,
        "droprate": 0.0,
        "newbob_decay": 0.5,
        "newbob_max_decay": 1e-06,
        "num_gpus": 1,
        "timestamp": "",
    }

config_FSMPAE_10009 = {
        "fs": 8000,#16000,
        "num_freq": 256, #512,#1024,
        "init_delay": False,
        "dataset": ['ir_fs8000_s1024_m1331_room4.0x6.0x3.0_rt200'],
        "src_index": {
            'ir_fs8000_s1024_m1331_room4.0x6.0x3.0_rt200': {
                'train': range(0, 820),
                'valid': range(820, 922),
                'test': range(922, 1024),
                'all': range(0, 1024)
            }
        },
        "idx_mes_pos_mat_path": 'idx_mes_pos_s1024_m1331.npy',
        "num_mes_list": list(reversed([100])),
        "num_mes_test": 100,
        "M_mes_norm": 100,
        "model": 'FSMPAE',
        "learning_rate": 1e-3,
        "epochs": 1400,
        "lr_update": 'step',
        "lr_milestones": [800, 1200],
        "lr_gamma": 0.1,
        "batch_size": 1, #16,
        "loss_weights": {
            "lsd": 1.0
        },
        "save_frequency": 500,
        "activation_function": 'nn.Mish()',
        "data_type_interp": ['atf_mag'],
        "data_type_hyper_en": ['src_position', 'mic_position', 'frequency', 'M_mes'],
        "data_type_hyper_de": ['src_position', 'mic_position', 'frequency'],
        "dim_data_hyper": {
            'src_position': 3,
            'mic_position': 3,
            'frequency': 1,
            'M_mes': 1
        },
        "mid_mean_dim": (1,),
        "aggr_mean": True,
        # Fourier feature mapping
        "data_type_ffm": ['src_position', 'mic_position', 'frequency'],
        "num_ff": {
            'src_position': 16,
            'mic_position': 16,
            'frequency': 8,
            'M_mes': 16
        },
        "ffm_trainable": True,
        # Hypernetwork
        "de_2_skip": True,
        "dim_z": 64,
        "channel_En_0": 16,
        "channel_En_z": 128,
        "channel_De_z": 128,
        "channel_De_-1": 16,
        "channel_hyper": 64,
        "hyper_use_res": True,
        "hlayers_En_0": 2,
        "hlayers_En_z": 0,
        "hyperlinear_en_0": True,
        "hlayers_De_z": 0,
        "hlayers_De_-1": 2,
        "hlayers_hyper": 2,
        "hyperlinear_de_0": True,
        "hyperlinear_de_-1": True,
        "droprate": 0.0,
        "newbob_decay": 0.5,
        "newbob_max_decay": 1e-06,
        "num_gpus": 1,
        "timestamp": "",
    }

config_FSMPAE_10010 = {
        "fs": 8000,#16000,
        "num_freq": 256, #512,#1024,
        "init_delay": False,
        "dataset": ['ir_fs8000_s1024_m1331_room4.0x6.0x3.0_rt200'],
        "src_index": {
            'ir_fs8000_s1024_m1331_room4.0x6.0x3.0_rt200': {
                'train': range(0, 820),
                'valid': range(820, 922),
                'test': range(922, 1024),
                'all': range(0, 1024)
            }
        },
        "idx_mes_pos_mat_path": 'idx_mes_pos_s1024_m1331.npy',
        "num_mes_list": list(reversed([25])),
        "num_mes_test": 25,
        "M_mes_norm": 25,
        "model": 'FSMPAE',
        "learning_rate": 1e-3,
        "epochs": 1400,
        "lr_update": 'step',
        "lr_milestones": [800, 1200],
        "lr_gamma": 0.1,
        "batch_size": 1, #16,
        "loss_weights": {
            "lsd": 1.0
        },
        "save_frequency": 500,
        "activation_function": 'nn.Mish()',
        "data_type_interp": ['atf_mag'],
        "data_type_hyper_en": ['src_position', 'mic_position', 'frequency', 'M_mes'],
        "data_type_hyper_de": ['src_position', 'mic_position', 'frequency'],
        "dim_data_hyper": {
            'src_position': 3,
            'mic_position': 3,
            'frequency': 1,
            'M_mes': 1
        },
        "mid_mean_dim": (1,),
        "aggr_mean": True,
        # Fourier feature mapping
        "data_type_ffm": ['src_position', 'mic_position', 'frequency'],
        "num_ff": {
            'src_position': 16,
            'mic_position': 16,
            'frequency': 8,
            'M_mes': 16
        },
        "ffm_trainable": True,
        # Hypernetwork
        "de_2_skip": True,
        "dim_z": 64,
        "channel_En_0": 16,
        "channel_En_z": 128,
        "channel_De_z": 128,
        "channel_De_-1": 16,
        "channel_hyper": 64,
        "hyper_use_res": True,
        "hlayers_En_0": 2,
        "hlayers_En_z": 0,
        "hyperlinear_en_0": True,
        "hlayers_De_z": 0,
        "hlayers_De_-1": 2,
        "hlayers_hyper": 2,
        "hyperlinear_de_0": True,
        "hyperlinear_de_-1": True,
        "droprate": 0.0,
        "newbob_decay": 0.5,
        "newbob_max_decay": 1e-06,
        "num_gpus": 1,
        "timestamp": "",
    }

config_FSMPAE_10011 = {
        "fs": 8000,#16000,
        "num_freq": 256, #512,#1024,
        "init_delay": False,
        "dataset": ['ir_fs8000_s1024_m1331_room4.0x6.0x3.0_rt200'],
        "src_index": {
            'ir_fs8000_s1024_m1331_room4.0x6.0x3.0_rt200': {
                'train': range(0, 820),
                'valid': range(820, 922),
                'test': range(922, 1024),
                'all': range(0, 1024)
            }
        },
        "idx_mes_pos_mat_path": 'idx_mes_pos_s1024_m1331.npy',
        "num_mes_list": list(reversed([10])),
        "num_mes_test": 10,
        "M_mes_norm": 10,
        "model": 'FSMPAE',
        "learning_rate": 1e-3,
        "epochs": 1400,
        "lr_update": 'step',
        "lr_milestones": [800, 1200],
        "lr_gamma": 0.1,
        "batch_size": 1, #16,
        "loss_weights": {
            "lsd": 1.0
        },
        "save_frequency": 500,
        "activation_function": 'nn.Mish()',
        "data_type_interp": ['atf_mag'],
        "data_type_hyper_en": ['src_position', 'mic_position', 'frequency', 'M_mes'],
        "data_type_hyper_de": ['src_position', 'mic_position', 'frequency'],
        "dim_data_hyper": {
            'src_position': 3,
            'mic_position': 3,
            'frequency': 1,
            'M_mes': 1
        },
        "mid_mean_dim": (1,),
        "aggr_mean": True,
        # Fourier feature mapping
        "data_type_ffm": ['src_position', 'mic_position', 'frequency'],
        "num_ff": {
            'src_position': 16,
            'mic_position': 16,
            'frequency': 8,
            'M_mes': 16
        },
        "ffm_trainable": True,
        # Hypernetwork
        "de_2_skip": True,
        "dim_z": 64,
        "channel_En_0": 16,
        "channel_En_z": 128,
        "channel_De_z": 128,
        "channel_De_-1": 16,
        "channel_hyper": 64,
        "hyper_use_res": True,
        "hlayers_En_0": 2,
        "hlayers_En_z": 0,
        "hyperlinear_en_0": True,
        "hlayers_De_z": 0,
        "hlayers_De_-1": 2,
        "hlayers_hyper": 2,
        "hyperlinear_de_0": True,
        "hyperlinear_de_-1": True,
        "droprate": 0.0,
        "newbob_decay": 0.5,
        "newbob_max_decay": 1e-06,
        "num_gpus": 1,
        "timestamp": "",
    }

config_FSMPAE_10012 = {
        "fs": 8000,#16000,
        "num_freq": 256, #512,#1024,
        "init_delay": False,
        "dataset": ['ir_fs8000_s1024_m1331_room4.0x6.0x3.0_rt200'],
        "src_index": {
            'ir_fs8000_s1024_m1331_room4.0x6.0x3.0_rt200': {
                'train': range(0, 820),
                'valid': range(820, 922),
                'test': range(922, 1024),
                'all': range(0, 1024)
            }
        },
        "idx_mes_pos_mat_path": 'idx_mes_pos_s1024_m1331.npy',
        "num_mes_list": list(reversed([5])),
        "num_mes_test": 5,
        "M_mes_norm": 5,
        "model": 'FSMPAE',
        "learning_rate": 1e-3,
        "epochs": 1400,
        "lr_update": 'step',
        "lr_milestones": [800, 1200],
        "lr_gamma": 0.1,
        "batch_size": 1, #16,
        "loss_weights": {
            "lsd": 1.0
        },
        "save_frequency": 500,
        "activation_function": 'nn.Mish()',
        "data_type_interp": ['atf_mag'],
        "data_type_hyper_en": ['src_position', 'mic_position', 'frequency', 'M_mes'],
        "data_type_hyper_de": ['src_position', 'mic_position', 'frequency'],
        "dim_data_hyper": {
            'src_position': 3,
            'mic_position': 3,
            'frequency': 1,
            'M_mes': 1
        },
        "mid_mean_dim": (1,),
        "aggr_mean": True,
        # Fourier feature mapping
        "data_type_ffm": ['src_position', 'mic_position', 'frequency'],
        "num_ff": {
            'src_position': 16,
            'mic_position': 16,
            'frequency': 8,
            'M_mes': 16
        },
        "ffm_trainable": True,
        # Hypernetwork
        "de_2_skip": True,
        "dim_z": 64,
        "channel_En_0": 16,
        "channel_En_z": 128,
        "channel_De_z": 128,
        "channel_De_-1": 16,
        "channel_hyper": 64,
        "hyper_use_res": True,
        "hlayers_En_0": 2,
        "hlayers_En_z": 0,
        "hyperlinear_en_0": True,
        "hlayers_De_z": 0,
        "hlayers_De_-1": 2,
        "hlayers_hyper": 2,
        "hyperlinear_de_0": True,
        "hyperlinear_de_-1": True,
        "droprate": 0.0,
        "newbob_decay": 0.5,
        "newbob_max_decay": 1e-06,
        "num_gpus": 1,
        "timestamp": "",
    }

config_FSMPAE_10013 = {
        "fs": 8000,#16000,
        "num_freq": 256, #512,#1024,
        "init_delay": False,
        "dataset": ['ir_fs8000_s1024_m9261_room4.0x6.0x3.0_rt200'],
        "src_index": {
            'ir_fs8000_s1024_m9261_room4.0x6.0x3.0_rt200': {
                'train': range(0, 820),
                'valid': range(820, 922),
                'test': range(922, 1024),
                'all': range(0, 1024)
            }
        },
        "idx_mes_pos_mat_path": 'idx_mes_pos_s1024_m9261.npy',
        "num_mes_list": list(reversed([50])),
        "num_mes_test": 50,
        "M_mes_norm": 50,
        "model": 'FSMPAE',
        "learning_rate": 1e-3,
        "epochs": 1400,
        "lr_update": 'step',
        "lr_milestones": [800, 1200],
        "lr_gamma": 0.1,
        "batch_size": 1, #16,
        "loss_weights": {
            "lsd": 1.0
        },
        "save_frequency": 500,
        "activation_function": 'nn.Mish()',
        "data_type_interp": ['atf_mag'],
        "data_type_hyper_en": ['src_position', 'mic_position', 'frequency', 'M_mes'],
        "data_type_hyper_de": ['src_position', 'mic_position', 'frequency'],
        "dim_data_hyper": {
            'src_position': 3,
            'mic_position': 3,
            'frequency': 1,
            'M_mes': 1
        },
        "mid_mean_dim": (1,),
        "aggr_mean": True,
        # Fourier feature mapping
        "data_type_ffm": ['src_position', 'mic_position', 'frequency'],
        "num_ff": {
            'src_position': 16,
            'mic_position': 16,
            'frequency': 8,
            'M_mes': 16
        },
        "ffm_trainable": True,
        # Hypernetwork
        "de_2_skip": True,
        "dim_z": 64,
        "channel_En_0": 16,
        "channel_En_z": 128,
        "channel_De_z": 128,
        "channel_De_-1": 16,
        "channel_hyper": 64,
        "hyper_use_res": True,
        "hlayers_En_0": 2,
        "hlayers_En_z": 0,
        "hyperlinear_en_0": True,
        "hlayers_De_z": 0,
        "hlayers_De_-1": 2,
        "hlayers_hyper": 2,
        "hyperlinear_de_0": True,
        "hyperlinear_de_-1": True,
        "droprate": 0.0,
        "newbob_decay": 0.5,
        "newbob_max_decay": 1e-06,
        "num_gpus": 1,
        "timestamp": "",
    }

config_FSMPAE_10014 = {
        "fs": 8000,#16000,
        "num_freq": 256, #512,#1024,
        "init_delay": False,
        "dataset": ['ir_fs8000_s1024_m9261_room4.0x6.0x3.0_rt200'],
        "src_index": {
            'ir_fs8000_s1024_m9261_room4.0x6.0x3.0_rt200': {
                'train': range(0, 820),
                'valid': range(820, 922),
                'test': range(922, 1024),
                'all': range(0, 1024)
            }
        },
        "idx_mes_pos_mat_path": 'idx_mes_pos_s1024_m9261.npy',
        "num_mes_list": list(reversed([10])),
        "num_mes_test": 10,
        "M_mes_norm": 10,
        "model": 'FSMPAE',
        "learning_rate": 1e-3,
        "epochs": 1400,
        "lr_update": 'step',
        "lr_milestones": [800, 1200],
        "lr_gamma": 0.1,
        "batch_size": 1, #16,
        "loss_weights": {
            "lsd": 1.0
        },
        "save_frequency": 500,
        "activation_function": 'nn.Mish()',
        "data_type_interp": ['atf_mag'],
        "data_type_hyper_en": ['src_position', 'mic_position', 'frequency', 'M_mes'],
        "data_type_hyper_de": ['src_position', 'mic_position', 'frequency'],
        "dim_data_hyper": {
            'src_position': 3,
            'mic_position': 3,
            'frequency': 1,
            'M_mes': 1
        },
        "mid_mean_dim": (1,),
        "aggr_mean": True,
        # Fourier feature mapping
        "data_type_ffm": ['src_position', 'mic_position', 'frequency'],
        "num_ff": {
            'src_position': 16,
            'mic_position': 16,
            'frequency': 8,
            'M_mes': 16
        },
        "ffm_trainable": True,
        # Hypernetwork
        "de_2_skip": True,
        "dim_z": 64,
        "channel_En_0": 16,
        "channel_En_z": 128,
        "channel_De_z": 128,
        "channel_De_-1": 16,
        "channel_hyper": 64,
        "hyper_use_res": True,
        "hlayers_En_0": 2,
        "hlayers_En_z": 0,
        "hyperlinear_en_0": True,
        "hlayers_De_z": 0,
        "hlayers_De_-1": 2,
        "hlayers_hyper": 2,
        "hyperlinear_de_0": True,
        "hyperlinear_de_-1": True,
        "droprate": 0.0,
        "newbob_decay": 0.5,
        "newbob_max_decay": 1e-06,
        "num_gpus": 1,
        "timestamp": "",
    }

config_FSMPAE_10015 = {
        "fs": 8000,#16000,
        "num_freq": 256, #512,#1024,
        "init_delay": False,
        "dataset": ['ir_fs8000_s1024_m9261_room4.0x6.0x3.0_rt200'],
        "src_index": {
            'ir_fs8000_s1024_m9261_room4.0x6.0x3.0_rt200': {
                'train': range(0, 820),
                'valid': range(820, 922),
                'test': range(922, 1024),
                'all': range(0, 1024)
            }
        },
        "idx_mes_pos_mat_path": 'idx_mes_pos_s1024_m9261.npy',
        "num_mes_list": list(reversed([5])),
        "num_mes_test": 5,
        "M_mes_norm": 5,
        "model": 'FSMPAE',
        "learning_rate": 1e-3,
        "epochs": 1400,
        "lr_update": 'step',
        "lr_milestones": [800, 1200],
        "lr_gamma": 0.1,
        "batch_size": 1, #16,
        "loss_weights": {
            "lsd": 1.0
        },
        "save_frequency": 500,
        "activation_function": 'nn.Mish()',
        "data_type_interp": ['atf_mag'],
        "data_type_hyper_en": ['src_position', 'mic_position', 'frequency', 'M_mes'],
        "data_type_hyper_de": ['src_position', 'mic_position', 'frequency'],
        "dim_data_hyper": {
            'src_position': 3,
            'mic_position': 3,
            'frequency': 1,
            'M_mes': 1
        },
        "mid_mean_dim": (1,),
        "aggr_mean": True,
        # Fourier feature mapping
        "data_type_ffm": ['src_position', 'mic_position', 'frequency'],
        "num_ff": {
            'src_position': 16,
            'mic_position': 16,
            'frequency': 8,
            'M_mes': 16
        },
        "ffm_trainable": True,
        # Hypernetwork
        "de_2_skip": True,
        "dim_z": 64,
        "channel_En_0": 16,
        "channel_En_z": 128,
        "channel_De_z": 128,
        "channel_De_-1": 16,
        "channel_hyper": 64,
        "hyper_use_res": True,
        "hlayers_En_0": 2,
        "hlayers_En_z": 0,
        "hyperlinear_en_0": True,
        "hlayers_De_z": 0,
        "hlayers_De_-1": 2,
        "hlayers_hyper": 2,
        "hyperlinear_de_0": True,
        "hyperlinear_de_-1": True,
        "droprate": 0.0,
        "newbob_decay": 0.5,
        "newbob_max_decay": 1e-06,
        "num_gpus": 1,
        "timestamp": "",
    }

config_FSMPAE_10016 = {
        "fs": 8000,#16000,
        "num_freq": 256, #512,#1024,
        "init_delay": False,
        "dataset": ['ir_fs8000_s1024_m9261_room4.0x6.0x3.0_rt200'],
        "src_index": {
            'ir_fs8000_s1024_m9261_room4.0x6.0x3.0_rt200': {
                'train': range(0, 820),
                'valid': range(820, 922),
                'test': range(922, 1024),
                'all': range(0, 1024)
            }
        },
        "idx_mes_pos_mat_path": 'idx_mes_pos_s1024_m9261.npy',
        "num_mes_list": list(reversed([25])),
        "num_mes_test": 25,
        "M_mes_norm": 25,
        "model": 'FSMPAE',
        "learning_rate": 1e-3,
        "epochs": 1400,
        "lr_update": 'step',
        "lr_milestones": [800, 1200],
        "lr_gamma": 0.1,
        "batch_size": 1, #16,
        "loss_weights": {
            "lsd": 1.0
        },
        "save_frequency": 500,
        "activation_function": 'nn.Mish()',
        "data_type_interp": ['atf_mag'],
        "data_type_hyper_en": ['src_position', 'mic_position', 'frequency', 'M_mes'],
        "data_type_hyper_de": ['src_position', 'mic_position', 'frequency'],
        "dim_data_hyper": {
            'src_position': 3,
            'mic_position': 3,
            'frequency': 1,
            'M_mes': 1
        },
        "mid_mean_dim": (1,),
        "aggr_mean": True,
        # Fourier feature mapping
        "data_type_ffm": ['src_position', 'mic_position', 'frequency'],
        "num_ff": {
            'src_position': 16,
            'mic_position': 16,
            'frequency': 8,
            'M_mes': 16
        },
        "ffm_trainable": True,
        # Hypernetwork
        "de_2_skip": True,
        "dim_z": 64,
        "channel_En_0": 16,
        "channel_En_z": 128,
        "channel_De_z": 128,
        "channel_De_-1": 16,
        "channel_hyper": 64,
        "hyper_use_res": True,
        "hlayers_En_0": 2,
        "hlayers_En_z": 0,
        "hyperlinear_en_0": True,
        "hlayers_De_z": 0,
        "hlayers_De_-1": 2,
        "hlayers_hyper": 2,
        "hyperlinear_de_0": True,
        "hyperlinear_de_-1": True,
        "droprate": 0.0,
        "newbob_decay": 0.5,
        "newbob_max_decay": 1e-06,
        "num_gpus": 1,
        "timestamp": "",
    }

config_FSMPAE_10017 = {
        "fs": 8000,#16000,
        "num_freq": 256, #512,#1024,
        "init_delay": False,
        "dataset": ['ir_fs8000_s1024_m9261_room4.0x6.0x3.0_rt200'],
        "src_index": {
            'ir_fs8000_s1024_m9261_room4.0x6.0x3.0_rt200': {
                'train': range(0, 820),
                'valid': range(820, 922),
                'test': range(922, 1024),
                'all': range(0, 1024)
            }
        },
        "idx_mes_pos_mat_path": 'idx_mes_pos_s1024_m9261.npy',
        "num_mes_list": list(reversed([100])),
        "num_mes_test": 100,
        "M_mes_norm": 100,
        "model": 'FSMPAE',
        "learning_rate": 1e-3,
        "epochs": 1400,
        "lr_update": 'step',
        "lr_milestones": [800, 1200],
        "lr_gamma": 0.1,
        "batch_size": 1, #16,
        "loss_weights": {
            "lsd": 1.0
        },
        "save_frequency": 500,
        "activation_function": 'nn.Mish()',
        "data_type_interp": ['atf_mag'],
        "data_type_hyper_en": ['src_position', 'mic_position', 'frequency', 'M_mes'],
        "data_type_hyper_de": ['src_position', 'mic_position', 'frequency'],
        "dim_data_hyper": {
            'src_position': 3,
            'mic_position': 3,
            'frequency': 1,
            'M_mes': 1
        },
        "mid_mean_dim": (1,),
        "aggr_mean": True,
        # Fourier feature mapping
        "data_type_ffm": ['src_position', 'mic_position', 'frequency'],
        "num_ff": {
            'src_position': 16,
            'mic_position': 16,
            'frequency': 8,
            'M_mes': 16
        },
        "ffm_trainable": True,
        # Hypernetwork
        "de_2_skip": True,
        "dim_z": 64,
        "channel_En_0": 16,
        "channel_En_z": 128,
        "channel_De_z": 128,
        "channel_De_-1": 16,
        "channel_hyper": 64,
        "hyper_use_res": True,
        "hlayers_En_0": 2,
        "hlayers_En_z": 0,
        "hyperlinear_en_0": True,
        "hlayers_De_z": 0,
        "hlayers_De_-1": 2,
        "hlayers_hyper": 2,
        "hyperlinear_de_0": True,
        "hyperlinear_de_-1": True,
        "droprate": 0.0,
        "newbob_decay": 0.5,
        "newbob_max_decay": 1e-06,
        "num_gpus": 1,
        "timestamp": "",
    }

config_FSMPAE_10018 = {
        "fs": 2000,#16000,
        "num_freq": 64, #512,#1024,
        "init_delay": False,
        "dataset": ['ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200'],
        "src_index": {
            'ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200': {
                'train': range(0, 820),
                'valid': range(820, 922),
                'test': range(922, 1024),
                'all': range(0, 1024)
            }
        },
        "idx_mes_pos_mat_path": 'idx_mes_pos_s1024_m1331.npy',
        "num_mes_list": list(reversed([50])),
        "num_mes_test": 50,
        "M_mes_norm": 50,
        "model": 'FSMPAE',
        "learning_rate": 1e-3,
        "epochs": 1400,
        "lr_update": 'step',
        "lr_milestones": [800, 1200],
        "lr_gamma": 0.1,
        "batch_size": 1, #16,
        "loss_weights": {
            "lsd": 1.0
        },
        "save_frequency": 500,
        "activation_function": 'nn.Mish()',
        "data_type_interp": ['atf_mag'],
        "data_type_hyper_en": ['src_position', 'mic_position', 'frequency', 'M_mes'],
        "data_type_hyper_de": ['src_position', 'mic_position', 'frequency'],
        "dim_data_hyper": {
            'src_position': 3,
            'mic_position': 3,
            'frequency': 1,
            'M_mes': 1
        },
        "mid_mean_dim": (1,),
        "aggr_mean": True,
        # Fourier feature mapping
        "data_type_ffm": ['src_position', 'mic_position', 'frequency'],
        "num_ff": {
            'src_position': 16,
            'mic_position': 16,
            'frequency': 8,
            'M_mes': 16
        },
        "ffm_trainable": True,
        # Hypernetwork
        "de_2_skip": True,
        "dim_z": 64,
        "channel_En_0": 16,
        "channel_En_z": 128,
        "channel_De_z": 128,
        "channel_De_-1": 16,
        "channel_hyper": 64,
        "hyper_use_res": True,
        "hlayers_En_0": 2,
        "hlayers_En_z": 0,
        "hyperlinear_en_0": True,
        "hlayers_De_z": 0,
        "hlayers_De_-1": 2,
        "hlayers_hyper": 2,
        "hyperlinear_de_0": True,
        "hyperlinear_de_-1": True,
        "droprate": 0.0,
        "newbob_decay": 0.5,
        "newbob_max_decay": 1e-06,
        "num_gpus": 1,
        "timestamp": "",
    }

config_FSMPAE_10019 = {
        "fs": 2000,#16000,
        "num_freq": 64, #512,#1024,
        "init_delay": False,
        "dataset": ['ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200'],
        "src_index": {
            'ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200': {
                'train': range(0, 820),
                'valid': range(820, 922),
                'test': range(922, 1024),
                'all': range(0, 1024)
            }
        },
        "idx_mes_pos_mat_path": 'idx_mes_pos_s1024_m1331.npy',
        "num_mes_list": list(reversed([50])),
        "num_mes_test": 50,
        "M_mes_norm": 50,
        "model": 'FSMPAE',
        "learning_rate": 1e-3,
        "epochs": 1400,
        "lr_update": 'step',
        "lr_milestones": [800, 1200],
        "lr_gamma": 0.1,
        "batch_size": 1, #16,
        "loss_weights": {
            "lsd": 1.0
        },
        "save_frequency": 500,
        "activation_function": 'nn.Mish()',
        "data_type_interp": ['atf_mag'],
        "data_type_hyper_en": ['src_position', 'mic_position', 'frequency'],
        "data_type_hyper_de": ['src_position', 'mic_position', 'frequency'],
        "dim_data_hyper": {
            'src_position': 3,
            'mic_position': 3,
            'frequency': 1
        },
        "mid_mean_dim": (1,),
        "aggr_mean": True,
        # Fourier feature mapping
        "data_type_ffm": ['src_position', 'mic_position', 'frequency'],
        "num_ff": {
            'src_position': 16,
            'mic_position': 16,
            'frequency': 8
        },
        "ffm_trainable": True,
        # Hypernetwork
        "de_2_skip": True,
        "dim_z": 64,
        "channel_En_0": 16,
        "channel_En_z": 128,
        "channel_De_z": 128,
        "channel_De_-1": 16,
        "channel_hyper": 64,
        "hyper_use_res": True,
        "hlayers_En_0": 2,
        "hlayers_En_z": 0,
        "hyperlinear_en_0": True,
        "hlayers_De_z": 0,
        "hlayers_De_-1": 2,
        "hlayers_hyper": 2,
        "hyperlinear_de_0": True,
        "hyperlinear_de_-1": True,
        "droprate": 0.0,
        "newbob_decay": 0.5,
        "newbob_max_decay": 1e-06,
        "num_gpus": 1,
        "timestamp": "",
    }

config_FSMPAE_10020 = {
        "fs": 2000,#16000,
        "num_freq": 64, #512,#1024,
        "init_delay": False,
        "dataset": ['ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200'],
        "src_index": {
            'ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200': {
                'train': range(0, 820),
                'valid': range(820, 922),
                'test': range(922, 1024),
                'all': range(0, 1024)
            }
        },
        "idx_mes_pos_mat_path": 'idx_mes_pos_s1024_m1331.npy',
        "num_mes_list": list(reversed([10])),
        "num_mes_test": 10,
        "M_mes_norm": 10,
        "model": 'FSMPAE',
        "learning_rate": 1e-3,
        "epochs": 1400,
        "lr_update": 'step',
        "lr_milestones": [800, 1200],
        "lr_gamma": 0.1,
        "batch_size": 1, #16,
        "loss_weights": {
            "lsd": 1.0
        },
        "save_frequency": 500,
        "activation_function": 'nn.Mish()',
        "data_type_interp": ['atf_mag'],
        "data_type_hyper_en": ['src_position', 'mic_position', 'frequency'],
        "data_type_hyper_de": ['src_position', 'mic_position', 'frequency'],
        "dim_data_hyper": {
            'src_position': 3,
            'mic_position': 3,
            'frequency': 1
        },
        "mid_mean_dim": (1,),
        "aggr_mean": True,
        # Fourier feature mapping
        "data_type_ffm": ['src_position', 'mic_position', 'frequency'],
        "num_ff": {
            'src_position': 16,
            'mic_position': 16,
            'frequency': 8
        },
        "ffm_trainable": True,
        # Hypernetwork
        "de_2_skip": True,
        "dim_z": 64,
        "channel_En_0": 16,
        "channel_En_z": 128,
        "channel_De_z": 128,
        "channel_De_-1": 16,
        "channel_hyper": 64,
        "hyper_use_res": True,
        "hlayers_En_0": 2,
        "hlayers_En_z": 0,
        "hyperlinear_en_0": True,
        "hlayers_De_z": 0,
        "hlayers_De_-1": 2,
        "hlayers_hyper": 2,
        "hyperlinear_de_0": True,
        "hyperlinear_de_-1": True,
        "droprate": 0.0,
        "newbob_decay": 0.5,
        "newbob_max_decay": 1e-06,
        "num_gpus": 1,
        "timestamp": "",
    }

config_FSMPAE_10021 = {
        "fs": 2000,#16000,
        "num_freq": 64, #512,#1024,
        "init_delay": False,
        "dataset": ['ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200'],
        "src_index": {
            'ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200': {
                'train': range(0, 820),
                'valid': range(820, 922),
                'test': range(922, 1024),
                'all': range(0, 1024)
            }
        },
        "idx_mes_pos_mat_path": 'idx_mes_pos_s1024_m1331.npy',
        "num_mes_list": list(reversed([5])),
        "num_mes_test": 5,
        "M_mes_norm": 5,
        "model": 'FSMPAE',
        "learning_rate": 1e-3,
        "epochs": 1400,
        "lr_update": 'step',
        "lr_milestones": [800, 1200],
        "lr_gamma": 0.1,
        "batch_size": 1, #16,
        "loss_weights": {
            "lsd": 1.0
        },
        "save_frequency": 500,
        "activation_function": 'nn.Mish()',
        "data_type_interp": ['atf_mag'],
        "data_type_hyper_en": ['src_position', 'mic_position', 'frequency'],
        "data_type_hyper_de": ['src_position', 'mic_position', 'frequency'],
        "dim_data_hyper": {
            'src_position': 3,
            'mic_position': 3,
            'frequency': 1
        },
        "mid_mean_dim": (1,),
        "aggr_mean": True,
        # Fourier feature mapping
        "data_type_ffm": ['src_position', 'mic_position', 'frequency'],
        "num_ff": {
            'src_position': 16,
            'mic_position': 16,
            'frequency': 8
        },
        "ffm_trainable": True,
        # Hypernetwork
        "de_2_skip": True,
        "dim_z": 64,
        "channel_En_0": 16,
        "channel_En_z": 128,
        "channel_De_z": 128,
        "channel_De_-1": 16,
        "channel_hyper": 64,
        "hyper_use_res": True,
        "hlayers_En_0": 2,
        "hlayers_En_z": 0,
        "hyperlinear_en_0": True,
        "hlayers_De_z": 0,
        "hlayers_De_-1": 2,
        "hlayers_hyper": 2,
        "hyperlinear_de_0": True,
        "hyperlinear_de_-1": True,
        "droprate": 0.0,
        "newbob_decay": 0.5,
        "newbob_max_decay": 1e-06,
        "num_gpus": 1,
        "timestamp": "",
    }

config_FSMPAE_10022 = {
        "fs": 2000,#16000,
        "num_freq": 64, #512,#1024,
        "init_delay": False,
        "dataset": ['ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200'],
        "src_index": {
            'ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200': {
                'train': range(0, 820),
                'valid': range(820, 922),
                'test': range(922, 1024),
                'all': range(0, 1024)
            }
        },
        "idx_mes_pos_mat_path": 'idx_mes_pos_s1024_m1331.npy',
        "num_mes_list": list(reversed([20])),
        "num_mes_test": 20,
        "M_mes_norm": 20,
        "model": 'FSMPAE',
        "learning_rate": 1e-3,
        "epochs": 1400,
        "lr_update": 'step',
        "lr_milestones": [800, 1200],
        "lr_gamma": 0.1,
        "batch_size": 1, #16,
        "loss_weights": {
            "lsd": 1.0
        },
        "save_frequency": 500,
        "activation_function": 'nn.Mish()',
        "data_type_interp": ['atf_mag'],
        "data_type_hyper_en": ['src_position', 'mic_position', 'frequency'],
        "data_type_hyper_de": ['src_position', 'mic_position', 'frequency'],
        "dim_data_hyper": {
            'src_position': 3,
            'mic_position': 3,
            'frequency': 1
        },
        "mid_mean_dim": (1,),
        "aggr_mean": True,
        # Fourier feature mapping
        "data_type_ffm": ['src_position', 'mic_position', 'frequency'],
        "num_ff": {
            'src_position': 16,
            'mic_position': 16,
            'frequency': 8
        },
        "ffm_trainable": True,
        # Hypernetwork
        "de_2_skip": True,
        "dim_z": 64,
        "channel_En_0": 16,
        "channel_En_z": 128,
        "channel_De_z": 128,
        "channel_De_-1": 16,
        "channel_hyper": 64,
        "hyper_use_res": True,
        "hlayers_En_0": 2,
        "hlayers_En_z": 0,
        "hyperlinear_en_0": True,
        "hlayers_De_z": 0,
        "hlayers_De_-1": 2,
        "hlayers_hyper": 2,
        "hyperlinear_de_0": True,
        "hyperlinear_de_-1": True,
        "droprate": 0.0,
        "newbob_decay": 0.5,
        "newbob_max_decay": 1e-06,
        "num_gpus": 1,
        "timestamp": "",
    }

config_FSMPAE_10023 = {
        "fs": 2000,#16000,
        "num_freq": 64, #512,#1024,
        "init_delay": False,
        "dataset": ['ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200'],
        "src_index": {
            'ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200': {
                'train': range(0, 820),
                'valid': range(820, 922),
                'test': range(922, 1024),
                'all': range(0, 1024)
            }
        },
        "idx_mes_pos_mat_path": 'idx_mes_pos_s1024_m1331.npy',
        "num_mes_list": list(reversed([100])),
        "num_mes_test": 100,
        "M_mes_norm": 100,
        "model": 'FSMPAE',
        "learning_rate": 1e-3,
        "epochs": 1400,
        "lr_update": 'step',
        "lr_milestones": [800, 1200],
        "lr_gamma": 0.1,
        "batch_size": 1, #16,
        "loss_weights": {
            "lsd": 1.0
        },
        "save_frequency": 500,
        "activation_function": 'nn.Mish()',
        "data_type_interp": ['atf_mag'],
        "data_type_hyper_en": ['src_position', 'mic_position', 'frequency'],
        "data_type_hyper_de": ['src_position', 'mic_position', 'frequency'],
        "dim_data_hyper": {
            'src_position': 3,
            'mic_position': 3,
            'frequency': 1
        },
        "mid_mean_dim": (1,),
        "aggr_mean": True,
        # Fourier feature mapping
        "data_type_ffm": ['src_position', 'mic_position', 'frequency'],
        "num_ff": {
            'src_position': 16,
            'mic_position': 16,
            'frequency': 8
        },
        "ffm_trainable": True,
        # Hypernetwork
        "de_2_skip": True,
        "dim_z": 64,
        "channel_En_0": 16,
        "channel_En_z": 128,
        "channel_De_z": 128,
        "channel_De_-1": 16,
        "channel_hyper": 64,
        "hyper_use_res": True,
        "hlayers_En_0": 2,
        "hlayers_En_z": 0,
        "hyperlinear_en_0": True,
        "hlayers_De_z": 0,
        "hlayers_De_-1": 2,
        "hlayers_hyper": 2,
        "hyperlinear_de_0": True,
        "hyperlinear_de_-1": True,
        "droprate": 0.0,
        "newbob_decay": 0.5,
        "newbob_max_decay": 1e-06,
        "num_gpus": 1,
        "timestamp": "",
    }

config_FSMPAE_10024 = {
        "fs": 2000,#16000,
        "num_freq": 64, #512,#1024,
        "init_delay": False,
        "dataset": ['ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200'],
        "src_index": {
            'ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200': {
                'train': range(0, 820),
                'valid': range(820, 922),
                'test': range(922, 1024),
                'all': range(0, 1024)
            }
        },
        "idx_mes_pos_mat_path": 'idx_mes_pos_s1024_m1331.npy',
        "num_mes_list": list(reversed([10])),
        "num_mes_test": 10,
        "M_mes_norm": 10,
        "model": 'FSMPAE',
        "learning_rate": 1e-4,
        "epochs": 1400,
        "lr_update": 'step',
        "lr_milestones": [800, 1200],
        "lr_gamma": 0.1,
        "batch_size": 1, #16,
        "loss_weights": {
            "lsd": 1.0
        },
        "save_frequency": 500,
        "activation_function": 'nn.Mish()',
        "data_type_interp": ['atf_mag'],
        "data_type_hyper_en": ['src_position', 'mic_position', 'frequency'],
        "data_type_hyper_de": ['src_position', 'mic_position', 'frequency'],
        "dim_data_hyper": {
            'src_position': 3,
            'mic_position': 3,
            'frequency': 1
        },
        "mid_mean_dim": (1,),
        "aggr_mean": True,
        # Fourier feature mapping
        "data_type_ffm": ['src_position', 'mic_position', 'frequency'],
        "num_ff": {
            'src_position': 16,
            'mic_position': 16,
            'frequency': 8
        },
        "ffm_trainable": True,
        # Hypernetwork
        "de_2_skip": True,
        "dim_z": 64,
        "channel_En_0": 16,
        "channel_En_z": 128,
        "channel_De_z": 128,
        "channel_De_-1": 16,
        "channel_hyper": 64,
        "hyper_use_res": True,
        "hlayers_En_0": 2,
        "hlayers_En_z": 0,
        "hyperlinear_en_0": True,
        "hlayers_De_z": 0,
        "hlayers_De_-1": 2,
        "hlayers_hyper": 2,
        "hyperlinear_de_0": True,
        "hyperlinear_de_-1": True,
        "droprate": 0.0,
        "newbob_decay": 0.5,
        "newbob_max_decay": 1e-06,
        "num_gpus": 1,
        "timestamp": "",
    }

config_FSMPAE_10025 = {
        "fs": 2000,#16000,
        "num_freq": 64, #512,#1024,
        "init_delay": False,
        "dataset": ['ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200'],
        "src_index": {
            'ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200': {
                'train': range(0, 820),
                'valid': range(820, 922),
                'test': range(922, 1024),
                'all': range(0, 1024)
            }
        },
        "idx_mes_pos_mat_path": 'idx_mes_pos_s1024_m1331.npy',
        "num_mes_list": list(reversed([5])),
        "num_mes_test": 5,
        "M_mes_norm": 5,
        "model": 'FSMPAE',
        "learning_rate": 1e-4,
        "epochs": 1400,
        "lr_update": 'step',
        "lr_milestones": [800, 1200],
        "lr_gamma": 0.1,
        "batch_size": 1, #16,
        "loss_weights": {
            "lsd": 1.0
        },
        "save_frequency": 500,
        "activation_function": 'nn.Mish()',
        "data_type_interp": ['atf_mag'],
        "data_type_hyper_en": ['src_position', 'mic_position', 'frequency'],
        "data_type_hyper_de": ['src_position', 'mic_position', 'frequency'],
        "dim_data_hyper": {
            'src_position': 3,
            'mic_position': 3,
            'frequency': 1
        },
        "mid_mean_dim": (1,),
        "aggr_mean": True,
        # Fourier feature mapping
        "data_type_ffm": ['src_position', 'mic_position', 'frequency'],
        "num_ff": {
            'src_position': 16,
            'mic_position': 16,
            'frequency': 8
        },
        "ffm_trainable": True,
        # Hypernetwork
        "de_2_skip": True,
        "dim_z": 64,
        "channel_En_0": 16,
        "channel_En_z": 128,
        "channel_De_z": 128,
        "channel_De_-1": 16,
        "channel_hyper": 64,
        "hyper_use_res": True,
        "hlayers_En_0": 2,
        "hlayers_En_z": 0,
        "hyperlinear_en_0": True,
        "hlayers_De_z": 0,
        "hlayers_De_-1": 2,
        "hlayers_hyper": 2,
        "hyperlinear_de_0": True,
        "hyperlinear_de_-1": True,
        "droprate": 0.0,
        "newbob_decay": 0.5,
        "newbob_max_decay": 1e-06,
        "num_gpus": 1,
        "timestamp": "",
    }

config_FSMPAE_10026 = {
        "fs": 2000,#16000,
        "num_freq": 64, #512,#1024,
        "init_delay": False,
        "dataset": ['ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200'],
        "src_index": {
            'ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200': {
                'train': range(0, 820),
                'valid': range(820, 922),
                'test': range(922, 1024),
                'all': range(0, 1024)
            }
        },
        "idx_mes_pos_mat_path": 'idx_mes_pos_s1024_m1331.npy',
        "num_mes_list": list(reversed([5, 10, 20, 50, 100])),
        "num_mes_test": 5,
        "M_mes_norm": 100,
        "model": 'FSMPAE',
        "learning_rate": 1e-3,
        "epochs": 1400,
        "lr_update": 'step',
        "lr_milestones": [800, 1200],
        "lr_gamma": 0.1,
        "batch_size": 1, #16,
        "loss_weights": {
            "lsd": 1.0
        },
        "save_frequency": 500,
        "activation_function": 'nn.Mish()',
        "data_type_interp": ['atf_mag'],
        "data_type_hyper_en": ['src_position', 'mic_position', 'frequency', 'M_mes'],
        "data_type_hyper_de": ['src_position', 'mic_position', 'frequency'],
        "dim_data_hyper": {
            'src_position': 3,
            'mic_position': 3,
            'frequency': 1,
            'M_mes': 1
        },
        "mid_mean_dim": (1,),
        "aggr_mean": True,
        # Fourier feature mapping
        "data_type_ffm": ['src_position', 'mic_position', 'frequency'],
        "num_ff": {
            'src_position': 16,
            'mic_position': 16,
            'frequency': 8,
            'M_mes': 16
        },
        "ffm_trainable": True,
        # Hypernetwork
        "de_2_skip": True,
        "dim_z": 64,
        "channel_En_0": 16,
        "channel_En_z": 128,
        "channel_De_z": 128,
        "channel_De_-1": 16,
        "channel_hyper": 64,
        "hyper_use_res": True,
        "hlayers_En_0": 2,
        "hlayers_En_z": 0,
        "hyperlinear_en_0": True,
        "hlayers_De_z": 0,
        "hlayers_De_-1": 2,
        "hlayers_hyper": 2,
        "hyperlinear_de_0": True,
        "hyperlinear_de_-1": True,
        "droprate": 0.0,
        "newbob_decay": 0.5,
        "newbob_max_decay": 1e-06,
        "num_gpus": 1,
        "timestamp": "",
    }

config_FSMPAE_10027 = {
        "fs": 2000,#16000,
        "num_freq": 64, #512,#1024,
        "init_delay": False,
        "dataset": ['ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200'],
        "src_index": {
            'ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200': {
                'train': range(0, 820),
                'valid': range(820, 922),
                'test': range(922, 1024),
                'all': range(0, 1024)
            }
        },
        "idx_mes_pos_mat_path": 'idx_mes_pos_s1024_m1331.npy',
        "num_mes_list": list(reversed([10, 20, 50, 100])),
        "num_mes_test": 10,
        "M_mes_norm": 100,
        "model": 'FSMPAE',
        "learning_rate": 1e-3,
        "epochs": 1400,
        "lr_update": 'step',
        "lr_milestones": [800, 1200],
        "lr_gamma": 0.1,
        "batch_size": 1, #16,
        "loss_weights": {
            "lsd": 1.0
        },
        "save_frequency": 500,
        "activation_function": 'nn.Mish()',
        "data_type_interp": ['atf_mag'],
        "data_type_hyper_en": ['src_position', 'mic_position', 'frequency', 'M_mes'],
        "data_type_hyper_de": ['src_position', 'mic_position', 'frequency'],
        "dim_data_hyper": {
            'src_position': 3,
            'mic_position': 3,
            'frequency': 1,
            'M_mes': 1
        },
        "mid_mean_dim": (1,),
        "aggr_mean": True,
        # Fourier feature mapping
        "data_type_ffm": ['src_position', 'mic_position', 'frequency'],
        "num_ff": {
            'src_position': 16,
            'mic_position': 16,
            'frequency': 8,
            'M_mes': 16
        },
        "ffm_trainable": True,
        # Hypernetwork
        "de_2_skip": True,
        "dim_z": 64,
        "channel_En_0": 16,
        "channel_En_z": 128,
        "channel_De_z": 128,
        "channel_De_-1": 16,
        "channel_hyper": 64,
        "hyper_use_res": True,
        "hlayers_En_0": 2,
        "hlayers_En_z": 0,
        "hyperlinear_en_0": True,
        "hlayers_De_z": 0,
        "hlayers_De_-1": 2,
        "hlayers_hyper": 2,
        "hyperlinear_de_0": True,
        "hyperlinear_de_-1": True,
        "droprate": 0.0,
        "newbob_decay": 0.5,
        "newbob_max_decay": 1e-06,
        "num_gpus": 1,
        "timestamp": "",
    }

config_FSMPAE_10028 = {
        "fs": 2000,#16000,
        "num_freq": 64, #512,#1024,
        "init_delay": False,
        "dataset": ['ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200'],
        "src_index": {
            'ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200': {
                'train': range(0, 820),
                'valid': range(820, 922),
                'test': range(922, 1024),
                'all': range(0, 1024)
            }
        },
        "idx_mes_pos_mat_path": 'idx_mes_pos_s1024_m1331.npy',
        "num_mes_list": list(reversed([20, 50, 100])),
        "num_mes_test": 20,
        "M_mes_norm": 100,
        "model": 'FSMPAE',
        "learning_rate": 1e-3,
        "epochs": 1400,
        "lr_update": 'step',
        "lr_milestones": [800, 1200],
        "lr_gamma": 0.1,
        "batch_size": 1, #16,
        "loss_weights": {
            "lsd": 1.0
        },
        "save_frequency": 500,
        "activation_function": 'nn.Mish()',
        "data_type_interp": ['atf_mag'],
        "data_type_hyper_en": ['src_position', 'mic_position', 'frequency', 'M_mes'],
        "data_type_hyper_de": ['src_position', 'mic_position', 'frequency'],
        "dim_data_hyper": {
            'src_position': 3,
            'mic_position': 3,
            'frequency': 1,
            'M_mes': 1
        },
        "mid_mean_dim": (1,),
        "aggr_mean": True,
        # Fourier feature mapping
        "data_type_ffm": ['src_position', 'mic_position', 'frequency'],
        "num_ff": {
            'src_position': 16,
            'mic_position': 16,
            'frequency': 8,
            'M_mes': 16
        },
        "ffm_trainable": True,
        # Hypernetwork
        "de_2_skip": True,
        "dim_z": 64,
        "channel_En_0": 16,
        "channel_En_z": 128,
        "channel_De_z": 128,
        "channel_De_-1": 16,
        "channel_hyper": 64,
        "hyper_use_res": True,
        "hlayers_En_0": 2,
        "hlayers_En_z": 0,
        "hyperlinear_en_0": True,
        "hlayers_De_z": 0,
        "hlayers_De_-1": 2,
        "hlayers_hyper": 2,
        "hyperlinear_de_0": True,
        "hyperlinear_de_-1": True,
        "droprate": 0.0,
        "newbob_decay": 0.5,
        "newbob_max_decay": 1e-06,
        "num_gpus": 1,
        "timestamp": "",
    }

config_FSMPAE_10029 = {
        "fs": 2000,#16000,
        "num_freq": 64, #512,#1024,
        "init_delay": False,
        "dataset": ['ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200'],
        "src_index": {
            'ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200': {
                'train': range(0, 820),
                'valid': range(820, 922),
                'test': range(922, 1024),
                'all': range(0, 1024)
            }
        },
        "idx_mes_pos_mat_path": 'idx_mes_pos_s1024_m1331.npy',
        "num_mes_list": list(reversed([50, 100])),
        "num_mes_test": 50,
        "M_mes_norm": 100,
        "model": 'FSMPAE',
        "learning_rate": 1e-3,
        "epochs": 1400,
        "lr_update": 'step',
        "lr_milestones": [800, 1200],
        "lr_gamma": 0.1,
        "batch_size": 1, #16,
        "loss_weights": {
            "lsd": 1.0
        },
        "save_frequency": 500,
        "activation_function": 'nn.Mish()',
        "data_type_interp": ['atf_mag'],
        "data_type_hyper_en": ['src_position', 'mic_position', 'frequency', 'M_mes'],
        "data_type_hyper_de": ['src_position', 'mic_position', 'frequency'],
        "dim_data_hyper": {
            'src_position': 3,
            'mic_position': 3,
            'frequency': 1,
            'M_mes': 1
        },
        "mid_mean_dim": (1,),
        "aggr_mean": True,
        # Fourier feature mapping
        "data_type_ffm": ['src_position', 'mic_position', 'frequency'],
        "num_ff": {
            'src_position': 16,
            'mic_position': 16,
            'frequency': 8,
            'M_mes': 16
        },
        "ffm_trainable": True,
        # Hypernetwork
        "de_2_skip": True,
        "dim_z": 64,
        "channel_En_0": 16,
        "channel_En_z": 128,
        "channel_De_z": 128,
        "channel_De_-1": 16,
        "channel_hyper": 64,
        "hyper_use_res": True,
        "hlayers_En_0": 2,
        "hlayers_En_z": 0,
        "hyperlinear_en_0": True,
        "hlayers_De_z": 0,
        "hlayers_De_-1": 2,
        "hlayers_hyper": 2,
        "hyperlinear_de_0": True,
        "hyperlinear_de_-1": True,
        "droprate": 0.0,
        "newbob_decay": 0.5,
        "newbob_max_decay": 1e-06,
        "num_gpus": 1,
        "timestamp": "",
    }


config_FSMPAE_10030 = {
        "fs": 2000,#16000,
        "num_freq": 64, #512,#1024,
        "init_delay": False,
        "dataset": ['ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200'],
        "src_index": {
            'ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200': {
                'train': range(0, 820),
                'valid': range(820, 922),
                'test': range(922, 1024),
                'all': range(0, 1024)
            }
        },
        "idx_mes_pos_mat_path": 'idx_mes_pos_s1024_m1331.npy',
        "num_mes_list": list(reversed([100])),
        "num_mes_test": 100,
        "M_mes_norm": 100,
        "model": 'FSMPAE',
        "learning_rate": 1e-3,
        "epochs": 1400,
        "lr_update": 'step',
        "lr_milestones": [800, 1200],
        "lr_gamma": 0.1,
        "batch_size": 1, #16,
        "loss_weights": {
            "lsd": 1.0
        },
        "save_frequency": 500,
        "activation_function": 'nn.Mish()',
        "data_type_interp": ['atf_mag'],
        "data_type_hyper_en": ['src_position', 'mic_position', 'frequency', 'M_mes'],
        "data_type_hyper_de": ['src_position', 'mic_position', 'frequency'],
        "dim_data_hyper": {
            'src_position': 3,
            'mic_position': 3,
            'frequency': 1,
            'M_mes': 1
        },
        "mid_mean_dim": (1,),
        "aggr_mean": True,
        # Fourier feature mapping
        "data_type_ffm": ['src_position', 'mic_position', 'frequency'],
        "num_ff": {
            'src_position': 16,
            'mic_position': 16,
            'frequency': 8,
            'M_mes': 16
        },
        "ffm_trainable": True,
        # Hypernetwork
        "de_2_skip": True,
        "dim_z": 64,
        "channel_En_0": 16,
        "channel_En_z": 128,
        "channel_De_z": 128,
        "channel_De_-1": 16,
        "channel_hyper": 64,
        "hyper_use_res": True,
        "hlayers_En_0": 2,
        "hlayers_En_z": 0,
        "hyperlinear_en_0": True,
        "hlayers_De_z": 0,
        "hlayers_De_-1": 2,
        "hlayers_hyper": 2,
        "hyperlinear_de_0": True,
        "hyperlinear_de_-1": True,
        "droprate": 0.0,
        "newbob_decay": 0.5,
        "newbob_max_decay": 1e-06,
        "num_gpus": 1,
        "timestamp": "",
    }


config_FSMPAEPI_10001 = {
        "fs": 8000,#16000,
        "num_freq": 256, #512,#1024,
        "init_delay": False,
        "dataset": ['ir_fs8000_s1024_m1331_room4.0x6.0x3.0_rt200'],
        "src_index": {
            'ir_fs8000_s1024_m1331_room4.0x6.0x3.0_rt200': {
                'train': range(0, 820),
                'valid': range(820, 922),
                'test': range(922, 1024),
                'all': range(0, 1024)
            }
        },
        "idx_mes_pos_mat_path": 'idx_mes_pos_s1024_m1331.npy',
        "num_mes_list": list(reversed([50])),
        "num_mes_test": 50,
        "M_mes_norm": 50,
        "model": 'FSMPAEPI',
        "learning_rate": 1e-3,
        "epochs": 1400,
        "lr_update": 'step',
        "lr_milestones": [800, 1200],
        "lr_gamma": 0.1,
        "batch_size": 1, #16,
        "loss_weights": {
            "lsd": 1.0
        },
        "save_frequency": 500,
        "activation_function": 'nn.Mish()',
        "data_type_interp": ['atf_mag'],
        "data_type_hyper_en": ['src_position', 'mic_position', 'frequency', 'M_mes'],
        "data_type_hyper_de": ['src_position', 'mic_position', 'frequency'],
        "dim_data_hyper": {
            'src_position': 3,
            'mic_position': 3,
            'frequency': 1,
            'M_mes': 1
        },
        "mid_mean_dim": (1,),
        "aggr_mean": True,
        # Fourier feature mapping
        "data_type_ffm": ['src_position', 'mic_position', 'frequency'],
        "num_ff": {
            'src_position': 16,
            'mic_position': 16,
            'frequency': 8,
            'M_mes': 16
        },
        "ffm_trainable": True,
        # Hypernetwork
        "de_2_skip": True,
        "dim_z": 64,
        "channel_En_0": 16,
        "channel_En_z": 128,
        "channel_De_z": 128,
        "channel_De_-1": 16,
        "channel_hyper": 64,
        "hyper_use_res": True,
        "hlayers_En_0": 2,
        "hlayers_En_z": 0,
        "hyperlinear_en_0": True,
        "hlayers_De_z": 0,
        "hlayers_De_-1": 2,
        "hlayers_hyper": 2,
        "hyperlinear_de_0": True,
        "hyperlinear_de_-1": True,
        "droprate": 0.0,
        "newbob_decay": 0.5,
        "newbob_max_decay": 1e-06,
        "num_gpus": 1,
        "timestamp": "",
    }

config_FSMPAEPI_10002 = {
        "fs": 8000,#16000,
        "num_freq": 256, #512,#1024,
        "init_delay": False,
        "dataset": ['ir_fs8000_s1024_m1331_room4.0x6.0x3.0_rt200'],
        "src_index": {
            'ir_fs8000_s1024_m1331_room4.0x6.0x3.0_rt200': {
                'train': range(0, 820),
                'valid': range(820, 922),
                'test': range(922, 1024),
                'all': range(0, 1024)
            }
        },
        "idx_mes_pos_mat_path": 'idx_mes_pos_s1024_m1331.npy',
        "num_mes_list": list(reversed([10])),
        "num_mes_test": 10,
        "M_mes_norm": 10,
        "model": 'FSMPAEPI',
        "learning_rate": 1e-3,
        "epochs": 1400,
        "lr_update": 'step',
        "lr_milestones": [800, 1200],
        "lr_gamma": 0.1,
        "batch_size": 1, #16,
        "loss_weights": {
            "lsd": 1.0
        },
        "save_frequency": 500,
        "activation_function": 'nn.Mish()',
        "data_type_interp": ['atf_mag'],
        "data_type_hyper_en": ['src_position', 'mic_position', 'frequency', 'M_mes'],
        "data_type_hyper_de": ['src_position', 'mic_position', 'frequency'],
        "dim_data_hyper": {
            'src_position': 3,
            'mic_position': 3,
            'frequency': 1,
            'M_mes': 1
        },
        "mid_mean_dim": (1,),
        "aggr_mean": True,
        # Fourier feature mapping
        "data_type_ffm": ['src_position', 'mic_position', 'frequency'],
        "num_ff": {
            'src_position': 16,
            'mic_position': 16,
            'frequency': 8,
            'M_mes': 16
        },
        "ffm_trainable": True,
        # Hypernetwork
        "de_2_skip": True,
        "dim_z": 64,
        "channel_En_0": 16,
        "channel_En_z": 128,
        "channel_De_z": 128,
        "channel_De_-1": 16,
        "channel_hyper": 64,
        "hyper_use_res": True,
        "hlayers_En_0": 2,
        "hlayers_En_z": 0,
        "hyperlinear_en_0": True,
        "hlayers_De_z": 0,
        "hlayers_De_-1": 2,
        "hlayers_hyper": 2,
        "hyperlinear_de_0": True,
        "hyperlinear_de_-1": True,
        "droprate": 0.0,
        "newbob_decay": 0.5,
        "newbob_max_decay": 1e-06,
        "num_gpus": 1,
        "timestamp": "",
    }


config_FSMPAEPI_10003 = {
        "fs": 8000,#16000,
        "num_freq": 256, #512,#1024,
        "init_delay": False,
        "dataset": ['ir_fs8000_s1024_m1331_room4.0x6.0x3.0_rt200'],
        "src_index": {
            'ir_fs8000_s1024_m1331_room4.0x6.0x3.0_rt200': {
                'train': range(0, 820),
                'valid': range(820, 922),
                'test': range(922, 1024),
                'all': range(0, 1024)
            }
        },
        "idx_mes_pos_mat_path": 'idx_mes_pos_s1024_m1331.npy',
        "num_mes_list": list(reversed([5])),
        "num_mes_test": 5,
        "M_mes_norm": 5,
        "model": 'FSMPAEPI',
        "learning_rate": 1e-3,
        "epochs": 1400,
        "lr_update": 'step',
        "lr_milestones": [800, 1200],
        "lr_gamma": 0.1,
        "batch_size": 1, #16,
        "loss_weights": {
            "lsd": 1.0
        },
        "save_frequency": 500,
        "activation_function": 'nn.Mish()',
        "data_type_interp": ['atf_mag'],
        "data_type_hyper_en": ['src_position', 'mic_position', 'frequency', 'M_mes'],
        "data_type_hyper_de": ['src_position', 'mic_position', 'frequency'],
        "dim_data_hyper": {
            'src_position': 3,
            'mic_position': 3,
            'frequency': 1,
            'M_mes': 1
        },
        "mid_mean_dim": (1,),
        "aggr_mean": True,
        # Fourier feature mapping
        "data_type_ffm": ['src_position', 'mic_position', 'frequency'],
        "num_ff": {
            'src_position': 16,
            'mic_position': 16,
            'frequency': 8,
            'M_mes': 16
        },
        "ffm_trainable": True,
        # Hypernetwork
        "de_2_skip": True,
        "dim_z": 64,
        "channel_En_0": 16,
        "channel_En_z": 128,
        "channel_De_z": 128,
        "channel_De_-1": 16,
        "channel_hyper": 64,
        "hyper_use_res": True,
        "hlayers_En_0": 2,
        "hlayers_En_z": 0,
        "hyperlinear_en_0": True,
        "hlayers_De_z": 0,
        "hlayers_De_-1": 2,
        "hlayers_hyper": 2,
        "hyperlinear_de_0": True,
        "hyperlinear_de_-1": True,
        "droprate": 0.0,
        "newbob_decay": 0.5,
        "newbob_max_decay": 1e-06,
        "num_gpus": 1,
        "timestamp": "",
    }

config_FSMPAEPI_10004 = {
        "fs": 8000,#16000,
        "num_freq": 256, #512,#1024,
        "init_delay": False,
        "dataset": ['ir_fs8000_s1024_m1331_room4.0x6.0x3.0_rt200'],
        "src_index": {
            'ir_fs8000_s1024_m1331_room4.0x6.0x3.0_rt200': {
                'train': range(0, 820),
                'valid': range(820, 922),
                'test': range(922, 1024),
                'all': range(0, 1024)
            }
        },
        "idx_mes_pos_mat_path": 'idx_mes_pos_s1024_m1331.npy',
        "num_mes_list": list(reversed([25])),
        "num_mes_test": 25,
        "M_mes_norm": 25,
        "model": 'FSMPAEPI',
        "learning_rate": 1e-3,
        "epochs": 1400,
        "lr_update": 'step',
        "lr_milestones": [800, 1200],
        "lr_gamma": 0.1,
        "batch_size": 1, #16,
        "loss_weights": {
            "lsd": 1.0
        },
        "save_frequency": 500,
        "activation_function": 'nn.Mish()',
        "data_type_interp": ['atf_mag'],
        "data_type_hyper_en": ['src_position', 'mic_position', 'frequency', 'M_mes'],
        "data_type_hyper_de": ['src_position', 'mic_position', 'frequency'],
        "dim_data_hyper": {
            'src_position': 3,
            'mic_position': 3,
            'frequency': 1,
            'M_mes': 1
        },
        "mid_mean_dim": (1,),
        "aggr_mean": True,
        # Fourier feature mapping
        "data_type_ffm": ['src_position', 'mic_position', 'frequency'],
        "num_ff": {
            'src_position': 16,
            'mic_position': 16,
            'frequency': 8,
            'M_mes': 16
        },
        "ffm_trainable": True,
        # Hypernetwork
        "de_2_skip": True,
        "dim_z": 64,
        "channel_En_0": 16,
        "channel_En_z": 128,
        "channel_De_z": 128,
        "channel_De_-1": 16,
        "channel_hyper": 64,
        "hyper_use_res": True,
        "hlayers_En_0": 2,
        "hlayers_En_z": 0,
        "hyperlinear_en_0": True,
        "hlayers_De_z": 0,
        "hlayers_De_-1": 2,
        "hlayers_hyper": 2,
        "hyperlinear_de_0": True,
        "hyperlinear_de_-1": True,
        "droprate": 0.0,
        "newbob_decay": 0.5,
        "newbob_max_decay": 1e-06,
        "num_gpus": 1,
        "timestamp": "",
    }

config_FSMPAEPI_10005 = {
        "fs": 8000,#16000,
        "num_freq": 256, #512,#1024,
        "init_delay": False,
        "dataset": ['ir_fs8000_s1024_m1331_room4.0x6.0x3.0_rt200'],
        "src_index": {
            'ir_fs8000_s1024_m1331_room4.0x6.0x3.0_rt200': {
                'train': range(0, 820),
                'valid': range(820, 922),
                'test': range(922, 1024),
                'all': range(0, 1024)
            }
        },
        "idx_mes_pos_mat_path": 'idx_mes_pos_s1024_m1331.npy',
        "num_mes_list": list(reversed([100])),
        "num_mes_test": 100,
        "M_mes_norm": 100,
        "model": 'FSMPAEPI',
        "learning_rate": 1e-3,
        "epochs": 1400,
        "lr_update": 'step',
        "lr_milestones": [800, 1200],
        "lr_gamma": 0.1,
        "batch_size": 1, #16,
        "loss_weights": {
            "lsd": 1.0
        },
        "save_frequency": 500,
        "activation_function": 'nn.Mish()',
        "data_type_interp": ['atf_mag'],
        "data_type_hyper_en": ['src_position', 'mic_position', 'frequency', 'M_mes'],
        "data_type_hyper_de": ['src_position', 'mic_position', 'frequency'],
        "dim_data_hyper": {
            'src_position': 3,
            'mic_position': 3,
            'frequency': 1,
            'M_mes': 1
        },
        "mid_mean_dim": (1,),
        "aggr_mean": True,
        # Fourier feature mapping
        "data_type_ffm": ['src_position', 'mic_position', 'frequency'],
        "num_ff": {
            'src_position': 16,
            'mic_position': 16,
            'frequency': 8,
            'M_mes': 16
        },
        "ffm_trainable": True,
        # Hypernetwork
        "de_2_skip": True,
        "dim_z": 64,
        "channel_En_0": 16,
        "channel_En_z": 128,
        "channel_De_z": 128,
        "channel_De_-1": 16,
        "channel_hyper": 64,
        "hyper_use_res": True,
        "hlayers_En_0": 2,
        "hlayers_En_z": 0,
        "hyperlinear_en_0": True,
        "hlayers_De_z": 0,
        "hlayers_De_-1": 2,
        "hlayers_hyper": 2,
        "hyperlinear_de_0": True,
        "hyperlinear_de_-1": True,
        "droprate": 0.0,
        "newbob_decay": 0.5,
        "newbob_max_decay": 1e-06,
        "num_gpus": 1,
        "timestamp": "",
    }

config_FSMPAEPI_10006 = {
        "fs": 8000,#16000,
        "num_freq": 256, #512,#1024,
        "init_delay": False,
        "dataset": ['ir_fs8000_s1024_m1331_room4.0x6.0x3.0_rt200'],
        "src_index": {
            'ir_fs8000_s1024_m1331_room4.0x6.0x3.0_rt200': {
                'train': range(0, 820),
                'valid': range(820, 922),
                'test': range(922, 1024),
                'all': range(0, 1024)
            }
        },
        "idx_mes_pos_mat_path": 'idx_mes_pos_s1024_m1331.npy',
        "num_mes_list": list(reversed([50])),
        "num_mes_test": 50,
        "M_mes_norm": 50,
        "model": 'FSMPAEPI',
        "learning_rate": 1e-3,
        "epochs": 1400,
        "lr_update": 'step',
        "lr_milestones": [800, 1200],
        "lr_gamma": 0.1,
        "batch_size": 1, #16,
        "loss_weights": {
            "lsd": 1.0
        },
        "save_frequency": 500,
        "activation_function": 'nn.Mish()',
        "data_type_interp": ['atf_mag'],
        "data_type_hyper_en": ['src_position', 'mic_position', 'frequency', 'M_mes'],
        "data_type_hyper_de": ['src_position', 'mic_position', 'frequency'],
        "dim_data_hyper": {
            'src_position': 3,
            'mic_position': 3,
            'frequency': 1,
            'M_mes': 1
        },
        "mid_mean_dim": (1,),
        "aggr_mean": True,
        # Fourier feature mapping
        "data_type_ffm": ['src_position', 'mic_position', 'frequency'],
        "num_ff": {
            'src_position': 16,
            'mic_position': 16,
            'frequency': 8,
            'M_mes': 16
        },
        "ffm_trainable": True,
        # Hypernetwork
        "de_2_skip": True,
        "dim_z": 128, #64,
        "channel_En_0": 32, #16,
        "channel_En_z": 256, #128,
        "channel_De_z": 256, #128,
        "channel_De_-1": 32, #16,
        "channel_hyper": 64,
        "hyper_use_res": True,
        "hlayers_En_0": 2,
        "hlayers_En_z": 0,
        "hyperlinear_en_0": True,
        "hlayers_De_z": 0,
        "hlayers_De_-1": 2,
        "hlayers_hyper": 2,
        "hyperlinear_de_0": True,
        "hyperlinear_de_-1": True,
        "droprate": 0.0,
        "newbob_decay": 0.5,
        "newbob_max_decay": 1e-06,
        "num_gpus": 1,
        "timestamp": "",
    }

config_FSMPAEPI_10007 = {
        "fs": 2000,
        "num_freq": 64, #256, #512,#1024,
        "init_delay": False,
        "dataset": ['ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200'],
        "src_index": {
            'ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200': {
                'train': range(0, 820),
                'valid': range(820, 922),
                'test': range(922, 1024),
                'all': range(0, 1024)
            }
        },
        "idx_mes_pos_mat_path": 'idx_mes_pos_s1024_m1331.npy',
        "num_mes_list": list(reversed([50])),
        "num_mes_test": 50,
        "M_mes_norm": 50,
        "model": 'FSMPAEPI',
        "learning_rate": 1e-3,
        "epochs": 1400,
        "lr_update": 'step',
        "lr_milestones": [800, 1200],
        "lr_gamma": 0.1,
        "batch_size": 1, #16,
        "loss_weights": {
            "lsd": 1.0
        },
        "save_frequency": 500,
        "activation_function": 'nn.Mish()',
        "data_type_interp": ['atf_mag'],
        "data_type_hyper_en": ['src_position', 'mic_position', 'frequency', 'M_mes'],
        "data_type_hyper_de": ['src_position', 'mic_position', 'frequency'],
        "dim_data_hyper": {
            'src_position': 3,
            'mic_position': 3,
            'frequency': 1,
            'M_mes': 1
        },
        "mid_mean_dim": (1,),
        "aggr_mean": True,
        # Fourier feature mapping
        "data_type_ffm": ['src_position', 'mic_position', 'frequency'],
        "num_ff": {
            'src_position': 16,
            'mic_position': 16,
            'frequency': 8,
            'M_mes': 16
        },
        "ffm_trainable": True,
        # Hypernetwork
        "de_2_skip": True,
        "dim_z": 64,
        "channel_En_0": 16,
        "channel_En_z": 128,
        "channel_De_z": 128,
        "channel_De_-1": 16,
        "channel_hyper": 64,
        "hyper_use_res": True,
        "hlayers_En_0": 2,
        "hlayers_En_z": 0,
        "hyperlinear_en_0": True,
        "hlayers_De_z": 0,
        "hlayers_De_-1": 2,
        "hlayers_hyper": 2,
        "hyperlinear_de_0": True,
        "hyperlinear_de_-1": True,
        "droprate": 0.0,
        "newbob_decay": 0.5,
        "newbob_max_decay": 1e-06,
        "num_gpus": 1,
        "timestamp": "",
    }

config_FSMPAEPI_10008 = {
        "fs": 2000,
        "num_freq": 64, #256, #512,#1024,
        "init_delay": False,
        "dataset": ['ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200'],
        "src_index": {
            'ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200': {
                'train': range(0, 820),
                'valid': range(820, 922),
                'test': range(922, 1024),
                'all': range(0, 1024)
            }
        },
        "idx_mes_pos_mat_path": 'idx_mes_pos_s1024_m1331.npy',
        "num_mes_list": list(reversed([50])),
        "num_mes_test": 50,
        "M_mes_norm": 50,
        "model": 'FSMPAEPI',
        "learning_rate": 1e-3,
        "epochs": 1400,
        "lr_update": 'step',
        "lr_milestones": [800, 1200],
        "lr_gamma": 0.1,
        "batch_size": 1, #16,
        "loss_weights": {
            "lsd": 1.0
        },
        "save_frequency": 500,
        "activation_function": 'nn.Mish()',
        "data_type_interp": ['atf_mag'],
        "data_type_hyper_en": ['src_position', 'mic_position', 'frequency'],
        "data_type_hyper_de": ['src_position', 'mic_position', 'frequency'],
        "dim_data_hyper": {
            'src_position': 3,
            'mic_position': 3,
            'frequency': 1
        },
        "mid_mean_dim": (1,),
        "aggr_mean": True,
        # Fourier feature mapping
        "data_type_ffm": ['src_position', 'mic_position', 'frequency'],
        "num_ff": {
            'src_position': 16,
            'mic_position': 16,
            'frequency': 8
        },
        "ffm_trainable": True,
        # Hypernetwork
        "de_2_skip": True,
        "dim_z": 64,
        "channel_En_0": 16,
        "channel_En_z": 128,
        "channel_De_z": 128,
        "channel_De_-1": 16,
        "channel_hyper": 64,
        "hyper_use_res": True,
        "hlayers_En_0": 2,
        "hlayers_En_z": 0,
        "hyperlinear_en_0": True,
        "hlayers_De_z": 0,
        "hlayers_De_-1": 2,
        "hlayers_hyper": 2,
        "hyperlinear_de_0": True,
        "hyperlinear_de_-1": True,
        "droprate": 0.0,
        "newbob_decay": 0.5,
        "newbob_max_decay": 1e-06,
        "num_gpus": 1,
        "timestamp": "",
    }

config_PINN_10001 = {
        "fs": 8000,#16000,
        "c": 343,
        "num_freq": 512, #1024, #256, #
        "init_delay": True,
        "dataset": ['ir_fs8000_s1024_m1331_room4.0x6.0x3.0_rt200'],
        "src_index": {
            'ir_fs8000_s1024_m1331_room4.0x6.0x3.0_rt200': {
                'train': range(0, 820),
                'valid': range(820, 922),
                'test': range(922, 1024),
                'all': range(0, 1024)
            }
        },
        "idx_mes_pos_mat_path": 'idx_mes_pos_s1024_m1331.npy',
        "num_mes_list": list(reversed([50])),
        "num_mes_test": 50,
        "model": 'PINN',
        "learning_rate": 1e-3,
        "epochs": 1000,#1400,
        "lr_update": 'step',
        "lr_milestones": [800, 1200],
        "lr_gamma": 0.1,
        "batch_size": 1, #16,
        "loss_weights": {
            "lsd": 1.0,
            "mse": 1e2,
            "helmholtz": 1e4
        },
        "save_frequency": 500,
        "dim_data_pinn": {
            'src_position': 3,
            'mic_position': 3,
            'frequency': 1
        },
        "data_type_interp": ['atf', 'atf_mag'],
        "data_type_pinn": ['src_position', 'mic_position', 'frequency'],
        # Fourier feature mapping
        "data_type_ffm": ['src_position', 'mic_position', 'frequency'],
        "num_ff": {
            'src_position': 16,
            'mic_position': 16,
            'frequency': 8
        },
        "ffm_trainable": True,
        "droprate": 0.0,
        "newbob_decay": 0.5,
        "newbob_max_decay": 1e-06,
        "num_gpus": 1,
        "timestamp": "",
    }


config_PINN_10002 = {
        "fs": 8000,#16000,
        "c": 343,
        "num_freq": 512, #1024, #256, #
        "init_delay": True,
        "dataset": ['ir_fs8000_s1024_m1331_room4.0x6.0x3.0_rt200'],
        "src_index": {
            'ir_fs8000_s1024_m1331_room4.0x6.0x3.0_rt200': {
                'train': range(0, 820),
                'valid': range(820, 922),
                'test': range(922, 1024),
                'all': range(0, 1024)
            }
        },
        "idx_mes_pos_mat_path": 'idx_mes_pos_s1024_m1331.npy',
        "num_mes_list": list(reversed([25])),
        "num_mes_test": 25,
        "model": 'PINN',
        "learning_rate": 1e-3,
        "epochs": 1000,#1400,
        "lr_update": 'step',
        "lr_milestones": [800, 1200],
        "lr_gamma": 0.1,
        "batch_size": 1, #16,
        "loss_weights": {
            "lsd": 1.0,
            "mse": 1e2,
            "helmholtz": 1e4
        },
        "save_frequency": 500,
        "dim_data_pinn": {
            'src_position': 3,
            'mic_position': 3,
            'frequency': 1
        },
        "data_type_interp": ['atf', 'atf_mag'],
        "data_type_pinn": ['src_position', 'mic_position', 'frequency'],
        # Fourier feature mapping
        "data_type_ffm": ['src_position', 'mic_position', 'frequency'],
        "num_ff": {
            'src_position': 16,
            'mic_position': 16,
            'frequency': 8
        },
        "ffm_trainable": True,
        "droprate": 0.0,
        "newbob_decay": 0.5,
        "newbob_max_decay": 1e-06,
        "num_gpus": 1,
        "timestamp": "",
    }

config_PINN_10003 = {
        "fs": 8000,#16000,
        "c": 343,
        "num_freq": 512, #1024, #256, #
        "init_delay": True,
        "dataset": ['ir_fs8000_s1024_m1331_room4.0x6.0x3.0_rt200'],
        "src_index": {
            'ir_fs8000_s1024_m1331_room4.0x6.0x3.0_rt200': {
                'train': range(0, 820),
                'valid': range(820, 922),
                'test': range(922, 1024),
                'all': range(0, 1024)
            }
        },
        "idx_mes_pos_mat_path": 'idx_mes_pos_s1024_m1331.npy',
        "num_mes_list": list(reversed([10])),
        "num_mes_test": 10,
        "model": 'PINN',
        "learning_rate": 1e-3,
        "epochs": 1000,#1400,
        "lr_update": 'step',
        "lr_milestones": [800, 1200],
        "lr_gamma": 0.1,
        "batch_size": 1, #16,
        "loss_weights": {
            "lsd": 1.0,
            "mse": 1e2,
            "helmholtz": 1e4
        },
        "save_frequency": 500,
        "dim_data_pinn": {
            'src_position': 3,
            'mic_position': 3,
            'frequency': 1
        },
        "data_type_interp": ['atf', 'atf_mag'],
        "data_type_pinn": ['src_position', 'mic_position', 'frequency'],
        # Fourier feature mapping
        "data_type_ffm": ['src_position', 'mic_position', 'frequency'],
        "num_ff": {
            'src_position': 16,
            'mic_position': 16,
            'frequency': 8
        },
        "ffm_trainable": True,
        "droprate": 0.0,
        "newbob_decay": 0.5,
        "newbob_max_decay": 1e-06,
        "num_gpus": 1,
        "timestamp": "",
    }

config_PINN_10004 = {
        "fs": 8000,#16000,
        "c": 343,
        "num_freq": 512, #1024, #256, #
        "init_delay": True,
        "dataset": ['ir_fs8000_s1024_m1331_room4.0x6.0x3.0_rt200'],
        "src_index": {
            'ir_fs8000_s1024_m1331_room4.0x6.0x3.0_rt200': {
                'train': range(0, 820),
                'valid': range(820, 922),
                'test': range(922, 1024),
                'all': range(0, 1024)
            }
        },
        "idx_mes_pos_mat_path": 'idx_mes_pos_s1024_m1331.npy',
        "num_mes_list": list(reversed([5])),
        "num_mes_test": 5,
        "model": 'PINN',
        "learning_rate": 1e-3,
        "epochs": 1000,#1400,
        "lr_update": 'step',
        "lr_milestones": [800, 1200],
        "lr_gamma": 0.1,
        "batch_size": 1, #16,
        "loss_weights": {
            "lsd": 1.0,
            "mse": 1e2,
            "helmholtz": 1e4
        },
        "save_frequency": 500,
        "dim_data_pinn": {
            'src_position': 3,
            'mic_position': 3,
            'frequency': 1
        },
        "data_type_interp": ['atf', 'atf_mag'],
        "data_type_pinn": ['src_position', 'mic_position', 'frequency'],
        # Fourier feature mapping
        "data_type_ffm": ['src_position', 'mic_position', 'frequency'],
        "num_ff": {
            'src_position': 16,
            'mic_position': 16,
            'frequency': 8
        },
        "ffm_trainable": True,
        "droprate": 0.0,
        "newbob_decay": 0.5,
        "newbob_max_decay": 1e-06,
        "num_gpus": 1,
        "timestamp": "",
    }

config_PINN_10005 = {
        "fs": 8000,#16000,
        "c": 343,
        "num_freq": 512, #1024, #256, #
        "init_delay": True,
        "dataset": ['ir_fs8000_s1024_m1331_room4.0x6.0x3.0_rt200'],
        "src_index": {
            'ir_fs8000_s1024_m1331_room4.0x6.0x3.0_rt200': {
                'train': range(0, 820),
                'valid': range(820, 922),
                'test': range(922, 1024),
                'all': range(0, 1024)
            }
        },
        "idx_mes_pos_mat_path": 'idx_mes_pos_s1024_m1331.npy',
        "num_mes_list": list(reversed([100])),
        "num_mes_test": 100,
        "model": 'PINN',
        "learning_rate": 1e-3,
        "epochs": 1000,#1400,
        "lr_update": 'step',
        "lr_milestones": [800, 1200],
        "lr_gamma": 0.1,
        "batch_size": 1, #16,
        "loss_weights": {
            "lsd": 1.0,
            "mse": 1e2,
            "helmholtz": 1e4
        },
        "save_frequency": 500,
        "dim_data_pinn": {
            'src_position': 3,
            'mic_position': 3,
            'frequency': 1
        },
        "data_type_interp": ['atf', 'atf_mag'],
        "data_type_pinn": ['src_position', 'mic_position', 'frequency'],
        # Fourier feature mapping
        "data_type_ffm": ['src_position', 'mic_position', 'frequency'],
        "num_ff": {
            'src_position': 16,
            'mic_position': 16,
            'frequency': 8
        },
        "ffm_trainable": True,
        "droprate": 0.0,
        "newbob_decay": 0.5,
        "newbob_max_decay": 1e-06,
        "num_gpus": 1,
        "timestamp": "",
    }

config_NF_10001 = {
        "fs": 8000,#16000,
        "c": 343,
        "num_freq": 256, #512, #1024, #
        "init_delay": False,
        "dataset": ['ir_fs8000_s1024_m1331_room4.0x6.0x3.0_rt200'],
        "src_index": {
            'ir_fs8000_s1024_m1331_room4.0x6.0x3.0_rt200': {
                'train': range(0, 820),
                'valid': range(820, 922),
                'test': range(922, 1024),
                'all': range(0, 1024)
            }
        },
        "idx_mes_pos_mat_path": 'idx_mes_pos_s1024_m1331.npy',
        "num_mes_list": list(reversed([50])),
        "num_mes_test": 50,
        "model": 'NF',
        "learning_rate": 1e-5,
        "epochs": 1400,
        "epochs_test": 100,
        "lr_update": 'step',
        "lr_milestones": [800, 1200],
        "lr_gamma": 0.1,
        "batch_size": 1, #16,
        "loss_weights": {
            "lsd": 1.0,
            "mse": 1e2,
            "helmholtz": 1e4
        },
        "save_frequency": 500,
        "dim_data_nf": {
            'src_position': 3,
            'mic_position': 3,
            'frequency': 1
        },
        "data_type_interp": ['atf_mag'],
        "data_type_pinn": ['src_position', 'mic_position', 'frequency'],
        # Fourier feature mapping
        "data_type_ffm": ['src_position', 'mic_position', 'frequency'],
        "num_ff": {
            'src_position': 16,
            'mic_position': 16,
            'frequency': 8
        },
        "ffm_trainable": True,
        "droprate": 0.0,
        "newbob_decay": 0.5,
        "newbob_max_decay": 1e-06,
        "num_gpus": 1,
        "timestamp": "",
    }

config_NF_10002 = {
        "fs": 2000,#16000,
        "c": 343,
        "num_freq": 64, #512, #1024, #
        "init_delay": False,
        "dataset": ['ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200'],
        "src_index": {
            'ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200': {
                'train': range(0, 820),
                'valid': range(820, 922),
                'test': range(922, 1024),
                'all': range(0, 1024)
            }
        },
        "idx_mes_pos_mat_path": 'idx_mes_pos_s1024_m1331.npy',
        "num_mes_list": list(reversed([50])),
        "num_mes_test": 50,
        "model": 'NF',
        "learning_rate": 1e-5,
        "epochs": 1400,
        "epochs_test": 10,
        "lr_update": 'step',
        "lr_milestones": [800, 1200],
        "lr_gamma": 0.1,
        "batch_size": 1, #16,
        "loss_weights": {
            "lsd": 1.0
        },
        "save_frequency": 500,
        "dim_data_nf": {
            'src_position': 3,
            'mic_position': 3,
            'frequency': 1
        },
        "data_type_interp": ['atf_mag'],
        "data_type_pinn": ['src_position', 'mic_position', 'frequency'],
        # Fourier feature mapping
        "data_type_ffm": ['src_position', 'mic_position', 'frequency'],
        "num_ff": {
            'src_position': 16,
            'mic_position': 16,
            'frequency': 8
        },
        "ffm_trainable": True,
        "droprate": 0.0,
        "newbob_decay": 0.5,
        "newbob_max_decay": 1e-06,
        "num_gpus": 1,
        "timestamp": "",
    }

config_NF_10003 = {
        "fs": 2000,#16000,
        "c": 343,
        "num_freq": 64, #512, #1024, #
        "init_delay": False,
        "dataset": ['ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200'],
        "src_index": {
            'ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200': {
                'train': range(0, 820),
                'valid': range(820, 922),
                'test': range(922, 1024),
                'all': range(0, 1024)
            }
        },
        "idx_mes_pos_mat_path": 'idx_mes_pos_s1024_m1331.npy',
        "num_mes_list": list(reversed([5])),
        "num_mes_test": 5,
        "model": 'NF',
        "learning_rate": 1e-5,
        "epochs": 1400,
        "epochs_test": 10,
        "lr_update": 'step',
        "lr_milestones": [800, 1200],
        "lr_gamma": 0.1,
        "batch_size": 1, #16,
        "loss_weights": {
            "lsd": 1.0
        },
        "save_frequency": 500,
        "dim_data_nf": {
            'src_position': 3,
            'mic_position': 3,
            'frequency': 1
        },
        "data_type_interp": ['atf_mag'],
        "data_type_pinn": ['src_position', 'mic_position', 'frequency'],
        # Fourier feature mapping
        "data_type_ffm": ['src_position', 'mic_position', 'frequency'],
        "num_ff": {
            'src_position': 16,
            'mic_position': 16,
            'frequency': 8
        },
        "ffm_trainable": True,
        "droprate": 0.0,
        "newbob_decay": 0.5,
        "newbob_max_decay": 1e-06,
        "num_gpus": 1,
        "timestamp": "",
    }

config_NF_10004 = {
        "fs": 2000,#16000,
        "c": 343,
        "num_freq": 64, #512, #1024, #
        "init_delay": False,
        "dataset": ['ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200'],
        "src_index": {
            'ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200': {
                'train': range(0, 820),
                'valid': range(820, 922),
                'test': range(922, 1024),
                'all': range(0, 1024)
            }
        },
        "idx_mes_pos_mat_path": 'idx_mes_pos_s1024_m1331.npy',
        "num_mes_list": list(reversed([10])),
        "num_mes_test": 10,
        "model": 'NF',
        "learning_rate": 1e-5,
        "epochs": 1400,
        "epochs_test": 10,
        "lr_update": 'step',
        "lr_milestones": [800, 1200],
        "lr_gamma": 0.1,
        "batch_size": 1, #16,
        "loss_weights": {
            "lsd": 1.0
        },
        "save_frequency": 500,
        "dim_data_nf": {
            'src_position': 3,
            'mic_position': 3,
            'frequency': 1
        },
        "data_type_interp": ['atf_mag'],
        "data_type_pinn": ['src_position', 'mic_position', 'frequency'],
        # Fourier feature mapping
        "data_type_ffm": ['src_position', 'mic_position', 'frequency'],
        "num_ff": {
            'src_position': 16,
            'mic_position': 16,
            'frequency': 8
        },
        "ffm_trainable": True,
        "droprate": 0.0,
        "newbob_decay": 0.5,
        "newbob_max_decay": 1e-06,
        "num_gpus": 1,
        "timestamp": "",
    }

config_NF_10005 = {
        "fs": 2000,#16000,
        "c": 343,
        "num_freq": 64, #512, #1024, #
        "init_delay": False,
        "dataset": ['ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200'],
        "src_index": {
            'ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200': {
                'train': range(0, 820),
                'valid': range(820, 922),
                'test': range(922, 1024),
                'all': range(0, 1024)
            }
        },
        "idx_mes_pos_mat_path": 'idx_mes_pos_s1024_m1331.npy',
        "num_mes_list": list(reversed([20])),
        "num_mes_test": 20,
        "model": 'NF',
        "learning_rate": 1e-5,
        "epochs": 1400,
        "epochs_test": 10,
        "lr_update": 'step',
        "lr_milestones": [800, 1200],
        "lr_gamma": 0.1,
        "batch_size": 1, #16,
        "loss_weights": {
            "lsd": 1.0
        },
        "save_frequency": 500,
        "dim_data_nf": {
            'src_position': 3,
            'mic_position': 3,
            'frequency': 1
        },
        "data_type_interp": ['atf_mag'],
        "data_type_pinn": ['src_position', 'mic_position', 'frequency'],
        # Fourier feature mapping
        "data_type_ffm": ['src_position', 'mic_position', 'frequency'],
        "num_ff": {
            'src_position': 16,
            'mic_position': 16,
            'frequency': 8
        },
        "ffm_trainable": True,
        "droprate": 0.0,
        "newbob_decay": 0.5,
        "newbob_max_decay": 1e-06,
        "num_gpus": 1,
        "timestamp": "",
    }

config_NF_10006 = {
        "fs": 2000,#16000,
        "c": 343,
        "num_freq": 64, #512, #1024, #
        "init_delay": False,
        "dataset": ['ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200'],
        "src_index": {
            'ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200': {
                'train': range(0, 820),
                'valid': range(820, 922),
                'test': range(922, 1024),
                'all': range(0, 1024)
            }
        },
        "idx_mes_pos_mat_path": 'idx_mes_pos_s1024_m1331.npy',
        "num_mes_list": list(reversed([100])),
        "num_mes_test": 100,
        "model": 'NF',
        "learning_rate": 1e-5,
        "epochs": 1400,
        "epochs_test": 10,
        "lr_update": 'step',
        "lr_milestones": [800, 1200],
        "lr_gamma": 0.1,
        "batch_size": 1, #16,
        "loss_weights": {
            "lsd": 1.0
        },
        "save_frequency": 500,
        "dim_data_nf": {
            'src_position': 3,
            'mic_position': 3,
            'frequency': 1
        },
        "data_type_interp": ['atf_mag'],
        "data_type_pinn": ['src_position', 'mic_position', 'frequency'],
        # Fourier feature mapping
        "data_type_ffm": ['src_position', 'mic_position', 'frequency'],
        "num_ff": {
            'src_position': 16,
            'mic_position': 16,
            'frequency': 8
        },
        "ffm_trainable": True,
        "droprate": 0.0,
        "newbob_decay": 0.5,
        "newbob_max_decay": 1e-06,
        "num_gpus": 1,
        "timestamp": "",
    }

config_KRR_10001 = {
        "fs": 8000,#16000,
        "c": 343,
        "num_freq": 512, #1024, #256, #
        "init_delay": True,
        "dataset": ['ir_fs8000_s1024_m1331_room4.0x6.0x3.0_rt200'],
        "src_index": {
            'ir_fs8000_s1024_m1331_room4.0x6.0x3.0_rt200': {
                'train': range(0, 820),
                'valid': range(820, 922),
                'test': range(922, 1024),
                'all': range(0, 1024)
            }
        },
        "idx_mes_pos_mat_path": 'idx_mes_pos_s1024_m1331.npy',
        "num_mes_list": list(reversed([50])),
        "num_mes_test": 50,
        "model": 'KRR',
        "learning_rate": 1e-3,
        "epochs": 1000,#1400,
        "lr_update": 'step',
        "lr_milestones": [800, 1200],
        "lr_gamma": 0.1,
        "batch_size": 1, #16,
        "loss_weights": {
            "lsd": 1.0,
            "mse": 1e2,
            "helmholtz": 1e4
        },
        "save_frequency": 500,
        "dim_data_pinn": {
            'src_position': 3,
            'mic_position': 3,
            'frequency': 1
        },
        "data_type_interp": ['atf_mag'],
        "data_type_pinn": ['src_position', 'mic_position', 'frequency'],
        # Fourier feature mapping
        "data_type_ffm": ['src_position', 'mic_position', 'frequency'],
        "num_ff": {
            'src_position': 16,
            'mic_position': 16,
            'frequency': 8
        },
        "ffm_trainable": True,
        "droprate": 0.0,
        "newbob_decay": 0.5,
        "newbob_max_decay": 1e-06,
        "num_gpus": 1,
        "timestamp": "",
    }

config_KRR_10002 = {
        "fs": 2000,#16000,
        "c": 343,
        "num_freq": 64, #512, #1024, #256, #
        "init_delay": False,
        "dataset": ['ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200'],
        "src_index": {
            'ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200': {
                'train': range(0, 820),
                'valid': range(820, 922),
                'test': range(922, 1024),
                'all': range(0, 1024)
            }
        },
        "idx_mes_pos_mat_path": 'idx_mes_pos_s1024_m1331.npy',
        "num_mes_list": list(reversed([50])),
        "num_mes_test": 50,
        "model": 'KRR',
        "learning_rate": 1e-3,
        "epochs": 1000,#1400,
        "lr_update": 'step',
        "lr_milestones": [800, 1200],
        "lr_gamma": 0.1,
        "batch_size": 1, #16,
        "loss_weights": {
            "lsd": 1.0,
            "mse": 1e2,
            "helmholtz": 1e4
        },
        "save_frequency": 500,
        "dim_data_pinn": {
            'src_position': 3,
            'mic_position': 3,
            'frequency': 1
        },
        "data_type_interp": ['atf_mag'],
        "data_type_pinn": ['src_position', 'mic_position', 'frequency'],
        # Fourier feature mapping
        "data_type_ffm": ['src_position', 'mic_position', 'frequency'],
        "num_ff": {
            'src_position': 16,
            'mic_position': 16,
            'frequency': 8
        },
        "ffm_trainable": True,
        "droprate": 0.0,
        "newbob_decay": 0.5,
        "newbob_max_decay": 1e-06,
        "num_gpus": 1,
        "timestamp": "",
    }

config_KRR_10003 = {
        "fs": 2000,#16000,
        "c": 343,
        "num_freq": 64, #512, #1024, #256, #
        "init_delay": False,
        "dataset": ['ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200'],
        "src_index": {
            'ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200': {
                'train': range(0, 820),
                'valid': range(820, 922),
                'test': range(922, 1024),
                'all': range(0, 1024)
            }
        },
        "idx_mes_pos_mat_path": 'idx_mes_pos_s1024_m1331.npy',
        "num_mes_list": list(reversed([100])),
        "num_mes_test": 100,
        "model": 'KRR',
        "learning_rate": 1e-3,
        "epochs": 1000,#1400,
        "lr_update": 'step',
        "lr_milestones": [800, 1200],
        "lr_gamma": 0.1,
        "batch_size": 1, #16,
        "loss_weights": {
            "lsd": 1.0,
            "mse": 1e2,
            "helmholtz": 1e4
        },
        "save_frequency": 500,
        "dim_data_pinn": {
            'src_position': 3,
            'mic_position': 3,
            'frequency': 1
        },
        "data_type_interp": ['atf_mag'],
        "data_type_pinn": ['src_position', 'mic_position', 'frequency'],
        # Fourier feature mapping
        "data_type_ffm": ['src_position', 'mic_position', 'frequency'],
        "num_ff": {
            'src_position': 16,
            'mic_position': 16,
            'frequency': 8
        },
        "ffm_trainable": True,
        "droprate": 0.0,
        "newbob_decay": 0.5,
        "newbob_max_decay": 1e-06,
        "num_gpus": 1,
        "timestamp": "",
    }

config_KRR_10004 = {
        "fs": 2000,#16000,
        "c": 343,
        "num_freq": 64, #512, #1024, #256, #
        "init_delay": False,
        "dataset": ['ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200'],
        "src_index": {
            'ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200': {
                'train': range(0, 820),
                'valid': range(820, 922),
                'test': range(922, 1024),
                'all': range(0, 1024)
            }
        },
        "idx_mes_pos_mat_path": 'idx_mes_pos_s1024_m1331.npy',
        "num_mes_list": list(reversed([5])),
        "num_mes_test": 5,
        "model": 'KRR',
        "learning_rate": 1e-3,
        "epochs": 1000,#1400,
        "lr_update": 'step',
        "lr_milestones": [800, 1200],
        "lr_gamma": 0.1,
        "batch_size": 1, #16,
        "loss_weights": {
            "lsd": 1.0,
            "mse": 1e2,
            "helmholtz": 1e4
        },
        "save_frequency": 500,
        "dim_data_pinn": {
            'src_position': 3,
            'mic_position': 3,
            'frequency': 1
        },
        "data_type_interp": ['atf_mag'],
        "data_type_pinn": ['src_position', 'mic_position', 'frequency'],
        # Fourier feature mapping
        "data_type_ffm": ['src_position', 'mic_position', 'frequency'],
        "num_ff": {
            'src_position': 16,
            'mic_position': 16,
            'frequency': 8
        },
        "ffm_trainable": True,
        "droprate": 0.0,
        "newbob_decay": 0.5,
        "newbob_max_decay": 1e-06,
        "num_gpus": 1,
        "timestamp": "",
    }

config_KRR_10005 = {
        "fs": 2000,#16000,
        "c": 343,
        "num_freq": 64, #512, #1024, #256, #
        "init_delay": False,
        "dataset": ['ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200'],
        "src_index": {
            'ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200': {
                'train': range(0, 820),
                'valid': range(820, 922),
                'test': range(922, 1024),
                'all': range(0, 1024)
            }
        },
        "idx_mes_pos_mat_path": 'idx_mes_pos_s1024_m1331.npy',
        "num_mes_list": list(reversed([10])),
        "num_mes_test": 10,
        "model": 'KRR',
        "learning_rate": 1e-3,
        "epochs": 1000,#1400,
        "lr_update": 'step',
        "lr_milestones": [800, 1200],
        "lr_gamma": 0.1,
        "batch_size": 1, #16,
        "loss_weights": {
            "lsd": 1.0,
            "mse": 1e2,
            "helmholtz": 1e4
        },
        "save_frequency": 500,
        "dim_data_pinn": {
            'src_position': 3,
            'mic_position': 3,
            'frequency': 1
        },
        "data_type_interp": ['atf_mag'],
        "data_type_pinn": ['src_position', 'mic_position', 'frequency'],
        # Fourier feature mapping
        "data_type_ffm": ['src_position', 'mic_position', 'frequency'],
        "num_ff": {
            'src_position': 16,
            'mic_position': 16,
            'frequency': 8
        },
        "ffm_trainable": True,
        "droprate": 0.0,
        "newbob_decay": 0.5,
        "newbob_max_decay": 1e-06,
        "num_gpus": 1,
        "timestamp": "",
    }

config_KRR_10006 = {
        "fs": 2000,#16000,
        "c": 343,
        "num_freq": 64, #512, #1024, #256, #
        "init_delay": False,
        "dataset": ['ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200'],
        "src_index": {
            'ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200': {
                'train': range(0, 820),
                'valid': range(820, 922),
                'test': range(922, 1024),
                'all': range(0, 1024)
            }
        },
        "idx_mes_pos_mat_path": 'idx_mes_pos_s1024_m1331.npy',
        "num_mes_list": list(reversed([20])),
        "num_mes_test": 20,
        "model": 'KRR',
        "learning_rate": 1e-3,
        "epochs": 1000,#1400,
        "lr_update": 'step',
        "lr_milestones": [800, 1200],
        "lr_gamma": 0.1,
        "batch_size": 1, #16,
        "loss_weights": {
            "lsd": 1.0,
            "mse": 1e2,
            "helmholtz": 1e4
        },
        "save_frequency": 500,
        "dim_data_pinn": {
            'src_position': 3,
            'mic_position': 3,
            'frequency': 1
        },
        "data_type_interp": ['atf_mag'],
        "data_type_pinn": ['src_position', 'mic_position', 'frequency'],
        # Fourier feature mapping
        "data_type_ffm": ['src_position', 'mic_position', 'frequency'],
        "num_ff": {
            'src_position': 16,
            'mic_position': 16,
            'frequency': 8
        },
        "ffm_trainable": True,
        "droprate": 0.0,
        "newbob_decay": 0.5,
        "newbob_max_decay": 1e-06,
        "num_gpus": 1,
        "timestamp": "",
    }