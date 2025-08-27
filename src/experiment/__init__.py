# For debugging
test_conf_qh9 = {
    "model": {
        "model_name": "QHFlow",
        "version": "QHFlow",
        "hidden_size": 128,
        "bottle_hidden_size": 32,
        "use_block_S": True,
        "use_block_H": False,
    },
    "dataset": {
        "dataset_full_name": "QH9Stable-random",
        "dataset_name": "QH9Stable",
        "version": "100k",
        "split": "random",
        "batch_size": 4,
        "train_batch_size": 4,
        "valid_batch_size": 4,
        "test_batch_size": 4,
        "learning_rate": 5e-4,
        "validation_interval": 1,
        "use_gradient_clipping": True,
        "clip_norm": 5.0,
        "pin_memory": False,
        "num_workers": 8,
        "warmup_steps": 1000,
        "total_steps": 260000,
        "lr_end": 1e-7,
        "train_batch_interval": 100,
        "validation_batch_interval": 1000,
    },
    "ckpt_dir": "checkpoints",
    "split_seed": 42,
    "check_val_every_n_epoch": 1,
    "optimizer": "AdamW",
    "decay_factor": 0.5,
    "decay_patience": 5,
    "ema_start_epoch": -1,
    "warmup_step": 1000,
    "num_training_steps": 260000,
    "end_lr": 1e-7,
    "scheduler_power": 1.0,
    "use_init_hamiltonian": True,
    "use_init_hamiltonian_residue": True,
    "test_mode": "test",
    "qh9": True,
    "data_type": "float32",
    "loss_weights": {"hamiltonian": 10.0},
    "use_mse_and_mae": True,
    "prefix": "_debug",
    "mode": "predict",
    "wandb": {"mode": "offline "},
    "flow": {
        "batch_mul": 1,
        "use_t_scale": False,
        "init_gauss": True,
        "init_gauss_center": False,
        "use_mse_and_mae": True,
        "use_res_target": True,
    }
}

test_conf_md17 = {
    "model": {
        "model_name": "QHFlow",
        "version": "QHFlow",
        "hidden_size": 128,
        "bottle_hidden_size": 32,
        "use_block_S": False,
        "use_block_H": True,
    },
    "dataset": {
        "dataset_name": "water",
        "num_train": 500,
        "num_valid": 500,
        "density_loss": 0.01,
        "max_radius": 15.0,
        "batch_size": 16,
        "train_batch_size": 16,
        "valid_batch_size": 16,
        "test_batch_size": 16,
    },
    "ckpt_dir": "checkpoints",
    "split_seed": 42,
    "check_val_every_n_epoch": 10,
    "optimizer": "AdamW",
    "decay_factor": 0.5,
    "decay_patience": 5,
    "ema_start_epoch": 40,
    "warmup_step": 1000,
    "num_training_steps": 200000,
    "end_lr": 1e-9,
    "scheduler_power": 1,
    "data_type": "float64",
    "loss_weights": {"hamiltonian": 10.0},
    "use_init_hamiltonian_residue": True,
    "use_mse_and_mae": True,
    "prefix": "_debug",
    "wandb": {"mode": "offline"},
}



class ConfigNamespace:
    """딕셔너리를 속성 접근과 딕셔너리 접근 모두 지원하는 namespace로 변환하는 클래스"""
    
    def __init__(self, config_dict):
        self._config = config_dict
        # 딕셔너리의 모든 키를 속성으로 설정
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigNamespace(value))
            else:
                setattr(self, key, value)
    
    def __getitem__(self, key):
        """딕셔너리 스타일 접근 지원"""
        return self._config[key]
    
    def __contains__(self, key):
        """in 연산자 지원"""
        return key in self._config
    
    def get(self, key, default=None):
        """딕셔너리의 get 메서드 지원"""
        return self._config.get(key, default)
    
    def items(self):
        """딕셔너리의 items 메서드 지원"""
        return self._config.items()
    
    def keys(self):
        """딕셔너리의 keys 메서드 지원"""
        return self._config.keys()
    
    def values(self):
        """딕셔너리의 values 메서드 지원"""
        return self._config.values()
    
    def __repr__(self):
        return f"ConfigNamespace({self._config})"
    
    def to_dict(self):
        """원본 딕셔너리 반환"""
        return self._config

def get_test_conf_qh9():
    return ConfigNamespace(test_conf_qh9)

def get_test_conf_md17():
    return ConfigNamespace(test_conf_md17)