from .QHFlow import QHFlow
from .Real_QHNet import QHNet as Real_QHNet
from .Real_QHNet_qh9 import QHNet as Real_QHNet_qh9

import logging
logger = logging.getLogger(__name__)

__all__ = ["get_model", "get_default_model_args", "default_model_args_qh9", "default_model_args_md17"]

def get_model(args):
    model_args = {
        "in_node_features": getattr(args, "in_node_features", 1),
        "sh_lmax": getattr(args, "sh_lmax", 4),
        "hidden_size": getattr(args, "hidden_size", 128),
        "bottle_hidden_size": getattr(args, "bottle_hidden_size", 32),
        "num_gnn_layers": getattr(args, "num_gnn_layers", 5),
        "max_radius": getattr(args, "max_radius", 15),
        "num_nodes": getattr(args, "num_nodes", 10),
        "radius_embed_dim": getattr(args, "radius_embed_dim", 16),
        "max_T": getattr(args, "max_T", 15),
        "use_block_S": getattr(args, "use_block_S", True),
        "ham_dim": getattr(args, "ham_dim", 24),
        "ham_hidden": getattr(args, "ham_hidden", 24 * 24 // 2),
        "dataset_type": getattr(args, "dataset_type", "qh9"),
    }
    logging.info(f"model_args: {model_args}")
    model_dict ={
        "Real_QHNet".lower():Real_QHNet,
        "Real_QHNet_qh9".lower():Real_QHNet_qh9,
        "QHFlow".lower():QHFlow,
        "QHFlow_qh9".lower():QHFlow,
    }
    
    if args == None:
        print("args is None, using QHFlow for default")
        return model_dict["QHFlow"](**model_args)

    model_name = args.version.lower()
    model = model_dict.get(model_name, None)
    
    if model is None:
        raise NotImplementedError(f"the version {args.version} is not implemented.")
    else:
        return model(**model_args)

# For debugging
def get_default_model_args(dataset_type):
    if dataset_type == "qh9":
        return default_model_args_qh9
    elif dataset_type == "md17":
        return default_model_args_md17
    else:
        raise ValueError(f"Invalid dataset type: {dataset_type}")

default_model_args_qh9 = {
    "in_node_features": 1,
    "sh_lmax": 4,
    "hidden_size": 128,
    "bottle_hidden_size": 32,
    "num_gnn_layers": 5,
    "max_radius": 15,
    "num_nodes": 10,
    "radius_embed_dim": 16,
    "max_T": 15,
    "use_block_S": True,
    "use_block_H": True,
    "ham_dim": 24,
    "ham_hidden": 24 * 24 // 2,
    "dataset_type": "qh9",
}

default_model_args_md17 = {
    "in_node_features": 1,
    "sh_lmax": 4,
    "hidden_size": 128,
    "bottle_hidden_size": 32,
    "num_gnn_layers": 5,
    "max_radius": 15,
    "num_nodes": 10,
    "radius_embed_dim": 16,
    "max_T": 15,
    "use_block_S": False,
    "use_block_H": True,
    "ham_dim": 24,
    "ham_hidden": 24 * 24 // 2,
    "dataset_type": "md17",
}