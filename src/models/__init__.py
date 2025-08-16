from .QHFlow import QHFlow
from .QHFlow_qh9 import QHFlow as QHFlow_qh9
from .Real_QHNet import QHNet as Real_QHNet
from .Real_QHNet_qh9 import QHNet as Real_QHNet_qh9

import logging
logger = logging.getLogger(__name__)

__all__ = ["get_model"]

# version: wo bias and with bias model are used to load the model for the paper reproduction
# QHNet is the clean version, and we use QHNet to build the benchmark performance

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
    }
    logging.info(f"model_args: {model_args}")
    model_dict ={
        "Real_QHNet".lower():Real_QHNet,
        "Real_QHNet_qh9".lower():Real_QHNet_qh9,
        "QHFlow".lower():QHFlow,
        "QHFlow_qh9".lower():QHFlow_qh9
    }
    
    model_name = args.version.lower()
    model = model_dict.get(model_name, None)
    
    if model is None:
        raise NotImplementedError(f"the version {args.version} is not implemented.")
    else:
        return model(**model_args)


from pl_module.base_module import LitModel
from pl_module.base_module_inf import LitModel as LitModel_inf
from pl_module.flow_module import LitModel_flow
from pl_module.flow_module_inf import LitModel_flow as LitModel_flow_inf
from pl_module.flow_module_inf_scf import LitModel_flow as LitModel_flow_inf_scf
from pl_module.flow_module_qh9 import LitModel_flow as LitModel_flow_qh9
from pl_module.flow_module_qh9_inf import LitModel_flow as LitModel_flow_qh9_inf
from pl_module.flow_module_qh9_finetune import  LitModel_flow as LitModel_flow_qh9_finetune
from pl_module.flow_module_finetune import LitModel_flow as LitModel_flow_finetune


def get_pl_model(conf):
    version = conf.model.version.lower()
    pl_type = conf.get("pl_type", None)
    cur_mode = conf.get("mode", None)
    
    # Define inference modes
    inference_modes = ["inference", "inf", "predict", "predict-mul"]
    is_inference_mode = cur_mode in inference_modes
    
    # If pl_type is specified, use it directly
    if pl_type is not None:
        pl_type = pl_type.lower()
        return _get_model_by_pl_type(pl_type, is_inference_mode)
    
    # Fallback to version-based selection
    return _get_model_by_version(version, is_inference_mode)


def _get_model_by_pl_type(pl_type, is_inference_mode):
    """Get model class based on pl_type and inference mode."""
    
    # Define model mappings
    model_mappings = {
        "flow_inf_scf": LitModel_flow_inf_scf,
        "flow_inf": LitModel_flow_inf,
        "base": LitModel_inf if is_inference_mode else LitModel,
        "flow": LitModel_flow_inf if is_inference_mode else LitModel_flow,
        "flow_finetune": LitModel_flow_inf if is_inference_mode else LitModel_flow_finetune,
        "flow_qh9": LitModel_flow_qh9_inf if is_inference_mode else LitModel_flow_qh9,
        "flow_qh9_finetune": LitModel_flow_qh9_inf if is_inference_mode else LitModel_flow_qh9_finetune,
    }
    
    if pl_type in model_mappings:
        return model_mappings[pl_type]
    
    raise NotImplementedError(f"The pl_type '{pl_type}' is not implemented.")


def _get_model_by_version(version, is_inference_mode):
    """Get model class based on version and inference mode."""
    
    if "flow" in version:
        return LitModel_flow_inf if is_inference_mode else LitModel_flow
    
    return LitModel_inf if is_inference_mode else LitModel
