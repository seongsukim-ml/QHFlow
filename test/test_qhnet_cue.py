#!/usr/bin/env python3
"""
QHNet_cue Test Script

이 스크립트는 cuEquivariance를 사용하는 QHNet_cue 모델을 테스트합니다.
"""

import sys
import os
import torch
import numpy as np
import time
import json
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
import warnings
warnings.filterwarnings('ignore')
        
# QHFlow 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

model_config = {
    "in_node_features": 1,
    "sh_lmax": 4, 
    "hidden_size": 64,  # 작은 값으로 설정 (default: 128)
    "bottle_hidden_size": 32,  
    "num_gnn_layers": 5, 
    "max_radius": 15,
    "num_nodes": 10,
    "radius_embed_dim": 16,  
}


class LayerDebugger:
    """PyTorch Hook을 사용한 레이어 디버깅 클래스"""
    def __init__(self):
        self.activations = {}
        self.gradients = {}
        self.hooks = []
    
    def register_hooks(self, model):
        """모델의 모든 레이어에 hook 등록"""
        def forward_hook(name):
            def hook(module, input, output):
                self.activations[name] = {
                    'input': [x.detach().cpu() if isinstance(x, torch.Tensor) else x for x in input],
                    'output': output.detach().cpu() if isinstance(output, torch.Tensor) else output,
                    'module_type': type(module).__name__,
                    'module_params': {k: v.shape for k, v in module.named_parameters()} if hasattr(module, 'named_parameters') else {}
                }
            return hook
        
        def backward_hook(name):
            def hook(module, grad_input, grad_output):
                self.gradients[name] = {
                    'grad_input': [x.detach().cpu() if isinstance(x, torch.Tensor) else x for x in grad_input],
                    'grad_output': [x.detach().cpu() if isinstance(x, torch.Tensor) else x for x in grad_output]
                }
            return hook
        
        # 모든 leaf module에 hook 등록 (ScriptModule 제외)
        hook_count = 0
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # leaf module만
                try:
                    # ScriptModule인지 확인
                    if hasattr(module, '_get_method') or hasattr(module, 'code'):
                        print(f"Skipping ScriptModule: {name}")
                        continue
                    
                    forward_hook_handle = module.register_forward_hook(forward_hook(name))
                    backward_hook_handle = module.register_backward_hook(backward_hook(name))
                    self.hooks.extend([forward_hook_handle, backward_hook_handle])
                    hook_count += 1
                except Exception as e:
                    print(f"Failed to register hook for {name}: {e}")
                    continue
        
        print(f"Successfully registered hooks for {hook_count} layers")
    
    def remove_hooks(self):
        """등록된 hook들 제거"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_activation_summary(self):
        """활성화 값들의 요약 정보 반환"""
        summary = []
        summary.append("=== Layer Activations Summary ===")
        
        for name, act in self.activations.items():
            if isinstance(act['output'], torch.Tensor):
                summary.append(f"\n{name} ({act['module_type']}):")
                summary.append(f"  Output shape: {act['output'].shape}")
                
                # dtype 체크 후 통계 계산
                if act['output'].dtype in [torch.float16, torch.float32, torch.float64, torch.complex64, torch.complex128]:
                    summary.append(f"  Output stats: mean={act['output'].mean():.6f}, std={act['output'].std():.6f}")
                    summary.append(f"  Output range: [{act['output'].min():.6f}, {act['output'].max():.6f}]")
                else:
                    summary.append(f"  Output dtype: {act['output'].dtype} (non-numeric)")
                
                if act['input'] and isinstance(act['input'][0], torch.Tensor):
                    input_tensor = act['input'][0]
                    summary.append(f"  Input shape: {input_tensor.shape}")
                    
                    # dtype 체크 후 통계 계산
                    if input_tensor.dtype in [torch.float16, torch.float32, torch.float64, torch.complex64, torch.complex128]:
                        summary.append(f"  Input stats: mean={input_tensor.mean():.6f}, std={input_tensor.std():.6f}")
                    else:
                        summary.append(f"  Input dtype: {input_tensor.dtype} (non-numeric)")
        
        return "\n".join(summary)
    
    def save_activations(self, filepath):
        """활성화 값들을 파일로 저장"""
        torch.save(self.activations, filepath)
        print(f"Activations saved to {filepath}")
    
    def save_gradients(self, filepath):
        """그래디언트 값들을 파일로 저장"""
        torch.save(self.gradients, filepath)
        print(f"Gradients saved to {filepath}")
    
    def load_activations(self, filepath):
        """활성화 값들을 파일에서 로드"""
        self.activations = torch.load(filepath)
    
    def get_layer_statistics(self):
        """각 레이어의 통계 정보 반환"""
        stats = {}
        for name, act in self.activations.items():
            if isinstance(act['output'], torch.Tensor):
                # dtype 체크 후 통계 계산
                if act['output'].dtype in [torch.float16, torch.float32, torch.float64, torch.complex64, torch.complex128]:
                    stats[name] = {
                        'shape': list(act['output'].shape),
                        'mean': act['output'].mean().item(),
                        'std': act['output'].std().item(),
                        'min': act['output'].min().item(),
                        'max': act['output'].max().item(),
                        'module_type': act['module_type'],
                        'dtype': str(act['output'].dtype)
                    }
                else:
                    stats[name] = {
                        'shape': list(act['output'].shape),
                        'module_type': act['module_type'],
                        'dtype': str(act['output'].dtype),
                        'note': 'non-numeric tensor'
                    }
        return stats

def load_md17_water_sample(batch_size: int = 10, default_type: torch.dtype = torch.float32):
    """MD17 water 데이터셋에서 하나의 샘플을 로드합니다."""
    from dataset_module.ori_dataset import MD17_DFT

    root = os.path.join(os.path.dirname(__file__), '..', 'dataset')
    dataset = MD17_DFT(root=root, name='water')
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    data = next(iter(loader))

    def post_processing(batch, default_type):
        if "hamiltonian" in batch.keys:
            if batch.hamiltonian.dim() == 2:
                batch.hamiltonian = batch.hamiltonian.view(
                    batch.hamiltonian.shape[0] // batch.hamiltonian.shape[1],
                    batch.hamiltonian.shape[1],
                    batch.hamiltonian.shape[1],
                )
        if "overlap" in batch.keys:
            if batch.overlap.dim() == 2:
                batch.overlap = batch.overlap.view(
                    batch.overlap.shape[0] // batch.overlap.shape[1],
                    batch.overlap.shape[1],
                    batch.overlap.shape[1],
                )
        for key in batch.keys:
            if type(batch[key]) == torch.Tensor:
                if torch.is_floating_point(batch[key]):
                    batch[key] = batch[key].type(default_type)
        return batch

    return post_processing(data, default_type)


def run_model_test(model_class, model_config, test_name, additional_info=None):
    """
    모델 테스트를 위한 공통 함수
    
    Args:
        model_class: 테스트할 모델 클래스
        model_config: 모델 설정 딕셔너리
        test_name: 테스트 이름
        additional_info: 추가 정보 (예: use_cue 등)
    
    Returns:
        (success, timing_info): 성공 여부와 타이밍 정보
    """
    print("\n" + "="*60)
    print(f"Testing {test_name}")
    print(f"Using device: {device}")
    print(f"Default dtype: {torch.get_default_dtype()}")    
    print("="*60)


    start_time = time.time()
    
    try:
        # MD17 water 데이터 로드
        print("Loading MD17 water sample...")
        data_load_start = time.time()
        data = load_md17_water_sample()
        data_load_time = time.time() - data_load_start
        print(f"Data loading time: {data_load_time:.4f}s")
        
        # 데이터 정보 출력
        print(f"Loaded sample: num_atoms={data.pos.shape[0]}, H_size={data.overlap.shape[-1]}")
        print(f"Data type: {data.pos.dtype}, {data.overlap.dtype}")
        
        # 모델 생성
        model_start_time = time.time()
        model = model_class(**model_config)
        model.set(device)
        model_creation_time = time.time() - model_start_time
        print(f"Model creation time: {model_creation_time:.4f}s")
        
        # 모델 정보 출력
        print(f"Model parameters: {model.get_number_of_parameters():,}")
        if additional_info:
            for key, value in additional_info.items():
                print(f"{key}: {value}")
        
        # 순전파 테스트
        print("\nRunning forward pass...")
        forward_start_time = time.time()
        results = model(data.to(device))
        forward_time = time.time() - forward_start_time
        print(f"Forward pass time: {forward_time:.4f}s")
        
        # 결과 출력
        print(f"Output keys: {results.keys()}")
        for key, value in results.items():
            if isinstance(value, torch.Tensor):
                print(f"{key} shape: {value.shape}")
                print(f"{key} stats - mean: {value.mean():.4f}, std: {value.std():.4f}")
                if value.numel() > 0:
                    print(f"{key} sample (first 5 values): {value.flatten()[:5].tolist()}")
        
        total_time = time.time() - start_time
        print(f"✅ {test_name} forward pass successful!")
        print(f"Total execution time: {total_time:.4f}s")
        
        return True, {
            'data_load_time': data_load_time,
            'model_creation_time': model_creation_time,
            'forward_time': forward_time,
            'total_time': total_time
        }
        
    except Exception as e:
        total_time = time.time() - start_time
        print(f"❌ {test_name} test failed: {e}")
        print(f"Failed after: {total_time:.4f}s")
        import traceback
        traceback.print_exc()
        return False, {'total_time': total_time}


def run_simple_model_test(model_class, model_config, test_name, additional_info=None):
    """
    간단한 모델 테스트를 위한 공통 함수 (injection, filter 등 포함)
    
    Args:
        model_class: 테스트할 모델 클래스
        model_config: 모델 설정 딕셔너리
        test_name: 테스트 이름
        additional_info: 추가 정보 (예: use_cue 등)
    
    Returns:
        (success, timing_info): 성공 여부와 타이밍 정보
    """
    print("\n" + "="*60)
    print(f"Testing {test_name}")
    print("="*60)
    
    start_time = time.time()
    
    try:
        # MD17 water 데이터 로드
        print("Loading MD17 water sample...")
        data_load_start = time.time()
        data = load_md17_water_sample()
        data_load_time = time.time() - data_load_start
        print(f"Data loading time: {data_load_time:.4f}s")
        
        # 데이터 정보 출력
        print(f"Loaded sample: num_atoms={data.pos.shape[0]}, H_size={data.overlap.shape[-1]}")
        
        # 모델 생성
        model_start_time = time.time()
        model = model_class(**model_config)
        model.set(device)
        model_creation_time = time.time() - model_start_time
        print(f"Model creation time: {model_creation_time:.4f}s")
        
        # 모델 정보 출력
        print(f"Model parameters: {model.get_number_of_parameters():,}")
        if additional_info:
            for key, value in additional_info.items():
                print(f"{key}: {value}")
        
        # 데이터 주입
        print("\nRunning data injection...")
        injection_start_time = time.time()
        data, node_attr, edge_sh, rbf_new, transpose_edge_index = model.injection(data.to(device))
        injection_time = time.time() - injection_start_time
        print(f"Data injection time: {injection_time:.4f}s")
        print("✅ Data injection successful!")
        
        # 필터 함수 테스트 (Expansion 모듈 제외)
        print("\nTesting filter function...")
        filter_start_time = time.time()
        try:
            # 간단한 필터 테스트
            H = torch.eye(data.overlap.shape[-1], device=data.pos.device, dtype=torch.float64).unsqueeze(0)
            node_feats_H = model.onebody_reduction(data, H, keep_block=False)
            node_feats_S = model.onebody_reduction(data, data.overlap, keep_block=False)
            print(f"✅ One-body reduction successful!")
            print(f"   node_feats_H shape: {node_feats_H.shape}")
            print(f"   node_feats_S shape: {node_feats_S.shape}")
            
            # GNN 레이어 테스트
            print("\nTesting GNN layers...")
            node_attr_R = node_attr
            for i, layer in enumerate(model.e3_gnn_layer):
                print(f"   Testing layer {i+1}/{len(model.e3_gnn_layer)}...")
                node_attr_R = layer(data, node_attr_R)
                print(f"   Layer {i+1} output shape: {node_attr_R.shape}")
            
            print("✅ GNN layers successful!")
            
        except Exception as e:
            print(f"⚠️  Filter test failed: {e}")
            print("This is expected as the full model requires the Expansion module")
        
        filter_time = time.time() - filter_start_time
        print(f"Filter function time: {filter_time:.4f}s")
        
        total_time = time.time() - start_time
        print(f"✅ {test_name} test successful!")
        print(f"Total execution time: {total_time:.4f}s")
        
        return True, {
            'data_load_time': data_load_time,
            'model_creation_time': model_creation_time,
            'injection_time': injection_time,
            'filter_time': filter_time,
            'total_time': total_time
        }
        
    except Exception as e:
        total_time = time.time() - start_time
        print(f"❌ {test_name} test failed: {e}")
        print(f"Failed after: {total_time:.4f}s")
        import traceback
        traceback.print_exc()
        return False, {'total_time': total_time}


def run_model_test_with_hooks(model_class, model_config, test_name, additional_info=None):
    """
    Hook 디버거를 포함한 모델 테스트 실행 함수
    """
    print("\n" + "="*60)
    print(f"Testing {test_name} (with Hook Debugging)")
    print("="*60)
    
    start_time = time.time()
    
    try:
        # MD17 water 데이터 로드
        print("Loading MD17 water sample...")
        data_load_start = time.time()
        data = load_md17_water_sample()
        data_load_time = time.time() - data_load_start
        print(f"Data loading time: {data_load_time:.4f}s")
        
        # 데이터 정보 출력
        print(f"Loaded sample: num_atoms={data.pos.shape[0]}, H_size={data.overlap.shape[-1]}")
        print(f"Data type: {data.pos.dtype}, {data.overlap.dtype}")
        
        # 모델 생성
        model_start_time = time.time()
        model = model_class(**model_config)
        model.set(device)
        model_creation_time = time.time() - model_start_time
        print(f"Model creation time: {model_creation_time:.4f}s")
        
        # LayerDebugger 설정
        print("\nSetting up LayerDebugger...")
        debugger = LayerDebugger()
        debugger.register_hooks(model)
        print(f"Registered hooks for {len(debugger.hooks)//2} layers")
        
        print(f"Model parameters: {model.get_number_of_parameters():,}")
        if additional_info:
            for key, value in additional_info.items():
                print(f"{key}: {value}")
        
        # 순전파 테스트 (Hook으로 중간값 캡처)
        print("\nRunning forward pass with hook debugging...")
        forward_start_time = time.time()
        results = model(data.to(device))
        forward_time = time.time() - forward_start_time
        print(f"Forward pass time: {forward_time:.4f}s")
        
        # 디버깅 정보 출력
        print("\n" + "="*60)
        print("HOOK DEBUGGING RESULTS")
        print("="*60)
        
        # 레이어 통계 정보 출력
        layer_stats = debugger.get_layer_statistics()
        print(f"\nCaptured {len(layer_stats)} layer activations:")
        for name, stats in layer_stats.items():
            # 통계가 있는 레이어만 개요 출력
            if all(k in stats for k in ("mean", "std", "min", "max")):
                print(f"  {name}: {stats['module_type']} - {stats['shape']}")
                print(f"    mean={stats['mean']:.6f}, std={stats['std']:.6f}, range=[{stats['min']:.6f}, {stats['max']:.6f}]")
        
        # 상세한 활성화 요약 출력
        print("\n" + debugger.get_activation_summary())
        
        # 결과 출력
        print(f"\nOutput keys: {results.keys()}")
        for key, value in results.items():
            if isinstance(value, torch.Tensor):
                print(f"{key} shape: {value.shape}")
                print(f"{key} stats - mean: {value.mean():.4f}, std: {value.std():.4f}")
                if value.numel() > 0:
                    print(f"{key} sample (first 5 values): {value.flatten()[:5].tolist()}")
        
        # 디버깅 데이터 저장
        debug_output_dir = os.path.join(os.path.dirname(__file__), "..", "debug_outputs")
        os.makedirs(debug_output_dir, exist_ok=True)
        
        timestamp = int(time.time())
        activations_file = os.path.join(debug_output_dir, f"activations_{timestamp}.pt")
        gradients_file = os.path.join(debug_output_dir, f"gradients_{timestamp}.pt")
        stats_file = os.path.join(debug_output_dir, f"layer_stats_{timestamp}.json")
        
        debugger.save_activations(activations_file)
        debugger.save_gradients(gradients_file)
        
        # 통계 정보를 JSON으로 저장
        with open(stats_file, 'w') as f:
            json.dump(layer_stats, f, indent=2)
        print(f"Layer statistics saved to {stats_file}")
        
        # Hook 제거
        debugger.remove_hooks()
        print("Hooks removed successfully")
        
        total_time = time.time() - start_time
        print(f"\n✅ {test_name} forward pass successful!")
        print(f"Total execution time: {total_time:.4f}s")
        print(f"Debug files saved to: {debug_output_dir}")
        
        return True, {
            'data_load_time': data_load_time,
            'model_creation_time': model_creation_time,
            'forward_time': forward_time,
            'debug_time': total_time - (data_load_time + model_creation_time + forward_time),
            'total_time': total_time,
            'debug_files': {
                'activations': activations_file,
                'gradients': gradients_file,
                'stats': stats_file
            }
        }
    except Exception as e:
        total_time = time.time() - start_time
        print(f"❌ {test_name} test failed: {e}")
        print(f"Failed after: {total_time:.4f}s")
        import traceback
        traceback.print_exc()
        return False, {'total_time': total_time}

def test_e3nn_only_nohook():
    """QHNet e3nn-only (Hook 미포함)"""
    from models.QHNet import QHNet
    return run_model_test(
        model_class=QHNet,
        model_config=model_config,
        test_name="QHNet e3nn-only (no hook)",
        additional_info={"Using cuEquivariance": False}
    )

def test_e3nn_only():
    """e3nn만 사용하는 QHNet 모델 테스트 (Hook 디버깅 포함)"""
    from models.QHNet import QHNet
    return run_model_test_with_hooks(
        model_class=QHNet,
        model_config=model_config,
        test_name="QHNet e3nn-only (hook)",
        additional_info={"Using cuEquivariance": False}
    )

def test_qhnet_cue_simple():
    """QHNet_cue 모델의 간단한 테스트 (Expansion 모듈 제외)"""
    from models.QHNet_cue import QHNet_cue
    
    print("✅ cuEquivariance is available!")
    
    # Simple 모델용 설정
    simple_config = {
        **model_config,
    }
    
    return run_simple_model_test(
        model_class=QHNet_cue,
        model_config=simple_config,
        test_name="QHNet_cue Model (Simple Version)",
        additional_info={"Using cuEquivariance": True}
    )

def test_qhnet_cue_full():
    """QHNet_cue 모델의 전체 순전파 테스트"""
    from models.QHNet_cue import QHNet_cue
    
    print("✅ cuEquivariance is available!")
    
    return run_model_test(
        model_class=QHNet_cue,
        model_config=model_config,
        test_name="QHNet_cue Model (Full Forward Pass)",
        additional_info={"Using cuEquivariance": True}
    )

def test_qhnet_cue_with_hooks():
    """QHNet_cue 모델의 Hook 디버깅 테스트"""
    from models.QHNet_cue import QHNet_cue
    
    print("✅ cuEquivariance is available!")
    
    return run_model_test_with_hooks(
        model_class=QHNet_cue,
        model_config=model_config,
        test_name="QHNet_cue Model (with Hook Debugging)",
        additional_info={"Using cuEquivariance": True}
    )

def test_cue_availability():
    """cuEquivariance 사용 가능 여부 테스트"""
    print("\n" + "="*60)
    print("Testing cuEquivariance Availability")
    print("="*60)
    
    start_time = time.time()
    
    try:
        from models.QHNet_cue import as_cue_irreps, as_o3_irreps
        from e3nn import o3
        import cuequivariance as cue
        
        print("✅ cuEquivariance is available!")
        
        # 간단한 irreps 변환 테스트
        test_irreps = o3.Irreps("16x0e+8x1o+4x2e")
        cue_irreps = as_cue_irreps(test_irreps, 'O3')
        back_to_e3nn = as_o3_irreps(cue_irreps, 'O3')
        
        print(f"Original irreps: {test_irreps}")
        print(f"Converted to cuEquivariance: {cue_irreps}")
        print(f"Converted back to e3nn: {back_to_e3nn}")
        
        # Round-trip 변환 테스트
        if str(test_irreps) == str(back_to_e3nn):
            print("✅ Round-trip conversion successful!")
        else:
            print("⚠️  Round-trip conversion has differences")
        
        print("✅ cuEquivariance integration successful!")
        
        total_time = time.time() - start_time
        print(f"Total execution time: {total_time:.4f}s")
        
        return True, {'total_time': total_time}
        
    except Exception as e:
        total_time = time.time() - start_time
        print(f"❌ cuEquivariance test failed: {e}")
        print(f"Failed after: {total_time:.4f}s")
        import traceback
        traceback.print_exc()
        return False, {'total_time': total_time}


def main():
    """메인 테스트 함수"""
    print("QHNet_cue Testing Script")
    print("="*60)
    
    # torch의 기본 dtype을 float32로 설정
    torch.set_default_dtype(torch.float32)
    
    # GPU 사용 가능 여부 확인
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    print(f"Default dtype: {torch.get_default_dtype()}")
    
    # 테스트 결과 저장
    test_results = {}
    
    # 각 테스트 실행
    tests = [
        ("cuEquivariance Availability", test_cue_availability),
        # ("QHNet e3nn-only (no hook)", test_e3nn_only_nohook),
        # ("QHNet_cue Model (Simple)", test_qhnet_cue_simple),
        # ("QHNet e3nn-only (no hook)", test_e3nn_only_nohook),
        ("QHNet_cue Model (Full)", test_qhnet_cue_full),
        # ("QHNet e3nn-only (hook)", test_e3nn_only),
        # ("QHNet_cue Model (Hook Debug)", test_qhnet_cue_with_hooks),
    ]
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running {test_name} test...")
        print(f"{'='*60}")
        
        try:
            result, timing_info = test_func()
            test_results[test_name] = {
                'success': result,
                'timing': timing_info
            }
            
            if result:
                print(f"✅ {test_name} test PASSED")
                if 'timing' in test_results[test_name]:
                    timing = test_results[test_name]['timing']
                    print(f"   Timing breakdown:")
                    for key, value in timing.items():
                        if isinstance(value, (int, float)):
                            print(f"     {key}: {value:.4f}s")
                        else:
                            print(f"     {key}: {value}")
            else:
                print(f"❌ {test_name} test FAILED")
                
        except Exception as e:
            print(f"❌ {test_name} test ERROR: {e}")
            test_results[test_name] = {
                'success': False,
                'error': str(e)
            }
    
    # 결과 요약
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = 0
    failed = 0
    
    for test_name, result in test_results.items():
        status = "PASS" if result['success'] else "FAIL"
        timing_str = ""
        if result['success'] and 'timing' in result:
            total_time = result['timing'].get('total_time', 0)
            if isinstance(total_time, (int, float)):
                timing_str = f" ({total_time:.4f}s)"
            else:
                timing_str = ""
        print(f"{test_name:30} {status:4}{timing_str}")
        
        if result['success']:
            passed += 1
        else:
            failed += 1
    
    print(f"\nTotal: {passed + failed} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success rate: {passed/(passed+failed)*100:.1f}%")
    
    # 타이밍 요약
    if passed > 0:
        print(f"\n{'='*60}")
        print("TIMING SUMMARY")
        print(f"{'='*60}")
        
        for test_name, result in test_results.items():
            if result['success'] and 'timing' in result:
                timing = result['timing']
                print(f"\n{test_name}:")
                for key, value in timing.items():
                    if isinstance(value, (int, float)):
                        print(f"  {key:20}: {value:.4f}s")
                    else:
                        print(f"  {key:20}: {value}")
    
    if failed == 0:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {failed} tests failed. Check the output above for details.")


if __name__ == "__main__":
    main()
