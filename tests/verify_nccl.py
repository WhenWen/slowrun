"""Bounded CUDA collective preflight; run with torchrun on reserved devices."""
import os
import argparse
import json
import time
from datetime import timedelta
import torch
import torch.distributed as dist

parser = argparse.ArgumentParser()
parser.add_argument('--benchmark-elements', type=int, default=0,
                    help='Optional FP32 buffer size for collective timing')
parser.add_argument('--iterations', type=int, default=5)
args = parser.parse_args()
if args.benchmark_elements < 0 or args.iterations < 1:
    parser.error('Invalid benchmark size or iteration count')

rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(rank)
print(f'RANK {rank} INITIALIZING', flush=True)
dist.init_process_group('nccl', device_id=torch.device('cuda', rank), timeout=timedelta(seconds=30))
print(f'RANK {rank} INITIALIZED', flush=True)
value = torch.tensor(float(rank + 1), device='cuda')
dist.all_reduce(value)
expected = dist.get_world_size() * (dist.get_world_size() + 1) / 2
assert value.item() == expected, (value.item(), expected)
dist.barrier()
print(f'RANK {rank} COLLECTIVE_PASS sum={value.item()}', flush=True)
if args.benchmark_elements:
    world_size = dist.get_world_size()
    assert args.benchmark_elements % world_size == 0
    buffer = torch.full((args.benchmark_elements,), float(rank + 1), device='cuda')
    shard = torch.empty(args.benchmark_elements // world_size, device='cuda')
    gathered = torch.empty_like(buffer)
    operations = {
        'all_reduce_avg': lambda: dist.all_reduce(buffer, op=dist.ReduceOp.AVG),
        'reduce_scatter_avg': lambda: dist.reduce_scatter_tensor(shard, buffer, op=dist.ReduceOp.AVG),
        'all_gather': lambda: dist.all_gather_into_tensor(gathered, shard),
    }
    timings = {}
    for name, operation in operations.items():
        operation()  # warmup; AVG remains fixed at (world_size + 1) / 2
        dist.barrier()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(args.iterations):
            operation()
        torch.cuda.synchronize()
        elapsed = torch.tensor((time.perf_counter() - start) / args.iterations,
                               dtype=torch.float64, device='cuda')
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        timings[name + '_ms'] = elapsed.item() * 1000
    expected_mean = (world_size + 1) / 2
    for tensor in (buffer, shard, gathered):
        assert (tensor == expected_mean).all().item()
    if rank == 0:
        print(json.dumps({'status': 'BENCHMARK_PASS', 'world_size': world_size,
                          'buffer_mib': buffer.numel() * buffer.element_size() / 2**20,
                          'iterations': args.iterations,
                          'transport_environment': {key: os.environ.get(key) for key in
                              ('NCCL_P2P_DISABLE', 'NCCL_SHM_DISABLE', 'NCCL_NET')},
                          **timings}), flush=True)
dist.destroy_process_group()
