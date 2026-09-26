"""Bounded CUDA collective preflight; run with torchrun on reserved devices."""
import os
from datetime import timedelta
import torch
import torch.distributed as dist

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
dist.destroy_process_group()
