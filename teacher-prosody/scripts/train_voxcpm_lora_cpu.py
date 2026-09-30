"""Usage: MALLOC_MMAP_THRESHOLD_=1048576 MALLOC_TRIM_THRESHOLD_=1048576 MALLOC_ARENA_MAX=2 \\
  VOXCPM_TRAIN_SCRIPT=<VoxCPM repo>/scripts/train_voxcpm_finetune.py python scripts/train_voxcpm_lora_cpu.py --config_path configs/voxcpm2_teacher_style_lora.yaml

Run OpenBMB's VoxCPM2 LoRA fine-tune script on CPU: frozen base weights in bf16 (halves RAM, uses AMX),
LoRA weights in fp32, CPU bf16 autocast instead of the script's CUDA autocast."""
import ctypes
import os
import runpy
import sys

import torch
from voxcpm import training
from voxcpm.model import voxcpm2

torch.set_num_threads(4)

training.Accelerator.autocast = lambda self, *a, **k: torch.amp.autocast("cpu", dtype=torch.bfloat16)

# Variable-length batches fragment glibc's heap on CPU (RSS crept from 7 to 14 GB in 40 steps and the run
# was OOM-killed). Hand freed memory back after every optimizer step, and launch with
# MALLOC_MMAP_THRESHOLD_=1048576 MALLOC_TRIM_THRESHOLD_=1048576 MALLOC_ARENA_MAX=2: RSS then stays ~8 GB
# (peaks ~11.5 GB) at about twice the step time.
_libc = ctypes.CDLL("libc.so.6")
_step = training.Accelerator.step


def _step_and_trim(self, *a, **k):
    out = _step(self, *a, **k)
    _libc.malloc_trim(0)
    return out


training.Accelerator.step = _step_and_trim


_orig = voxcpm2.VoxCPM2Model.from_local.__func__


def _from_local(cls, *a, **k):
    m = _orig(cls, *a, **k)
    n = 0
    for name, p in m.named_parameters():
        if not p.requires_grad and "audio_vae" not in name:
            p.data = p.data.to(torch.bfloat16)
            n += p.numel()
    print(f"[cpu] frozen base weights -> bf16: {n/1e9:.2f}B params", file=sys.stderr)
    return m


voxcpm2.VoxCPM2Model.from_local = classmethod(_from_local)
sys.argv = ["train_voxcpm_finetune.py"] + sys.argv[1:]
runpy.run_path(os.environ.get("VOXCPM_TRAIN_SCRIPT", "VoxCPM/scripts/train_voxcpm_finetune.py"), run_name="__main__")
