"""Usage: VOXCPM_TRAIN_SCRIPT=<VoxCPM repo>/scripts/train_voxcpm_finetune.py python scripts/train_voxcpm_lora_cpu.py --config_path configs/voxcpm2_teacher_style_lora.yaml

Run OpenBMB's VoxCPM2 LoRA fine-tune script on CPU: frozen base weights in bf16 (halves RAM, uses AMX),
LoRA weights in fp32, CPU bf16 autocast instead of the script's CUDA autocast."""
import os
import runpy
import sys

import torch
from voxcpm import training
from voxcpm.model import voxcpm2

torch.set_num_threads(4)

training.Accelerator.autocast = lambda self, *a, **k: torch.amp.autocast("cpu", dtype=torch.bfloat16)


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
