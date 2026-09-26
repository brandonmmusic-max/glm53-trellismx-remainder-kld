#!/usr/bin/env python3
"""Inside the production image: write the latents with the real GLM_NEXT NVFP4 cache writer
and dequantize with b12x's own test reference helper."""
import sys

import torch

sys.path.insert(0, "/opt/glm53-flash/b12x")
from b12x.attention._shared.mla.kv_cache import concat_and_cache_glm_next_mla  # noqa: E402
from tests._reference.helpers import dequantize_nvfp4_mla_nope  # noqa: E402

kv_c = torch.load("/work/results/testC/mirror_inputs.pt")["kv_c"].cuda().contiguous()
n, page = kv_c.shape[0], 64
cache = torch.zeros((n + page - 1) // page, page, 304, dtype=torch.uint8, device="cuda")
slots = torch.arange(n, dtype=torch.int64, device="cuda")
concat_and_cache_glm_next_mla(kv_c, cache, slots)
torch.cuda.synchronize()
records = cache.reshape(-1, 304)[:n].cpu()
deq, _ = dequantize_nvfp4_mla_nope(records, nope_bytes=256, group_scales_offset=256, group_scales_end=288,
                                   latent_scale_offset=292)
torch.save({"deq": deq.float().cpu()}, "/work/results/testC/mirror_kernel_deq.pt")
print("kernel records", tuple(records.shape), "deq", tuple(deq.shape))
