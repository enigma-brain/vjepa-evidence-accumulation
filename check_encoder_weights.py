"""Are two checkpoints' V-JEPA 2 encoders identical? (CPU, downloads only the safetensors.)

Meta's task probes (SSv2, Diving48) are trained on a frozen encoder, so their HF checkpoints should carry exactly
the pretrained encoder; if so, features from a task checkpoint ARE the pretrained encoder's features.
Also use it to confirm which pretrained checkpoint a fine-tuned model started from (differences per layer).

Usage: python check_encoder_weights.py facebook/vjepa2-vitl-fpc64-256 facebook/vjepa2-vitl-fpc16-256-ssv2 ...
       (the first is the reference; a local .safetensors path also works)
"""
import sys

import config  # noqa: F401  (sets HF_HOME)
import torch
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors import safe_open


def encoder_tensors(src):
    if src.endswith(".safetensors"):
        files = [src]
    else:
        files = [hf_hub_download(src, f) for f in list_repo_files(src) if f.endswith(".safetensors")]
    out = {}
    for fn in files:
        with safe_open(fn, "pt") as f:
            for k in f.keys():
                if ".encoder." in f".{k}":  # "encoder.*" (VJEPA2Model) or "vjepa2.encoder.*" (task checkpoints)
                    out[k[k.index("encoder."):]] = f.get_tensor(k)
    return out


def main():
    ref_id, *others = sys.argv[1:]
    ref = encoder_tensors(ref_id)
    print(f"{ref_id}: {len(ref)} encoder tensors, {sum(t.numel() for t in ref.values()) / 1e6:.1f}M params")
    for o in others:
        t = encoder_tensors(o)
        missing, extra = sorted(set(ref) - set(t)), sorted(set(t) - set(ref))
        diff = {k: (ref[k].float() - t[k].float()).abs().max().item() for k in sorted(set(ref) & set(t))
                if ref[k].shape != t[k].shape or not torch.equal(ref[k], t[k])}
        same = not missing and not extra and not diff
        print(f"{o}: {'IDENTICAL encoder' if same else 'DIFFERENT'}  "
              f"(missing {len(missing)}, extra {len(extra)}, differing {len(diff)} / {len(ref)})")
        for k in (missing + extra)[:5]:
            print(f"    key mismatch: {k}")
        for k, v in list(diff.items())[:10]:
            print(f"    {k}: max |diff| {v:.3g}")


if __name__ == "__main__":
    main()
