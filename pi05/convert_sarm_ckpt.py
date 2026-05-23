"""
Convert SARM/Evo-RL pi05 LoRA checkpoint to RoboTwin openpi format.

SARM checkpoint structure:
  {src}/
    adapter_config.json          # LoRA config (r=16, alpha=8)
    adapter_model.safetensors    # LoRA delta weights (PEFT format)
    policy_preprocessor*.safetensors  # norm stats (Evo-RL format)
    config.json / train_config.json

RoboTwin expected structure:
  {dst}/
    model.safetensors            # merged full weights (PI0Pytorch keys)
    assets/{asset_id}/
      norm_stats.json            # openpi NormStats format

Usage:
  cd /root/gpufree-data/ljw/RoboTwin/policy/pi05
  python convert_sarm_ckpt.py \
    --src  /root/gpufree-data/ljw/outputs/pi05_rabc_fold_cloth/checkpoints/040000/pretrained_model \
    --base /root/gpufree-data/data/models/pi05_base/model.safetensors \
    --dst  /root/gpufree-data/ljw/RoboTwin/policy/pi05/checkpoints/pi05_aloha_full_base/sarm_fold_cloth/040000 \
    --asset-id fold_cloth
"""

import argparse
import json
import pathlib

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file


# ---------------------------------------------------------------------------
# 1. LoRA merge
# ---------------------------------------------------------------------------

def load_safetensors(path: str) -> dict[str, torch.Tensor]:
    out = {}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            out[k] = f.get_tensor(k)
    return out


def merge_lora(base_sd: dict, adapter_sd: dict, lora_r: int, lora_alpha: float) -> dict[str, torch.Tensor]:
    """Merge LoRA deltas into the base weights.

    PEFT keys look like:
      base_model.model.model.<name>.lora_A.weight
      base_model.model.model.<name>.lora_B.weight

    Base model keys look like:
      <name>.weight
    """
    scaling = lora_alpha / lora_r
    prefix = "base_model.model.model."

    # Build {bare_name -> {A, B}} from adapter
    lora_pairs: dict[str, dict] = {}
    for k, v in adapter_sd.items():
        if not k.startswith(prefix):
            continue
        bare = k[len(prefix):]                         # e.g. "action_in_proj.lora_A.weight"
        parts = bare.rsplit(".", 2)                    # ["action_in_proj", "lora_A", "weight"]
        if len(parts) != 3 or parts[2] != "weight":
            continue
        param_name = parts[0]                          # "action_in_proj"
        ab = parts[1]                                  # "lora_A" or "lora_B"
        lora_pairs.setdefault(param_name, {})[ab] = v

    merged = {k: v.clone() for k, v in base_sd.items()}

    for param_name, ab in lora_pairs.items():
        if "lora_A" not in ab or "lora_B" not in ab:
            print(f"  [WARN] incomplete LoRA pair for {param_name}, skipping")
            continue
        weight_key = f"{param_name}.weight"
        if weight_key not in merged:
            print(f"  [WARN] base key not found: {weight_key}, skipping")
            continue

        lora_A = ab["lora_A"].float()   # (r, in_features)
        lora_B = ab["lora_B"].float()   # (out_features, r)
        delta = (lora_B @ lora_A) * scaling

        base_w = merged[weight_key].float()
        merged[weight_key] = (base_w + delta).to(merged[weight_key].dtype)
        print(f"  merged: {weight_key}  delta_norm={delta.norm():.4f}")

    return merged


# ---------------------------------------------------------------------------
# 2. Norm stats conversion
# ---------------------------------------------------------------------------

def load_evorl_norm_stats(safetensors_path: str) -> dict[str, dict[str, np.ndarray]]:
    """Load Evo-RL normalizer safetensors into {feature: {stat: array}}."""
    stats: dict[str, dict[str, np.ndarray]] = {}
    with safe_open(safetensors_path, framework="pt") as f:
        for key in f.keys():
            # key format: "observation.state.mean", "action.q01", ...
            parts = key.rsplit(".", 1)
            if len(parts) != 2:
                continue
            feature, stat = parts[0], parts[1]
            stats.setdefault(feature, {})[stat] = f.get_tensor(key).numpy()
    return stats


EVORL_TO_OPENPI_KEYS = {
    "observation.state": "state",
    "action": "action",
}


def convert_norm_stats(evorl_stats: dict, feature_map: dict) -> dict:
    """Convert to openpi NormStats format: {name: {mean, std, q01, q99}}."""
    openpi_stats = {}
    for evorl_key, openpi_key in feature_map.items():
        if evorl_key not in evorl_stats:
            print(f"  [WARN] norm stat key not found: {evorl_key}")
            continue
        src = evorl_stats[evorl_key]
        entry = {}
        for stat in ("mean", "std", "q01", "q99"):
            if stat in src:
                arr = src[stat].flatten().tolist()
                entry[stat] = arr
            else:
                print(f"  [WARN] missing stat '{stat}' for {evorl_key}")
        openpi_stats[openpi_key] = entry
        print(f"  norm stat: {evorl_key} -> {openpi_key} (shape {src.get('mean', np.array([])).shape})")
    return openpi_stats


def save_openpi_norm_stats(norm_stats: dict, path: pathlib.Path):
    """Write openpi norm_stats.json (pydantic-compatible format)."""
    payload = {"norm_stats": norm_stats}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(f"  saved norm stats -> {path}")


# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True,
                        help="SARM pretrained_model dir (contains adapter_model.safetensors)")
    parser.add_argument("--base", required=True,
                        help="pi05_base full weights (model.safetensors)")
    parser.add_argument("--dst", required=True,
                        help="Output checkpoint dir for RoboTwin")
    parser.add_argument("--asset-id", default="fold_cloth",
                        help="Dataset asset id (used as subdir under assets/)")
    parser.add_argument("--step", type=int, default=None,
                        help="Override checkpoint step (auto-detected from src dir name)")
    args = parser.parse_args()

    src = pathlib.Path(args.src)
    dst = pathlib.Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    # --- read LoRA config ---
    adapter_cfg_path = src / "adapter_config.json"
    with open(adapter_cfg_path) as f:
        adapter_cfg = json.load(f)
    lora_r     = adapter_cfg["r"]
    lora_alpha = adapter_cfg["lora_alpha"]
    print(f"LoRA config: r={lora_r}, alpha={lora_alpha}, scaling={lora_alpha/lora_r:.3f}")

    # --- merge weights ---
    print("\n[1/3] Loading base weights ...")
    base_sd = load_safetensors(args.base)
    print(f"  base keys: {len(base_sd)}")

    print("[1/3] Loading LoRA adapter ...")
    adapter_sd = load_safetensors(str(src / "adapter_model.safetensors"))
    print(f"  adapter keys: {len(adapter_sd)}")

    print("[1/3] Merging LoRA ...")
    merged_sd = merge_lora(base_sd, adapter_sd, lora_r, lora_alpha)

    out_model = dst / "model.safetensors"
    save_file(merged_sd, str(out_model))
    print(f"  saved merged model -> {out_model}  ({out_model.stat().st_size/1e9:.2f} GB)")

    # --- convert norm stats ---
    print("\n[2/3] Converting norm stats ...")
    norm_sf = src / "policy_preprocessor_step_2_normalizer_processor.safetensors"
    if not norm_sf.exists():
        # fallback: try postprocessor
        norm_sf = src / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"

    if norm_sf.exists():
        evorl_stats = load_evorl_norm_stats(str(norm_sf))
        openpi_stats = convert_norm_stats(evorl_stats, EVORL_TO_OPENPI_KEYS)
        norm_out = dst / "assets" / args.asset_id / "norm_stats.json"
        save_openpi_norm_stats(openpi_stats, norm_out)
    else:
        print("  [WARN] no norm stats file found, skipping")

    # --- summary ---
    print("\n[3/3] Done. Output structure:")
    for p in sorted(dst.rglob("*")):
        if p.is_file():
            size = p.stat().st_size
            unit = "MB" if size > 1e6 else "KB"
            print(f"  {p.relative_to(dst)}  ({size/1e6:.1f} MB)" if size > 1e6 else
                  f"  {p.relative_to(dst)}  ({size/1e3:.1f} KB)")

    print(f"""
To run in RoboTwin simulation, call PI0 with:
  train_config_name = "pi05_aloha_full_base"
  model_name        = "{dst.parent.name}"
  checkpoint_id     = "{dst.name}"
  asset_id          = "{args.asset_id}"  (auto-detected from assets/ dir)
""")


if __name__ == "__main__":
    main()
