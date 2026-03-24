# FocusGS

FocusGS is an object-focused derivative of [2D Gaussian Splatting](https://github.com/hbb1/2d-gaussian-splatting). It uses `Polarization Loss` and `ObjectMark Filtering` to bias training toward the target object, suppress background Gaussians, and extract object-centric meshes more cleanly.

This repository is not the official 2DGS implementation. It is a derivative research codebase built on top of [hbb1/2d-gaussian-splatting](https://github.com/hbb1/2d-gaussian-splatting) and, transitively, [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting).

> License notice
>
> This repository keeps the upstream Gaussian-Splatting license in [LICENSE.md](LICENSE.md). Redistribution is allowed with attribution, but usage remains limited to non-commercial research and evaluation unless separate permission is obtained from the original licensors. See [NOTICE.md](NOTICE.md) for derivative-work and attribution details.

## What FocusGS Adds

- `Polarization Loss`: uses the renderer's `polarization_alpha` signal to align the rendered object region with the supervision mask.
- `ObjectMark Guidance`: learns a per-Gaussian `objectmark_score` that separates object Gaussians from background Gaussians.
- `ObjectMark Filtering`: prunes low-score Gaussians during training so later reconstruction and meshing focus on the object.
- `Masked RGB Supervision`: if a mask is provided, RGB reconstruction can be restricted to the target region.
- `Backward Compatibility`: saved checkpoints and PLY files remain readable even when older files use `mask_label` instead of `objectmark_score`.

## Method Overview

FocusGS keeps the 2DGS training pipeline and adds object-aware supervision on top of it:

1. Load an object mask from `<dataset>/mask`.
2. Compute RGB reconstruction on the object region.
3. Apply `Polarization Loss` so rendered alpha matches the object mask.
4. Learn `objectmark_score` for each Gaussian with foreground/background guidance.
5. Periodically prune Gaussians whose `objectmark_score` falls below a threshold.

As a result, the learned Gaussian set becomes increasingly object-centric, which helps produce cleaner object-only meshes at export time.

## Dataset Layout

FocusGS follows the same COLMAP-style scene structure as 2DGS / 3DGS and adds an optional `mask` directory:

```text
<scene>/
  images/
    000001.png
    000002.png
    ...
  mask/
    000001.png
    000002.png
    ...
  sparse/
    0/
      cameras.bin
      images.bin
      points3D.bin
```

Mask notes:

- Put object masks under `<scene>/mask`.
- Use the same base filename as the source image whenever possible.
- Grayscale binary masks are recommended: object = white, background = black.
- If no mask is found, FocusGS falls back to standard RGB-driven behavior and the object-focused losses are skipped.

## Installation

```bash
git clone https://github.com/WooSungHyun03/FocusGS.git --recursive
cd FocusGS

conda env create --file environment.yml
conda activate focusgs
```

If you update submodules later:

```bash
git submodule update --init --recursive
```

## Training

Basic training:

```bash
python train.py -s <path_to_scene> -m output/<scene_name>
```

FocusGS automatically looks for masks in `<path_to_scene>/mask`.

Example with object-focused settings:

```bash
python train.py \
  -s <path_to_scene> \
  -m output/<scene_name> \
  --lambda_polarization 0.1 \
  --lambda_objectmark_foreground 0.2 \
  --lambda_objectmark_background 0.8 \
  --objectmark_guidance_from_iter 0 \
  --objectmark_pruning_threshold 0.5 \
  --objectmark_pruning_iterations 5000 10000 15000 20000 25000 30000
```

### FocusGS-Specific Arguments

| Argument | Default | Purpose |
| --- | --- | --- |
| `--lambda_polarization` | `0.1` | Strength of Polarization Loss |
| `--lambda_objectmark_foreground` | `0.2` | Encourages high `objectmark_score` on the object region |
| `--lambda_objectmark_background` | `0.8` | Suppresses `objectmark_score` on background regions |
| `--objectmark_guidance_from_iter` | `0` | Iteration to start ObjectMark guidance |
| `--objectmark_pruning_threshold` | `0.5` | Gaussians below this score are pruned |
| `--objectmark_pruning_iterations` | `5000 10000 15000 20000 25000 30000` | Iterations where ObjectMark Filtering is applied |
| `--disable_mask_l1` | `False` | Disable masked RGB L1 loss and use full-image L1 instead |

Other useful inherited arguments:

- `--lambda_normal`
- `--lambda_dist`
- `--depth_ratio`
- `--densify_until_iter`

## Mesh Extraction

After training, mesh extraction follows the same workflow as 2DGS:

```bash
python render.py -s <path_to_scene> -m output/<scene_name>
```

For larger or unbounded scenes:

```bash
python render.py -s <path_to_scene> -m output/<scene_name> --unbounded --mesh_res 1024
```

Because FocusGS prunes low-confidence background Gaussians during training, the exported mesh is typically more object-centric than a vanilla 2DGS reconstruction on the same masked scene.

If you want stronger filtering:

- increase `--objectmark_pruning_threshold`
- prune later, after `objectmark_score` has stabilized
- keep masks as clean and binary as possible

## Output Notes

- Saved PLY files include `objectmark_score`.
- Older PLY files with `mask_label` are still readable.
- Training writes lightweight logs such as `train_vram.txt`, `gs.txt`, and `time.txt` into the model output directory.

## Upstream References

FocusGS is built on the following upstream works:

- 2DGS repository: https://github.com/hbb1/2d-gaussian-splatting
- 2DGS paper: https://arxiv.org/pdf/2403.17888
- 3DGS repository: https://github.com/graphdeco-inria/gaussian-splatting

## Acknowledgements

FocusGS is a derivative work of [2DGS](https://github.com/hbb1/2d-gaussian-splatting), which itself builds upon [3DGS](https://github.com/graphdeco-inria/gaussian-splatting). Unmodified upstream files remain under their original copyright and license notices. Additional third-party dependencies retain their own notices as distributed in this repository.

## Citation

If you use FocusGS, please cite the original 2DGS paper and acknowledge this repository as a derivative implementation.

```bibtex
@inproceedings{Huang2DGS2024,
    title={2D Gaussian Splatting for Geometrically Accurate Radiance Fields},
    author={Huang, Binbin and Yu, Zehao and Chen, Anpei and Geiger, Andreas and Gao, Shenghua},
    publisher = {Association for Computing Machinery},
    booktitle = {SIGGRAPH 2024 Conference Papers},
    year      = {2024},
    doi       = {10.1145/3641519.3657428}
}
```
