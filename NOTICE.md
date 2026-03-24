## FocusGS Notice

FocusGS is a derivative research codebase based on the following upstream projects:

- [hbb1/2d-gaussian-splatting](https://github.com/hbb1/2d-gaussian-splatting)
- [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)

The repository redistributes upstream code under the existing `Gaussian-Splatting License` in [LICENSE.md](LICENSE.md). That license allows redistribution of derivative works, but it keeps the original attribution requirements and limits use to non-commercial research and evaluation unless separate permission is obtained from the licensors.

At the time of this publication, FocusGS-specific changes are centered in these files:

- `arguments/__init__.py`
- `train.py`
- `scene/gaussian_model.py`
- `utils/camera_utils.py`
- `environment.yml`

This repository also contains or references third-party components that keep their own upstream notices, including:

- `submodules/diff-surfel-rasterization`
- `submodules/simple-knn`

No claim of ownership is made over unmodified upstream source files. When redistributing FocusGS, keep `LICENSE.md`, existing copyright/attribution notices, and this notice file together.
