# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Seam carving (content-aware image resizing and object removal) in Python, with two
interchangeable backends — a NumPy/OpenCV CPU implementation and a PyCUDA/CUDA-C GPU
implementation — behind one `Engine` API. Also ships a small Flask front-end.

## Commands

```bash
python3 -m pip install -r requirements.txt

# resize (writes res.jpg by default)
python cml_ui.py -r 512 -c 700 images/rem.jpeg

# object removal (masks must be the SAME shape as the source image)
python cml_ui.py image.jpg --protect mask_protect.jpg --remove mask_remove.jpg

# interactive mask drawing (OpenCV window; draw, ESC to exit, writes mask.jpg)
python scripts/draw_mask.py images/rem.jpeg

# flask app on :10800 — requires HOSTNAME to be exported (see Gotchas)
python web_ui.py --ip 0.0.0.0

# docker images (build from repo root)
bash scripts/make_image.sh build_cpu|build_cuda|clear
```

There is no test suite, linter, or formatter configured. Verification is manual: run
`cml_ui.py` on an image in `images/` and inspect the output.

Releases are cut by pushing a `v*.*.*` tag, which triggers
`.github/workflows/docker-publish.yml` to build and push both images to Docker Hub.
`VERSION` is read only by `scripts/make_image.sh` for local tags.

## Architecture

**Backend selection happens at import time** in `carver/__init__.py`: it tries
`from carver.carve_cuda import Core` and silently falls back to `carver.carve_slow`
on `ModuleNotFoundError` (i.e. no `pycuda`). Both modules expose the same
`Core.carve_column(img, mask_remove, mask_protect) -> (img, mask_remove, mask_protect)`
contract — removing exactly one vertical seam and carrying the masks along with it.
Any new backend just needs to implement that one method.

**`Engine` (`carver/__init__.py`) is the only public API** and is backend-agnostic:

- `resize(img, w, h, mask_protect=None)` — first `cv.resize`s up by the *larger* of the
  two scale factors so only downward carving is ever needed, then carves rows and columns.
- `remove(img, mask_remove, mask_protect=None)` — carves until the removal mask is empty,
  or (when a protect mask is present, which can pin some masked pixels forever) until the
  mask stops shrinking.
- Horizontal seams are done by rotating the image and masks 90° (`_crop_r`), carving
  columns, and rotating back. There is only ever one seam-finding direction.

**Masks** are 3-channel images on disk; `prepare_mask` reduces them to a `uint8` 0/1
single-channel array. They bias the energy map by `±MAX_PIXEL_BACKWARD_ENERGY * height`
— a value large enough to dominate any real seam cost, making removal seams effectively
free and protected seams effectively forbidden. Note the CPU and CUDA backends use
*different* constants for this (`8*255*2` vs `16*255*4092`); both are just "large enough".

**Energy** is backward energy: grayscale, then |Sobel_x| + |Sobel_y|. `carve_slow` has
both a scipy `convolve` version and the (used) `cv.Sobel` version.

**CUDA path** (`carver/carve_cuda.py` + `carver/carve.cu`): `carve.cu` is read from disk
at `Core.__init__` and JIT-compiled via `SourceModule`, so kernel edits need no build
step. The context is explicitly pushed/popped around every GPU section (`try/finally`).
Kernels: `bgr2gray`, `sobel_abs`, `add_mask_by_factor`, `min_energy_at_row`,
`get_min_index`. The DP over the energy matrix is parallelized *per row* — one kernel
launch per image row, all columns of that row in parallel — which is why the Python loop
`for row in range(1, height)` issues `height-1` launches on one stream. Backtracking and
the actual pixel deletion stay on the CPU in both backends.

## Gotchas

- `carve_slow.py` and `carve_cuda.py` use `np.int` / `np.bool`, removed in NumPy 1.24.
  This code needs `numpy < 1.24`; `requirements.txt` does not pin it.
- `web_ui.py` is stale: it calls `carver.run(...)`, which no longer exists on `Engine`
  (use `resize`), and reads `os.environ['HOSTNAME']` at import, which raises `KeyError`
  on most shells unless you `export HOSTNAME=...`.
- `scripts/create_gif.py` imports `minimum_seam` / `backward_energy_opencv` from
  `carve_slow`, which were renamed with a leading underscore.
- `scripts/make_image.sh` and the GitHub workflow expect `Dockerfile` and
  `cuda.Dockerfile` at the repo root, but the only Dockerfile lives at
  `scripts/Dockerfile` (it is the CUDA one; base image is overridable via `--build-arg BASE`).
- `numba` is still in `requirements.txt` but unused since commit `20cdcdc`.
- The CUDA backend casts the image to `float32` and returns it as `float32`;
  `cv.imwrite` still handles it, but downstream code should not assume `uint8`.
