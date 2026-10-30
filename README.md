# ThreeDGaussianSplatting

[![Build Status](https://github.com/LaurensDiels/ThreeDGaussianSplatting.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/ladiels/ThreeDGaussianSplatting.jl/actions/workflows/CI.yml?query=branch%3Amaster)

Note that the actual tests are skipped as CUDA is not available.

## Quick start

```julia-repl
julia> using Pkg; Pkg.add(url="https://github.com/LaurensDiels/ThreeDGaussianSplatting.jl")

julia> using ThreeDGaussianSplatting

julia> Scripts.train("/path/to/images/directory", "/path/to/output/directory")

julia> Scripts.render("/path/to/trained/3DGS/model")
```


## License

We distribute our code under the MIT license. Note that our implementation is not a port of the original PyTorch + CUDA [implementation](https://github.com/graphdeco-inria/gaussian-splatting), but a full reimplementation based on the 3DGS [paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/3d_gaussian_splatting_high.pdf) starting from scratch (code-wise). For some smaller algorithmic details, such as the order of densification, pruning and the optimizer step, we did refer to their code (though our Adam implementation is already completely different). We also have some consistency tests with the reference output. The original 3DGS implementation is distributed under a custom Inria and MPII Gaussian-Splatting [license](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md).


## Citing

If you used this software for academic research, please cite
```bib
@article{diels2024fast,
  title     = {{Fast 3D Gaussian Splatting Rendering via Easily Integrable Improvements}},
  author    = {Diels, Laurens and Vlaminck, Michiel and Philips, Wilfried and Luong, Hiep},
  journal   = {IEEE Signal Processing Letters},
  volume    = {32},
  pages     = {381--385},
  year      = {2024},
  publisher = {IEEE}
}
```