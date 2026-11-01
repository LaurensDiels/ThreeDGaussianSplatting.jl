# ThreeDGaussianSplatting.jl

An implementation of 3D Gaussian Splatting in the Julia programming language.

## Installation

### Prerequisites

- An NVIDIA graphics card.

- Julia 1.13 (or newer). The recommended way to install Julia is via [Juliaup](https://github.com/julialang/juliaup). You can find instructions at https://julialang.org/downloads/.

- CUDA toolkit. 

    You can download it from <https://developer.nvidia.com/cuda-downloads>. Note that on Windows you also need [Visual Studio](https://visualstudio.microsoft.com/) integration. Properly installing CUDA toolkit can be tricky both on Windows and Linux. NVIDIA provides detailed instructions:
    - For Windows: <https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html>
    - For Linux: <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html>

    In particular we need `nvcc`. By default we assume it is on the `PATH` (i.e. you do not get an error when entering `nvcc --version` in a terminal). If not, or if you want to manually select the correct version if you have multiple versions of CUDA toolkit installed, you can use `ENV["TDGS_NVCC"] = ...` to point to the correct executable. This environment variable must be set before the build step, so normally before `add`ing `ThreeDGaussianSplatting` (see below).
    - Note that `cl.exe` comes in 32- and 64-bit versions. You almost certainly need the 64-bit one.

- A compatible CPU (host) compiler.
    On Windows this is `cl.exe` from Visual Studio. On Linux you can use `g++`. By default we assume this compiler is on the `PATH`. If not, use `ENV["TDGS_HOST_COMPILER"] = ...` to point to the correct executable, including the version. For example, you could use `ENV["TDGS_HOST_COMPILER"] = g++-12`.

- [COLMAP](https://colmap.github.io) with CUDA support, if you want to train 3DGS models starting from images. We require version 3.13 or newer.


### Installing ThreeDGaussianSplatting.jl

Our package is not in the general Julia registry, and we have to plans to request inclusion. Therefore, in a Julia REPL, use the GitHub link:
```julia-repl
julia> using Pkg; Pkg.add(url="https://github.com/LaurensDiels/ThreeDGaussianSplatting.jl")
```

This will automatically trigger the build step, where we compile a CUDA C wrapper for a CUB sort to a shared library we can use in Julia. This step require an appropriate device (`nvcc`) and host (`cl`, `g++`) compiler, cf. above.


### Build

If the build step failed because the compilers were not set properly, or if you have to build again because you changed graphics card (compute capability), you can force rebuilding using
```julia-repl
julia> Pkg.build("ThreeDGaussianSplatting.jl")
```


## Use

### Training

To train a 3DGS model from a directory containing images taken from the same camera, use
```julia-repl
julia> using ThreeDGaussianSplatting

julia> Scripts.train("/path/to/images/directory", "/path/to/output/directory")
```
(using appropriate paths). By default this will show the training process using an interactive GUI. This will make training a bit slower. If you do not need the gui, you can set the keyword argument `show_gui` to `false`:
```julia-repl
julia> Scripts.train(...; show_gui = false)
```
Important to note is that the visualization settings for the GUI are independent from the training settings. E.g. the GUI renders and training renders might use different background colors.

If you're brave or need custom functionality, you can of course also work directly with the `Trainer` struct. The implementation of `train` illustrates its use and can serve as a starting point.


### Rendering

You can start in interactive rendering GUI via
```julia-repl
julia> using ThreeDGaussianSplatting

julia> Scripts.render("/path/to/trained/3DGS/model")
```
The path here can refer to a trained `.ply` file, or the output directory from the training (by us or the reference 3DGS implementation).


## Links
- Reference 3DGS implementation: <https://github.com/graphdeco-inria/gaussian-splatting>
- Another Julia implementation: <https://github.com/JuliaNeuralGraphics/GaussianSplatting.jl>


## Contents
```@contents
Pages = (p -> p.second).(Main.PAGE_NAMES_AND_FILES)
```