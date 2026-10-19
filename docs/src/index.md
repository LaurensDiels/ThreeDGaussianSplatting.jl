# ThreeDGaussianSplatting.jl

An implementation of 3D Gaussian Splatting in the Julia programming language.

A work in progress...

## Install

### Prerequisites

NVIDIA graphics card

nvcc, either on PATH, or via ENV["TDGS_NVCC"]

compatible compiler (cl.exe for Windows (normally the 64-bit version), from compatible VS version; g++ for Linux)

colmap (for training only), either on PATH, or via ENV["TDGS_COLMAP]

Julia (v1.13+?)

### Build

Automatic, but `] build ThreeDGaussianSplatting` if necessary. For example, after switching GPU (compute capability)


## Use

scripts

Later: app


Mention GaussianSplatting.jl somewhere

## Contents
```@contents
Pages = (p -> p.second).(Main.PAGE_NAMES_AND_FILES)
```