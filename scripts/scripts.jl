module Scripts

using ..ThreeDGaussianSplatting
using StaticArrays
using CUDA
using FastCUDASSIM

include("render.jl")
include("train.jl")

export render
export train

end