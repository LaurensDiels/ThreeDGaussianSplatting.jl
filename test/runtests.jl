using ThreeDGaussianSplatting
using Test

using CUDA, GPUArrays
using Adapt
using StaticArrays
using ColorTypes
using LinearAlgebra
using Statistics

const TDGS = ThreeDGaussianSplatting

if CUDA.functional()
    @testset "ThreeDGaussianSplatting.jl" begin
        include("adapt.jl")
        include("gaussians.jl")
        include("image_coordinates.jl")
        include("cameras.jl")
        include("special_sarrays.jl")
        include("forward_structs.jl")
        include("forward_core.jl")
        include("backward_core.jl")
        include("training.jl")  # Also tests the AdamOptimizer to some extent
    end
else
    @warn "CUDA is not functional! Skipping tests..."
end
