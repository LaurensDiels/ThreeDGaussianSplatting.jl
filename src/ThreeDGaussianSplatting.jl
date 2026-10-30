module ThreeDGaussianSplatting

using CUDA
using CUDA: i32
using GPUArrays
using Adapt
using StaticArrays
using StaticArrays: SUnitRange
using MacroTools
using FastCUDASSIM
using Underscores
using PlyIO
using JSON
using Images
using Rotations
using ColorTypes
using LinearAlgebra
using Statistics
using Random
using NearestNeighbors
import CImGui as ig
import GLFW
import ModernGL as GL
import CImGui.lib as lib
import CSyntax: @c


include("adapt.jl")
include("image_coordinates.jl")
include("cameras.jl")
include("misc.jl")
include("special_sarrays.jl")
include("sh.jl")
include("gaussians.jl")
include("scene.jl")
include("forward_structs.jl")
include("forward_core.jl")
include("backward_structs.jl")
include("backward_core.jl")
include("training_structs.jl")
include("optimizer.jl") # Uses TrainingOptions
include("training.jl")  # Uses AbstractOptimizer
include("gui.jl")

include("../scripts/scripts.jl")


# TODO: preferences.jl, either everything or nothing?
for n in names(@__MODULE__; all = true)
    if (Base.isidentifier(n)  # false for macros
          || startswith(string(n), '@')
        ) && n ∉ (Symbol(@__MODULE__), :eval, :include)  # always present
        @eval export $n
    end
end
# Note: the submodule Scripts is exported, but the methods within are not

# TODO: Precompilation (if CUDA.functional())

end
