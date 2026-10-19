# Will compile cub_sort_pairs_wrapper.cu for Device 0
# nvcc and the cpu compiler need to be on PATH, or specified via ENV

using CUDA

dev = CUDA.device()
cc = capability(dev)                     # E.g. v"8.6.0"
arch_flag = "sm_$(cc.major)$(cc.minor)"  # E.g. sm_86

cu_path = joinpath(@__DIR__, "cub_sort_pairs", "cub_sort_pairs_wrapper.cu")
so_path = joinpath(@__DIR__, "cub_sort_pairs", "cub_sort_pairs_wrapper.so")

cmd_args = [
    get(ENV, "TDGS_NVCC", "nvcc"),  # Where the latter option assumes it is on PATH
    cu_path, "-o", so_path, "-shared",
    "-std=c++17",
    "-arch=$arch_flag"
]

# Host compiler args
let x = get(ENV, "TDGS_HOST_COMPILER", nothing)
    if !isnothing(x)
        append!(cmd_args, ["-ccbin", x])
    end
end
if Sys.iswindows()
    append!(cmd_args, ["-Xcompiler", "/Zc:preprocessor"])  # for cl
else
    append!(cmd_args, ["-Xcompiler", "-fPIC"])  # Position Independent Code
end

# Loosen compatibility requirements
append!(cmd_args, 
    ["-allow-unsupported-compiler", 
     "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"])

cmd = Cmd(cmd_args)
println("Running command: $cmd")  # Will show up in the log file
run(cmd)