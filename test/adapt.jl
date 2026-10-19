@testset "ArrayDomain" begin
    struct S{A, B}; a::A; b::B; end
    Adapt.@adapt_structure S
    s_gpu = S(CUDA.rand(2), CUDA.rand(3))      # S{CuArray{...}, CuArray{...}}
    s_cpu = adapt(CPU, s_gpu)                  # S{Vector{...}, Vector{...}}
    @test s_cpu.a isa Vector
    s_gpu2 = adapt(GPU, s_cpu)
    @test s_gpu2.a isa CuVector{Float32, CUDA.DeviceMemory}
    s_gpu3 = adapt(GPU{CUDA.UnifiedMemory}, s_cpu)
    @test s_gpu3.a isa CuVector{Float32, CUDA.UnifiedMemory}
    s_device = adapt(Device, s_cpu)
    @test s_device.a isa CuDeviceVector

    s_gpu3 = GPU(s_cpu)
    @test s_gpu3.a isa CuVector
end

@testset "im_mutable" begin
    @im_mutable Mutable 'I' "Abstract" struct S1 end
    @test @isdefined AbstractS1
    @test @isdefined MutableS1
    @test @isdefined IS1
    @test MutableS1 <: AbstractS1
    @test IS1 <: AbstractS1


    @im_mutable_with_adapt struct S2{T <: AbstractFloat, A <: AbstractVector{T}} <: AbstractFloat
        t::T
        a::A

        S2() = new{Float32, Vector{Float32}}(0., Float32[0.])
        function :common(t::T, a::A) where {T <: AbstractFloat, A <: AbstractVector{T}}
            return new{T, A}(t, a)
        end
        function S2(i::Integer)
            f = i + Float16(0.5)
            return new{Float16, Vector{Float16}}(f, [f])
        end

        :mutable(s::AbstractString) = new{Float64, Vector{Float64}}(parse(Float64, s), [2.])
        function MutableS2(::Tuple{})
            return new{Float32, Vector{Float32}}(0.75, Float32[])
        end

        ImmutableS2(s::AbstractString) = new{Float64, Vector{Float64}}(
            parse(Float64, s) + 2, [3.])
        function :immutable(a::A) where {T <: AbstractFloat, A <: AbstractArray{T}}
            @assert !isempty(a)
            return new{T, A}(a[1])
        end
    end
    @test MutableS2 <: AbstractFloat
    @test ImmutableS2 <: AbstractFloat
    MS2 = MutableS2; IS2 = ImmutableS2;
    @test MS2().t === IS2().t === Float32(0.)
    @test MS2(1).t === IS2(1).t === Float16(1.5)
    @test MS2("1").t === Float64(1.)
    @test MS2(()).t === Float32(0.75)
    @test_throws MethodError MS2([1.])
    @test IS2("1").t === Float64(3.)
    @test IS2([1.]).t === 1.
    @test_throws MethodError IS2(())


    @test adapt(GPU, MS2()) isa MS2{Float32, <:CuVector{Float32}}
    @test adapt(CPU, adapt(GPU, MS2())) isa MS2{Float32, <:Vector{Float32}}
    @test adapt(Device, MS2()) isa IS2{Float32, <:CuDeviceVector{Float32}}
    @test adapt(Device, adapt(GPU, MS2())) isa IS2{Float32, <:CuDeviceVector{Float32}}
    # Cannot convert back to CPU or GPU
    @test adapt(GPU, IS2()) isa IS2{Float32, <:CuVector{Float32}}
    @test adapt(CPU, adapt(GPU, IS2())) isa IS2{Float32, <:Vector{Float32}}
    @test adapt(Device, IS2()) isa IS2{Float32, <:CuDeviceVector{Float32}}

    abstract type Coolness end
    struct Amazing <: Coolness end; struct Boring <: Coolness end
    @im_mutable struct S3{C <: Coolness, A <: AbstractVector}
        a::A

        function S3(::Type{C}, a::A) where {C <: Coolness, A <: AbstractVector}
            new{C, A}(a)
        end
    end
    @adapt_im_mutable Adapt.adapt_structure(to, s::S3{C}) where C = S3(C, adapt(to, s.a))
    @test adapt(Device, MutableS3(Amazing, [0])) isa
        ImmutableS3{Amazing, <:CuDeviceVector{Int64}}
end

@testset "replace" begin
    @replace Int => Float64 f(x::Int) = typeof(x)
    @test_throws MethodError f(1) 
    @test f(1.) == Float64

    @replace x => (a, b, c, d) x = 1
    @test a == b == c == d == 1
end