"""
Definitions for converting (via Adapt.jl) between CPU (`Array`), GPU (`CuArray`) and
Device (`CuDeviceArray`) versions. Because mutable structs are not allowed in kernels,
we also include macros for creating both mutable and immutable structs from one single
definition.
"""

"Abstract type for the domain of an AbstractArray (CPU, GPU or Device)."
abstract type ArrayDomain end

"`ArrayDomain` accessible from the CPU"
abstract type CPUAccessibleDomain <: ArrayDomain end

"""
    CPU <: CPUAccessibleDomain

`ArrayDomain` backed by CPU-memory, accessed from the CPU, like Array.

---

    CPU(x)

Adapt `x` to `CPU`.
"""
struct CPU    <: CPUAccessibleDomain end
# The function is defined later, after the @replace macro.

"""
    GPU{M <: CUDACore.AbstractMemory} <: CPUAccessibleDomain

`ArrayDomain` backed by GPU-memory (CUDA.DeviceMemory) CPU/GPU-memory (CUDA.UnifiedMemory),
accessed from the CPU (the host) (like CuArray).

---

    GPU(x)

Adapt `x` to `GPU{CUDA.DeviceMemory}`.

---

    GPU{M}(x) where M <: CUDACore.AbstractMemory

Adapt `x` to `GPU{M}`.
"""
struct GPU{M <: CUDACore.AbstractMemory} <: CPUAccessibleDomain end

"""
    Device <: ArrayDomain

`ArrayDomain` backed by GPU-memory, accessed from the GPU (the device) (like CuDeviceArray).

---

    Device(x)

Adapt `x` to `Device`.
"""
struct Device <: ArrayDomain end

"""
    base_parent(x)

Recursively apply `parent` until we stagnate.
"""
function base_parent(x)
    p = parent(x) 
    p === x ? x : base_parent(p)
end

"""
    array_domain(x)

Return the most likely `ArrayDomain` of `x`, by checking if `x`'s `base_parent` is a
`CuDeviceArray` (-> `Device`), `CuArray` (-> `GPU{...}`) or something else (-> `CPU`).
""" 
function array_domain(x) 
    bp = base_parent(x)
    if bp isa CuDeviceArray
        return Device
    elseif bp isa CuArray
        return GPU{memory_type(bp)}
    else
        return CPU
    end
end

"""
    array_domain(x::Type)

Return the most likely `ArrayDomain` of the `Type` `x`, by checking if its
`Adapt.unwrap_type` is `CuDeviceArray` (-> `Device`), `CuArray` (-> `GPU{...}`) or something
else (-> `CPU`).
""" 
function array_domain(x::Type) 
    bt = unwrap_type(x)  # the type version of base_parent
    if bt <: CuDeviceArray
        return Device
    elseif bt <: CuArray
        return GPU{memory_type(bt)}
    else
        return CPU
    end
end


"""
    create_array(::Type{<:CPUAccessibleDomain}, T, dims)

Create an unitilized array of the provided `array_domain` type, size `dims` and `eltype`
`T`.
"""
function create_array end
create_array(::Type{CPU}, T, dims) = Array{T}(undef, dims)
create_array(::Type{GPU{M}}, T, dims) where M = CuArray{T, length(dims), M}(undef, dims)
create_array(::Type{GPU}, T, dims) = create_array(GPU{CUDA.DeviceMemory}, T, dims)

"""
    create_zeros(::Type{<:CPUAccessibleDomain}, T, dims)

Create a zero-filled array of the provided `array_domain` type, size `dims` and `eltype`
`T` (which must support `zero`).
"""
create_zeros(::Type{D}, T, dims) where {D <: CPUAccessibleDomain} = 
    fill!(create_array(D, T, dims), zero(T))
# (Could use Base.zeros, but let's stay consistent with create_array)

# Above we had the gneric version, in particular for CUDA.UnifiedMemory, but calling
# zeros / CUDA.zeros directly might be every so slightly faster
create_zeros(::Type{CPU}, T, dims) = zeros(T, dims)
create_zeros(::Type{GPU{CUDA.DeviceMemory}}, T, dims) = CUDA.zeros(T, dims)
create_zeros(::Type{GPU}, T, dims) = create_zeros(GPU{CUDA.DeviceMemory}, T, dims)

Adapt.adapt(::Type{CPU}, x) = adapt(Array, x)
Adapt.adapt(::Type{GPU}, x) = adapt(CuArray, x)
Adapt.adapt_storage(::Type{GPU{M}}, x::AbstractArray{T, N}) where {M, T, N} = 
    adapt(CuArray{T, N, M}, x)
# (Could use the same approach for CPU, GPU and Device, but a simple adapt also works there)
Adapt.adapt(::Type{Device}, x) = adapt(CUDA.KernelAdaptor(), adapt(GPU, x))
# Cannot adapt directly from Array to CuDeviceArray. Conversely, we cannot adapt away from
# a CuDeviceArray (neither to an Array, nor to a CuArray).
# In practice we will not adapt(Device, ...), as @cuda already calls
# adapt(CUDA.KernelAdaptor(), ...) automatically.
# (Note that we cannot use @cuda on a CPU version, though this is quite natural.)

# We can now write something like
#   struct S{A, B}; a::A; b::B; end
#   Adapt.@adapt_structure S
#   s = S(CUDA.rand(2), CUDA.rand(3))  # S{CuArray{...}, CuArray{...}}
#   adapt(CPU, s)                      # S{Vector{...}, Vector{...}}
# Note that this requires all Array-like fields of S to be generic.

# One problem is that we cannot use mutable structs for kernel arguments. Of course we won't
# be mutating the struct itself (not to be confused with the arrays within), so this
# is not really an issue, except we will need an immutable variant and an adaptation from
# the mutable struct to the immutable version for CUDA.KernelAdaptor (for Device).
# Hence the following macro.

const DEFAULT_MUTABLE_PREFIX = :Mutable
const DEFAULT_IMMUTABLE_PREFIX = :Immutable
const DEFAULT_ABSTRACT_PREFIX = :Abstract

"""
    @im_mutable [mutable_prefix immutable_prefix abstract_prefix] struct_def

A macro to create both a mutable and immutable version of a struct. We also create an
abstract supertype for both. If the supplied struct expression declares a supertype,
it will be the supertype of the new abstract (super)type.

Sometimes you want different inner constructors for the mutable and immutable versions,
for example to avoid string interpolation in error messages inside of CUDA kernels.
These different constructors can be explicitly declared using
`function :mutable(...) ... end` and `function :immutable(...) ... end`, or by using the
name of the mutable and immutable types. For common inner constructors the supplied struct
name can be used, or alternatively `function :common(...) ... end`.

The use of `@im_mutable` cannot be preceeded by documentation. Therefore, this must be
written afterwards. E.g.
```julia
@im_mutable struct S end

"Immutable version"
ImmutableS
```

# Arguments
- `mutable_prefix`: The name of the mutable version of the struct will be the supplied
    name prepended by the (`Symbol`, `String`, `Char`, etc.) `mutable_prefix`.
    Defaults to $DEFAULT_MUTABLE_PREFIX.
- `immutable_prefix`: Similar, for the immutable version. Defaults to
    $DEFAULT_IMMUTABLE_PREFIX.
- `abstract_prefix`: Similar, for the abstract supertype. Defaults to
    $DEFAULT_ABSTRACT_PREFIX.
- `struct_def`: The `struct ... end` expression to modify.

# Example
```julia-repl
julia> @im_mutable M I A struct S{T, N, A <: AbstractArray{T, N}} <: AbstractArray{T, N}
    a::A

    function S()
        println("Common constructor (supplied name)")
        new{Int, 1, Vector{Int}}([1])
    end

    function MS(a::A) where {T, N, A <: AbstractArray{T, N}}
        println("Mutable constructor (mutable name)")
        new{T, N, A}(a)
    end

    function :immutable(a::A) where {T, N, A <: AbstractArray{T, N}}
        println("Immutable constructor (:immutable)")
        new{T, N, A}(a)
    end
end

julia> function Base.show(io::IO, mime::MIME"text/plain", x::AS)
           show(io, mime, typeof(x))
           print(":\n")
           show(io, mime, x.a)
       end

julia> m = MS()
Common constructor (supplied name)
MS{Int64, 1, Vector{Int64}}:
1-element Vector{Int64}:
 1

julia> m isa AS
true

julia> m isa AbstractArray
true

julia> m = MS([2.])
Mutable constructor (mutable name)
MS{Float64, 1, Vector{Float64}}:
1-element Vector{Float64}:
 2.0

julia> i = IS()
Common constructor (supplied name)
IS{Int64, 1, Vector{Int64}}:
1-element Vector{Int64}:
 1

julia> i = IS([2.])
Immutable constructor (:immutable)
IS{Float64, 1, Vector{Float64}}:
1-element Vector{Float64}:
 2.0
```
"""
macro im_mutable(mutable_prefix, immutable_prefix, abstract_prefix, struct_def)
    esc(_im_mutable(mutable_prefix, immutable_prefix, abstract_prefix, struct_def))
end

macro im_mutable(struct_def)
    esc(_im_mutable(
        DEFAULT_MUTABLE_PREFIX, DEFAULT_IMMUTABLE_PREFIX, DEFAULT_ABSTRACT_PREFIX, 
        struct_def))
end

function _im_mutable(
    mutable_prefix, immutable_prefix, abstract_prefix, 
    struct_def)

    name, parameter_names, parameter_supertypes, supertype, body = 
        _destructure_struct_def(struct_def)

    # Create new names and name-parameter combinations
    mutable_name = Symbol(mutable_prefix, name)
    immutable_name = Symbol(immutable_prefix, name)
    abstract_name = Symbol(abstract_prefix, name)

    if !isempty(parameter_names)
        # Recombine parameter_names and parameter_supertypes
        parameters = map(parameter_names, parameter_supertypes) do n, t
            if isnothing(t)
                return n
            else
                return :($n <: $t)
            end
        end

        abstract_name_and_parameters = :($abstract_name{$(parameters...)})
        abstract_name_and_param_names = :($abstract_name{$(parameter_names...)})
        mutable_name_and_parameters = :($mutable_name{$(parameters...)})
        immutable_name_and_parameters = :($immutable_name{$(parameters...)})
    else
        abstract_name_and_parameters = abstract_name
        # Used in abstract type definition
        # Not important:   abstract type AS{} end   also works fine
        abstract_name_and_param_names = abstract_name
        # Used in concrete struct definition
        # Important:   struct IS{} <: AS{} (...) end   does not work
        mutable_name_and_parameters = mutable_name
        immutable_name_and_parameters = immutable_name
    end


    # Declare new abstract supertype
    if isnothing(supertype)
        abstract_def = :(abstract type $abstract_name_and_parameters end)
    else
        abstract_def = :(abstract type $abstract_name_and_parameters <: $supertype end)
    end

    # Declare new mutable and immutable structs
    # Extract inner constructors and distribute them over these two
    inner_constructors = []
    inner_constructor_names = []
    struct_rest = []
    for ex in body
        fn_name, _, _, _ = _destructure_function_def(ex)
        if isnothing(fn_name)  # not a constructor
            push!(struct_rest, ex)
        else
            push!(inner_constructors, ex)
            push!(inner_constructor_names, fn_name)      
        end
    end

    # First for the mutable struct
    mutable_inner_constructor_names = (:(:common), name, :(:mutable), mutable_name)
    # (Not esc_mutable_name)
    mutable_inner_ids = filter(eachindex(inner_constructor_names)) do i
        inner_constructor_names[i] in mutable_inner_constructor_names
    end  # indices of the relevant inner constructors for the mutable struct
    # Rename these to the mutable struct name
    mutable_inner_constructors = map(mutable_inner_ids) do i
        MacroTools.postwalk(inner_constructors[i]) do ex
            ex in mutable_inner_constructor_names ? mutable_name : ex
        end
    end

    mutable_struct_def = quote
        mutable struct $mutable_name_and_parameters <: 
                $abstract_name_and_param_names
            $(struct_rest...)
            $(mutable_inner_constructors...)
        end
    end

    # Completely similar for the immutable struct
    immutable_inner_constructor_names = (:(:common), name, :(:immutable), immutable_name)
    immutable_inner_ids = filter(eachindex(inner_constructor_names)) do i
        inner_constructor_names[i] in immutable_inner_constructor_names
    end
    immutable_inner_constructors = map(immutable_inner_ids) do i
        MacroTools.postwalk(inner_constructors[i]) do ex
            ex in immutable_inner_constructor_names ? immutable_name : ex
        end
    end

    immutable_struct_def = quote
        struct $immutable_name_and_parameters <: 
                $abstract_name_and_param_names
            $(struct_rest...)
            $(immutable_inner_constructors...)
        end
    end
    # Note: we cannot use something like   $mutability_keyword struct ... end   ,
    # for   mutability_keyword = (:mutable, :())   , making it hard to combine the mutable
    # and immutable paths into a single loop

    return quote
        $abstract_def
        $mutable_struct_def
        $immutable_struct_def
    end
end

"""
    _destructure_struct_def(struct_def)

Attempt to split the supplied struct definition into the
+ non-parametric struct name
+ parameter names
+ parameter supertypes
+ struct supertype
+ struct body.
The parameter names, supertypes and struct body are returned as a (possibly empty) `Vector`.
If the struct or a parameter does not have a supertype, we use `nothing`.

# Example
```
julia> foreach(println, enumerate(_destructure_struct_def(quote
            struct S{A, B <: Integer} <: AbstractFloat
                a::A
                b::B

                S(a::A) = new{A, Int}(a, 0)
            end
        end)))
(1, :S)
(2, Any[:A, :B])
(3, Any[nothing, :Integer])
(4, :AbstractFloat)
(5, Any[:(a::A), :(b::B), :(S(a::A) = begin
          new{A, Int}(a, 0)
      end)])

julia> foreach(println, enumerate(_destructure_struct_def(:(struct S end))))
(1, :S)
(2, Any[])
(3, Any[])
(4, nothing)
(5, Any[])
```
"""
function _destructure_struct_def(struct_def)
    if !@capture(struct_def, struct struct_head_ body__ end)
        throw(ArgumentError("Did not find a struct definition in the supplied code!"))
    end
    # Otherwise we now have an Expr struct_head and Vector{Expr} body
    # (though as Vector{Any})

    if !@capture(struct_head, name_and_params_ <: supertype_)
        name_and_params = struct_head
    end
    # otherwise automatically supertype = nothing
    if !@capture(name_and_params, name_{parameters__})
        name = name_and_params
        parameters = []
    end
    # parameters includes supertypes ({A <: B})

    if !isempty(parameters)
        parameter_names = []
        parameter_supertypes = []
        for ex in parameters
            if @capture(ex, A_ <: B_)
                push!(parameter_names, A)
                push!(parameter_supertypes, B)
            else
                push!(parameter_names, ex)
                push!(parameter_supertypes, nothing)
            end
        end
    else
        parameter_names = []
        parameter_supertypes = []
    end

    return name, parameter_names, parameter_supertypes, supertype, body
end


"""
    _destructure_function_def(function_def)

Attempt to split the supplied function definition into a tuple consisting of the
function name, function arguments (including types), where declaration (without the
`where` itself or the braces) and function body.

When a part is missing we return `nothing` in this tuple entry. In particular, when we
cannot find the function name (e.g. because this in fact is not a function definition)
we return a tuple consisting of four `nothing`s.

# Example
```julia-repl
julia> _destructure_function_def(:(function f(x::S, y::T) where {S, T} x + y end))
(:f, Any[:(x::S), :(y::T)], Any[:S, :T], quote
    x + y
end)

julia> _destructure_function_def(:(g(x) = x^2))
(:g, Any[:x], nothing, quote
    x ^ 2
end)

julia> _destructure_function_def(:(function h end))
(:g, nothing, nothing, nothing)
```
"""
function _destructure_function_def(function_def)
    if @capture(
        function_def, 
        function fn_name_(fn_args__) fn_body_ end
    )
        return fn_name, fn_args, nothing, fn_body
    elseif @capture(
        function_def, 
        function fn_name_(fn_args__) where fn_where__ fn_body_ end
    )
        return fn_name, fn_args, fn_where, fn_body
    elseif @capture(
        function_def, 
        function fn_name_ end
    )
        return fn_name, nothing, nothing, nothing
    elseif @capture(
        function_def, 
        fn_name_(fn_args__) = fn_body_
    )
        return fn_name, fn_args, nothing, fn_body
    elseif @capture(
        function_def, 
        fn_name_(fn_args__) where fn_where__ = fn_body_
    )
        return fn_name, fn_args, fn_where, fn_body
    else
        return nothing, nothing, nothing, nothing
    end
end

"""
    _restructure_function_def(fn_name, fn_args, fn_where, fn_body)

Inverse of [`_destructure_function_def`](@ref). If `fn_name` is `nothing`, we return
an empty `Expr`.
"""
function _restructure_function_def(fn_name, fn_args, fn_where, fn_body)
    if isnothing(fn_name)
        return :()  # invalid
    elseif isnothing(fn_args)
        return :(function $fn_name end)  # only way to define a nullary function
    # in the remaining cases also fn_body is not nothing
    elseif isnothing(fn_where)
        return quote
            function $fn_name($(fn_args...))
                $fn_body
            end
        end
    else
        return quote
          function $fn_name($(fn_args...)) where {$(fn_where...)}
                $fn_body
            end
        end 
    end
end


"""
    @replace pair expr

A macro to replace every occurance of `first(pair)` in `expr` by (the entries of)
`last(pair)`. 

# Example
```julia-repl
julia> @replace Int => Float64 f(x::Int) = x * x
f (generic function with 1 method)

julia> methods(f)
# 1 method for generic function "f" from Main:
 [1] f(x::Float64)

julia> @replace x => (a, b, c, d) x = 1
1

julia> (a, b, c, d)
(1, 1, 1, 1)
```
"""
macro replace(pair, expr)
    if @capture(pair, a_ => b_)
        esc(_replace(expr, a, _to_symbols(b)))
    else
         throw(ArgumentError(
            "The first argument of `@replace` should be an expression of the form A => B \
             or A => (B, C, ...)!"))
    end
end

_to_symbols(x::Symbol) = (x,)
_to_symbols(x::Expr) = x.args  # e.g. when x.head in (:tuple, :vect) 


function _replace(org_expr, org_symb, new_symbs)
    new_exprs = map(new_symbs) do ns
        MacroTools.postwalk(org_expr) do ex
            ex == org_symb ? ns : ex
        end
    end

    ret = reduce((e1, e2) -> :($e1; $e2), new_exprs)
    return ret
end

# For convenience, use e.g. GPU(x) = adapt(GPU, x)
@replace D => (CPU, GPU, Device) D(x) = adapt(D, x)
GPU{M}(x) where M <: CUDACore.AbstractMemory = adapt(GPU{M}, x)

"""
    @adapt_im_mutable [mutable_prefix, immutable_prefix,] adapt_structure_base_expr

A macro to create or convert an `Adapt.adapt_structure` definition for the non-existing base
type for a [`@im_mutable`](@ref) struct to versions for the (existing) mutable and immutable
structs.

Also adds an extra `CUDA.KernelAdaptor` version returning the immutable version for the
mutable version, as mutable structs are not allowed in GPU kernels.

# Arguments
- `mutable_prefix` and `immutable_prefix`: The same as in `@im_mutable`.
- `adapt_structure_base_expr`: Either a full `Adapt.adapt_structure` method definition for
    the base type, or simply the name of the base type. In the latter case there needs to
    exist a generic non-typed inner constructor taking in the fields, i.e.
    `S(fields...) [where ...] = new{...}(fields...)` (and not (only)
    `S{...}(fields...) [where ...] = new{...}(fields...)`) for both the mutable and
    immutable structs.

# Example
Continuing with the `@im_mutable` example:
```julia-repl
julia> @im_mutable M I A struct S{T, N, A <: AbstractArray{T, N}} <: AbstractArray{T, N}
    a::A

    function S()
        println("Common constructor (supplied name)")
        new{Int, 1, Vector{Int}}([1])
    end

    function MS(a::A) where {T, N, A <: AbstractArray{T, N}}
        println("Mutable constructor (mutable name)")
        new{T, N, A}(a)
    end

    function :immutable(a::A) where {T, N, A <: AbstractArray{T, N}}
        println("Immutable constructor (:immutable)")
        new{T, N, A}(a)
    end
end

julia> function Base.show(io::IO, mime::MIME"text/plain", x::AS)
           show(io, mime, typeof(x))
           print(":\n")
           show(io, mime, x.a)
       end

julia> m = MS([2.])
Mutable constructor (mutable name)
MS{Float64, 1, Vector{Float64}}:
1-element Vector{Float64}:
 2.0

julia> @adapt_im_mutable M I S

julia> adapt(GPU, m)
Mutable constructor (mutable name)
MS{Int64, 1, CuArray{Int64, 1, CUDACore.DeviceMemory}}:
1-element CuArray{Int64, 1, CUDACore.DeviceMemory}:
 1

julia> adapt(Device, m)  # First goes to GPU (mutable), then to Device (immutable) 
Mutable constructor (mutable name)
Immutable constructor (:immutable)
IS{Int64, 1, CuDeviceVector{Int64, 1}}:
1-element device array at 0x0000000204c00200
```

# Example
```
julia> abstract type Coolness end

julia> struct Amazing <: Coolness end; struct Boring <: Coolness end

julia> @im_mutable struct S{C <: Coolness, A <: AbstractVector}
    a::A

    function S(::Type{C}, a::A) where {C <: Coolness, A <: AbstractVector}
        new{C, A}(a)
    end
end

julia> @adapt_im_mutable Adapt.adapt_structure(to, s::S{C}) where C = S(C, adapt(to, s.a))

julia> adapt(Device, MutableS(Amazing, [1]))
ImmutableS{Amazing, CuDeviceVector{Int64, 1}}(1-element device array at 0x0000000204c00000)
```
"""
macro adapt_im_mutable(
    mutable_prefix, immutable_prefix,
    adapt_structure_base_expr
)
    esc(_adapt_im_mutable(
        mutable_prefix, immutable_prefix, 
        adapt_structure_base_expr))
end

macro adapt_im_mutable(
    adapt_structure_base_expr
)
    esc(_adapt_im_mutable(
        DEFAULT_MUTABLE_PREFIX, DEFAULT_IMMUTABLE_PREFIX, 
        adapt_structure_base_expr))
end

function _adapt_im_mutable(
    mutable_prefix, immutable_prefix,
    base_name::Symbol)
    
    # Turn the structs' base name into a full `Adapt.adapt_structure` declaration.
    _adapt_im_mutable(
        mutable_prefix, immutable_prefix,
        quote
            function Adapt.adapt_structure(to, x::$base_name)
                return $base_name(
                    (adapt(to, getfield(x, fieldname)) 
                    for fieldname in fieldnames($base_name))...
                )
            end
            # (Don't escape Adapt: refers to the one loaded by ThreeDGaussianSplatting)
        end
    )
end

function _adapt_im_mutable(
    mutable_prefix, immutable_prefix,
    adapt_structure_base_expr
)
    # adapt_structure_base_expr should (now) be an `adapt_structure` definition

    fn_name, fn_args, fn_where, fn_body = _destructure_function_def(
        adapt_structure_base_expr)
    if fn_name != :(Adapt.adapt_structure)
        throw(ArgumentError("The expression must include `Adapt.adapt_structure`!"))
    end

    # (From specific to general)
    if @capture(fn_args[2], x_::base_name_{type_params__})
    elseif @capture(fn_args[2], x_::base_name_)
        # If we would remove the first if (which is more specific than this one)
        # our base_name might be something like var"S{C}" (not var"S"{C}),
        # which will lead to problems in the _replace below.
    else
        throw(ArgumentError(
            "Could not find the base struct name in the second \
             `Adapt.adapt_structure` argument!"))
    end

    # Now base_name should be something non-parametric.
    # (type_params is declared but might be nothing.)
    mutable_name = Symbol(mutable_prefix, base_name)
    immutable_name = Symbol(immutable_prefix, base_name)

    return quote
        $(_restructure_function_def(
            fn_name, 
            [fn_args[1], 
                _replace(fn_args[2], base_name, (mutable_name,))],
            fn_where,
            _replace(fn_body, base_name, (mutable_name,))
        ))  # Mutable -> Mutable
        $(_restructure_function_def(
            fn_name, 
            [fn_args[1], 
                _replace(fn_args[2], base_name, (immutable_name,))],
            fn_where,
            _replace(fn_body, base_name, (immutable_name,))
        ))  # Immutable -> Immutable
        $(_restructure_function_def(
            fn_name, 
            [:(to::CUDA.KernelAdaptor), 
                _replace(fn_args[2], base_name, (mutable_name,))],
            fn_where,
            _replace(fn_body, base_name, (immutable_name,))
        ))  # KernelAdaptor: Mutable -> Immutable
    end
end

"""
    @im_mutable_with_adapt(
        [[mutable_prefix, immutable_prefix, abstract_prefix,]]
        struct_def,
        [adapt_structure_base_expr]
    )

A macro combining `@im_mutable` and `@adapt_im_mutable`.
"""
macro im_mutable_with_adapt(
    mutable_prefix, immutable_prefix, abstract_prefix,
    struct_def,
    adapt_structure_base_expr
)
    esc(_im_mutable_with_adapt(
        mutable_prefix, immutable_prefix, abstract_prefix,
        struct_def,
        adapt_structure_base_expr
    ))
end

macro im_mutable_with_adapt(
    struct_def,
    adapt_structure_base_expr
)
    esc(_im_mutable_with_adapt(
        DEFAULT_MUTABLE_PREFIX, DEFAULT_IMMUTABLE_PREFIX, DEFAULT_ABSTRACT_PREFIX,
        struct_def,
        adapt_structure_base_expr
    ))
end


macro im_mutable_with_adapt(
    struct_def
)
    esc(_im_mutable_with_adapt(
        DEFAULT_MUTABLE_PREFIX, DEFAULT_IMMUTABLE_PREFIX, DEFAULT_ABSTRACT_PREFIX,
        struct_def,
        _destructure_struct_def(struct_def)[1],
    ))
end

function _im_mutable_with_adapt(
    mutable_prefix, immutable_prefix, abstract_prefix,
    struct_def,
    adapt_structure_base_expr
)
    new_struct_defs = _im_mutable(
        mutable_prefix, immutable_prefix, abstract_prefix, struct_def)
    new_adapts = _adapt_im_mutable(
        mutable_prefix, immutable_prefix, adapt_structure_base_expr)
    return :($new_struct_defs; $new_adapts)
end