# Some special SArrays (mainly) SMatrices for transparent (hopefully) more efficient
# computations.

"""
    SpecialSArray{S, T, D} <: StaticArray{S, T, D}

Abstract type for our Special SArrays. By default (unless overridden) we will forward many
important methods to use `SArray(ssa)` for a `ssa::SpecialSArray`. The idea here is that
typically `getindex(::SpecialSArray, ...)` (which you must implement) will be slower than
`getindex(::SArray)`, but we might be able to create an efficient `@generated SArray`
conversion at compile-time. (If not, the default `SArray` conversion is used, which just
calls `getindex` anyway.)

Also broadcasting will be forwarded to `SArray`.

If a method is internal, i.e. when applied to a `SpecialSArray` you could get another
`SpecialSArray`, you must override it manually. For example, `transpose` for a
`SpecialSMatrix` is often internal.


# Parameters
- `S`: `size` tuple.
- `T`: `eltype`.
- `D`: number of dimensions (axes).
"""
abstract type SpecialSArray{S, T, D} <: StaticArray{S, T, D} end

const SpecialSVector{N, T} = SpecialSArray{Tuple{N}, T, 1}
const SpecialSMatrix{M, N, T} = SpecialSArray{Tuple{M, N}, T, 2}

######################
# Forward methods to SArray variant

@inline Base.collect(ssa::SpecialSArray) = collect(SArray(ssa))
@inline Base.collect(::Type{T}, ssa::SpecialSArray) where T = collect(T, SArray(ssa))
# The fact that functions can have multiple methods like this, make a metaprogramming
# approach using @eval difficult.
@inline Base.vec(ssa::SpecialSArray) = vec(SArray(ssa))
@inline Base.Array(ssa::SpecialSArray) = Array(SArray(ssa))
@inline Base.Matrix(ssa::SpecialSArray) = Matrix(SArray(ssa))
@inline Base.Vector(ssa::SpecialSArray) = Vector(SArray(ssa))

@inline Base.:(==)(A::SpecialSArray, B::AbstractArray) = SArray(A) == B
@inline Base.:(==)(A::AbstractArray, B::SpecialSArray) = A == SArray(B)
@inline Base.:(==)(A::SpecialSArray, B::SpecialSArray) = SArray(A) == SArray(B)
# Needed to avoid ambiguity, as both of the earlier methods would have applied 
@inline Base.isapprox(A::SpecialSArray, B::AbstractArray; kwargs...) = 
    isapprox(SArray(A), B; kwargs...)
# kwargs do not count for multiple dispatch, so this is fine (but args... would not be)
@inline Base.isapprox(A::AbstractArray, B::SpecialSArray; kwargs...) = 
    isapprox(A, SArray(B); kwargs...)
@inline Base.isapprox(A::SpecialSArray, B::SpecialSArray; kwargs...) = 
    isapprox(SArray(A), SArray(B); kwargs...)


@inline Base.:+(A::SpecialSArray) = +(SArray(A))
@inline Base.:+(A::SpecialSArray, B::AbstractArray) = SArray(A) + B
@inline Base.:+(A::SpecialSArray, B::StaticArray) = SArray(A) + B
# Needed to avoid ambiguity, as ::SpecialSArray + ::StaticArray could use
# ::SpecialSArray + ::AbstractArray or ::StaticArray + ::StaticArray
@inline Base.:+(A::SpecialSArray, b::Number) = SArray(A) + b
@inline Base.:+(A::AbstractArray, B::SpecialSArray) = A + SArray(B)
@inline Base.:+(A::StaticArray, B::SpecialSArray) = A + SArray(B)
@inline Base.:+(a::Number, B::SpecialSArray) = a + SArray(B)
@inline Base.:+(A::SpecialSArray, B::SpecialSArray) = SArray(A) + SArray(B)

@inline Base.:-(A::SpecialSArray) = -(SArray(A))
@inline Base.:-(A::SpecialSArray, B::AbstractArray) = SArray(A) - B
@inline Base.:-(A::SpecialSArray, B::StaticArray) = SArray(A) - B
@inline Base.:-(A::SpecialSArray, b::Number) = SArray(A) - b
@inline Base.:-(A::AbstractArray, B::SpecialSArray) = A - SArray(B)
@inline Base.:-(A::StaticArray, B::SpecialSArray) = A - SArray(B)
@inline Base.:-(a::Number, B::SpecialSArray) = a - SArray(B)
@inline Base.:-(A::SpecialSArray, B::SpecialSArray) = SArray(A) - SArray(B)

@inline Base.:*(A::SpecialSArray, b::Number) = SArray(A) * b
@inline Base.:*(a::Number, B::SpecialSArray) = a * SArray(B)
@inline Base.:/(A::SpecialSArray, b::Number) = SArray(A) / b
@inline Base.:\(a::Number, B::SpecialSArray) = a \ SArray(B)

@inline Base.muladd(s::Number, A::SpecialSArray, B::StaticArray) = muladd(s, SArray(A), B)
@inline Base.muladd(s::Number, A::StaticArray, B::SpecialSArray) = muladd(s, A, SArray(B))
@inline Base.muladd(s::Number, A::SpecialSArray, B::SpecialSArray) = 
    muladd(s, SArray(A), SArray(B))
@inline Base.muladd(A::SpecialSArray, s::Number, B::StaticArray) = muladd(SArray(A), s, B)
@inline Base.muladd(A::StaticArray, s::Number, B::SpecialSArray) = muladd(A, s, SArray(B))

@inline Base.FastMath.mul_fast(a::Number, B::SpecialSArray) = 
    Base.FasthMath.mul_fast(a, SArray(B))
@inline Base.FastMath.mul_fast(A::SpecialSArray, b::Number) = 
    Base.FasthMath.mul_fast(SArray(A), b)

@inline Base.FastMath.add_fast(A::SpecialSArray, B::StaticArray) = 
    Base.FastMath.add_fast(SArray(A), B)
@inline Base.FastMath.add_fast(A::StaticArray, B::SpecialSArray) = 
    Base.FastMath.add_fast(A, SArray(B))
@inline Base.FastMath.add_fast(A::SpecialSArray, B::SpecialSArray) = 
    Base.FastMath.add_fast(SArray(A), SArray(B))
@inline Base.FastMath.sub_fast(A::SpecialSArray, B::StaticArray) = 
    Base.FastMath.sub_fast(SArray(A), B)
@inline Base.FastMath.sub_fast(A::StaticArray, B::SpecialSArray) = 
    Base.FastMath.sub_fast(A, SArray(B))
@inline Base.FastMath.sub_fast(A::SpecialSArray, B::SpecialSArray) = 
    Base.FastMath.sub_fast(SArray(A), SArray(B))

@inline LinearAlgebra.transpose(ssa::SpecialSArray) = transpose(SArray(ssa))
@inline LinearAlgebra.adjoint(ssa::SpecialSArray) = adjoint(SArray(ssa))
@inline LinearAlgebra.det(ssa::SpecialSArray) = det(SArray(ssa))
@inline LinearAlgebra.tr(ssa::SpecialSArray) = tr(SArray(ssa))
@inline Base.inv(ssa::SpecialSArray) = inv(SArray(ssa))
@inline LinearAlgebra.norm(ssa::SpecialSArray, p::Real = 2) = norm(SArray(ssa), p)
@inline LinearAlgebra.opnorm(ssa::SpecialSArray, p::Real = 2) = opnorm(SArray(ssa), p)
@inline LinearAlgebra.diag(ssa::SpecialSArray, n::Integer = 0) = diag(SArray(ssa), n)
@inline LinearAlgebra.eigen(ssa::SpecialSArray; kwargs...) = eigen(SArray(ssa); kwargs...)
# We do not explicitly support the two argument version (neither does StaticArrays)
@inline LinearAlgebra.eigvals(ssa::SpecialSArray; kwargs...) = 
    eigvals(SArray(ssa); kwargs...)
# Only supported for hermitian StaticMatrices, but we might just be one
# The multiple argument versions are either not supported by StaticArrays, or too
# complicated to attempt to recreate their signature
@inline LinearAlgebra.eigvecs(ssa::SpecialSArray; kwargs...) = 
    eigvecs(SArray(ssa); kwargs...)

@inline LinearAlgebra.cross(a::SpecialSArray, b::StaticVector) = cross(SArray(a), b)
# Normally a will of course be a SpecialSVector, but there's no harm in being overly general
# We need to keep the StaticVector though, to be more specific than the StaticArrays method
@inline LinearAlgebra.cross(a::StaticVector, b::SpecialSArray) = cross(a, SArray(b))
@inline LinearAlgebra.cross(a::SpecialSArray, b::SpecialSArray) = 
    cross(SArray(a), SArray(b))
@inline LinearAlgebra.dot(a::SpecialSArray, b::StaticArray) = dot(SArray(a), b)
@inline LinearAlgebra.dot(a::StaticArray, b::SpecialSArray) = dot(a, SArray(b))
@inline LinearAlgebra.dot(a::SpecialSArray, b::SpecialSArray) = dot(SArray(a), SArray(b))

# kron is defined for different ndims. We forward via generic ndims method _kron.
# To be more specific than the StaticArray methods at first we will need to specify the
# ndims 
@inline LinearAlgebra.kron(a::SpecialSArray, b::StaticVector) = _kron(a, b)
@inline LinearAlgebra.kron(a::StaticVector, b::SpecialSArray) = _kron(a, b)
@inline LinearAlgebra.kron(a::SpecialSArray, b::StaticMatrix) = _kron(a, b)
@inline LinearAlgebra.kron(a::StaticMatrix, b::SpecialSArray) = _kron(a, b)
@inline LinearAlgebra.kron(a::SpecialSArray, b::SpecialSArray) = _kron(a, b)

_kron(a::StaticArray, b::StaticArray) = kron(a, b)
_kron(a::SpecialSArray, b::StaticArray) = kron(SArray(a), b)
_kron(a::StaticArray, b::SpecialSArray) = kron(a, SArray(b))
_kron(a::SpecialSArray, b::SpecialSArray) = kron(SArray(a), SArray(b))
# But for specific subtypes of SpecialSArray we can now just overload _kron (regardless of
# ndims)

@inline LinearAlgebra.triu(ssa::SpecialSArray, k::Int = 0) = triu(SArray(ssa), k)
@inline LinearAlgebra.tril(ssa::SpecialSArray, k::Int = 0) = tril(SArray(ssa), k)

# Here we do need to specify Matrix, since the *-methods for StaticArrays are only
# defined for matrices. Similar as for kron we use an in-between method.
@inline Base.:*(
    A::SpecialSMatrix{P, Q}, B::SpecialSMatrix{Q, R}
) where {P, Q, R} = matmul(A, B)
@inline Base.:*(
    A::SpecialSMatrix{P, Q}, B::StaticMatrix{Q, R}
) where {P, Q, R} = matmul(A, B)
@inline Base.:*(
    A::StaticMatrix{P, Q}, B::SpecialSMatrix{Q, R}
) where {P, Q, R} = matmul(A, B)
@inline Base.:*(
    A::SpecialSMatrix{P, Q}, B::SpecialSVector{Q}
) where {P, Q} = matmul(A, B)
@inline Base.:*(
    A::SpecialSMatrix{P, Q}, B::StaticVector{Q}
) where {P, Q} = matmul(A, B)
@inline Base.:*(
    A::StaticMatrix{P, Q}, B::SpecialSVector{Q}
) where {P, Q} = matmul(A, B)
# We get ::SpecialSMatrix ^ ::Int (i.e. 
#   literal_pow(::typeof(^), ::SpecialMatrix, ::Val{p}) where p
# ) for free

"""
     matmul(
        A::StaticArray, B::StaticArray, 
        ::Val{use_adjoint_A} = Val(false), ::Val{use_adjoint_B} = Val(false)
    ) where {use_adjoint_A, use_adjoint_B}

Compute the matrix-matrix or matrix-vector product `A * B` or variants with adjoints, like
`A' * B` (`use_adjoint_A && !use_adjoint_B`).
"""
@inline matmul(A::StaticArray, B::StaticArray) = A * B
@inline matmul(A::SpecialSArray, B::StaticArray) = SArray(A) * B
@inline matmul(A::StaticArray, B::SpecialSArray) = A * SArray(B)
@inline matmul(A::SpecialSArray, B::SpecialSArray) = SArray(A) * SArray(B)
# Size-generic method. Can be overridden by concrete SpecialSArrays (or abstract subtypes)

# (We don't include e.g. show because this is inherently slow)
# We might have missed a few

# Forward methods to SArray variant
######################
# Broadcast via forwarding

abstract type AbstractSpecialSArrayStyle{D} <: Broadcast.AbstractArrayStyle{D} end

# Our style takes priority

Base.BroadcastStyle(
    ::AbstractSpecialSArrayStyle{M}, 
    ::Broadcast.AbstractArrayStyle{N}
) where {M, N} = DefaultSpecialSArrayStyle{max(M, N)}()
# Needed for e.g. .+ Diagonal([1, 2])

# Note: these rules are automatically reflexive and symmetric

# DefaultArrayStyle has priority over AbstractArrayStyle, so we have to announce we have
# priority over it
Base.BroadcastStyle(
    ::AbstractSpecialSArrayStyle{M}, 
    ::Base.Broadcast.DefaultArrayStyle{N}
) where {M, N} = DefaultSpecialSArrayStyle{max(M, N)}()
# Needed for e.g. .+ 1 or .+ ::Array

# Note: these rules are not automatically transitive, so we still need the first one

# Similar for StaticArrayStyle
Base.BroadcastStyle(
    ::AbstractSpecialSArrayStyle{M}, 
    ::StaticArrays.StaticArrayStyle{N}
) where {M, N} = DefaultSpecialSArrayStyle{max(M, N)}()
# Needed for e.g. .+ ::SArray

Base.copy(bc::Broadcast.Broadcasted{<:AbstractSpecialSArrayStyle}) = 
    Base.copy(_rewrite_bc_special_sarray(bc))

"""
    _rewrite_bc_special_sarray(x)

Rewrite (the contents of) a `Broadcast.Broadcasted`, replacing appearances of
`SpecialSArray`s by `SArray` conversions thereof.
"""
_rewrite_bc_special_sarray(ssa::SpecialSArray) = SArray(ssa)
_rewrite_bc_special_sarray(x) = x
_rewrite_bc_special_sarray(bc::Broadcast.Broadcasted) = 
    Broadcast.broadcasted(bc.f, map(_rewrite_bc_special_sarray, bc.args)...)

struct DefaultSpecialSArrayStyle{D} <: AbstractSpecialSArrayStyle{D} end
# Promotion constructors
DefaultSpecialSArrayStyle(::Val{D}) where D = DefaultSpecialSArrayStyle{D}()
DefaultSpecialSArrayStyle{D}(::Val{N}) where {D, N} = DefaultSpecialSArrayStyle{max(D, N)}()
Base.BroadcastStyle(::Type{<:SpecialSArray{<:Any, <:Any, D}}) where D = 
    DefaultSpecialSArrayStyle{D}()


### SpecialSArray
############################################################################################
### SpecialIndexingSArray

"""
    SpecialIndexingSArray{S, T, D} <: SpecialSArray{S, T, D}

Abstract type for special SArrays, which have a different (smaller) underlying storage, but
act as a normal SArray via a custom `getindex`. Examples include arrays with known zero
entries, or symmetric matrices. To avoid the overhead of the `getindex`, for supported
methods like matrix multiplication (when `D == 2`) we will unroll loops and rewrite the
indexing at compile-time.
"""
abstract type SpecialIndexingSArray{S, T, D} <: SpecialSArray{S, T, D} end


"""
    SpecialIndexingSMatrix{M, N, T}

2D `SpecialIndexingSArray` of static size `Tuple{M, N}`.
"""
const SpecialIndexingSMatrix{M, N, T} = SpecialIndexingSArray{Tuple{M, N}, T, 2}

"""
    SpecialIndexingSVector{N, T}

1D `SpecialIndexingSArray` of static size `Tuple{N}`.
"""
const SpecialIndexingSVector{N, T} = SpecialIndexingSArray{Tuple{N}, T, 1}

######################
# Indexing and SArray


"""
    flatten_blocks(expr)

Recursively flatten an expression containing nested blocks into a single block expression.
Also removes empty blocks. Has no effect if `expr` is not a block.

This is particularly useful for when expressions are combined using `quote ... end` and
interpolation. For example, `quote \$ex1; \$ex2 end` results in a block containing two
subblocks.
"""
function flatten_blocks(expr)
    expr isa Expr && expr.head == :block || return expr

    new_expr = Expr(:block)
    for a in expr.args
        if !(a isa Expr)
            push!(new_expr.args, a)
        elseif a.head == :block
            append!(new_expr.args, flatten_blocks(a).args)
        elseif a.head == :tuple && isempty(a.args)  # do nothing: remove
        else
            push!(new_expr.args, a)
        end
    end
    return new_expr
end

"""
    join_exprs(exprs)

Combine a collections of expressions into a single block expression. If a single expression
is provided (also when not as a singleton) we just return it.
"""
function join_exprs(exprs)
    return flatten_blocks(_join_exprs(exprs))
end

"""
    _join_exprs(exprs)

Internal helper method. Like [`join_exprs`](@ref), but the output will contain nested
blocks.
"""
function _join_exprs(exprs)
    if exprs isa Expr
        return exprs
    elseif length(exprs) == 1
        return only(exprs)
    else
        return :($(first(exprs)); $(_join_exprs(exprs[2:end])))
    end
    # Will error on an empty collection, but that's fine
end

"""
    adjust_indexing_expression(
        split_expr, ::Type{data_type}, data_symbol
    ) where data_type

Recursively rewrite the indexing operations in `expr` for the instance of the `data_type`
represented therein using `data_symbol`, to directly use the actual underlying
representation.

The expression is split into a head (variable declarations (new and ones inherited from the
input)) and the body. (When traversing recursively from top-level to low-level we need to
append to the head.) We also return a split expression, i.e. a `Tuple`. 

For the base case (in particular for `NTuple`) we just return `split_expr`.
"""
adjust_indexing_expression(split_expr, _, _) = split_expr


"""
    adjust_indexing_expression(
        split_expr, ::Type{<:SArray{S}}, sa_symb
    ) where S

Replace the appearance of `sa_symb`'s (linear of multi-dimensional) indexing (where
`sa_symb` represents an `SArray`) in `expr_body` (i.e. `split_expr[2]`) by directly
(linearly) indexing the underlying `NTuple`.

!!! The indexing expressions in `expr_body` cannot contain variables. That is, e.g.
`M[k, 1]` is not allowed. 
"""
function adjust_indexing_expression(
    split_expr, ::Type{<:SArray{S}}, sa_symb
) where S
    expr_head, expr_body = split_expr
    data_tuple_symb = gensym(Symbol(sa_symb, :_data_tuple))
    expr_head = join_exprs((
        expr_head, 
        :($data_tuple_symb = Tuple($sa_symb))  # i.e. $sa_symb.data
    ))
    # (Could use 
    #       expr_head = quote
    #           $expr_head
    #           $data_tuple_symb = Tuple($sa_symb)  # i.e. $sa_symb.data
    #       end
    # which will create nested blocks, as eventually we will join the head and body using
    # join_exprs. But using cleaner intermediate expressions make everythign more legible
    # for debugging)
    expr_body = MacroTools.postwalk(expr_body) do ex
        if @capture(ex, a_[I__]) && a === sa_symb
            # I isa Vector{Int} (but stored as a Vector{Any})
            :($data_tuple_symb[$(LinearIndices(fieldtypes(S))[Tuple(I)...])])
            # (Also fine when length(I) == 1, as we can linearly index LinearIndices.)
        else
            ex
        end
    end
    # No need to recursively call our method, as for NTuple we just return the expr
    return expr_head, expr_body
end


"""
    adjust_indexing_expression(
        split_expr, ts::Tuple{Type, Symbol}...;
        finalize = false
    )

Iteratively apply the base 3-argument [`adjust_indexing_expression`](@ref) on `split_expr`
for all provided `Tuple` pairs of a `data_type` and `data_symbol`.

When `finalize` we combine the resulting split expression and process literal zeros
via [`adjust_zeros_expression`](@ref). (This step is mainly relevant for `SArrayWithZeros`.)

The produced expression is then ready to be used in a (`@inline`) `@generated` function.
"""
function adjust_indexing_expression(
    split_expr, ts::Tuple{Type, Symbol}...;
    finalize = false
)
    for (data_type, data_symbol) in ts
        split_expr = adjust_indexing_expression(split_expr, data_type, data_symbol)
    end
    if finalize
        return adjust_zeros_expression(join_exprs(split_expr))
    else
       return split_expr
    end
end

const REWRITE_EXPRESSION_ZERO_TYPES = (Int16, Int32, Int64, UInt16, UInt32, UInt64, Float16, Float32, Float64)

"""
    adjust_zeros_expression(expr)

Rewrite the `expr`, simplifying `+`, `-`, `*`, `/`, `\\`, `div`/`÷` and `muladd` when
zeros are involved. We only check for zeros of the following types: 
$(join(REWRITE_EXPRESSION_ZERO_TYPES, ", ", " and ")).

# Example
```
julia> adjust_zeros_expression(:(
    muladd(a, muladd(-0, b, c), d + muladd(0, e, 0) + f * g * div(0 * h, i))
))
:(muladd(a, c, d))
```
"""
function adjust_zeros_expression(expr)
    zeros = zero.(REWRITE_EXPRESSION_ZERO_TYPES)
    MacroTools.postwalk(expr) do ex
        if @capture(ex, muladd(a_, b_, c_))
            if a in zeros || b in zeros
                return c
            elseif c in zeros
                return :($a * $b)
            else
                return ex
            end

        elseif @capture(ex, +(as__))
            # (Does indeed capture a + b, a + b + c, etc.)
            filtered_as = filter(!in(zeros), as)  # Non-zero terms
            if isempty(filtered_as)
                # Only zero terms
                return first(as)
                # A zero of the appropriate type
            elseif length(filtered_as) == 1
                return only(filtered_as)
                # Avoid +(a)
            else
                return :(+($(filtered_as...)))
                # (E.g. a + 0 + c --> a + c) 
            end

        elseif @capture(ex, -a_)
            # In particular, captures -0 (not in zeros) and canonizes it to 0 (in zeros)
            if a in zeros
                return a
            else
                ex
            end

        elseif @capture(ex, a_ - b_)
            if a in zeros
                return :(-$b)
            elseif b in zeros
                return a
            else
                return ex
            end

        # (- is not ternary or higher)

        elseif @capture(ex, *(as__))
            idx = findfirst(in(zeros), as)
            if isnothing(idx)
                return ex
            else
                return as[idx]
                # A zero of the appropriate type
            end

        elseif @capture(ex, a_ / b_) || @capture(ex, div(a_, b_)) || @capture(ex, a_ ÷ b_)
            # Only a binary operation
            if a in zeros
                return a
            end

        elseif @capture(ex, a_ \ b_)
            if b in zeros
                return b
            end

        else
            return ex
        end
    end        
end


"""
    sarray_generation_expression_array(sisa_symb, S)

Create an `Array{Expr}` of `size` `S` containing all (multi-dimensional) indexing
expressions for array symbol `sisa_symb`.
"""
function sarray_generation_expression_array(sisa_symb, S)
    return [
        :($sisa_symb[$(Tuple(I)...)]) for I in CartesianIndices(fieldtypes(S))]
end


"""
    sarray_generation_expression(sa_symb, S, T)

Create the expression which would initialize an `SArray{S, T}` corresponding to a
`StaticArray` (in particular a `SpecialIndexingSArray`) represented by `sa_symb`. This
expression uses standard (multi-dimensional) array indexing and includes `@inbounds`.

The output expression is split into a(n empty) head (`:()`) and body (the actual `SArray`
call). See also [`adjust_indexing_expression`](@ref).

For example, `sarray_generation_expression(:M, Tuple{2, 3}, Float32)` returns
`(:(), 
  :(@inbounds SArray{Tuple{2, 3}, Float32}(M[1, 1], M[2, 1], M[1, 2], ..., M[2, 3]))
)`.
"""
function sarray_generation_expression(sa_symb, S, T)
    out_array_exprs = sarray_generation_expression_array(sa_symb, S)
    return :(), :(
        @inbounds SArray{$S, $T}($(out_array_exprs...))
    )
end
# (Note: no dispatch on the type of the sisa, like in smatrix_multiplcation_expression,
# since the output in the expression is fixed as an SArray and there is no reason to
# construct an SArray in any other way than by listing the entires (as opposed to e.g. 
# in unrolled_smatrix_determinant_expression, where different types might use different
# determinant calculations))

  
"""
    SArray(sisa::SpecialIndexingSArray)

Create the normal `SArray` equivalent of this `SpecialIndexingSArray`.
"""
@inline @generated StaticArrays.SArray(sisa::SpecialIndexingSArray) = _sarray_code(
    sisa, :sisa)
# (Note: in contrast to earlier only SpecialIndexingSArray, not StaticArray, as we don't
# want type piracy)

_sarray_code(A::Type{<:StaticArray{S, T}}, sa_symb) where {S, T} = 
    adjust_indexing_expression(
        sarray_generation_expression(sa_symb, S, T),
        (A, sa_symb);
        finalize = true
    )
# Having this as a separate function, rather than just writing it in the @generated SArray
# method makes debugging easier, as you can now easily inspect the generated code directly.


@inline StaticArrays.SMatrix(sism::SpecialIndexingSMatrix) = SArray(sism)
@inline StaticArrays.SVector(sisv::SpecialIndexingSVector) = SArray(sisv)

@inline Base.convert(::Type{SArray}, sisa::SpecialIndexingSArray) = SArray(sisa)
@inline Base.convert(::Type{SMatrix}, sism::SpecialIndexingSMatrix) = SMatrix(sism)
@inline Base.convert(::Type{SVector}, sisv::SpecialIndexingSVector) = SMatrix(sisv)

Base.@propagate_inbounds Base.getindex(sisa::SpecialIndexingSArray, inds::Int...) = 
    SArray(sisa)[inds...]
# (Note that generically SArray(...) would be defined using getindex, while we define
#  getindex using SArray(...))

# Indexing and SArray 
######################
# Adjoint and transpose

"""
    adjoint_or_transposed_smatrix_generation_expressions_matrix(
        sisa_symb, S, use_transpose
    )

Create expressions for the entries of the adjoint or transpose of the
`SpecialIndexingSArray{S}`, which will be denoted by `sisa_symb`. This is only possible
in case of a matrix or a vector; an error will be thrown otherwise. In both cases we want a
matrix output (possible `1 x N`), so we return a `Matrix{Expr}`. In case of an input matrix
of `size` `(M, N)` this output will have `size` `(N, M)` and its entries use standard 2D
matrix indexing. In case of a vector we will get `(1, N)` with 1D indexing.

For example, 
`adjoint_or_transposed_smatrix_generation_expressions_matrix(
    :M, Tuple{2, 3}, Float32, true)[2, 1]`
is `:(M[1, 2])`.

With `!use_transpose` (i.e. use_adjoint) we would have `(M[j, i])'` (e.g. complex
conjugation) everywhere instead.
"""
function adjoint_or_transposed_smatrix_generation_expressions_matrix(
    sisa_symb, S, use_transpose
)
    sz = fieldtypes(S)
    if length(sz) >= 3 || length(sz) == 0
        throw(ArgumentError(
            "Taking the adjoint or transpose is only supported when `ndims in (1, 2)`!
             Received $(length(sz))"))
    end

    if use_transpose
        if length(sz) == 2
            return [
                :($sisa_symb[$(reverse(Tuple(I))...)]) 
                for I in CartesianIndices((sz[2], sz[1]))
            ]
        else
            return [:($sisa_symb[$i]) for _ = 1:1, i = 1:sz[1]]
            # _ to get a Matrix
        end
    else
        if length(sz) == 2
            return [
                :(($sisa_symb[$(reverse(Tuple(I))...)])') 
                for I in CartesianIndices((sz[2], sz[1]))
            ]
        else
            return [:(($sisa_symb[$i])') for _ = 1:1, i = 1:sz[1]]
            
        end
    end
end

"""
    adjoint_or_transposed_smatrix_generation_expression(
        ::Type{<:StaticArray{S, T}}, sa_symb, use_transpose
    ) where {S, T}

Create the expression which would initialize an `SMatrix{N, M, T}` corresponding to the
adjoint or transpose of our `StaticArray` subtype (in particular `SpecialIndexingSArray`).
This is only supported for `StaticMatrix{M, N}` or `StaticVector{N}`. The latter is
considered to be a matrix of size `Tuple{N, 1}`.

The output expression is split into a(n empty) head (`:()`) and body (the actual `SMatrix`
call). See also [`adjust_indexing_expression`](@ref).

For example, 
`adjoint_or_transposed_smatrix_generation_expression(SMatrix{2, 3, Float32}, :M, true)`
returns
`(:(), 
  :(@inbounds SMatrix{3, 2, Float32}(M[1, 1], M[1, 2], M[1, 3], M[2, 1], M[2, 2], M[2, 3]))
)`.
With `!use_transpose && !(T <: Real)` we would have `(M[j, i])'` everywhere
(e.g. complex conjugation) instead. (The `T <: Real` exception is just to produce a cleaner
expression. Indeed, for a `r::Real` we have `r' == adjoint(r) = r`.)
"""
function adjoint_or_transposed_smatrix_generation_expression(
    ::Type{<:StaticArray{S, T}}, sa_symb, use_transpose
) where {S, T}
    adjoint_or_transposed_matrix_exprs = 
        adjoint_or_transposed_smatrix_generation_expressions_matrix(
            sa_symb, S, use_transpose || T <: Real)
    return :(), :(
        @inbounds SMatrix{$(size(adjoint_or_transposed_matrix_exprs)...), $T}(
            $(adjoint_or_transposed_matrix_exprs...))
    )
end


"""
    adjoint_or_transpose(
        sa::StaticArray, ::Val{use_transpose}
    ) where use_transpose

Compute either the adjoint (when `!use_transpose`) or transpose (when `use_transpose`) of
`sism`.
"""
@inline @generated adjoint_or_transpose(
    sa::StaticArray, ::Val{use_transpose}
) where use_transpose = _adjoint_or_transpose_code(sa, :sa, use_transpose)

_adjoint_or_transpose_code(
    A::Type{<:StaticArray}, sa_symb, use_transpose
) = adjust_indexing_expression(
    adjoint_or_transposed_smatrix_generation_expression(
        A, sa_symb, use_transpose
    ),
    (A, sa_symb);
    finalize = true
)

@inline LinearAlgebra.adjoint(sisa::SpecialIndexingSArray) = 
    adjoint_or_transpose(sisa, Val(false))
# Note that this returns an SMatrix. For concrete SpecialIndexingSMatrices we can often
# override this to return a matrix of the same type.
@inline LinearAlgebra.transpose(sisa::SpecialIndexingSArray) = 
    adjoint_or_transpose(sisa, Val(true))


# Transpose and adjoint
######################
# Matrix multiplcation

"""
    unrolled_smatrix_multiplication_expressions_matrix(
        A_symb, B_symb, SA, SB, 
        use_adjoint_or_transposed_A = false, use_transpose_A = false,
        use_adjoint_or_transposed_B = false, use_transpose_B = false
    )

Create expressions for the entries of the product of arrays represented as `A_symb` and
`B_symb`, and of sizes `SA` and `SB` (e.g. `Tuple{2, 3}`). When the sizes are not compatible
with matrix multiplication we will throw an error. When they are, optionally (depending on
`use_adjoint_or_transposed_⋅`) we can take the adjoint (when `!use_transpose_⋅`) or transpose
(when `use_transpose_⋅`) of the matrices, in particular reversing the order of indices.

The output is a `Matrix{Expr}`, or a `Vector{Expr}` in case of a matrix-vector product.
We do not include `@inbounds`.

For example, 
`unrolled_smatrix_multiplication_expressions_matrix(:A, :B, Tuple{2, 2}, Tuple{2, 3})[1, 3]`
is
`:(muladd(A[1, 2], B[2, 3], A[1, 1] * B[1, 3]))`.
"""
function unrolled_smatrix_multiplication_expressions_matrix(
    A_symb, B_symb, SA, SB,
    use_adjoint_or_transposed_A = false, use_transpose_A = false,
    use_adjoint_or_transposed_B = false, use_transpose_B = false
)

    function _normalize_size(S)
    # Checks and normalizes the number of dimensions. In particular, convert vectors into
    # column matrices.
        sz = fieldtypes(S)
        if length(sz) >= 3 
            throw(ArgumentError(
                "In matrix multiplication the arguments must at most two axes! \
                Received: $(length(sz))"))
        elseif length(sz) == 1
            sz = (only(sz), 1)
        end
        return sz
    end
    size_A = _normalize_size(SA)
    size_aA = use_adjoint_or_transposed_A ? reverse(size_A) : size_A
    # Size of the actual first input of the matrix multiplication (aA is either A'/A^T or A)
    
    size_B = _normalize_size(SB)
    vector_out = fieldcount(SB) == 1 && !use_adjoint_or_transposed_B
    # matrix * column vector -> column vector
    size_aB = use_adjoint_or_transposed_B ? reverse(size_B) : size_B
    if size_aA[2] != size_aB[1]
        throw(DimensionMismatch(
            "In matrix multiplication the number of columns of the first matrix must equal \
             the number of rows of the second! Received: $(size_aA[2]) and $(size_aB[1])"))
    end

    function _get_a_exprs(symb, S, use_adjoint_or_transposed, use_transpose)
        if use_adjoint_or_transposed
            return adjoint_or_transposed_smatrix_generation_expressions_matrix(
                symb, S, use_transpose)
        else
            return sarray_generation_expression_array(symb, S)
        end
    end
    aA_exprs = _get_a_exprs(A_symb, SA, use_adjoint_or_transposed_A, use_transpose_A)
    # Contains at [i, j] the aA expression producing aA[i, j]
    # (i.e. A[i, j], A[j, i], or A[j, i]')
    aB_exprs = _get_a_exprs(B_symb, SB, use_adjoint_or_transposed_B, use_transpose_B)
    # This might be a vector using 1D indices. We can still index it with aB[k, 1]
    # which is identical to aB[k] and contains a valid 1D vector indexing expression
    out_matrix_exprs = Matrix{Expr}(undef, size_aA[1], size_aB[2])
    for j = 1:size_aB[2]
        for i = 1:size_aA[1]
            out_expr = :($(aA_exprs[i, 1]) * $(aB_exprs[1, j]))
            for k = 2:size_aA[2]
                out_expr = :(muladd($(aA_exprs[i, k]), $(aB_exprs[k, j]), $out_expr))  
            end
            out_matrix_exprs[i, j] = out_expr
        end
    end
    if vector_out
        return vec(out_matrix_exprs)
    else
        return out_matrix_exprs
    end
end

"""
    unrolled_smatrix_multiplication_expression(
        ::Type{<:StaticArray{SA}}, ::Type{<:StaticArray{SB}}, A_symb, B_symb, T,
        use_adjoint_or_transposed_A = false, use_transpose_A = false,
        use_adjoint_or_transposed_B = false, use_transpose_B = false
    ) where {SA, SB}

Create the expression for matrix multiplying two arrays of (`Tuple{...}`) sizes `SA` and
`SB`, if possible (when they are actually matrices with matching dimensions). The matrices
will be represented in the output expression as `A_symb` and `B_symb`. The created `SMatrix`
in the output will have `eltype` `T`. In case of a matrix-vector product we output an
`SVector`. When `use_adjoint_or_transposed_⋅.`, we will take the adjoint (`!use_transpose_⋅`)
or tranpose (`use_transpose_⋅`) of the corresponding matrix in the output expression, to
generate e.g. `A * B'` when 
`use_adjoint_or_transposed_A && use_adjoint_or_transposed_B && !use_transpose_N`. If
`T <: Real` and we request adjoints, we will use the equivalent but slightly cleaner
transposes instead.

Like in [`sarray_generation_expression`](@ref) and [`adjust_indexing_expression`](@ref) we
split the return expression into a(n empty) head (`:()`) and the body containing the actual
matrix multiplication expression. The latter uses
[`unrolled_smatrix_multiplication_expressions_matrix`](@ref). Now we include a single
`@inbounds`.

For example, 
`unrolled_smatrix_multiplication_expression(
    SMatrix{2, 2}, SMatrix{2, 3}, :A, :B, Float32)` 
returns
`(:(), 
  :(@inbounds SMatrix{2, 3, Float32}(
        muladd(A[1, 2], B[2, 1], A[1, 1] * B[1, 1]),  # (A * B)[1, 1]
        ...  # [2, 1], [1, 2], [2, 2], [1, 3], [2, 3] 
   )
)`.
"""
function unrolled_smatrix_multiplication_expression(
    ::Type{<:StaticArray{SA}}, ::Type{<:StaticArray{SB}}, A_symb, B_symb, T,
    use_adjoint_or_transposed_A = false, use_transpose_A = false,
    use_adjoint_or_transposed_B = false, use_transpose_B = false
) where {SA, SB}
    out_matrix_exprs = unrolled_smatrix_multiplication_expressions_matrix(
        A_symb, B_symb, SA, SB, 
        use_adjoint_or_transposed_A, use_transpose_A || T <: Real,
        use_adjoint_or_transposed_B, use_transpose_B || T <: Real)
    out_type_expr = if ndims(out_matrix_exprs) == 1 
        :(SVector{$(size(out_matrix_exprs)...), $T})
    else
        :(SMatrix{$(size(out_matrix_exprs)...), $T})
    end
    return :(), :(
        @inbounds $out_type_expr($(out_matrix_exprs...))
    )
end


"""  
    unrolled_matmul(
        A::StaticArray{SA, TA}, B::StaticArray{SB, TB}, 
        ::Val{use_adjoint_or_transposed_A} = Val(false), 
        ::Val{use_transpose_A} = Val(false),
        ::Val{use_adjoint_or_transposed_B} = Val(false), 
        ::Val{use_transpose_B} = Val(false),
    ) where {
        SA, SB, TA, TB, 
        use_adjoint_or_transposed_A, use_transpose_A,
        use_adjoint_or_transposed_B, use_transpose_B
    }

Compute the matrix-matrix or matrix-vector product `A * B`, or variants with adjoints or
transposes (e.g. `A * B'`), by unrolling the loops and rewriting the indexing operations.

The `use_transpose_⋅` arguments are only relevant when `use_adjoint_or_transposed_⋅`,
in which case they determine whether we use the transpose or adjoint.
"""
@inline @generated unrolled_matmul(
    A::StaticArray{SA, TA}, B::StaticArray{SB, TB}, 
    ::Val{use_adjoint_or_transposed_A} = Val(false), ::Val{use_transpose_A} = Val(false),
    ::Val{use_adjoint_or_transposed_B} = Val(false), ::Val{use_transpose_B} = Val(false)
) where {
    SA, SB, TA, TB, 
    use_adjoint_or_transposed_A, use_transpose_A,
    use_adjoint_or_transposed_B, use_transpose_B
} = _unrolled_matmul_code(
    A, B, :A, :B, 
    use_adjoint_or_transposed_A, use_transpose_A,
    use_adjoint_or_transposed_B, use_transpose_B
)

_unrolled_matmul_code(
    A::Type{<:StaticArray{SA, TA}}, B::Type{<:StaticArray{SB, TB}}, 
    A_symb, B_symb,
    use_adjoint_or_transposed_A, use_transpose_A,
    use_adjoint_or_transposed_B, use_transpose_B
) where {SA, SB, TA, TB} = adjust_indexing_expression(
    unrolled_smatrix_multiplication_expression(
        A, B, A_symb, B_symb, Base.promote_op(LinearAlgebra.matprod, TA, TB),
        use_adjoint_or_transposed_A, use_transpose_A,
        use_adjoint_or_transposed_B, use_transpose_B
    ),
    (A, A_symb), (B, B_symb);
    finalize = true
)

# Override matmul
@inline matmul(A::SpecialIndexingSArray, B::StaticArray) = unrolled_matmul(A, B)
@inline matmul(A::StaticArray, B::SpecialIndexingSArray) = unrolled_matmul(A, B)
@inline matmul(A::SpecialIndexingSArray, B::SpecialIndexingSArray) = unrolled_matmul(A, B)
# SpecialSArray will take care of the bindings to Base.:*


# Matrix multiplcation
######################
# Determinant and inverse

"""
    unrolled_smatrix_determinant_expression(A::Type{<:StaticMatrix{N, N}}, sm_symb) where N

Create the expression computing the determinant of a `size` `(N, N)` `StaticMatrix` which
will be represented using `sm_symb`. We use cofactor expansion along the first column.
For larger `N` this is probably not the most efficient approach, but for all reasonable
sizes `(N <= 8)` it is faster than StaticArrays's implementation. In our testing from
`N == 9` onwards the compiler struggles a lot.

Like in [`sarray_generation_expression`](@ref) and [`adjust_indexing_expression`](@ref)
we split the return expression into a(n empty) head (`:()`) and the body containing the
actual determinant computation expression.

We include `@inbounds`.
"""
function unrolled_smatrix_determinant_expression(
    A::Type{<:StaticMatrix{N, N}}, sm_symb
) where N
    return :(),
        :(@inbounds $(unrolled_smatrix_minor_expression(
            A, sm_symb, Tuple(1:N), Tuple(1:N))))
end

"""
    unrolled_smatrix_minor_expression(A::Type{<:StaticMatrix}, sm_symb, rows, columns)

Create the expression computing the minor of the matrix represented by `sm_symb` using
the row and column indices specified in `rows` and `columns`.

For example, `unrolled_smatrix_minor_expression(:M, (1, 3), (2, 3))` returns
`:(muladd(-(M[3, 2]), M[1, 3], M[1, 2] * M[3, 3]))`
"""
function unrolled_smatrix_minor_expression(A::Type{<:StaticMatrix}, sm_symb, rows, columns)
    if length(columns) == 1  # == length(rows)
        return :($sm_symb[$(only(rows)), $(only(columns))])
    else
        # cofactor expansion along first (remaining) column: [rows[k], first(columns)]
        det_expr = :(
            $sm_symb[$(first(rows)), $(first(columns))]
            * $(unrolled_smatrix_minor_expression(A, sm_symb, rows[2:end], columns[2:end]))
        )
        for k = 2:length(columns)
            minor_rows = (rows[begin : k - 1]..., rows[k + 1 : end]...)  # drop [k]
            minor_columns = columns[2:end]  # drop [1]
            det_expr = :(
                muladd(
                    $(_maybe_add_sign(
                        :($sm_symb[$(rows[k]), $(first(columns))]), 
                        iseven(k))), 
                    $(unrolled_smatrix_minor_expression(
                        A, sm_symb, minor_rows, minor_columns)), 
                    $det_expr
                )
            )
        end
        return det_expr
    end
end

"""
    _maybe_add_sign(expr, neg)

Add a `-` in front of the `expr` if `neg`. Return `expr` otherwise (i.e. we do not add a
`+`).
"""
function _maybe_add_sign(expr, neg)
    if neg
        return :(-$expr)
    else
        return expr
    end
end

"""
    unrolled_det(sm::StaticMatrix)

Compute the determinant of `sm` using the unrolled cofactor expansion along the first
column, with adjusted indexing.
"""
@inline @generated unrolled_det(sm::StaticMatrix) = _unrolled_det_code(sm, :sm)

_unrolled_det_code(
    A::Type{<:StaticMatrix{N, N}}, sm_symb
) where N = adjust_indexing_expression(
    unrolled_smatrix_determinant_expression(A, sm_symb),
    (A, sm_symb);
    finalize = true
)

# Only use the unrolled cofactor version when N <= 8
for N = 1:8
    @eval(
        @inline LinearAlgebra.det(sism::SpecialIndexingSMatrix{$N, $N}) = 
            unrolled_det(sism)
    )
end
# For N >= 9 we fall back to SpecialSArray's det (via SArray), which is slower at runtime,
# but at least it gets compiled)
# (Potentially subtypes can get away with a larger N)


"""
    unrolled_smatrix_inverse_expression(
        A::Type{<:StaticMatrix{N, N}}, sm_symb, use_scaled_smatrix
    ) where N

Create an expression for creating the inverse of a `size` `(N, N)` matrix which will be
represented using `sm_symb`. We use the adjugate matrix formula and unrolled determinant
based on cofactors. In our output expression we return a `ScaledSMatrix` if
`use_scaled_smatrix`, or an `SMatrix` otherwise. In both cases we include `@inbounds`.

For larger `N` this approach for computing the inverse is presumably not optimal. However,
for `1 <= N <= 4` it seems to have the same speed as the standard `StaticMatrix`
implementation. For `5 <= N <= 6` we are much faster. The gap starts closing at `N == 7`.
For `N == 8` onwards there is too much pressure on the compiler.

Like in [`sarray_generation_expression`](@ref) and [`adjust_indexing_expression`](@ref)
we split the return expression into a head with variable declarations and the body
containing the actual computations.
"""
function unrolled_smatrix_inverse_expression(
    A::Type{<:StaticMatrix{N, N}}, sm_symb, use_scaled_smatrix
) where N
    if use_scaled_smatrix
        adjugate_matrix_entries_exprs = unrolled_adjugate_matrix_expressions_matrix(
            A, sm_symb)
        split_det_expr = unrolled_smatrix_determinant_expression(A, sm_symb)

        return split_det_expr[1], :(
            @inbounds ScaledSMatrix(
                inv($(split_det_expr[2])),
                SMatrix{$N, $N}($(adjugate_matrix_entries_exprs...))
            )
        )
        # (Will have nested @inbounds, but that's fine)
    else
        inverse_matrix_expr_head, inverse_matrix_entries_exprs = 
            unrolled_inverse_matrix_expressions_matrix(A, sm_symb)

        return inverse_matrix_expr_head, :(
            @inbounds SMatrix{$N, $N}($(inverse_matrix_entries_exprs...))
        )
    end
end

"""
    unrolled_adjugate_matrix_expressions_matrix(
        A::Type{<:StaticMatrix{N, N}}, sm_symb
    ) where N

Return a `Matrix{Expr}` containing the entries of the adjugate matrix of the input `(N, N)`
represented as `sm_symb`. 
"""
function unrolled_adjugate_matrix_expressions_matrix(
    A::Type{<:StaticMatrix{N, N}}, sm_symb
) where N
    return [
        _maybe_add_sign(
            unrolled_smatrix_minor_expression(
                A,
                sm_symb, 
                ((1:i-1)..., (i+1:N)...),  # 1:N except for i
                ((1:j-1)..., (j+1:N)...),
            ),
            isodd(i + j)
    ) for j = 1:N, i = 1:N]  # (i.e. transpose of cofactor matrix)
end

"""
    unrolled_inverse_matrix_expressions_matrix(
        A::Type{<:StaticMatrix{N, N}}, sm_symb
    ) where N

Return an `Expr` declaring a new inverse determinant variable, and `Matrix{Expr}` containing
the entries of the adjugate matrix of the input `(N, N)` represented as `sm_symb`. This
entries will use the inverse determinant variable. The first returned expression is
consistent with the split in e.g. [`adjust_indexing_expression`](@ref).
"""
function unrolled_inverse_matrix_expressions_matrix(
    A::Type{<:StaticMatrix{N, N}}, sm_symb
) where N
    split_det_expr = unrolled_smatrix_determinant_expression(A, sm_symb)
    idet_variable = gensym(Symbol(sm_symb, :_idet))
    idet_declaration = :($idet_variable = inv($(split_det_expr[2])))
    head = join_exprs((
        split_det_expr[1],
        idet_declaration
    ))

    adjugate_matrix_exprs = unrolled_adjugate_matrix_expressions_matrix(A, sm_symb)
    return head, map(adjugate_matrix_exprs) do ame
        :($idet_variable * $ame)
    end
end

"""
    unrolled_inv(
        sm::StaticMatrix, ::Val{use_scaled_smatrix} = Val(false)
    ) where use_scaled_smatrix

Compute the inverse of `sm` using the unrolled adjoint matrix formula. By default we return
an `SMatrix`, but with `use_scaled_smatrix`, the return type is `ScaledSMatrix`, where the
scale factor is the inverse determinant.
"""
@inline @generated unrolled_inv(
    sm::StaticMatrix, ::Val{use_scaled_smatrix} = Val(false)
) where use_scaled_smatrix = _unrolled_inv_code(sm, :sm, use_scaled_smatrix)

_unrolled_inv_code(
    A::Type{<:StaticMatrix{N, N}}, sm_symb, use_scaled_smatrix
) where N = adjust_indexing_expression(
    unrolled_smatrix_inverse_expression(A, sm_symb, use_scaled_smatrix),
    (A, sm_symb);
    finalize = true
)

# Use unrolled_inv for N <= 7
for N = 1:7
    @eval(
        @inline Base.inv(sism::SpecialIndexingSMatrix{$N, $N}) = 
            unrolled_inv(sism)
    )
end


# Determinant and inverse
######################
# Broadcasting

"""
    componentwise_sarray_operation_expression(op_symb, A_symb, B_symb, S)

Create an `Array` of size `S` containing expressions for the componentwise application of
the operation represented by `op_symb` on the `SArray{S}`s represented by `A_symb` and
`B_symb`.
"""
function componentwise_sarray_operation_expression_array(op_symb, A_symb, B_symb, S)
    A_exprs = sarray_generation_expression_array(A_symb, S)
    B_exprs = sarray_generation_expression_array(B_symb, S)
    return map(A_exprs, B_exprs) do A_expr, B_expr
        return :($op_symb($A_expr, $B_expr))
    end
end

"""
    componentwise_sarray_operation_expression(op_symb, A_symb, B_symb, S)

Create the split expression (cf. [`sarray_generation_expression`](@ref)) for construction
an `SArray{S}` obtained from the componentwise application of the operation represented by
`op_symb` on the `SArray{S}`s represented by `A_symb` and `B_symb`.
"""
function componentwise_sarray_operation_expression(op_symb, A_symb, B_symb, S)
    A_op_B_exprs = componentwise_sarray_operation_expression_array(
        op_symb, A_symb, B_symb, S)
    return :(), :(
        @inbounds SArray{$S}($(A_op_B_exprs...))
    )
end

"""
    componentwise_sarray_operation(
        op::OP, A::StaticArray{S}, B::StaticArray{S}
    ) where {OP <: Function, S}

Compute the `SArray{S}` obtained by applying the binary operation `op` (e.g. `+`) on
`StaticArray{S}`s `A` and `B`.
"""
@inline @generated componentwise_sarray_operation(
    ::OP, A::StaticArray{S}, B::StaticArray{S}
) where {OP <: Function, S} = _componentwise_sarray_operation_code(OP, A, B, :A, :B)

function _componentwise_sarray_operation_code(
    ::Type{OP}, A::Type{<:StaticArray{S}}, B::Type{<:StaticArray{S}}, a_symb, b_symb
) where {OP <: Function, S}
    # E.g. OP == typeof(+), OP.instance == +, Symbol(OP.instance) == :(+)
    return adjust_indexing_expression(
        componentwise_sarray_operation_expression(
            Symbol(OP.instance), a_symb, b_symb, S),
        (A, :A), (B, :B);
        finalize = true
    )
end

# We could now try to modify broadcasting to use this method.
# But if the compiler is clever (and it presumably is), it can produce the same code
# starting from our `SArray(::SpecialIndexingSArray)` implementation with resolved indexing.
# So we'll just let it do its thing.


### SpecialIndexingSArray
############################################################################################
### SelfAdjointOrSymmetricSMatrix

const StaticVectorOrNTuple{N, T} = Union{StaticVector{N, T}, NTuple{N, T}}

abstract type SelfAdjointOrSymmetric end
struct SelfAdjoint <: SelfAdjointOrSymmetric end
struct Symmetric <: SelfAdjointOrSymmetric end

"""
    SelfAdjointOrSymmetricSMatrix{
        N, H, T, V <: StaticVectorOrNTuple{H, T}, K <: SelfAdjointOrSymmetric
    } <: SpecialIndexingSMatrix{N, N, T}

Struct for compactly representing self-adjoint `(A == adjoint(A))` or symmetric
`(A == transpose(A))` `N x N` matrices using only their lower halves, as a
`V <: StaticVectorOrNTuple{H, T}` where `H == N(N + 1)/2`. The parameter `K` determines
the kind: `SelfAdjoint` or `Symmetric`.

---
    SelfAdjointOrSymmetricSMatrix{N, H, T, V, K}(v::V) where {
        N, H, T, V <: StaticVectorOrNTuple{H, T}, K <: SelfAdjointOrSymmetric
    }

Create a `SelfAdjointOrSymmetricSMatrix{N, H, T, V, K}` using the lower triangle vector `v`.
Internal constructor which does not check compatibility between `N` and `H` (which should
be `div(N * (N + 1), 2)`). In practice you should use the outer constructor
`SelfAdjointOrSymmetricSMatrixMatrix(v::V, ::K) where {...}`.

In case of `K === SelfAdjoint` we also do not check that the entries in `v` corresponding
to the diagonal are self-adjoint themselves (i.e. typically real). (Such checks cannot
be performed statically, i.e. would have to occur at runtime, making it slower.)
"""
struct SelfAdjointOrSymmetricSMatrix{
    N, H, T, V <: StaticVectorOrNTuple{H, T}, K <: SelfAdjointOrSymmetric
} <: SpecialIndexingSMatrix{N, N, T}
    lower_v::V

    # We need this inner constructor, because the default one
    #   SelfAdjointOrSymmetricSMatrix{N, H, T, V, K}(lower_v) where {
    #       N, H, T, V <: StaticVectorOrNTuple{H, T}, K <: SelfAdjointOrSymmetric}
    # is not more specific than StaticArray's 
    #   (::Type{SA})(sa::StaticArray) where SA <: StaticArray
    @inline function SelfAdjointOrSymmetricSMatrix{N, H, T, V, K}(v::V) where {
        N, H, T, V <: StaticVectorOrNTuple{H, T}, K <: SelfAdjointOrSymmetric
    }
        new{N, H, T, V, K}(v)
    end
end
# In particular allows V to an NTuple, SVector or SVectorWithZeros (see later)

# SelfAdjointOrSymmetricSMatrix might support more involved T than just Real or Complex
# (e.g. matrices themselves), but the code is written with (only) the simpler ones in mind

@inline get_lower_v(sassm::SelfAdjointOrSymmetricSMatrix) = sassm.lower_v
@inline get_kind(
    ::SelfAdjointOrSymmetricSMatrix{<:Any, <:Any, <:Any, <:Any, K}
) where K = K


"""
    self_adjoint_or_symmetric_smatrix_from_lower_expression(
        V::Type{<:StaticVectorOrNTuple{H, T}}, lower_v_symb, K
    ) where {H, T}

Return the (split) expression creating a `SelfAdjointOrSymmetricSMatrix` from its lower
triangle vector of type `V`, represented as `v_symb`. We also need the kind `K`
(`SelfAdjoint` or `Symmetric`).

This expression will be used for the corresponding `SelfAdjointOrSymmetricSMatrix`
constructor.
"""
function self_adjoint_or_symmetric_smatrix_from_lower_expression(
    V::Type{<:StaticVectorOrNTuple{H, T}}, lower_v_symb, K
) where {H, T}
    N = isqrt(2 * H)
    H == div(N * (N + 1), 2) || throw(ArgumentError(
        "The `length` `H` of a `SelfAdjointOrSymmetricSMatrix`'s under triangle vector must
         be of the form `H = N * (N + 1) / 2` for some integer `N`! \
         Received: $H (!= $N * ($N + 1) / 2)"
    ))
    isbitstype_check(T, "SelfAdjointOrSymmetricSMatrix")
    # SVector can sometimes be used more generally, but with e.g. (1, im, 2)
    # of eltype Union{Int, Complex{Int}} we run into strange errors
    return :(), :(
        SelfAdjointOrSymmetricSMatrix{$N, $H, $T, $V, $K}($lower_v_symb)
    )  
end

"""
   isbitstype_check(T, type_name)
   
Check if the provided type `T` `isbitstype`. If not, throw an error declaring that
`type_name` only supports `isbitstype` eltypes.
"""
isbitstype_check(T, type_name) = isbitstype(T) || throw(ArgumentError(
    "`$type_name` only supports `isbitstype` `eltype`s. Received: $T"
))



"""
    SelfAdjointOrSymmetricSMatrix(
        v::StaticVectorOrNTuple{H, T}, ::Type{K}
    ) where {H, T, K <: SelfAdjointOrSymmetric}

Create a `SelfAdjointOrSymmetricSMatrix` from the lower triangular matrix entries, as a
vector, and its kind `K` (`SelfAdjoint` or `Symmetric`).
"""
@inline @generated SelfAdjointOrSymmetricSMatrix(
    lower_v::StaticVectorOrNTuple{H, T}, ::Type{K}
) where {H, T, K <: SelfAdjointOrSymmetric} = _self_adjoint_or_symmetric_smatrix_code(
    lower_v, :lower_v, K
)
# Notes: 
# 1) not SelfAdjointOrSymmetricSMatrix(v::V, ::Type{K}) where {...}
#    as this is again less specific than (::Type{SA})(x...) where SA <: StaticArray
# 2) we need a @generated function as N Fails to be computed statically on the GPU without
#    it
# 3) Alternatively we could have chosen to use ::K as argument. In constructor calls we then
#    have to the singleton, e.g. SelfAdjoint(). Now we just use the type SelfAdjoint itself

_self_adjoint_or_symmetric_smatrix_code(
    V::Type{<:StaticVectorOrNTuple{H, T}}, lower_v_symb, K
) where {H, T} = adjust_indexing_expression(
    self_adjoint_or_symmetric_smatrix_from_lower_expression(
        V, lower_v_symb, K 
    );
    finalize = true
)  
# There is no indexing, so no real need for adjust_indexing_expression. But we do want the
# finalize part (including rewriting zeros).


"""
    SelfAdjointSMatrix{N, H, T, V <: StaticVectorOrNTuple{H, T}}

Self-adjoint version of [`SelfAdjointOrSymmetricSMatrix`](@ref). Such a matrix satisfies
`A == adjoint(A)` (== `A'`). If `T <: Real` there is no difference between self-adjoint and
symmmetric matrices.
"""
const SelfAdjointSMatrix{N, H, T, V} = 
    SelfAdjointOrSymmetricSMatrix{N, H, T, V, SelfAdjoint}

"""
    SymmetricSMatrix{N, H, T, V <: StaticVectorOrNTuple{H, T}}

Symmetric version of [`SelfAdjointOrSymmetricSMatrix`](@ref). Such a matrix satisfies
`A == transpose(A)`. If `T <: Real` there is no difference between self-adjoint and
symmmetric matrices.
"""
const SymmetricSMatrix{N, H, T, V} = SelfAdjointOrSymmetricSMatrix{N, H, T, V, Symmetric}

"""
    SelfAdjointSMatrix(v::StaticVectorOrNTuple)

Create a `SelfAdjointSMatrix` from the lower triangular matrix entries, as a vector.

!!! We do not check that the entries of `v` corresponding to the diagonal are self-adjoint
(i.e. typically: `Real`) themselves.
"""
@inline SelfAdjointSMatrix(v::StaticVectorOrNTuple) = 
    SelfAdjointOrSymmetricSMatrix(v, SelfAdjoint)

"""
    SymmetricSMatrix(v::StaticVectorOrNTuple)

Create a `SymmetricSMatrix` from the lower triangular matrix entries, as a vector.
"""
@inline SymmetricSMatrix(v::StaticVectorOrNTuple) =
    SelfAdjointOrSymmetricSMatrix(v, Symmetric)

######################
# From StaticMatrix

"""
    create_self_adjoint_or_symmetric_smatrix_indices_1D_to_2D(N)

Create an `SVector{⋅, CartesianIndex{2}}` containing at entry `i` the lower matrix
coordinates stored by a `SelfAdjointOrSymmetricSMatrix{N}`'s lower triangle vector at 1D
coordinate `i`.
"""
function create_self_adjoint_or_symmetric_smatrix_indices_1D_to_2D(N)
    H = div(N * (N + 1), 2)
    indices_1D_to_2D = MVector{H, CartesianIndex{2}}(undef)
    ctr = 1
    for j = 1:N
        for i = j:N
            indices_1D_to_2D[ctr] = CartesianIndex(i, j)
            ctr += 1
        end
    end
    return SVector(indices_1D_to_2D)
end

"""
   create_self_adjoint_or_symmetric_smatrix_indices_2D_to_1D(N)

Create an `SMatrix{N, N, Int}` containing at index `[i, j]` the 1D index in a
`SelfAdjointOrSymmetricSMatrix{N}`'s lower triangle vector which (possibly up to taking the
scalar `adjoint`) represents this matrix position.
"""
function create_self_adjoint_or_symmetric_smatrix_indices_2D_to_1D(N)
    indices_2D_to_1D = MMatrix{N, N, Int}(undef)
    ctr = 1
    for j = 1:N
        indices_2D_to_1D[j, j] = ctr
        ctr += 1
        for i = j + 1 : N
            indices_2D_to_1D[i, j] = ctr
            indices_2D_to_1D[j, i] = ctr
            ctr += 1
        end
    end
    return SMatrix(indices_2D_to_1D)
end

"""
   create_self_adjoint_or_symmetric_smatrix_should_take_adjoint_matrix(N, K)

Create an `SMatrix{N, N, Bool}` containing at index `[i, j]` a boolean value of whether to
get the `SelfAdjointOrSymmetricSMatrix{N, ..., K}`'s value at `[i, j]` we need to take the
adjoint of the corresponding lower triangle vector entry.
"""
function create_self_adjoint_or_symmetric_smatrix_should_take_adjoint_matrix(N, K)
    return SMatrix{N, N}([K === SelfAdjoint && i < j for i = 1:N, j = 1:N])
end

"""
    lower_matrix_svector_from_static_matrix_expression(
        sm_symb, N, T
    )

Return the expression for creating an `SVector{⋅, T}` containing the lower triangle entries
from a (full) square `StaticMatrix{N, N, T}`. In the expression the input matrix will be
represented as `sm_symb`.

For compatibility with [`adjust_indexing_expression`](@ref) we return a split expression,
where the head is empty, and the body contains the actual (`@inbounds`) `SVector` call.
"""
function lower_matrix_svector_from_static_matrix_expression(
    sm_symb, N, T
)
    H = div(N * (N + 1), 2)
    indices_1D_to_2D = create_self_adjoint_or_symmetric_smatrix_indices_1D_to_2D(N)
    lower_element_exprs = [:($sm_symb[$(Tuple(I)...)]) for I in indices_1D_to_2D]
    return :(), :(
        @inbounds SVector{$H, $T}($(lower_element_exprs...))
    )
end

"""
    lower_matrix_sv(
        M::StaticMatrix{N, N, T}
    ) where {N, T}

Create a `StaticVector` (typically an `SVector`) of the lower triangle entries of the
provided square matrix.
"""
@inline @generated lower_matrix_sv(
    M::StaticMatrix{N, N, T}
) where {N, T} = _lower_matrix_sv_code(M, :M)

_lower_matrix_sv_code(M::Type{<:StaticMatrix{N, N, T}}, sm_symb) where {N, T} = 
    adjust_indexing_expression(
        lower_matrix_svector_from_static_matrix_expression(
            sm_symb, N, T
        ),
        (M, sm_symb);
        finalize = true
    )

"""
    lower_matrix_sv(
        M::SelfAdjointOrSymmetricSMatrix
    )

Returns the underlying lower triangle vector, like `get_lower_v`.
"""
@inline lower_matrix_sv(M::SelfAdjointOrSymmetricSMatrix) = get_lower_v(M)


"""
    self_adjoint_or_symmetric_smatrix_from_static_matrix_expression(
        sm_symb, N, T, K
    )

Return the expression for creating a `SelfAdjointOrSymmetricSMatrix` from a (full)
square `StaticMatrix{N, N, T}` and its kind `K` (`SelfAdjoint` or `Symmetric`). In the
expression the input matrix will be represented as `sm_symb`.

In our expression we will not check whether the matrix is self-adjoint or symmetric. We just
take the lower triangle entries, as an `SVector` (see also
[`lower_matrix_svector_from_static_matrix_expression`](@ref)). This means that for
`SelfAdjoint` we can get an invalid `SelfAdjointSMatrix` when the diagonal entries are not
self-adjoint (`Real`) themselves.

For compatibility with [`adjust_indexing_expression`](@ref) we return a split expression,
where the head is empty, and the body contains the actual (`@inbounds`)
`SelfAdjointOrSymmetricSMatrix` call.
"""
function self_adjoint_or_symmetric_smatrix_from_static_matrix_expression(
    sm_symb, N, T, K
)
    lmsv_expr = lower_matrix_svector_from_static_matrix_expression(sm_symb, N, T)
    return lmsv_expr[1], :(
        SelfAdjointOrSymmetricSMatrix(
            $(lmsv_expr[2]),
            $K
        )
    )
end

"""
    SelfAdjointOrSymmetricSMatrix(
        M::StaticMatrix{N, N, T}, ::Type{K}
    ) where {N, T, K <: SelfAdjointOrSymmetric}

Create a `SelfAdjointOrSymmetricSMatrix` from a (full) `StaticMatrix` and its kind `K`
(`SelfAdjoint` or `Symmetric`). We take the lower half of `M` and do not check whether `M`
itself is in fact symmetric.

!!! For `SelfAdjoint` we need `M`'s diagonal to consist of self-adjoint (i.e. typically:
`Real`) entries itself, but do not check this.
"""
@inline @generated SelfAdjointOrSymmetricSMatrix(
    M::StaticMatrix{N, N, T}, ::Type{K}
) where {N, T, K <: SelfAdjointOrSymmetric} = _self_adjoint_or_symmetric_smatrix_code(
    M, :M, K
)
# (Or we could just call SelfAdjointOrSymmetricSMatrix(lower_matrix_sv(M)), K) )

_self_adjoint_or_symmetric_smatrix_code(
    M::Type{<:StaticMatrix{N, N, T}}, sm_symb, K
) where {N, T} = adjust_indexing_expression(
    self_adjoint_or_symmetric_smatrix_from_static_matrix_expression(
        sm_symb, N, T, K
    ),
    (M, sm_symb);
    finalize = true
)
# (We probably do not need to adjust the indexing, as normally M <: SMatrix. But we could in
# principle get a SpecialIndexingSMatrix)


"""
    SelfAdjointSMatrix(M::StaticMatrix{N, N, T}) where {N, T}

Create a `SelfAdjointSMatrix` from a (full) `StaticMatrix`, by taking the lower half of `M`.
We do not check whether whether `M` itself is in fact symmetric.

!!! The diagonal entries of `M` have to each be self-adjoint themselves (i.e. typically:
`Real`) entries, but do not check this.
"""
@inline SelfAdjointSMatrix(M::StaticMatrix{N, N, T}) where {N, T} = 
    SelfAdjointOrSymmetricSMatrix(M, SelfAdjoint)


"""
    SymmetricSMatrix(M::StaticMatrix{N, N, T}) where {N, T}

Create a `SymmetricSMatrix` from a (full) `StaticMatrix`, by taking the lower half of `M`.
We do not check whether whether `M` itself is in fact symmetric.
"""
@inline SymmetricSMatrix(M::StaticMatrix{N, N, T}) where {N, T} = 
    SelfAdjointOrSymmetricSMatrix(M, Symmetric)

# From StaticMatrix
######################
# Indexing

"""
    adjust_indexing_expression(
        split_expr,
        ::Type{<:SelfAdjointOrSymmetricSMatrix{N, <:Any, <:Any, V, K}}, 
        sassm_symb, 
    ) where {N, V, K}

Rewrite the split `expr` containing an `SelfAdjointOrSymmetricSMatrix` represented as
`ssm_symb` by replacing the 2D indexing by 1D indexing of the underlying data.

!!! The indexing expressions in `expr_body` (`split_expr[2]`) cannot contain variables.
That is, `M[k, 1]` is not allowed. 
"""
function adjust_indexing_expression(
    split_expr, ::Type{<:SelfAdjointOrSymmetricSMatrix{N, <:Any, <:Any, V, K}}, sassm_symb
) where {N, V, K}
    expr_head, expr_body = split_expr
    ssm_lower_v_symb = gensym(Symbol(sassm_symb, :_lower_v))
    expr_head = join_exprs((
        expr_head,
        :($ssm_lower_v_symb = get_lower_v($sassm_symb))
    ))
    
    indices_2D_to_1D = create_self_adjoint_or_symmetric_smatrix_indices_2D_to_1D(N)
    should_take_adjoint = 
        create_self_adjoint_or_symmetric_smatrix_should_take_adjoint_matrix(N, K)
    expr_body = MacroTools.postwalk(expr_body) do ex
        if @capture(ex, A_[I__]) && A === sassm_symb 
            # I is a Vector{Any} (but really a Vector{Int})
            tI = Tuple(I)
            if !should_take_adjoint[tI...]
                :($ssm_lower_v_symb[$(indices_2D_to_1D[tI...])])
                # (Note that this also works for linear indexing. E.g. when N == 2, then
                # M[3] (i.e. M[1, 2]) gets rewritten to M.lower_v[2])
            else
                 :(($ssm_lower_v_symb[$(indices_2D_to_1D[tI...])])')  
            end
        else
            ex
        end
    end
    
    return adjust_indexing_expression((expr_head, expr_body), V, ssm_lower_v_symb)
end

Base.@propagate_inbounds function Base.getindex(
    sassm::SelfAdjointOrSymmetricSMatrix, inds::Int...
)
    return SMatrix(sassm)[inds...]  # (Just the SpecialSArray method, so not really needed)
    # A more direct approach for 2D indexing a SelfAdjointOrSymmetricSMatrix{N, ..., K} at 
    # [i, j] is:
    #=   
        if j > i
            (i, j) = (j, i)
            take_adjoint = true
        else
            take_adjoint = false
        end
        # Now i >= j
        # The last matrix row's 1D indices (i.e. at [N, j]) are
        #   N, N + (N - 1), N + (N - 1) + (N - 2),  ..., jN - j(j - 1)/2
        # At [i, j] we then get jN - j(j - 1)/2 - (N - i) = (j - 1)(N - j/2) + i
        idx = div((j - 1) * (2 * N - j), 2) + i
        if take_adjoint
            return sassm.lower_sv[idx]'
        else
            return sassm.lower_sv[idx]
        end
    =#
    # But this seems to be slower, maybe because the compiler can simplify e.g.
    # SMatrix(sassm)[1, 1] directly to ssm.lower_v[1]?
end

# Indexing
######################
# Basic methods

@inline SelfAdjointOrSymmetricSMatrix(sassm::SelfAdjointOrSymmetricSMatrix) = sassm

@inline function Base.:(==)(
    A::SelfAdjointOrSymmetricSMatrix{<:Any, <:Any, TA, <:Any, KA}, 
    B::SelfAdjointOrSymmetricSMatrix{<:Any, <:Any, TB, <:Any, KB}
) where {TA, TB, KA, KB}
    lower_v_A = get_lower_v(A); lower_v_B = get_lower_v(B)
    lower_v_A == lower_v_B || return false
    KA == KB && return true
    upper_v_A = if KA === Symmetric || TA <: Real
        lower_v_A
    else
        adjoint.(lower_v_A)
    end
    upper_v_B = if KB === Symmetric || TB <: Real
        lower_v_B
    else
        adjoint.(lower_v_B)
    end
    return upper_v_A == upper_v_B
end
        
# In particular
@inline Base.:(==)(A::SelfAdjointSMatrix, B::SelfAdjointSMatrix) =
    get_lower_v(A) == get_lower_v(B)
    # Then V, T, N and consequently H are also all equal.
@inline Base.:(==)(A::SymmetricSMatrix, B::SymmetricSMatrix) =
    get_lower_v(A) == get_lower_v(B)

@inline Base.:+(sassm::SelfAdjointOrSymmetricSMatrix) = 
    SelfAdjointOrSymmetricSMatrix(+get_lower_v(sassm), get_kind(sassm))
# (Not sure what's the point of the unary +)
@inline Base.:-(sassm::SelfAdjointOrSymmetricSMatrix) = 
    SelfAdjointOrSymmetricSMatrix(-get_lower_v(ssm), get_kind(sassm))
# We assume that -(::V) exists and returns a StaticVectorOrNTuple (which is already not the
# case for an NTuple).
# Similar statements apply below
@inline Base.:+(
    A::SelfAdjointOrSymmetricSMatrix{N, <:Any, <:Any, <:Any, K}, 
    B::SelfAdjointOrSymmetricSMatrix{N, <:Any, <:Any, <:Any, K}
) where {N, K} = SelfAdjointOrSymmetricSMatrix(get_lower_v(A) + get_lower_v(B), K)
# (Alternatively, use broadcasted +, see later)
@inline Base.:-(
    A::SelfAdjointOrSymmetricSMatrix{N, <:Any, <:Any, <:Any, K}, 
    B::SelfAdjointOrSymmetricSMatrix{N, <:Any, <:Any, <:Any, K}
) where {N, K} = 
    SelfAdjointOrSymmetricSMatrix(get_lower_v(A) - get_lower_v(B), K)

@inline Base.:+(x::Number, ssm::SymmetricSMatrix) = SymmetricSMatrix(x + get_lower_v(ssm))
@inline Base.:+(ssm::SymmetricSMatrix, x::Number) = SymmetricSMatrix(get_lower_v(ssm) + x)
@inline Base.:-(x::Number, ssm::SymmetricSMatrix) = SymmetricSMatrix(x - get_lower_v(ssm))
@inline Base.:-(ssm::SymmetricSMatrix, x::Number) = SymmetricSMatrix(get_lower_v(ssm) - x)
@inline Base.:*(x::Number, ssm::SymmetricSMatrix) = SymmetricSMatrix(x * get_lower_v(ssm))
@inline Base.:*(ssm::SymmetricSMatrix, x::Number) = SymmetricSMatrix(get_lower_v(ssm) * x)
@inline Base.:/(ssm::SymmetricSMatrix, x::Number) = 
    SymmetricSMatrix(get_lower_v(ssm) .* inv(x))
@inline Base.:\(x::Number, ssm::SymmetricSMatrix) = ssm / x

# For SelfAdjointSMatrices this is all only true if we add/multiply with/... a real number
@inline Base.:+(x::Real, sasm::SelfAdjointSMatrix) = 
    SelfAdjointSMatrix(x + get_lower_v(sasm))
@inline Base.:+(sasm::SelfAdjointSMatrix, x::Real) = 
    SelfAdjointSMatrix(get_lower_v(sasm) + x)
@inline Base.:-(x::Real, sasm::SelfAdjointSMatrix) = 
    SelfAdjointSMatrix(x - get_lower_v(sasm))
@inline Base.:-(sasm::SelfAdjointSMatrix, x::Real) = 
    SelfAdjointSMatrix(get_lower_v(sasm) - x)
@inline Base.:*(x::Real, sasm::SelfAdjointSMatrix) = 
    SelfAdjointSMatrix(x * get_lower_v(sasm))
@inline Base.:*(sasm::SelfAdjointSMatrix, x::Real) = 
    SelfAdjointSMatrix(get_lower_v(sasm) * x)
@inline Base.:/(sasm::SelfAdjointSMatrix, x::Real) = 
    SelfAdjointSMatrix(get_lower_v(sasm) .* inv(x))
@inline Base.:\(x::Real, sasm::SelfAdjointSMatrix) = sasm / x

# Basic methods
######################
# Broadcasting

# Like above, SelfAdjointSMatrices are not closed under many broadcasting operations,
# as we would need the broadcasted function to commute with complex conjugation (or adjoint)
# Translations (with a non-Real shift) are already non-examples!
# So we will only consider SymmetricSMatrices

struct SymmetricSMatrixStyle <: AbstractSpecialSArrayStyle{2} end
Base.BroadcastStyle(::Type{<:SymmetricSMatrix}) = SymmetricSMatrixStyle()

# Scalar broadcasting results in a SymmetricSMatrix.
Base.BroadcastStyle(s::SymmetricSMatrixStyle, ::Broadcast.AbstractArrayStyle{0}) = s
Base.BroadcastStyle(s::SymmetricSMatrixStyle, ::Broadcast.DefaultArrayStyle{0}) = s
Base.BroadcastStyle(s::SymmetricSMatrixStyle, ::StaticArrays.StaticArrayStyle{0}) = s
Base.BroadcastStyle(s::SymmetricSMatrixStyle, ::AbstractSpecialSArrayStyle{0}) = s

# Array broadcasting does not, so should fall back to the DefaultSpecialSArrayStyle,
# which is automatic due to the AbstractSpecialArrayStyle rules

# Idea:
# 1) In broadcasting, replace the SymmetricSMatrix by its lower_v
# 2) Broadcast using lower_v
# 3) Put the resulting vector in a SymmetricSMatrix
Base.copy(bc::Broadcast.Broadcasted{<:SymmetricSMatrixStyle}) = 
    SymmetricSMatrix(Base.copy(_rewrite_bc_symmetric_smatrix(bc)))

"""
    _rewrite_bc_symmetric_smatrix(x)

Rewrite (the contents of) a `Broadcast.Broadcasted`, replacing appearances of
SymmetricSMatrices their `lower_v`.
"""
_rewrite_bc_symmetric_smatrix(ssm::SymmetricSMatrix) = get_lower_v(ssm)
_rewrite_bc_symmetric_smatrix(x) = x
_rewrite_bc_symmetric_smatrix(bc::Broadcast.Broadcasted) = 
    Broadcast.broadcasted(bc.f, map(_rewrite_bc_symmetric_smatrix, bc.args)...)

# Broadcasting
######################
# Adjoint and transpose

@inline LinearAlgebra.adjoint(sasm::SelfAdjointSMatrix) = sasm
@inline LinearAlgebra.transpose(ssm::SymmetricSMatrix) = ssm

@inline LinearAlgebra.adjoint(ssm::SymmetricSMatrix{N, H, T, V}) where {N, H, T, V} =
    SymmetricSMatrix{N, H, T, V}(adjoint.(get_lower_v(ssm)))
# E.g. M = [1 im; im 2] --> v = [1, im, 2] // M' = [1 -im; -im 2] --> v = [1, -im, 2]
@inline LinearAlgebra.adjoint(ssm::SymmetricSMatrix{<:Any, <:Any, <:Real, <:Any}) = ssm
@inline LinearAlgebra.transpose(sasm::SelfAdjointSMatrix{N, H, T, V}) where {N, H, T, V} =
    SelfAdjointSMatrix{N, H, T, V}(adjoint.(get_lower_v(sasm)))
# E.g. M = [1 -im; im 2] --> v = [1, im, 2] // M^T = [1 im; -im 2] --> v = [1, -im, 2]
@inline LinearAlgebra.transpose(sasm::SelfAdjointSMatrix{<:Any, <:Any, <:Real, <:Any}) = 
    sasm


# Adjoint and transpose
######################
# Congruence transform

"""
    congruence_transform(A, B)
    
Compute `A * B * transpose(A)`. Usually this requires `A` and `B` to be AbstractMatrices of
sizes `(M, N)` and `(N, N)`.

When `A` is a `StaticArray` (normally a `StaticMatrix` to avoid errors) and `B` is a
`SymmetricSMatrix` the result is again a `SymmetricSMatrix`.

For the version with `adjoint` instead of `transpose`, see
[`adjoint_congruence_transform`](@ref).
"""
@inline function congruence_transform(A::StaticArray, B::StaticArray)
    return (A * B) * transpose(A)
end

@inline function congruence_transform(
    A::StaticArray, B::SymmetricSMatrix
)
    AB = A * B  # (matmul; not a SymmetricSMatrix)
    return SymmetricSMatrix(unrolled_matmul_lower(
        AB, A, 
        Val(false), Val(false),
        Val(true), Val(true)))
end

"""
    congruence_transform(A)

Compute `A * transpose(A)`. When `A` is a `StaticArray` (in particular a `StaticMatrix` or
`StaticVector`), the result is a `SymmetricSMatrix`.
"""
@inline function congruence_transform(A)
    return A * transpose(A)
end

@inline function congruence_transform(A::StaticArray)
    return SymmetricSMatrix(unrolled_matmul_lower(
        A, A, 
        Val(false), Val(false),
        Val(true), Val(true)))
end

"""
    adjoint_congruence_transform(A, B)
    
Compute `A * B * A'`. Usually this requires `A` and `B` to be AbstractMatrices of
sizes `(M, N)` and `(N, N)`. 
see 

When `A` is a `StaticArray` (normally a `StaticMatrix` to avoid errors) and `B` is a
`SelfAdjointSMatrix` the result is again a `SelfAdjointSMatrix`.

For the version with `transpose` instead of `adjoint`, see [`congruence_transform`](@ref).
"""
@inline function adjoint_congruence_transform(A::StaticArray, B::StaticArray)
    return (A * B) * transpose(A)
end

@inline function adjoint_congruence_transform(
    A::StaticArray, B::SelfAdjointSMatrix
)
    AB = A * B  # (matmul; not a SelfAdjointSMatrix)
    return SelfAdjointSMatrix(unrolled_matmul_lower(
        AB, A, 
        Val(false), Val(false),
        Val(true), Val(false)))
end

"""
    adjoint_congruence_transform(A)

Compute `A * A'`. When `A` is a `StaticMatrix`, the result is a `SelfAdjointSMatrix`.
"""
@inline function congruence_transform_adjoint(A)
    return A * A'
end

@inline function congruence_transform_adjoint(A::StaticMatrix)
    return SelfAdjointSMatrix(unrolled_matmul_lower(
        A, A, 
        Val(false), Val(false), 
        Val(true), Val(false)))
end

"""
    unrolled_matmul_lower(
        A::StaticArray{SA, TA}, B::StaticArray{SB, TB},
        ::Val{use_adjoint_or_transposed_A} = Val(false), 
        ::Val{use_transpose_A} = Val(false),
        ::Val{use_adjoint_or_transposed_B} = Val(false), 
        ::Val{use_transpose_B} = Val(false),
    ) where {
        SA, SB, TA, TB, 
        use_adjoint_or_transposed_A, use_transpose_A,
        use_adjoint_or_transposed_B, use_transpose_B
    }

Compute the lower half of the matrix product `A * B` or transposed variants (e.g.
`A * transpose(B)`), if this matrix product results in a square matrix. In this case we
return an `SVector` of the lower triangle entries.
"""
@inline @generated unrolled_matmul_lower(
    A::StaticArray{SA, TA}, B::StaticArray{SB, TB},
    ::Val{use_adjoint_or_transposed_A} = Val(false), ::Val{use_transpose_A} = Val(false),
    ::Val{use_adjoint_or_transposed_B} = Val(false), ::Val{use_transpose_B} = Val(false)
) where {
    SA, SB, TA, TB, 
    use_adjoint_or_transposed_A, use_transpose_A,
    use_adjoint_or_transposed_B, use_transpose_B
} = _unrolled_matmul_lower_code(
    A, B, :A, :B, 
    use_adjoint_or_transposed_A, use_transpose_A,
    use_adjoint_or_transposed_B, use_transpose_B
)

_unrolled_matmul_lower_code(
    A::Type{<:StaticArray{SA, TA}}, B::Type{<:StaticArray{SB, TB}}, 
    A_symb, B_symb, 
    use_adjoint_or_transposed_A, use_transpose_A,
    use_adjoint_or_transposed_B, use_transpose_B
) where {SA, SB, TA, TB} = adjust_indexing_expression(
    unrolled_smatrix_multiplication_lower_expression(
        A, B, A_symb, B_symb, Base.promote_op(LinearAlgebra.matprod, TA, TB),
        use_adjoint_or_transposed_A, use_transpose_A,
        use_adjoint_or_transposed_B, use_transpose_B
    ),
    (A, A_symb), (B, B_symb);
    finalize = true
)

"""
    unrolled_smatrix_multiplication_lower_expression(
        ::Type{<:StaticArray{SA}}, ::Type{<:StaticArray{SB}}, A_symb, B_symb, T, 
        use_adjoint_or_transposed_A = false, use_transpose_A = false,
        use_adjoint_or_transposed_B = false, use_transpose_B = false
    ) where {SA, SB}

Create the expression for obtaining the lower half of the matrix product of arrays of
(`Tuple{}`) sizes `SA` and `SB`, represented as `A_symb` and `B_symb`. The output
expression will create an `SVector` of `eltype` `T`. Optionally uses adjoints or transposes
(e.g. `A'` or `transpose(B)`) in the matrix product. If adjoints are requested,
but `T <: Real`, we will use the equivalent, but slightly cleaner transposes instead.

We require the matrix product to result in a square matrix. If not, we throw an error.
(Note that this is always the case for congruence transforms.)

Like in [`unrolled_smatrix_multiplication_expression`](@ref) we split the return expression
into a head (`:()`) and the body containing the actual matrix multiplication lower half
`SVector` creation expression. 
"""
function unrolled_smatrix_multiplication_lower_expression(
    ::Type{<:StaticArray{SA}}, ::Type{<:StaticArray{SB}}, A_symb, B_symb, T, 
    use_adjoint_or_transposed_A = false, use_transpose_A = false,
    use_adjoint_or_transposed_B = false, use_transpose_B = false
) where {SA, SB}
    out_matrix_exprs = unrolled_smatrix_multiplication_expressions_matrix(
        A_symb, B_symb, SA, SB, 
        use_adjoint_or_transposed_A, use_transpose_A || T <: Real,
        use_adjoint_or_transposed_B, use_transpose_B || T <: Real)
    M, N = size(out_matrix_exprs)
    M == N || throw(DimensionMismatch(
        "We only support taking the lower half of a matrix product when the result is \
         square! Received: ($M, $N)"))
    indices_1D_to_2D = create_self_adjoint_or_symmetric_smatrix_indices_1D_to_2D(N)
    out_vector_exprs = [out_matrix_exprs[I] for I in indices_1D_to_2D]
    H = length(out_vector_exprs)  # == div(N * (N + 1), 2)
    return :(), :(
        @inbounds SVector{$H, $T}($(out_vector_exprs...))
    )
end


# Congruence transform
######################
# Other matrix-level methods

function unrolled_smatrix_determinant_expression(
    ::Type{<:SelfAdjointSMatrix{2}}, sasm_symb)
    return :(),
        :(@inbounds muladd($sasm_symb[1, 1], $sasm_symb[2, 2], -abs2($sasm_symb[2, 1])))
end
# The abs2(x) might be a bit faster than x * adjoint(x)


"""
   unrolled_smatrix_inverse_expression(
        A::Type{<:SelfAdjointOrSymmetricSMatrix{N, <:Any, <:Any, <:Any, K}}, 
        sassm_symb, use_scaled_smatrix
    ) where {N, K}

Create an expression for creating the inverse of a `SelfAdjointOrSymmetricSMatrix` of `size`
`(N, N)`, which will be represented using `sassm_symb`. The difference with the more general
`StaticMatrix` version is that we only compute the lower half.
"""
function unrolled_smatrix_inverse_expression(
    A::Type{<:SelfAdjointOrSymmetricSMatrix{N, <:Any, <:Any, <:Any, K}}, 
    sassm_symb, use_scaled_smatrix
) where {N, K}
    indices_1D_to_2D = create_self_adjoint_or_symmetric_smatrix_indices_1D_to_2D(N)
    H = length(indices_1D_to_2D)
    if use_scaled_smatrix
        adjugate_matrix_lower_entries_exprs = unrolled_adjugate_matrix_expressions_matrix(
            A, sassm_symb)[indices_1D_to_2D]

        split_det_expr = unrolled_smatrix_determinant_expression(A, sassm_symb)
        return split_det_expr[1], :(
            @inbounds ScaledSMatrix(
                inv($(split_det_expr[2])),
                SelfAdjointOrSymmetricSMatrix(
                    SVector{$H}($(adjugate_matrix_lower_entries_exprs...)),
                    $K
                )
            )
        )
        # Note that for the lower_v we cannot necessarily use the same type V as for sassm
        # itself (for example, zeros can change position in case we had
        # V <: SVectorWithZeros), hence the default SVector
    else
        inverse_matrix_expr_head, inverse_matrix_entries_exprs = 
            unrolled_inverse_matrix_expressions_matrix(A, sassm_symb)
        inverse_matrix_lower_entries_exprs = inverse_matrix_entries_exprs[indices_1D_to_2D]

        return inverse_matrix_expr_head, :(
            @inbounds SelfAdjointOrSymmetricSMatrix(
                SVector{$H}($(inverse_matrix_lower_entries_exprs...)),
                $K
            )
        )
    end
end

# For eigen-related methods we have type instability if we do not know in advance
# our matrix is symmetric (as we might need complex values). So do not use our normal
# fallback method, but explicitly inform we are dealing with (LinearAlgebra.)Symmetric
# matrices
@inline LinearAlgebra.eigen(ssm::SymmetricSMatrix; kwargs...) = 
    eigen(LinearAlgebra.Symmetric(SArray(ssm)); kwargs...)
@inline LinearAlgebra.eigvals(ssm::SymmetricSMatrix; kwargs...) = 
    eigvals(LinearAlgebra.Symmetric(SArray(ssm)); kwargs...)
@inline LinearAlgebra.eigvecs(ssm::SymmetricSMatrix; kwargs...) = 
    eigvecs(LinearAlgebra.Symmetric(SArray(ssm)); kwargs...)


# Other matrix-level methods
######################
# Gradients

"""
    gradient_wrt_symmetric_matrix_to_gradient_wrt_lower_vector_expression(
        sm_grad_symb, N, T
    )

Create the expression to go from the gradient with respect to a symmetric `N × N` matrix to
the gradient with respect to the lower half vector of this matrix. Concretely, diagonal
entries are preserved, but the two off-diagonal entries (in the lower and upper half)
corresponding to one vector entry will be added.

As always, we split the return expression into a head (`:()`) and the body containing the
expression creating the `SVector{⋅, T}` of the gradient with respect to the lower half
of the symmetric matrix. The gradient with respect to the full matrix will be represented as
`sm_grad_symb`.
"""
function gradient_wrt_symmetric_matrix_to_gradient_wrt_lower_vector_expression(
    sm_grad_symb, N, T
)
    grad_full_matrix_exprs = sarray_generation_expression_array(sm_grad_symb, Tuple{N, N})
    grad_lower_v_exprs = Expr[]
    for j = 1:N
        push!(grad_lower_v_exprs, grad_full_matrix_exprs[j, j])
        for i = j+1 : N
            push!(grad_lower_v_exprs, 
                :($(grad_full_matrix_exprs[i, j]) + $(grad_full_matrix_exprs[j, i])))
        end
    end

    return :(), :(
        SVector{$(length(grad_lower_v_exprs)), T}($(grad_lower_v_exprs...))
    )
end

"""
    gradient_wrt_symmetric_to_gradient_wrt_lower_vector(grad_sm::StaticMatrix{N, N, T})

Convert a gradient `grad_sm` with respect to a symmetric `N × N` matrix into the gradient
with respect to the lower half representation of the matrix. We will return a
`SVector{H, T}` where `H = N * (N - 1) / 2`.
"""
@inline @generated gradient_wrt_symmetric_matrix_to_gradient_wrt_lower_vector(
    grad_sm::StaticMatrix{N, N, T}
) where {N, T} = _gradient_wrt_symmetric_matrix_to_gradient_wrt_lower_vector_code(
    grad_sm, :grad_sm
)
# Note that the gradient itself does not necessarily need to be symmetric. But of course,
# we do support grad_sm::SymmetricSMatrix if appropriate

_gradient_wrt_symmetric_matrix_to_gradient_wrt_lower_vector_code(
    grad_sm::Type{<:StaticMatrix{N, N, T}}, grad_sm_symb,
) where {N, T} = adjust_indexing_expression(
    gradient_wrt_symmetric_matrix_to_gradient_wrt_lower_vector_expression(
        grad_sm_symb, N, T
    ),
    (grad_sm, grad_sm_symb);
    finalize = true
)


### SymmetricSMatrix
############################################################################################
### SArrayWithZeros

const SVectorOrNTuple{N, T} = Union{SVector{N, T}, NTuple{N, T}}

"""
    SArrayWithZeros{
        S, T, D, ZIds, L, V <: SVectorOrNTuple{L, T}
    } <: SpecialIndexingSArray{S, T, D}

`StaticArray` with known zero entries at an `NTuple` of `D`-D coordinates `ZIds` (which
hence is an `NTuple{⋅, NTuple{D, Int}})`.` We store a vector of type `V` containing the
`L` non-zero entrieS.

---

    SArrayWithZeros{S, T, D, ZIds, L, V}(data_vector::V) where {
        S, T, D, ZIds, L, V <: SVectorOrNTuple{L, T}
    }

Create a `SArrayWithZeros` from types and the underlying data vector. Internal constructor
which does not check these are compatible.
"""
struct SArrayWithZeros{
    S, T, D, ZIds, L, V <: SVectorOrNTuple{L, T}
} <: SpecialIndexingSArray{S, T, D}
    data_vector::V

    @inline function SArrayWithZeros{S, T, D, ZIds, L, V}(data_vector::V) where {
        S, T, D, ZIds, L, V <: SVectorOrNTuple{L, T}
    }
        new{S, T, D, ZIds, L, V}(data_vector)
    end
end
# We use SVectorOrNTuple instead of StaticVectorOrNTuple, since backing an SArrayWithZeros
# by e.g. a SVectorWithZeros makes no sense. Additionally, we can construct an
# SVectorOrNTuple using an NTuple (as we will do in adjoint/transpose).
# Normally we'll just use SVector though (as in the constructor from an SArray).

@inline get_data_vector(sawz::SArrayWithZeros) = sawz.data_vector
@inline get_zero_indices(::SArrayWithZeros{<:Any, <:Any, <:Any, ZIds}) where ZIds = ZIds


"""
    SMatrixWithZeros{M, N, T, ZIds, L, V}

2D `SArrayWithZeros` of static size `NTuple{M, N}`.
"""
const SMatrixWithZeros{M, N, T, ZIds, L, V} = SArrayWithZeros{Tuple{M, N}, T, 2, ZIds, L, V}

"""
    SVectorWithZeros{N, T, ZIds, L, V}

1D `SArrayWithZeros` of static size `NTuple{N}`.
"""
const SVectorWithZeros{N, T, ZIds, L, V} = SArrayWithZeros{Tuple{N}, T, 1, ZIds, L, V}


"""
    sarray_with_zeros_from_data_vector_expression(
        V::Type{<:SVectorOrNTuple{L, T}}, data_vector_symb, ::Type{S}, ::Val{ZIds}
    ) where {S <: Tuple, ZIds, L, T}

Return the (split) expression for the `SArrayWithZeros` constructor from the underlying data
vector (in the expression represented as `data_vector_symb`) of type `V`, together
with the desired size and tuple of zero indices.
"""
function sarray_with_zeros_from_data_vector_expression(
    V::Type{<:SVectorOrNTuple{L, T}}, data_vector_symb, S, ZIds
) where {L, T}
    isbitstype_check(T, "SArrayWithZeros")
    cZIds = _canonicalize_ZIds(ZIds)
    _check_sawz_args(S, cZIds, L)
    D = fieldcount(S)
    return :(), :(
        SArrayWithZeros{$S, $T, $D, $cZIds, $L, $V}($data_vector_symb)
    )
end

"""
    SArrayWithZeros(
        data_vector::SVectorOrNTuple{L, T}, ::Type{S}, ::Val{ZIds}
    ) where {S <: Tuple, ZIds, L, T}

Create an `SArrayWithZeros` from the underlying data vector, desired size (as a
`Tuple{...}`) and `NTuple{⋅, NTuple{D, Int}}` of indices with zero entries (as a `Val`).
We will statically check the compatibility of these arguments.
"""
@inline @generated SArrayWithZeros(
    data_vector::SVectorOrNTuple{L, T}, ::Type{S}, ::Val{ZIds}
) where {S <: Tuple, ZIds, L, T} = _sarray_with_zeros_code(
    data_vector, :data_vector, S, ZIds
)

_sarray_with_zeros_code(
    V::Type{<:SVectorOrNTuple{L, T}}, data_vector_symb, S, ZIds
) where {L, T} = adjust_indexing_expression(
    sarray_with_zeros_from_data_vector_expression(V, data_vector_symb, S, ZIds);
    finalize = true
)
# (Same remark as for _self_adjoint_or_symmetric_smatrix_code: we have no idexing, but just
#  want the finalize)


"""
    _check_sawz_args(S, ZIds, L)

Check that the `SArrayWithZeros` arguments of static size `S`, non-zero index locations
`ZIds` and representation length `L` are compatible.
"""
function _check_sawz_args(S, ZIds, L)
    sz = fieldtypes(S)
    D = length(sz)  # == fieldcount(S)
    L isa Int || throw(ArgumentError(
        "An `SArrayWithZeros`'s underlying length parameter `L` must be an `Int`! \
         Received: $(typeof(L))"
    ))
    S <: Tuple && sz isa NTuple{D, Int} && all(>=(0), sz) || throw(ArgumentError(
        "An `SArrayWithZeros`'s size parameter `S` must be a `Tuple{}` containing `D` \
         (`== $D` here)` non-negative `Int` entries! Received: $S"
    ))  
    ZIds isa NTuple{<:Any, NTuple{D, Int}} || throw(ArgumentError(
        "An `SArrayWithZeros`'s zero indices parameter `ZIds` must be an \
         `NTuple{., NTuple{D, Int}}`! Received $(typeof(ZIds))"
    ))
    L == prod(sz) - length(ZIds) || throw(ArgumentError(
        "An `SArrayWithZeros`'s data vector's `length` must equal the number of non-zero \
         entries in the array! Received: $L != $(prod(sz)) - $(length(ZIds))"
    ))
end

"""
    _canonicalize_ZIds(ZIds)

Remove duplicates and sort `ZIds` to canonical order.
"""
_canonicalize_ZIds(ZIds) = Tuple(sort(unique(ZIds)))


"""
    SMatrixWithZeros(sv::StaticVectorOrNTuple, ::Val{M}, ::Val{N}, ZIds_val) where {M, N}

Create an `SMatrixWithZeros` from a partial `StaticVectorOrNTuple` of non-zero entries, the
total size including zeros `(M, N)`, and a `Val` of 2D zero locations.
"""
@inline SMatrixWithZeros(
    sv::StaticVectorOrNTuple, ::Val{M}, ::Val{N}, ZIds_val
) where {M, N} = SArrayWithZeros(sv, Tuple{M, N}, ZIds_val)

"""
    SMatrixWithZeros(sv::StaticVectorOrNTuple, S::Tuple{M, N}, ZIds_val) where {M, N}

Create an `SMatrixWithZeros` from a partial `StaticVectorOrNTuple` of non-zero entries, the
total size including zeros `(M, N)`, and a `Val` of 2D zero locations.
"""
@inline SMatrixWithZeros(
    sv::StaticVectorOrNTuple, S::Type{Tuple{M, N}}, ZIds_val
) where {M, N} = SArrayWithZeros(sv, S, ZIds_val)

"""
    SVectorWithZeros(sv::StaticVectorOrNTuple, ::Val{N}, ZIds_val) where N

Create an `SVectorWithZeros` from a partial `StaticVectorOrNTuple` of non-zero entries, the
total length including zeros `N`, and a `Val` of 1D zero locations.
"""
@inline SVectorWithZeros(sv::StaticVectorOrNTuple, ::Val{N}, ZIds_val) where N = 
    SArrayWithZeros(sv, Tuple{N}, ZIds_val)

"""
    SVectorWithZeros(sv::StaticVectorOrNTuple, S::Type{Tuple{N}}, ZIds_val) where N

Create an `SVectorWithZeros` from a partial `StaticVectorOrNTuple` of non-zero entries, the
total length including zeros `N`, and a `Val` of 1D zero locations.
"""
@inline SVectorWithZeros(sv::StaticVectorOrNTuple, S::Type{Tuple{N}}, ZIds_val) where N = 
    SArrayWithZeros(sv, S, ZIds_val)

######################
# From StaticArray

"""
    create_sarray_with_zeros_indices_1D_to_DD(S, ZIds)

Create an `SVector{⋅, CartesianIndex{D}}` containing at entry `i` the `D`-D coordinates
stored by a `SArrayWithZeros{S, ..., ZIds}` data vector at 1D coordinate `i`.
"""
function create_sarray_with_zeros_indices_1D_to_DD(S, ZIds)
    sz = fieldtypes(S)
    L = prod(sz) - length(ZIds)
    D = length(sz)
    indices_1D_to_DD = MVector{L, CartesianIndex{D}}(undef)
    CI = CartesianIndices(sz)
    ctr = 1
    for I in CI
        if !(Tuple(I) in ZIds)
            indices_1D_to_DD[ctr] = I
            ctr += 1
        end
    end
    return SVector(indices_1D_to_DD)
end

"""
   create_sarray_with_zeros_indices_DD_to_1D(S, ZIds)

Create an `SArray{S, Int}` containing at `D`-D index `I` the 1D index in a
`SArrayWithZeros`'s data vector which represents this array position. At the zero locations
`ZIds` we put the invalid 1D index `0`.
"""
function create_sarray_with_zeros_indices_DD_to_1D(S, ZIds)
    indices_DD_to_1D = MArray{S, Int}(undef)
    sz = fieldtypes(S)
    CI = CartesianIndices(sz)
    ctr = 1
    for I in CI
        if Tuple(I) in ZIds
            indices_DD_to_1D[I] = 0
        else
            indices_DD_to_1D[I] = ctr
            ctr += 1
        end
    end
    return SArray(indices_DD_to_1D)
end

"""
    sarray_with_zeros_from_static_array_expression(
        sa_symb, S, T, ZIds
    )

Return the expression for creating a `SArrayWithZeros{S, T, ..., ZIds}` from a `StaticArray`
by extracting the `data_vector` and calling the corresponding constructor (`@inbounds`).
"""
function sarray_with_zeros_from_static_array_expression(
    sa_symb, S, T, ZIds
)
    sz = fieldtypes(S)
    L = prod(sz) - length(ZIds)
    _check_sawz_args(S, ZIds, L)
    # (We don't need to canonicalize ZIds, as this will be done in the SArrayWithZeros call)

    indices_1D_to_DD = create_sarray_with_zeros_indices_1D_to_DD(S, ZIds)
    non_zero_element_exprs = [:($sa_symb[$I]) for I in indices_1D_to_DD]
    return :(), :(
        @inbounds SArrayWithZeros(
            SVector{$L, $T}($(non_zero_element_exprs...)), 
            $S, $(Val(ZIds))) 
    )
end


"""
    SArrayWithZeros(
        sa::StaticArray{S, T}, ::Val{ZIds}
    ) where {S, T, ZIds}

Create an `SArrayWithZeros` from a full `StaticArray` `sa` and the zero locations `ZIds`.
It does not matter whether the input `sa` actually has zero values at these locations. 
The data vector type will be `SVector`.
"""
@inline @generated SArrayWithZeros(
    sa::StaticArray{S, T}, ::Val{ZIds}
) where {S, T, ZIds} = _sarray_with_zeros_code(sa, :sa, ZIds)

_sarray_with_zeros_code(
    A::Type{<:StaticArray{S, T}}, sa_symb, ZIds
) where {S, T} = adjust_indexing_expression(
    sarray_with_zeros_from_static_array_expression(sa_symb, S, T, ZIds),
    (A, sa_symb);
    finalize = true
)
    
"""
    SMatrixWithZeros(sm::StaticMatrix, ZIds_val)

Create an `SMatrixWithZeros` from a (full) `StaticMatrix` and a `Val` of 2D zero locations.
"""
@inline SMatrixWithZeros(sm::StaticMatrix, ZIds_val) = SArrayWithZeros(sm, ZIds_val)

"""
    SVectorWithZeros(sv::StaticVector, ZIds_val)

Create an `SVectorWithZeros` from a full `StaticVector` and a `Val` of 1D zero
locations.
"""
@inline SVectorWithZeros(sv::StaticVector, ZIds_val) = 
    SArrayWithZeros(sv, ZIds_val)


# From StaticArray
######################
# Indexing

"""
    adjust_indexing_expression(
        split_expr, ::Type{<:SArrayWithZeros{S, T, D, ZIds, V}}, sawz_symb
    ) where S

Rewrite the split `expr` containing an `SArrayWithZeros` represented as `sawz_symb` by
replacing the `D`-D indexing by 1D indexing of the underlying data vector or by explcit
zeros at `ZIds`.

!!! The indexing expressions in `expr_body` (`split_expr[2]`) cannot contain variables.
That is, e.g. `M[k, 1]` is not allowed. 
"""
function adjust_indexing_expression(
    split_expr, ::Type{<:SArrayWithZeros{S, T, <:Any, ZIds, V}}, sawz_symb
) where {S, T, ZIds, V}
    expr_head, expr_body = split_expr
    sawz_data_vector_symb = gensym(Symbol(sawz_symb, :_data_vector))    
    expr_head = join_exprs((
        expr_head,
        :($sawz_data_vector_symb = get_data_vector($sawz_symb))
    ))

    indices_DD_to_1D = create_sarray_with_zeros_indices_DD_to_1D(S, ZIds)
    expr_body = MacroTools.postwalk(expr_body) do ex
        if @capture(ex, A_[I__]) && A === sawz_symb
            tI = Tuple(I)
            if tI in ZIds
                :($(zero(T)))
                # The compiler can mostly figure out it does not need to compute 0 * ...,
                # though it does not fully propagate it. E.g. det of a SMatrixWithZeros
                # with a zero row does not just return 0.
                # To account for this (and to produce nicer expressions in general) we have
                # adjust_zeros_expression.
            else
                :($sawz_data_vector_symb[$(indices_DD_to_1D[tI...])])
            end
        else
            ex
        end
    end
    return adjust_indexing_expression(
        (expr_head, expr_body), V, sawz_data_vector_symb)
end

# Indexing
######################
# Basic methods and broadcasting

@inline SArrayWithZeros(sawz::SArrayWithZeros) = sawz
@inline SMatrixWithZeros(smwz::SMatrixWithZeros) = smwz
@inline SVectorWithZeros(svwz::SVectorWithZeros) = svwz

@inline function Base.:(==)(
    sawz_A::SArrayWithZeros{S_A, <:Any, <:Any, ZIds_A},
    sawz_B::SArrayWithZeros{S_B, <:Any, <:Any, ZIds_B}
) where {S_A, S_B, ZIds_A, ZIds_B} 
    return S_A == S_B && ZIds_A == ZIds_B && sawz_A.data_vector == sawz_B.data_vector
    # Then also L, T and D are equal.
    # Note that we need ZIds_A and ZIds_B to be in canonical format. If one bypasses the
    # canonicalization step by using the inner constructor directly, then 
    # ZIds_A = ((1, 2), (2, 1)) and ZIds_B = ((2, 1), (1, 2)) will cause sawz_A and sawz_B
    # to also be different.
end

@inline Base.:+(sawz) = sawz
@inline Base.:-(sawz::SArrayWithZeros{S, T, D, ZIds, L, V}) where {S, T, D, ZIds, L, V} = 
    SArrayWithZeros{S, T, D, ZIds, L, V}(-get_data_vector(sawz))
# (This assumes that -(::V) is internal (again of type V). Similar statements apply below)

# Addition and subtraction of SArrayWithZeros of the same size:
# known zeros at intersection of the zero positions

# In particular
@inline Base.:+(
    A::SArrayWithZeros{S, T, D, ZIds, L, V},
    B::SArrayWithZeros{S, T, D, ZIds, L, V}
) where {S, T, D, ZIds, L, V <: SVectorOrNTuple{L, T}} = 
    SArrayWithZeros{S, T, D, ZIds, L, V}(
        get_data_vector(A) + get_data_vector(B)
    )
# We need the V constraint to be stronger than just ::SArrayWithZeros (later)

@inline Base.:-(
    A::SArrayWithZeros{S, T, D, ZIds, L, V},
    B::SArrayWithZeros{S, T, D, ZIds, L, V}
) where {S, T, D, ZIds, L, V <: SVectorOrNTuple{L, T}} =
    SArrayWithZeros{S, T, D, ZIds, L, V}(
        get_data_vector(A) - get_data_vector(B)
    )

# More generally
# Combine componentwise_sarray_operation and `SArrayWithZeros(::SArray, .)`
"""
    componentwise_sarray_with_zeros_operation_expression(
        op_symb, 
        ::Type{<:SArrayWithZeros{S}}, ::Type{<:SArrayWithZeros{S}},
        A_symb, B_symb,
        new_ZIds
    ) where S

Return the (split) expression creating the `SArrayWithZeros{S, <:Any, D, new_ZIds}` obtained
by applying the operation represented by `op_symb` (e.g. :+) componentwise on the
`SArrayWithZeros` subtypes (`A` and `B`), the instances of which will be represented by
`A_symb` and `B_symb`. This method assumes we have computed the zero indices `new_ZIds` in
advanced given the zero indices from `A` and `B`.
"""
function componentwise_sarray_with_zeros_operation_expression(
    op_symb, 
    ::Type{<:SArrayWithZeros{S}}, ::Type{<:SArrayWithZeros{S}},
    A_symb, B_symb,
    new_ZIds
) where S   
    A_op_B_exprs_array = componentwise_sarray_operation_expression_array(
        op_symb, A_symb, B_symb, S)
    indices_1D_to_DD = create_sarray_with_zeros_indices_1D_to_DD(S, new_ZIds)
    non_zero_element_exprs = [:($(A_op_B_exprs_array[I])) for I in indices_1D_to_DD]
    return :(), :(
        @inbounds SArrayWithZeros(
            SVector{$(length(non_zero_element_exprs))}($(non_zero_element_exprs...)), 
            $S, $(Val(new_ZIds)))
    )
end

"""
    componentwise_sarray_with_zeros_operation(
        op::OP, 
        A::SArrayWithZeros{S}, B::SArrayWithZeros{S},
        A_symb, B_symb,
        ::Val{new_ZIds}
    ) where {OP <: Function, S, D, ZIds_A, ZIds_B, new_ZIds}

Create the `SArrayWithZeros{S, ..., new_ZIds}` obtained by applying the operation `op`
(e.g. `+`) componentwise on the `SArrayWithZeros` `A` and `B`. This method assumes we have
computed the zero indices `new_ZIds` in advance from the zero indices of `A` and `B`.
"""
@inline @generated componentwise_sarray_with_zeros_operation(
    ::OP, 
    A::SArrayWithZeros{S}, B::SArrayWithZeros{S},
    A_symb, B_symb,
    ::Val{new_ZIds}
) where {OP <: Function, S, new_ZIds} = _componentwise_sarray_with_zeros_operation_code(
    OP, A, B, A_symb, B_symb, new_ZIds
)

_componentwise_sarray_with_zeros_operation_code(
    ::Type{OP}, 
    A::Type{<:SArrayWithZeros{S}}, B::Type{<:SArrayWithZeros{S}},
    A_symb, B_symb,
    new_ZIds
) where {OP <: Function, S} = adjust_indexing_expression(
    componentwise_sarray_with_zeros_operation_expression(
        Symbol(OP.instance), A, B, A_symb, B_symb, new_ZIds
    ),
    (A, A_symb), (B, B_symb);
    finalize = true
)


"""
    componentwise_sarray_with_zeros_operation_intersection_expression(
        op_symb, 
        A::Type{<:SArrayWithZeros{S, <:Any, D, ZIds_A}},
        B::Type{<:SArrayWithZeros{S, <:Any, D, ZIds_B}},
        A_symb, B_symb
    ) where {S, D, ZIds_A, ZIds_B}

Like [`componentwise_sarray_with_zeros_operation_expression`](@ref), but
specifically when the `new_ZIds` is the intersection of `ZIds_A` and `ZIds_B`. 
"""
function componentwise_sarray_with_zeros_operation_intersection_expression(
    op_symb, 
    A::Type{<:SArrayWithZeros{S, <:Any, D, ZIds_A}},
    B::Type{<:SArrayWithZeros{S, <:Any, D, ZIds_B}},
    A_symb, B_symb
) where {S, D, ZIds_A, ZIds_B}
    new_ZIds = Tuple(intersect(ZIds_A, ZIds_B))
    # Might be empty, but that's fine
    return componentwise_sarray_with_zeros_operation_expression(
        op_symb, A, B, A_symb, B_symb, new_ZIds
    )
end

"""
    componentwise_sarray_with_zeros_operation_intersection(
        ::OP, 
        A::SArrayWithZeros{S, <:Any, D, ZIds_A}, B::SArrayWithZeros{S, <:Any, D, ZIds_B},
    ) where {OP <: Function, S, D, ZIds_A, ZIds_B}

Create the `SArrayWithZeros{S}` obtained by applying the operation `op` (e.g. `+`)
componentwise on the `SArrayWithZeros` `A` and `B`, when the output `SArrayWithZeros`'s
zero indices is the intersection of those of `A` and `B` (i.e. `ZIds_A` and `ZIds_B`).
"""
@inline @generated componentwise_sarray_with_zeros_operation_intersection(
    ::OP, 
    A::SArrayWithZeros{S, <:Any, D, ZIds_A}, B::SArrayWithZeros{S, <:Any, D, ZIds_B},
) where {OP <: Function, S, D, ZIds_A, ZIds_B} = 
    _componentwise_sarray_with_zeros_operation_intersection_code(OP, A, B, :A, :B)

_componentwise_sarray_with_zeros_operation_intersection_code(
    ::Type{OP}, 
    A::Type{<:SArrayWithZeros{S, <:Any, D, ZIds_A}},
    B::Type{<:SArrayWithZeros{S, <:Any, D, ZIds_B}},
    A_symb, B_symb
) where {OP <: Function, S, D, ZIds_A, ZIds_B} = adjust_indexing_expression(
    componentwise_sarray_with_zeros_operation_intersection_expression(
        Symbol(OP.instance), A, B, A_symb, B_symb
    ),
    (A, A_symb), (B, B_symb);
    finalize = true
)
# Alternatively we could have directly called componentwise_sarray_with_zeros_operation
# for the intersection statically computed in the @generated. But we will follow our
# established approach of 1) expression generation 2) rewritten _code 3) @generated,
# as it makes it a bit easier to inspect the concrete generated code
# (It is more verbose, though)


"""
    componentwise_sarray_with_zeros_operation_union_expression(
        op_symb, 
        A::Type{<:SArrayWithZeros{S, <:Any, D, ZIds_A}},
        B::Type{<:SArrayWithZeros{S, <:Any, D, ZIds_B}},
        A_symb, B_symb
    ) where {S, D, ZIds_A, ZIds_B}

Like [`componentwise_sarray_with_zeros_operation_expression`](@ref), but
specifically when the `new_ZIds` is the union of `ZIds_A` and `ZIds_B`. 
"""
function componentwise_sarray_with_zeros_operation_union_expression(
    op_symb, 
    A::Type{<:SArrayWithZeros{S, <:Any, D, ZIds_A}},
    B::Type{<:SArrayWithZeros{S, <:Any, D, ZIds_B}},
    A_symb, B_symb
) where {S, D, ZIds_A, ZIds_B}
    new_ZIds = Tuple(unique((ZIds_A..., ZIds_B...)))
    return componentwise_sarray_with_zeros_operation_expression(
        op_symb, A, B, A_symb, B_symb, new_ZIds
    )
end

"""
    componentwise_sarray_with_zeros_operation_union(
        ::OP, 
        A::SArrayWithZeros{S, <:Any, D, ZIds_A}, B::SArrayWithZeros{S, <:Any, D, ZIds_B},
    ) where {OP <: Function, S, D, ZIds_A, ZIds_B}

Create the `SArrayWithZeros{S}` obtained by applying the operation `op` (e.g. `*`)
componentwise on the `SArrayWithZeros` `A` and `B`, when the output `SArrayWithZeros`'s
zero indices is the union of those of `A` and `B` (i.e. `ZIds_A` and `ZIds_B`).
"""
@inline @generated componentwise_sarray_with_zeros_operation_union(
    ::OP, 
    A::SArrayWithZeros{S, <:Any, D, ZIds_A}, B::SArrayWithZeros{S, <:Any, D, ZIds_B},
) where {OP <: Function, S, D, ZIds_A, ZIds_B} = 
    _componentwise_sarray_with_zeros_operation_union_code(OP, A, B, :A, :B)

_componentwise_sarray_with_zeros_operation_union_code(
    ::Type{OP}, 
    A::Type{<:SArrayWithZeros{S, <:Any, D, ZIds_A}},
    B::Type{<:SArrayWithZeros{S, <:Any, D, ZIds_B}},
    A_symb, B_symb
) where {OP <: Function, S, D, ZIds_A, ZIds_B} = adjust_indexing_expression(
    componentwise_sarray_with_zeros_operation_union_expression(
        Symbol(OP.instance), A, B, A_symb, B_symb
    ),
    (A, A_symb), (B, B_symb);
    finalize = true
)


@inline Base.:+(A::SArrayWithZeros, B::SArrayWithZeros) =
    componentwise_sarray_with_zeros_operation_intersection(+, A, B)
# (With lower precedence than the version with matching ZIds)

@inline Base.:-(A::SArrayWithZeros, B::SArrayWithZeros) =
    componentwise_sarray_with_zeros_operation_intersection(-, A, B)


# Multiplication and division by a scalar is again easy
@inline Base.:*(
    sawz::SArrayWithZeros{S, T, D, ZIds, L, V}, x::Number
) where {S, T, D, ZIds, L, V} = SArrayWithZeros{S, T, D, ZIds, L, V}(
    get_data_vector(sawz) * x
)
@inline Base.:*(
    x::Number, sawz::SArrayWithZeros{S, T, D, ZIds, L, V}
) where {S, T, D, ZIds, L, V} = SArrayWithZeros{S, T, D, ZIds, L, V}(
    x * get_data_vector(sawz)
)
@inline Base.:/(
    sawz::SArrayWithZeros{S, T, D, ZIds, L, V}, x::Number
) where {S, T, D, ZIds, L, V} = SArrayWithZeros{S, T, D, ZIds, L, V}(
    get_data_vector(sawz) / x
)
@inline Base.:\(
    x::Number, sawz::SArrayWithZeros{S, T, D, ZIds, L, V}
) where {S, T, D, ZIds, L, V} = SArrayWithZeros{S, T, D, ZIds, L, V}(
    x \ get_data_vector(sawz)
)

# Most other operations do not result in an SArrayWithZeros
# Broadcasting is also not internal for the most part, except for the .+ already defined
# above via + and a similar .*
Base.Broadcast.broadcasted(::typeof(+), A::SArrayWithZeros, B::SArrayWithZeros) = A + B
# (This might break broadcast fusion, but it probably does not matter?)
Base.Broadcast.broadcasted(::typeof(-), A::SArrayWithZeros, B::SArrayWithZeros) = A - B
Base.Broadcast.broadcasted(::typeof(*), A::SArrayWithZeros, B::SArrayWithZeros) =
    componentwise_sarray_with_zeros_operation_union(*, A, B)
Base.Broadcast.broadcasted(::typeof(*), A::SArrayWithZeros, x::Number) = A * x
Base.Broadcast.broadcasted(::typeof(*), x::Number, A::SArrayWithZeros) = x * A
Base.Broadcast.broadcasted(::typeof(/), A::SArrayWithZeros, x::Number) = A / x
Base.Broadcast.broadcasted(::typeof(\), x::Number, A::SArrayWithZeros) = x \ A


# Basic methods and broadcasting
######################
# Transpose and adjoint

"""
    adjoint_or_transposed_smatrix_generation_expression(
        ::Type{<:SArrayWithZeros{S, T, D, ZIds, L, V}}, sawz_symb, use_transpose
    ) where {S, T}

More specific method of `adjoint_or_transposed_smatrix_generation_expression`
for `SArrayWithZeros`. The output expression (body) creates a `SArrayWithZeros`.
"""
function adjoint_or_transposed_smatrix_generation_expression(
    ::Type{<:SArrayWithZeros{S, T, D, ZIds, L, V}}, sawz_symb, use_transpose
) where {S, T, D, ZIds, L, V}
    # Running examples D = 2: S = Tuple{2, 3}, ZIds = ((1, 2),), L = 5, use_adjoint = false
    #                  D = 1: S = Tuple{3},    ZIds = ((2,),),   L = 2, use_adjoint = false
    # In both cases we use sawz_symb = :sawz
    at_matrix_exprs = adjoint_or_transposed_smatrix_generation_expressions_matrix(
        sawz_symb, S, use_transpose)
    # Matrix{Expr}
    # D = 2: [:(sawz[1, 1])  :(sawz[2, 1])
    #         :(sawz[1, 2])  :(sawz[2, 2])
    #         :(sawz[1, 3])  :(sawz[2, 3])
    #
    # D = 1: [:(sawz[1])  :(sawz[2])  :(sawz[3])]
    N, M = size(at_matrix_exprs)  # N == 1 if D == 1

    # Remove the ZIds entries (i.e. the reversed indices in the transpose)
    if D == 2
        ZIds_T = reverse.(ZIds)
        # ((2, 1),)
    else  # D == 1
        ZIds_T = map(t -> (1, t[1]), ZIds)
        # ((1, 2),)
    end
    at_non_zero_element_exprs = [
        :($(at_matrix_exprs[I])) 
        for I = CartesianIndices(at_matrix_exprs)
        if !(Tuple(I) in ZIds_T)
    ]
    # Vector{Expr}
    # D = 2: [:(sawz[1, 1]), :(sawz[1, 3]), :(sawz[2, 1]), :(sawz[2, 2]), :(sawz[2, 3])]
    # D = 1: [:(sawz[1]), :(sawz[3])]
    return (:(),
        :(
            @inbounds SArrayWithZeros{Tuple{$N, $M}, $T, 2, $ZIds_T, $L, $V}(
                $V(tuple($(at_non_zero_element_exprs...)))
            )
        )
    )  # Use the inner constructor
end
# The index rewriting will be done automatically in _adjoint_or_transpose_code.
# Forwarding to transpose and adjoint is also already done via the SpecialIndexingSArray


### SArrayWithZeros
############################################################################################
### ScaledSArray


"""
    ScaledSArray{S, T, D, F, A <: StaticArray{S, T, D}} <: SpecialSArray{S, T, D}

Struct for lazily representing the product of a `StaticArray` (of type `A`) with a scale
factor (of unrelated type `F`, expected to be scalar). 

This can be beneficial in situations like scale matrix - vector product. For example, if `s`
is a scalar, `sm` a `(M, N)` (static) matrix and `sv` a length-`N` vector, then
`s * sm * sv` (i.e. `(s * sm) * sv`) requires `M * N` multiplications with `s`, while
`SMatrix(ScaledSMatrix(s, sm) * sv)` delays the multiplications with `s` until the end
in the `SMatrix` conversion, requiring only `N` multiplications. On the other hand for
matrix-matrix multiplications with increasing dimensions, delaying it means we need more
multiplications. Additionally, if we do not convert back to a `SArray`, a `ScaledSArray`
takes up more memory, which especially at smaller sizes might also make everything
noticeably slower. For these reason we do not eagerly introduce `ScaledSArrays` (e.g. in
`::Number * ::SpecialSArray` or in `inv`, to avoid dividing by the determinant). The user
must create them explicitly (via `ScaledSArray(::Number, ::StaticArray)` and 
`unrolled_inv(⋅, Val(true))`).

---

    ScaledSArray{S, T, D, F, A}(
        scale_factor, sa
    ) where {S, T, D, F, A <: StaticArray{S, T, D}}

Create a `ScaledSArray{S, T, D, F, A}` from a scale factor and `StaticArray`.

---

    ScaledSArray(
        scale_factor::F, sa::A
    ) where {S, T, D, F, A <: StaticArray{S, T, D}}

Create a `ScaledSArray` from a scale factor and `StaticArray`. The `ScaledSArray`
parameters are taken over from the argument.
"""
struct ScaledSArray{S, T, D, F, A <: StaticArray{S, T, D}} <: SpecialSArray{S, T, D}
    scale_factor::F
    sa::A

    @inline function ScaledSArray{S, T, D, F, A}(
            scale_factor, sa
        ) where {S, T, D, F, A <: StaticArray{S, T, D}}
            return new{S, T, D, F, A}(scale_factor, sa)
        end

    @inline function ScaledSArray(
        scale_factor::F, sa::A
    ) where {S, T, D, F, A <: StaticArray{S, T, D}}
        return new{S, T, D, F, A}(scale_factor, sa)
    end
end

"""
   ScaledSMatrix{M, N, T, F, A}
   
2D `ScaledSArray` of static size `NTuple{M, N}`.
"""
const ScaledSMatrix{M, N, T, F, A} = ScaledSArray{Tuple{M, N}, T, 2, F, A}

"""
   ScaledSVector{N, T, F, A}
   
1D `ScaledSArray` of static size `NTuple{N}`.
"""
const ScaledSVector{N, T, F, A} = ScaledSArray{Tuple{N}, T, 1, F, A}

"""
    ScaledSMatrix(scale_factor, sm::StaticMatrix)

Create a `ScaledSMatrix` from a scale factor and `StaticMatrix`.
"""
@inline ScaledSMatrix(scale_factor, sm::StaticMatrix) = ScaledSArray(scale_factor, sm)

"""
    ScaledSVector(scale_factor, sv::StaticVector)

Create a `ScaledSVector` from a scale factor and `StaticVector`.
"""
@inline ScaledSVector(scale_factor, sv::StaticVector) = ScaledSArray(scale_factor, sv)

@inline get_scale_factor(ssa::ScaledSArray) = ssa.scale_factor
@inline get_underlying_array(ssa::ScaledSArray) = ssa.sa
@inline get_underlying_matrix(ssm::ScaledSMatrix) = get_underlying_array(ssm)
@inline get_underlying_vector(ssv::ScaledSVector) = get_underlying_array(ssv)

@inline get_scale_factor_type(::ScaledSArray{<:Any, <:Any, <:Any, F}) where F = F
@inline get_underlying_array_type(::ScaledSArray{<:Any, <:Any, <:Any, <:Any, A}) where A = A

Base.@propagate_inbounds Base.getindex(ssa::ScaledSArray, inds::Int...) = 
    get_scale_factor(ssa) * get_underlying_array(ssa)[inds...]

@inline StaticArrays.SArray(ssa::ScaledSArray) = 
    get_scale_factor(ssa) .* SArray(get_underlying_array(ssa))
# Conceivably slightly faster than the ::StaticArray version relying on getindex
# Also note that .sa might be e.g. an SArrayWithZeros
@inline StaticArrays.SMatrix(ssm::ScaledSMatrix) = 
    get_scale_factor(ssm) .* SMatrix(get_underlying_matrix(ssm))
@inline StaticArrays.SVector(ssv::ScaledSVector) = 
    get_scale_factor(ssv) .* SVector(get_underlying_vector(ssv))

######################
# Matrix multiplication

@inline matmul(A::ScaledSArray, B::ScaledSArray) = 
    ScaledSArray(
        get_scale_factor(A) * get_scale_factor(B), 
        get_underlying_array(A) * get_underlying_array(B))
@inline matmul(A::ScaledSArray, B::StaticArray) = 
    ScaledSArray(get_scale_factor(A), get_underlying_array(A) * B)
@inline matmul(A::StaticArray, B::ScaledSArray) = 
    ScaledSArray(get_scale_factor(B), A * get_underlying_array(B))

# To resolve ambiguity, we also need to declare ::SpecialIndexingSArray variants
@inline matmul(A::ScaledSArray, B::SpecialIndexingSArray) = 
    ScaledSArray(get_scale_factor(A), get_underlying_array(A) * B)
@inline matmul(A::SpecialIndexingSArray, B::ScaledSArray) = 
    ScaledSArray(get_scale_factor(B), A * get_underlying_array(B))


# Matrix multiplication
######################
# Adjoint and transpose

@inline LinearAlgebra.adjoint(A::ScaledSArray) =
    ScaledSArray(get_scale_factor(A), adjoint(get_underlying_array(A)))
@inline LinearAlgebra.transpose(A::ScaledSArray) =
    ScaledSArray(get_scale_factor(A), transpose(get_underlying_array(A)))

# For ::ScaledSVector we have a problem: Adjoint(⋅, SVector) cannot be (implicitly)
# converted to an SMatrix (in contrast to Adjoint(⋅, SMatrix)).
@inline LinearAlgebra.adjoint(
    A::ScaledSVector{N, <:Any, <:Any, V}
) where {N, V <: SVector} =
    ScaledSArray(get_scale_factor(A), SMatrix{1, N}(adjoint(get_underlying_vector(A))))
@inline LinearAlgebra.transpose(
    A::ScaledSVector{N, <:Any, <:Any, V}
) where {N, V <: SVector} =
    ScaledSArray(get_scale_factor(A), SMatrix{1, N}(transpose(get_underlying_vector(A))))

# For other V <: StaticVector we might get errors, though not for <: SymbolicIndexingSVector
# as we have internal transpose and adjoints (yielding new SymbolicIndexingSArrays)

# Adjoint and transpose
######################
# Other matrix-level methods
# (even though we define them for ScaledSArrays and let the underlying calls on the
#  .sa error if we are not dealing with matrices)

@inline LinearAlgebra.det(A::ScaledSArray{S}) where S =
    get_scale_factor(A) ^ first(fieldtypes(S)) * det(get_underlying_array(A))
    # i.e. in practice A::ScaledSMatrix{N, N} and we use factor ^ N

@inline LinearAlgebra.tr(A::ScaledSArray) = 
    get_scale_factor(A) * tr(get_underlying_array(A))
@inline Base.inv(A::ScaledSArray) = 
    ScaledSArray(inv(get_scale_factor(A)), inv(get_underlying_array(a)))
@inline LinearAlgebra.norm(A::ScaledSArray, p::Real = 2) = 
    abs(get_scale_factor(A)) * norm(get_underlying_array(A), p)
@inline LinearAlgebra.opnorm(A::ScaledSArray, p::Real = 2) = 
    abs(get_scale_factor(A)) * opnorm(get_underlying_array(A), p)
@inline LinearAlgebra.diag(A::ScaledSArray, n::Integer = 0) = 
    get_scale_factor(A) * diag(get_underlying_array(A), n)
@inline function LinearAlgebra.eigen(A::ScaledSArray; kwargs...)
    egn = eigen(get_underlying_array(A); kwargs...)
    return Eigen(get_scale_factor(A) .* egn.values, egn.vectors)
end
@inline LinearAlgebra.eigvals(A::ScaledSArray; kwargs...) = 
    get_scale_factor(A) .* eigvals(get_underlying_array(A); kwargs...)
@inline LinearAlgebra.eigvecs(A::ScaledSArray; kwargs...) = eigvecs(
    get_underlying_array(A); kwargs...)

@inline LinearAlgebra.cross(A::ScaledSArray, B::StaticArray) =
    ScaledSArray(get_scale_factor(A), cross(get_underlying_array(A), B))
@inline LinearAlgebra.cross(A::StaticArray, B::ScaledSArray) =
    ScaledSArray(get_scale_factor(B), cross(A, get_underlying_array(B)))
@inline LinearAlgebra.cross(A::ScaledSArray, B::ScaledSArray) =
    ScaledSArray(
        get_scale_factor(A) * get_scale_factor(B), 
        cross(get_underlying_array(A), get_underlying_array(B)))
@inline LinearAlgebra.dot(A::ScaledSArray, B::StaticArray) =
    adjoint(get_scale_factor(A)) * dot(get_underlying_array(A), B)
    # Note that Julia adopts the convention that the inner product is linear in the second
    # component and conjugate linear in the first. (Blasphemy!)
    # We just adjoint instead of conj just in case the code would also work when the scale
    # factor is a matrix
@inline LinearAlgebra.dot(A::StaticArray, B::ScaledSArray) =
    get_scale_factor(B) * dot(A, get_underlying_array(B))
@inline LinearAlgebra.dot(A::ScaledSArray, B::ScaledSArray) =
    adjoint(get_scale_factor(A)) * get_scale_factor(B) *dot(
        get_underlying_array(A), get_underlying_array(B))
@inline _kron(A::ScaledSArray, B::StaticArray) = 
    ScaledSArray(get_scale_factor(A), _kron(get_underlying_array(A), B))
@inline _kron(A::StaticArray, B::ScaledSArray) = 
    ScaledSArray(get_scale_factor(B), _kron(A, get_underlying_array(B)))
@inline _kron(A::ScaledSArray, B::ScaledSArray) = 
    ScaledSArray(
        get_scale_factor(A) * get_scale_factor(B), 
        _kron(get_underlying_array(A), get_underlying_array(B)))

@inline LinearAlgebra.triu(A::ScaledSArray, k::Int = 0) = ScaledSArray(
    get_scale_factor(A), triu(get_underlying_array(A), k))
@inline LinearAlgebra.tril(A::ScaledSArray, k::Int = 0) = ScaledSArray(
    get_scale_factor(A), tril(get_underlying_array(A), k))

# Other matrix-level methods
######################
# Basic methods and broadcasting


@inline Base.:*(A::ScaledSArray, x::Number) = ScaledSArray(
    get_scale_factor(A) * x, get_underlying_array(A))
@inline Base.:*(x::Number, A::ScaledSArray) = ScaledSArray(
    x * get_scale_factor(A), get_underlying_array(A))
@inline Base.:/(A::ScaledSArray, x::Number) = ScaledSArray(
    get_scale_factor(A) / x, get_underlying_array(A))
@inline Base.:\(x::Number, A::ScaledSArray) = ScaledSArray(
    x \ get_scale_factor(A), get_underlying_array(A))

Base.Broadcast.broadcasted(::typeof(*), A::ScaledSArray, x::Number) = A * x
Base.Broadcast.broadcasted(::typeof(*), x::Number, A::ScaledSArray) = x * A
Base.Broadcast.broadcasted(::typeof(/), A::ScaledSArray, x::Number) = A / x
Base.Broadcast.broadcasted(::typeof(\), x::Number, A::ScaledSArray) = x \ A

Base.Broadcast.broadcasted(::typeof(*), A::ScaledSArray, B::ScaledSArray) =
    ScaledSArray(
        get_scale_factor(A) * get_scale_factor(B), 
        get_underlying_array(A) .* get_underlying_array(B))
Base.Broadcast.broadcasted(::typeof(^), A::ScaledSArray, p) =
    ScaledSArray(
        get_scale_factor(A) ^ p, 
        get_underlying_array(A) .^ p)
Base.Broadcast.broadcasted(::typeof(Base.literal_pow), _, A::ScaledSArray, p_val) =
    ScaledSArray(
        Base.literal_pow(^, get_scale_factor(A), p_val), 
        Base.literal_pow.(^, get_underlying_array(A), p_val))
# The first function argument is Base.RefValue{typeof(^)}(^), of type
# Base.RefValue{typeof(^)}

# Basic methods and broadcasting
######################
# Interactions

# We already had Base.inv(::SelfAdjointOrSymmetricSMatrix) earlier

# Congruence transform with ScaledSMatrices can make optimal use of the scale factors to
# produce a SymmetricSMatrix{..., ScaledSVector}
unrolled_smatrix_multiplication_lower_expression(
    A::Type{<:ScaledSArray}, B::Type{<:StaticArray}, args...
) = unrolled_smatrix_multiplication_expression_scaled_sarray(A, B, args...)
smatrix_multiplication_lower_expression(
    A::Type{<:StaticArray}, B::Type{<:ScaledSArray}, args...
) = unrolled_smatrix_multiplication_expression_scaled_sarray(A, B, args...)
smatrix_multiplication_lower_expression(
    A::Type{<:ScaledSArray}, B::Type{<:ScaledSArray}, args...
) = unrolled_smatrix_multiplication_expression_scaled_sarray(A, B, args...)

"""
    unrolled_smatrix_multiplication_expression_scaled_sarray(
        A::Type, B::Type, 
        A_symb, B_symb, T, 
        use_adjoint_or_transposed_A, use_transpose_A,
        use_adjoint_or_transposed_B, use_transpose_B
    )

More specific version of [`unrolled_smatrix_multiplication_lower_expression`](@ref) when we
know at least one of the input array (matrix) types is a ScaledSArray.

We generate the expression for the underlying arrays, then wrap the result expression in a
`ScaledSVector`.
"""
function unrolled_smatrix_multiplication_expression_scaled_sarray(
    A::Type, B::Type, 
    A_symb, B_symb, T, 
    use_adjoint_or_transposed_A, use_transpose_A,
    use_adjoint_or_transposed_B, use_transpose_B
)
    # Attempt to go to ScaledSArray.sa, if we actually have a ScaledSArray
    function _get_underlying(type::Type{T}, symb) where T
        if T <: ScaledSArray
            underlying_symb = gensym(Symbol(symb, :_sa))
            underlying_defs = :($underlying_symb = get_underlying_array($symb))
            underlying_array_type = _get_underlying_array_type(T)
        else
            underlying_symb = symb
            underlying_defs = :()
            underlying_array_type = type
        end
        return underlying_symb, underlying_defs, underlying_array_type
    end

    _get_underlying_array_type(
        ::Type{<:ScaledSArray{<:Any, <:Any, <:Any, <:Any, UA}}
    ) where UA = UA

    underlying_A_symb, underlying_A_defs, underlying_A_array_type = _get_underlying(
        A, A_symb)
    underlying_B_symb, underlying_B_defs, underlying_B_array_type = _get_underlying(
        B, B_symb)
    
    # Generate expression of lower half matmul using the underlying arrays
    inner_split_expr = unrolled_smatrix_multiplication_lower_expression(
        underlying_A_array_type, underlying_B_array_type,
        underlying_A_symb, underlying_B_symb, T,
        use_adjoint_or_transposed_A, use_transpose_A,
        use_adjoint_or_transposed_B, use_transpose_B
    ) 
    # head: declarations under underlying_A_symb
    # body: @inbounds SVector{...}(...)  with adjusted indexing
    expr_head = join_exprs((
        underlying_A_defs,
        underlying_B_defs,
        inner_split_expr[1]
    ))

    function _get_individual_scale_factor_expr_or_nothing(
        ::Type{T}, symb, use_adjoint_or_transposed, use_transpose
    )  where T
        T <: ScaledSArray || return nothing
        if use_adjoint_or_transposed && !use_transpose
            return :(adjoint(get_scale_factor($symb)))
        else
            return :(get_scale_factor($symb))
        end
    end
    scale_factor_expr_A = _get_individual_scale_factor_expr_or_nothing(
        A, A_symb, use_adjoint_or_transposed_A, use_transpose_A)
    scale_factor_expr_B = _get_individual_scale_factor_expr_or_nothing(
        B, B_symb, use_adjoint_or_transposed_B, use_transpose_B)
    scale_factor_expr = if !isnothing(scale_factor_expr_A) && 
                           !isnothing(scale_factor_expr_B)
        # i.e. both ScaledSArrays
        :($scale_factor_expr_A * $scale_factor_expr_B)
    elseif isnothing(scale_factor_expr_B)  # only A is a ScaledSArray
        scale_factor_expr_A
    else  # only B
        scale_factor_expr_B
    end
    expr_body = :(
        ScaledSVector($scale_factor_expr, $(inner_split_expr[2]))
    )
    return expr_head, expr_body
end