@inline remove_parameters(::Type{T}) where T = Core.typename(T).wrapper
@inline typeof_noparams(::T) where T = remove_parameters(T)

"""
   array_convert(::Type{S}, arr::AbstractArray{T, N}) where {S, T, N}
   
Convert `arr` to `Array{S, N}`. (Just returns `arr` when it is already of `eltype` `S`.)
"""
function array_convert(::Type{S}, arr::AbstractArray{T, N}) where {S, T, N}
    convert(Array{S, N}, arr)
end

"""
    indexlastaxis(A::AbstractArray{<:Any, N}, I) where N

Equivalent to A[:, :, ..., :, I].
"""
Base.@propagate_inbounds indexlastaxis(A::AbstractArray{<:Any, N}, I) where N = 
    A[ntuple(_ -> :, N - 1)..., I]


"""
    viewlastaxis(A::AbstractArray{<:Any, N}, I) where N

Equivalent to view(A, :, :, ..., :, I).
"""
Base.@propagate_inbounds viewlastaxis(A::AbstractArray{<:Any, N}, I) where N = 
    view(A, ntuple(_ -> :, N - 1)..., I)
# (or selectdim)


"""
    maybe_unsafe_free!(x)

Call `unsafe_free!` if `x` is a `CuArray` or has a custom implementation. Otherwise do
nothing.
"""
maybe_unsafe_free!(_) = nothing
maybe_unsafe_free!(x::CuArray) = unsafe_free!(x)


"""
    reuse_memory(x, new_size; free = true)

Wrap the first (contiguous) part of `x`'s memory into a new similar array of `size`
`new_size`, which we return. When `free` we call `maybe_unsafe_free!` on `x`.
Note that `unsafe_free!` does not deallocate memory still in use by another variable (the
one we return). So there is little downside to using `free`. Therefore, we default to
`true`.
"""
function reuse_memory(x, new_size; free = true)
    new_length = prod(new_size)
    new_length <= length(x) || 
        throw(ArgumentError("Can only reuse memory to produce smaller arrays!"))
    y = reshape(view(vec(x), Base.OneTo(Int64(new_length))), new_size)
    # An underlying CUDA.jl method expects Int64, not UInt64
    free && maybe_unsafe_free!(x)
    return y
end

"""
    resize_memory(x, new_size; reuse = true, free = true)

Return an array of `new_size`, where we either do not care about the values it contains
(`zero`, the default), or we place zeros (`!zero)`.

Normally, when `free` (`true` by default), we `maybe_unsafe_free!` the old one `x` and
allocate a new array. When shrinking the total `length` and if `reuse` (`true` by default),
we return a reshaped view of `x`.
"""
function resize_memory(x, new_size; zero = false, reuse = true, free = true)
    if reuse && prod(new_size) <= length(x)
        new_x = reuse_memory(x, new_size; free)
        zero && fill!(new_x, Base.zero(eltype(new_x)))
        return new_x
    else
        free && maybe_unsafe_free!(x)
        if zero
            return create_zeros(array_domain(x), eltype(x), new_size)
        else
            return similar(x, new_size)
            # (or create_array(...))
        end
    end
end

"""
    resize_array(x, new_size; zero = true, free = true)

Return an array of `new_size` containing the same values as `x` on overlapping indices.
We require `length(new_size) == ndims(x)`, i.e. the number of axes must be the same.
For new indices, we use zero values when `zero` (`true` by default). Otherwise they remain
uninitialized.

Normally, this requires us to allocate a new array and copy over the relevant values. When
`free` (`true` by default) we `maybe_unsafe_free!` `x`. But if the new array would be a
contiguous subset of `x` we could just use a view. When `reuse` (`true` by default) we
follow this strategy, if possible. Otherwise, we just use the allocation approach.
"""
function resize_array(x, new_size; zero = true, reuse = true, free = true)
    length(new_size) == ndims(x) || throw(ArgumentError(
        "The number of axes in the original and new size must coincide! \
         Received: $(ndims(x)) and $(length(new_size))"
    ))

    if reuse && 
          new_size[1:end-1] == size(x)[1:end-1] && new_size[ndims(x)] <= size(x, ndims(x))
        return reuse_memory(x, new_size; free) 
    else
        new_x = if zero
            create_zeros(array_domain(x), eltype(x), new_size)
        else
            create_array(array_domain(x), eltype(x), new_size)
        end
        ids = ntuple(ndims(x)) do i
            1 : min(size(x, i), new_size[i])
            # Note that we assume axes(x, i) == Base.OneTo(⋅)
        end
        new_x[ids...] .= view(x, ids...)

        free && maybe_unsafe_free!(x)
    end

    return new_x
end


"""
    quaternion_to_rotation_matrix(q::StaticVector{4}) -> SMatrix{3, 3}

Convert a(n unnormalized) quaternion `q` into a 3 x 3 rotation matrix.
"""
@inline function quaternion_to_rotation_matrix(q::StaticVector{4})
    sqrt2_inv_norm = sqrt(2 * inv(sum(t -> t^2, q))) 
    # sum(...) is faster than norm, and equally fast as manually writing it out
    r, x, y, z = q .* sqrt2_inv_norm
    return @SMatrix [1 - (y*y + z*z)         x*y - r*z           x*z + r*y
                          x*y + r*z     1 - (x*x + z*z)          y*z - r*x
                          x*z - r*y          y*z + r*x      1 - (x*x + y*y)]  
    # e.g. 1 - 2(q[2]*q[2] + q[3]*q[3]) / norm_sq
end

"""
    rotation_matrix_to_quaternion(R::SMatrix{3, 3}) -> SVector{4}

Convert a rotation matrix `R` into a normalized quaternion.
"""
@inline function rotation_matrix_to_quaternion(R::StaticMatrix{3, 3})
    q = QuatRotation(R).q
    return SVector{4}(q.s, q.v1, q.v2, q.v3)
end
# Not used in the forward or backward pass, so performance (even though it is probably fine)
# is not important


"""
    normalization_jacobian(unnormed::StaticVector{N})

Create a `StaticMatrix{N, N}` representing the Jacobian of the normalization
`unnormed ↦ normed := normed ./ norm(normed)`, where `norm` takes the Euclidean (2-)norm.
"""
@inline function normalization_jacobian(unnormed::StaticVector{N}) where N
    inv_norm = inv(sqrt(sum(t -> t^2, unnormed)))
    normed = inv_norm .* unnormed  # (Or ScaledSVector)
    return ScaledSMatrix(inv_norm, normalization_jacobian_times_norm(normed))
end

"""
    transposed_normalization_jacobian(unnormed::StaticVector{N})

Alias for [`normalization_jacobian`](@ref), as this Jacobian is symmetric.
"""
const transposed_normalization_jacobian = normalization_jacobian
# But (potentially) useful for clarity/consistency with the other Transposed_... in 
# backward_project_fgaussians_kernel!

"""
    normalization_jacobian_times_norm_from_normed_expression(
        ::Type{<:StaticVector{N, T}}, normed_symb
    ) where {N, T}

Create the expression constructing a `SymmetricSMatrix` of the Jacobian of the function
normalizing a `StaticVector`, by dividing it by its norm. As the resulting expression is
easier we actually scale the Jacobian by this norm.

As input we need a symbol `normed_symb` representing the already normalized vector.

For compatibility with [`adjust_indexing_expression`](@ref), we return a split expression.
"""
function normalization_jacobian_times_norm_from_normed_expression(
    ::Type{<:StaticVector{N, T}}, normed_symb
) where {N, T}
    jacobian_lower_entry_exprs = Expr[]
    for j = 1:N
        push!(jacobian_lower_entry_exprs, 
              :($(one(T)) - $normed_symb[$j] * $normed_symb[$j]))  # [j, j]
        for i = j+1 : N
            push!(jacobian_lower_entry_exprs, :(-$normed_symb[$j] * $normed_symb[$i]))  
            # [i, j] == [j, i]
        end
    end
    return :(), :(
        SymmetricSMatrix(
            SVector{$(length(jacobian_lower_entry_exprs)), $T}(
                $(jacobian_lower_entry_exprs...)))
    )
end

"""
    normalization_jacobian_times_norm(normed::StaticVector{N, T}) where {N, T}

Construct a `SymmetricSMatrix` containing the Jacobian of the vector normalization function,
but where this Jacobian is scaled by the norm. As input we take the already normalized
vector `normed`.
"""
@inline @generated normalization_jacobian_times_norm(
    normed::StaticVector{N, T}
) where {N, T} = _normalization_jacobian_times_norm_code(normed, :normed)

_normalization_jacobian_times_norm_code(normed_type::Type{<:StaticVector}, normed_symb) = 
    adjust_indexing_expression(
        normalization_jacobian_times_norm_from_normed_expression(
            normed_type, normed_symb
        );
        finalize = true
    )


"""
    sigmoid(x)

Return the sigmoid of x: ``1 / (1 + exp(-x))``.
"""
@inline function sigmoid(x)
    return one(x) / (one(x) + exp(-x))
end

"""
    logit(x)

Return the logit of x: ``- \\log(1/x - 1)``. Note that logit is the inverse of sigmoid.
"""
@inline function logit(x)
    return -log(inv(x) - one(x))
end

"""
    @run_1d_kernel data_len_expr call_expr

Run the CUDA kernel in `call_expr` using a tuned launch configuration, where the grid and
block structure is one-dimensional. The number of threads will be the one in `data_len_expr`
(or slightly more). Typically this is a `length` or `size` which will be evaluated in the
produced code. In the end the output code calls `@cuda threads=... blocks=... call_expr`.
"""
macro run_1d_kernel(data_len_expr, call_expr)
    ret_expr = quote
        data_len = $data_len_expr  
        # (In the output code define a variable evaluating data_len_expr. This is then
        # executed only once.)
        local kernel = @cuda launch=false $call_expr
        local config = launch_configuration(kernel.fun)
        local nb_threads = min(data_len, config.threads)
        local nb_blocks = clamp(cld(data_len, nb_threads), config.blocks, typemax(Int32))
        kernel($(call_expr.args[2:end]...); threads=nb_threads, blocks=nb_blocks)
    end

    return esc(ret_expr)
end 

"""
    AbstractRectangle{T}

Abstract struct for representing axis-aligned rectanbles, with coordinates of type `T`.
"""
abstract type AbstractRectangle{T} end

"""
    Rectangle{T} <: AbstractRectangle{T}

Standard `AbstractRectangle` implementation.

---

    Rectangle(left::T, top::T, right::T, bottom::T) where T

Construct a Rectangle{T} from the provided extreme coordinates.
"""
struct Rectangle{T} <: AbstractRectangle{T}
    left::T
    top::T
    right::T
    bottom::T
end

"""
    Rectangle(top_left::SVector{2, T}, bottom_right::SVector{2, T}) where T

Construct a `Rectangle{T}` from the provided top-left (left, top) and bottom-right
(right, bottom) coordinates.
"""
@inline function Rectangle(top_left::SVector{2, T}, bottom_right::SVector{2, T}) where T
    return Rectangle(top_left[1], top_left[2], bottom_right[1], bottom_right[2]) 
end

@inline get_left(r::Rectangle) = r.left
@inline get_top(r::Rectangle) = r.top
@inline get_right(r::Rectangle) = r.right
@inline get_bottom(r::Rectangle) = r.bottom

"""
    get_top_left(r::Rectangle{T}) where T

Return (left, top) as an SVector{2, T}.
"""
@inline get_top_left(r::Rectangle{T}) where T = SVector{2, T}(get_left(r), get_top(r))

"""
    get_bottom_right(r::Rectangle{T}) where T

Return (right, bottom) as an SVector{2, T}.
"""
@inline get_bottom_right(r::Rectangle{T}) where T = SVector{2, T}(get_right(r), get_bottom(r))


"""
    get_top_right(r::Rectangle{T}) where T

Return (right, top) as an SVector{2, T}.
"""
@inline get_top_right(r::Rectangle{T}) where T = SVector{2, T}(get_right(r), get_top(r))

"""
    get_bottom_left(r::Rectangle{T}) where T

Return (left, bottom) as an SVector{2, T}.
"""
@inline get_bottom_left(r::Rectangle{T}) where T = SVector{2, T}(get_left(r), get_bottom(r))

@inline Base.isempty(r::Rectangle) = 
    get_left(r) > get_right(r) || get_top(r) > get_bottom(r)

"""
    EmptyRectangle{T} <: AbstractRectangle{T}

Canonical representation of an empty `AbstractRectangle{T}`.
"""
struct EmptyRectangle{T} <: AbstractRectangle{T} end

@inline Base.isempty(er::EmptyRectangle) = true