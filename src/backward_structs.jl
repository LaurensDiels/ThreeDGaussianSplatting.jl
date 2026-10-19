@im_mutable_with_adapt struct FilteredGaussiansGradientsVector{
    D <: ArrayDomain, I <: Integer, F <: AbstractFloat,
    F2DGV <: AbstractFiltered2DGaussiansVector{D, I, F},
    M2 <: AbstractMatrix{F}, 
    O <: AbstractVector{F},
    IC <: AbstractMatrix{F},
    C <: AbstractMatrix{F}, 
    M3 <: AbstractMatrix{F}, 
    OF <: AbstractVector{F},
    SF <: AbstractMatrix{F},
    RF <: AbstractMatrix{F},
    SH <: AbstractArray{F, 3}
}
    # We again distinguish between length and capacity. But since these are identical as for
    # the Filtered2DGaussiansVector, we don't have to store them explicitly
    f2dgv::F2DGV                 # capacity: n

    mean2d_grads::M2             # 2 x n  
    opacity_grads::O             # n
    licov2d_grads::IC            # 3 x n
    # Gradient with respect to the lower half of the symmetric inverse covariance matrix
    # (Not: lower half of the gradient with respect to the inverse covariance matrix)
    color_grads::C               # 3 x n

    mean3d_grads::M3             # 3 x n (TODO: n x 3, same below?)
    opacity_feature_grads::OF    # n
    scaling_features_grads::SF   # 3 x n
    rotation_features_grads::RF  # 4 x n 
    sh_coeffs_grads::SH          # (sh_degree + 1)^2 x 3 x n

    function FilteredGaussiansGradientsVector(
        f2dgv::F2DGV,
        mean2d_grads::M2, opacity_grads::O, licov2d_grads::IC, color_grads::C,
        mean3d_grads::M3, opacity_feature_grads::OF, 
        scaling_features_grads::SF, rotation_features_grads::RF,
        sh_coeffs_grads::SH
    ) where {
        D <: ArrayDomain, I <: Integer, F <: AbstractFloat,
        F2DGV <: AbstractFiltered2DGaussiansVector{D, I, F},
        M2 <: AbstractMatrix{F},
        O <: AbstractVector{F}, 
        IC <: AbstractMatrix{F},
        C <: AbstractMatrix{F},
        M3 <: AbstractMatrix{F}, 
        OF <: AbstractVector{F},
        SF <: AbstractMatrix{F},
        RF <: AbstractMatrix{F},
        SH <: AbstractArray{F, 3}
    }
        type_str = "FilteredGaussiansGradientsVector"
        arrays = (
            mean2d_grads, licov2d_grads, opacity_grads, color_grads,
            mean3d_grads, opacity_feature_grads, 
            scaling_features_grads, rotation_features_grads,
            sh_coeffs_grads
        )

        D2 = _get_common_array_domain(type_str, arrays)
        D2 === D || throw(ArgumentError(
            "The arrays of a `$type_str` need to have the same `ArrayDomain` as the
             corresponding `Filtered2DGaussiansVector`! Received: $D2 and $D."))
        n = capacity(f2dgv)
        _check_same_last_axis_size(type_str, arrays, n)
        # TODO: if we use n x 3 for means3d etc., update this

        _size_check(mean2d_grads, (2, n), "mean2d_grads")
        _size_check(opacity_grads, (n,), "opacity_grads")
        size(licov2d_grads) == (3, n) || 
            throw(DimensionMismatch(
                "Expected size (3, $n) for `licov2d_grads` (where 3 is the length of the \
                 lower matrix of the symmetric inverse 2D covariance matrix)! \
                 Received: $(size(licov2d_grads))"))
        size(color_grads) == (3, n) || 
            throw(DimensionMismatch(
                "Expected size (3, $n) for `color_grads` (where 3 comes from the number of
                 color components (R, G and B)! Received: $(size(color_grads))"))
            
        _size_check(mean3d_grads, (3, n), "mean3d_grads")
        _size_check(opacity_feature_grads, (n,), "opacity_feature_grads")
        _size_check(scaling_features_grads, (3, n), "scaling_features_grads")
        _size_check(rotation_features_grads, (4, n), "rotation_features_grads")
        
        _check_sh_coeffs(type_str, sh_coeffs_grads, "sh_coeffs_grads")

        return new{D, I, F, F2DGV, M2, O, IC, C, M3, OF, SF, RF, SH}(
            f2dgv,
            mean2d_grads, opacity_grads, licov2d_grads, color_grads,
            mean3d_grads, opacity_feature_grads, 
            scaling_features_grads, rotation_features_grads,
            sh_coeffs_grads
        )
    end
end

"""
    AbstractFilteredGaussiansGradientsVector{
        D <: ArrayDomain, I <: Integer, F <: AbstractFloat,
        F2DGV <: AbstractFiltered2DGaussiansVector{D, I, F},
        M2 <: AbstractMatrix{F}, 
        O <: AbstractVector{F}, 
        IC <: AbstractMatrix{F},
        C <: AbstractMatrix{F},
        M3 <: AbstractMatrix{F}, 
        OF <: AbstractVector{F},
        SF <: AbstractMatrix{F},
        RF <: AbstractMatrix{F},
        SH <: AbstractArray{F, 3}
    }

Abstract supertype of [`MutableFilteredGaussiansGradientsVector`](@ref) and
[`ImmutableFilteredGaussiansGradientsVector`](@ref).
"""
AbstractFilteredGaussiansGradientsVector

"""
    ImmutableFilteredGaussiansGradientsVector{
        D <: ArrayDomain, I <: Integer, F <: AbstractFloat,
        F2DGV <: AbstractFiltered2DGaussiansVector{D, I, F},
        M2 <: AbstractMatrix{F}, 
        O <: AbstractVector{F},
        IC <: AbstractMatrix{F},
        C <: AbstractMatrix{F}, 
        M3 <: AbstractMatrix{F}, 
        OF <: AbstractVector{F},
        SF <: AbstractMatrix{F},
        RF <: AbstractMatrix{F},
        SH <: AbstractArray{F, 3}
    } <: AbstractFilteredGaussiansGradientsVector{
            D, I, F, M2, O, IC, C, M3, OF, SF, RF, SH
        }

Immutable version of [`MutableFilteredGaussiansGradientsVector`](@ref). For adaptive density
control we need the mutable version.
"""
ImmutableFilteredGaussiansGradientsVector

@replace(
    X => (MutableFilteredGaussiansGradientsVector, 
          ImmutableFilteredGaussiansGradientsVector),
    function X(f2dgv::AbstractFiltered2DGaussiansVector{D, I, F}) where {D, I, F}
        n = capacity(f2dgv)
        return X(
            f2dgv,
            create_array(D, F, (2, n)),
            create_array(D, F, n),
            create_array(D, F, (3, n)),
            create_array(D, F, (3, n)),
            create_array(D, F, (3, n)),
            create_array(D, F, n),
            create_array(D, F, (3, n)),
            create_array(D, F, (4, n)),
            create_array(D, F, (
                (get_sh_degree(f2dgv.all_3d_gaussians) + one(I)) ^ 2, 3, n)),
        )
    end
)

"""
    FilteredGaussiansGradientsVector(
        f2dgv::AbstractFiltered2DGaussiansVector{D, I, F}
    ) where {D, I, F}

Construct a Mutable-/ImmutableFilteredGaussiansGradientsVector from `f2dgv`, depending on
its mutability. See the similar [`MutableFilteredGaussiansGradientsVector`](@ref) for
more information.
"""
function FilteredGaussiansGradientsVector(
    f2dgv::AbstractFiltered2DGaussiansVector{D, I, F}
) where {D, I, F}
    FGGV = if ismutabletype(f2dgv) 
        MutableFilteredGaussiansGradientsVector
    else
        ImmutableFilteredGaussiansGradientsVector
    end
    return FGGV(f2dgv)
end
# (You would have to go out of your way to get a correctly filled
#  ImmutableFiltered2DGaussiansVector, though, as `render!` does not support it.)

"""
    MutableFilteredGaussiansGradientsVector{
        D <: ArrayDomain, I <: Integer, F <: AbstractFloat,
        F2DGV <: AbstractFiltered2DGaussiansVector{D, I, F},
        M2 <: AbstractMatrix{F}, 
        O <: AbstractVector{F}, 
        IC <: AbstractMatrix{F},
        C <: AbstractMatrix{F},
        M3 <: AbstractMatrix{F}, 
        OF <: AbstractVector{F},
        SF <: AbstractMatrix{F},
        RF <: AbstractMatrix{F},
        SH <: AbstractArray{F, 3}
    } <: AbstractFilteredGaussiansGradientsVector{
            D, I, F, M2, O, IC, C, M3, OF, SF, RF, SH
        }

Mutable struct storing the gradients of the rendering loss with respect to the parameters
of filtered 2D/3D Gaussians. Like `Filtered2DGaussiansVector`s our arrays have a capacity
potentially larger than our `length`, which is the number of filtered Gaussians. We also
support broadcasting, with element type `[FilteredGaussianGradients](@ref)`.

The gradient values will be filled in (in steps) in [`backward!`](@ref).

---

    MutableFilteredGaussiansGradientsVector(
        f2dgv::F2DGV,
        mean2d_grads::M2, opacity_grads::O, licov2d_grads::IC, color_grads::C,
        mean3d_grads::M3, opacity_feature_grads::OF, 
        scaling_features_grads::SF, rotation_features_grads::RF,
        sh_coeffs_grads::SH
    ) where {
        D <: ArrayDomain, I <: Integer, F <: AbstractFloat,
        F2DGV <: AbstractFiltered2DGaussiansVector{D, I, F},
        M2 <: AbstractMatrix{F}, 
        O <: AbstractVector{F},
        IC <: AbstractMatrix{F},
        C <: AbstractMatrix{F}, 
        M3 <: AbstractMatrix{F}, 
        OF <: AbstractVector{F},
        SF <: AbstractMatrix{F},
        RF <: AbstractMatrix{F},
        SH <: AbstractArray{F, 3}
    }

Construct a `MutableFilteredGaussiansGradientsVector` from its fields. Inner constructor
which performs `size` and `ArrayDomain` checks.

---

    MutableFilteredGaussiansGradientsVector(
        f2dgv::AbstractFiltered2DGaussiansVector{D, I, F}
    ) where {D, I, F}

Construct a `MutableFilteredGaussiansGradientsVector` from the corresponding
`Filtered2DGaussiansVector`. We will create new gradient arrays initially filled with
garbage values.
"""
MutableFilteredGaussiansGradientsVector


"""
    zero_2d_grads!(
        fggv::AbstractFilteredGaussiansGradientsVector{<:Any, <:Any, F}
    ) where F

Set the gradients corresponding to 2D Gaussian attributes (i.e. after the projection phase)
to zero.
"""
@inline function zero_2d_grads!(
    fggv::AbstractFilteredGaussiansGradientsVector{<:Any, <:Any, F}
) where F
    fill!(fggv.mean2d_grads, zero(F))
    fill!(fggv.opacity_grads, zero(F))
    fill!(fggv.licov2d_grads, zero(F))
    fill!(fggv.color_grads, zero(F))
end

# Similar to Filtered2DGaussiansVector
@inline array_domain(::AbstractFilteredGaussiansGradientsVector{D}) where D = D
@inline int_type(::AbstractFilteredGaussiansGradientsVector{<:Any, I}) where I = I
@inline float_type(::AbstractFilteredGaussiansGradientsVector{<:Any, <:Any, F}) where F = F

@inline Base.length(fggv::AbstractFilteredGaussiansGradientsVector) = length(fggv.f2dgv)
@inline capacity(fggv::AbstractFilteredGaussiansGradientsVector) = capacity(fggv.f2dgv)
@inline get_sh_degree(fggv::AbstractFilteredGaussiansGradientsVector{<:Any, I}) where I = 
    isqrt(I(size(fggv.sh_coeffs_grads, 1))) - one(I)

function Base.show(
    io::IO, ::MIME"text/plain", 
    fggv::AbstractFilteredGaussiansGradientsVector{D, I, F}
) where {D, I, F}
    print(io, 
        "$(typeof_noparams(fggv)){$D, $I, $F}\n",
        "  length: ", length(fggv), " (out of ", length(parent(fggv.f2dgv)), 
            " 3D Gaussian$(_s_when_not_1(length(parent(fggv.f2dgv)))))\n",
        "  capacity: ", capacity(fggv), "\n"
    )
end

@inline Base.size(fggv::AbstractFilteredGaussiansGradientsVector) = (length(fggv),)
@inline Base.ndims(::Type{<:AbstractFilteredGaussiansGradientsVector}) = 1
@inline Base.firstindex(::AbstractFilteredGaussiansGradientsVector{<:Any, I}) where I = 
    one(I)
@inline Base.lastindex(fggv::AbstractFilteredGaussiansGradientsVector) = length(fggv)

# slice
Base.@propagate_inbounds function Base.getindex(
    fggv::AbstractFilteredGaussiansGradientsVector, i
)
    FGGV = typeof(fggv)
    remove_parameters(FGGV)(
        fggv.f2dgv[i],  # (not an AbstractArray) 
        (indexlastaxis(_length_normalized_getfield(fggv, fn, i), i) 
         for (fn, ft) in zip(fieldnames(FGGV), fieldtypes(FGGV))
         if ft <: AbstractArray
        )...
    )
end

Base.@propagate_inbounds function Base.view(
    fggv::AbstractFilteredGaussiansGradientsVector, i
)
    FGGV = typeof(fggv)
    remove_parameters(FGGV)(
        view(fggv.f2dgv, i),
        (viewlastaxis(_length_normalized_getfield(fggv, fn, i), i) 
         for (fn, ft) in zip(fieldnames(FGGV), fieldtypes(FGGV))
         if ft <: AbstractArray
        )...
    )
end

"""
    maybe_unsafe_free!(
        fggv::FGGV
    ) where {D, FGGV <: AbstractFilteredGaussiansGradientsVector{D}}

Call `maybe_unsafe_free!` on all arrays in the Filtered2DGaussiansVector. If D <: GPU
these will then be `unsafe_free!`d. If not, nothing happens.

We will never call `unsafe_free!` on the corresponding `Filtered2DGaussiansVector`.
"""
function maybe_unsafe_free!(
    fggv::FGGV
) where FGGV <: AbstractFilteredGaussiansGradientsVector
    for (fn, ft) in zip(fieldnames(FGGV), fieldtypes(FGGV))
        ft <: AbstractArray && maybe_unsafe_free!(getfield(fggv, fn))
    end
    return nothing
end

"""
    set_sh_degree!(
        mfggv::MutableFilteredGaussiansGradientsVector{D, I, F}, sh_degree::I
    ) where {D, I, F}

Resize the underlying spherical harmonics coefficients gradients array to a new size
corresponding to the supplied `sh_degree` (if it is valid). The new array will contain
garbage values.
"""
function set_sh_degree!(
    mfggv::MutableFilteredGaussiansGradientsVector{<:Any, I}, sh_degree::I
) where I
    sh_degree == get_sh_degree(mfggv) && return
    _check_sh_degree(sh_degree)

    new_max_sh_order = (sh_degree + one(I))^2
    new_size = (new_max_sh_order, 3, capacity(mfggv))

    mfggv.sh_coeffs_grads = resize_memory(mfggv.sh_coeffs_grads, new_size)
   
    return nothing
end

"""
    set_nb_gaussians!(mfggv::MutableFilteredGaussiansGradientsVector, nb_gaussians)

Unless this was already within `capacity`, resize the arrays to the new `nb_gaussians`.

!!! We do not alter the Filtered2DGaussiansVector. In the intended use-case of adaptive
density control, this adjustment will be performed by `set_nb_gaussians!` on the
`render_buffers`.
"""
function set_nb_gaussians!(mfggv::MutableFilteredGaussiansGradientsVector, nb_gaussians)
    nb_gaussians <= capacity(mfggv) && return

    for fn in fieldnames(MutableFilteredGaussiansGradientsVector)
        fn == :f2dgv && continue
        array = getfield(mfggv, fn)
        setfield!(mfggv, fn, resize_memory(array, (size(array)[1:end-1]..., nb_gaussians)))
    end
    return nothing
end


"""
    GaussianGradients{
        D <: ArrayDomain, 
        I <: Integer, 
        F <: AbstractFloat,
        FGGV <: AbstractFilteredGaussiansGradientsVector{D, I, F}
    }

The loss gradients of a 3D/2D Gaussian.

!!! A `GaussianGradients` is defined by its index into the FilteredGaussiansGradientsVector.
If the latter is mutated, the index and hence the `GaussianGradients` might become invalid. 
"""
struct GaussianGradients{
    D <: ArrayDomain,
    I <: Integer,
    F <: AbstractFloat,
    FGGV <: AbstractFilteredGaussiansGradientsVector{D, I, F}
}
    fggv::FGGV
    fgidx::I

    @inline function GaussianGradients(fggv::FGGV, fgidx::Integer) where {
        D <: ArrayDomain, I <: Integer, F <: AbstractFloat,
        FGGV <: AbstractFilteredGaussiansGradientsVector{D, I, F}
    }
        @boundscheck 1 <= fgidx <= length(fggv) || throw(ArgumentError(
            "The provided GaussianGradients filtered index $fgidx is out of bounds 
             (1:$(length(fggv)))!"))
        return new{D, I, F, FGGV}(fggv, fgidx)
    end

    # For Device: no string interpolation
    @inline function GaussianGradients(fggv::FGGV, fgidx::Integer) where {
        I <: Integer, F <: AbstractFloat,
        FGGV <: AbstractFilteredGaussiansGradientsVector{Device, I, F}
    }
        @boundscheck 1 <= fgidx <= length(fggv) || throw(ArgumentError(
            "The provided GaussianGradients index is out of bounds!"))
        return new{Device, I, F, FGGV}(fggv, fgidx)
    end
end

Base.@propagate_inbounds Base.getindex(
    fggv::AbstractFilteredGaussiansGradientsVector, fgidx::Integer
) = GaussianGradients(fggv, fgidx)
Base.@propagate_inbounds Base.first(fggv::AbstractFilteredGaussiansGradientsVector) = 
    fggv[1]
Base.@propagate_inbounds Base.last(fggv::AbstractFilteredGaussiansGradientsVector) = 
    fggv[length(fggv)]
@inline Base.eltype(::FGGV) where {
    D, I, F, FGGV <: AbstractFilteredGaussiansGradientsVector{D, I, F}
} = GaussianGradients{D, I, F, FGGV}
@inline Base.broadcastable(fggv::AbstractFilteredGaussiansGradientsVector) = fggv
@inline Base.BroadcastStyle(::Type{FGGV}) where 
    {M, FGGV <: AbstractFilteredGaussiansGradientsVector{GPU{M}}
} = CUDACore.CuArrayStyle{1, M}()

function Base.iterate(
    fggv::AbstractFilteredGaussiansGradientsVector{<:Any, I}, 
    state::I = one(I)
) where I
    state <= length(fggv) || return nothing
    return (@inbounds fggv[state], state + one(I))
end

@inline array_domain(::GaussianGradients{D}) where D = D
@inline int_type(::GaussianGradients{<:Any, I}) where I = I
@inline float_type(::GaussianGradients{<:Any, <:Any, F}) where F = F

@inline Base.parent(gg::GaussianGradients) = gg.fggv
@inline get_fgidx(gg::GaussianGradients) = gg.fgidx
@inline get_gidx(gg::GaussianGradients) = 
    @inbounds gg.fggv.f2dgv.fgidx_to_gidx[get_fgidx(gg)]


@inline get_mean2d_grad_view(gg::GaussianGradients) = @inbounds view(
    gg.fggv.mean2d_grads, SOneTo(2), gg.fgidx
)
@inline get_mean2d_grad_sv(gg::GaussianGradients) = SVector{2}(get_mean2d_grad_view(gg))
@inline function set_mean2d_grad!(gg::GaussianGradients, xg, yg)
    m2g = get_mean2d_grad_view(gg)
    @inbounds begin m2g[1] = xg; m2g[2] = yg end
    return nothing
end
@inline set_mean2d_grad!(gg::GaussianGradients, xy_grad) =
    set_mean2d_grad!(gg, xy_grad...)
@inline function cu_atomically_add_to_mean2d_grad(
    gg::GaussianGradients, xg, yg
)
    @inbounds CUDA.@atomic gg.fggv.mean2d_grads[1, gg.fgidx] += xg
    @inbounds CUDA.@atomic gg.fggv.mean2d_grads[2, gg.fgidx] += yg
    return nothing
end
@inline cu_atomically_add_to_mean2d_grad(gg::GaussianGradients, xy_grad) =
    cu_atomically_add_to_mean2d_grad(gg, xy_grad...)

@inline get_opacity_grad_view(gg::GaussianGradients) = @inbounds view(
    gg.fggv.opacity_grads, gg.fgidx:gg.fgidx
)
@inline get_opacity_grad(gg::GaussianGradients) = @inbounds gg.fggv.opacity_grads[gg.fgidx]
@inline set_opacity_grad!(gg::GaussianGradients, og) =
   @inbounds gg.fggv.opacity_grads[gg.fgidx] = og
@inline function cu_atomically_add_to_opacity_grad(
    gg::GaussianGradients, og,
)
    @inbounds CUDA.@atomic gg.fggv.opacity_grads[gg.fgidx] += og
    return nothing
end

@inline get_inverse_covariance_lower_grad_view(gg::GaussianGradients) = @inbounds view(
    gg.fggv.licov2d_grads, SOneTo(3), gg.fgidx
)
@inline get_inverse_covariance_lower_grad_sv(gg::GaussianGradients) = 
    SVector{3}(get_inverse_covariance_lower_grad_view(gg))
@inline function set_inverse_covariance_lower_grad!(
    gg::GaussianGradients, licov1g, licov2g, licov3g
)  
# (Note: don't use ic21g etc., as we do not need the gradient with respect to this single
#  entry, but rather the sum of the gradients of those entries corresponding to the second
#  entry in the lower half vector representation)
    licov2dg = get_inverse_covariance_lower_grad_view(gg)
    @inbounds begin licov2dg[1] = licov1g; licov2dg[2] = licov2g; licov2dg[3] = licov3g end
    return nothing
end
@inline set_inverse_covariance_lower_grad!(gg::GaussianGradients, licov2d_Grad) =
    set_inverse_covariance_lower_grad!(gg, licov2d...)
@inline function cu_atomically_add_to_inverse_covariance_lower_grad(
    gg::GaussianGradients, licov1g, licov2g, licov3g
)
    @inbounds CUDA.@atomic gg.fggv.licov2d_grads[1, gg.fgidx] += licov1g
    @inbounds CUDA.@atomic gg.fggv.licov2d_grads[2, gg.fgidx] += licov2g
    @inbounds CUDA.@atomic gg.fggv.licov2d_grads[3, gg.fgidx] += licov3g
    return nothing
end
@inline cu_atomically_add_to_inverse_covariance_lower_grad(
    gg::GaussianGradients, licov2d_grad
) = cu_atomically_add_to_inverse_covariance_lower_grad(gg, licov2d_grad...)

@inline get_color_grad_view(gg::GaussianGradients) = @inbounds view(
    gg.fggv.color_grads, SOneTo(3), gg.fgidx
)
@inline get_color_grad_sv(gg::GaussianGradients) = SVector{3}(get_color_grad_view(gg))
@inline function set_color_grad!(gg::GaussianGradients, red_g, green_g, blue_g)
    cg = get_color_grad_view(gg)
    @inbounds begin cg[1] = red_g; cg[2] = green_g; cg[3] = blue_g end
    return nothing
end
@inline set_color_grad!(gg::GaussianGradients, color_grad) =
    set_color_grad!(gg, color_grad...)
@inline function cu_atomically_add_to_color_grad(
    gg::GaussianGradients, red_g, green_g, blue_g
)
    @inbounds CUDA.@atomic gg.fggv.color_grads[1, gg.fgidx] += red_g
    @inbounds CUDA.@atomic gg.fggv.color_grads[2, gg.fgidx] += green_g
    @inbounds CUDA.@atomic gg.fggv.color_grads[3, gg.fgidx] += blue_g
    return nothing
end
@inline cu_atomically_add_to_color_grad(gg::GaussianGradients, color_grad) =
    cu_atomically_add_to_color_grad(gg, color_grad...)

@inline get_mean3d_grad_view(gg::GaussianGradients) = @inbounds view(
    gg.fggv.mean3d_grads, SOneTo(3), gg.fgidx
)
@inline get_mean3d_grad_sv(gg::GaussianGradients) = SVector{3}(get_mean3d_grad_view(gg))
@inline function set_mean3d_grad!(gg::GaussianGradients, xg, yg, zg)
    m3g = get_mean3d_grad_view(gg)
    @inbounds begin m3g[1] = xg; m3g[2] = yg; m3g[3] = zg end
    return nothing
end
@inline set_mean3d_grad!(gg::GaussianGradients, xyz_grad) =
    set_mean3d_grad!(gg, xyz_grad...)

@inline get_opacity_feature_grad_view(gg::GaussianGradients) = @inbounds view(
    gg.fggv.opacity_feature_grads, gg.fgidx:gg.fgidx
)
@inline get_opacity_feature_grad(gg::GaussianGradients) = 
    @inbounds gg.fggv.opacity_feature_grads[gg.fgidx]
@inline function set_opacity_feature_grad!(gg::GaussianGradients, ofg)
    @inbounds gg.fggv.opacity_feature_grads[gg.fgidx] = ofg
end

@inline get_scaling_features_grad_view(gg::GaussianGradients) = @inbounds view(
    gg.fggv.scaling_features_grads, SOneTo(3), gg.fgidx
)
@inline get_scaling_features_grad_sv(gg::GaussianGradients) = 
    SVector{3}(get_scaling_features_grad_view(gg))
@inline function set_scaling_features_grad!(gg::GaussianGradients, sf1g, sf2g, sf3g)
    sfg = get_scaling_features_grad_view(gg)
    @inbounds begin sfg[1] = sf1g; sfg[2] = sf2g; sfg[3] = sf3g end
    return nothing
end
@inline set_scaling_features_grad!(gg::GaussianGradients, sfs_grad) =
    set_scaling_features_grad!(gg, sfs_grad...)

@inline get_rotation_features_grad_view(gg::GaussianGradients) = @inbounds view(
    gg.fggv.rotation_features_grads, SOneTo(4), gg.fgidx
)
@inline get_rotation_features_grad_sv(gg::GaussianGradients) = 
    SVector{4}(get_rotation_features_grad_view(gg))
@inline function set_rotation_features_grad!(gg::GaussianGradients, rf1g, rf2g, rf3g, rf4g)
    rfg = get_rotation_features_grad_view(gg)
    @inbounds begin rfg[1] = rf1g; rfg[2] = rf2g; rfg[3] = rf3g; rfg[4] = rf4g end
    return nothing
end
@inline set_rotation_features_grad!(gg::GaussianGradients, rfs_grad) =
    set_rotation_features_grad!(gg, rfs_grad...)

@inline get_sh_degree(gg::GaussianGradients) = get_sh_degree(gg.fggv)
@inline get_sh_coeffs_grad_view(gg::GaussianGradients) = @inbounds view(
    gg.fggv.sh_coeffs_grads, :, SOneTo(3), gg.fgidx
)
@inline get_sh_coeffs_grad_view(gg::GaussianGradients, max_sh_degree::I) where I = 
    @inbounds view(
        gg.fggv.sh_coeffs_grads, 
        one(I):(max_sh_degree + one(I))^2, SOneTo(3), gg.fgidx)
@inline get_sh_coeffs_grad_view(
    gg::GaussianGradients, ::Val{MAX_SH_DEGREE_I64}
) where MAX_SH_DEGREE_I64 = @inbounds view(
        gg.fggv.sh_coeffs_grads, 
        SOneTo((MAX_SH_DEGREE_I64 + 1)^2), SOneTo(3), gg.fgidx)


# With all the same caveats as set_sh_coeffs!, get_sh_coeffs_sm_val and get_sh_coeffs_sm:
@inline function set_sh_coeffs_grad!(gg::GaussianGradients, sh_coeffs_grad)
    shgv = get_sh_coeffs_grad_view(gg)
    shgv .= sh_coeffs_grad
end
@inline get_sh_coeffs_grad_sm_val(
    gg::GaussianGradients, ::Val{nb_coeffs_per_color_channel}
) where {
    nb_coeffs_per_color_channel
} = SMatrix{nb_coeffs_per_color_channel, 3}(get_sh_coeffs_grad_view(gg))
@inline function get_sh_coeffs_grad_sm(gg::GaussianGradients)
    sh_degree = get_sh_degree(gg.fggv.f2dgv.all_3d_gaussians)
    if sh_degree == 0
        return get_sh_coeffs_grad_sm_val(gg, Val(1))
    elseif sh_degree == 1
        return get_sh_coeffs_grad_sm_val(gg, Val(4))
    elseif sh_degree == 2
        return get_sh_coeffs_grad_sm_val(gg, Val(9))
    else sh_degree == 3
        return get_sh_coeffs_grad_sm_val(gg, Val(16))
    end
end

@inline get_sh_coeffs_dc_grad_view(gg::GaussianGradients) = 
    @inbounds view(gg.fggv.sh_coeffs_grads, 1, SOneTo(3), gg.fgidx)
@inline get_sh_coeffs_dc_grad_sv(gg::GaussianGradients) =
    SVector{3}(get_sh_coeffs_dc_grad_view(gg))
@inline function set_sh_coeffs_dc_grad!(
    gg::GaussianGradients, red_sh_g, green_sh_g, blue_sh_g
)
    shdcgv = get_sh_coeffs_dc_grad_view(gg)
    @inbounds begin
        shdcgv[1] = red_sh_g; shdcgv[2] = green_sh_g; shdcgv[3] = blue_sh_g
    end
    return nothing
end
@inline set_sh_coeffs_dc_grad!(gg::GaussianGradients, sh_dc_g) =
    set_sh_coeffs_dc_grad!(gg, sh_dc_g...)

@inline get_sh_coeffs_rest_grad_view(gg::GaussianGradients) = 
    @inbounds @view gg.fggv.sh_coeffs_grads[2:end, SOneTo(3), gg.fgidx]
@inline get_sh_coeffs_rest_grad_view(gg::GaussianGradients, max_sh_degree::I) where I = 
    @inbounds view(
        gg.fggv.sh_coeffs_grads, I(2):(max_sh_degree + one(I))^2, SOneTo(3), gg.fgidx)
@inline get_sh_coeffs_rest_grad_view(
    gg::GaussianGradients, ::Val{MAX_SH_DEGREE_I64}
) where MAX_SH_DEGREE_I64 = 
    @inbounds view(
        gg.fggv.sh_coeffs_grads, 
        SUnitRange(2, (MAX_SH_DEGREE_I64 + 1)^2), SOneTo(3), gg.fgidx)


function Base.show(io::IO, ::MIME"text/plain", gg::GaussianGradients)
    @allowscalar print(io, 
        "GaussianGradients $(gg.fgidx) out of $(length(gg.fggv)) corresponding to Gaussian \
        $(get_gidx(gg)), with gradients for the \n",
        "  2D Gaussian mean: ", get_mean2d_grad_view(gg), "\n",
        "  opacity: ", get_opacity_grad(gg), "\n",
        "  lower inverse covariance matrix: ", 
            get_inverse_covariance_lower_grad_view(gg), "\n", 
        "  colors: ", get_color_grad_view(gg), "\n",
        "  3D Gaussian mean: ", get_mean3d_grad_view(gg), "\n",
        "  opacity feature: ", get_opacity_feature_grad(gg), "\n",
        "  scaling features: ", get_scaling_features_grad_view(gg), "\n",
        "  rotation features: ", get_rotation_features_grad_view(gg), "\n", 
        "  sherical harmonics coefficients: ", get_sh_coeffs_grad_view(gg)
    )
end
