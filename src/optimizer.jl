"""
    AbstractOptimizer{D <: ArrayDomain, I <: Integer, F <: AbstractFloat}

Abstract supertype for optimizers for TrainableGaussiansVectors. Concrete types should
implement [`update!`](@ref).
"""
abstract type AbstractOptimizer{D <: ArrayDomain, I <: Integer, F <: AbstractFloat} end

@im_mutable_with_adapt struct GaussianAdamMomentsLearningRatesVector{
    D <: ArrayDomain, I <: Integer, F <: AbstractFloat,
    MM <: AbstractArray{F, 3},
    OFM <: AbstractArray{F, 2},
    SFM <: AbstractArray{F, 3},
    RFM <: AbstractArray{F, 3},
    SHM <: AbstractArray{F, 4}
}

    # All arrays have first (mean) [1, ...] and second [2, ...] moment components
    # Because we have to go from filtered Gaussians (in fgradients) to all Gaussians,
    # it's better to rely on caching than on memory coalescing, so ... x n should be
    # faster than n x ...
    means_moments::MM               # 2 x 3 x n
    opacity_features_moments::OFM   # 2 x n
    # TODO: combine means and opacity?
    scaling_features_moments::SFM   # 2 x 3 x n
    rotation_features_moments::RFM  # 2 x 4 x n
    sh_coeffs_moments::SHM          # 2 x (sh_degree + 1)^2 x 3 x n

    means_lr::F
    opacity_features_lr::F
    scaling_features_lr::F
    rotation_features_lr::F
    sh_coeffs_dc_lr::F
    sh_coeffs_rest_lr::F

    function GaussianAdamMomentsLearningRatesVector(
        means_moments::MM,
        opacity_features_moments::OFM,
        scaling_features_moments::SFM,
        rotation_features_moments::RFM,
        sh_coeffs_moments::SHM,
        means_lr::F,
        opacity_features_lr::F,
        scaling_features_lr::F,
        rotation_features_lr::F,
        sh_coeffs_dc_lr::F, 
        sh_coeffs_rest_lr::F;
        int_type::Type{<:Integer} = Int32,
    ) where {
        F <: AbstractFloat,
        MM <: AbstractArray{F, 3},
        OFM <: AbstractArray{F, 2},
        SFM <: AbstractArray{F, 3},
        RFM <: AbstractArray{F, 3},
        SHM <: AbstractArray{F, 4}
    }
        type_str = "GaussianAdamMomentsVector"
        arrays = (
            means_moments, opacity_features_moments,
            scaling_features_moments, rotation_features_moments,
            sh_coeffs_moments
        )
        D = _get_common_array_domain(type_str, arrays)
        n = _check_same_last_axis_size(type_str, arrays)

        _size_check(means_moments, (2, 3, n), "means_moments")
        _size_check(opacity_features_moments, (2, n), "opacity_features_moments")
        _size_check(scaling_features_moments, (2, 3, n), "scaling_features_moments")
        _size_check(rotation_features_moments, (2, 4, n), "rotation_features_moments")

        h = size(sh_coeffs_moments, 2)  # max order
        isqrt(h) ^ 2 == h || 
            throw(DimensionMismatch(
                "The second dimension of a `$type_str`'s `sh_coeffs_moments` must be \
                a perfect square `(sh_degree + 1)^2`!"))
        _size_check(sh_coeffs_moments, (2, h, 3, n), "sh_coeffs_moments")

        _check_positive(means_lr, type_str, "means_lr")
        _check_positive(opacity_features_lr, type_str, "opacity_features_lr")
        _check_positive(scaling_features_lr, type_str, "scaling_features_lr")
        _check_positive(rotation_features_lr, type_str, "rotation_features_lr")
        _check_positive(sh_coeffs_dc_lr, type_str, "sh_coeffs_dc_lr")
        _check_positive(sh_coeffs_rest_lr, type_str, "sh_coeffs_rest_lr")

        return new{D, int_type, F, MM, OFM, SFM, RFM, SHM}(
            means_moments, opacity_features_moments,
            scaling_features_moments, rotation_features_moments,
            sh_coeffs_moments,
            means_lr, opacity_features_lr,
            scaling_features_lr, rotation_features_lr,
            sh_coeffs_dc_lr, sh_coeffs_rest_lr
        )
    end
end
# The immutable version is fine as long as we do not not need the resize the arrays,
# like we need for densification

"""
    _check_positive(value, type_str, value_name)

Check if `value > 0`. If not, throw a descriptive error involving the parent `type_str`,
which has `value_name` as a field.
"""
function _check_positive(value, type_str, value_name)
    value > 0 || throw(DomainError(
        value,
        "A$(_n_when_starts_with_vowel(type_str)) `$type_str`s `$value_name` parameter \
         should be strictly positive!"
    ))
end

"""
    _check_nonnegative(value, type_str, value_name)

Check if `value >= 0`. If not, throw a descriptive error involving the parent `type_str`,
which has `value_name` as a field.
"""
function _check_nonnegative(value, type_str, value_name)
    value >= 0 || throw(DomainError(
        value,
        "A$(_n_when_starts_with_vowel(type_str)) `$type_str`s `$value_name` parameter \
         should be non-negative!"
    ))
end

"""
    _check_01(value, type_str, value_name)

Check if `0 < value < 1`. If not, throw a descriptive error involving the parent `type_str`,
which has `value_name` as a field.
"""
function _check_01(value, type_str, value_name)
    0 < value < 1 || throw(DomainError(
        value,
        "A$(_n_when_starts_with_vowel(type_str)) `$type_str`s `$value_name` parameter \
         should be non-negative!"
    ))
end

"""
    AbstractGaussianAdamMomentsLearningRatesVector{
        D <: ArrayDomain, I <: Integer, F <: AbstractFloat,
        MM <: AbstractArray{F, 3},
        OFM <: AbstractArray{F, 2},
        SFM <: AbstractArray{F, 3},
        RFM <: AbstractArray{F, 3},
        SHM <: AbstractArray{F, 4}
    }

Abstract supertype of [`MutableGaussianAdamMomentsLearningRatesVector`](@ref) and
[`ImmutableGaussianAdamMomentsLearningRatesVector`](@ref).
"""
AbstractGaussianAdamMomentsLearningRatesVector

"""
    ImmutableGaussianAdamMomentsLearningRatesVector{
        D <: ArrayDomain, I <: Integer, F <: AbstractFloat,
        MM <: AbstractArray{F, 3},
        OFM <: AbstractArray{F, 2},
        SFM <: AbstractArray{F, 3},
        RFM <: AbstractArray{F, 3},
        SHM <: AbstractArray{F, 4}
    } <: AbstractGaussianAdamMomentsLearningRatesVector{
            D, I, F, MM, OFM, SFM, RFM, SHM
        }

Immutable version of [`MutableGaussianAdamMomentsLearningRatesVector`](@ref). Should
not be used directly, only possibly inside of kernels.
"""
ImmutableGaussianAdamMomentsLearningRatesVector

@replace(
    X => (MutableGaussianAdamMomentsLearningRatesVector, 
          ImmutableGaussianAdamMomentsLearningRatesVector),
    function X(
        gv::AbstractTrainableGaussiansVector{D, I, F},
        to::TrainingOptions{I, F},
    ) where {D, I, F}
        n = length(gv)
        return X(
            create_zeros(D, F, (2, 3, n)),
            create_zeros(D, F, (2, n)),
            create_zeros(D, F, (2, 3, n)),
            create_zeros(D, F, (2, 4, n)),
            create_zeros(D, F, (2, (get_sh_degree(gv) + 1) ^ 2, 3, n)),
            get_initial_means_learning_rate(to),
            get_opacity_features_learning_rate(to),
            get_scaling_features_learning_rate(to),
            get_rotation_features_learning_rate(to),
            get_sh_coeffs_dc_learning_rate(to),
            get_sh_coeffs_rest_learning_rate(to);
            int_type = I
        )
    end
)

"""
    MutableGaussianAdamMomentsLearningRatesVector{
        D <: ArrayDomain, I <: Integer, F <: AbstractFloat,
        MM <: AbstractArray{F, 3},
        OFM <: AbstractArray{F, 2},
        SFM <: AbstractArray{F, 3},
        RFM <: AbstractArray{F, 3},
        SHM <: AbstractArray{F, 4}
    } <: AbstractGaussianAdamMomentsLearningRatesVector{
            D, I, F, MM, OFM, SFM, RFM, SHM
        }

Mutable struct storing the Adam-style running first and second moments of gradients of
Gaussians. We support broadcasting, with element type
`[GaussianAdamMomentsLearningRate](@ref)`.

This struct is used as a field for `AdamOptimizer`s.

---

    MutableGaussianAdamMomentsLearningRatesVector(
        means_moments::MM,
        opacity_features_moments::OFM,
        scaling_features_moments::SFM,
        rotation_features_moments::RFM,
        sh_coeffs_moments::SHM;
        int_type::Type{<:Integer} = Int32
    ) where {
         F <: AbstractFloat,
        MM <: AbstractArray{F, 3},
        OFM <: AbstractArray{F, 2},
        SFM <: AbstractArray{F, 3},
        RFM <: AbstractArray{F, 3},
        SHM <: AbstractArray{F, 4}
    }

Construct a `MutableGaussianAdamMomentsLearningRatesVector` from its fields. Inner
constructor which performs `size` and `ArrayDomain` checks.

---

    MutableGaussianAdamMomentsLearningRatesVector(
        gv::AbstractTrainableGaussiansVector{D, I, F},
        to::TrainingOptions{I, F},
    ) where {D, I, F}

Construct a `MutableGaussianAdamMomentsLearningRatesVector` from the corresponding
`TrainableGaussiansVector` and `TrainingOptions`. We will create new arrays filled with
zeros.
"""
MutableGaussianAdamMomentsLearningRatesVector

"""
    GaussianAdamMomentsLearningRatesVector(
        gv::AbstractTrainableGaussiansVector{D, I, F},
        to::TrainingOptions{I, F},
    ) where {D, I, F}

Construct a Mutable/ImmutableGaussianAdamMomentsLearningRatesVector from the
`TrainableGaussiansVector` and `TrainingOptions`, where we inherit the former's mutability.
"""
function GaussianAdamMomentsLearningRatesVector(
    gv::AbstractTrainableGaussiansVector{D, I, F},
    to::TrainingOptions{I, F},
) where {D, I, F}
    GAMLRV = if ismutable(gv) 
        MutableGaussianAdamMomentsLearningRatesVector
    else
        ImmutableGaussianAdamMomentsLearningRatesVector
    end
    return GAMLRV(gv, to)
end

# Like for AbstractGaussiansVector
@inline array_domain(::AbstractGaussianAdamMomentsLearningRatesVector{D}) where D = D
@inline int_type(
    ::AbstractGaussianAdamMomentsLearningRatesVector{<:Any, I}
) where I = I
@inline float_type(
    ::AbstractGaussianAdamMomentsLearningRatesVector{<:Any, <:Any, F}
) where F = F

@inline Base.length(
    gamlrv::AbstractGaussianAdamMomentsLearningRatesVector{<:Any, I}
) where I = size(gamlrv.means_moments, 3)
@inline get_sh_degree(
    gamlrv::AbstractGaussianAdamMomentsLearningRatesVector{<:Any, I}
) where I = isqrt(I(size(gamlrv.sh_coeffs_moments, 2))) - one(I)


@inline get_means_learning_rate(
    gamlrv::AbstractGaussianAdamMomentsLearningRatesVector
) = gamlrv.means_lr
@inline function set_means_learning_rate!(
    gamlrv::MutableGaussianAdamMomentsLearningRatesVector,
    means_lr
) 
    _check_positive(means_lr, 
        "MutableGaussianAdamMomentsLearningRatesVector", "means_lr")
    gamlrv.means_lr = means_lr
    return nothing
end
@inline get_opacity_features_learning_rate(
    gamlrv::AbstractGaussianAdamMomentsLearningRatesVector
) = gamlrv.opacity_features_lr
@inline function set_opacity_features_learning_rate!(
    gamlrv::MutableGaussianAdamMomentsLearningRatesVector,
    opacity_features_lr
) 
    _check_positive(opacity_features_lr, 
        "MutableGaussianAdamMomentsLearningRatesVector", "opacity_features_lr")
    gamlrv.opacity_features_lr = opacity_features_lr
    return nothing
end
@inline get_scaling_features_learning_rate(
    gamlrv::AbstractGaussianAdamMomentsLearningRatesVector
) = gamlrv.scaling_features_lr
@inline function set_scaling_features_learning_rate!(
    gamlrv::MutableGaussianAdamMomentsLearningRatesVector,
    scaling_features_lr
) 
    _check_positive(scaling_features_lr, 
        "MutableGaussianAdamMomentsLearningRatesVector", "scaling_features_lr")
    gamlrv.scaling_features_lr = scaling_features_lr
    return nothing
end
@inline get_rotation_features_learning_rate(
    gamlrv::AbstractGaussianAdamMomentsLearningRatesVector
) = gamlrv.rotation_features_lr
@inline function set_rotation_features_learning_rate!(
    gamlrv::MutableGaussianAdamMomentsLearningRatesVector,
    rotation_features_lr
) 
    _check_positive(rotation_features_lr, 
        "MutableGaussianAdamMomentsLearningRatesVector", "rotation_features_lr")
    gamlrv.rotation_features_lr = rotation_features_lr
    return nothing
end
@inline get_sh_coeffs_dc_learning_rate(
    gamlrv::AbstractGaussianAdamMomentsLearningRatesVector
) = gamlrv.sh_coeffs_dc_lr
@inline function set_sh_coeffs_dc_learning_rate!(
    gamlrv::MutableGaussianAdamMomentsLearningRatesVector,
    sh_coeffs_dc_lr
) 
    _check_positive(sh_coeffs_dc_lr, 
        "MutableGaussianAdamMomentsLearningRatesVector", "sh_coeffs_dc_lr")
    gamlrv.sh_coeffs_dc_lr = sh_coeffs_dc_lr
    return nothing
end
@inline get_sh_coeffs_rest_learning_rate(
    gamlrv::AbstractGaussianAdamMomentsLearningRatesVector
) = gamlrv.sh_coeffs_rest_lr
@inline function set_sh_coeffs_rest_learning_rate!(
    gamlrv::MutableGaussianAdamMomentsLearningRatesVector,
    sh_coeffs_rest_lr
) 
    _check_positive(sh_coeffs_rest_lr, 
        "MutableGaussianAdamMomentsLearningRatesVector", "sh_coeffs_rest_lr")
    gamlrv.sh_coeffs_rest_lr = sh_coeffs_rest_lr
    return nothing
end

function Base.show(
    io::IO, ::MIME"text/plain", 
    gamlrv::AbstractGaussianAdamMomentsLearningRatesVector{D, I, F}
) where {D, I, F}
    print(io, 
        "$(typeof_noparams(gamlrv)){$D, $I, $F}\n",
        "  length: ", length(gamlrv), "\n",
        "  spherical harmonics degree: ", get_sh_degree(gamlrv)
    )
end

@inline Base.size(gamlrv::AbstractGaussianAdamMomentsLearningRatesVector) = 
    (length(gamlrv),)
@inline Base.ndims(::Type{<:AbstractGaussianAdamMomentsLearningRatesVector}) = 1
@inline Base.firstindex(
    ::AbstractGaussianAdamMomentsLearningRatesVector{<:Any, I}
) where I = one(I)
@inline Base.lastindex(gamlrv::AbstractGaussianAdamMomentsLearningRatesVector) = 
    length(gamlrv)

# slice
Base.@propagate_inbounds function Base.getindex(
    gamlrv::AbstractGaussianAdamMomentsLearningRatesVector, i
)
    GAMLRV = typeof(gamlrv)
    remove_parameters(GAMLRV)(
        (indexlastaxis(getfield(gamlrv, fn), i)
         for fn in fieldnames(GAMLRV)
        )...;
        int_type = int_type(gamlrv)
    )
end

Base.@propagate_inbounds function Base.view(
    gamlrv::AbstractGaussianAdamMomentsLearningRatesVector, i
)
    GAMLRV = typeof(gamlrv)
    remove_parameters(GAMLRV)(
        (viewlastaxis(getfield(gamlrv, fn), i)
         for fn in fieldnames(GAMLRV)
        )...;
        int_type = int_type(gamlrv)
    )
end

struct GaussianAdamMomentsLearningRate{
    D <: ArrayDomain,
    I <: Integer,
    F <: AbstractFloat,
    GAMLRV <: AbstractGaussianAdamMomentsLearningRatesVector{D, I, F}
}
    gamlrv::GAMLRV
    idx::I

    # With string interpolation
    @inline function GaussianAdamMomentsLearningRate(
        gamlrv::GAMLRV, idx::Integer
    ) where {  
        D <: ArrayDomain,
        I <: Integer,
        F <: AbstractFloat,
        GAMLRV <: AbstractGaussianAdamMomentsLearningRatesVector{D, I, F}
    }
        @boundscheck 1 <= idx <= length(gamlrv) || 
            @noinline throw(ArgumentError(
                "The provided Gaussian index $idx is out of bounds \
                 (1:$(length(gamlrv)))!"))
        return new{D, I, F, GAMLRV}(gamlrv, idx)
    end

    # Without string interpolation
    @inline function GaussianAdamMomentsLearningRate(
        gamlrv::GAMLRV, idx::Integer
    ) where {
        I <: Integer,
        F <: AbstractFloat,
        GAMLRV <: AbstractGaussianAdamMomentsLearningRatesVector{Device, I, F}
    }
        @boundscheck 1 <= idx <= length(gamlrv) || 
            @noinline throw(ArgumentError(
                "The provided Gaussian index is out of bounds!"))
        return new{Device, I, F, GAMLRV}(gamlrv, idx)
    end
end

Base.@propagate_inbounds Base.getindex(
    gamlrv::AbstractGaussianAdamMomentsLearningRatesVector, idx::Integer
) = GaussianAdamMomentsLearningRate(gamlrv, idx)
Base.@propagate_inbounds Base.first(
    gamlrv::AbstractGaussianAdamMomentsLearningRatesVector
) = gamlrv[1]
Base.@propagate_inbounds Base.last(
    gamlrv::AbstractGaussianAdamMomentsLearningRatesVector
) = gamlrv[length(gamlrv)]
@inline Base.eltype(::GAMLRV) where {
    D, I, F, GAMLRV <:AbstractGaussianAdamMomentsLearningRatesVector{D, I, F}
} = GaussianAdamMomentsLearningRate{D, I, F, GAMLRV}
@inline Base.broadcastable(
    gamlrv::AbstractGaussianAdamMomentsLearningRatesVector
) = gamlrv
@inline Base.BroadcastStyle(::Type{GAMLRV}) where {
    M, GAMLRV <: AbstractGaussianAdamMomentsLearningRatesVector{GPU{M}}
} = CUDACore.CuArrayStyle{1, M}()
function Base.iterate(
    gamlrv::AbstractGaussianAdamMomentsLearningRatesVector{<:Any, I}, 
    state::I = one(I)
) where I
    state <= length(gamlrv) || return nothing
    return (@inbounds gamlrv[state], state + one(I))
end

@inline array_domain(::GaussianAdamMomentsLearningRate{D}) where D = D
@inline int_type(::GaussianAdamMomentsLearningRate{<:Any, I}) where I = I
@inline float_type(::GaussianAdamMomentsLearningRate{<:Any, <:Any, F}) where F = F

@inline Base.parent(gamlr::GaussianAdamMomentsLearningRate) = gamlr.gamlrv
@inline get_idx(gamlr::GaussianAdamMomentsLearningRate) = gamlr.idx
@inline get_gidx(gamlr::GaussianAdamMomentsLearningRate) = gamlr.idx

# In practice, we will only need view (both for getting and setting), but we'll provide
# extra helpers, which might be useful for debugging
@inline get_mean_adam_moments_view(gamlr::GaussianAdamMomentsLearningRate) =
    @inbounds view(gamlr.gamlrv.means_moments, SOneTo(2), SOneTo(3), gamlr.idx)
@inline get_mean_adam_moments_sm(gamlr::GaussianAdamMomentsLearningRate) =
    SMatrix{2, 3}(get_mean_adam_moments_view(gamlr))
@inline function set_mean_adam_moments!(
    gamlr::GaussianAdamMomentsLearningRate, 
    x1, x2, y1, y2, z1, z2
)
    ms = get_mean_adam_moments_view(gamlr)
    @inbounds begin
        ms[1, 1] = x1; ms[2, 1] = x2; 
        ms[1, 2] = y1; ms[2, 2] = y2; 
        ms[1, 3] = z1; ms[2, 3] = z2
    end
    return nothing
end
@inline set_mean_adam_moments!(
    gamlr::GaussianAdamMomentsLearningRate, moments
) = set_mean_adam_moments!(gamlr, moments...)

@inline get_opacity_feature_adam_moments_view(
    gamlr::GaussianAdamMomentsLearningRate
) = @inbounds view(gamlr.gamlrv.opacity_features_moments, SOneTo(2), gamlr.idx)
@inline get_opacity_feature_adam_moments_sv(
    gamlr::GaussianAdamMomentsLearningRate
) = SVector{2}(get_opacity_feature_adam_moments_view(gamlr))
@inline function set_opacity_feature_adam_moments!(
    gamlr::GaussianAdamMomentsLearningRate, 
    m1, m2
)
    ms_view = get_opacity_feature_adam_moments_view(gamlr)
    @inbounds begin ms_view[1] = m1; ms_view[2] = m2 end
    return nothing
end
@inline set_opacity_feature_adam_moments!(gamlr::GaussianAdamMomentsLearningRate, ofms) =
    set_opacity_feature_adam_moments!(gamlr, ofms...)

@inline get_scaling_features_adam_moments_view(gamlr::GaussianAdamMomentsLearningRate) =
    @inbounds view(gamlr.gamlrv.scaling_features_moments, SOneTo(2), SOneTo(3), gamlr.idx)
@inline get_scaling_features_adam_moments_sm(gamlr::GaussianAdamMomentsLearningRate) =
    SMatrix{2, 3}(get_scaling_features_adam_moments_view(gamlr))
@inline function set_scaling_features_adam_moments!(
    gamlr::GaussianAdamMomentsLearningRate, 
    sx1, sx2, sy1, sy2, sz1, sz2
)
    sms = get_scaling_features_adam_moments_view(gamlr)
    @inbounds begin
        sms[1, 1] = sx1; sms[2, 1] = sx2; 
        sms[1, 2] = sy1; sms[2, 2] = sy2; 
        sms[1, 3] = sz1; sms[2, 3] = sz2
    end
    return nothing
end
@inline set_scaling_features_adam_moments!(
    gamlr::GaussianAdamMomentsLearningRate, sms
) = set_scaling_features_adam_moments!(gamlr, sms...)

@inline get_rotation_features_adam_moments_view(
    gamlr::GaussianAdamMomentsLearningRate
) = @inbounds view(gamlr.gamlrv.rotation_features_moments, SOneTo(2), SOneTo(4), gamlr.idx)
@inline get_rotation_features_adam_moments_sm(
    gamlr::GaussianAdamMomentsLearningRate
) = SMatrix{2, 4}(get_rotation_features_adam_moments_view(gamlr))
@inline function set_rotation_features_adam_moments!(
    gamlr::GaussianAdamMomentsLearningRate, 
    qr1, qr2, qx1, qx2, qy1, qy2, qz1, qz2
)
    rms = get_rotation_features_adam_moments_view(gamlr)
    @inbounds begin
        rms[1, 1] = qr1; rms[2, 1] = qr2; 
        rms[1, 2] = qx1; rms[2, 2] = qx2; 
        rms[1, 3] = qy1; rms[2, 3] = qy2;
        rms[1, 4] = qz1; rms[2, 4] = qz2
    end
    return nothing
end
@inline set_rotation_features_adam_moments!(
    gamlr::GaussianAdamMomentsLearningRate, qms
) = set_rotation_features_adam_moments!(gamlr, qms...)

@inline get_sh_degree(gamlr::GaussianAdamMomentsLearningRate) = 
    get_sh_degree(gamlr.gamlrv)
@inline get_sh_coeffs_adam_moments_view(gamlr::GaussianAdamMomentsLearningRate) = 
    @inbounds view(gamlr.gamlrv.sh_coeffs_moments, SOneTo(2), :, SOneTo(3), gamlr.idx)
@inline get_sh_coeffs_adam_moments_view(
    gamlr::GaussianAdamMomentsLearningRate, max_sh_degree::I
) where I = @inbounds view(
    gamlr.gamlrv.sh_coeffs_moments, 
    SOneTo(2), one(I):(max_sh_degree + one(I))^2, SOneTo(3), gamlr.idx)
@inline get_sh_coeffs_adam_moments_view(
    gamlr::GaussianAdamMomentsLearningRate, ::Val{MAX_SH_DEGREE_I64}
) where MAX_SH_DEGREE_I64 = @inbounds view(
    gamlr.gamlrv.sh_coeffs_moments, 
    SOneTo(2), SOneTo((MAX_SH_DEGREE_I64 + 1)^2), SOneTo(3), gamlr.idx)

# As always, getting a SArray of sh_coeffs-relating things might be ill-advised.
# Nevertheless:
@inline get_sh_coeffs_adam_moments_sa_val(
    gamlr::GaussianAdamMomentsLearningRate, ::Val{nb_coeffs_per_color_channel}
) where nb_coeffs_per_color_channel = 
    SArray{Tuple{2, nb_coeffs_per_color_channel, 3}}(
        get_sh_coeffs_adam_moments_view(gamlr))
@inline function get_sh_coeffs_adam_moments_sa(
    gamlr::GaussianAdamMomentsLearningRate
)
    sh_degree = get_sh_degree(gamlr)
    if sh_degree == 0
        return get_sh_coeffs_adam_moments_sa_val(gamlr, Val(1))
    elseif sh_degree == 1
        return get_sh_coeffs_adam_moments_sa_val(gamlr, Val(4))
    elseif sh_degree == 2
        return get_sh_coeffs_adam_moments_sa_val(gamlr, Val(9))
    else sh_degree == 3
        return get_sh_coeffs_adam_moments_sa_val(gamlr, Val(16))
    end
end
@inline function set_sh_coeffs_adam_moments!(
    gamlr::GaussianAdamMomentsLearningRate, shm
)
    shm_view = get_sh_coeffs_adam_moments_view(gamlr)
    shm_view .= shm
end

@inline get_sh_coeffs_dc_adam_moments_view(
    gamlr::GaussianAdamMomentsLearningRate
) = @inbounds view(gamlr.gamlrv.sh_coeffs_moments, SOneTo(2), 1, SOneTo(3), gamlr.idx)
@inline get_sh_coeffs_dc_adam_moments_sm(
    gamlr::GaussianAdamMomentsLearningRate
) = SMatrix{2, 3}(get_sh_coeffs_dc_adam_moments_view(gamlr))
@inline function set_sh_coeffs_dc_adam_moments!(
    gamlr::GaussianAdamMomentsLearningRate, 
    sh_r_1, sh_r_2, sh_g_1, sh_g_2, sh_b_1, sh_b_2
)
    sh_dc_ms = get_sh_coeffs_dc_adam_moments_view(gamlr)
    @inbounds begin
        sh_dc_ms[1, 1] = sh_r_1; sh_dc_ms[2, 1] = sh_r_2; 
        sh_dc_ms[1, 2] = sh_g_1; sh_dc_ms[2, 2] = sh_g_2; 
        sh_dc_ms[1, 3] = sh_b_1; sh_dc_ms[2, 3] = sh_b_2
    end
    return nothing
end
@inline set_sh_coeffs_dc_adam_moments!(
    gamlr::GaussianAdamMomentsLearningRate, sh_dc_ms
) = set_sh_coeffs_dc_adam_moments!(gamlr, sh_dc_ms...)


@inline get_sh_coeffs_rest_adam_moments_view(
    gamlr::GaussianAdamMomentsLearningRate
) = @inbounds @view gamlr.gamlrv.sh_coeffs_moments[SOneTo(2), 2:end, SOneTo(3), gamlr.idx]
@inline get_sh_coeffs_rest_adam_moments_view(
    gamlr::GaussianAdamMomentsLearningRate, max_sh_degree::I
) where I = @inbounds view(
    gamlr.gamlrv.sh_coeffs_moments, 
    SOneTo(2), I(2):(max_sh_degree + one(I))^2, SOneTo(3), gamlr.idx
)
@inline get_sh_coeffs_rest_adam_moments_view(
    gamlr::GaussianAdamMomentsLearningRate, ::Val{MAX_SH_DEGREE_I64}
) where MAX_SH_DEGREE_I64 = @inbounds view(
    gamlr.gamlrv.sh_coeffs_moments, 
    SOneTo(2), SUnitRange(2, (MAX_SH_DEGREE_I64 + 1)^2), SOneTo(3), gamlr.idx
)
@inline get_sh_coeffs_rest_adam_moments_sa_val(
    gamlr::GaussianAdamMomentsLearningRate, ::Val{MAX_SH_ORDER_I64}
    # i.e. nb_coeffs_per_color_channel, but more explicit that we still include dc
) where MAX_SH_ORDER_I64 = SArray{Tuple{2, MAX_SH_ORDER_I64 - 1, 3}}(
    get_sh_coeffs_adam_rest_moments_view(gamlr))
@inline function get_sh_coeffs_rest_adam_moments_sa(
    gamlr::GaussianAdamMomentsLearningRate
)
    sh_degree = get_sh_degree(gamlr)
    if sh_degree == 0
        return get_sh_coeffs_rest_adam_moments_sa_val(gamlr, Val(1))
    elseif sh_degree == 1
        return get_sh_coeffs_rest_adam_moments_sa_val(gamlr, Val(4))
    elseif sh_degree == 2
        return get_sh_coeffs_rest_adam_moments_sa_val(gamlr, Val(9))
    else sh_degree == 3
        return get_sh_coeffs_rest_adam_moments_sa_val(gamlr, Val(16))
    end
end
@inline function set_sh_coeffs_rest_adam_moments!(
    gamlr::GaussianAdamMomentsLearningRate, shrm
)
    shrm_view = get_sh_coeffs_rest_adam_moments_view(gamlr)
    shrm_view .= shrm
end

@inline get_mean_learning_rate(gamlr::GaussianAdamMomentsLearningRate) =
    get_means_learning_rate(gamlr.gamlrv)
# Learning rates are not per Gaussian, but per parameter type (e.g. opacity_features),
# so we cannot have a set_..._learning_rate!(gamlr::GaussianAdamMomentsLearningRate)
@inline get_opacity_feature_learning_rate(gamlr::GaussianAdamMomentsLearningRate) =
    get_opacity_features_learning_rate(gamlr.gamlrv)
@inline get_scaling_features_learning_rate(gamlr::GaussianAdamMomentsLearningRate) =
    get_scaling_features_learning_rate(gamlr.gamlrv)
@inline get_rotation_features_learning_rate(
    gamlr::GaussianAdamMomentsLearningRate
) = get_rotation_features_learning_rate(gamlr.gamlrv)
@inline get_sh_coeffs_dc_learning_rate(gamlr::GaussianAdamMomentsLearningRate) =
    get_sh_coeffs_dc_learning_rate(gamlr.gamlrv)
@inline get_sh_coeffs_rest_learning_rate(gamlr::GaussianAdamMomentsLearningRate) =
    get_sh_coeffs_rest_learning_rate(gamlr.gamlrv)


function Base.show(
    io::IO, ::MIME"text/plain", gamlr::GaussianAdamMomentsLearningRate
)
    @allowscalar print(io, 
        "GaussianAdamMomentsLearningRate ", 
            gamlr.idx, " out of ", length(gamlr.gamlrv), "\n",
        "  mean:\n",
        "    moments:       ", get_mean_adam_moments_sm(gamlr), "\n",
        "    learning rate: ", get_mean_learning_rate(gamlr), "\n",
        "  opacity feature:\n",
        "    moments:       ", get_opacity_feature_adam_moments_sv(gamlr), "\n",
        "    learning rate: ", get_opacity_feature_learning_rate(gamlr), "\n",
        "  scaling features:\n",
        "    moments:       ", get_scaling_features_adam_moments_sm(gamlr), "\n",
        "    learning rate: ", get_scaling_features_learning_rate(gamlr), "\n",
        "  rotation features:\n",
        "    moments:       ", get_rotation_features_adam_moments_sm(gamlr), "\n",
        "    learning rate: ", get_rotation_features_learning_rate(gamlr), "\n",
        "  spherical harmonics coefficients: \n",
        "    degree:        ", get_sh_degree(gamlr), "\n",
        "    DC color:\n",
        "      moments:     ", get_sh_coeffs_dc_adam_moments_sm(gamlr), "\n",
        "      learning rate: ", get_sh_coeffs_dc_learning_rate(gamlr), "\n",
        "   rest:\n",
        "    learning rate:   ", get_sh_coeffs_rest_learning_rate(gamlr)
    )
end

"""
    set_sh_degree!(
        mgamlrv::MutableGaussianAdamMomentsLearningRatesVector{D, I, F},
        sh_degree::I
    ) where {D, I, F}

Resize the underlying running Adam spherical harmonics coefficients gradients moments array
to a new size corresponding to the supplied `sh_degree` (if it is valid). When we are
    increasing the degree, the newly added values will be zero.
"""  # TODO: but the newly added parameters still will use the global Adam update_count... (same in ref. 3DGS)
function set_sh_degree!(
    mgamlrv::MutableGaussianAdamMomentsLearningRatesVector{D, I, F}, sh_degree::I
) where {D, I, F}
    sh_degree == get_sh_degree(mgamlrv) && return
    _check_sh_degree(sh_degree)
    
    new_max_sh_order = (sh_degree + one(I))^2
    new_size = (2, new_max_sh_order, 3, length(mgamlrv))
    mgamlrv.sh_coeffs_moments = resize_array(mgamlrv.sh_coeffs_moments, new_size)
    return nothing
end

"""
    zero_opacity_features_adam_moments!(opt::AbstractAdamOptimizer)

Set the Adam running moments for the opacity feature gradients to zero.
"""
zero_opacity_features_adam_moments!(
    gamlrv::AbstractGaussianAdamMomentsLearningRatesVector{<:Any, <:Any, F}
) where F = fill!(gamlrv.opacity_features_moments, zero(F))




@im_mutable_with_adapt struct AdamOptimizer{
    D <: ArrayDomain, I <: Integer, F <: AbstractFloat,
    GV <: AbstractTrainableGaussiansVector{D, I, F},
    MLR <: AbstractGaussianAdamMomentsLearningRatesVector{D, I, F}
} <: AbstractOptimizer{D, I, F}

    gaussians_vector::GV
    moments_learning_rates::MLR

    beta1::F
    beta2::F
    epsilon::F
    update_count::I

    function AdamOptimizer(
        gaussians_vector::GV,
        moments_learning_rates::MLR,
        beta1::F, beta2::F, epsilon::F,
        update_count::I
    ) where {
        D <: ArrayDomain, I <: Integer, F <: AbstractFloat,
        GV <: AbstractTrainableGaussiansVector{D, I, F},
        MLR <: AbstractGaussianAdamMomentsLearningRatesVector{D, I, F}
    }
        type_str = "AdamOptimizer"
        length(gaussians_vector) == length(moments_learning_rates) ||
            throw(DimensionMismatch(
                "An `AdamOptimizer`'s `gaussian_vector` and `moments_learning_rates` \
                 should  have the same `length`!
                 Received: $(length(gaussians_vector)) and $(length(moments_learning_rates))"
            ))

        _check_01(beta1, type_str, "beta1")
        _check_01(beta2, type_str, "beta2")
        _check_nonnegative(epsilon, type_str, "epsilon")
        _check_nonnegative(update_count, type_str, "update_count")
    
        return new{D, I, F, GV, MLR}(
            gaussians_vector, 
            moments_learning_rates,
            beta1, beta2, epsilon,
            update_count
        )
    end
end
# For general use we really need the mutable version, to increment update_count

"""
    struct AbstractAdamOptimizer{
        D <: ArrayDomain, I <: Integer, F <: AbstractFloat,
        GV <: AbstractTrainableGaussiansVector{D, I, F},
        MLR <: AbstractGaussianAdamMomentsLearningRatesVector{D, I, F}
    } <: AbstractOptimizer{D, I, F}

Abstract supertype of [`MutableAdamOptimizer`](@ref) and [`ImmutableAdamOptimizer`](@ref).
"""
AbstractAdamOptimizer

"""
    struct ImmutableAdamOptimizer{
        D <: ArrayDomain, I <: Integer, F <: AbstractFloat,
        GV <: AbstractTrainableGaussiansVector{D, I, F},
        MLR <: AbstractGaussianAdamMomentsLearningRatesVector{D, I, F}
    } <: AbstractAdamOptimizer{D, I, F, GV, MLR}

Immutable version of [`MutableAdamOptimizer`](@ref). Should not be used directly, only
possibly inside of kernels.
"""
ImmutableAdamOptimizer

const DEFAULT_ADAM_BETA1_F64 = 0.9
const DEFAULT_ADAM_BETA2_F64 = 0.999
const DEFAULT_ADAM_EPSILON_F64 = 1e-15  # cf. reference 3DGS 
# (PyTorch defaults to 1e-8. This actually behaves quite differently regarding e.g. opacity
# resets and pruning)

@replace(
    X => (MutableAdamOptimizer, ImmutableAdamOptimizer),
    function X(
        gv::AbstractTrainableGaussiansVector{D, I, F},
        to::TrainingOptions{I, F};
        beta1 = F(DEFAULT_ADAM_BETA1_F64), 
        beta2 = F(DEFAULT_ADAM_BETA2_F64), 
        epsilon = F(DEFAULT_ADAM_EPSILON_F64),
        update_count = zero(I)
    ) where {D, I, F}
        return X(
            gv, 
            GaussianAdamMomentsLearningRatesVector(gv, to),
            F(beta1), F(beta2), F(epsilon),
            I(update_count)
        )
    end
)

"""
    struct MutableAdamOptimizer{
        D <: ArrayDomain, I <: Integer, F <: AbstractFloat,
        GV <: AbstractTrainableGaussiansVector{D, I, F},
        MLR <: AbstractGaussianAdamMomentsLearningRatesVector{D, I, F}
    } <: AbstractAdamOptimizer{D, I, F, GV, MLR}


Mutable struct representing an Adam optimizer for a `TrainableGaussiansVector`.

---

    MutableAdamOptimizer(
        gaussians_vector::GV,
        moments_learning_rates::MLR,
        beta1::F, beta2::F, epsilon::F,
        update_count::I
    ) where {
        D <: ArrayDomain, I <: Integer, F <: AbstractFloat,
        GV <: AbstractTrainableGaussiansVector{D, I, F},
        MLR <: AbstractGaussianAdamMomentsLearningRatesVector{D, I, F}
    }

Construct a `MutableAdamOptimizer` from its `TrainableGaussiansVector`,
`GaussianAdamMomentsLearningRatesVector` and the Adam hyperparameters
(factors for first and second moment exponential moving averages (`beta1` and `beta2`),
and small positive `epsilon` to avoid division by zero).

---

    MutableAdamOptimizer(
        gv::AbstractTrainableGaussiansVector{D, I, F},
        to::TrainingOptions{I, F};
        beta1 = F($DEFAULT_ADAM_BETA1_F64), 
        beta2 = F($DEFAULT_ADAM_BETA2_F64), 
        epsilon = F($DEFAULT_ADAM_EPSILON_F64),
        update_count = zero(I)
    ) where {D, I, F}

Construct a `MutableAdamOptimizer` from its `TrainableGaussiansVector` and its the
`TrainingOptions`. Unless manually specified otherwise via the keyword arguments, we will
use the standard Adam hyperparameters.
"""
MutableAdamOptimizer

@inline get_beta1(ao::AbstractAdamOptimizer) = ao.beta1
@inline function set_beta1!(mao::MutableAdamOptimizer, beta1)
    _check_01(beta1, "MutableAdamOptimizer", "beta1")
    mao.beta1 = beta1
    return nothing
end
@inline get_beta2(ao::AbstractAdamOptimizer) = ao.beta2
@inline function set_beta2!(mao::MutableAdamOptimizer, beta2)
    _check_01(beta2, "MutableAdamOptimizer", "beta2")
    mao.beta2 = beta2
    return nothing
end
@inline get_epsilon(ao::AbstractAdamOptimizer) = ao.epsilon
@inline function set_epsilon!(mao::MutableAdamOptimizer, epsilon)
    _check_nonnegative(epsilon, "MutableAdamOptimizer", "epsilon")
    mao.epsilon = epsilon
    return nothing
end
@inline get_update_count(ao::AbstractAdamOptimizer) = ao.update_count
@inline function set_update_count!(mao::MutableAdamOptimizer, update_count)
    _check_nonnegative(update_count, "MutableAdamOptimizer", "update_count")
    mao.update_count = update_count
    return nothing
end

"""
    scalar_adam_update(
        param::F, grad::F,
        m1::F, m2::F, learning_rate::F, update_count::I,
        beta1::F, beta2::F, epsilon::F
    ) where {I <: Integer, F <: AbstractFloat}

Perform a single Adam update step for a scalar parameter `param`, using (scalar) gradient
`grad`, exponential moving average Adam mean (i.e. first moment) gradient `m1` and second
moment gradient `m2`, the `update_count` before this call (used for bias correction), the
`learning_rate`, and the other Adam hyperparameters.

Returns the `Tuple` consisting of [1]: the updated parameter, [2]: the updated running
average of the first moment, [3]: of the second moment.
"""
function scalar_adam_update(
    param::F, grad::F,
    m1::F, m2::F, learning_rate::F, update_count::I,
    beta1::F, beta2::F, epsilon::F
) where {I <: Integer, F <: AbstractFloat}
    t = update_count + one(I)  # 1 for the first update
    new_m1 = beta1 * m1 + (one(F) - beta1) * grad      # exponential moving average
    new_m2 = beta2 * m2 + (one(F) - beta2) * grad ^ 2  #   for second moment
    new_m1_hat = new_m1 / (one(F) - beta1 ^ t)  # bias compensation
    new_m2_hat = new_m2 / (one(F) - beta2 ^ t)
    new_param = param - learning_rate * new_m1_hat / (sqrt(new_m2_hat) + epsilon)
    return new_param, new_m1, new_m2
end

"""
    update!(
        opt::MutableAdamOptimizer{<:GPU, I}, 
        fgradients::AbstractFilteredGaussiansGradientsVector,
        sh_degree_val::Val{SH_DEGREE}
    ) where {I, SH_DEGREE}

Update all Gaussian parameters using this `AdamOptimizer` and the provided gradients.
"""
function update!(
    opt::MutableAdamOptimizer{<:GPU, I}, 
    fgradients::AbstractFilteredGaussiansGradientsVector,
    sh_degree_val::Val{SH_DEGREE}
) where {I, SH_DEGREE}
    # We update each component (array in opt.moments_learning_rates) separately
    # because this is optimal for memory locality
    adam_component_update!(
        opt, fgradients, 
        get_mean_view, get_mean3d_grad_view,
        get_mean_adam_moments_view,
        get_mean_learning_rate,
        SOneTo(3)
    )
    adam_component_update!(
        opt, fgradients, 
        get_opacity_feature_view, get_opacity_feature_grad_view,
        get_opacity_feature_adam_moments_view, 
        get_opacity_feature_learning_rate,
        SOneTo(1)
    )
    adam_component_update!(
        opt, fgradients, 
        get_scaling_features_view, get_scaling_features_grad_view,
        get_scaling_features_adam_moments_view, 
        get_scaling_features_learning_rate,
        SOneTo(3)
    )
    adam_component_update!(
        opt, fgradients, 
        get_rotation_features_view, get_rotation_features_grad_view,
        get_rotation_features_adam_moments_view,  
        get_rotation_features_learning_rate,
        SOneTo(4)
    )

    # We could use the same function for the sh_coeffs, but as we have different learning
    # rates for the degree 0 and other degrees, this would require
    # get_component_learning_rates to produce up to an `SVector{48}`
    use_all_sh_val = Val(get_sh_degree(fgradients) == SH_DEGREE)
    adam_sh_coeffs_update!(opt, fgradients, sh_degree_val, use_all_sh_val)

    opt.update_count += one(I)

    return opt.gaussians_vector
end

"""
    adam_component_update!(
        opt::AbstractAdamOptimizer{G},
        fgradients::AbstractFilteredGaussiansGradientsVector{G},
        get_component_view,
        get_component_fgradient,
        get_component_adam_moments_view,
        get_component_learning_rates,
        component_1d_range
    ) where {G <: GPU}

Update a specific component (e.g. the means, opacity features, or SH coefficients) of the
`TrainableGaussiansVector` wrapped by the optimizer `opt`, using the provided gradients.

# Arguments
- `opt`: The Adam optimizer to use, containing the Gaussians to optimize, the running
    Adam gradient moments, and the learning rates.
- `fgradients`: The gradients of the parameters of the Gaussians which took part in the
    forward pass
- `get_component_view`: A function mapping a single `Gaussian` to a view of its component
    values. Will be used for both reading and writing.
- `get_component_fgradient`: A function mapping a single `GaussianGradients` (from
    `fgradients`) to the gradients for our component.
- `get_component_adam_moments_view`: A function mapping a single 
    `GaussianAdamMomentsLearningRate` to a view of the running Adam average of the first
    two gradient moments for our component. Used for both reading and writing.
- `get_component_learning_rates`: Similarly maps a single   
    `GaussianAdamMomentsLearningRate` to the corresponding learning rate(s). The
    (type stable) output of the function can either be a scalar (of the same floating type
    as `opt`), or an `AbstractArray` with this floating type. The latter is useful if
    individual coordinates of our component vector has different learning rates. This is the
    case, for example, if we group the 3D means and opacity features into a single
    component, in which case we'd use a `SVector{4}`.
- `component_1d_range`: The 1D coordinate indices for the component vector of every
    Gaussian. If the `length` is statically known, use `SOneTo` or `SUnitRange` instead of
    `⋅ : ⋅`.
"""
function adam_component_update!(
    opt::AbstractAdamOptimizer{G},
    fgradients::AbstractFilteredGaussiansGradientsVector{G},
    get_component_view,
    get_component_fgradient,
    get_component_adam_moments_view,
    get_component_learning_rates,
    component_1d_range
) where {G <: GPU}
    nb_fgaussians = length(fgradients)
    @run_1d_kernel nb_fgaussians adam_component_update_kernel!(
        opt, fgradients, nb_fgaussians,
        get_component_view,
        get_component_fgradient,
        get_component_adam_moments_view,
        get_component_learning_rates,
        component_1d_range
    )
end

"""
    adam_component_update_kernel!(
        opt::AbstractAdamOptimizer{Device, I, F},
        fgradients::AbstractFilteredGaussiansGradientsVector{Device, I, F}, nb_fgaussians,
        get_component_view,
        get_component_fgradient,
        get_component_adam_moments_view
        get_component_learning_rates,
        component_1d_range
    ) where {I, F}

CUDA kernel for [`adam_component_update!`](@ref). Every thread corresponds to a filtered
Gaussian, of which there are `nb_fgaussians`. When a (filtered) GaussianGradients `fgg` has
`get_gidx(fgg) <= 0`, we consider the gradients to be invalid and skip the Gaussian.
(This is used in adaptive density control.)
"""
function adam_component_update_kernel!(
    opt::AbstractAdamOptimizer{Device, I, F},
    fgradients::AbstractFilteredGaussiansGradientsVector{Device, I, F}, nb_fgaussians,
    get_component_view,
    get_component_fgradient,
    get_component_adam_moments_view,
    get_component_learning_rates,
    component_1d_range
) where {I, F}

    fgidx = threadIdx().x + (blockIdx().x - one(I)) * blockDim().x
    stride = blockDim().x * gridDim().x

    @inbounds while fgidx <= nb_fgaussians
        fgg = fgradients[fgidx]
        gidx = get_gidx(fgg)
        if gidx <= 0
            fgidx += stride
            continue
        end
        g = opt.gaussians_vector[gidx]
        gamlr = opt.moments_learning_rates[gidx]

        params = get_component_view(g)
        grads = get_component_fgradient(fgg)
        ms = get_component_adam_moments_view(gamlr)
        lrs = get_component_learning_rates(gamlr)

        for i in component_1d_range
            new_param, new_m1, new_m2 = scalar_adam_update(
                params[i], grads[i],
                ms[1, i], ms[2, i], lrs isa F ? lrs : lrs[i],
                opt.update_count,
                opt.beta1, opt.beta2, opt.epsilon
            )
            params[i] = new_param
            ms[1, i] = new_m1
            ms[2, i] = new_m2
        end

        fgidx += stride
    end
    return
end

"""
    adam_sh_coeffs_update!(
        opt::AbstractAdamOptimizer{G},
        fgradients::AbstractFilteredGaussiansGradientsVector{G},
        ::Val{SH_DEGREE}, ::Val{USE_ALL_SH}
    ) where {G <: GPU, SH_DEGREE, USE_ALL_SH}

Update the spherical harmonics coefficients using the provided gradients.

`SH_DEGREE` is the spherical harmonics degree we use, and `USE_ALL_SH` indicates whether
this is the maximum supported by the GaussiansVector. See also
[`project_fgaussians_kernel!`](@ref).
"""
function adam_sh_coeffs_update!(
    opt::AbstractAdamOptimizer{G},
    fgradients::AbstractFilteredGaussiansGradientsVector{G},
    sh_degree_val, use_all_sh_val
) where {G <: GPU}
    nb_fgaussians = length(fgradients)
    @run_1d_kernel nb_fgaussians adam_sh_coeffs_update_kernel!(
        opt, fgradients, nb_fgaussians,
        sh_degree_val, use_all_sh_val
    )
end

"""
    adam_sh_coeffs_update_kernel!(
        opt::AbstractAdamOptimizer{Device, I, F},
        fgradients::AbstractFilteredGaussiansGradientsVector{Device, I, F}, nb_fgaussians,
        ::Val{SH_DEGREE}, ::Val{USE_ALL_SH}
    ) where {I, F}

CUDA kernel for [`adam_sh_coeffs_update!`](@ref). Every thread corresponds to a filtered
Gaussian, of which there are `nb_fgaussians`. We skip gradients with non-positive
`get_gidx`.
"""
function adam_sh_coeffs_update_kernel!(
    opt::AbstractAdamOptimizer{Device, I, F},
    fgradients::AbstractFilteredGaussiansGradientsVector{Device, I, F}, nb_fgaussians,
    ::Val{SH_DEGREE}, ::Val{USE_ALL_SH}
) where {I, F, SH_DEGREE, USE_ALL_SH}

    fgidx = threadIdx().x + (blockIdx().x - one(I)) * blockDim().x
    stride = blockDim().x * gridDim().x

    lr_dc = get_sh_coeffs_dc_learning_rate(opt.moments_learning_rates)
    lr_rest = get_sh_coeffs_rest_learning_rate(opt.moments_learning_rates)
    MAX_SH_ORDER = (SH_DEGREE + one(I))^2

    @inbounds while fgidx <= nb_fgaussians
        fgg = fgradients[fgidx]
        gidx = get_gidx(fgg)
        if gidx <= 0
            fgidx += stride
            continue
        end
        g = opt.gaussians_vector[gidx]
        gamlr = opt.moments_learning_rates[gidx]
      
        if USE_ALL_SH
            sh_coeffs_view = get_sh_coeffs_view(g)
            grads = get_sh_coeffs_grad_view(fgg)
            ms = get_sh_coeffs_adam_moments_view(gamlr)
        else
            sh_degree_i64_val = Val(Int64(SH_DEGREE))
            sh_coeffs_view = get_sh_coeffs_view(g, sh_degree_i64_val)
            grads = get_sh_coeffs_grad_view(fgg, sh_degree_i64_val)
            ms = get_sh_coeffs_adam_moments_view(gamlr, sh_degree_i64_val)
        end

        c = one(I)  # color index
        while c <= I(3)
            # Handle [i, c] and [m, i, c] (m in {1, 2}: moment order) using 1D indices
            # idx and ms_idx:
            #   idx == i + (c - 1) * MAX_SH_ORDER == i + c_offset
            # and
            #   ms_idx == m + (i - 1) * 2 + (c - 1) * MAX_SH_ORDER * 2
            #          == m + (idx - 1) * 2
            # For m == 1:
            #   ms_idx == 2 * idx - 1
            # For m == 2 we just increment this by one
            c_offset = (c - one(I)) * MAX_SH_ORDER
            idx = one(I) + c_offset  # i = 1
            ms_idx = I(2) * idx - one(I)
            new_param, new_m1, new_m2 = scalar_adam_update(
                sh_coeffs_view[idx], grads[idx],
                ms[ms_idx], ms[ms_idx + one(I)], lr_dc,
                opt.update_count,
                opt.beta1, opt.beta2, opt.epsilon
            )
            sh_coeffs_view[idx] = new_param
            ms[ms_idx] = new_m1
            ms[ms_idx + one(I)] = new_m2

            i = I(2)
            while i <= MAX_SH_ORDER
                # TODO: test split like in sh.jl
                idx = i + c_offset
                ms_idx = I(2) * idx - one(I)
                new_param, new_m1, new_m2 = scalar_adam_update(
                    sh_coeffs_view[idx], grads[idx],
                    ms[ms_idx], ms[ms_idx + one(I)], lr_rest,
                    # the lr is the only difference
                    opt.update_count,
                    opt.beta1, opt.beta2, opt.epsilon
                )
                sh_coeffs_view[idx] = new_param
                ms[ms_idx] = new_m1
                ms[ms_idx + one(I)] = new_m2

                i += one(I)
            end

            c += one(I)
        end

        fgidx += stride
    end
    return
end

function Base.show(
    io::IO, ::MIME"text/plain", 
    opt::AbstractAdamOptimizer{D, I, F}
) where {D, I, F}
    gv = opt.gaussians_vector
    print(io, 
        "$(typeof_noparams(opt)){$D, $I, $F} for a $(typeof_noparams(gv)) consisting of \
            $(length(gv)) Gaussian$(_s_when_not_1(length(gv)))\n",
        "  beta1:       ", opt.beta1, "\n",
        "  beta2:       ", opt.beta2, "\n",
        "  epsilon:      ", opt.epsilon, "\n",
        "  update count: ", opt.update_count
    )
end

@inline get_sh_degree(opt::AbstractAdamOptimizer) = get_sh_degree(
    opt.moments_learning_rates)

"""
    set_sh_degree!(opt::AbstractAdamOptimizer, sh_degree)

Set the spherical harmonics degree of the underlying moments for the SH coefficients.
"""
set_sh_degree!(opt::AbstractAdamOptimizer, sh_degree) = set_sh_degree!(
    opt.moments_learning_rates, sh_degree)

"""
    zero_opacity_features_adam_moments!(opt::AbstractAdamOptimizer)

Set the Adam running moments for the opacity feature gradients to zero.
"""
zero_opacity_features_adam_moments!(opt::AbstractAdamOptimizer) = 
    zero_opacity_features_adam_moments!(opt.moments_learning_rates)


# Adaptive density control (here and not in training.jl as we will use the internal array
# structure)
"""
    adaptive_density_control!(
        gamlrv::MutableGaussianAdamMomentsLearningRatesVector{D, I, F},
        should_prune, should_densify, should_clone,
        new_gaussian_counts_cs, nb_ngaussians
    ) where {D <: GPU, I, F}

Modify `gamlrv` in place by applying adaptive density control. Learning rates will not be
adjusted. For to-clone Gaussians the first new version the Adam moments will be carried
over, while for the second clone these will be set to zeros. In case of splitting,
both new Gaussians will use zero moments. Finally, for copied-over Gaussians
we retain the moments. This is all consistent with the reference 3DGS implementation.

The overlapping arguments have the same meaning as in the `MutableTrainableGaussiansVector`
version. `nb_ngaussians == new_gaussian_counts_cs[end]` is the number of Gaussians after
the adaptive density control update.
"""
function adaptive_density_control!(
    mgamlrv::MutableGaussianAdamMomentsLearningRatesVector{D, I, F},
    should_prune, should_densify, should_clone,
    new_gaussian_counts_cs, nb_ngaussians
) where {D <: GPU, I, F}
    nb_gaussians = length(mgamlrv)
    MGAMLRV = typeof(mgamlrv)
    for (fn, ft) in zip(fieldnames(MGAMLRV), fieldtypes(MGAMLRV))
        ft <: AbstractArray || continue  # ..._lr: just keep
        # Now fn is ..._moments
        array = getfield(mgamlrv, fn)
        new_array = create_zeros(D, F, (size(array)[1:end-1]..., nb_ngaussians))
        @run_1d_kernel nb_gaussians fill_adam_moments_component_kernel!(
            new_array,
            array, nb_gaussians,
            should_prune, should_densify, should_clone,
            new_gaussian_counts_cs
        )
        maybe_unsafe_free!(array)
        setfield!(mgamlrv, fn, new_array)
    end
end

"""
    fill_adam_moments_component_kernel!(
        new_moments_component,
        old_moments_component, nb_gaussians::I,
        should_prune, should_densify, should_clone,
        new_gaussian_counts_cs
    ) where I

Fill in the newly created `new_moments_component` (e.g. new_scaling_features_moments)
from the old one. We assume that this has been initialized with zeros, as we will only
fill it in when we should copy over the old values, that is, when we carry over a single
Gaussian, or for the first clone.

Threads correspond to original Gaussian indices.
"""
function fill_adam_moments_component_kernel!(
    new_moments_component,
    old_moments_component, nb_gaussians::I,
    should_prune, should_densify, should_clone,
    new_gaussian_counts_cs
) where I
    gidx = threadIdx().x + (blockIdx().x - one(I)) * blockDim().x
    stride = blockDim().x * gridDim().x

    CI = CartesianIndices(size(old_moments_component)[1:end-1])
    CI_len = length(CI)
    @inbounds while gidx <= nb_gaussians
        ngidx = zero(I)  # invalid
        if !should_prune[gidx]
            if should_densify[gidx]
                if should_clone[gidx]
                    ngidx = new_gaussian_counts_cs[gidx] - one(I)  # Use first clone only
                end
                # Ignore splits: leave at zero
            else
                ngidx = new_gaussian_counts_cs[gidx]  # Carry over single Gaussian
            end
        end
        if ngidx > 0
            # Linearly copy over the values from gidx to ngidx
            CI_idx = one(I)
            while CI_idx <= CI_len
                new_moments_component[Tuple(CI[CI_idx])..., ngidx] =
                    old_moments_component[Tuple(CI[CI_idx])..., gidx]
                CI_idx += one(I)
            end
        end

        gidx += stride
    end
    return
end

"""
    adaptive_density_control!(
        opt::AbstractAdamOptimizer,
        should_prune, should_clone, should_split,
        new_gaussian_counts_cs, nb_ngaussians
    )

Adjust the Adam running moments for adaptive density control. Forwards to the
version for the underlying (mutable!) GaussianAdamMomentsLearningRatesVector.
We refer to this version for further information. 
"""
adaptive_density_control!(
    opt::AbstractAdamOptimizer,
    should_prune, should_clone, should_split,
    new_gaussian_counts_cs, nb_ngaussians
) = adaptive_density_control!(
    opt.moments_learning_rates,
    should_prune, should_clone, should_split,
    new_gaussian_counts_cs, nb_ngaussians
)