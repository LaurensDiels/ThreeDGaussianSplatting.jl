"""
    AbstractGaussiansVector{
        D <: ArrayDomain,
        I <: Integer,
        F <: AbstractFloat
    }

Abstract supertype for GaussiansVectors, which behave like (but are different from)
`Vector`s of `Gaussian`s.

# Parameters
- `D`: The ArrayDomain of the GaussiansVector: where the underlying memory resides (CPU/GPU)
    and how it is accessible.
- `I`: The type used to represent integer data.
- `F`: The type used to represent floating point data.
"""
abstract type AbstractGaussiansVector{
    D <: ArrayDomain,
    I <: Integer,
    F <: AbstractFloat
} end


# Cannot be created within kernels
# Needs immutable variant for passing to CUDA kernels, while a mutable version is more
# convenient for densification and pruning
# We'll collect all MutableTrainableGaussiansVector docstrings later
# We cannot use @im_mutable_with_adapt, as I is not inferrable from the fields
@im_mutable struct TrainableGaussiansVector{
    D <: ArrayDomain,
    I <: Integer,
    F <: AbstractFloat, 
    MO <: AbstractMatrix{F},
    S <: AbstractMatrix{F},
    R <: AbstractMatrix{F},
    H <: AbstractArray{F, 3}
} <: AbstractGaussiansVector{D, I, F}

    means_opacity_features::MO  # 4 x n 
    # TODO: update
    # n x 3  +  n, or n x 4 would be better for create_keep_gaussians_kernel!, due to memory coalescing,
    # but in fprojection_kernel! where we make a lot of (fgidx_to_gidx) jumps in memory, 4 x n is better 
    # (when loading a Gaussian's mean and opacity (feature) right after one another), due to caching.
    # As 4 is not too long, in our tests we only noticed a tiny (10 µs) performance loss in create_keep_gaussians_kernel! 
    # compared to n x 3, but a small (60 µs) performance increase in fprojection_kernel!.

    scaling_features::S         # 3 x n; better for caching behavior, cf. above
    rotation_features::R        # 4 x n 
    # (You would expect that combining scaling and rotation features into one should help this caching, 
    # but at best we got the same performance.)
    
    sh_coeffs::H                # (sh_degree + 1)^2 x 3 x n

    function TrainableGaussiansVector(
        means_opacity_features::MO, 
        scaling_features::S, rotation_features::R,
        sh_coeffs::H;
        int_type::Type{<:Integer},
    ) where {
        F <: AbstractFloat,
        MO <: AbstractMatrix{F},
        S <: AbstractMatrix{F},
        R <: AbstractMatrix{F},
        H <: AbstractArray{F, 3}
    }

        type_str = "TrainableGaussiansVector"
        # In principle Mutable... or Immutable..., but this should be clear enough
        arrays = (means_opacity_features, scaling_features, rotation_features, sh_coeffs)

        D = _get_common_array_domain(type_str, arrays)
        n = _check_same_last_axis_size(type_str, arrays)
        
        size(means_opacity_features) == (4, n) || 
            throw(DimensionMismatch(
                "Expected size (4, $n) for `means_opacity_features` \
                 (where 4 = 3 (`means`) + 1 (`opacity_features``))! \
                 Received: $(size(means_opacity_features))!"))
        _size_check(scaling_features, (3, n), "scaling_features")
        # The generic error message suffices here
        size(rotation_features) == (4, n) || 
            throw(DimensionMismatch(
                "Expected size (4, $n) for `rotation_features` (i.e. quaternions)! \
                 Received: $(size(rotation_features))!"))
        # (Our error messages are a bit inconsistent, as sometimes we mention the type_str,
        #  and sometimes we do not. I really don't think this is a problem, but at some
        #  point we might want to use a more uniform approach.)

        _check_sh_coeffs(type_str, sh_coeffs)

        return new{D, int_type, F, MO, S, R, H}(
            means_opacity_features, 
            scaling_features, 
            rotation_features, 
            sh_coeffs
        )
    end
end
# Manual adapt
Adapt.adapt_structure(to, mtgv::MutableTrainableGaussiansVector) = 
    MutableTrainableGaussiansVector(
        (adapt(to, getfield(mtgv, fieldname))
          for fieldname in fieldnames(MutableTrainableGaussiansVector)
        )...;
        int_type = int_type(mtgv)  # The reason we can't use adapt_im_mutable
    )
Adapt.adapt_structure(to, itgv::ImmutableTrainableGaussiansVector) = 
    ImmutableTrainableGaussiansVector(
        (adapt(to, getfield(itgv, fieldname))
          for fieldname in fieldnames(ImmutableTrainableGaussiansVector)
        )...;
        int_type = int_type(itgv)
    )
Adapt.adapt_structure(to::CUDA.KernelAdaptor, mtgv::MutableTrainableGaussiansVector) = 
    ImmutableTrainableGaussiansVector(
        (adapt(to, getfield(mtgv, fieldname))
          for fieldname in fieldnames(MutableTrainableGaussiansVector)
        )...;
        int_type = int_type(mtgv)
    )


"""
    _get_common_array_domain(type_str, arrays)

Check that the provided `arrays` (e.g. as a `Tuple`) have the same `ArrayDomain`. If so,
return this `ArrayDomain`. Otherwise, throw a descriptive error.
"""
function _get_common_array_domain(type_str, arrays)
    allequal(array_domain, arrays) || throw(ArgumentError(
            "All arrays of a$(_n_when_starts_with_vowel(type_str)) `$type_str` need to \
             have the same `ArrayDomain`. Received: $DMO, $DS, $DR and $DH."))
    return array_domain(first(arrays))
end


"""
    _n_when_starts_with_vowel(str)

Check if `str` starts with a vowel (including 'u'). If so, return "n", otherwise, return "".
This can then be appended to a "a" or "A".
"""
function _n_when_starts_with_vowel(str)
    if first(str) in ('a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U')
        return "n"
    else
        return ""
    end
end
# (We don't actually need this here, but only later for the AdamOptimizer.)

# In a similar vein
"""
    _s_when_not_1(val)

Return "s" when `val != 1` and "" otherwise. Can (often) be used to pluralize words.
"""
function _s_when_not_1(val)
    if val == 1
        return ""
    else
        return "s"
    end
end


"""
    _check_same_last_axis_size(
        type_str, arrays, val = nothing, arrays_type_str = "arrays"
    )

Check that all provided `arrays` have the same `size` in their last axis. If `val` is not
`nothing`, we also check that this common size is equal to `val`. If not, we throw
a descriptive error where we refer to the arrays as `arrays_type_str` (default: "arrays").
If everything is fine, we return the common length.
"""
function _check_same_last_axis_size(
    type_str, arrays, val = nothing, arrays_type_str = "arrays"
)
    szs = map(arrays) do a
        size(a, ndims(a))
    end
    allequal(szs) || throw(DimensionMismatch(
        "All $arrays_type_str of a$(_n_when_starts_with_vowel(type_str)) `$type_str` \
         should have the same `size` in their last axis. Received: \
         $(join(szs, ", ", " and "))."
    ))

    if !isnothing(val) && length(szs) > 0 && first(szs) != val
        throw(DimensionMismatch(
            "All $arrays_type_str of a$(_n_when_starts_with_vowel(type_str)) `$type_str` \
             should have the same `size` in their last axis, namely $val. Received: \
             $(join(szs, ", ", " and "))."
        ))
    end
    return first(szs)
end


"""
    _size_check(field, expected_size, field_name)

Check if the provided field` has size equals the `expected_size`. If not throws an error
with a basic message involving the `field_name`.
"""
function _size_check(field, expected_size, field_name)
    size(field) == expected_size || throw(DimensionMismatch(
        "Expected size $expected_sz for `$field_name`! \
         Received: $(size(field))"
    ))
end
# Will not be used much in this file, as we typically provide more descript error messages,
# but sees use in other files

"""
    _check_sh_coeffs(type_str, sh_coeffs, sh_coeffs_name = "sh_coeffs")

Check that `sh_coeffs` is in the expected format, and throw an error if not.
"""
function _check_sh_coeffs(type_str, sh_coeffs, sh_coeffs_name = "sh_coeffs")
    sh_degree = isqrt(size(sh_coeffs, 1)) - 1
    (sh_degree + 1) ^ 2 == size(sh_coeffs, 1) || 
        throw(DimensionMismatch(
            "The first dimension of a$(_n_when_starts_with_vowel(type_str)) `$type_str`'s \
             `$sh_coeffs_name` must be a perfect square `(sh_degree + 1)^2`! \
             Received: $(size(sh_coeffs, 1))"))
    sh_degree <= 3 || throw(
        DomainError(sh_degree, 
        "Only spherical harmonics up to degree 3 are supported!"))
        # No need to add a "Received: " as this is already handled by the DomainError
    size(sh_coeffs, 2) == 3 || 
        throw(DimensionMismatch(
            "The second dimension of a$(_n_when_starts_with_vowel(type_str)) `$type_str`'s \
             `$sh_coeffs_name` should have length 3, corresponding to the three RGB \
             components! Received: $(size(sh_coeffs, 2))"))
end

"""
    _check_sh_degree(sh_degree::Integer)

Throw an error if `sh_degree` is not valid, i.e. does not lie in {0, 1, 2, 3}.
"""
function _check_sh_degree(sh_degree::Integer)
    sh_degree <= 3 || throw(DomainError(
        sh_degree,
        "Only spherical harmonics degrees up to 3 are supported!"
    ))
    sh_degree >= 0 || throw(DomainError(
        sh_degree,
        "A spherical harmonics degree must be non-negative!"
    ))
end

"""
    AbstractTrainableGaussiansVector{
        D <: ArrayDomain,
        I <: Integer,
        F <: AbstractFloat,
        MO <: AbstractMatrix{F},
        S <: AbstractMatrix{F},
        R <: AbstractMatrix{F},
        H <: AbstractArray{F, 3}
    } <: AbstractGaussiansVector{D, I, F}

Abstract supertype of [`MutableTrainableGaussiansVector`](@ref) and
[`ImmutableTrainableGaussiansVector`](@ref).
"""
AbstractTrainableGaussiansVector

"""
    ImmutableTrainableGaussiansVector{
        D <: ArrayDomain,
        I <: Integer,
        F <: AbstractFloat,
        MO <: AbstractMatrix{F},
        S <: AbstractMatrix{F},
        R <: AbstractMatrix{F},
        H <: AbstractArray{F, 3}
    } <: AbstractTrainableGaussiansVector{D, I, F, MO, S, R, H}

Immutable version of [`MutableTrainableGaussiansVector`](@ref).
"""
ImmutableTrainableGaussiansVector

# From means and opacity separate
@replace(
    X => (MutableTrainableGaussiansVector, ImmutableTrainableGaussiansVector),
    function X(
        means::M,
        opacity_features::O, 
        scaling_features::S, rotation_features::R,
        sh_coeffs::H;
        int_type::Type{<:Integer} = Int32
    ) where {
        F <: AbstractFloat,
        M <: AbstractMatrix{F},
        O <: AbstractVector{F},
        S <: AbstractMatrix{F},
        R <: AbstractMatrix{F},
        H <: AbstractArray{F, 3}
    }
        return X(
            vcat(means, opacity_features'),
            scaling_features, rotation_features,
            sh_coeffs;
            int_type
        )
    end
)


"""
    RenderOnlyGaussiansVector{
        D <: ArrayDomain, 
        I <: Integer,
        F <: AbstractFloat, 
        MO <: AbstractMatrix{F},
        V <: AbstractMatrix{F},
        H <: AbstractArray{F, 3}
    } <: AbstractGaussiansVector{D, I, F}

`AbstractGaussiansVector` which is optimized for rendering alone. Cannot be used for
training. Supports broadcasting, with element type [`RenderOnlyGaussian`](@ref).

---

    RenderOnlyGaussiansVector(
        means_opacities::MO, 
        covariances::V,
        sh_coeffs::H;
        int_type::Type{<:Integer} = Int32
    ) where {
        F <: AbstractFloat,
        MO <: AbstractMatrix{F},
        V <: AbstractMatrix{F},
        H <: AbstractArray{F, 3}
    }

Construct a RenderOnlyGaussiansVector from its fields. Basic inner constructor which
performs the necessary `size` and `ArrayDomain` checks.
"""
struct RenderOnlyGaussiansVector{
    D <: ArrayDomain, 
    I <: Integer,
    F <: AbstractFloat, 
    MO <: AbstractMatrix{F},
    V <: AbstractMatrix{F},
    H <: AbstractArray{F, 3}
} <: AbstractGaussiansVector{D, I, F}

    means_opacities::MO    # 4 x n 
    covariances::V         # 6 x n; lower matrices
    # reinterpretable to SymmetricSMatrix{3, 6, F}
    sh_coeffs::H           # (sh_degree + 1)^2 x 3 x n

    function RenderOnlyGaussiansVector(
        means_opacities::MO, 
        covariances::V,
        sh_coeffs::H;
        int_type::Type{<:Integer} = Int32
    ) where {
        F <: AbstractFloat,
        MO <: AbstractMatrix{F},
        V <: AbstractMatrix{F},
        H <: AbstractArray{F, 3}
    }

        type_str = "RenderOnlyGaussiansVector"
        arrays = (means_opacities, covariances, sh_coeffs)

        D = _get_common_array_domain(type_str, arrays)
        n = _check_same_last_axis_size(type_str, arrays)
        
        size(means_opacities) == (4, n) || 
            throw(DimensionMismatch(
                "Expected size (4, $n) for `means_opacities` \
                 (where 4 = 3 (`means`) + 1 (`opacities``))! \
                 Received: $(size(means_opacities))!"))
        size(covariances) == (6, n) || 
            throw(DimensionMismatch(
                "Expected size (6, $n) for `covariances` (where 6 is the number of \
                 entries in the lower half of the symmetric covariance matrix)! \
                 Received: $(size(covariances))!"))
        _check_sh_coeffs(type_str, sh_coeffs)

        return new{D, int_type, F, MO, V, H}(
            means_opacities, 
            covariances, 
            sh_coeffs
        )
    end
end
Adapt.@adapt_structure RenderOnlyGaussiansVector


"""
    RenderOnlyGaussiansVector(
        means::M,
        opacities::O, 
        covariances::V,
        sh_coeffs::H;
        int_type::Type{<:Integer} = Int32
    ) where {
        F <: AbstractFloat,
        M <: AbstractMatrix{F},
        O <: AbstractVector{F},
        V <: AbstractMatrix{F},
        H <: AbstractArray{F, 3}
    }

Construct a `RenderOnlyGaussiansVector` from its underlying arrays (with the 3D means
and opacities separate).
"""
function RenderOnlyGaussiansVector(
    means::M,
    opacities::O, 
    covariances::V,
    sh_coeffs::H;
    int_type::Type{<:Integer} = Int32
) where {
    F <: AbstractFloat,
    M <: AbstractMatrix{F},
    O <: AbstractVector{F},
    V <: AbstractMatrix{F},
    H <: AbstractArray{F, 3}
}
    return RenderOnlyGaussiansVector(
        vcat(means, opacities'),
        covariances,
        sh_coeffs;
        int_type
    )
end


# Convert between Trainable and RenderOnly
"""
    RenderOnlyGaussiansVector(tgv::AbstractTrainableGaussiansVector, cpy = false)

Convert a (Mutable/Immutable)`TrainableGaussiansVector` into a `RenderOnlyGaussiansVector`.
When `!cpy` (the default) the output will share some memory with the input (namely
`sh_coeffs`). Set `cpy` to `true` if this is undesirable.
"""
function RenderOnlyGaussiansVector(tgv::AbstractTrainableGaussiansVector, cpy = false)
    means_opacities = copy(tgv.means_opacity_features)
    means_opacities[4, :] .= sigmoid.(means_opacities[4, :])
    covariances = _scaling_rotation_features_to_covariances(
        tgv.scaling_features, tgv.rotation_features)

    return RenderOnlyGaussiansVector(
        means_opacities,
        covariances,
        cpy ? copy(tgv.sh_coeffs) : tgv.sh_coeffs;
        int_type = int_type(tgv)
    )
end

"""
    _scaling_rotation_features_to_covariances(
        scaling_features::AbstractMatrix{F}, 
        rotation_features
    ) where F

Combine the scaling and rotation features into the corresponding (lower halves of the)
covariance matrices.
"""
function _scaling_rotation_features_to_covariances(
    scaling_features::AbstractMatrix{F}, 
    rotation_features
) where F
    covariances = similar(scaling_features, (6, size(scaling_features, 2)))
    map!(
        reinterpret(reshape, SVector{6, F}, covariances),
        reinterpret(reshape, SVector{3, F}, scaling_features),
        reinterpret(reshape, SVector{4, F}, rotation_features)
    ) do s, q
        R = quaternion_to_rotation_matrix(q)
        RS = R .* exp.(s)'
        return lower_matrix_sv(congruence_transform(RS))  #TODO
    end
    # (Could use reinterpret(map(...))), but this has a more complicated type.)
    return covariances
end

@replace(
    X => (MutableTrainableGaussiansVector, ImmutableTrainableGaussiansVector),
    function X(rogv::RenderOnlyGaussiansVector, cpy = false)
        means_opacity_features = copy(rogv.means_opacities)
        means_opacity_features[4, :] .= logit.(means_opacity_features[4, :])
        scaling_features, rotation_features = _covariances_to_scaling_rotation_features(
            rogv.covariances)

        return X(
            means_opacity_features,
            scaling_features, rotation_features,
            cpy ? copy(rogv.sh_coeffs) : rogv.sh_coeffs;
            int_type = int_type(rogv)
        )
    end
)

"""
    _covariances_to_scaling_rotation_features(
        covariance_lowers::AbstractMatrix{F}, 
    ) where F

Decompose the lower halves of covariance matrices into their corresponding scaling and
rotation features.
"""
function _covariances_to_scaling_rotation_features(
    covariance_lowers::AbstractMatrix{F}, 
) where F
    eigens = eigen.(SymmetricSMatrix.(
        reinterpret(reshape, SVector{6, F}, covariance_lowers)))
    scaling_features = similar(covariance_lowers, (3, size(covariance_lowers, 2)))
    rotation_features = similar(covariance_lowers, (4, size(covariance_lowers, 2)))
    map!(reinterpret(reshape, SVector{3, F}, scaling_features), eigens) do e
        log.(e.values)
    end
    map!(reinterpret(reshape, SVector{4, F}, rotation_features), eigens) do e
        rotation_matrix_to_quaternion(e.vectors)
    end
    
    maybe_unsafe_free!(eigens)
    return scaling_features, rotation_features
end


# Constructors from ply. MutableTrainableGaussiansVector docs follow below.
@replace(
    X => (MutableTrainableGaussiansVector, ImmutableTrainableGaussiansVector),
    function X(
        ply_path::AbstractString;
        array_domain::Type{<:ArrayDomain} = GPU{CUDA.DeviceMemory},
        int_type::Type{<:Integer} = Int32,
        float_type::Type{<:AbstractFloat} = Float32
    )
        D = array_domain; F = float_type
        arrs = map(_read_3DGS_ply(ply_path)) do arr
            D(array_convert(F, arr))
        end
        return X(arrs...; int_type)
    end
)


"""
    _read_3DGS_ply(ply_path)

Extract the 3D means, opacity, scaling and rotation features, and the spherical harmonics
coefficients stored in the `ply_path`. Normally the values are in `Float32`.
"""
function _read_3DGS_ply(ply_path)
    ply = load_ply(ply_path)
    vertices = ply["vertex"]
    n = length(vertices)

    means = stack((vertices["x"], vertices["y"], vertices["z"]), dims=1)  # 3 x n
    opacity_features = vertices["opacity"]
    
    sh_deg_0_coeffs = stack(
        (vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"]), 
        dims=1)  # 3 x n
    sh_other_coeffs_names = @_ vertices.properties .|> 
        plyname |> 
        filter(contains(_, "f_rest"), __) |> 
        sort(__, by=parse(Int, split(_, "_")[end]))
    # length d, where d is the spherical harmonics degree
    # Color channel 1 has features f_dc_0, f_rest_0, f_rest_1, ...; 
    #       channel 2 has f_dc_1, f_rest_<(d + 1)^2>, f_rest_<(d + 1)^2 + 1>, ...
    if isempty(sh_other_coeffs_names)
        sh_coeffs = reshape(sh_deg_0_coeffs, 1, 3, n)
    else
        sh_other_coeffs = @_ stack(vertices[_], sh_other_coeffs_names, dims=1) |> 
            reshape(__, :, 3, n)
        # ((d + 1)^2 - 1) ⋅ 3 x n ([:, 1] contains the first channel's features) 
        #   -> ((d + 1)^2 - 1) x 3 x n
        sh_coeffs = cat(sh_deg_0_coeffs[[CartesianIndex()], :, :], sh_other_coeffs, dims=1) 
        # (d + 1)^2 x 3 x n
        # We use this order as we the typical loop structure would be:
        #    for gaussian, for channel, for feature
    end

    scaling_features = stack(
        (vertices["scale_0"], vertices["scale_1"], vertices["scale_2"]), 
        dims=1)  # 3 x n
    rotation_features = stack(
        (vertices["rot_0"], vertices["rot_1"], vertices["rot_2"], vertices["rot_3"]),
        dims=1)  # 4 x n

    return means, opacity_features, scaling_features, rotation_features, sh_coeffs
end


# Final constructors: from a colored point cloud. MutableTrainableGaussiansVector docs
# follow below.
@replace(
    X => (MutableTrainableGaussiansVector, ImmutableTrainableGaussiansVector),
    function X(
        points_cpu, rgb_colors_cpu;
        array_domain::Type{<:ArrayDomain} = GPU{CUDA.DeviceMemory},
        int_type::Type{<:Integer} = Int32,
        float_type::Type{<:AbstractFloat} = Float32
    )
        D = array_domain; F = float_type
        
        n = size(points_cpu, 2)
        means_opacity_features = create_array(D, F, (4, n))
        copyto!(view(means_opacity_features, SOneTo(3), :), points_cpu)
        means_opacity_features[4, :] .= logit(F(0.1))

        sh_coeffs = create_array(D, F, (1, 3, n))  # sh_degree 0
        copyto!(view(sh_coeffs, 1, :, :), rgb_colors_cpu)
        # Still need to convert from RGB to SH coeffs
        sh_coeffs .= (sh_coeffs .- F(0.5)) ./ F(SH_CONSTS_F64[1])
        # Inverse of sh_coeffs_to_color(Val(0), ...)

        kdtree_cpu = KDTree(points_cpu)
        _, dists_cpu = knn(kdtree_cpu, points_cpu, 4)  
        # 4 nearest neighbour distances from each point, including the point itself
        mean_dists_cpu = sum.(dists_cpu) ./ 3
        # Exclude the point as its own neighbour (with dist 0)
        initial_scaling_features = log.(D(mean_dists_cpu))
        scaling_features = create_array(D, F, (3, n))
        scaling_features .= initial_scaling_features'
        maybe_unsafe_free!(initial_scaling_features)

        rotation_features = create_zeros(D, F, (4, n))
        rotation_features[1, :] .= one(F)  # [1, 0, 0, 0] corresponds to the identity matrix

        return X(
            means_opacity_features, 
            scaling_features, rotation_features, 
            sh_coeffs; 
            int_type
        )
    end
)

"""
    MutableTrainableGaussiansVector{
        D <: ArrayDomain, 
        I <: Integer,
        F <: AbstractFloat, 
        MO <: AbstractMatrix{F},
        S <: AbstractMatrix{F},
        R <: AbstractMatrix{F},
        H <: AbstractArray{F, 3}
    } <: AbstractTrainableGaussiansVector{D, I, F, MO, S, R, H}

`AbstractGaussiansVector` which can be used for training (and rendering). There is also
an immutable version `ImmutableTrainableGaussiansVector` for use in kernels.

Its fields are `AbstractArray`s which should all have the same `ArrayDomain` `D` and
compatible sizes. We support broadcasting, with element type [`TrainableGaussian`](@ref).

---

    MutableTrainableGaussiansVector(
        means_opacity_features::MO, 
        scaling_features::S, rotation_features::R,
        sh_coeffs::H;
        int_type::Type{<:Integer} = Int32
    ) where {
        F <: AbstractFloat,
        MO <: AbstractMatrix{F},
        S <: AbstractMatrix{F},
        R <: AbstractMatrix{F},
        H <: AbstractArray{F, 3}
    }

Construct a `MutableTrainableGaussiansVector` from its fields. Basic (inner) constructor
which performs the necessary `size` and `ArrayDomain` checks.

---

    MutableTrainableGaussiansVector(
        means::M,
        opacity_features::O, 
        scaling_features::S, rotation_features::R,
        sh_coeffs::H;
        int_type::Type{<:Integer} = Int32
    ) where {
        F <: AbstractFloat,
        M <: AbstractMatrix{F},
        O <: AbstractVector{F},
        S <: AbstractMatrix{F},
        R <: AbstractMatrix{F},
        H <: AbstractArray{F, 3}
    }

Construct a `MutableTrainableGaussiansVector` from its underlying arrays (with the 3D means
and opacity features separate).

---

    MutableTrainableGaussiansVector(rogv::RenderOnlyGaussiansVector, cpy = false)

Convert a `RenderOnlyGaussiansVector` into a `MutableTrainableGaussiansVector`.
When `!cpy` (the default) the output will share some memory with the input (namely
`sh_coeffs`). Set `cpy` to `true` if this is undesirable.

---

    MutableTrainableGaussiansVector(
        ply_path::AbstractString;
        array_domain::Type{<:ArrayDomain} = GPU{CUDA.DeviceMemory},
        int_type::Type{<:Integer} = Int32,
        float_type::Type{<:AbstractFloat} = Float32
    )

Construct a `MutableTrainableGaussiansVector` given a 3DGS model saved in a .ply file at the
`ply_path`.

---

    MutableTrainableGaussiansVector(
        points_cpu, rgb_colors_cpu,
        array_domain::Type{<:ArrayDomain} = GPU{CUDA.DeviceMemory},
        int_type::Type{<:Integer} = Int32,
        float_type::Type{<:AbstractFloat} = Float32
    )

Construct a `MutableTrainableGaussiansVector` from `(3, n)` CPU arrays of point coordinates
and colors (with components in [0, 1]). This is normally only useful to initialize a
GaussiansVector for training.
"""
MutableTrainableGaussiansVector

"""
    RenderOnlyGaussiansVector(
        ply_path::AbstractString;
        array_domain::Type{<:ArrayDomain} = GPU{CUDA.DeviceMemory},
        int_type::Type{<:Integer} = Int32,
        float_type::Type{<:AbstractFloat} = Float32
    )

Construct a `RenderOnlyGaussiansVector` given a 3DGS model saved in a .ply file at the
`ply_path`.
"""
function RenderOnlyGaussiansVector(
    ply_path::AbstractString;
    array_domain::Type{<:ArrayDomain} = GPU{CUDA.DeviceMemory},
    int_type::Type{<:Integer} = Int32,
    float_type::Type{<:AbstractFloat} = Float32
)
    D = array_domain; F = float_type
    means, opacity_features, scaling_features, rotation_features, sh_coeffs = 
        map(_read_3DGS_ply(ply_path)) do arr
            D(array_convert(F, arr))
        end
    opacities = sigmoid.(opacity_features)
    covariances = _scaling_rotation_features_to_covariances(
        scaling_features, rotation_features)
    
    return RenderOnlyGaussiansVector(
        vcat(means, opacities'),
        covariances,
        sh_coeffs;
        int_type
    )
end

"""
    RenderOnlyGaussiansVector(
        points_cpu, rgb_colors_cpu,
        array_domain::Type{<:ArrayDomain} = GPU{CUDA.DeviceMemory},
        int_type::Type{<:Integer} = Int32,
        float_type::Type{<:AbstractFloat} = Float32
    )

Construct a `RenderOnlyGaussiansVector` from `(3, n)` CPU arrays of point coordinates
and colors (with components in [0, 1]). This is unlikely to be useful, as you'd normally
only use a point cloud as initialization for training (which `RenderOnlyGaussiansVector`)
does not support.
"""
function RenderOnlyGaussiansVector(
    points_cpu, rgb_colors_cpu;
    array_domain::Type{<:ArrayDomain} = GPU{CUDA.DeviceMemory},
    int_type::Type{<:Integer} = Int32,
    float_type::Type{<:AbstractFloat} = Float32
)
    D = array_domain; F = float_type
    
    n = size(points_cpu, 2)
    means_opacities = create_array(D, F, (4, n))
    copyto!(view(means_opacities, SOneTo(3), :), points_cpu)
    means_opacities[4, :] .= F(0.1)

    # sh_coeffs is the same as for TrainableGaussiansVectors
    sh_coeffs = create_array(D, F, (1, 3, n))
    copyto!(view(sh_coeffs, 1, :, :), rgb_colors_cpu)
    sh_coeffs .= (sh_coeffs .- F(0.5)) ./ F(SH_CONSTS_F64[1])

    kdtree_cpu = KDTree(points_cpu)
    _, dists_cpu = knn(kdtree_cpu, points_cpu, 4)  
    s_sqs_D = D((sum.(dists_cpu) ./ 3) .^ 2)
    # Every 3x3 covariance matrix is R * S * S' * R = diagm(s_qs[i], s_sqs[i], s_sqs[i])
    # as R = I

    covariances = create_zeros(D, F, (6, n))  # ² 0 0 ² 0 ²
    covariances[1, :] .= s_sqs_D
    covariances[4, :] .= s_sqs_D
    covariances[6, :] .= s_sqs_D
    maybe_unsafe_free!(s_sqs_D)

    return RenderOnlyGaussiansVector(
        means_opacities, 
        covariances, 
        sh_coeffs; 
        int_type
    )
end


"""
    save_to_ply(ply_path, tgv::AbstractTrainableGaussiansVector)

Save a (`Mutable`/`Immutable`)`TrainableGaussiansVector` as a 3DGS .ply file. Compatible
with the reference implementation (and the inverse of [`_read_3DGS_ply`](@ref)).

# Arguments
- `ply_path`: The path (including .ply extension) of the output .ply file.
- `tgv`: The `AbstractTrainableGaussiansVector` to save. `RenderOnlyGaussiansVector`s
    need to be converted first.

# Keyword arguments
- `force_sh_degree_3`: Certain viewers like Inria's SIBR viewer do not support rendering
    3DGS models of degree different than 3. When this parameter is set to `true` we use
    zeros to extend the spherical harmonics coefficients to degree 3. Defaults to `false`.
- `include_normals`: Similarly, certain viewers will require adding (zero) normals. Defaults
    to `false`.
- `use_float_double`: PlyIO will use `float32` and `float64` as type names for `Float32`
    and `Float64`. Many Ply readers will understand this (including the reference 3DGS
    implementation and the SIBR viewer), but e.g. SuperSplat does not. Alternative type
    names are `float` and `double`. When this parameter is set to `true` we use the latter.
    Defaults to `false`. 
"""
save_to_ply(ply_path, tgv::AbstractTrainableGaussiansVector; kwargs...) = 
    save_to_ply(ply_path, CPU(tgv); kwargs...)

function save_to_ply(
    ply_path, tgv::AbstractTrainableGaussiansVector{CPU, I, F};
    force_sh_degree_3 = false, include_normals = false, use_float_double = false
) where {I <: Integer, F <: AbstractFloat}
# (We need these bounds on I and F, as AbstractTrainableGaussiansVector is
#  considered more specific than e.g. 
#  AbstractTrainableGaussiansVector{CPU, <:Any, F}) where F )
    h = size(tgv.sh_coeffs, 1)
    _maybe_convert_to_ply_float_double = if use_float_double 
        _convert_to_ply_float_double
    else
        identity
    end

    ply = Ply()
    means_aps = (ArrayProperty(
        n, 
        _maybe_convert_to_ply_float_double(tgv.means_opacity_features[i, :])
    ) for (n, i) in (("x", 1), ("y", 2), ("z", 3)))
    # Generator of ArrayProperties, where each contains a Vector of one of the three
    # components (x, y or z)
    # Note: ArrayProperty really requires Vector, so view etc. are not allowed 
    normals_aps = if include_normals
        (ArrayProperty(n, _maybe_convert_to_ply_float_double(zeros(F, length(tgv)))) 
        for n in ("nx", "ny", "nz"))
    else
        ()
    end
    opacity_features_ap = ArrayProperty(
        "opacity", 
        _maybe_convert_to_ply_float_double(tgv.means_opacity_features[4, :]))
    scaling_features_aps = (ArrayProperty(
        "scale_$(i - 1)", 
        _maybe_convert_to_ply_float_double(tgv.scaling_features[i, :])
    ) for i = 1:3)
    # Again, each component is one ArrayProperty; zero-indexed names
    rotation_features_aps = (ArrayProperty(
        "rot_$(i - 1)", 
        _maybe_convert_to_ply_float_double(tgv.rotation_features[i, :])
    ) for i = 1:4)
    sh_coeffs_dc_aps = (ArrayProperty(
        "f_dc_$(c - 1)", 
        _maybe_convert_to_ply_float_double(tgv.sh_coeffs[1, c, :]) 
    ) for c = 1:3)
    # DC (degree 0) SH coefficients
    sh_coeffs_rest_aps = if !force_sh_degree_3
        (ArrayProperty(
            "f_rest_$((c - 1) * (h - 1) + i - 2)", 
            _maybe_convert_to_ply_float_double(tgv.sh_coeffs[i, c, :])
        ) for c = 1:3 for i = 2:h)
    else
        (ArrayProperty(
            "f_rest_$((c - 1) * 15 + i - 2)", 
             _maybe_convert_to_ply_float_double(
                i <= h ? tgv.sh_coeffs[i, c, :] : zeros(F, length(tgv)))
        ) for c = 1:3 for i = 2:16)
    end
    # All other SH coefficients, in the order of first all R components ([0, 14] 
    # if sh_degree == 3, h == 16), then all G ([15, 29]), finally all B ([30, 44])

    push!(ply, 
        PlyElement("vertex",
            means_aps...,
            normals_aps...,
            sh_coeffs_dc_aps...,
            sh_coeffs_rest_aps...,
            opacity_features_ap,
            scaling_features_aps...,
            rotation_features_aps...
        )
    )  # The order matters for the SIBR viewer!
    save_ply(ply, ply_path; ascii = false)
end

struct PlyFloat
    x::Float32
end

struct PlyDouble
    x::Float64
end

PlyIO.ply_type_name(::PlyFloat)         = "float"
PlyIO.ply_type_name(::Type{PlyFloat})   = "float"
PlyIO.ply_type_name(::PlyDouble)        = "double"
PlyIO.ply_type_name(::Type{PlyDouble})  = "double"

"""
    _convert_to_ply_float_double(x)

Convert (without allocations) a `Vector{Float32}` into a `Vector{PlyFloat}` and a
`Vector{Float64}` into a `Vector{Float64}` into a `Vector{PlyDouble}`. Other inputs are
simply returned.
"""
_convert_to_ply_float_double(x) = x
_convert_to_ply_float_double(x::Vector{Float32}) = 
    unsafe_wrap(Array, Ptr{PlyFloat}(pointer(x)), size(x))
# Similar to reinterpret(PlyFloat, x), but returns a `Vector{PlyFloat}` instead of a
# reinterpret(PlyFloat, ::Vector{Float32}), which ArrayProperty does not support.
_convert_to_ply_float_double(x::Vector{Float64}) = 
    unsafe_wrap(Array, Ptr{PlyDouble}(pointer(x)), size(x))


# Inline every small helper method
@inline array_domain(::AbstractGaussiansVector{D}) where D = D
@inline int_type(::AbstractGaussiansVector{<:Any, I}) where I = I
@inline float_type(::AbstractGaussiansVector{<:Any, <:Any, F}) where F = F

@inline Base.length(gv::AbstractGaussiansVector{<:Any, I}) where I = 
    I(size(gv.sh_coeffs, 3))  # common to both types of GaussianVectors
@inline get_sh_degree(gv::AbstractGaussiansVector{<:Any, I}) where I = 
    isqrt(I(size(gv.sh_coeffs, 1))) - one(I)

function Base.show(
    io::IO, 
    ::MIME"text/plain", 
    tgv::AbstractTrainableGaussiansVector{D, I, F}
) where {D, I, F}
    print(io, 
        "$(typeof_noparams(tgv)){$D, $I, $F} consisting of $(length(tgv)) \
                Gaussian$(_s_when_not_1(length(tgv)))\n",
        "  spherical harmonics degree: ", get_sh_degree(tgv)
    )
end

function Base.show(
    io::IO, 
    ::MIME"text/plain", 
    rgv::RenderOnlyGaussiansVector{D, I, F}
) where {D, I, F}
    print(io, 
        "$(typeof_noparams(rgv)){$D, $I, $F} consisting of $(length(rgv)) \
                Gaussian$(_s_when_not_1(length(rgv)))\n",
        "  spherical harmonics degree: ", get_sh_degree(rgv)
    )
end

# Necessary for broadcasting
@inline Base.size(gv::AbstractGaussiansVector) = (length(gv),)
@inline Base.ndims(::Type{<:AbstractGaussiansVector}) = 1
@inline Base.firstindex(::AbstractGaussiansVector{<:Any, I}) where I = one(I)
@inline Base.lastindex(gv::AbstractGaussiansVector) = length(gv)


# For i::Integer we will define getindex later. Here i is e.g. a BitVector or a StepRange.
Base.@propagate_inbounds function Base.getindex(gv::AbstractGaussiansVector, i)
    GV = typeof(gv)
    remove_parameters(GV)(
        (indexlastaxis(getfield(gv, fn), i)
         for fn in fieldnames(GV)
        )...;
        int_type = int_type(gv)
    )
end

"""
    slice!(mtgv::MutableTrainableGaussiansVector, i)

Modify `mtgv` in place by slicing the vector. Mutating (but more efficient) equivalent of
`sliced_mtgv = mtgv[i]; maybe_unsafe_free!(mtgv)`. Supports `@inbounds` at the call site.
"""
Base.@propagate_inbounds function slice!(mtgv::MutableTrainableGaussiansVector, i)
    for fn in fieldnames(MutableTrainableGaussiansVector)
        array = getfield(mtgv, fn)
        new_array = indexlastaxis(array, i)
        maybe_unsafe_free!(array)
        setfield!(mtgv, fn, new_array)
    end
    return nothing
end

Base.@propagate_inbounds function Base.view(gv::AbstractGaussiansVector, i)
    GV = typeof(gv)
    remove_parameters(GV)(
        (viewlastaxis(getfield(gv, fn), i) 
         for fn in fieldnames(GV)
        )...;
        int_type = int_type(gv)
    )
end

"""
    maybe_unsafe_free!(gv::AbstractGaussiansVector{D}) where D

Call `maybe_unsafe_free!` on all arrays in the GaussiansVector. If D <: GPU this will
`unsafe_free!` them. If not, nothing happens.
"""
function maybe_unsafe_free!(gv::GV) where GV <: AbstractGaussiansVector
    for fn in fieldnames(GV)
        maybe_unsafe_free!(getfield(gv, fn))
    end
    return nothing
end

"""
    set_sh_degree!(
        mtgv::MutableTrainableGaussiansVector{D, I, F}, sh_degree::I
    ) where {D, I, F}

Resize the underlying `sh_coeffs` array to a new size corresponding to the supplied
`sh_degree` (if it is valid). When we are increasing the degree, the newly added
coefficients will be zero, so that this operation does not affect the rendering.
"""
function set_sh_degree!(
    mtgv::MutableTrainableGaussiansVector{D, I, F}, sh_degree::I
) where {D, I, F}
    sh_degree == get_sh_degree(mtgv) && return
    _check_sh_degree(sh_degree)
    
    new_max_sh_order = (sh_degree + one(I))^2
    new_size = (new_max_sh_order, 3, length(mtgv))
    mtgv.sh_coeffs = resize_array(mtgv.sh_coeffs, new_size)
    return nothing
end

"""
    reset_opacities!(
        gv::AbstractGaussiansVector{D, I, F},
        max_opacity::F
    ) where {D, I, F}

Set the opacity of every Gaussian to the supplied value, if it is lower than the Gaussian's
current opacity.
"""
function reset_opacities! end

function reset_opacities!(
     tgv::AbstractTrainableGaussiansVector{D, I, F},
     max_opacity::F
) where {D, I, F}
    max_opacity_feature = logit(max_opacity)
    @inbounds opacity_features_view = view(tgv.means_opacity_features, 4, :)
    opacity_features_view .= min.(opacity_features_view, max_opacity_feature)
    return nothing
end

function reset_opacities!(
     rgv::RenderOnlyGaussiansVector{D, I, F},
     max_opacity::F
) where {D, I, F}
    @inbounds opacities_view = view(rgv.means_opacities, 4, :)
    opacities_view .= min.(opacities_view, max_opacity)
    return nothing
end



"""
    Gaussian{
        D <: ArrayDomain, 
        I <: Integer, 
        F <: AbstractFloat,
        GV <: AbstractGaussiansVector{D, I, F}
    }

A 3D Gaussian primitive. Element of a GaussiansVector.

!!! A `Gaussian` is defined by its index into the GaussiansVector. If the latter is mutated,
the index and hence the `Gaussian` might become invalid. 
"""
struct Gaussian{
    D <: ArrayDomain, 
    I <: Integer, 
    F <: AbstractFloat,
    # In principle not needed, but relevant (e.g. eltype of get_mean)
    # and easier to access than through GV
    GV <: AbstractGaussiansVector{D, I, F}
}
    gv::GV
    idx::I

    # General (CPU, GPU, i.e. CPUAccessibleDomain) version
    @inline function Gaussian(gv::GV, idx::Integer) where {  
        # (Not idx::I as this messes up gv[1] / gv[1i32])
        D <: ArrayDomain,
        I <: Integer,
        F <: AbstractFloat,
        GV <: AbstractGaussiansVector{D, I, F}
    }
        @boundscheck 1 <= idx <= length(gv) || 
            @noinline throw(ArgumentError(
                "The provided Gaussian index $idx is out of bounds (1:$(length(gv)))!"))
        # Inlining the throw would make this constructor much longer (also when everything
        # is fine), which might negatively affect performance
        # BoundsError would be more appropriate, but the (non-customizable) message is
        # less informative
        return new{D, I, F, GV}(gv, idx)
    end

    # Device version: no string interpolation allowed
    @inline function Gaussian(gv::GV, idx::Integer) where {
        I <: Integer,
        F <: AbstractFloat,
        GV <: AbstractGaussiansVector{Device, I, F}
    }
        @boundscheck 1 <= idx <= length(gv) || 
            @noinline throw(ArgumentError(
                "The provided Gaussian index is out of bounds!"))
        return new{Device, I, F, GV}(gv, idx)
    end
end
# Don't document the inner constructors, as the canonical way to create a Gaussian is via
# getindex on a GaussiansVector

Base.@propagate_inbounds Base.getindex(gv::AbstractGaussiansVector, idx::Integer) = 
    Gaussian(gv, idx)
Base.@propagate_inbounds Base.first(gv::AbstractGaussiansVector) = gv[1]
# 1 will get converted to I
Base.@propagate_inbounds Base.last(gv::AbstractGaussiansVector) = gv[length(gv)]
# Don't just use @inbounds, as length might be 0
@inline Base.eltype(::GV) where {D, I, F, GV <:AbstractGaussiansVector{D, I, F}} = 
    Gaussian{D, I, F, GV}
@inline Base.broadcastable(gv::AbstractGaussiansVector) = gv
@inline Base.BroadcastStyle(::Type{GV}) where {M, GV <: AbstractGaussiansVector{GPU{M}}} =
    CUDACore.CuArrayStyle{1, M}()
# CPU can use the (default) DefaultArrayStyle. (Device does not support broadcasting.)
function Base.iterate(gv::AbstractGaussiansVector{<:Any, I}, state::I = one(I)) where I
    state <= length(gv) || return nothing
    return (@inbounds gv[state], state + one(I))
end

@inline array_domain(::Gaussian{D}) where D = D
@inline int_type(::Gaussian{<:Any, I}) where I = I
@inline float_type(::Gaussian{<:Any, <:Any, F}) where F = F
# (Or we could delegate to g.gv)

@inline Base.parent(g::Gaussian) = g.gv
@inline get_idx(g::Gaussian) = g.idx
@inline get_gidx(g::Gaussian) = g.idx
# Emphasizes this is a Gaussian index (and not a ((sorted) duplicated) filtered Gaussian
# one)

"""
    RenderOnlyGaussian

An element of a `RenderOnlyGaussiansVector`. Special kind of [`Gaussian`](@ref).
"""
const RenderOnlyGaussian = Gaussian{<:Any, <:Any, <:Any, <:RenderOnlyGaussiansVector}

"""
    TrainableGaussian

An element of a `TrainableGaussiansVector`. Special kind of [`Gaussian`](@ref).
"""
const TrainableGaussian = Gaussian{<:Any, <:Any, <:Any, <:AbstractTrainableGaussiansVector}

@inline get_mean_view(g::RenderOnlyGaussian) = 
    @inbounds view(g.gv.means_opacities, SOneTo(3), g.idx)
# (SOneTo is probably not better than 1:, but you never know, 
#  and it certainly should not be worse)
# SOneTo (and SUnitRAnge) only accepts Int64, so we're mixing indexing types, but this
# should be fine
@inline get_mean_view(g::TrainableGaussian) = 
    @inbounds view(g.gv.means_opacity_features, SOneTo(3), g.idx)
@inline get_mean_sv(g::Gaussian) = SVector{3}(get_mean_view(g))
@inline function set_mean!(g::Gaussian, x, y, z) 
    mv = get_mean_view(g)
    @inbounds begin mv[1] = x; mv[2] = y; mv[3] = z end
    return nothing  # Would otherwise return z
end
@inline set_mean!(g::Gaussian, xyz) = set_mean!(g, xyz...)
# We don't use something like StaticVector{3} since everything of static length 3 is fine,
# be it an SVector{3}, an NTuple{3}, a SMatrix{3, 1}, some transpose(::...), ...
# (For all of these except NTuple, we could call Length and dispatch on that. But if we have
#  an incorrect length, we'll get an error after splatting anyway.)

@inline get_opacity_feature_view(g::TrainableGaussian) = 
    @inbounds view(g.gv.means_opacity_features, SUnitRange(4, 4), g.idx)
@inline get_opacity_feature(g::TrainableGaussian) = 
    @inbounds g.gv.means_opacity_features[4, g.idx]  
    # Will throw "ERROR: Scalar indexing is disallowed." outside of kernels
    # (unless allowscalar)
@inline function set_opacity_feature!(g::TrainableGaussian, opacity_feature)
    @inbounds g.gv.means_opacity_features[4, g.idx] = opacity_feature
    return nothing  
    # Would return opacity_feature (which is not terrible, but we'll avoid for consistency)
end
@inline get_opacity_view(g::RenderOnlyGaussian) = @inbounds g.gv.means_opacities[
    SUnitRange(4, 4), g.idx]
@inline get_opacity(g::RenderOnlyGaussian) = @inbounds g.gv.means_opacities[4, g.idx]
@inline function set_opacity!(g::RenderOnlyGaussian, opacity)
    @inbounds g.gv.means_opacities[4, g.idx] = opacity
    return nothing
end
@inline get_opacity(g::TrainableGaussian) = sigmoid(get_opacity_feature(g))
@inline set_opacity!(g::TrainableGaussian, opacity) = 
    set_opacity_feature!(g, logit(opacity))
    # Without the return nothing above, this would return logit(opacity), which is weird

@inline get_scaling_features_view(g::TrainableGaussian) = 
    @inbounds view(g.gv.scaling_features, SOneTo(3), g.idx)
@inline get_scaling_features_sv(g::TrainableGaussian) = 
    SVector{3}(get_scaling_features_view(g))
@inline function set_scaling_features!(g::TrainableGaussian, sx, sy, sz)
    sv = get_scaling_features_view(g)
    @inbounds begin sv[1] = sx; sv[2] = sy; sv[3] = sz end
    return nothing
end
@inline set_scaling_features!(g::TrainableGaussian, s) = 
    set_scaling_features!(g, s...)

@inline uniformly_scale!(g::TrainableGaussian, factor) = 
    uniformly_scale_features!(g, log(factor))
@inline function uniformly_scale_features!(g::TrainableGaussian, factor_feature)
    # factor_feature is the log of the usual factor
    sfv = get_scaling_features_view(g)
    @inbounds begin
        sfv[1] += factor_feature; sfv[2] += factor_feature; sfv[3] += factor_feature
    end
    return nothing
end

@inline get_scaling_sv(g::TrainableGaussian) = exp.(get_scaling_features_sv(g))

@inline get_rotation_features_view(g::TrainableGaussian) = 
    @inbounds view(g.gv.rotation_features, SOneTo(4), g.idx)
@inline get_rotation_features_sv(g::TrainableGaussian) = 
    SVector{4}(get_rotation_features_view(g))
@inline function set_rotation_features!(g::TrainableGaussian, qr, qx, qy, qz)
    qv = get_rotation_features_view(g)
    @inbounds begin qv[1] = qr; qv[2] = qx; qv[3] = qy; qv[3] = qz end
    return nothing
end
@inline set_rotation_features!(g::TrainableGaussian, q) = 
    set_rotation_features!(g, q...)
@inline get_rotation_sm(g::TrainableGaussian) = 
    quaternion_to_rotation_matrix(get_rotation_features_sv(g))

@inline get_covariance_lower_view(g::RenderOnlyGaussian) = 
    @inbounds @view g.gv.covariances[SOneTo(6), g.idx]
@inline get_covariance_lower_sv(g::RenderOnlyGaussian) = 
    SVector{6}(get_covariance_lower_view(g))
@inline function set_covariance_lower!(g::RenderOnlyGaussian, c11, c21, c31, c22, c32, c33)
    cv = get_covariance_lower_view(g)
    @inbounds begin 
        cv[1] = c11; cv[2] = c21; cv[3] = c31; cv[4] = c22; cv[5] = c32; cv[6] = c33
    end
    return nothing
end
@inline set_covariance_lower!(g::RenderOnlyGaussian, c) =
    set_covariance_lower!(g, c...)
@inline function uniformly_scale!(g::RenderOnlyGaussian, factor)
    factor_sq = factor * factor
    clv = get_covariance_lower_view(g)
    @inbounds begin
        clv[1] *= factor_sq; clv[2] *= factor_sq; clv[3] *= factor_sq
        clv[4] *= factor_sq; clv[5] *= factor_sq; clv[6] *= factor_sq;
    end
end
@inline get_covariance_ssm(g::RenderOnlyGaussian) = 
    SymmetricSMatrix(get_covariance_lower_sv(g))
@inline get_covariance_sm(g::RenderOnlyGaussian) = SMatrix(get_covariance_ssm(g))
# TODO: ssm everywhere
@inline function get_covariance_sm(g::TrainableGaussian)
    RS = get_rotation_sm(g) .* get_scaling_sv(g)'
    return RS * RS'
end
@inline function project_covariance_lower_sv(g::RenderOnlyGaussian, J::StaticMatrix{2, 3})
    V = get_covariance_ssm(g)
    return lower_matrix_sv(congruence_transform(J, V))
end
@inline function project_covariance_lower_sv(g::TrainableGaussian, J::StaticMatrix{2, 3})
    JRS = J * (get_rotation_sm(g) .* get_scaling_sv(g)')  # cf. get_covariance_sm
    return lower_matrix_sv(congruence_transform(JRS))
end

@inline get_sh_degree(g::Gaussian) = get_sh_degree(g.gv)
@inline get_sh_coeffs_view(g::Gaussian) = 
    @inbounds view(g.gv.sh_coeffs, :, SOneTo(3), g.idx)
@inline get_sh_coeffs_view(g::Gaussian, max_sh_degree::I) where I = 
    @inbounds view(g.gv.sh_coeffs, one(I):(max_sh_degree + one(I))^2, SOneTo(3), g.idx)
@inline get_sh_coeffs_view(g::Gaussian, ::Val{MAX_SH_DEGREE_I64}) where MAX_SH_DEGREE_I64 = 
    @inbounds view(g.gv.sh_coeffs, SOneTo((MAX_SH_DEGREE_I64 + 1)^2), SOneTo(3), g.idx)
# Recall that SOneTo only accepts Int64 (if you supply Int32, the + 1 will convert to Int64)

"""
    set_sh_coeffs!(g::Gaussian, coeffs)

Set this `Gaussian`'s spherical harmonics coefficients to the supplied values. Normally
this would be a `SMatrix{nb_coeffs_per_color_channel, 3}` where
`nb_coeffs_per_color_channel == (get_sh_degree(g) + 1)^2`. However, any type of the correct
statically known `length` is also fine (e.g. a flattened `SVector`). 

!!! We do not check that `length(coeffs)` is compatible with the `sh_degree`. Additionally
`coeffs` can require a lot of memory (in one go, e.g. as a `SMatrix`), potentially leading
to performance degradation.
"""
@inline function set_sh_coeffs!(g::Gaussian, coeffs)
    shv = get_sh_coeffs_view(g)
    shv .= coeffs
end

"""
    get_sh_coeffs_sm_val(
        g::Gaussian, ::Val{nb_coeffs_per_color_channel}
    ) where {nb_coeffs_per_color_channel

Return the spherical harmonics coefficients of this `Gaussian`, as an `SMatrix`. Type-stable
method where we input the size of the resulting `SMatrix` on its first axis as a `Val`.
For example, when the `get_sh_degree(g) == 3` we should use
`get_sh_coeffs_sm_val(g, Val(48))`.

!!! We do not check that the supplied `Val` is compatible with the actual `sh_degree`. 

!!! The returned matrix will have 48 values when `get_sh_degree(g) == 3`. For
performance-critical places this method is then best avoided, as e.g. in kernels this might
take up too many registers, causing spills to global memory. Use
`get_sh_coeffs_view` in this case.
"""
@inline get_sh_coeffs_sm_val(g::Gaussian, ::Val{nb_coeffs_per_color_channel}) where {
    nb_coeffs_per_color_channel
} = SMatrix{nb_coeffs_per_color_channel, 3}(get_sh_coeffs_view(g))

"""
    get_sh_coeffs_sm(g::Gaussian)

Return the spherical harmonics coefficients of this `Gaussian`, as an `SMatrix`.

As we have to look up the `sh_degree`, which affects the return type, this method is
type-unstable. On top of this, the same potential performance issues are applicable as in
[`get_sh_coeffs_sm_val`](@ref). Nevertheless, this method can still be useful for debugging.
"""
@inline function get_sh_coeffs_sm(g::Gaussian)
    sh_degree = get_sh_degree(g)
    if sh_degree == 0
        return get_sh_coeffs_sm_val(g, Val(1))
        # Much (250x) faster than get_sh_coeffs_sm_val(g, Val((sh_degree + 1)^2), as this
        # requires dynamic dispatch (the type of the second argument is unknown in advance)
    elseif sh_degree == 1
        return get_sh_coeffs_sm_val(g, Val(4))
    elseif sh_degree == 2
        return get_sh_coeffs_sm_val(g, Val(9))
    else sh_degree == 3
        return get_sh_coeffs_sm_val(g, Val(16))
    end
    # Other SH degrees are not supported. We already checked for this in the constructor.
end
@inline get_sh_coeffs_dc_view(g::Gaussian) = 
    @inbounds view(g.gv.sh_coeffs, 1, SOneTo(3), g.idx)
@inline get_sh_coeffs_dc_sv(g::Gaussian) = SVector{3}(get_sh_coeffs_dc_view(g))
@inline function set_sh_coeffs_dc!(g::Gaussian, fx, fy, fz)
    dcv = get_sh_coeffs_dc_view(g)
    @inbounds begin dcv[1] = fx; dcv[2] = fy; dcv[3] = fz end
    return nothing
end
@inline get_dc_color_sv(g::Gaussian) = sh_coeffs_to_color(
    Val(0), get_sh_coeffs_dc_sv(g), nothing)

"""
    get_dc_color_rgb(g::Gaussian)

Return the `RGB` DC color (corresponding to spherical harmonics degree 0; with the same
floating point `eltype` as `g`'s parent GaussiansVector) of this Gaussian.
We clamp the values to [0, 1].
"""
@inline get_dc_color_rgb(g::Gaussian) = RGB(clamp.(get_dc_color_sv(g), 0, 1)...)
@inline set_dc_color!(g, rgb) =  # StaticVector{3}-like
    set_sh_coeffs_dc!(g, color_to_sh_coeffs_dc(rgb)...)
@inline set_dc_color!(gaussian, r, g, b) = 
    set_sh_coeffs_dc!(gaussian, color_to_sh_coeffs_dc(SVector(r, g, b))...)
@inline set_dc_color!(g, rgb::RGB) = 
    set_sh_coeffs_dc!(g, red(rgb), green(rgb), blue(rgb))


@inline get_sh_coeffs_rest_view(g::Gaussian) = 
    @inbounds @view g.gv.sh_coeffs[2:end, SOneTo(3), g.idx]
@inline get_sh_coeffs_rest_view(g::Gaussian, max_sh_degree::I) where I = 
    @inbounds view(g.gv.sh_coeffs, I(2):(max_sh_degree + one(I))^2, SOneTo(3), g.idx)
@inline get_sh_coeffs_rest_view(
    g::Gaussian, ::Val{MAX_SH_DEGREE_I64}
) where MAX_SH_DEGREE_I64 = 
    @inbounds view(g.gv.sh_coeffs, 
                   SUnitRange(2, (MAX_SH_DEGREE_I64 + 1)^2), SOneTo(3), g.idx)

"""
    sample(g::TrainableGaussian, ::Val{N})

Independently sample the associated normal distribution to this TrainableGaussian `N` times,
resulting in a `SMatrix{3, N}`. 
"""
@inline function sample(g::TrainableGaussian, ::Val{N}) where N
    F = float_type(g)
    shift = get_rotation_sm(g) * (randn(SMatrix{3, N, F}) .* get_scaling_sv(g))
    # x ~ N(0, I₃) -> R * (s .* x) ~ N(0, R sI₃ (sI₃)^T R^T) = N(0, get_covariance_sm(g))
    return get_mean_sv(g) .+ shift  # ~ N(get_mean_sv(g), get_covariance_sm(g))
end
@inline resample_and_scale!(g::TrainableGaussian, factor) = 
    resample_and_scale_features!(g, log(factor))
@inline function resample_and_scale_features!(g::TrainableGaussian, factor_feature)
    set_mean!(g, sample(g, Val(1)))  # (An SMatrix{3, 1} splats the same as an SVector{3})
    uniformly_scale_features!(g, factor_feature)
end


function Base.show(io::IO, ::MIME"text/plain", g::TrainableGaussian)
    @allowscalar print(io, 
        "TrainableGaussian ", g.idx, " out of ", length(g.gv), "\n",
        "  mean: ", get_mean_view(g), "\n",
        "  opacity feature: ", get_opacity_feature(g), "\n",
        "    --> opacity: ", get_opacity(g), "\n",
        "  scaling features: ", get_scaling_features_view(g), "\n",
        "    --> scaling: ", get_scaling_sv(g), "\n",
        "  rotation features: ", get_rotation_features_view(g), "\n",
        "    --> rotation: ", get_rotation_sm(g), "\n",
        "  spherical harmonics coefficients: ", get_sh_coeffs_view(g), "\n",
        "    degree: ", get_sh_degree(g), "\n",
        "    DC color: ", get_dc_color_sv(g)
    )
end

function Base.show(io::IO, ::MIME"text/plain", g::RenderOnlyGaussian)
    @allowscalar print(io, 
        "RenderOnlyGaussian ", g.idx, " out of ", length(g.gv), "\n",
        "  mean: ", get_mean_view(g), "\n",
        "  opacity: ", get_opacity(g), "\n",
        "  covariance: ", get_covariance_sm(g), "\n", 
        "  spherical harmonics coefficients: ", get_sh_coeffs_view(g), "\n",
        "    degree: ", get_sh_degree(g), "\n",
        "    DC color: ", get_dc_color_sv(g)
    )
end


# We place the following methods in this file, as for efficiency we want to work one
# array at a time, which exposes the internal representation. In other words, we cannot
# rely on e.g. our get_mean abstraction. Conceptually, this all belongs more to
# training.jl
"""
    adaptive_density_control!(
        mtgv::MutableTrainableGaussiansVector{<:GPU},
        should_prune, should_densify, should_clone,
        new_gaussian_counts_cs, nb_ngaussians,
        splitting_scale_factor
    )

Modify `mtgv` in place by applying adaptive density control. To-prune Gaussians will be
removed, to-clone Gaussians will be copied over twice, to-split Gaussians will for the
most part also be copied over twice, but their means will be resampled and their scaling
adjusted. The remaining Gaussians are simply copied over once.

# Arguments
- `should_prune`: `Bool` vector which indicates for each Gaussian if we should prune (i.e.
    remove) it.
- `should_densify`: Similar vector, indicating whether we should densify a Gaussian.
    Should be mutually exclusive with `should_prune`, i.e. they cannot be both `true` at
    the same time. They can be both `false`, in which case we just carry the Gaussian over.
- `should_clone`: Indicates for those Gaussians we densify, whether we should clone (`true`)
    or split (`false`).
- `new_gaussian_counts_cs == accumulate(+, .!should_prune .+ should_densify)`
    where the second argument indicates how many new Gaussians an old Gaussian should spawn
    (0 if pruned, 2 if cloned or split (in which case we should not prune), 1 if carried
    over).
- `ngidx_to_gidx`: A vector of indices pointing from each new Gaussian to the original one
    that spawned it.
- `nb_ngaussians == new_gaussian_counts_cs[end]`: The total number of Gaussians after this
    adaptive density control step.
- `splitting_scale_factor`: Training options which determines how much split Gaussians
    get shrunk down.
"""
function adaptive_density_control!(
    mtgv::MutableTrainableGaussiansVector{<:GPU},
    should_prune, should_densify, should_clone,
    new_gaussian_counts_cs, ngidx_to_gidx,
    splitting_scale_factor
)
    nb_gaussians = length(mtgv)
    nb_ngaussians = length(ngidx_to_gidx)

    # When splitting, we need to sample new means and adjust the scaling_features
    new_means_opacity_features = similar(
        mtgv.means_opacity_features, (4, nb_ngaussians))
    @run_1d_kernel nb_gaussians fill_means_opacity_features_adc_kernel!(
        new_means_opacity_features,
        mtgv, nb_gaussians,
        should_prune, should_densify, should_clone,
        new_gaussian_counts_cs
    )
    maybe_unsafe_free!(mtgv.means_opacity_features)
    mtgv.means_opacity_features = new_means_opacity_features

    new_scaling_features = similar(
        mtgv.scaling_features, (3, nb_ngaussians))
    @run_1d_kernel nb_gaussians fill_scaling_features_adc_kernel!(
        new_scaling_features,
        mtgv.scaling_features, nb_gaussians,
        # Note that we cannot supply mtgv here, as the
        # ImmutableTrainableGaussiansVector{Device} constructor in the
        # adapt(KernelAdaptor, ...) would complain (rightly so) that the array sizes are
        # incompatible
        should_prune, should_densify, should_clone,
        new_gaussian_counts_cs,
        log(splitting_scale_factor)
    )
    maybe_unsafe_free!(mtgv.scaling_features)
    mtgv.scaling_features = new_scaling_features
    
    # The others arrays do not need to be adjusted, so can just be sliced (cf. slice!)
    # via ngidx_to_gidx
    for fn in fieldnames(MutableTrainableGaussiansVector)
        fn in (:means_opacity_features, :scaling_features) && continue
        array = getfield(mtgv, fn)
        new_array = indexlastaxis(array, ngidx_to_gidx)
        maybe_unsafe_free!(array)
        setfield!(mtgv, fn, new_array)
    end
    return nothing
end

"""
    fill_means_opacity_features_adc_kernel!(
        new_means_opacity_features,
        mtgv::ImmutableTrainableGaussiansVector{Device, I, F}, nb_gaussians,
        should_prune, should_densify, should_clone,
        new_gaussian_counts_cs
    ) where {I, F}

Fill in the newly created `new_means_opacity_features` from the original one. Carried-over
Gaussians retain their values, both clones receive the original values, split Gaussians
get resampled means and the original opacity features.

Threads correspond to original Gaussian indices.
"""
function fill_means_opacity_features_adc_kernel!(
    new_means_opacity_features,
    mtgv::ImmutableTrainableGaussiansVector{Device, I, F}, nb_gaussians,
    should_prune, should_densify, should_clone,
    new_gaussian_counts_cs
) where {I, F}
    gidx = threadIdx().x + (blockIdx().x - one(I)) * blockDim().x
    stride = blockDim().x * gridDim().x

    # Reinterpret as CuVector{SVector{4}} to allow block-copies (instead of having to loop)
    old_means_opacity_features_svs = reinterpret(
        reshape, SVector{4, F}, mtgv.means_opacity_features)
    new_means_opacity_features_svs = reinterpret(
        reshape, SVector{4, F}, new_means_opacity_features)

    @inbounds while gidx <= nb_gaussians
        if !should_prune[gidx]
            last_ngidx = new_gaussian_counts_cs[gidx]
            if should_densify[gidx]
                if should_clone[gidx]
                    # Just copy over, twice
                    first_ngidx = last_ngidx - one(I)
                    old_mean_opacity_sv = old_means_opacity_features_svs[gidx]
                    new_means_opacity_features_svs[first_ngidx] = old_mean_opacity_sv
                    new_means_opacity_features_svs[last_ngidx] = old_mean_opacity_sv
                else  # Split
                    # Resample, cf. sample
                    g = mtgv[gidx]
                    shifts = get_rotation_sm(g) * (randn(SMatrix{3, 2, F}) .* get_scaling_sv(g))
                    old_mean = get_mean_sv(g)
                    opacity_feature = get_opacity_feature(g)

                    first_ngidx = last_ngidx - one(I)
                    new_means_opacity_features_svs[first_ngidx] = 
                        [old_mean .+ shifts[SOneTo(3), 1]; opacity_feature]
                    new_means_opacity_features_svs[last_ngidx] = 
                        [old_mean .+ shifts[SOneTo(3), 2]; opacity_feature]
                end
            else  # Carry over
                # Just copy over, once
                new_means_opacity_features_svs[last_ngidx] = 
                    old_means_opacity_features_svs[gidx]
            end
        end
        gidx += stride
    end
    return
end

"""
    fill_scaling_features_adc_kernel!(
        new_scaling_features::CuDeviceVector{F},
        old_scaling_features, nb_gaussians,
        should_prune, should_densify, should_clone,
        new_gaussian_counts_cs::CuDeviceVector{I},
        splitting_scale_factor_feature
    ) where {I, F}

Fill in the newly created `new_scaling_features` from the original one. In general new
Gaussians receive (a copy) of the original values, but split Gaussians get shrink. In
feature (log) space, this means we add a `splitting_scale_factor_feature`.

Threads correspond to original Gaussian indices.
"""
function fill_scaling_features_adc_kernel!(
    new_scaling_features::CuDeviceMatrix{F},
    old_scaling_features, nb_gaussians,
    should_prune, should_densify, should_clone,
    new_gaussian_counts_cs::CuDeviceVector{I},
    splitting_scale_factor_feature
) where {I, F}
    gidx = threadIdx().x + (blockIdx().x - one(I)) * blockDim().x
    stride = blockDim().x * gridDim().x

    old_scaling_features_svs = reinterpret(
        reshape, SVector{3, F}, old_scaling_features)
    new_scaling_features_svs = reinterpret(
        reshape, SVector{3, F}, new_scaling_features)

    @inbounds while gidx <= nb_gaussians
        if !should_prune[gidx]
            last_ngidx = new_gaussian_counts_cs[gidx]
            old_scaling_features_sv = old_scaling_features_svs[gidx]
            if should_densify[gidx]
                if should_clone[gidx]
                    # Just copy over, twice
                    first_ngidx = last_ngidx - one(I)
                    new_scaling_features_svs[first_ngidx] = old_scaling_features_sv
                    new_scaling_features_svs[last_ngidx] = old_scaling_features_sv
                else
                    # Split: shrink
                    first_ngidx = last_ngidx - one(I)
                    shrunk_scaling_features_sv = old_scaling_features_sv .+ 
                        splitting_scale_factor_feature
                    new_scaling_features_svs[first_ngidx] = shrunk_scaling_features_sv
                    new_scaling_features_svs[last_ngidx] = shrunk_scaling_features_sv
                end
            else
                # Just copy over, once
                new_scaling_features_svs[last_ngidx] = old_scaling_features_sv
            end
        end
        gidx += stride
    end
    return
end