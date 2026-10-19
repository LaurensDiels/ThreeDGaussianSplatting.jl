# This file cannot be merged with training.jl, as optimizer.jl uses TrainingOptions,
# and training.jl uses AbstractOptimizer.

"""
    TrainingOptions{I <: Integer, F <: AbstractFloat}

Struct which contains most options (hyperparameters) for training (apart from e.g. those
for the optimizer).
"""
struct TrainingOptions{
    I <: Integer, F <: AbstractFloat, 
}
    nb_iterations::I

    loss_dssim_vs_L1_weight::F

    initial_means_lr::F
    final_means_lr::F
    opacity_features_lr::F
    scaling_features_lr::F
    rotation_features_lr::F
    sh_coeffs_dc_lr::F
    sh_coeffs_rest_lr::F
    nb_lr_annealing_steps::I

    sh_degree_increase_interval::I

    opacity_reset_interval::I
    opacity_reset_start_iteration_nb::I  # inclusive
    opacity_reset_final_iteration_nb::I  # inclusive
    opacity_reset_max_value::F

    adc_interval::I  # We couple pruning and densification (as adaptive density control)
    adc_start_iteration_nb::I
    adc_final_iteration_nb::I
    densification_mean2d_grad_norm_threshold_ndc::F
    densification_size_threshold::F  # big: split, small: clone
    splitting_scale_factor::F  # 0 < ⋅ < 1
    pruning_min_opacity::F
    pruning_3d_max_scale::F
end

const DEFAULT_TRAINING_OPTIONS_I64_F64 = (
    nb_iterations                                = 30_000,

    loss_dssim_vs_L1_weight                      = 0.2,
    # Meaning 20% of the weight comes from the DSSIM, and 80% from the L1 loss

    base_initial_means_lr                        = 0.00016,
    # To be multiplied by the scene radius
    base_final_means_lr_rel_factor               = inv(100.),
    opacity_features_lr                          = 0.05,
    scaling_features_lr                          = 0.005,
    rotation_features_lr                         = 0.001,
    sh_coeffs_dc_lr                              = 0.0025,
    sh_coeffs_rest_lr_rel_factor                 = inv(20.),

    sh_degree_increase_interval                  = 1_000,

    opacity_reset_interval                       = 3_000,
    opacity_reset_start_iteration_nb             = 0,
    # I.e. from the start (equivalently: default_opacity_reset_interval_i64)
    opacity_reset_final_iteration_nb             = 12_000,
    # Inclusive; in reference 3DGS: 15_000, exclusive
    opacity_reset_max_value                      = 0.01,

    adc_interval                                 = 100,
    adc_start_iteration_nb                       = 600,
    # In reference 3DGS: 500, exclusive
    # (The first densification step will occur by default at iteration 600, but we need
    #  to start keeping track of accumulated gradients earlier. Follwoing the reference
    #  implementation, we do this from the start (iteration 1))
    adc_final_iteration_nb                       = 14900,
    # In reference 3DGS: 15000, exclusive
    # (No point in keeping track of the gradients between 14901 and 15000, as we won't
    #  densify anyway)
    densification_mean2d_grad_norm_threshold_ndc = 0.0002,
    base_densification_size_threshold            = 0.01,
    splitting_scale_factor                       = inv(0.8 * 2),
    # Where 2 is the number of Gaussians a split produces
    # TODO: This is consistent with the reference implementation, but should this factor not
    #  also scale with the scene radius?
    pruning_min_opacity                          = 0.005,
    base_pruning_3d_max_scale                    = 0.1
)

"""
    _default_if_nothing(I, F, symb)

Macro that binds the variable `symb` to the default value from
`DEFAULT_TRAINING_OPTIONS_I64_F64`, if the current binding is to `nothing`. We also convert
to the appropriate integer type `I` or floating point type `F`.
"""
macro _default_if_nothing(I, F, symb)
    default = DEFAULT_TRAINING_OPTIONS_I64_F64[symb]
    type = default isa Int ? I : F
    return esc(quote
        $symb = if isnothing($symb)
            $type($default)
        else
            $type($symb)
        end
    end)
end

"""
    TrainingOptions(
        scene::Scene;
        nb_iterations = nothing,
        loss_dssim_vs_L1_weight = nothing,
        base_initial_means_lr = nothing, base_final_means_lr = nothing,
        opacity_features_lr = nothing,
        scaling_features_lr = nothing, rotation_features_lr = nothing,
        sh_coeffs_dc_lr = nothing, sh_coeffs_rest_lr = nothing,
        nb_lr_annealing_steps = nothing,
        sh_degree_increase_interval = nothing,
        opacity_reset_interval = nothing,
        opacity_reset_start_iteration_nb = nothing,
        opacity_reset_final_iteration_nb = nothing,
        opacity_reset_max_value = nothing,
        adc_interval = nothing,
        adc_start_iteration_nb = nothing,
        adc_final_iteration_nb = nothing,
        densification_mean2d_grad_norm_threshold_ndc = nothing,
        base_densification_size_threshold = nothing,
        splitting_scale_factor = nothing,
        pruning_min_opacity = nothing,
        base_pruning_3d_max_scale = nothing,
        int_type = Int32, float_type = Float32
    )

Create a `TrainingOptions` from its components. When values are set to `nothing`, we will
use the 3DGS default values. For `nb_lr_annealing_steps` we default to the (initialized)
value of `nb_iterations`.

The 'base' prefix in e.g. `base_initial_means_lr`refers to the fact we still scale this by
the `scene` radius.
"""
function TrainingOptions(
    scene::Scene;
    nb_iterations = nothing,
    loss_dssim_vs_L1_weight = nothing,
    base_initial_means_lr = nothing, base_final_means_lr = nothing,
    opacity_features_lr = nothing,
    scaling_features_lr = nothing, rotation_features_lr = nothing,
    sh_coeffs_dc_lr = nothing, sh_coeffs_rest_lr = nothing,
    nb_lr_annealing_steps = nothing,
    sh_degree_increase_interval = nothing,
    opacity_reset_interval = nothing,
    opacity_reset_start_iteration_nb = nothing, 
    opacity_reset_final_iteration_nb = nothing,
    opacity_reset_max_value = nothing,
    adc_interval = nothing,
    adc_start_iteration_nb = nothing,
    adc_final_iteration_nb = nothing,
    densification_mean2d_grad_norm_threshold_ndc = nothing,
    base_densification_size_threshold = nothing,
    splitting_scale_factor = nothing,
    pruning_min_opacity = nothing,
    base_pruning_3d_max_scale = nothing,
    int_type = Int32, float_type = Float32
)
    I = int_type; F = float_type
    @_default_if_nothing I F nb_iterations
    @_default_if_nothing I F loss_dssim_vs_L1_weight
    @_default_if_nothing I F base_initial_means_lr
    base_final_means_lr = if isnothing(base_final_means_lr) 
        base_initial_means_lr * 
            F(DEFAULT_TRAINING_OPTIONS_I64_F64.base_final_means_lr_rel_factor)
    else
        F(base_final_means_lr)
    end
    @_default_if_nothing I F opacity_features_lr
    @_default_if_nothing I F scaling_features_lr
    @_default_if_nothing I F rotation_features_lr
    @_default_if_nothing I F sh_coeffs_dc_lr
    sh_coeffs_rest_lr = if isnothing(sh_coeffs_rest_lr)
        sh_coeffs_dc_lr * 
            F(DEFAULT_TRAINING_OPTIONS_I64_F64.sh_coeffs_rest_lr_rel_factor)
    else
        F(sh_coeffs_rest_lr)
    end
    nb_lr_annealing_steps = if isnothing(nb_lr_annealing_steps)
        nb_iterations
    else
        I(nb_lr_annealing_steps)
    end

    @_default_if_nothing I F sh_degree_increase_interval 
    
    @_default_if_nothing I F opacity_reset_interval   
    @_default_if_nothing I F opacity_reset_start_iteration_nb
    @_default_if_nothing I F opacity_reset_final_iteration_nb
    @_default_if_nothing I F opacity_reset_max_value

    @_default_if_nothing I F adc_interval
    @_default_if_nothing I F adc_start_iteration_nb
    @_default_if_nothing I F adc_final_iteration_nb
    @_default_if_nothing I F densification_mean2d_grad_norm_threshold_ndc
    @_default_if_nothing I F base_densification_size_threshold
    @_default_if_nothing I F splitting_scale_factor
    @_default_if_nothing I F pruning_min_opacity
    @_default_if_nothing I F base_pruning_3d_max_scale

    scene_radius = get_radius(scene)
    return TrainingOptions{I, F}(
        nb_iterations,
        loss_dssim_vs_L1_weight,
        base_initial_means_lr * scene_radius, base_final_means_lr * scene_radius,
        opacity_features_lr,
        scaling_features_lr, rotation_features_lr,
        sh_coeffs_dc_lr, sh_coeffs_rest_lr,
        nb_lr_annealing_steps,
        sh_degree_increase_interval,
        opacity_reset_interval,
        opacity_reset_start_iteration_nb, 
        opacity_reset_final_iteration_nb,
        opacity_reset_max_value,
        adc_interval,
        adc_start_iteration_nb,
        adc_final_iteration_nb,
        densification_mean2d_grad_norm_threshold_ndc,
        base_densification_size_threshold * scene_radius,
        splitting_scale_factor,
        pruning_min_opacity,
        base_pruning_3d_max_scale * scene_radius
    )
end

@inline get_nb_iterations(to::TrainingOptions) = to.nb_iterations

@inline get_loss_dssim_vs_L1_weight(to::TrainingOptions) = to.loss_dssim_vs_L1_weight

@inline get_initial_means_learning_rate(to::TrainingOptions) = to.initial_means_lr
@inline get_final_means_learning_rate(to::TrainingOptions) = to.final_means_lr
@inline get_opacity_features_learning_rate(to::TrainingOptions) = to.opacity_features_lr
@inline get_scaling_features_learning_rate(to::TrainingOptions) = to.scaling_features_lr
@inline get_rotation_features_learning_rate(to::TrainingOptions) = to.rotation_features_lr
@inline get_sh_coeffs_dc_learning_rate(to::TrainingOptions) = to.sh_coeffs_dc_lr
@inline get_sh_coeffs_rest_learning_rate(to::TrainingOptions) = to.sh_coeffs_rest_lr
@inline get_nb_lr_annealing_steps(to::TrainingOptions) = to.nb_lr_annealing_steps

@inline get_sh_degree_increase_interval(to::TrainingOptions) = 
    to.sh_degree_increase_interval

@inline get_opacity_reset_interval(to::TrainingOptions) = to.opacity_reset_interval
@inline get_opacity_reset_start_iteration_nb(to::TrainingOptions) = 
    to.opacity_reset_start_iteration_nb
@inline get_opacity_reset_final_iteration_nb(to::TrainingOptions) = 
    to.opacity_reset_final_iteration_nb
@inline get_opacity_reset_max_value(to::TrainingOptions) = to.opacity_reset_max_value

@inline get_adc_interval(to::TrainingOptions) = to.adc_interval
@inline get_adc_start_iteration_nb(to::TrainingOptions) = to.adc_start_iteration_nb
@inline get_adc_final_iteration_nb(to::TrainingOptions) = to.adc_final_iteration_nb
@inline get_densification_mean2d_grad_norm_threshold_ndc(to::TrainingOptions) = 
    to.densification_mean2d_grad_norm_threshold_ndc
@inline get_densification_size_threshold(to::TrainingOptions) =
    to.densification_size_threshold
@inline get_splitting_scale_factor(to::TrainingOptions) = to.splitting_scale_factor
@inline get_pruning_min_opacity(to::TrainingOptions) = to.pruning_min_opacity
@inline get_pruning_3d_max_scale(to::TrainingOptions) = to.pruning_3d_max_scale


@im_mutable_with_adapt struct AdaptiveDensityControlBuffers{
    D <: ArrayDomain, I <: Integer, F <: AbstractFloat,
    AMGN <: AbstractVector{F}, RCC <: AbstractVector{I},
    MSF <: AbstractVector{F},
    SP <: AbstractVector{Bool}, SD <: AbstractVector{Bool}, SC <: AbstractVector{Bool},
    NGC <: AbstractVector{UInt8}, NGCCS <: AbstractVector{I}, 
    NGTG <: AbstractVector{I}
}

    # TODO: max_aabb_diameter (cf. ref. max_radii)
    #   but 1) max_radii is not actually used in ref. as it gets reset before use
    #       2) the ref. threshold does not take into account resolution

    # All vectors except for ngidx_to_gidx have length nb_gaussians
    accumulated_mean2d_grad_norms_ndc::AMGN
    # Will store the per-Gaussian sums of the norms of the 2D mean gradients, in NDC.
    rendering_contribution_counts::RCC
    # E.g. when a Gaussian is filtered out due to invisibility, it did not contribute

    max_scale_features::MSF

    should_prune::SP
    should_densify::SD
    should_clone::SC
    
    new_gaussian_counts::NGC  # How many new Gaussians does our Gaussian spawn (0, 1 or 2)?
    new_gaussian_counts_cs::NGCCS


    ngidx_to_gidx::NGTG  # length == sum(new_gaussian_counts) == new_gaussian_counts_cs[end]
    # Maps a new Gaussian index to the original one that spawned it

    function AdaptiveDensityControlBuffers(
        accumulated_mean2d_grad_norms_ndc::AMGN, rendering_contribution_counts::RCC,
        max_scale_features::MSF,
        should_prune::SP, should_densify::SD, should_clone::SC,
        new_gaussian_counts::NGC, new_gaussian_counts_cs::NGCCS,
        ngidx_to_gidx::NGTG
    ) where {
        I <: Integer, F <: AbstractFloat,
        AMGN <: AbstractVector{F}, RCC <: AbstractVector{I},
        MSF <: AbstractVector{F},
        SP <: AbstractVector{Bool}, SD <: AbstractVector{Bool}, SC <: AbstractVector{Bool},
        NGC <: AbstractVector{UInt8}, NGCCS <: AbstractVector{I},
        NGTG <: AbstractVector{I}
    }
        type_str = "AdaptiveDensityControlBuffers"
        garrays = (
            accumulated_mean2d_grad_norms_ndc, rendering_contribution_counts,
            max_scale_features,
            should_prune, should_densify, should_clone,
            new_gaussian_counts, new_gaussian_counts_cs
        )  # per-Gaussian, size based on nb_gaussians
        arrays = (garrays..., ngidx_to_gidx)

        D = _get_common_array_domain(type_str, arrays)
        _check_same_last_axis_size(type_str, garrays)

        return new{D, I, F, AMGN, RCC, MSF, SP, SD, SC, NGC, NGCCS, NGTG}(
            accumulated_mean2d_grad_norms_ndc, rendering_contribution_counts,
            max_scale_features,
            should_prune, should_densify, should_clone,
            new_gaussian_counts, new_gaussian_counts_cs,
            ngidx_to_gidx
        )
    end
end

"""
    AbstractAdaptiveDensityControlBuffers{
        D <: ArrayDomain, I <: Integer, F <: AbstractFloat, 
        AMGN <: AbstractVector{F}, RCC <: AbstractVector{I},
        MSF <: AbstractVector{F},
        SP <: AbstractVector{Bool}, SD <: AbstractVector{Bool}, SC <: AbstractVector{Bool},
        NGC <: AbstractVector{UInt8}, NGCCS <: AbstractVector{I},
        NGTG <: AbstractVector{I}
    }

Abstract supertype for [`MutableAdaptiveDensityControlBuffers`](@ref) and
[`ImmutableAdaptiveDensityControlBuffers`](@ref).
"""
AbstractAdaptiveDensityControlBuffers

"""
    ImmutableAdaptiveDensityControlBuffers{
        D <: ArrayDomain, I <: Integer, F <: AbstractFloat, 
        AMGN <: AbstractVector{F}, RCC <: AbstractVector{I},
        MSF <: AbstractVector{F},
        SP <: AbstractVector{Bool}, SD <: AbstractVector{Bool}, SC <: AbstractVector{Bool},
        NGC <: AbstractVector{UInt8}, NGCCS <: AbstractVector{I},
        NGTG <: AbstractVector{I}
    } <: AbstractAdaptiveDensityControlBuffers{
        D, I, F, AMGN, RCC, SP, SD, SC, NGC, NGCCS, NGTG}

Immutable version of [`MutableAdaptiveDensityControlBuffers`](@ref). Only used in kernels.
"""
ImmutableAdaptiveDensityControlBuffers

function MutableAdaptiveDensityControlBuffers(
    gv::AbstractGaussiansVector{D, I, F}
) where {D, I, F}
    n = length(gv)
    return MutableAdaptiveDensityControlBuffers(
        create_zeros(D, F, n), create_zeros(D, I, n),
        create_array(D, F, n),
        create_array(D, Bool, n), create_array(D, Bool, n), create_array(D, Bool, n),
        create_array(D, UInt8, n), create_array(D, I, n),
        create_array(D, I, 0)
    )
end

"""
    MutableAdaptiveDensityControlBuffers{
        D <: ArrayDomain, I <: Integer, F <: AbstractFloat, 
        AMGN <: AbstractVector{F}, RCC <: AbstractVector{I},
        MSF <: AbstractVector{F},
        SP <: AbstractVector{Bool}, SD <: AbstractVector{Bool}, SC <: AbstractVector{Bool},
        NGC <: AbstractVector{UInt8}, NGCCS <: AbstractVector{I},
        NGTG <: AbstractVector{I}
    } <: AbstractAdaptiveDensityControlBuffers{
        D, I, F, AMGN, RCC, SP, SD, SC, NGC, NGCCS, NGTG}

Mutable struct containing (resizeable) buffers needed for adaptive density control.

---

    MutableAdaptiveDensityControlBuffers(
        accumulated_mean2d_grad_norms_ndc::AMGN, rendering_contribution_counts::RCC,
        max_scale_features::MSF,
        should_prune::SP, should_densify::SD, should_clone::SC,
        new_gaussian_counts::NGC, new_gaussian_counts_cs::NGCCS,
        ngidx_to_gidx::NGTG
    ) where {
        I <: Integer, F <: AbstractFloat,
        AMGN <: AbstractVector{F}, RCC <: AbstractVector{I},
        MSF <: AbstractVector{F},
        SP <: AbstractVector{Bool}, SD <: AbstractVector{Bool}, SC <: AbstractVector{Bool},
        NGC <: AbstractVector{UInt8}, NGCCS <: AbstractVector{I},
        NGTG <: AbstractVector{I}
    }

Construct a `MutableAdaptiveDensityControlBuffers` from its fields. Internal constructor
which performs `ArrayDomain` and `size` compatibility checks.

---

    MutableAdaptiveDensityControlBuffers(
        gv::AbstractGaussiansVector{D, I, F}
    ) where {D, I, F}

Create a `MutableAdaptiveDensityControlBuffers` from a GaussiansVector (normally a 
`MutableTrainableGaussiansVector`). Most arrays are initialized with garbage values,
except on the one hand the accumulating gradients and counts (initialized at zeros),
and on the other `ngidx_to_gidx` as we do not know in advance how many ('new') Gaussians
there will be after an adaptive density control step.
"""
MutableAdaptiveDensityControlBuffers


"""
    set_nb_gaussians!(madcb::MutableAdaptiveDensityControlBuffers, nb_gaussians)

Resize the per-Gaussian arrays to the new `nb_gaussians`. This also resets the values these
contain (to garbage, or zeros).
"""
function set_nb_gaussians!(madcb::MutableAdaptiveDensityControlBuffers, nb_gaussians)
    for fn in fieldnames(MutableAdaptiveDensityControlBuffers)
        fn == :ngidx_to_gidx && continue
        should_zero = fn in (
            :accumulated_mean2d_grad_norms_ndc, :rendering_contribution_counts)
        array = getfield(madcb, fn)
        setfield!(madcb, fn, 
            resize_memory(array, (size(array)[1:end-1]..., nb_gaussians);
                          zero = should_zero))
    end
end

"""
    set_nb_ngaussians!(madcb::MutableAdaptiveDensityControlBuffers, nb_ngaussians)

Resize the per-new-Gaussian array(s) to the new `nb_ngaussians`. The new values will be
uninitialized.
"""
function set_nb_ngaussians!(madcb::MutableAdaptiveDensityControlBuffers, nb_ngaussians)
    madcb.ngidx_to_gidx = resize_memory(madcb.ngidx_to_gidx, nb_ngaussians)
end

"""
    maybe_unsafe_free!(t::Trainer{D, I, F})

Call `maybe_unsafe_free!` on the underlying arrays.
"""
function maybe_unsafe_free!(adc::AbstractAdaptiveDensityControlBuffers)
    maybe_unsafe_free!(adc.accumulated_mean2d_grad_norms_ndc)
    maybe_unsafe_free!(adc.rendering_contribution_counts)
    maybe_unsafe_free!(adc.max_scale_features)
    maybe_unsafe_free!(adc.should_prune)
    maybe_unsafe_free!(adc.should_densify)
    maybe_unsafe_free!(adc.should_clone) 
    maybe_unsafe_free!(adc.new_gaussian_counts)
    maybe_unsafe_free!(adc.new_gaussian_counts_cs)
    maybe_unsafe_free!(adc.ngidx_to_gidx)
    return nothing
end