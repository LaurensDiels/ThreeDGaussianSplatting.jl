"""
    Trainer{
        D <: ArrayDomain, I <: Integer, F <: AbstractFloat,
        RGB <: AbstractArray{F, 3}, D_GT <: ArrayDomain, 
        GV <: MutableTrainableGaussiansVector{D, I, F},
        CWIV <: AbstractVector{<:CameraWithImage{D_GT, I, F}},
        T <: Tiling{I},
        RB <: MutableRenderBuffers{D, I, F},
        ROUT <: RenderOutputs{D, <:Any, F, RGB},  
        TO <: TrainingOptions{I, F},
        GTB <: Union{Nothing, RGB},
        LOSS <: AbstractVector{F},
        DSSIM <: AbstractArray{F, 4},
        FGRADS <: MutableFilteredGaussiansGradientsVector{D, I, F},
        OPT <: AbstractOptimizer{D, I, F},
        ADC <: MutableAdaptiveDensityControlBuffers{D, I, F}
    }

Struct for 3DGS optimization and adaptive density control.

---

    Trainer(
        gaussians_vector::GV, cameras_with_images_vector::CWIV,
        render_options::RenderOptions{I, F}, tiling::T,
        training_options::TO, optimizer::OPT
    ) where {
        D <: ArrayDomain, I <: Integer, F <: AbstractFloat, D_GT <: ArrayDomain,
        GV <: MutableTrainableGaussiansVector{D, I, F},
        CWIV <: AbstractVector{<:CameraWithImage{D_GT, I, F}},
        T <: Tiling{I},
        TO <: TrainingOptions{I, F},
        OPT <: AbstractOptimizer{D, I, F}
    }

Construct a `Trainer` from a (mutable) `gaussians_vector` to optimize, the ground truth
cameras, `render_options`, `tiling` configuration, `training_options` and (mutable)
`optimizer`.
"""
mutable struct Trainer{
    D <: ArrayDomain, I <: Integer, F <: AbstractFloat,
    RGB <: AbstractArray{F, 3}, D_GT <: ArrayDomain, 
    GV <: MutableTrainableGaussiansVector{D, I, F},
    CWIV <: AbstractVector{<:CameraWithImage{D_GT, I, F}},
    T <: Tiling{I},
    RB <: MutableRenderBuffers{D, I, F},
    ROUT <: RenderOutputs{D, <:Any, F, RGB},  
    TO <: TrainingOptions{I, F},
    GTB <: Union{Nothing, RGB},
    LOSS <: AbstractVector{F},
    DSSIM <: AbstractArray{F, 4},
    FGRADS <: MutableFilteredGaussiansGradientsVector{D, I, F},
    OPT <: AbstractOptimizer{D, I, F},
    ADC <: MutableAdaptiveDensityControlBuffers{D, I, F}
}

    const gaussians_vector::GV
    const cameras_with_images_vector::CWIV
    render_options::RenderOptions{I, F}
    # We cannot make this constant, as we need to increase the SH_DEGREE.
    # More importantly, we can then also not fully specify its type (e.g. generically,
    # via ::RO, RO <: RenderOptions{I, F}, as for the rest)
    # The complete Trainer is then not fully specified, which will lead to type instability
    # every time we need render_options, i.e. in render! and backward!. But the runtime
    # for each call completely dwarves the dynamic dispatch cost (and in total remains
    # faster than using a runtime sh_degree value), this is fine.
    const tiling::T
    const render_buffers::RB
    const render_outputs::ROUT
    # TODO: what if the images/cameras have different resolutions?

    const training_options::TO
    const gt_image_buffer::GTB
    const loss_singleton::LOSS
    const dssim_buffers::DSSIM  # TODO: via FastCUDASSIM only <: GPU is supported
    const rgb_image_grad::RGB 
    const fgradients::FGRADS
    const optimizer::OPT

    const adc_buffers::ADC

    iteration_nb::I  # of the last training_step, i.e. (typically) nb_training_steps_so_far
    const cameras_order::Vector{I}
    cameras_order_idx::I
end

function Trainer(
    gaussians_vector::GV, cameras_with_images_vector::CWIV,
    render_options::RenderOptions{I, F}, tiling::T,
    training_options::TO, optimizer::OPT
) where {
    D <: ArrayDomain, I <: Integer, F <: AbstractFloat, D_GT <: ArrayDomain,
    GV <: MutableTrainableGaussiansVector{D, I, F},
    CWIV <: AbstractVector{<:CameraWithImage{D_GT, I, F}},
    T <: Tiling{I},
    TO <: TrainingOptions{I, F},
    OPT <: AbstractOptimizer{D, I, F}
}
    gaussians_vector === optimizer.gaussians_vector || throw(ArgumentError(
        "The provided `gaussians_vector` needs to be identical to the ones for the
         `optimizer`!"
    ))

    # TODO: what if the images/cameras have different resolutions?
    first_camera = get_camera(first(cameras_with_images_vector))
    width = get_width(first_camera)
    height = get_height(first_camera)

    render_buffers = MutableRenderBuffers(gaussians_vector, tiling)
    render_outputs = RenderOutputs(
        width, height; 
        array_domain = D, int_type = I, float_type = F,
        include_transmittance = true, include_depth = false,
        # (as long as we do not have ground truth depths)
        include_last_contributing_indices = true)

    if D_GT == D
        gt_image_buffer = nothing
        # We do not need the buffer, as the loader directly gives us the image with the
        # correct array domain
    else
        gt_image_buffer = create_array(D, F, (3, height, width))
    end
    loss_singleton = create_array(D, F, 1)
    rgb_image_grad = create_array(D, F, (3, height, width))
    dssim_buffers = create_array(D, F, (height, 3, 3, width))
    fgradients = MutableFilteredGaussiansGradientsVector(render_buffers.f2dgv)
    adc_buffers = MutableAdaptiveDensityControlBuffers(gaussians_vector)
    return Trainer{
        D, I, F, typeof(rgb_image_grad), D_GT,
        GV, CWIV, T, 
        typeof(render_buffers), typeof(render_outputs), 
        TO,
        typeof(gt_image_buffer), typeof(loss_singleton), 
        typeof(dssim_buffers), typeof(fgradients),
        OPT,
        typeof(adc_buffers)
    }(
        gaussians_vector, cameras_with_images_vector,
        render_options, tiling, render_buffers, render_outputs,
        training_options, 
        gt_image_buffer, loss_singleton, dssim_buffers, rgb_image_grad, fgradients, 
        optimizer,
        adc_buffers,
        
        zero(I), 
        randperm(I(length(cameras_with_images_vector))), 
        zero(I)
    )
end

@inline array_domain(::Trainer{D}) where D = D
@inline int_type(::Trainer{<:Any, I}) where I = I
@inline float_type(::Trainer{<:Any, <:Any, F}) where F = F

# TODO: show

@inline get_iteration_nb(t::Trainer) = t.iteration_nb

@inline get_current_camera_with_image(t::Trainer) = t.cameras_with_images_vector[
    t.cameras_order[t.cameras_order_idx]]
@inline get_current_camera(t::Trainer) = get_camera(get_current_camera_with_image(t))

"""
    training_step!(t::Trainer{D, I, F, <:Any, D_GT}) where {D <: GPU, I, F, D_GT}

Perform a single training step. This consists of rendering, computing the loss,
backpropagating the gradients, and performing a optimizer step. Depending on the iteration
number, we might also increase the sherical harmonics degree, prune or clone/split
Gaussians, or reset opacities.
"""
function training_step!(t::Trainer{D, I, F, <:Any, D_GT}) where {D <: GPU, I, F, D_GT}
    t.iteration_nb += one(I)
    t.cameras_order_idx += one(I)
    if t.cameras_order_idx > length(t.cameras_order)
        randperm!(t.cameras_order)
        t.cameras_order_idx = one(I)
    end
    
    maybe_increase_sh_degree!(t)

    camera_with_image = get_current_camera_with_image(t)
    camera = get_camera(camera_with_image)

    render!(
        t.render_outputs,
        t.gaussians_vector, camera,
        t.tiling, t.render_options,
        t.render_buffers
    )
    # Dynamic dispatch on t.render_options, as we do not know statically know the SH_DEGREE
    # here. But after the dispatch, render! does statically know it, so is faster than
    # without, more than the cost of the dynamic dispatch
    rendered_image = get_rgb_image(t.render_outputs)
    with_image(camera_with_image) do gt_image
        # gt_image has array_domain D_GT
        if D_GT != D
            copyto!(t.gt_image_buffer, gt_image)
            gt_image = t.gt_image_buffer
        end
        dssim_with_gradient!(
            t.loss_singleton, t.rgb_image_grad, 
            rendered_image, gt_image,
            t.dssim_buffers)
        L1_loss = mapreduce(+, rendered_image, gt_image) do a, b
            return abs(a - b)
        end / length(rendered_image)
       
        weight = get_loss_dssim_vs_L1_weight(t.training_options)
        t.loss_singleton .= (one(F) - weight) * L1_loss .+ weight .* t.loss_singleton
        l1_px_weight = (one(F) - weight) / length(rendered_image)
        t.rgb_image_grad .= (
               ifelse.(rendered_image .> gt_image, l1_px_weight, -l1_px_weight)
            .+ weight .* t.rgb_image_grad)
        # TODO: dssim_with_gradient_and_Lp_loss! in FastCUDASSIM?
    end
    # TODO: move to function (compute_loss_with_grad?)?
    #   would also be useful to write a test

    backward!(
        t.fgradients,
        t.rgb_image_grad,
        t.render_outputs,
        t.gaussians_vector,
        camera,
        t.tiling,
        t.render_options,
        t.render_buffers
    )
    # Densify before the optimizer step, as we still need to move (only) the original
    # versions of cloned Gaussians via its gradients
    in_adc_step = maybe_densify_and_prune!(t)
    update_learning_rates!(t)
    
    update!(t.optimizer, t.fgradients, Val(get_sh_degree(t.render_options)))
    # Dynamic dispatch

    in_adc_step && complete_adc_step!(t)

    maybe_opacity_reset!(t)  
    # Reference 3DGS puts this before the optimizer step, but implicitly zeros the
    # opacity_feature gradients, so we still end up with get_opacity_reset_max_value(to)
end


"""
    maybe_increase_sh_degree!(t::Trainer{<:Any, I}) where I

Increase the spherical harmonics degree by one, if we are not already at degree 3,
and if we are at the appropriate iteration number (as determined by the training options).
"""
@inline function maybe_increase_sh_degree!(t::Trainer{<:Any, I}) where I
    t.iteration_nb % get_sh_degree_increase_interval(t.training_options) == 0 || return
    sh_degree = get_sh_degree(t.gaussians_vector)
    sh_degree < 3 || return

    sh_degree += one(I)
    t.render_options = set_sh_degree(t.render_options, sh_degree)
    set_sh_degree!(t.gaussians_vector, sh_degree)
    set_sh_degree!(t.fgradients, sh_degree)
    set_sh_degree!(t.optimizer, sh_degree)
end

"""
    maybe_opacity_reset!(t::Trainer)

Reset the opacities of all Gaussians to the value set in the training options, but only
if this amounts to an opacity decrease, and only if we are at the appropriate iteration
number.
"""
@inline function maybe_opacity_reset!(t::Trainer)
    to = t.training_options
    ((get_opacity_reset_start_iteration_nb(to) <= t.iteration_nb <= 
        get_opacity_reset_final_iteration_nb(to)) && 
            t.iteration_nb % get_opacity_reset_interval(to) == 0) || return

    reset_opacities!(t.gaussians_vector, get_opacity_reset_max_value(to))
    zero_opacity_features_adam_moments!(t.optimizer)
end

"""
    update_learning_rates!(t::Trainer{<:Any, I, F}) where {I, F}

Update (anneal) the learning rates used for the Gaussian parameters.
"""
function update_learning_rates!(t::Trainer{<:Any, I, F}) where {I, F}
    to = t.training_options
    x = clamp(
        (t.iteration_nb - one(I)) / F(get_nb_lr_annealing_steps(to) - one(I)), 
        zero(F), one(F))
    # In the first iteration (1): x == 0
    # In the last annealing iteration (nb_lr_annealing_steps) and beyond: x == 1
    # (Note that indeed if e.g. nb_lr_annealing_steps == 2, then we want to use initial_lr
    #  in iteration 1 and final_lr in iteration 2, with two steps (values))
    set_means_learning_rate!(
        t.optimizer.moments_learning_rates,
        exp(
            (1 - x) * log(get_initial_means_learning_rate(to)) + 
                 x  * log(get_final_means_learning_rate(to)))
    )
end

"""
    maybe_densify_and_prune!(t::Trainer{<:GPU, I, F}) where {I, F}

If we are at the appropriate iteration number (as determined by the training options),
apply adaptive density control, i.e. densify and prune Gaussians if needed.

We return a `Bool` indicating whether we actually performed an ADC step. In that case,
we still need to finish it after invoking the optimizer, via [`complete_adc_step!`](@ref).
"""
@inline function maybe_densify_and_prune!(t::Trainer{<:GPU, I, F}) where {I, F}
    to = t.training_options
    if t.iteration_nb <= get_adc_final_iteration_nb(to)
        # Also before get_adc_start_iteration_nb(to)
        nb_fgaussians = length(t.fgradients)
        @run_1d_kernel nb_fgaussians update_adc_stats_kernel!(
            t.adc_buffers, 
            nb_fgaussians, t.fgradients, get_current_camera(t))

        if t.iteration_nb >= get_adc_start_iteration_nb(to) && 
            t.iteration_nb % get_adc_interval(to) == 0

            average_means2d_grad_norm_ndc = t.adc_buffers.accumulated_mean2d_grad_norms_ndc
            # reuse memory
            map!(
                average_means2d_grad_norm_ndc, 
                t.adc_buffers.accumulated_mean2d_grad_norms_ndc,
                t.adc_buffers.rendering_contribution_counts
            ) do acc_grad, count
                if count >= 1
                    return acc_grad / count
                else
                    return zero(F)
                end
            end

            t.adc_buffers.max_scale_features .= maximum.(
                get_scaling_features_sv.(t.gaussians_vector))
            if t.iteration_nb <= get_opacity_reset_interval(to)
                 t.adc_buffers.should_prune .=
                    get_opacity_feature.(t.gaussians_vector) .< logit(
                        get_pruning_min_opacity(to))
            else
                t.adc_buffers.should_prune .= (
                        get_opacity_feature.(t.gaussians_vector) .< logit(
                            get_pruning_min_opacity(to))
                    .|| t.adc_buffers.max_scale_features .> log(
                            get_pruning_3d_max_scale(to))
                )
                # TODO? or 2D size
            end
            t.adc_buffers.should_densify .= (
                    .!t.adc_buffers.should_prune 
                .&& average_means2d_grad_norm_ndc .>= 
                    get_densification_mean2d_grad_norm_threshold_ndc(to)
            )
            t.adc_buffers.should_clone .= (
                    t.adc_buffers.should_densify 
                .&& t.adc_buffers.max_scale_features .<= log(
                    get_densification_size_threshold(to))
            )

            # Gaussians to prune do not spawn any new Gaussians, Gaussians to densify
            # (and not prune) spawn two, Gaussians which we don't need to densify nor prune,
            # spawn a single one (itself). We want to get an index lookup ngidx_to_gidx
            # pointing from a new Gaussian index to the original one (similar to
            # fgidx_to_gidx). For example:
            #
            # gaussian indices:          1   2   3   4   5   6   7   8
            # should_prune:              1   0   0   0   0   0   1   0
            # should_densify:            0   1   1   1   0   0   0   1
            # (should_clone:             0   0   1   0   0   0   0   1
            #  should_split:             0   1   0   1   0   0   0   0)
            # new_gaussian_counts:       0   2   2   2   1   1   0   2
            # new_gaussian_counts_cs:    0   2   4   6   7   8   8   10  -> 10 new Gaussians
            # ---------------------------------------------------------
            # new gaussian indices:         1,2 3,4 4,5  7   8      9,10
            # ngidx_to_gidx:                2,2 3,3 4,4  5   6      8,8

            t.adc_buffers.new_gaussian_counts .= 
                I.(.!t.adc_buffers.should_prune) .+ I.(t.adc_buffers.should_densify)
            accumulate!(+, 
                t.adc_buffers.new_gaussian_counts_cs, 
                t.adc_buffers.new_gaussian_counts
            )
            nb_ngaussians = @allowscalar t.adc_buffers.new_gaussian_counts_cs[end]
            set_nb_ngaussians!(t.adc_buffers, nb_ngaussians)

            nb_gaussians = length(t.gaussians_vector)  # (old Gaussians)
            @run_1d_kernel nb_gaussians fill_ngidx_to_gidx_kernel!(
                t.adc_buffers.ngidx_to_gidx,
                nb_gaussians,
                t.adc_buffers.should_prune, t.adc_buffers.should_densify,
                t.adc_buffers.new_gaussian_counts_cs
            )

            adaptive_density_control!(
                t.gaussians_vector,
                t.adc_buffers.should_prune,
                t.adc_buffers.should_densify,
                t.adc_buffers.should_clone,
                t.adc_buffers.new_gaussian_counts_cs,
                t.adc_buffers.ngidx_to_gidx,
                get_splitting_scale_factor(to)
            )

            adaptive_density_control!(
                t.optimizer,
                t.adc_buffers.should_prune,
                t.adc_buffers.should_densify,
                t.adc_buffers.should_clone,
                t.adc_buffers.new_gaussian_counts_cs,
                nb_ngaussians
            )

            # Following the reference 3DGS implementation we still perform the optimizer
            # step, but only for original Gaussians, i.e. those that were neither pruned
            # nor densified, and the first clones. We will achieve this implicitly, by
            # altering fgidx_to_gidx to point to the ngidx if applicable and zero
            # (which gets ignored by the optimizer) otherwise.
            nb_fgaussians = length(t.fgradients)
            @run_1d_kernel nb_fgaussians adjust_fgidx_to_gidx_adc_kernel!(
                t.render_buffers.f2dgv.fgidx_to_gidx,
                nb_fgaussians, 
                t.adc_buffers.should_prune,
                t.adc_buffers.should_densify,
                t.adc_buffers.should_clone,
                t.adc_buffers.new_gaussian_counts_cs
            )

            # We still need to resize all fgradient buffers, but this can only happen
            # after the optimizer step. Also the render_buffers (containing fgidx_to_gidx)
            # should be resized, but can only be done after that step.

            # We can already resize the ADC buffers, though
            set_nb_gaussians!(t.adc_buffers, nb_ngaussians)
            # Also resets the 2D mean gradient stats to zero
            return true  # Informs us we still have to resize fradients and render_buffers
        end
    end
    return false
end

"""
    complete_adc_step!(t::Trainer{<:GPU, I, F}) where {I, F}

Complete an adaptive density control step started by [`maybe_densify_and_prune!`](@ref).
Conceretely, we resize the gradient and render buffers.

!!! This method should only be called when `maybe_densify_and_prune!` returns `true` and
after we have applied the optimizer step.
"""
function complete_adc_step!(t::Trainer{<:GPU, I, F}) where {I, F}
    nb_ngaussians = length(t.gaussians_vector)
    set_nb_gaussians!(t.fgradients, nb_ngaussians)
    set_nb_gaussians!(t.render_buffers, nb_ngaussians)
end

"""
    update_adc_stats_kernel!(
        adc_buffers::ImmutableAdaptiveDensityControlBuffers{Device, I, F}, 
        nb_fgaussians, fgradients, camera
    ) where {I, F}

Update the `adc_buffers` to keep track of the 2D mean gradient norms. We need these
statistics to decide when we need to densify.

Threads correspond to filtered Gaussian indices.
"""
function update_adc_stats_kernel!(
    adc_buffers::ImmutableAdaptiveDensityControlBuffers{Device, I, F}, 
    nb_fgaussians, fgradients, camera
) where {I, F}
    fgidx = threadIdx().x + (blockIdx().x - one(I)) * blockDim().x
    stride = blockDim().x * gridDim().x

    sq_h_div_4, sq_w_div_4 = (size(camera) .^ 2) ./ I(4)
    @inbounds while fgidx <= nb_fgaussians
        gg = fgradients[fgidx]
        gidx = get_gidx(gg)
        m2dgx, m2dgy = get_mean2d_grad_sv(gg)
        
        if m2dgx != 0 || m2dgy != 0
            norm_ndc = sqrt(m2dgx^2 * sq_w_div_4 + m2dgy^2 * sq_h_div_4)
            # I.e. sqrt((m2dgx * w/2)^2 + (m2dgy * h/2)^2)
            adc_buffers.accumulated_mean2d_grad_norms_ndc[gidx] += norm_ndc
            adc_buffers.rendering_contribution_counts[gidx] += one(I)
        end
        # Otherwise our Gaussian did not contribute (even though it survived filtering,
        # e.g. because its bbox is empty)

        fgidx += stride
    end
    return
end

"""
    adjust_fgidx_to_gidx_adc_kernel!(
        fgidx_to_gidx::CuDeviceVector{I},
        nb_fgaussians, 
        should_prune, should_densify, should_clone,
        new_gaussian_counts_cs
    ) where I

Adjust `fgidx_to_gidx` to use the newly created Gaussian indices, instead of the old ones.
When an old Gaussian does not correspond to a new one (when pruning), or the new one(s)
should not receive gradients (the second clone, or when splitting) we will indicate this
using a zero index.

Threads correspond to filtered Gaussian indices.
"""
function adjust_fgidx_to_gidx_adc_kernel!(
    fgidx_to_gidx::CuDeviceVector{I},
    nb_fgaussians, 
    should_prune, should_densify, should_clone,
    new_gaussian_counts_cs
) where I
    fgidx = threadIdx().x + (blockIdx().x - one(I)) * blockDim().x
    stride = blockDim().x * gridDim().x

    @inbounds while fgidx <= nb_fgaussians
        gidx = fgidx_to_gidx[fgidx]
        ngidx = zero(I)
        if !should_prune[gidx]
            if should_densify[gidx]
                if should_clone[gidx]
                    ngidx = new_gaussian_counts_cs[gidx] - one(I)  # Use first clone
                end
                # Neither split Gaussian gets gradients
            else  
                ngidx = new_gaussian_counts_cs[gidx]
            end
        end
        fgidx_to_gidx[fgidx] = ngidx

        fgidx += stride
    end
    return
end

"""
    fill_ngidx_to_gidx_kernel!(
        ngidx_to_gidx::CuDeviceVector{I},
        nb_gaussians, 
        should_prune, should_densify, new_gaussian_counts_cs
    ) where I

Fill in the values of `ngidx_to_gidx`: each new Gaussian index will point to the original
Gaussian index which spawned it. For this we need to know which original Gaussians are
pruned or densified. We also need a cumulative sum of the number of new Gaussian each
original Gaussian spawns.

Threads correspond to original Gaussian indices.
"""
function fill_ngidx_to_gidx_kernel!(
    ngidx_to_gidx::CuDeviceVector{I},
    nb_gaussians, 
    should_prune, should_densify, new_gaussian_counts_cs
) where I
    gidx = threadIdx().x + (blockIdx().x - one(I)) * blockDim().x
    stride = blockDim().x * gridDim().x
    @inbounds while gidx <= nb_gaussians
        if !should_prune[gidx]
            last_ngidx = new_gaussian_counts_cs[gidx]
            if should_densify[gidx]
                # Spawns two new Gaussian
                first_ngidx = last_ngidx - one(I)
                ngidx_to_gidx[first_ngidx] = gidx
                ngidx_to_gidx[last_ngidx] = gidx
            else
                # One 'new' Gaussian
                ngidx_to_gidx[last_ngidx] = gidx
            end
        end
        gidx += stride
    end
    return nothing
end


function Base.show(
    io::IO, ::MIME"text/plain", 
    t::Trainer{D, I, F}
) where {D, I, F}
    lines = [
        "3DGS Trainer{$D, $I, $F}",
        "  previous iteration number:  $(get_iteration_nb(t)) \
                                        / $(get_nb_iterations(t.training_options))",
    ]
    if t.cameras_order_idx > 0  # Exclude before first training_step!
        cam_idx = t.cameras_order[t.cameras_order_idx]
        cwi = t.cameras_with_images_vector[cam_idx]
        append!(
            lines, 
            ("  previous camera: ",
             "    index:                    $cam_idx",
             "    filename:                 $(splitpath(cwi.image_loader.path)[end])"
            )  # aligned with e.g. iteration_nb
        )
    end
    append!(lines,
        ("  number of Gaussians:        $(length(t.gaussians_vector))",
         "  spherical harmonics degree: $(get_sh_degree(t.gaussians_vector))")
    )
    print(io, join(lines, '\n'))
end

"""
    maybe_unsafe_free!(t::Trainer)

Call `maybe_unsafe_free!` on (GPU) structures we allocated ourselves in the constructor.
"""
function maybe_unsafe_free!(t::Trainer)
    maybe_unsafe_free!(t.render_buffers)
    maybe_unsafe_free!(t.render_outputs)
    maybe_unsafe_free!(t.gt_image_buffer)
    maybe_unsafe_free!(t.loss_singleton)
    maybe_unsafe_free!(t.dssim_buffers)
    maybe_unsafe_free!(t.rgb_image_grad) 
    maybe_unsafe_free!(t.fgradients)
    maybe_unsafe_free!(t.adc_buffers)
    return nothing
end