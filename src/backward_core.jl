# Include Adam
# (Not sure about densification, pruning)
# Two large kernel (tile rendering, projection)
# Helper functions compute_gradient_X(gradient_Y, X) (something like that)

# Projection backward kernel with Val flag for including Adam update
# (skips writing gradients to global memory)


# We do not support the backward pass for the CPU version, as this would be way too slow
# to use for training in practice.


"""
    backward!(
        fgradients::AbstractFilteredGaussiansGradientsVector{G, I, F},
        rgb_image_grad,
        render_outputs::RenderOutputs{G, <:Any, F},
        gaussians::AbstractTrainableGaussiansVector{G, I, F},
        camera::AbstractCamera{I, F},
        tiling::Tiling{I},
        render_options::RenderOptions{I, F},
        render_buffers::AbstractRenderBuffers{G, I, F}
    ) where {G <: GPU, I, F} -> gradients

GPU version for computing the gradients of the loss with respect to the Gaussian parameters,
starting from the gradients `rgb_image_grad` with respect to the rendered image. Only
Gaussians which survived the filtering in the forward pass receive gradients.

We will zero the necessary `fgradients` within this method.

# Requirements
- The forward [`render!`](@ref) pass has performed first, with the same arguments (except
    `gradients`). In particular, all argument requirements there are met.
- `render_outputs` includes RGB, transmittance and last_contributing_sdfgids
"""
function backward!(
    fgradients::AbstractFilteredGaussiansGradientsVector{G, I, F},
    rgb_image_grad,
    render_outputs::RenderOutputs{G, <:Any, F},
    gaussians::AbstractGaussiansVector{G, I, F},  #TODO: not used (directly)?
    camera::AbstractCamera{I, F},
    tiling::Tiling{I},
    render_options::RenderOptions{I, F},
    render_buffers::AbstractRenderBuffers{G, I, F}
) where {G <: GPU, I, F}

    zero_2d_grads!(fgradients)

    tile_width = get_tile_width(tiling); tile_height = get_tile_height(tiling)
    threads = (tile_width, tile_height)
    blocks = (get_nb_tiles_x(tiling), get_nb_tiles_y(tiling))
    @cuda threads blocks backward_render_tiles_kernel!(
        fgradients,
        rgb_image_grad,
        get_transmittance_image(render_outputs), 
        get_last_contributing_sdfgids(render_outputs),
        render_buffers.sdfgidx_to_fgidx, render_buffers.tiles_end_sdfgidx,
        render_buffers.f2dgv,
        tiling, render_options
    )

    nb_fgaussians = length(render_buffers.f2dgv)
    @run_1d_kernel nb_fgaussians backward_project_fgaussians_kernel!(
        fgradients,
        nb_fgaussians, render_buffers.f2dgv,
        camera,
        render_options
    )
    
    return fgradients
end


"""
    backward_render_tiles_kernel!(
        fgradients,
        rgb_image_grad,
        transmittance_image, last_contributing_sdfgids,
        sdfgidx_to_fgidx, tiles_end_sdfgidx,
        f2dgv::ImmutableFiltered2DGaussiansVector{Device, I, F},
        tiling, render_options,
    ) where {I, F}

Perform the backward pass of the tile rendering kernel [`render_tiles_kernel!`](@ref),
filling the gradients with respect to the filtered 2D Gaussians' parameters in `fgradients`.
`rgb_image_grad` contains the gradient image of the loss with respect to the `rgb_image` and the
`RenderOutputs`.

We use the same thread and block structure as in the forward pass, i.e. in
[`render_tiles_kernel!`](@ref).

!!! As we will add to the 2D Gaussian's gradients in `fgradients` these should be zeroed
before running this kernel!
"""
function backward_render_tiles_kernel!(
    fgradients,
    rgb_image_grad,
    transmittance_image, last_contributing_sdfgids,
    sdfgidx_to_fgidx, tiles_end_sdfgidx,
    f2dgv::ImmutableFiltered2DGaussiansVector{Device, I, F},
    tiling, render_options,
) where {I, F}
    # Similar setup as in the forward method, including loading the same values to shared
    # memory
    px_u_in_tile = threadIdx().x; px_v_in_tile = threadIdx().y 
    tile_idx_u = blockIdx().x; tile_idx_v = blockIdx().y

    TILE_WIDTH = get_tile_width(tiling)
    TILE_HEIGHT = get_tile_height(tiling)
    NB_THREADS_PER_BLOCK = TILE_WIDTH * TILE_HEIGHT
    NB_SHMEM_GAUSSIANS = NB_THREADS_PER_BLOCK

    shmem_gaussian_eval_params = CuStaticSharedArray(F, (6, NB_SHMEM_GAUSSIANS))
    shmem_colors = CuStaticSharedArray(SVector{3, F}, NB_SHMEM_GAUSSIANS)

    # TODO: depth regularization

    thread_idx_within_block = threadIdx().x + (threadIdx().y - one(I)) * TILE_WIDTH
    tile_idx = tile_ids_to_linear_idx(tiling, tile_idx_u, tile_idx_v)

    tile_u_start_cts = (tile_idx_u - one(I)) * TILE_WIDTH
    tile_v_start_cts = (tile_idx_v - one(I)) * TILE_HEIGHT

    px_u = px_u_in_tile + tile_u_start_cts
    px_v = px_v_in_tile + tile_v_start_cts
    img_height, img_width = size(transmittance_image)
    px_within_screen = one(I) <= px_u <= img_width && one(I) <= px_v <= img_height
    
    cts_px_u = px_u - F(0.5)
    cts_px_v = px_v - F(0.5)

    px_transmittance = zero(F)
    # (Always declare px_transmittance, also when we do not use it, cf. px_color_r
    #  in the forward pass)
    px_last_contributing_sdfgidx = zero(I)
    px_rgb_grad = zero(SVector{3, F})
    px_partial_color = zero(MVector{3, F})
    # (same as forward's px_color, but now we care more about intermediate values)

    min_alpha = get_min_alpha(render_options)
    max_alpha = get_max_alpha(render_options)
    background_color = get_background_color(render_options)

    @inbounds if px_within_screen
        px_last_contributing_sdfgidx = last_contributing_sdfgids[px_v, px_u]
        px_transmittance = transmittance_image[px_v, px_u]  
        # Tᵘ_∞ on the online mathematics page
        px_rgb_grad = SVector{3}(view(rgb_image_grad, SOneTo(3), px_v, px_u))  # ∂L/∂Kᵘ
        px_partial_color .= px_transmittance .* background_color  # ψ_∞
        # TODO: compare = (SVector) to .+ (MVector)
    end


    # We now process the Gaussians from back to frond
    tile_first_sdfgidx = if tile_idx == one(I)
        one(I)
    else
        @inbounds tiles_end_sdfgidx[tile_idx - one(I)] + one(I)
    end
  
    @inbounds tile_first_loaded_sdfgidx = tiles_end_sdfgidx[tile_idx] + one(I)
    # TODO: this should really be the maximum + 1 of the last_contributing_sdfgids for the pixels in this tile
    #   --> max-reduce in forward, only save per tile

    @inbounds while tile_first_loaded_sdfgidx > tile_first_sdfgidx
        # Load tile_first_loaded_sdfgidx - NB_THREADS 
        #      : tile_first_loaded_sdfgidx - 1
        next_tile_first_loaded_sdfgidx = tile_first_loaded_sdfgidx - NB_SHMEM_GAUSSIANS
        nb_shmem_gaussians_to_load = if next_tile_first_loaded_sdfgidx >= tile_first_sdfgidx
            NB_SHMEM_GAUSSIANS
        else
            tile_first_loaded_sdfgidx - tile_first_sdfgidx
        end

        
        if thread_idx_within_block <= nb_shmem_gaussians_to_load
            sdfgidx_to_load = (tile_first_loaded_sdfgidx - nb_shmem_gaussians_to_load 
                                - one(I) + thread_idx_within_block)
            # The Gaussians will be loaded by threads [1, nb_shmem_gaussians_to_load]
            # Alternatives:
            # a) sdfgidx_to_load = tile_first_loaded_sdfgidx - thread_idx_within_block
            #   but with + subsequent threads might load in contiguous data 
            # b) sdfgidx_to_load = (tile_first_loaded_sdfgidx - NB_THREADS_PER_BLOCK  
            #       - one(I) + thread_idx_within_block)
            #   but then we always fill the last part of the shared memory, i.e. the
            #   threads/shmem_idx used (with a sdfgidx_to_load >= tile_first_sdfgidx check)
            #   are [NB_THREADS_PER_BLOCK - nb_shmem_gaussians_to_load + 1, 
            #          NB_THREADS_PER_BLOCK]
            fgidx_to_load = sdfgidx_to_fgidx[sdfgidx_to_load]
            f2dg = f2dgv[fgidx_to_load]

            # No precomputations (TODO? we would need different ones)
            x, y = get_mean_sv(f2dg)
            opacity = get_opacity(f2dg)
            ic11, ic21, ic22 = get_inverse_covariance_lower_sv(f2dg)   

            shmem_gaussian_eval_params[1, thread_idx_within_block] = x
            shmem_gaussian_eval_params[2, thread_idx_within_block] = y
            shmem_gaussian_eval_params[3, thread_idx_within_block] = opacity
            shmem_gaussian_eval_params[4, thread_idx_within_block] = ic11
            shmem_gaussian_eval_params[5, thread_idx_within_block] = ic21
            shmem_gaussian_eval_params[6, thread_idx_within_block] = ic22

            shmem_colors[thread_idx_within_block] = get_color_sv(f2dg)
        end

        sync_threads()
        nb_loaded_shmem_gaussians = nb_shmem_gaussians_to_load
        # (A more appropriate name at this point)

        shmem_idx = nb_loaded_shmem_gaussians
        sdfgidx = tile_first_loaded_sdfgidx - one(I)
        # These are always linked as 
        #   shmem_idx = ( sdfgidx - tile_first_loaded_sdfgidx + one(I) 
        #                 + nb_loaded_shmem_gaussians )
        if sdfgidx > px_last_contributing_sdfgidx
            # Skip all Gaussians beyond the last contributing one, simply by starting at
            # this last one
            shmem_idx = ( px_last_contributing_sdfgidx - tile_first_loaded_sdfgidx + one(I) 
                          + nb_loaded_shmem_gaussians )
            # Always >= nb_loaded_shmem_gaussians, but might be <= 0
            sdfgidx = px_last_contributing_sdfgidx
        end

        while px_within_screen && shmem_idx >= one(I)
            # (Fine to go backward in memory, as shared memory is fast)
            x = shmem_gaussian_eval_params[1, shmem_idx]
            y = shmem_gaussian_eval_params[2, shmem_idx]
            opacity = shmem_gaussian_eval_params[3, shmem_idx]
            ic11 = shmem_gaussian_eval_params[4, shmem_idx]
            ic21 = shmem_gaussian_eval_params[5, shmem_idx]
            ic22 = shmem_gaussian_eval_params[6, shmem_idx]
            icov2d = SymmetricSMatrix(SVector{3}(ic11, ic21, ic22))

            px_du = cts_px_u - x
            px_dv = cts_px_v - y
            px_gaussian_eval = exp( 
                F(-0.5) * (ic11 * px_du^2 + ic22 * px_dv^2) - ic21 * px_du * px_dv )
            px_alpha = min(max_alpha, opacity * px_gaussian_eval)
                
            if px_alpha < min_alpha
                # Still skip a Gaussian if the contribution is too small --> no gradient
                shmem_idx -= one(I)
                sdfgidx -= one(I)
                continue
            end
            
            inv_1malpha = inv(one(F) - px_alpha)
            px_transmittance *= inv_1malpha  # Tᵘₜ from Tᵘₜ₊₁ (or Tᵘ_∞)
            px_trans_color = px_transmittance .* shmem_colors[shmem_idx]  
            # Tᵘₜ cₜ
            px_gaussian_color_contribution = px_alpha .* px_trans_color  # wᵘₜ 
            px_partial_color .+= px_gaussian_color_contribution
            # ψᵘₜ from ψᵘₜ₊₁
            
            
            px_alpha_grad = inv_1malpha * dot(px_rgb_grad, px_trans_color .- px_partial_color)
            px_opacity_grad = px_gaussian_eval * px_alpha_grad
            px_dus = SVector{2}(px_du, px_dv)
            px_alpha_grad_alpha = px_alpha * px_alpha_grad
            px_icov2d_grad = (F(-0.5) * px_alpha_grad_alpha) .* congruence_transform(px_dus)
            # 2x2 SymmetricSMatrix

            px_mean2d_grad = px_alpha_grad_alpha .* (transpose(px_dus) * icov2d)
            # transpose(::SVector{2}); uses unrolled_matmul,
            # (Could also directly use 
            #   unrolled_matmul(px_dus, icov2d, 
            #                   Val(true), Val(true), Val(false), Val(false))
            # TODO: Scaled?
            # TODO: Delay matrix-vector product until backward projection kernel?
            px_color_grad = (px_transmittance * px_alpha) .* px_rgb_grad
            
            fgidx = sdfgidx_to_fgidx[sdfgidx]
            fgg = fgradients[fgidx]
            
            cu_atomically_add_to_mean2d_grad(fgg, px_mean2d_grad)
            cu_atomically_add_to_opacity_grad(fgg, px_opacity_grad)
            cu_atomically_add_to_inverse_covariance_lower_grad(fgg, 
                gradient_wrt_symmetric_matrix_to_gradient_wrt_lower_vector(px_icov2d_grad))
            cu_atomically_add_to_color_grad(fgg, px_color_grad)
            # TODO: first reduce on the block or warp level
            
            shmem_idx -= one(I)
            sdfgidx -= one(I)
        end

        tile_first_loaded_sdfgidx = next_tile_first_loaded_sdfgidx
        sync_threads()
    end

    return nothing    
end


"""
    backward_project_fgaussians_kernel!(
        fgradients,
        nb_fgaussians, f2dgv::ImmutableFiltered2DGaussiansVector{Device, I, F},
        camera, 
        render_options
    ) where {I, F}

Perform the backward pass of the projection kernel [`project_fgaussians_kernel!`](@ref),
using the 2D Gaussian gradients in `fgradients`, previously filled by
[`backward_render_tiles_kernel!`](@ref), to set the 3D gradients.

"""
function backward_project_fgaussians_kernel!(
    fgradients,
    nb_fgaussians, f2dgv::ImmutableFiltered2DGaussiansVector{Device, I, F},
    camera, 
    render_options
) where {I, F}

    # TODO: skip (just set to zero) if no associated duplicated Gaussians (nb_overlapped == 0)
    #   (should not be necessary for correctness, as the incoming gradients are zero in this case)

    fgidx = threadIdx().x + (blockIdx().x - one(I)) * blockDim().x
    stride = blockDim().x * gridDim().x

    use_antialiasing = get_use_antialiasing(render_options)
    sh_degree_val = Val(get_sh_degree(render_options)) # TODO: right?

    @inbounds while fgidx <= nb_fgaussians
        f2dg = f2dgv[fgidx]
        g = get_3d_gaussian(f2dg)
        fgg = fgradients[fgidx]

        mean3d = get_mean_sv(g)
        original_opacity = get_opacity(g)  # always without antialiasing correction
        depth = get_depth(mean3d, camera)

        # --- opacity_feature_grad ---
        if use_antialiasing
            opacity = get_opacity(f2dg)  # corrected
        else
            opacity = original_opacity
        end
        opacity_grad = get_opacity_grad(fgg)
        opacity_feature_grad = if !use_antialiasing
            opacity_grad * opacity * (one(F) - opacity)
        else
            opacity_grad * opacity * (one(F) - original_opacity)
        end
        set_opacity_feature_grad!(fgg, opacity_feature_grad)
        # ---

        # --- cov2d_lower_grad ---
        # Required for mean3d_grad, scaling_features_grad and rotation_features_grad
        cov2d_lower_grad = compute_cov2d_lower_grad(
            get_inverse_covariance_lower_grad_sv(fgg),
            get_inverse_covariance_lower_sv(f2dg),
            original_opacity,
            render_options
        )
        # ---

        # --- mean3d_grad ---
        # From mean2d_grad (perspective projection)
        mean2d = get_mean_sv(f2dg)
        if get_boundary_clipping_mode(render_options) == CovarianceBoundaryClamp
            clamped_mean2d = clamp_to_extended_screen(mean2d, render_options)  
        else
            clamped_mean2d = mean2d
        end
        clamped_J = get_linearized_projection_2d(clamped_mean2d, depth, camera)

        mean2d_grad = get_mean2d_grad_sv(fgg)
        Ps = get_projection_matrix_3x4(camera)[SOneTo(3), SOneTo(3)]

        if get_boundary_clipping_mode(render_options) != CovarianceBoundaryClamp
            unclamped_J = clamped_J
            mean3d_grad = transpose(unclamped_J) * mean2d_grad 
            # We transpose as mean2d_grad is a StaticVector{2} and we want mean3d_grad to be
            # a StaticVector{3}, even though mathematically we would have a
            # StaticMatrix{1, 3}. Therefore, we should use
            # (∂L/∂(mean3d))^T = (∂L/∂(mean2d) * ∂(mean2d)/∂(mean3d))^T 
            #                  = (∂L/∂(mean2d) * J)^T 
            #                  = J^T * mean2d_grad
        else
            # We could use the same approach, but that would now require us to compute
            # unclamped_J, requiring a (2 x 3) ⋅ (3 x 3) matrix product.
            # Therefore, we use the formula from the online documentation instead
            x2dg = mean2d_grad[I(1)]; y2dg = mean2d_grad[I(2)]
            mean3d_grad = transpose(Ps) * ScaledSVector(inv(depth), SVector{3}(
                x2dg, y2dg, -(mean2d[I(1)] * x2dg + mean2d[I(2)] * y2dg))
            )
        end

        # From cov2d_grad (linearization of perspective transformation at mean3d)
        mean3d_grad = mean3d_grad .+ Transposed_dσ_dm(
            clamped_mean2d, depth, clamped_J, 
            Ps, lower_matrix_sv(get_covariance_sm(g)),
            # TODO: Symmetric (ssm)
            render_options
        ) * cov2d_lower_grad
        
        # From color (viewpoint dependency)
        viewing_direction = mean3d .- get_position(camera)
        inv_vdnorm = inv(sqrt(sum(t -> t^2, viewing_direction)))
        normed_vd = viewing_direction .* inv_vdnorm
        color = get_color_sv(f2dg)
        color_grad = get_color_grad_sv(fgg)
        sh_coeffs = get_sh_coeffs_view(g)
        
        normalized_viewing_direction_grad = transposed_color_normalized_direction_jacobian(
            color, normed_vd, sh_coeffs, sh_degree_val
        ) * color_grad
        mean3d_grad = mean3d_grad .+ ScaledSMatrix(
                    inv_vdnorm, 
                    normalization_jacobian_times_norm(normed_vd)
                ) * normalized_viewing_direction_grad
        # The normalization Jacobian is symmetric, so already transposed
        
        set_mean3d_grad!(fgg, mean3d_grad)
        # ---
        
        # --- scaling_features_grad ---
        cov3d_lower_grad = Transposed_dσ_dv(clamped_J) * cov2d_lower_grad
        s = get_scaling_sv(g)
        R = get_rotation_sm(g)
        scaling_grad = Transposed_dv_ds(s, R) * cov3d_lower_grad
        scaling_feature_grad = scaling_grad .* s
        set_scaling_features_grad!(fgg, scaling_feature_grad)
        # ---

        # --- rotation_features_grad
        s_sq = s .* s
        R_grad = DL_dR(cov3d_lower_grad, R, s_sq)

        q = get_rotation_features_sv(g)
        inv_qnorm = inv(sqrt(sum(t -> t^2, q)))
        normed_q = q .* inv_qnorm
        normed_q_grad = compute_normed_q_grad(R_grad, normed_q)
        q_grad = ScaledSMatrix(inv_qnorm, 
                               normalization_jacobian_times_norm(normed_q)
                ) * normed_q_grad
        set_rotation_features_grad!(fgg, q_grad)
        # ---

        # --- sh_coeffs_grad ---
        compute_sh_coeffs_grad!(
            get_sh_coeffs_grad_view(fgg), color_grad, color, normed_vd, sh_degree_val
        )
        # ---

        fgidx += stride
    end
    return nothing
end

"""
    compute_cov2d_lower_grad(
        icov2d_lower_grad, icov2d_lower, original_opacity, render_options
    )

Compute the gradient with respect to the lower half of the 2D covariance matrix,
starting from the gradient with respect to the lower half `icov2d_lower` of the inverse 2D
covariance matrix, and the lower half `icov2d_lower` of this inverse matrix itself.
When we use antialiasing we also need the `original_opacity` (before antialiasing)
and dilations in `render_options`.
"""
@inline function compute_cov2d_lower_grad(
    icov2d_lower_grad, icov2d_lower, original_opacity, render_options
)
    a, b, d = icov2d_lower
    a_sq = a^2; ab = a * b; b_sq = b^2; bd = b * d; d_sq = d^2; ad = a * d
    ab2 = ab + ab; bd2 = bd + bd
    cov2d_lower_grad = -(SMatrix{3, 3}(
        a_sq,    ab2,          b^2,
        ab,      b_sq + ad,    bd,
        b_sq,    bd2,          d_sq
        ) * icov2d_lower_grad)
    # TODO: scaled (-)?
    # Note that this matrix is listed in column-major order, so that it is the transposition
    # of ∂(icov2d_lower)/∂(cov2d_lower). Because we use icov2d_lower_grad as a (column)
    # vector instead of a row vector, we need transpositions
    if get_use_antialiasing(render_options)
        # Contribution from opacity-correction
        Dx = get_dilation_x(render_options)
        Dy = get_dilation_y(render_options)
        DxDy_det = (ad - b_sq) * Dx * Dy
        b2 = b + b
        cov2d_lower_grad = cov2d_lower_grad .+ original_opacity .* SA[
            a_sq * Dx + b_sq * Dx - a * DxDy_det,
            b2 * (a * Dx + d * Dy - DxDy_det),
            b_sq * Dx + d^2 * Dy - d * DxDy_det
        ]
    end
    return cov2d_lower_grad
end

# The idea with the SpecialIndexingMatrices below is that we have a complicated getindex,
# but which (mostly) gets evaluted at compile-time
"""
    Transposed_dσ_dv{
        T, J_type <: StaticMatrix{2, 3, T}
    } <: SpecialIndexingSMatrix{3, 6, T}

Struct symbolically representing the 6 × 3 transpose of the 3 × 6 Jacobian `∂σ_∂v`, where
`σ` is the lower half of the 2D covariance matrix `Σ` (after dilation) and `v` the lower
half of the 3D covariance matrix `V`. The entries are fully determined by the Jacobian `J`
of the perspective projection at the 3D Gaussian mean.

---

    Transposed_dσ_dv(J)

Construct the `Transposed_dσ_dv` corresponding to perspective projection Jacobian `J`.
"""
struct Transposed_dσ_dv{
    T, J_type <: StaticMatrix{2, 3, T}
} <: SpecialIndexingSMatrix{6, 3, T}
    J::J_type
end

"""
    adjust_indexing_expression(
        split_expr, ::Type{<:Transposed_dσ_dv{T, J_type}}, t_dσ_dv_symb
    ) where {T, J_type}

Rewrite the split `expr` containing `Transposed_dσ_dv` indexing to use the perspective
projection Jacobian `J`. In the input expression the `Transposed_dσ_dv` should be
represented as `t_dσ_dv_symb`.

!!! The indexing expressions in `expr_body` (`split_expr[2]`) must only contain 
literals (e.g. [1], not [i]).
"""
function adjust_indexing_expression(
    split_expr, ::Type{<:Transposed_dσ_dv{T, J_type}}, t_dσ_dv_symb
) where {T, J_type}
    expr_head, expr_body = split_expr
    J_symb = gensym(Symbol(t_dσ_dv_symb, :_J))
    
    expr_head = join_exprs((
        expr_head, 
        :($J_symb = $t_dσ_dv_symb.J)
    ))

    indices_3_to_2x2 = create_self_adjoint_or_symmetric_smatrix_indices_1D_to_2D(2)
    indices_6_to_3x3 = create_self_adjoint_or_symmetric_smatrix_indices_1D_to_2D(3)
    expr_body = MacroTools.postwalk(expr_body) do ex
        if @capture(ex, a_[I__]) && a === t_dσ_dv_symb
            j, i = I  # transpose
            p, q = Tuple(indices_3_to_2x2[i])
            r, s = Tuple(indices_6_to_3x3[j])
            # (∂σ_∂v)[i, j] == ∂(σ[i])/∂(v[j])
            #             == ∂(Σ[p, q])/∂(V[r, s]) [ + ∂(Σ[p, q])/∂(V[s, r]) ]
            #             == J[p, r] * J[q, s]     [ + J[p, s] * J[q, r] ]
            # where the last term is only present if r != s
            new_ex = :($J_symb[$p, $r] * $J_symb[$q, $s])
            if r != s
                new_ex = :($new_ex + $J_symb[$p, $s] * $J_symb[$q, $r])
            end
            return new_ex
        else
            return ex
        end
    end
    return adjust_indexing_expression((expr_head, expr_body), J_type, J_symb)
end



"""
    Transposed_dv_ds(s, R)

Return a symbolic representation of the 3 × 6 transposition of the 6 × 3 Jacobian `∂v_∂s`,
where `v` is the lower half of the 3D covariance matrix `V` and `s` is the (decoded) scaling
vector. Apart from the latter we also need the 3 × 3 rotation matrix `R` for our Gaussian.
"""
@inline Transposed_dv_ds(s, R) = ScaledSMatrix(eltype(s)(2), HalfTransposed_dv_ds(s, R))
# Capital letter for consistency with (the constructor) Transposed_dσ_dv

"""
    HalfTransposed_dv_ds{
        T, s_type <: StaticVector{3, T}, R_type <: StaticMatrix{3, 3, T}
    } <: SpecialIndexingSMatrix{6, 3, T}

Struct containing the underlying representation used in [`Transposed_dv_ds`](@ref).

---

    HalfTransposed_dv_ds(s, R)

Construct the `HalfTransposed_dv_ds` from the scaling vector `s` and rotation matrix `R`.
"""
struct HalfTransposed_dv_ds{
    T, s_type <: StaticVector{3, T}, R_type <: StaticMatrix{3, 3, T}
} <: SpecialIndexingSMatrix{3, 6, T}
    s::s_type
    R::R_type
end

"""
    adjust_indexing_expression(
        split_expr, ::Type{<:HalfTransposed_dv_ds{T, s_type, R_type}}, half_t_dv_ds_symb
    ) where {T, s_type, R_type}

Rewrite the split `expr` containing `HalfTransposed_dv_ds` indexing to use the scaling
vector `s` and rotation matrix `R`. In the input expression the `HalfTransposed_dv_ds`
should be represented as `half_t_dv_ds_symb`.

!!! The indexing expressions in `expr_body` (`split_expr[2]`) must only contain 
literals (e.g. [1], not [i]).
"""
function adjust_indexing_expression(
    split_expr, ::Type{<:HalfTransposed_dv_ds{T, s_type, R_type}}, half_t_dv_ds_symb
) where {T, s_type, R_type}
    expr_head, expr_body = split_expr
    s_symb = gensym(Symbol(half_t_dv_ds_symb, :_s))
    R_symb = gensym(Symbol(half_t_dv_ds_symb, :_R))
    
    expr_head = join_exprs((
        expr_head, 
        :($s_symb = $half_t_dv_ds_symb.s),
        :($R_symb = $half_t_dv_ds_symb.R)
    ))

    indices_6_to_3x3 = create_self_adjoint_or_symmetric_smatrix_indices_1D_to_2D(3)
    expr_body = MacroTools.postwalk(expr_body) do ex
        if @capture(ex, a_[I__]) && a === half_t_dv_ds_symb
            j, i = I  # transpose
            p, q = Tuple(indices_6_to_3x3[i])
            # (∂v_∂s)[i, j] == ∂(v[i])/∂(s[j]) == ∂(V[p, q])/∂(s[j])
            #               == 2 * s[j] * R[p, j] * R[q, j] 
            return :($s_symb[$j] * $R_symb[$p, $j] * $R_symb[$q, $j])
        else
            return ex
        end
    end
    return adjust_indexing_expression(
        (expr_head, expr_body), 
        (s_type, s_symb), (R_type, R_symb))
end

"""
    DL_dR(dL_dv, R, s_sq)

Construct a 3 × 3 `SMatrix` containing the gradient of the loss `L` with respect to the
3 × 3 rotation matrix `R`. To create it, we also need the gradient `dL_dv` with respect to
the lower part `v` of the 3D covariance matrix, and the squared scaling vector `s_sq`.
"""
@inline DL_dR(dL_dv, R, s_sq) = SMatrix(_DL_dR(dL_dv, R, s_sq))
# Materialize the _DL_dR, so that we only compute its entries once

"""
    _DL_dR{
        T, 
        dL_dv_type <: StaticVector{6, T},
        R_type <: StaticMatrix{3, 3, T}, 
        s_sq_type <: StaticVector{3, T}
    } <: SpecialIndexingSMatrix{3, 3, T}

Struct containing the underlying representation used in [`DL_dR`](@ref). Since it is better
to materialize it into a `SMatrix`, we dissuade direct usageof the struct via the `_`
prefix.

---

    _DL_dR(dL_dv, R, s_sq)

Construct the `_DL_dR` from the 3D covariance lower vector gradient `dL_dv`, rotation matrix
`R` and squared scaling vector `s_sq`.
"""
struct _DL_dR{
    T, 
    dL_dv_type <: StaticVector{6, T},
    R_type <: StaticMatrix{3, 3, T}, 
    s_sq_type <: StaticVector{3, T}
} <: SpecialIndexingSMatrix{3, 3, T}
    dL_dv::dL_dv_type
    R::R_type
    s_sq::s_sq_type
end

"""
    adjust_indexing_expression(
        split_expr, ::Type{<:_DL_dR{T, dL_dv_type, R_type, s_sq_type}}, dL_dR_symb
    ) where {T, s_type, R_type}

Rewrite the split `expr` containing `_DL_dR` indexing to use the lower 3D covariance vector
gradient `dL_dv`, rotation matrix `R` and squared scaling vector `s_sq`. In the input
expression the `_DL_dR` should be represented as `dL_dR_symb`.

!!! The indexing expressions in `expr_body` (`split_expr[2]`) must only contain 
literals (e.g. [1], not [i]).
"""
function adjust_indexing_expression(
    split_expr, ::Type{<:_DL_dR{T, dL_dv_type, R_type, s_sq_type}}, dL_dR_symb
) where {T, dL_dv_type, R_type, s_sq_type}
    expr_head, expr_body = split_expr
    dL_dv_symb = gensym(Symbol(dL_dR_symb, :_dL_dv))
    R_symb = gensym(Symbol(dL_dR_symb, :_R))
    s_sq_symb = gensym(Symbol(dL_dR_symb, :_s_sq))
    
    expr_head = join_exprs((
        expr_head, 
        :($dL_dv_symb = $dL_dR_symb.dL_dv),
        :($R_symb = $dL_dR_symb.R),
        :($s_sq_symb = $dL_dR_symb.s_sq)
    ))

    indices_3x3_to_6 = create_self_adjoint_or_symmetric_smatrix_indices_2D_to_1D(3)
    # ϕ in the online documentation
    expr_body = MacroTools.postwalk(expr_body) do ex
        if @capture(ex, a_[I__]) && a === dL_dR_symb
            k, l = I
            p = indices_3x3_to_6[k, k]
            new_ex = :($dL_dv_symb[$p] * $R_symb[$k, $l])
            for i = 1:3
                p = indices_3x3_to_6[i, k]
                new_ex = :($new_ex + $dL_dv_symb[$p] * $R_symb[$i, $l])
            end
            new_ex = :($s_sq_symb[$l] * $new_ex)
            return new_ex
        else
            return ex
        end
    end
    return adjust_indexing_expression(
        (expr_head, expr_body), 
        (dL_dv_type, dL_dv_symb), (R_type, R_symb), (s_sq_type, s_sq_symb))
end

"""
    compute_normed_q_grad(R_grad, normed_q)

Compute the gradient with respect to the normalized quaternion `normed_q`, starting from
the gradient `R_grad` with respect to the rotation matrix.
"""
@inline function compute_normed_q_grad(R_grad, normed_q)
    F = eltype(R_grad)
    r, x, y, z = normed_q

    m2x = -(x + x)
    m2y = -(y + y)
    m2z = -(z + z)
    return ScaledSVector(
        F(2),
        SVector{4}(
            dot(
                SA[  # 1/2 .* ∂R/∂r (not transposed)
                       zero(F)         -z          y
                             z    zero(F)         -x
                            -y          x    zero(F)], 
                R_grad
            ),
            dot(
                SA[  # 1/2 .* ∂R/∂x
                       zero(F)      y       z
                             y    m2x      -r
                             z      r     m2x],  
                R_grad
            ),
            dot(
                SA[  # 1/2 .* ∂R/∂y
                       m2y          x      r
                         x    zero(F)      z
                        -r          z    m2y],  
                R_grad
            ),
            dot(
                SA[  # 1/2 .* ∂R/∂z
                       m2z     -r          x
                         r    m2z          y
                         x      y    zero(F)],  
                R_grad
            ),
        )
    )
end


"""
    Transposed_dσ_dm(mean2dc, depth::F, Jc, Ps, v, render_options) where F

Construct a `StaticMatrix{3, 3}` containing the transposition of the Jacobian of the lower
half `σ` of the 2D covariance matrix `Σ` with respect to the 3D mean `m`. To create it, we
need the (possibly clamped) 2D mean `mean2dc`, 3D Gaussian `depth`, (possibly clamped
version of the) Jacobian `Jc` of the projection transformation, the `[1:3, 1:3]` submatrix 
`Ps` of projection matrix itself, and the 3D covariance matrix lower half `v`. To determine
from the `mean2dc` whether we clamped, we need the boundary values stored in
`render_options`. This parameter also allows us to check whether clamping is even allowed a
priori. If not, the formulas become slightly simpler.
"""
@inline function Transposed_dσ_dm(mean2dc, depth::F, Jc, Ps, v, render_options) where F
    mean2dc2 = mean2dc .+ mean2dc  # 2μ⁽ᶜ⁾
    clamping_possible = get_boundary_clipping_mode(render_options) === 
        CovarianceBoundaryClamp
    deltas = if clamping_possible
        @inbounds (
            render_options.clip_x_min < mean2dc[1] <= render_options.clip_x_max,
            render_options.clip_y_min < mean2dc[2] <= render_options.clip_y_max
        )
    else
        (true, true)
        # Hopefully, since clamping_possible is statically known, the compiler can simplify
        # deltas[1] * x to x in this case. (If not, this presumably does not matter much
        # for performance) 
    end
    dJc_dp_mrsq = @inbounds SMatrixWithZeros(
        SVector{12, F}(
            deltas[1] * Ps[3, 1],               # [1, 1, 1] <-> [1, 1]
            deltas[1] * Ps[3, 2],               # [1, 2, 1] <-> [3, 1]
            deltas[1] * Ps[3, 3],               # [1, 3, 1] <-> [5, 1]
            deltas[2] * Ps[3, 1],               # [2, 1, 2] <-> [2, 2]
            deltas[2] * Ps[3, 2],               # [2, 2, 2] <-> [4, 2]
            deltas[2] * Ps[3, 3],               # [2, 3, 2] <-> [6, 2]
            Ps[1, 1] - mean2dc2[1] * Ps[3, 1],  # [1, 1, 3] <-> [1, 3]
            Ps[2, 1] - mean2dc2[2] * Ps[3, 1],  # [2, 1, 3] <-> [2, 3]
            Ps[1, 2] - mean2dc2[1] * Ps[3, 2],  # [1, 2, 3] <-> [3, 3]
            Ps[2, 2] - mean2dc2[2] * Ps[3, 2],  # [1, 2, 3] <-> [4, 3]
            Ps[1, 3] - mean2dc2[1] * Ps[3, 3],  # [1, 3, 3] <-> [5, 3]
            Ps[2, 3] - mean2dc2[2] * Ps[3, 3],  # [1, 3, 3] <-> [6, 3]
        ),
        Tuple{6, 3},
        Val(((2, 1), (4, 1), (6, 1), (1, 2), (3, 2), (5, 2)))
    )  # ∂𝐉⁽ᶜ⁾/∂𝒑 from the online documentation, except for the -1/r^2 prefactor
    dJc_dm_mrsq = SArray{Tuple{2, 3, 3}}(dJc_dp_mrsq * Ps)
    return ScaledSMatrix(
        -inv(depth ^ 2),
        SMatrix(
            Transposed_dσ_dm_mrsq(dJc_dm_mrsq, Jc, v)
        ) # Materialize the Transposed_dσ_dm_mrsq
    )
end

"""
    Transposed_dσ_dm_mrsq{
        T, 
        dJc_dm_mrsq_type <: StaticArray{Tuple{2, 3, 3}, T},
        Jc_type <: StaticMatrix{2, 3, T}, 
        v_type <: StaticVector{3, T}
    } <: SpecialIndexingSMatrix{3, 3, T}
        dJc_dm_mrsq::dJc_dm_mrsq_type
        Jc::Jc_type
        v::v_type
    end

Struct containing the underlying representation used in [`Transposed_dσ_dm`](@ref). It
stores the same matrix up to division by the negative squared depth.

---

    Transposed_dσ_dm_mrsq(dJc_dm_mrsq, Jc, v)

Construct the `Transposed_dσ_dm_mrsq` from the negative square depth scaled second order
Jacobian of the clamped projection mapping `dJc_dm_mrsq = -1/r^2 * dJc_dm`, the clamped
first order Jacobian `Jc`, and the lower half `v` of the 3D covariance matrix.
"""
struct Transposed_dσ_dm_mrsq{
    T, 
    dJc_dm_mrsq_type <: StaticArray{Tuple{2, 3, 3}, T},
    Jc_type <: StaticMatrix{2, 3, T}, 
    v_type <: StaticVector{6, T}
} <: SpecialIndexingSMatrix{3, 3, T}
    dJc_dm_mrsq::dJc_dm_mrsq_type
    Jc::Jc_type
    v::v_type
end

"""
    adjust_indexing_expression(
        split_expr, 
        ::Type{<:Transposed_dσ_dm_mrsq{T, dJc_dm_mrsq_type, Jc_type, v_type}}, 
        dσ_dm_mrsq_symb
    ) where {T, dJc_dm_mrsq_type, Jc_type, v_type}

Rewrite the split `expr` containing `Transposed_dσ_dm_mrsq` indexing to use its field. In
the input expression the `Transposed_dσ_dm_mrsq` should be represented as
`t_dσ_dm_mrsq_symb`.

!!! The indexing expressions in `expr_body` (`split_expr[2]`) must only contain 
literals (e.g. [1], not [i]).
"""
function adjust_indexing_expression(
    split_expr, 
    ::Type{<:Transposed_dσ_dm_mrsq{T, dJc_dm_mrsq_type, Jc_type, v_type}}, 
    t_dσ_dm_mrsq_symb
) where {T, dJc_dm_mrsq_type, Jc_type, v_type}

    expr_head, expr_body = split_expr
    dJc_dm_mrsq_symb = gensym(Symbol(t_dσ_dm_mrsq_symb, :_dJc_dm_mrsq))
    Jc_symb = gensym(Symbol(t_dσ_dm_mrsq_symb, :_Jc))
    v_symb = gensym(Symbol(t_dσ_dm_mrsq_symb, :_v))
    
    expr_head = join_exprs((
        expr_head, 
        :($dJc_dm_mrsq_symb = $t_dσ_dm_mrsq_symb.dJc_dm_mrsq),
        :($Jc_symb = $t_dσ_dm_mrsq_symb.Jc),
        :($v_symb = $t_dσ_dm_mrsq_symb.v)
    ))

    indices_3_to_2x2 = create_self_adjoint_or_symmetric_smatrix_indices_1D_to_2D(2)
    indices_3x3_to_6 = create_self_adjoint_or_symmetric_smatrix_indices_2D_to_1D(3)
    expr_body = MacroTools.postwalk(expr_body) do ex
        if @capture(ex, a_[I__]) && a === t_dσ_dm_mrsq_symb
            k, p = I  # transpose
            i, j = Tuple(indices_3_to_2x2[p])
            # i, j, k now have the same meaning as in the formula for ∂Σᵢⱼ/∂mₖ in the online
            # documentation
            # ∂σₚ/∂mₖ == ∂Σᵢⱼ/∂mₖ 
            new_ex = :($(zero(T)))
            for α = 1:3, β = 1:3
                q = indices_3x3_to_6[α, β]
                new_ex = :($new_ex + 
                    $v_symb[$q] * ($dJc_dm_mrsq_symb[$i, $α, $k] * $Jc_symb[$j, $β]
                                   + $Jc_symb[$i, $α] * $dJc_dm_mrsq_symb[$j, $β, $k])
                )
            end
            return new_ex
        else
            return ex
        end
    end
    return adjust_indexing_expression(
        (expr_head, expr_body), 
        (dJc_dm_mrsq_type, dJc_dm_mrsq_symb), (Jc_type, Jc_symb), (v_type, v_symb))
end


"""
    compute_sh_coeffs_grad!(
        shg, color_grad, color, normed_vd, ::Val{SH_DEGREE}
    ) where SH_DEGREE
    
Compute the gradient with respect to the spherical harmonics coefficients, and write them
to the output gradients matrix (or view thereof) `shg`. For this we need the gradient
`color_grad` with respect to the Gaussian's color `color` for the normalized viewing
direction `normed_vd`. For efficiency reasons we also need the spherical harmonics degree,
as a `Val`.
"""
@inline function compute_sh_coeffs_grad!(
    shg, color_grad, color, normed_vd, ::Val{SH_DEGREE}
) where SH_DEGREE
    I = eltype(SH_DEGREE)
    F = eltype(shg)

    MAX_SH_ORDER = (SH_DEGREE + one(I)) ^ 2
    l = one(I)
    @inbounds while l <= 3
        if color[l] >= 0
            color_grad_l = color_grad[l]
            k = one(I)
            while k <= MAX_SH_ORDER
                shg[k, l] = color_grad_l * evaluate_sh_basis_function(
                    k, normed_vd, I, F)
                k += one(I)
            end
        else
            k = one(I)
            while k <= MAX_SH_ORDER
                shg[k, l] = zero(F)
                k += one(I)
            end
        end
        l += one(I)
    end
    # TODO: looks like 3 x h x n memory order would be better than h x 3 x n
    #   also for forward, as we can reuse evaluate_sh_basis_function
end