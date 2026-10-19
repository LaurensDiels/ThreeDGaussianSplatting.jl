"""
    render!(
        outputs::RenderOutputs{CPU, <:Any, F},
        gaussians::AbstractGaussiansVector{CPU, I, F},
        camera::AbstractCamera{I, F},
        _,
        options::RenderOptions{I, F},
        buffers::AbstractRenderBuffers{CPU, I, F}
    ) where {I, F} -> outputs

Single-threaded CPU-only baseline version for rendering the 3D `gaussians` into the
`render_outputs`, for the provided `camera`. For the actual rendering we loop over the
filtered 2D Gaussians and loop over precisely those pixels they affect.
    
Does not use tiles or duplicated Gaussians. Therefore, a `RenderOutputs`'s
`last_contributing_sdfgids` will be ignored. For consistency with the GPU version we do
provide an unused tiling parameter slot.

# Requirements
- `outputs` and `buffers` must be compatible in size with `gaussians` and
`camera`. 
- `outputs` must include `transmittance_image`.
- `buffers`'s Filtered2DGaussiansVector is mutable.
"""
function render!(
    outputs::RenderOutputs{CPU, <:Any, F},
    gaussians::AbstractGaussiansVector{CPU, I, F},
    camera::AbstractCamera{I, F},
    _,
    options::RenderOptions{I, F},
    buffers::AbstractRenderBuffers{CPU, I, F}
) where {I, F}

    # Initialize
    f2dgv = buffers.f2dgv
    empty!(f2dgv)

    rgb_image = get_rgb_image(outputs)
    fill!(rgb_image, zero(F))
    has_transmittance_image(outputs) || throw(ArgumentError(
        "The `RenderOutputs` has to include the transmittance image!"))
    transmittance_image = get_transmittance_image(outputs)
    fill!(transmittance_image, one(F))
    if has_depth_image(outputs) 
        fill!(get_depth_image(outputs), zero(F))
    end

    # Project and filter
    min_alpha = get_min_alpha(options)
    use_antialiasing = get_use_antialiasing(options)
    aa_min_factor = get_antialiasing_min_factor(options)
    for g in gaussians
        # Project mean and filter
        mean3d = get_mean_sv(g)
        opacity = get_opacity(g)
        opacity >= min_alpha || continue

        mean2d, depth = project_with_depth(mean3d, camera)
        check_depth(depth, camera) || continue
        if get_boundary_clipping_mode(options) == BoundaryFilter
            lies_on_extended_screen(mean2d, options) || continue
        end

        f2dgv.length += 1
        @inbounds f2dg = f2dgv[end]
        set_mean!(f2dg, mean2d)
        set_opacity!(f2dg, opacity)
        set_depth!(f2dg, depth)

        # Project covariance matrix
        if get_boundary_clipping_mode(options) == CovarianceBoundaryClamp
            clamped_mean2d = clamp_to_extended_screen(mean2d, options)
            J = get_linearized_projection_2d(clamped_mean2d, depth, camera)
        else
            J = get_linearized_projection_2d(mean2d, depth, camera)
        end
        cov2d_11, cov2d_21, cov2d_22 = project_covariance_lower_sv(g, J)
        if use_antialiasing
            undilated_cov2d_det = cov2d_11 * cov2d_22 - cov2d_21 * cov2d_21
        end
        cov2d_11 += get_dilation_x(options)
        cov2d_22 += get_dilation_y(options)
        icov2d_det = inv(cov2d_11 * cov2d_22 - cov2d_21^2)
        if use_antialiasing
            opacity *= max(aa_min_factor, sqrt(undilated_cov2d_det * icov2d_det))
        end
        icov2d_11 = cov2d_22 * icov2d_det
        icov2d_21 = -cov2d_21 * icov2d_det
        icov2d_22 = cov2d_11 * icov2d_det
        set_inverse_covariance_lower!(f2dg, icov2d_11, icov2d_21, icov2d_22)
        # TODO: class Symmetric{N}, with matrix multiplication and inversion definitions
        #   (for N = 2, 3)
        # --> SymmetricSMatrix, but still update code to use it everywhere

        viewing_direction = normalize(mean3d - get_position(camera))
        color = sh_coeffs_to_color(
            Val(get_sh_degree(options)), 
            get_sh_coeffs_view(g), 
            viewing_direction)
        set_color!(f2dg, color)
    end

    h, w = size(camera)
    max_alpha = get_max_alpha(options)
    min_transmittance = get_min_transmittance(options)

    # Sort
    sp = sortperm(@view f2dgv.depths[1:length(f2dgv)])

    # Render 2D Gaussians
    for fgidx in sp
        @inbounds f2dg = f2dgv[fgidx]

        # cf. get_bbox_cts_px and get_bbox_discrete_px,
        idet = get_inverse_covariance_determinant(f2dg)
        gamma = 2 * log(get_opacity(f2dg) / min_alpha) / idet
        x, y = get_mean_sv(f2dg)  # continuous pixel coordinates
        _, q, r = get_inverse_covariance_lower_sv(f2dg)

        # But now explicitly solve the quadratic equation
        #   opacity * Gaussian((u_cts, v_cts); (x, y, cov) = min_alpha
        # for (continuous) v_cts as a funtion of u_cts:
        # v_cts(u_cts) = 
        #   (-q * (u_cts - x) ± sqrt(idet * (r * gamma - (u_cts - x)^2))) / r 

        x_radius = @inbounds sqrt(r * gamma)
        
        u_start = max(one(I), ceil(I, x - x_radius + F(0.5)))
        u_end = min(w, round(I, x + x_radius))
        for u = u_start : u_end  # discrete pixel coordinates
            u_cts = u - F(0.5)
            du = u_cts - x         
            discriminant_div_4idet = r * gamma - du^2  # D / (4 * idet)
            if discriminant_div_4idet <= 0
                # < 0 is mathematically impossible, might occur due to low numerical
                # precision
                sqrt_in_v = zero(F)
            else
                sqrt_in_v = sqrt(idet * discriminant_div_4idet)  # sqrt(discriminant) / 2
            end
            q_du = q * du
            ir = inv(r)
            cts_top = y + (-q_du - sqrt_in_v) * ir
            cts_bottom = y + (-q_du + sqrt_in_v) * ir
            v_start = max(one(I), ceil( I, cts_top    + F(0.5)))
            v_stop =  min(h,      round(I, cts_bottom))
            for v = v_start : v_stop
                alpha = min(max_alpha, get_value_discrete_px(f2dg, u, v))
                # alpha >= min_alpha because of uv-loop
                current_T = transmittance_image[v, u]
                next_T = current_T * (one(F) - alpha)
                next_T < min_transmittance && continue
                
                weight = alpha * current_T
                @view(rgb_image[:, v, u]) .+= weight .* get_color_sv(f2dg)
                transmittance_image[v, u] = next_T
                if has_depth_image(outputs)  # Check gets compiled away
                    get_depth_image(outputs)[v, u] += weight * get_depth(f2dg)
                end
                # last_contributing_sdfgids does not make sense as we don't duplicate
                # Gaussians
            end
        end

    end

    rgb_image .+= reshape(transmittance_image, (1, h, w)) .* 
                    reshape(get_background_color(options), (3, 1, 1))
    return outputs
end


"""
    render!(
        outputs::RenderOutputs{G, <:Any, F},
        gaussians::AbstractGaussiansVector{G, I, F},
        camera::AbstractCamera{I, F},
        tiling::Tiling{I},
        options::RenderOptions{I, F},
        buffers::AbstractRenderBuffers{G, I, F}
    ) where {G <: GPU, I, F} -> outputs

GPU version for rendering the 3D `gaussians` into the `outputs`, for the provided `camera`.

# Requirements
- `outputs` and `buffers` must be compatible in size with `gaussians` and `camera`. 
- `buffers`'s Filtered2DGaussiansVector is mutable and must point to `gaussians`.
"""
function render!(
    outputs::RenderOutputs{G, <:Any, F},
    gaussians::AbstractGaussiansVector{G, I, F},
    camera::AbstractCamera{I, F},
    tiling::Tiling{I},
    options::RenderOptions{I, F},
    buffers::AbstractRenderBuffers{G, I, F}
) where {G <: GPU, I, F}

    nb_gaussians = length(gaussians)
    gaussians_sh_degree = get_sh_degree(gaussians)
    options_sh_degree = get_sh_degree(options)
    use_all_sh_val = Val(gaussians_sh_degree == options_sh_degree)
    # If we use them all, it is faster to use get_sh_coeffs_view(::AbstractGaussian),
    # rather than the variant were we supply a max_sh_degree (also when this is equal
    # to gaussians_sh_degree).

    ## Filter
    # (basically a non-allocating findall to get fgidx_to_gidx)
    # E.g. 
    # gaussian indices:     1 2 3 4 5 6 7
    # keep_gaussians:       0 1 1 0 0 0 1
    # keep_gaussians_cs:    0 1 2 2 2 2 3   -> 3 kept Gaussians (= count(keep_gaussians))
    # ------------------------------------
    # fgaussian indices:      1 2       3   from keep_gaussians_cs
    # fgidx_to_gidx:          2 3       7   from gaussian indices
    @run_1d_kernel nb_gaussians fill_keep_gaussians_kernel!(
        buffers.keep_gaussians, 
        nb_gaussians, gaussians, 
        camera, options
    )
    # We don't have to include nb_gaussians here. But for consistency with other calls
    # we will always include it or its (duplicated) filtered variant.
    # More generally we will typically explicitly include the actual usable length 
    # of the involved arrays.
    accumulate!(+, buffers.keep_gaussians_cs, buffers.keep_gaussians)
    nb_fgaussians = @allowscalar @inbounds buffers.keep_gaussians_cs[nb_gaussians]
    set_length!(buffers.f2dgv, nb_fgaussians) 
    
    if nb_fgaussians == 0
        # When no Gaussians are in view
        render_void!(outputs, options)
        return outputs
    end
    @run_1d_kernel nb_gaussians fill_fgidx_to_gidx_kernel!(
        buffers.f2dgv.fgidx_to_gidx, 
        nb_gaussians, 
        buffers.keep_gaussians, buffers.keep_gaussians_cs
    )


    ## Project the filtered 3D Gaussians
    # Fully sets buffers.f2dgv
    @run_1d_kernel nb_fgaussians project_fgaussians_kernel!(
        buffers.f2dgv, buffers.f2dgs_nb_overlapped_tiles,
        nb_fgaussians, 
        camera,
        tiling, options, use_all_sh_val
    )


    ## Duplicate filtered 2D Gaussians (or indices thereof)
    # Each filtered Gaussian contributes one for every tile it intersects
    accumulate!(+, 
        view(buffers.f2dgs_nb_overlapped_tiles_cs, 1:nb_fgaussians),
        view(buffers.f2dgs_nb_overlapped_tiles, 1:nb_fgaussians))
    nb_dfgaussians = 
        @allowscalar @inbounds buffers.f2dgs_nb_overlapped_tiles_cs[nb_fgaussians]
    set_nb_dfgaussians!(buffers, nb_dfgaussians)

    if nb_dfgaussians == 0
        # Could happen as we only filter on depth above
        render_void!(outputs, options)
        return outputs
    end

    ## Sort
    # Construct the sorting codes
    # E.g.
    # fgaussian indices:            1, 2, 3, 4, 5
    # f2dgs_nb_overlapped_tiles:    1, 2, 0, 2, 1
    # f2dgs_nb_overlapped_tiles_cs: 1, 3, 3, 5, 6
    # -------------------------------------------
    # dfgaussian indices:           1, 2, 3, 4, 5, 6
    # dfgidx_to_fgidx:              1, 2, 2, 4, 4, 5
    #   e.g. fgidx 4 is repeated 2 times (from f2dgs_nb_overlapped_tiles[4]),
    #        namely at dfgids f2dgs_nb_overlapped_tiles_cs[3] + 1 :
    #                            f2dgs_nb_overlapped_tiles_cs[4]
    # For df2dgs_codes we put at some dfgidx range for some fgidx a code for 
    # the each tile index and the 2D filtered Gaussian's depth.
    @run_1d_kernel nb_fgaussians fill_df2dgs_codes_and_dfgidx_to_fgidx_kernel!(
        buffers.df2dgs_codes, buffers.dfgidx_to_fgidx,
        nb_fgaussians, buffers.f2dgv, buffers.f2dgs_nb_overlapped_tiles_cs,
        tiling
    )

    # Actual sort
    sort_duplicated_filtered_2d_gaussians!(buffers, tiling)

    # Determine which sorted dfgaussians belong to which tile
    @run_1d_kernel nb_dfgaussians fill_tiles_end_sdfgidx_kernel!(
        buffers.tiles_end_sdfgidx, 
        nb_dfgaussians, buffers.sdf2dgs_codes, get_nb_tiles(tiling)
    )

    ## Render the pixels in each tile using the duplicated filtered 2D Gaussians here
    tile_width = get_tile_width(tiling); tile_height = get_tile_height(tiling)
    # always (horizontal, vertical) (!)
    threads = (tile_width, tile_height)
    blocks = (get_nb_tiles_x(tiling), get_nb_tiles_y(tiling))
    @cuda threads blocks render_tiles_kernel!(
        outputs,
        buffers.sdfgidx_to_fgidx, buffers.tiles_end_sdfgidx,
        buffers.f2dgv,
        tiling, options
    )

    return outputs
end

"""
    fill_keep_gaussians_kernel!(
        keep_gaussians,
        nb_gaussians, gaussians::AbstractGaussiansVector{Device, I, F},
        camera,
        options
    ) where {I, F}

Fill the `RenderBuffers`'s `keep_gaussians`. Threads correspond to Gaussian indices.
"""
function fill_keep_gaussians_kernel!(
    keep_gaussians,
    nb_gaussians, gaussians::AbstractGaussiansVector{Device, I, F}, # (always immutable)
    camera,
    options
) where {I, F}

    gidx = threadIdx().x + (blockIdx().x - one(I)) * blockDim().x
    # (Of type I. On the GPU only I === Int32 makes sense, though, also for debugging.)
    stride = blockDim().x * gridDim().x
    min_alpha = get_min_alpha(options)
    @inbounds while gidx <= nb_gaussians
        g = gaussians[gidx]
        keep_gaussian = get_opacity(g) >= min_alpha
        if keep_gaussian 
        # We can skip the projection if not, at the cost of thread divergence
        # This does not seem to make much difference
            if get_boundary_clipping_mode(options) == BoundaryFilter
                mean2d, depth = project_with_depth(get_mean_sv(g), camera) 
                keep_gaussian = keep_gaussian && lies_on_extended_screen(mean2d, options) &&   
                    check_depth(depth, camera) && get_opacity(g) > min_alpha
            else  # CovarianceBoundaryClamp
                keep_gaussian = keep_gaussian && check_depth(get_mean_sv(g), camera)
            end
        end
        keep_gaussians[gidx] = keep_gaussian
        gidx += stride
    end
    return
end

"""
    fill_fgidx_to_gidx_kernel!(
        fgidx_to_gidx::CuDeviceVector{I},
        nb_gaussians,
        keep_gaussians, keep_gaussians_cs,
    ) where I

Fill the `RenderBuffers`'s `f2dgv.fgidx_to_gidx`, given correctly set `keep_gaussians` and
`keep_gaussians_cs`. Note that only `fgidx_to_gidx`'s values at (fg) indices
`1:length(f2dgv)` will be set. The others (`length(f2dgv) + 1 : capacity(f2dgv)`) will
remain garbage.

Threads correspond to Gaussian indices.
"""
function fill_fgidx_to_gidx_kernel!(
    fgidx_to_gidx::CuDeviceVector{I},
    nb_gaussians,
    keep_gaussians, keep_gaussians_cs
) where I

    gidx = threadIdx().x + (blockIdx().x - one(I)) * blockDim().x
    stride = blockDim().x * gridDim().x
    @inbounds while gidx <= nb_gaussians
        if keep_gaussians[gidx]
            fgidx_to_gidx[keep_gaussians_cs[gidx]] = gidx
        end
        gidx += stride
    end
    return nothing
end

"""
    project_fgaussians_kernel!(
        f2dgv::ImmutableFiltered2DGaussiansVector{Device, I, F}, f2dgs_nb_overlapped_tiles,
        nb_fgaussians, 
        camera, 
        tiling, options, ::Val{USE_ALL_SH}
    ) where {I, F}

Complete setting the `RenderBuffers`'s `f2dgv` by projecting the filtered 3D Gaussians
to 2D. Also fills `f2dgs_nb_overlapped_tiles`. All indices above `length(f2dgv)` will
keep having garbage values. 

!!! For `f2dgv` also 2D Gaussians for which the bounding box is empty
(`f2dgs_nb_overlapped_tiles[fgidx] == 0`) will not have proper values set (as we want
to avoid slow unnecessary writes).

Before launching this kernel `f2dgv`'s `fgidx_to_gidx` and parent `all_3d_gaussians` should
have been set. When we want to render a model using the full SH degree the model supports,
setting `USE_ALL_SH` to `true` (statically) is more efficient. When set to `false` we will
view-slice the sh_coeffs using the SH degree from `options`.

Threads correspond to filtered Gaussian indices.
"""
function project_fgaussians_kernel!(
    f2dgv::ImmutableFiltered2DGaussiansVector{Device, I, F}, f2dgs_nb_overlapped_tiles,
    nb_fgaussians, 
    camera, 
    tiling, options, ::Val{USE_ALL_SH}
) where {I, F, USE_ALL_SH}

    fgidx = threadIdx().x + (blockIdx().x - one(I)) * blockDim().x
    stride = blockDim().x * gridDim().x

    h, w = size(camera)
    min_alpha = get_min_alpha(options)
    use_antialiasing = get_use_antialiasing(options)
    aa_min_factor = get_antialiasing_min_factor(options)
    @inbounds while fgidx <= nb_fgaussians
    
        f2dg = f2dgv[fgidx]         # to set
        g = get_3d_gaussian(f2dg)   # from its pre-projection 3D Gaussian

        # Group reads
        mean3d = get_mean_sv(g)
        opacity = get_opacity(g)
        # (We had already computed these in fill_keep_gaussians_kernel!, but it is faster
        #  to recompute than, than to store and retrieve them from memory.)

        mean2d, depth = project_with_depth(mean3d, camera)
        # Depth is fine, has already been checked when setting keep_gaussians
        # Delay (grouped) writes, since these are only relevant when the bounding box is
        # not empty

        if get_boundary_clipping_mode(options) == CovarianceBoundaryClamp
            clamped_mean2d = clamp_to_extended_screen(mean2d, options)
            J = get_linearized_projection_2d(clamped_mean2d, depth, camera)
        else
            J = get_linearized_projection_2d(mean2d, depth, camera)
        end
        cov2d_11, cov2d_21, cov2d_22 = project_covariance_lower_sv(g, J)
        # TODO: could use SymmetricSMatrix cov2d
        #   det, unrolled_inv (which contains det)
        if use_antialiasing
            undilated_cov2d_det = cov2d_11 * cov2d_22 - cov2d_21 * cov2d_21
        end
        cov2d_11 += get_dilation_x(options)
        cov2d_22 += get_dilation_y(options)
        # TODO: DiagonalSMatrix?
        #   (Maybe also the dual to SMatrixWithZeros (few zeros): SparseSMatrix (many zeros))
        icov2d_det = inv(cov2d_11 * cov2d_22 - cov2d_21^2)
        if use_antialiasing
            opacity *= max(aa_min_factor, sqrt(undilated_cov2d_det * icov2d_det))
            # Note that now opacity could fall below min_alpha
            # This will just result in an empty bbox
        end
        # TODO: put in a separate function

        cts_bbox = _get_bbox_cts_px_cov(opacity, mean2d..., cov2d_11, cov2d_22, min_alpha)
        # TODO: for training ADC, maybe compute and store diameter (squared)
        #   --> optional (Nothing or CuArray) type in RenderBuffers, cf. RenderOutputs
        discrete_bbox = _get_bbox_discrete_px(cts_bbox, w, h)
           
        if isempty(discrete_bbox)
            nb_overlapped_tiles = zero(I)
            # Do not write to fg2dv, do not compute color
        else
            tile_ids_rect = rectangle_to_tile_ids(discrete_bbox, tiling, Val(false))
            # We already clamped in _get_bbox_discrete_px

            set_depth!(f2dg, depth)
            set_intersected_tile_range!(f2dg, tile_ids_rect)

            # Compute the inverse covariance matrix and color, but again delay writing
            icov2d_11 = cov2d_22 * icov2d_det
            icov2d_21 = -cov2d_21 * icov2d_det
            icov2d_22 = cov2d_11 * icov2d_det 

            viewing_direction = normalize(mean3d - get_position(camera))
            color = sh_coeffs_to_color(
                Val(get_sh_degree(options)),
                USE_ALL_SH ? 
                    get_sh_coeffs_view(g) : 
                    get_sh_coeffs_view(g, Val(Int64(get_sh_degree(options)))), 
                viewing_direction)
            # (Note that get_sh_degree(options) is already statically known,
            #  so there is no runtime dispatch here.)

            set_mean!(f2dg, mean2d)
            set_opacity!(f2dg, opacity)
            set_inverse_covariance_lower!(f2dg, icov2d_11, icov2d_21, icov2d_22)
            set_color!(f2dg, color)

            nb_overlapped_tiles = 
                (get_right(tile_ids_rect) - get_left(tile_ids_rect) + one(I)) * 
                (get_bottom(tile_ids_rect) - get_top(tile_ids_rect) + one(I))
            # TODO: Test allowing for non-axis-aligned bounding boxes by (cleverly)
            # iterating over the axis-aligned tile ranges
            # --> smaller nb_overlapped_tiles
            # In fill_df2dgs_codes_and_dfgidx_to_fgidx_kernel! we would then again
            # have to iterate, but we already need to do so now.
        end

        # In all cases, do write away nb_overlapped_tiles
        f2dgs_nb_overlapped_tiles[fgidx] = nb_overlapped_tiles

        fgidx += stride
    end
    return nothing
end

"""
    fill_df2dgs_codes_and_dfgidx_to_fgidx_kernel!(
        df2dgs_codes, dfgidx_to_fgidx,
        nb_fgaussians, f2dgv::ImmutableFiltered2DGaussiansVector{Device, I, F},
        f2dgs_nb_overlapped_tiles_cs,
        tiling
    ) where {I, F}

Fill the `RenderBuffers`'s `df2dgs_codes` and `dfgidx_to_fgidx`.

Threads correspond to filtered Gaussian indices.
"""
function fill_df2dgs_codes_and_dfgidx_to_fgidx_kernel!(
    df2dgs_codes, dfgidx_to_fgidx,
    nb_fgaussians, f2dgv::ImmutableFiltered2DGaussiansVector{Device, I, F},
    f2dgs_nb_overlapped_tiles_cs,
    tiling
) where {I, F}

    fgidx = threadIdx().x + (blockIdx().x - one(I)) * blockDim().x
    stride = blockDim().x * gridDim().x
    @inbounds while fgidx <= nb_fgaussians
        
        f2dg = f2dgv[fgidx]
        if fgidx == 1
            dfgidx = one(I)
        else
            dfgidx = f2dgs_nb_overlapped_tiles_cs[fgidx - one(I)] + one(I)
        end
        # The range of dgids for this fgidx is in general (see example render!)
        # f2dgs_nb_overlapped_tiles_cs[fgidx - 1] + 1 : f2dgs_nb_overlapped_tiles_cs[fgidx]
        # At all these entries of dfgidx_to_fgidx we need to place fgidx.
        # Start dgidx at the left boundary, or 1 if we are the first f2dg.
        
        if dfgidx > f2dgs_nb_overlapped_tiles_cs[fgidx]      
            # This fgaussian does not intersect any tile (i.e.
            # f2dgs_nb_overlapped_tiles[fgidx] == 0).
            # Because we skip setting the intersected_tile_range in this case
            # and the values might thus be arbitrary, we must avoid looping over the 
            # intersected_tile_range
            fgidx += stride
            continue
        end

        depth_code = reinterpret(Int32, Float32(get_depth(f2dg)))
        # Irrespective of I and F
        # For depth > 0 (guaranteed by the near clipping plane filtering),
        # depth_code (Int32) has the same order as depth (Float)

        # Loop over the intersected tiles
        tile_range_rect = get_intersected_tile_range_rect(f2dg)
        tile_xidx_start = get_left(tile_range_rect)
        tile_yidx_start = get_top(tile_range_rect)
        tile_xidx_end = get_right(tile_range_rect)
        tile_yidx_end = get_bottom(tile_range_rect)

        # x-contiguous order (not tile-column-major!)
        # See tile_ids_to_linear_idx and the comments in render_tiles_kernel!
        tile_yidx = tile_yidx_start
        while tile_yidx <= tile_yidx_end
            tile_y_code = _tile_y_idx_to_linear_idx_contribution(tiling, tile_yidx)
            # y-part of the full tile code
            tile_xidx = tile_xidx_start
            while tile_xidx <= tile_xidx_end
                tile_code = Int32(
                    _tile_x_idx_and_y_contribution_to_linear_idx(tile_xidx, tile_y_code))
                # 1 <= linear_tile_idx <= get_nb_tiles_x(tiling) * get_nb_tiles_y(tiling)
                # so should definitely fit into 32 bits
                tile_depth_code = Int64(tile_code) << 32 | depth_code
                # Put the tile_code in the 32 most significant digits, the depth_code in the
                # least significant digits
                df2dgs_codes[dfgidx] = tile_depth_code
                dfgidx_to_fgidx[dfgidx] = fgidx

                tile_xidx += one(I)
                dfgidx += one(I)
            end
            tile_yidx += one(I)
        end
        # (Now dfgidx == f2dgs_nb_overlapped_tiles_cs[fgidx] + 1)

        fgidx += stride
    end
    return nothing
end

const cub_sort_pairs_so_path = joinpath(
    @__DIR__, "..", "deps", "cub_sort_pairs", "cub_sort_pairs_wrapper.so")

"""
    sort_duplicated_filtered_2d_gaussians!(
        render_buffers::MutableRenderBuffers{<:GPU, I},
        tiling::Tiling
    ) where I

Call cub's SortPairs (radix sort) to set the `render_buffers`'s `sdf2dgs_codes` and
`sdfgidx_to_fgidx`, provided the non-sorted `df2dgs_codes` and `dfgidx_to_fgidx` were
properly set.
"""
function sort_duplicated_filtered_2d_gaussians!(
    render_buffers::MutableRenderBuffers{<:GPU, I},
    tiling::Tiling
) where I
    nb_dfgaussians_ci = Cint(render_buffers.nb_dfgaussians)
    highest_key_bit_ci = Cint(I(33) + floor(I, log2(get_nb_tiles(tiling))))
    # We do not use all 64 bits. The top 32 bits ([33, 64]) have to encode up to the 
    # maximal linear tile index. This requires floor(log2(...)) + 1 binary digits.
    # Add 32 for the lower bits offets.

    # Make sure we have a large enough sorting buffer
    required_sorting_buffer_bytes = ccall(
        (:determine_sort_pairs_temp_storage, cub_sort_pairs_so_path),
        Csize_t,
        (CuPtr{Int64}, CuPtr{Int64}, 
         CuPtr{Int32}, CuPtr{Int32}, 
         Cint, Cint),
        render_buffers.df2dgs_codes, render_buffers.sdf2dgs_codes,
        render_buffers.dfgidx_to_fgidx, render_buffers.sdfgidx_to_fgidx,
        nb_dfgaussians_ci, highest_key_bit_ci
    )
    render_buffers.sorting_buffer = resize_memory(
        render_buffers.sorting_buffer,
        required_sorting_buffer_bytes;
        free = true)
    # Now length(sorting_buffer) >= required_sorting_buffer_bytes

    # Actual sort
    ccall(
        (:sort_pairs, cub_sort_pairs_so_path),
        Cvoid,
        (Ptr{Cvoid}, 
         Csize_t, 
         CuPtr{Int64}, CuPtr{Int64}, 
         CuPtr{Int32}, CuPtr{Int32}, 
         Cint, Cint),
        reinterpret(Ptr{Cvoid}, pointer(render_buffers.sorting_buffer)), 
        required_sorting_buffer_bytes,
        render_buffers.df2dgs_codes, render_buffers.sdf2dgs_codes,
        render_buffers.dfgidx_to_fgidx, render_buffers.sdfgidx_to_fgidx,
        nb_dfgaussians_ci, highest_key_bit_ci
    )
end


"""
    fill_tiles_end_sdfgidx_kernel!(
        tiles_end_sdfgidx::AbstractVector{I},
        nb_dfgaussians, sdf2dgs_codes, nb_tiles
    ) where I

Fill the `RenderBuffers`'s `tiles_end_sdfgidx`. A tile with linear index `tidx` will
corresponds to sorted duplicated filtered 2D Gaussian indices
`(tiles_end_sdfgidx[tidx - 1], tiles_end_sdfgidx[tidx]]`.

Threads correspond to sorted duplicated filtered Gaussian indices (of which there are
`nb_dfgaussians`).
"""
function fill_tiles_end_sdfgidx_kernel!(
    tiles_end_sdfgidx::AbstractVector{I},
    nb_dfgaussians, sdf2dgs_codes, nb_tiles
) where I

    sdfgidx = threadIdx().x + (blockIdx().x - one(I)) * blockDim().x
    stride = blockDim().x * gridDim().x

    # A tile boundary occurs when the next sdfgidx's (linear) tile index is different
    # from the current one. All tiles in between these two indices are empty.
    # Initial empty tiles will get end_sdfgidx 0, final empty tiles
    # nb_dfgaussians.

    @inbounds if sdfgidx == 1
        # Being the first sdfgaussian in general, we are the first one in the first
        # non-empty tile. Set the end_sdfgidx for all previous (hence empty) `tile_idx` to
        # 0. In that case `[tiles_end_sdfgidx[tidx - 1] + 1, tiles_end_sdfgidx[tidx]]` (with 
        # [1, tiles_end_sdfgidx[1]] for tidx == 1) is indeed empty.
        first_non_empty_tile_idx = I(first(sdf2dgs_codes) >> 32)  # top 32 bits
        tile_idx = one(I)
        while tile_idx < first_non_empty_tile_idx
            tiles_end_sdfgidx[tile_idx] = zero(I)
            tile_idx += one(I)
        end
        # tiles_end_sdfgidx[first_non_empty_tile_idx] will be set below in the general non-
        # boundary case, for sdfgidx == 1
    end

    # (Since we need sdfgidx + 1, we cannot include nb_dfgaussians.)
    @inbounds while sdfgidx < nb_dfgaussians
        current_tile_idx = I(sdf2dgs_codes[sdfgidx] >> 32)
        next_tile_idx = I(sdf2dgs_codes[sdfgidx + one(I)] >> 32)
        while current_tile_idx < next_tile_idx
            # All tiles between next_tile_idx and current_tile_idx are empty.
            # As the dfg range for a tile with index tidx will be
            # [tiles_end_sdfgidx[tidx - 1] + 1, tiles_end_sdfgidx[tidx]],
            # so we can indicate emptyness by setting tiles_end_sdfgidx to the same
            # value for all inbetween tiles. Hence the while-loop.
            @inbounds tiles_end_sdfgidx[current_tile_idx] = sdfgidx
            current_tile_idx += one(I)
        end
        sdfgidx += stride
    end

    @inbounds if sdfgidx == nb_dfgaussians
        # Similar to the case of the first: as the last sdfgaussian in total, we are clearly
        # the final one in our tile. All subsequent tiles will necessarily be empty, so we 
        # again just set sdfgidx everywhere.
        # (In the general while loop we could have included a final dfgaussian check and
        #  in this case set next_tile_idx = nb_tiles + 1. But then this check would have
        #  to occur every iteration, which for smaller strides might carry some overhead.)
        current_tile_idx = I(sdf2dgs_codes[sdfgidx] >> 32)
        while current_tile_idx <= nb_tiles
            tiles_end_sdfgidx[current_tile_idx] = sdfgidx
            current_tile_idx += one(I)
        end
    end
    return nothing
end

"""
    render_tiles_kernel!(
        outputs,
        sdfgidx_to_fgidx, tiles_end_sdfgidx,
        f2dgv::ImmutableFiltered2DGaussiansVector{Device, I, F},
        tiling, options,
    ) where {I, F}

Render the (duplicated) filtered 2D Gaussians (per tile) to produce the output images.

We use a 2D thread and block structure, where each block corresponds to a tile, and each
thread to a pixel. 

The blocks must exactly correspond to the tiles in number and 2D layout, as we will not
loop over blocks, nor check if blocks are out of bounds. (Pixels within a block are still
allowed to be OOB.) In particular, we require 
`blockDim().x == get_tile_width(tiling) && blockDim().y == get_tile_height(tiling)`,
(where the tile width and height should be statically known),
and `gridDim().x == get_nb_tiles_x(tiling) && grimDim().y == get_nb_tiles_y(tiling)`.
"""
function render_tiles_kernel!(
    outputs,
    sdfgidx_to_fgidx, tiles_end_sdfgidx,
    f2dgv::ImmutableFiltered2DGaussiansVector{Device, I, F},
    tiling, options,
) where {I, F}
    # Note that e.g. `threadIdx().x` is changing faster than `threadIdx().y`. I.e. for
    # column-major order `x` corresponds to the vertical component (row index) and y to the
    # horizontal one (column index).)
    # Despite this we empirically found that letting x correspond the horizontal index
    # is faster, potentially because the 3D Gaussians are somehow more naturally
    # horizontally sorted than vertically (e.g. due to the camera trajectory), or
    # because images to render are wider than they are high.
    # TODO: optimal image shape?
    #   maybe add a parameter like CHW or CWH and dispatch (or use if)
    #   might also be better for GUI, where we need 4 x width x height
    #       (could use 1 - T for the alpha-channel, or just 1)

    # Use less confusing, more descriptive names using u and v instead of x and y,
    # where u is unambiguously horizontal and v vertical.
    px_u_in_tile = threadIdx().x; px_v_in_tile = threadIdx().y  # discrete pixel coordinates
    tile_idx_u = blockIdx().x; tile_idx_v = blockIdx().y  # (in image)

    TILE_WIDTH = get_tile_width(tiling)       # == blockDim().x
    TILE_HEIGHT = get_tile_height(tiling)
    NB_THREADS_PER_BLOCK = TILE_WIDTH * TILE_HEIGHT
    NB_SHMEM_GAUSSIANS = NB_THREADS_PER_BLOCK
    # Every thread loads one 2D Gaussian. Concretely, we load those parameters necessary
    # to evaluate the 2D Gaussian: its mean (2), opacity (1) and inverse covariance lower
    # matrix (3). Conveniently we store these continguously in a single array.
    # We also need to load in the colors.
    shmem_gaussian_eval_params = CuStaticSharedArray(F, (6, NB_SHMEM_GAUSSIANS))
    shmem_colors = CuStaticSharedArray(SVector{3, F}, NB_SHMEM_GAUSSIANS)
    # (Faster separately than combined)
    # TODO: Explicit2DGaussians... (see _structs)
    if has_depth_image(outputs)  # statically known
        shmem_depths = CuStaticSharedArray(F, NB_SHMEM_GAUSSIANS)
    end

    # For loading to memory it makes more sense to use a linear thread index
    thread_idx_within_block = threadIdx().x + (threadIdx().y - one(I)) * TILE_WIDTH
    # (== px_idx_within_tile); because TILE_WIDTH == blockDim().x (but statically known)
    tile_idx = tile_ids_to_linear_idx(tiling, tile_idx_u, tile_idx_v)
    # Should be blockIdx().x + (blockIdx().y - one(I)) * gridDim().x

    # Pixel location variables
    tile_u_start_cts = (tile_idx_u - one(I)) * TILE_WIDTH
    tile_v_start_cts = (tile_idx_v - one(I)) * TILE_HEIGHT
    # In (global) continuous PixelCoordinates, of type I

    px_u = px_u_in_tile + tile_u_start_cts  # In global discrete pixel coordinates
    px_v = px_v_in_tile + tile_v_start_cts
    img_height, img_width = size(outputs)
    px_within_screen = one(I) <= px_u <= img_width && one(I) <= px_v <= img_height
    
    cts_px_in_tile_u = px_u_in_tile - F(0.5)
    cts_px_in_tile_v = px_v_in_tile - F(0.5)
    cts_px_in_tile_uu = cts_px_in_tile_u * cts_px_in_tile_u
    cts_px_in_tile_uv = cts_px_in_tile_u * cts_px_in_tile_v
    cts_px_in_tile_vv = cts_px_in_tile_v * cts_px_in_tile_v

    # Accumulating output variables
    px_should_render = px_within_screen

    px_transmittance = one(F)  # (also needed when !has_transmittance_image(outputs))
    # Transmittance after encountering the first n Gaussians (currently: n = 0)
    px_color = zero(MVector{3, F})
    # This is only needed if px_within_screen, but only conditionally declaring it
    # leads to worse performance
        
    if has_depth_image(outputs)  # (Fine here: statically known -> compiled away)
        px_depth = zero(F)
    end
    if has_last_contributing_sdfgids(outputs)
        px_last_contributing_sdfgidx = I(INVALID_LAST_CONTRIBUTING_SDFGIDX_I64)
    end

    # Extract the relevant rendering options
    min_alpha = get_min_alpha(options)
    max_alpha = get_max_alpha(options)
    min_transmittance = get_min_transmittance(options)
    background_color = get_background_color(options)

    # The idea is now to
    # 1) load in a group of NB_SHMEM_GAUSSIANS for our tile
    #   - these have sdgids 
    #       tiles_end_sdfgidx[tile_idx - 1] + 1 : tiles_end_sdfgidx[tile_idx]
    # 2) render them
    # 3) repeat 1 and 2 until all Gaussians have been exhausted

    tile_last_loaded_sdfgidx = if tile_idx == one(I)
        zero(I)
    else
        @inbounds tiles_end_sdfgidx[tile_idx - one(I)]
    end
    
    @inbounds tile_last_sdfgidx = tiles_end_sdfgidx[tile_idx]

    @inbounds while tile_last_loaded_sdfgidx < tile_last_sdfgidx
        # Load tile_last_loaded_sdfgidx + 1 : tile_last_loaded_sdfgidx + NB_THREADS
        # (Note that all threads need to help loading, also those with !px_should_render)
        sdfgidx_to_load = tile_last_loaded_sdfgidx + thread_idx_within_block
        if sdfgidx_to_load <= tile_last_sdfgidx

            fgidx_to_load = sdfgidx_to_fgidx[sdfgidx_to_load]
            f2dg = f2dgv[fgidx_to_load]
            # (Note that this indirection will most likely lead to poor memory access
            #  patterns; there is no hope of memory coalescing. Therefore, we should
            #  focus on cache locality, hence 9 x n in Filtered2DGaussiansVector.)

            x, y = get_mean_sv(f2dg)  # continuous pixel coordinates
            du = x - tile_u_start_cts
            dv = y - tile_v_start_cts

            opacity = get_opacity(f2dg)
            ic11, ic21, ic22 = get_inverse_covariance_lower_sv(f2dg)   

            # Precomputations:
            # Expand quadratic form in Gaussian function's exponential, absorb opacity
            ic11_du = ic11 * du
            ic21_du = ic21 * du
            ic22_dv = ic22 * dv

            shmem_gaussian_eval_params[1, thread_idx_within_block] = F(-0.5) * ic11
            shmem_gaussian_eval_params[2, thread_idx_within_block] = F(-0.5) * ic22
            shmem_gaussian_eval_params[3, thread_idx_within_block] = -ic21
            shmem_gaussian_eval_params[4, thread_idx_within_block] = ic11_du + ic21 * dv
            shmem_gaussian_eval_params[5, thread_idx_within_block] = ic21_du + ic22_dv
            shmem_gaussian_eval_params[6, thread_idx_within_block] =
                F(-0.5) * (ic11_du * du + ic22_dv * dv) - ic21_du * dv + log(opacity)

            shmem_colors[thread_idx_within_block] = get_color_sv(f2dg)

            if has_depth_image(outputs)
                shmem_depths[thread_idx_within_block] = get_depth(f2dg)
            end
        end

        if has_last_contributing_sdfgids(outputs)
            shmem_idx_to_sdfgidx_offset = tile_last_loaded_sdfgidx
        end
        next_tile_last_loaded_sdfgidx = tile_last_loaded_sdfgidx + NB_SHMEM_GAUSSIANS
        # We have tried to load [tile_last_loaded_sdfgidx, next_tile_last_loaded_sdfgidx)
        nb_loaded_shmem_gaussians = if next_tile_last_loaded_sdfgidx <= tile_last_sdfgidx
            NB_SHMEM_GAUSSIANS
        else
            tile_last_sdfgidx - tile_last_loaded_sdfgidx
            # (We use if ... else ... end instead of ifelse as all thread in a warp (a 
            #  fortiori all threads in a block) will follow the same branch.)
            #  ((Additionally, in order for ifelse to be type-stable, both branches must
            #   result in the same type (I). If not, we get a confusing "illegal memory
            #   access error", which is incredibly difficult to debug. Of course, we still
            #   want both branches to result in I for other reasons.))
        end
        # <= NB_SHMEM_GAUSSIANS
        # E.g. if NB_SHMEM_GAUSSIANS == 4, tile_last_loaded_sdfgidx == 8 and 
        # tile_last_sdfgidx == 11, then we only loaded 3 sdgids: 9, 10, 11
        tile_last_loaded_sdfgidx = next_tile_last_loaded_sdfgidx
        # (To be fully correct, this should be 
        #      next_tile_last_loaded_sdfgidx + nb_loaded_shmem_gaussians.
        #  But this is only different when we reached the last Gaussian for our tile, in
        #  which case we stop after this batch anyway.)
        sync_threads()

        # Render the loaded 2D Gaussians in our pixel
        shmem_idx = one(I)
        while px_should_render && shmem_idx <= nb_loaded_shmem_gaussians
            px_alpha = min(max_alpha, exp(
                  shmem_gaussian_eval_params[1, shmem_idx] * cts_px_in_tile_uu
                + shmem_gaussian_eval_params[2, shmem_idx] * cts_px_in_tile_vv
                + shmem_gaussian_eval_params[3, shmem_idx] * cts_px_in_tile_uv
                + shmem_gaussian_eval_params[4, shmem_idx] * cts_px_in_tile_u
                + shmem_gaussian_eval_params[5, shmem_idx] * cts_px_in_tile_v
                + shmem_gaussian_eval_params[6, shmem_idx]
            ))
                
            if px_alpha < min_alpha
                # Skip Gaussian if the contribution is too small
                shmem_idx += one(I)
                continue
            end
            
            px_next_transmittance = px_transmittance * (one(F) - px_alpha)
            # (The transmittance after the current Gaussian)
            if px_next_transmittance < min_transmittance
                # All subsequent Gaussian contributions will be too small
                # (Not necessarily this one's though. But following 3DGS: stop
                # (for numerical stability))
                px_should_render = false
                continue  # Much faster than break, for some reason
                # Do not return, as we still need to help loading
            end
            
            # (px_transmittance is still the transmittance after the previous Gaussian)
            px_weight = px_transmittance * px_alpha
            px_color .+= px_weight .* shmem_colors[shmem_idx]
            px_transmittance = px_next_transmittance
            if has_depth_image(outputs) 
                px_depth += px_weight * shmem_depths[shmem_idx]
            end
            if has_last_contributing_sdfgids(outputs)
                px_last_contributing_sdfgidx = shmem_idx + shmem_idx_to_sdfgidx_offset
            end
            shmem_idx += one(I)
        end

        sync_threads()
        # Make sure we are not still reading from shared memory when other threads start
        # writing to it.
    end

    @inbounds if px_within_screen
        px_color .+= px_transmittance .* background_color

        rgb_image = get_rgb_image(outputs)
        view(rgb_image, SOneTo(3), px_v, px_u) .= px_color

        if has_transmittance_image(outputs)
            get_transmittance_image(outputs)[px_v, px_u] = px_transmittance
        end
        if has_depth_image(outputs)
            get_depth_image(outputs)[px_v, px_u] = px_depth
        end
        if has_last_contributing_sdfgids(outputs)
            get_last_contributing_sdfgids(outputs)[px_v, px_u] = 
                px_last_contributing_sdfgidx
        end
    end

    return nothing    
end