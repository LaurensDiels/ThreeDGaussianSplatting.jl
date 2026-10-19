@testset "Adaptive density control" begin
    # Example from maybe_densify_and_prune!
    nb_gaussians = 8
    should_prune_cpu   = Bool[1, 0, 0, 0, 0, 0, 1, 0]
    should_densify_cpu = Bool[0, 1, 1, 1, 0, 0, 0, 1]
    new_gaussian_counts_cpu = .!should_prune_cpu .+ should_densify_cpu
    @test new_gaussian_counts_cpu == [0, 2, 2, 2, 1, 1, 0, 2]
    new_gaussian_counts_cs_cpu = accumulate(+, new_gaussian_counts_cpu)
    @test new_gaussian_counts_cs_cpu == [0, 2, 4, 6, 7, 8, 8, 10]
    nb_ngaussians = 10
    
    ngidx_to_gidx_cpu = Vector{Int}(undef, nb_ngaussians)
    idx = 1
    for i = 1:nb_gaussians
        should_prune_cpu[i] && continue
        if should_densify_cpu[i]
            ngidx_to_gidx_cpu[idx] = i
            ngidx_to_gidx_cpu[idx + 1] = i
            idx += 2
        else
            ngidx_to_gidx_cpu[idx] = i
            idx += 1
        end
    end
    @test ngidx_to_gidx_cpu == [2, 2, 3, 3, 4, 4, 5, 6, 8, 8]

    ngidx_to_gidx = CuVector{Int32}(undef, nb_ngaussians)
    should_prune = cu(should_prune_cpu)
    should_densify = cu(should_densify_cpu)
    new_gaussian_counts_cs = CuVector{Int32}(new_gaussian_counts_cs_cpu)
    @run_1d_kernel nb_gaussians fill_ngidx_to_gidx_kernel!(
        ngidx_to_gidx,
        nb_gaussians,
        should_prune, should_densify,
        new_gaussian_counts_cs
    )
    @test Vector(ngidx_to_gidx) == ngidx_to_gidx_cpu


    # -- Check adaptive_density_control!
    sh_degree = Int32(1)
    gaussians = MutableTrainableGaussiansVector(
        CuArray([Float32(k) for i=1:3, k=1:8]),         # means
        CuArray([Float32(k) for k=1:8]),                # opacity features
        CuArray([Float32(k) for i=1:3, k=1:8]),         # scaling features
        CuArray([Float32(k) for i=1:4, k=1:8]),         # rotation features
        CuArray([Float32(k) for i=1:4, j=1:3, k=1:8])   # sh coeffs
    )
    cameras = [
        PinholeCamera(
            Int32(320), Int32(240), 
            1.f0, 
            one(SMatrix{3, 3, Float32}), SVector{3, Float32}(x, 0, 0)
        )
        for x in (-1, 1)
    ]
    scene = Scene(gaussians, cameras)
    optimizer = MutableAdamOptimizer(gaussians, TrainingOptions(scene))

    should_clone = cu(Bool[0, 0, 1, 0, 0, 0, 0, 1])
    should_split = cu(Bool[0, 1, 0, 1, 0, 0, 0, 0])
    @assert (should_clone .|| should_split) == should_densify

    # New gaussians are:
    # 1, 2: splits of old 2
    # 3, 4: clones of old 3
    # 5, 6: splits of old 4
    # 7: carried-over old 5
    # 8: carried-over old 6
    # 9, 10: clones of old 8
    # Note that the old ones are just ngidx_to_gidx
    adaptive_density_control!(
        gaussians,
        should_prune, should_densify, should_clone,
        new_gaussian_counts_cs,
        ngidx_to_gidx,
        exp(10.f0)  # for scaling_features, splitting adds 10
    )
    @test length(gaussians) == nb_ngaussians
    gaussians_cpu = CPU(gaussians)
    @test reinterpret(reshape, Float32,
        get_mean_sv.(view(gaussians_cpu, [3, 4, 7, 8, 9, 10]))
    ) == repeat(Float32[3, 3, 5, 6, 8, 8]', 3)
    # Skip the splits, as the new means are randomly sampled
    @test get_opacity_feature.(gaussians_cpu) == ngidx_to_gidx_cpu
    @test reinterpret(reshape, Float32,
        get_scaling_features_sv.(gaussians_cpu)
    ) ≈ repeat(Float32[12, 12, 3, 3, 14, 14, 5, 6, 8, 8]', 3)
    @test reinterpret(reshape, Float32,
        get_rotation_features_sv.(gaussians_cpu)
    ) == repeat(ngidx_to_gidx_cpu', 4)
    @test reshape(
        reinterpret(reshape, Float32, get_sh_coeffs_sm.(gaussians_cpu)), 
        (4, 3, 10)
    ) == repeat(reshape(ngidx_to_gidx_cpu, (1, 1, :)), 4, 3, 1)

    # For optimizer
    # Set Adam running values to 1, 2, 3, ...
    gamlrv_cpu = CPU(optimizer.moments_learning_rates)
    for k = 1:nb_gaussians
        gamlr = gamlrv_cpu[k]
        set_mean_adam_moments!(gamlr, @SMatrix fill(k, 2, 3))
        set_opacity_feature_adam_moments!(gamlr, @SVector fill(k, 2))
        set_scaling_features_adam_moments!(gamlr, @SMatrix fill(k, 2, 3))
        set_rotation_features_adam_moments!(gamlr, @SMatrix fill(k, 2, 4))
        set_sh_coeffs_adam_moments!(gamlr, @SArray fill(k, 2, 4, 3))
    end
    optimizer.moments_learning_rates = GPU(gamlrv_cpu)

    adaptive_density_control!(
        optimizer,
        should_prune, should_densify, should_clone,
        new_gaussian_counts_cs,
        nb_ngaussians
    )
    old_gamlrv_cpu = gamlrv_cpu
    gamlrv_cpu = CPU(optimizer.moments_learning_rates)
    @test length(gamlrv_cpu) == nb_ngaussians
    @test reshape(
        reinterpret(Float32, get_mean_adam_moments_sm.(gamlrv_cpu)),
        (2, 3, 10)
    ) == repeat(reshape(Float32[0, 0, 3, 0, 0, 0, 5, 6, 8, 0], (1, 1, :)), 2, 3, 1)
    # Splits, second clones: 0
    @test get_means_learning_rate(gamlrv_cpu) == get_means_learning_rate(old_gamlrv_cpu)
    # If everything works for the means, it is very unlikely that something will have gone
    # wrong for the other parameters, but we'll check just to be safe
    @test reinterpret(reshape, Float32, 
        get_opacity_feature_adam_moments_sv.(gamlrv_cpu)
    ) == repeat(Float32[0, 0, 3, 0, 0, 0, 5, 6, 8, 0]', 2)
    @test get_opacity_features_learning_rate(gamlrv_cpu) == 
        get_opacity_features_learning_rate(old_gamlrv_cpu)
    @test reshape(
        reinterpret(Float32, get_scaling_features_adam_moments_sm.(gamlrv_cpu)),
        (2, 3, 10)
    ) == repeat(reshape(Float32[0, 0, 3, 0, 0, 0, 5, 6, 8, 0], (1, 1, :)), 2, 3, 1)
    @test get_scaling_features_learning_rate(gamlrv_cpu) == 
        get_scaling_features_learning_rate(old_gamlrv_cpu)
    @test reshape(
        reinterpret(Float32, get_rotation_features_adam_moments_sm.(gamlrv_cpu)),
        (2, 4, 10)
    ) == repeat(reshape(Float32[0, 0, 3, 0, 0, 0, 5, 6, 8, 0], (1, 1, :)), 2, 4, 1)
    @test get_rotation_features_learning_rate(gamlrv_cpu) == 
        get_rotation_features_learning_rate(old_gamlrv_cpu)
    @test reshape(
        reinterpret(Float32, get_sh_coeffs_adam_moments_sa.(gamlrv_cpu)),
        (2, 4, 3, 10)
    ) == repeat(reshape(Float32[0, 0, 3, 0, 0, 0, 5, 6, 8, 0], (1, 1, 1, :)), 2, 4, 3, 1)
    @test get_sh_coeffs_dc_learning_rate(gamlrv_cpu) == 
        get_sh_coeffs_dc_learning_rate(old_gamlrv_cpu)
    @test get_sh_coeffs_rest_learning_rate(gamlrv_cpu) == 
        get_sh_coeffs_rest_learning_rate(old_gamlrv_cpu)

    # For fgidx_to_gidx and fgradients
    tiling = Tiling(first(cameras))
    render_buffers = MutableRenderBuffers(gaussians, tiling)
    nb_fgaussians = 3
    set_nb_fgaussians!(render_buffers.f2dgv, nb_fgaussians)
    original_fgidx_to_gidx_cpu = [2, 6, 8]  # all other Gaussians are filtered out
    copyto!(render_buffers.f2dgv.fgidx_to_gidx, original_fgidx_to_gidx_cpu)
    fgradients = MutableFilteredGaussiansGradientsVector(render_buffers.f2dgv)
    fgradients_cpu = CPU(fgradients)
    for k = 1:nb_fgaussians
        fgg = fgradients_cpu[k]
        set_mean3d_grad!(fgg, @SVector fill(k, 3))
        set_opacity_feature_grad!(fgg, k)
        set_scaling_features_grad!(fgg, @SVector fill(k, 3))
        set_rotation_features_grad!(fgg, @SVector fill(k, 4))
        set_sh_coeffs_grad!(fgg, @SMatrix fill(k, 4, 3))
    end
    fgradients = GPU(fgradients_cpu)
    # (Note that fgradients.f2dgv !== render_buffers.f2dgv, even though all of their fields
    # are equal, but not identical. Since our change to render_buffers.f2dgv.fgidx_to_gidx
    # should also affect fgradients.f2dgv.fgidx_to_gidx, we need to be a bit careful
    fgradients.f2dgv = render_buffers.f2dgv

    # fgidx 1 -> old gidx 2 (split) -> 0 (no splits get gradients) 
    # fgidx 2 -> old gidx 6 (carried-over) -> ngidx 8
    # fgidx 3 -> old gidx 8 (copied) -> ngidx 9 (first copy)
    @run_1d_kernel nb_fgaussians adjust_fgidx_to_gidx_adc_kernel!(
        render_buffers.f2dgv.fgidx_to_gidx,
        nb_fgaussians, 
        should_prune,
        should_densify,
        should_clone,
        new_gaussian_counts_cs
    )
    @test view(render_buffers.f2dgv.fgidx_to_gidx, 1:nb_fgaussians) == cu[0, 8, 9]

    gaussians_before_cpu = CPU(gaussians)
    # Set optimizer settings to more easy values to calculate with
    gamlrv = optimizer.moments_learning_rates
    set_means_learning_rate!(gamlrv, 1)
    set_opacity_features_learning_rate!(gamlrv, 1)
    set_scaling_features_learning_rate!(gamlrv, 1)
    set_rotation_features_learning_rate!(gamlrv, 1)
    set_sh_coeffs_dc_learning_rate!(gamlrv, 1)
    set_sh_coeffs_rest_learning_rate!(gamlrv, 10)
    set_beta1!(optimizer, 0.5)
    set_beta2!(optimizer, 0.5)
    set_epsilon!(optimizer, 0)
    set_update_count!(optimizer, 0)  # (was already the case)
    update!(optimizer, fgradients, Val(sh_degree))
    gaussians_after_cpu = CPU(gaussians)
    # Check that we did not alter the splits and second copies
    splits_second_copies_ids = [1, 2, 4, 5, 6, 10]
    # or filtered-out gaussians, i.e. those coming from old indices 1, 3, 4, 5, 7
    filtered_out_indices = [3, 4, 5, 6, 7]
    
    for ids in (splits_second_copies_ids, filtered_out_indices)
        @test get_mean_sv.(view(gaussians_before_cpu, ids)) == 
            get_mean_sv.(view(gaussians_after_cpu, ids))
        @test get_opacity_feature.(view(gaussians_before_cpu, ids)) == 
            get_opacity_feature.(view(gaussians_after_cpu, ids))
        @test get_scaling_features_sv.(view(gaussians_before_cpu, ids)) == 
            get_scaling_features_sv.(view(gaussians_after_cpu, ids))
        @test get_rotation_features_sv.(view(gaussians_before_cpu, ids)) == 
            get_rotation_features_sv.(view(gaussians_after_cpu, ids))
        @test get_sh_coeffs_sm.(view(gaussians_before_cpu, ids)) == 
            get_sh_coeffs_sm.(view(gaussians_after_cpu, ids)) 
    end

    # Finally, check that we did update the carried-over Gaussians and first clones,
    # when the original spawning Gaussian was not filtered out, namely ngids 8 and 9
    # (coming from gids 6 and 8, which correspond to fgids 2 and 3)
    function compute_new_values(gidx, fgidx, learning_rate = 1)
        grad = fgidx
        old_value = old_m1 = old_m2 = gidx
        # new_mx = betax * old_mx + (one(F) - betax) * grad = (old_mx + grad) / 2
        #   as betax = 0.5
        # new_mx_hat = new_mx / (one(F) - betax ^ t) = 2 * new_mx = old_mx + grad
        #   as additionally t = 1
        new_m1_hat = old_m1 + grad
        new_m2_hat = old_m2 + grad ^ 2
        return old_value - learning_rate * new_m1_hat / sqrt(new_m2_hat),
            new_m1_hat / 2, new_m2_hat / 2
    end
    gamlrv_cpu = CPU(optimizer.moments_learning_rates)
    for (ngidx, gidx, fgidx) in ((8, 6, 2), (9, 8, 3))
        p, m, v = compute_new_values(gidx, fgidx)
        @test all(get_mean_sv(gaussians_after_cpu[ngidx]) .≈ p)
        @test all(get_mean_adam_moments_view(gamlrv_cpu[ngidx]) .≈ (m, v))
        @test get_opacity_feature(gaussians_after_cpu[ngidx]) .≈ p
        @test all(get_opacity_feature_adam_moments_view(gamlrv_cpu[ngidx]) .≈ (m, v))
        @test all(get_scaling_features_sv(gaussians_after_cpu[ngidx]) .≈ p)
        @test all(get_scaling_features_adam_moments_view(gamlrv_cpu[ngidx]) .≈ (m, v))
        @test all(get_rotation_features_sv(gaussians_after_cpu[ngidx]) .≈ p)
        @test all(get_rotation_features_adam_moments_view(gamlrv_cpu[ngidx]) .≈ (m, v))
        @test all(get_sh_coeffs_dc_sv(gaussians_after_cpu[ngidx]) .≈ p)
        @test all(get_sh_coeffs_dc_adam_moments_view(gamlrv_cpu[ngidx]) .≈ (m, v))

        p_rest, m_rest, v_rest = compute_new_values(gidx, fgidx, 10.)
        @test all(get_sh_coeffs_rest_view(gaussians_after_cpu[ngidx]) .≈ p_rest)
        @test all(
            get_sh_coeffs_rest_adam_moments_view(gamlrv_cpu[ngidx]) .≈ (m_rest, v_rest))
    end

end

@testset "sh_degree_increase" begin
    sh_degree = Int32(0)
    scene = create_test_scene(; sh_degree, two_cameras = true)
    gaussians = scene.gaussians; camera = first(scene.cameras)
    render_options = RenderOptions(first(scene.cameras), sh_degree)
    tiling = Tiling(first(scene.cameras))
    training_options = TrainingOptions(scene; sh_degree_increase_interval = 1)
    # Always increase the degree when asked (up to 3)

    trainer = Trainer(
        scene.gaussians,
        map(scene.cameras) do cam
            CameraWithImage(
                cam,
                EagerImageLoader("", create_array(GPU, Float32, (0, 0)))
            )
        end,
        render_options, tiling, training_options,
        MutableAdamOptimizer(scene.gaussians, training_options)
    )
     
    @test get_sh_degree(trainer.fgradients) == 0
    sh_dc = get_sh_coeffs_dc_sv.(gaussians)
    org_sh_dc_ams = SArray{Tuple{2, 1, 3}}(1, 10, 2, 20, 3, 30)
    @allowscalar set_sh_coeffs_adam_moments!(
        trainer.optimizer.moments_learning_rates[end],
        org_sh_dc_ams)
    maybe_increase_sh_degree!(trainer)
    @test get_sh_degree(trainer.render_options) == 1
    @test get_sh_degree(trainer.gaussians_vector) == 1
    @test get_sh_coeffs_dc_sv.(gaussians) == sh_dc 
    @test all(v -> all(==(0), v), get_sh_coeffs_rest_view.(CPU(gaussians)))
    @test get_sh_degree(trainer.fgradients) == 1
    @test get_sh_degree(trainer.optimizer.moments_learning_rates) == 1
    new_sh_ams = @allowscalar get_sh_coeffs_adam_moments_sa(
        trainer.optimizer.moments_learning_rates[end])
    @test new_sh_ams[:, 1:1, :] == org_sh_dc_ams

    # For other sh degree increases, the behavior should also be fine. Let's explicitly
    # check we cap at 3 through
    maybe_increase_sh_degree!(trainer)
    @test get_sh_degree(gaussians) == 2  # (or trainer.gaussian_vector)
    maybe_increase_sh_degree!(trainer)
    @test get_sh_degree(gaussians) == 3
    maybe_increase_sh_degree!(trainer)
    @test get_sh_degree(gaussians) == 3
end

@testset "opacity_reset" begin
    scene = create_test_scene(; two_cameras = true)
    gaussians = scene.gaussians; camera = first(scene.cameras)
    render_options = RenderOptions(first(scene.cameras), Int32(0))
    tiling = Tiling(first(scene.cameras))
    training_options = TrainingOptions(scene; 
        opacity_reset_interval = 1,
        opacity_reset_start_iteration_nb = 0,
        opacity_reset_max_value = 0.8)

    trainer = Trainer(
        scene.gaussians,
        map(scene.cameras) do cam
            CameraWithImage(
                cam,
                EagerImageLoader("", create_array(GPU, Float32, (0, 0)))
            )
        end,
        render_options, tiling, training_options,
        MutableAdamOptimizer(scene.gaussians, training_options)
    )
     
    opacities = get_opacity.(gaussians)
    @allowscalar set_opacity_feature_adam_moments!(
        trainer.optimizer.moments_learning_rates[end],
        1, 10
    )
    maybe_opacity_reset!(trainer)
    b = opacities .<= 0.8
    @test all(get_opacity.(view(gaussians, b)) .≈ opacities[b])
    @test all(get_opacity.(view(gaussians, .!b)) .≈ 0.8)
end
