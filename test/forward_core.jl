@testset "RenderHelpers" begin
    # Not the projection or tile rendering itself
    # Cf. examples in comments in render!

    # Filtering
    keep_gaussians = CuVector(Bool[0, 1, 1, 0, 0, 0, 1])
    nb_gaussians = length(keep_gaussians)
    keep_gaussians_cs = CuVector{Int32}(undef, nb_gaussians)
    accumulate!(+, keep_gaussians_cs, keep_gaussians)
    nb_fgaussians = @allowscalar keep_gaussians_cs[nb_gaussians]
    @test nb_fgaussians == count(keep_gaussians) == 3
    @test all(keep_gaussians_cs[keep_gaussians] .== 1:nb_fgaussians)

    fgidx_to_gidx = CuVector{Int32}(undef, nb_gaussians)
    @run_1d_kernel nb_gaussians fill_fgidx_to_gidx_kernel!(
        fgidx_to_gidx, 
        nb_gaussians, 
        keep_gaussians, keep_gaussians_cs
    )
    @test fgidx_to_gidx[1:nb_fgaussians] == CuVector(Int32[2, 3, 7]) == 
        findall(keep_gaussians) == CuArray(1:nb_gaussians)[keep_gaussians]
    # The last two have eltype Int64, but that's fine


    # Duplication and sorting
    nb_gaussians = nb_fgaussians = 5
    gv = MutableTrainableGaussiansVector(
        CUDA.zeros(Float32, 3, nb_gaussians),
        CUDA.zeros(Float32, nb_gaussians),
        CUDA.zeros(Float32, 3, nb_gaussians),
        CUDA.zeros(Float32, 4, nb_gaussians),
        CUDA.zeros(Float32, 1, 3, nb_gaussians),
    )
    tiling = Tiling(2, 2, 1, 1; int_type = Int32)
    buffers = MutableRenderBuffers(gv, tiling)
    set_length!(buffers.f2dgv, nb_fgaussians)
    buffers.f2dgv.fgidx_to_gidx .= 1:nb_gaussians
    
    buffers.f2dgs_nb_overlapped_tiles .= CuVector(Int32[1, 2, 0, 2, 1])
    buffers.f2dgv.depths .= 1:5
    buffers.f2dgv.intersected_tile_ranges .= CuArray(reshape(Int32[
        # x_min, y_min, x_max, y_max
        2, 1, 2, 1,  # one tile (2, 1), cf. f2dgs_nb_overlapped_tiles
        1, 1, 2, 1,  # two tiles (1, 1) and (2, 1)
        1, 1, 0, 1,  # no tiles
        1, 2, 2, 2,  # two tiles (1, 2) and (2, 2)
        1, 1, 1, 1   # one tile (1, 1)
    ], (2, 2, nb_gaussians)))

    accumulate!(+, buffers.f2dgs_nb_overlapped_tiles_cs, buffers.f2dgs_nb_overlapped_tiles)
    # (No need for view when nb_fgaussians == nb_gaussians)
    nb_dfgaussians = @allowscalar buffers.f2dgs_nb_overlapped_tiles_cs[nb_fgaussians]
    @test nb_dfgaussians == sum(buffers.f2dgs_nb_overlapped_tiles)

    set_nb_dfgaussians!(buffers, nb_dfgaussians)
    @run_1d_kernel nb_fgaussians fill_df2dgs_codes_and_dfgidx_to_fgidx_kernel!(
        buffers.df2dgs_codes, buffers.dfgidx_to_fgidx,
        nb_fgaussians, buffers.f2dgv, buffers.f2dgs_nb_overlapped_tiles_cs,
        tiling
    )
    @test buffers.dfgidx_to_fgidx == CuVector(Int32[1, 2, 2, 4, 4, 5])

    sort_duplicated_filtered_2d_gaussians!(buffers, tiling)
    @test buffers.sdfgidx_to_fgidx == CuVector(Int32[
        2, 5,  # tile (1, 1)
        1, 2,  # tile (2, 1)
        4,     # tile (1, 2)
        4      # tile (2, 2)
    ])

    @run_1d_kernel nb_dfgaussians fill_tiles_end_sdfgidx_kernel!(
        buffers.tiles_end_sdfgidx, 
        nb_dfgaussians, buffers.sdf2dgs_codes, get_nb_tiles(tiling)
    )
    @test buffers.tiles_end_sdfgidx == CuVector(Int32[2, 4, 5, 6])
end

@testset "RenderConsistency" begin
    # First compare our CPU and GPU renders
    args = (("GPU", GPU, Int32, Float32), ("CPU", CPU, Int64, Float64))
    all_outputs = Dict("GPU" => Dict(), "CPU" => Dict())
    # The inner dicts have keys the sh_degree and values the CPU-conversion of the render
    # outputs
    for (name, D, I, F) in args 
        for sh_degree = 0:3
            scene = create_test_scene(; sh_degree,
                array_domain = D, int_type = I, float_type = F)
            gaussians = scene.gaussians
            camera = only(scene.cameras)
            outputs = RenderOutputs(camera; array_domain = D, 
                include_transmittance = true, 
                include_depth = true, 
                include_last_contributing_indices = D <: GPU
            )
            options = RenderOptions(
                camera, get_sh_degree(gaussians),
                CovarianceBoundaryClamp;
                background_color = zero(SVector{3, F})
            )
            tiling = Tiling(camera, I(16), I(16))
            buffers = MutableRenderBuffers(gaussians, tiling)

            render!(
                outputs,
                gaussians, camera,
                tiling, options,
                buffers
            )

            all_outputs[name][sh_degree] = CPU(outputs)
        end
    end

    all_approx(x, y) = all(zip(x, y)) do (a, b)
        return isapprox(a, b, atol = 0.001)
    end
    # Stronger than 
    #   isapprox(x, y, ...)
    # as this only uses the Euclidean distance (which could be small, even though at one
    # component the difference is large)
    # Use atol to avoid issues like !(0. ≈ 1e-100) due to the default use of rtol

    for sh_degree = 0:3
        @test all_approx(get_rgb_image(all_outputs["GPU"][sh_degree]), 
                         get_rgb_image(all_outputs["CPU"][sh_degree]))
        # (In particular, for this easy example no numerical instability issues)
        @test all_approx(get_transmittance_image(all_outputs["GPU"][sh_degree]), 
                         get_transmittance_image(all_outputs["CPU"][sh_degree])) 
        @test all_approx(get_depth_image(all_outputs["GPU"][sh_degree]), 
                         get_depth_image(all_outputs["CPU"][sh_degree]))

        @test get_last_contributing_sdfgids(all_outputs["GPU"][sh_degree])[1, 1] == 
            INVALID_LAST_CONTRIBUTING_SDFGIDX_I64
        @test findall(
            ==(INVALID_LAST_CONTRIBUTING_SDFGIDX_I64), 
            get_last_contributing_sdfgids(all_outputs["GPU"][sh_degree])
        ) == findall(==(1), get_transmittance_image(all_outputs["GPU"][sh_degree]))
        # In both cases no Gaussians are active
    end
    

    # Compare to selected values from (original) reference implementation
    rgb_image_0 = get_rgb_image(all_outputs["GPU"][0])
    @test all_approx(rgb_image_0[:,  60,  80], [0.0341, 0.0400, 0.0138])
    @test all_approx(rgb_image_0[:,  60, 160], [0.1124, 0.2627, 0.0687])
    @test all_approx(rgb_image_0[:,  60, 240], [0.0136, 0.0367, 0.0092])
    @test all_approx(rgb_image_0[:, 120,  80], [0.2964, 0.1036, 0.0831])
    @test all_approx(rgb_image_0[:, 120, 160], [0.3058, 0.2397, 0.1646])
    @test all_approx(rgb_image_0[:, 120, 240], [0.1961, 0.1354, 0.2264])
    @test all_approx(rgb_image_0[:, 180,  80], [0.4537, 0.1370, 0.1238])
    @test all_approx(rgb_image_0[:, 180, 160], [0.4323, 0.1960, 0.2960])
    @test all_approx(rgb_image_0[:, 180, 240], [0.4103, 0.2414, 0.4873])
    
    rgb_image_3 = get_rgb_image(all_outputs["GPU"][3])
    @test all_approx(rgb_image_3[:,  60,  80], [0.0357, 0.0417, 0.0155])
    @test all_approx(rgb_image_3[:,  60, 160], [0.1227, 0.2731, 0.0791])
    @test all_approx(rgb_image_3[:,  60, 240], [0.0150, 0.0382, 0.0106])
    @test all_approx(rgb_image_3[:, 120,  80], [0.3019, 0.1091, 0.0886])
    @test all_approx(rgb_image_3[:, 120, 160], [0.3147, 0.2487, 0.1735])
    @test all_approx(rgb_image_3[:, 120, 240], [0.1975, 0.1367, 0.2277])
    @test all_approx(rgb_image_3[:, 180,  80], [0.4613, 0.1447, 0.1315])
    @test all_approx(rgb_image_3[:, 180, 160], [0.4370, 0.2007, 0.3008])
    @test all_approx(rgb_image_3[:, 180, 240], [0.4110, 0.2421, 0.4880])
end