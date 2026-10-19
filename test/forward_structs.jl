@testset "RenderOutputs" begin
    w = 1280; h = 720
    ro = RenderOutputs(
        ones(3, h, w), nothing, zeros(h, w), nothing
    )

    @test get_width(ro) == w
    @test get_height(ro) == h
    @test has_rgb_image(ro)
    @test all(==(1.), get_rgb_image(ro))
    @test !has_transmittance_image(ro)
    @test get_transmittance_image(ro) === nothing
    @test has_depth_image(ro)
    @test all(==(0.), get_depth_image(ro))
    @test !has_last_contributing_sdfgids(ro)
    @test get_last_contributing_sdfgids(ro) === nothing

    gro = GPU(ro)
    @test gro isa RenderOutputs{<:GPU}
    @test get_rgb_image(gro) isa CuArray
    @test get_transmittance_image(gro) === nothing

    maybe_unsafe_free!(ro)
    r = get_rgb_image(ro)  # test: does not error
    maybe_unsafe_free!(gro)
    @test_throws "freed" get_rgb_image(gro)[1:1]

    sro = TDGS.resize!(ro, (10, 20))  # shrink, reuse memory
    @test eltype(ro) === eltype(sro)  # inclusing all nothings
    @test size(sro) == (10, 20)
    @test get_width(sro) == 20 && get_height(sro) == 10
    get_rgb_image(sro)[1, 1, 1] = 123.
    @test get_rgb_image(ro)[1, 1, 1] == 123.  # same memory
    get_rgb_image(sro)[1, 1, 1] = 1.  # set it back

    lro = TDGS.resize!(sro, (h, w))  # enlarge, new memory
    @test eltype(lro) === eltype(sro)
    @test size(lro) === (h, w)
    get_rgb_image(lro) .= -1.
    @test get_rgb_image(ro)[1, 1, 1] == 1.
end

@testset "Tiling and Rectangle" begin
    w = 1920; h = 1080
    t = Tiling(
        w, h, 
        16, 32
    )
    r = Rectangle(16, 48, 32, 97)
    tids = rectangle_to_tile_ids(r, t) 
    @test get_top_left(tids) == SA[1, 2]
    @test get_bottom_right(tids) == SA[2, 4]

    @test get_tile_width(t) == 16 && get_tile_height(t) == 32
    tids_a = rectangle_to_tile_ids(
        Rectangle(0.f0, -0.5f0, Float32(w), h + 32.f0), t, Val(false)
    )
    @test tids_a isa Rectangle{Int64}
    @test all(==(0), get_top_left(tids_a))        # left boundary exclusive
    @test get_right(tids_a) == get_nb_tiles_x(t)  # right boundary inclusive
    @test get_bottom(tids_a) > get_nb_tiles_y(t)

    tids_b = rectangle_to_tile_ids(
        Rectangle(0.f0, -0.5f0, Float32(w), h + 32.f0), t  # Val(true)
    )
    @test all(==(1), get_top_left(tids_b))
    @test get_right(tids_b) == get_nb_tiles_x(t)
    @test get_bottom(tids_b) == get_nb_tiles_y(t)
end

@testset "Filtered2DGaussians" begin
    gv = ImmutableTrainableGaussiansVector(
        [1., 2., 3.] .* [1, 10, 100, 1000]',
        ones(4),
        ones(3, 4),
        ones(4, 4),
        ones(1, 3, 4);
        int_type = Int64 
    )

    # The below values are not consistent with gv, but this does not matter here.
    fgidx_to_gidx = [2, 3, -1, -1]  # skip the first and the last 3D Gaussian
    depths = [10., 100., -1., -1.]
    means2d_opacities_inverse_covariances_colors = [
         1.;  2.; 0.03; 0.4; 0.5; 0.6; 1.; 0.; 0.;; 
        10.; 20.; 0.3 ;  4.;  5.; 6. ; 0.; 1.1; 0.5;;
        fill(-1., (9, 2))]
    intersected_tile_ranges = [1; 2;; 3; 4;;; 10; 20;; 30; 40;;; fill(-1, (2, 2, 2))]
    f2dgv = ImmutableFiltered2DGaussiansVector(
        2,
        gv, fgidx_to_gidx,
        depths, means2d_opacities_inverse_covariances_colors,
        intersected_tile_ranges
    )

    @test length(f2dgv) == 2
    @test capacity(f2dgv) == 4

    g = f2dgv[2]
    @test get_depth(g) == 100.
    @test get_inverse_covariance_lower_sv(g) == SA[4., 5., 6.]
    @test get_mean_view(get_3d_gaussian(g))[1] == 100.

    @test_throws r"B|bounds" f2dgv[3]  # within capacity, but after length

    rgbs = get_color_rgb.(f2dgv)
    @test eltype(rgbs) === RGB{Float64}
    @test rgbs == [RGB{Float64}(1, 0, 0), RGB{Float64}(0, 1, 0.5)]  # clamped

    sf2dgv = f2dgv[[false, true, false, true]]
    @test length(sf2dgv) == 1
    @test capacity(sf2dgv) == 2
    @test get_depth(first(sf2dgv)) == get_depth(f2dgv[2])

    vgf2dgv = @view GPU(f2dgv)[2:end]
    @test length(vgf2dgv) == 1
    @test capacity(vgf2dgv) == 1
    cgf2dgv2 = GPU(f2dgv)[2:capacity(f2dgv)]
    @test length(cgf2dgv2) == 1
    @test capacity(cgf2dgv2) == 3
    @test @allowscalar get_mean_sv(get_3d_gaussian(cgf2dgv2[1]))[1] == 100.
end