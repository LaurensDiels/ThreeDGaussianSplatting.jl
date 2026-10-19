@testset "Gaussians" begin
    means = [1. 4. 7.
             2. 5. 8.
             3. 6. 9.]  # first Gaussians's mean is [1., 2., 3.]
    opacity_features = [0.5; 0.25; 0.75]  # (1 x) 3
    scaling_features = [log(1) log(4) log(7)
                        log(2) log(5) log(8)
                        log(3) log(6) log(9)]
    rotation_features = [sqrt(2)/2 1. 1.
                                0. 0. 0.
                                0. 0. 0.
                         sqrt(2)/2 0. 0.]
    # 90° rotation around z-axis, twice no rotation
    sh_coeffs = ones(1, 3, 3)

    tgv = MutableTrainableGaussiansVector(
        means,
        opacity_features,
        scaling_features, rotation_features,
        sh_coeffs;
        int_type = Int64
    )

    @test tgv isa MutableTrainableGaussiansVector{CPU, Int64, Float64}
    @test length(tgv) == 3
    @test get_sh_degree(tgv) == 0
    @test_throws r"B|bounds" tgv[4]  
    # Works both for ArgumentError with "out of bounds" message, and BoundsError
    # (in case we decide to change to the latter)
    
    g = first(tgv)
    @test get_mean_sv(g) == SA[1., 2., 3.]
    @test get_opacity_feature(g) == 0.5
    @test get_opacity(g) ≈ 0.622459331
    @test get_scaling_features_sv(g) == SA[0., 0.6931471805599453, 1.0986122886681098]
    @test get_scaling_sv(g) ≈ SA[1., 2., 3.]
    @test get_rotation_features_sv(g) == SA[sqrt(2)/2, 0., 0., sqrt(2)/2]
    @test get_rotation_sm(g) ≈ SA[
        0. -1. 0.
        1.  0. 0.
        0.  0. 1.
    ]
    @test get_covariance_sm(g) ≈ SA[
        4. 0. 0.
        0. 1. 0.
        0. 0. 9.
    ]
    @test get_sh_degree(g) == 0
    @test get_sh_coeffs_sm(g) == SA[1. 1. 1]  # 1 x 3
    @test get_sh_coeffs_dc_sv(g) == SA[1., 1., 1.]
    @test all(get_dc_color_sv(g) .≈ inv(sqrt(4 * pi)) + 0.5)
    set_dc_color!(g, SVector(0.1, 0.2, 0.3))
    @test get_dc_color_sv(g) ≈ SA[0.1, 0.2, 0.3]
    get_sh_coeffs_view(g) .= 1.
    # Set it back to the original value (which we might expect later)
    @test all(==(1), get_sh_coeffs_dc_sv(g))

    @test first.(get_mean_sv.(tgv)) == [1., 4., 7.]
    @test length(tgv[first.(get_mean_sv.(tgv)) .>= 2]) == 2

    stgv = tgv[[false, true, true]]
    @test length(stgv) == 2
    @test get_mean_sv(first(stgv)) == get_mean_sv(tgv[2])
    vtgv = @view tgv[2:2:end]
    @test length(vtgv) == 1
    @test get_mean_sv(first(vtgv)) == get_mean_sv(tgv[2])
    get_mean_view(first(vtgv)) .= 0
    @test all(==(0), get_mean_sv(tgv[2]))
    set_mean!(first(vtgv), 4., 5., 6.)
    @test get_mean_sv(vtgv[1]) ≈ SA[4., 5., 6.]


    gtgv = GPU(tgv)
    @test gtgv isa MutableTrainableGaussiansVector{<:GPU, Int64, Float64}
    @test first.(get_mean_sv.(gtgv)) == CuArray([1., 4., 7.])
    @test first.(get_mean_sv.(gtgv[cu([false, true, true])])) == CuArray([4., 7.])
    vgtgv = @view gtgv[cu([2, 2, 1, 3])]
    @test length(vgtgv) == 4
    @test first.(get_mean_sv.(vgtgv)) == CuArray([4., 4., 1., 7.])

    dtgv = Device(tgv)
    @test dtgv isa ImmutableTrainableGaussiansVector{Device, Int64, Float64}


    rgv = RenderOnlyGaussiansVector(tgv)
    @test rgv isa RenderOnlyGaussiansVector{CPU, Int64, Float64}
    @test length(rgv) == 3
    @test get_sh_degree(rgv) == 0

    g = first(rgv)
    @test get_opacity(g) ≈ 0.622459331
    @test get_covariance_sm(g) ≈ SA[
        4. 0. 0.
        0. 1. 0.
        0. 0. 9.
    ]

    @test first.(get_mean_sv.(rgv)) == [1., 4., 7.]

    set_sh_degree!(tgv, 1)
    @test get_sh_degree(tgv) == 1
    @test get_sh_coeffs_sm(first(tgv)) == SA[
        1. 1. 1.
        0. 0. 0.
        0. 0. 0.
        0. 0. 0.
    ]

    pcd_pts = stack(Float64[i, i, i] for i = 1:6)
    pcd_rgbs = pcd_pts ./ size(pcd_pts, 2)
    rgv_from_pcd = RenderOnlyGaussiansVector(
        pcd_pts, pcd_rgbs;
        array_domain = CPU,
        float_type = Float32
    )
    @test get_mean_sv(rgv_from_pcd[3]) == @SVector Float32[3, 3, 3]
    @test get_opacity(rgv_from_pcd[3]) == 0.1f0
    ds = map(1:length(rgv_from_pcd)) do i
        norm(view(pcd_pts, :, i) .- view(pcd_pts, :, 3))
    end
    mean_d = mean(sort(ds)[2:4])
    @test get_covariance_lower_sv(rgv_from_pcd[3])[1] ≈ mean_d ^ 2

    mtgv_from_pcd = MutableTrainableGaussiansVector(pcd_pts, pcd_rgbs) # GPU
    @allowscalar @test get_scaling_sv(mtgv_from_pcd[3]) ≈ SA[mean_d, mean_d, mean_d]

end