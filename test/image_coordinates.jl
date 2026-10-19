@testset "CoordinateConverter" begin
    w = 100
    ptn = create_coordinate_converter(PixelCoordinates, NormalizedCoordinates, Float32, w)
    @test convert_coordinate(ptn, 0) ≈ -1.f0
    @test convert_coordinate(ptn, w) ≈ 1.f0
    @test convert_length(ptn, w) ≈ 2.f0

    n = convert_coordinate(
        NormalizedToNormalizedConverter(Float64),
        123.f0
    )
    @test n isa Float64 && n ≈ 123.

    ntp = NormalizedToPixelConverter(Float32, w)
    p = 125; n = convert_coordinate(ptn, p); pp = convert_coordinate(ntp, n)
    @test n ≈ 1.5f0
    @test pp isa Float32 && pp ≈ p
end
