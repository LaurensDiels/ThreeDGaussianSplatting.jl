@testset "SymmetricSMatrix" begin
    sm = SA[1. 3.
            2. 4.]
    ssm = SymmetricSMatrix(sm)
    @test SMatrix(ssm) == SA[1. 2.
                             2. 4.]  # (In particular, take the lower triangle entry)

    c = congruence_transform(sm, ssm)
    @test SMatrix(c) ≈ sm * SMatrix(ssm) * sm' ≈ sm * ssm * sm'
    @test c isa SymmetricSMatrix

    @test transpose(ssm) == ssm == adjoint(ssm) == SMatrix(ssm)
    # also tests == for two StaticSMatrices
    @test ssm == SMatrix(ssm)  # generic AbstractArray fallback

    ssm2_a = ssm * 2; ssm2_b = ssm .* 2; ssm2_c = ssm .+ ssm; ssm2_d = ssm + ssm
    @test ssm2_a ≈ ssm2_b ≈ ssm2_c ≈ ssm2_d 
    @test typeof(ssm2_a) == typeof(ssm2_b) <: SymmetricSMatrix{2}
    # (If we wanted to used ScaledSArrays more eagerly, this could have been
    # a ScaledSMatrix{2, 2})
    @test typeof(ssm2_c) == typeof(ssm2_d) <: SymmetricSMatrix{2}

    @test ssm .+ 1 isa SymmetricSMatrix
    @test ssm .+ Diagonal([1, 2]) isa AbstractMatrix 
    @test ssm .+ Matrix(ssm) isa AbstractMatrix
    # Does not really matter what we get, we just should not throw an error
    @test ssm .+ sm isa SMatrix

    @test SMatrix(ssm .* ssm .* ssm) ≈ SMatrix(ssm) .* SMatrix(ssm) .* SMatrix(ssm)

    @test abs(det(ssm)) < 1e-5
    invertible_ssm = SymmetricSMatrix(SVector(1., 2., 3.))
    @test det(invertible_ssm) ≈ det(SMatrix(invertible_ssm)) # (≈ -1)
    @test inv(invertible_ssm) ≈ inv(SMatrix(invertible_ssm))

    sms_gpu = cu(fill(Float32.(sm), 2))
    ssms_gpu = SymmetricSMatrix.(sms_gpu)
    @test all(sms_gpu .* ssms_gpu .== Ref(sm * SMatrix(ssm)))
    # (Note: broadcasted matrix multiplication)


    sasm = SelfAdjointSMatrix((1 + 0im, 1im, 2 + 0im))
    # (1, im, 2) has a strange type (technically an NTuple, but with composite eltype)
    # (Also note that typeof(im) === Complex{Bool})
    # SVector(1, im, 2) does work (resulting in an SVector{3, Complex{Int64}}`,
    # but our code did not, so we now guard against this by requiring isbits eltypes
    @test transpose(sasm) == SA[1 im; -im 2]
    @test sasm' == sasm
    @test transpose(sasm) * adjoint(sasm) == unrolled_matmul(
        sasm, sasm, Val(true), Val(true), Val(true), Val(false))

    @test gradient_wrt_symmetric_matrix_to_gradient_wrt_lower_vector(sm) == SA[1., 5., 4.]
    @test gradient_wrt_symmetric_matrix_to_gradient_wrt_lower_vector(ssm) == SA[1., 4., 4.]
end

@testset "SArrayWithZeros" begin
    svwz = SVectorWithZeros(SVector(1, 2), Val(3), Val(((2,),)))
    @test SVector(svwz) == SA[1, 0, 2]
    tsvwz = transpose(svwz)
    @test tsvwz isa SMatrixWithZeros{1, 3}
    @test get_zero_indices(tsvwz) == ((1, 2),) 
    @test SMatrix(tsvwz) == SA[1 0 2]
    @test tsvwz == svwz'
    
    sm = SA[1. 3. 5.
            2. 4. 6.] .* im
    smwz = SMatrixWithZeros(sm, Val(((1, 2),)))
    @test SMatrix(smwz) == SA[1. 0. 5.
                              2. 4. 6.] .* im
    @test smwz' isa SMatrixWithZeros{3, 2, Complex{Float64}}
    @test smwz' == -transpose(smwz)

    @test smwz * smwz' == SMatrix(smwz) * SMatrix(smwz)'

    smwz2 = SMatrixWithZeros(sm, Val(((2, 3), (1, 2))))
    @test get_zero_indices(smwz - smwz2) == ((1, 2),) 
    # In particular, the difference is a SMatrixWithZeros
    @test SMatrix(smwz .- smwz2) == SMatrix(smwz - smwz2) == SMatrix(smwz) .- SMatrix(smwz2)
    @test get_zero_indices(smwz .* smwz2) == ((1, 2), (2, 3))  # canonical order
    @test SMatrix(smwz .* smwz2) == SMatrix(smwz) .* SMatrix(smwz2)

    
    smwz_zero_row = SMatrixWithZeros(
        SMatrix{4, 4, Float32}(1:16),
        Val(((2, 1), (2, 2), (2, 3), (2, 4)))
    )
    @test det(smwz_zero_row) == 0.f0
    @test _unrolled_det_code(typeof(smwz_zero_row), :M).args[end].args[end] == 0.0f0
    # This expression should (up to some useless LineNumberNodes, a variable declaration
    # for the underlying Tuple, and an @inbounds) just consist of returning 0.f0
end

@testset "ScaledSArrays" begin
    sa = SA[1; 2;; 3; 4;;; 5; 6;; 7; 8]
    scsa = ScaledSArray(2, sa)
    @test get_scale_factor(scsa) == 2
    @test get_scale_factor_type(scsa) == Int
    @test SArray(scsa) == 2 .* sa

    sa2 = @SArray [2 for i=1:2, j=1:2, k=1:2]
    scsa2 = ScaledSArray(3., sa2)
    @test scsa .* scsa2 isa ScaledSArray
    @test scsa .* scsa2 ./ 12. == sa 
    @test scsa .* 6 isa ScaledSArray
    @test scsa .* 6 == scsa .* scsa2

    @test scsa + scsa2 == scsa .+ scsa2 == 2 .* sa .+ 3 .* sa2
    @test (scsa + scsa2) isa SArray
    # Generic (SpecialSArray fallback)

    scsm = ScaledSArray(im, SA[1; 2;; 3; 4])
    @test det(scsm) == det(SMatrix(scsm)) == 2
    @test kron(scsm, scsm) == kron(SMatrix(scsm), SMatrix(scsm))
    @test dot(im * scsm, scsm) == -im * dot(scsm, scsm) == -dot(scsm, 1im * scsm) == 
        -im * norm(get_underlying_matrix(scsm)) ^ 2  # (as abs2(im) == 1)
end

@testset "SpecialSArray interactions" begin
    svwz = SVectorWithZeros(SVector(2, 2), Val(3),  Val(((2,),)))
    # [2, 0, 2]
    ssm = SymmetricSMatrix(svwz)
    # [2 0; 0 2]
    sv = SVector(2, 3)
    @test ssm * sv == 2 * sv

    @test unrolled_inv(ssm, Val(false)) isa SymmetricSMatrix
    @test unrolled_inv(ssm, Val(true)) isa ScaledSArray
    @test unrolled_inv(ssm, Val(false)) == unrolled_inv(ssm, Val(true)) == inv(ssm) == 
        inv(SMatrix(ssm))

    ssm2 = SymmetricSMatrix(SVector(1., 2., 3.))
    scsm2 = ScaledSMatrix(2., ssm2)
    ct = congruence_transform(scsm2, ssm2)
    @test ct isa SymmetricSMatrix
    @test get_lower_v(ct) isa ScaledSVector
    @test ct == scsm2 * ssm2 * transpose(scsm2) == 
        SMatrix(scsm2) * SMatrix(ssm2) * transpose(SMatrix(scsm2)) == 
        4 * ssm2 * ssm2 * ssm2
end