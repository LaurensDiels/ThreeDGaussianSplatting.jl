const SH_CONSTS_F64 = (
     0.28209479177387814,
     0.4886025119029199,
     0.4886025119029199,
     0.4886025119029199,
     1.0925484305920792,
    -1.0925484305920792,
     0.31539156525252005,
    -1.0925484305920792,
     0.5462742152960396,
    -0.5900435899266435,
     2.890611442640554,
    -0.4570457994644658,
     0.3731763325901154,
    -0.4570457994644658,
     1.445305721320277,
    -0.5900435899266435
)

"""
    sh_coeffs_to_color(::Val{SH_DEGREE}, sh_coeffs, direction) where SH_DEGREE

Convert the (1, 4, 9 or 16) spherical harmonics coefficients `sh_coeffs` into the
corresponding RGB color for the specified `normalized_direction`.

# Arguments
- `SH_DEGREE`: The maximal spherical harmonics degree. Only `0 <= SH_DEGREE <= 3` is
    supported.
- `sh_coeffs`: An array of `length` `(sh_degree + 1)^2 * 3` containing the coordinates of
    the viewpoint-dependent RGB function. Can be 1D or 2D of `size` `(..., 3)`.
- `normalized_direction`: A `length`-3 vector containing the viewing direction. Its
    Euclidean norm must be 1. Not used when `sh_degree == 0`.

# Output
The computed RGB color, as an `SVector{3}`. It is clamped from below by 0, but not clamped
from above. I.e. its components can exceed 1.
"""
function sh_coeffs_to_color(::Val{SH_DEGREE}, sh_coeffs, normalized_direction) where SH_DEGREE  
    I = typeof(SH_DEGREE)  # Int32 or Int64
    F = eltype(sh_coeffs)

    color = ntuple(Val(3)) do c
        # c (always Int64) is the channel index, and we output the color component cc
        # for this index

        MAX_SH_ORDER = (SH_DEGREE + one(I)) ^ 2  
        # == size(sh_coeffs, 1) if ndims(sh_coeffs) == 2
        # but known at compile-time
        # Slower (for some reason) if we bring it out of the loop

        c_offset = (I(c) - one(I)) * MAX_SH_ORDER
        # 1D indexing using c_offset is more efficient than 2D indexing (if applicable)
        # as size(sh_coeffs, 1) is not known at compile-time
        
        @inbounds cc = evaluate_sh_basis_function(one(I), normalized_direction, I, F) * 
                            sh_coeffs[one(I) + c_offset]
        @inbounds if SH_DEGREE >= 1  
        # Faster than a single
        #   @inbounds while i <= MAX_SH_ORDER; c += ...; i += one(I)
        # loop, or even its fully unrolled equivalent (for some reason)
            cc += (
                evaluate_sh_basis_function(I(2), normalized_direction, I, F) * 
                    sh_coeffs[I(2) + c_offset]
              + evaluate_sh_basis_function(I(3), normalized_direction, I, F) * 
                    sh_coeffs[I(3) + c_offset]
              + evaluate_sh_basis_function(I(4), normalized_direction, I, F) * 
                    sh_coeffs[I(4) + c_offset]
            )
            if SH_DEGREE >= 2
                cc += (
                    evaluate_sh_basis_function(I(5), normalized_direction, I, F) * 
                        sh_coeffs[I(5) + c_offset]
                  + evaluate_sh_basis_function(I(6), normalized_direction, I, F) * 
                        sh_coeffs[I(6) + c_offset]
                  + evaluate_sh_basis_function(I(7), normalized_direction, I, F) * 
                        sh_coeffs[I(7) + c_offset]
                  + evaluate_sh_basis_function(I(8), normalized_direction, I, F) * 
                        sh_coeffs[I(8) + c_offset]
                  + evaluate_sh_basis_function(I(9), normalized_direction, I, F) * 
                        sh_coeffs[I(9) + c_offset]
                )
                if SH_DEGREE >= 3
                    cc += (
                        evaluate_sh_basis_function(I(10), normalized_direction, I, F) * 
                            sh_coeffs[I(10) + c_offset]
                      + evaluate_sh_basis_function(I(11), normalized_direction, I, F) * 
                            sh_coeffs[I(11) + c_offset]
                      + evaluate_sh_basis_function(I(12), normalized_direction, I, F) * 
                            sh_coeffs[I(12) + c_offset]
                      + evaluate_sh_basis_function(I(13), normalized_direction, I, F) * 
                            sh_coeffs[I(13) + c_offset]
                      + evaluate_sh_basis_function(I(14), normalized_direction, I, F) * 
                            sh_coeffs[I(14) + c_offset]
                      + evaluate_sh_basis_function(I(15), normalized_direction, I, F) * 
                            sh_coeffs[I(15) + c_offset]
                      + evaluate_sh_basis_function(I(16), normalized_direction, I, F) * 
                            sh_coeffs[I(16) + c_offset]
                    )
                end
            end
        end
        # (We recompute evaluate_sh_basis_function(i, normalized_direction) for every c,
        #  but this turns out to be faster than a 3 x MAX_SH_ORDER order where we can use it
        #  for all color components at once)
        cc += F(0.5)
        if cc < 0
            cc = zero(F)
        end
        return cc
    end
    return SVector(color)
end

"""
    evaluate_sh_basis_function(i, direction, ::Type{I}, ::Type{F}) where {I, F}

Evaluate the `i`th spherical harmonics basis function in the supplied normalized
`direction`. We will use `I` for integer and `F` for floating point calculations.

When `direction` is `nothing` we always use the first basis function (which is viewpoint
indepentent, i.e. constant).
"""
@inline function evaluate_sh_basis_function(i, direction, ::Type{I}, ::Type{F}) where {I, F}
    @inbounds if i == 1 || isnothing(direction)
        return F(SH_CONSTS_F64[1])
        # (Don't (have to) use I for SH_CONSTS_F64 as this is known at compile-time.)
    else
        x, y, z = direction
        if i == 2; return -y * F(SH_CONSTS_F64[2]); end
        if i == 3; return  z * F(SH_CONSTS_F64[3]); end
        if i == 4; return -x * F(SH_CONSTS_F64[4]); end

        if i == 5; return                    x * y * F(SH_CONSTS_F64[5]); end
        if i == 6; return                    y * z * F(SH_CONSTS_F64[6]); end
        if i == 7; return (I(2) * z^2 - x^2 - y^2) * F(SH_CONSTS_F64[7]); end
        if i == 8; return                    x * z * F(SH_CONSTS_F64[8]); end
        if i == 9; return              (x^2 - y^2) * F(SH_CONSTS_F64[9]); end

        if i == 10; return                y * (I(3) * x^2 - y^2) * F(SH_CONSTS_F64[10]); end
        if i == 11; return                             x * y * z * F(SH_CONSTS_F64[11]); end
        if i == 12; return          y * (I(4) * z^2 - x^2 - y^2) * F(SH_CONSTS_F64[12]); end
        if i == 13; return z * (I(2) * z^2 - I(3) * (x^2 + y^2)) * F(SH_CONSTS_F64[13]); end
        if i == 14; return          x * (I(4) * z^2 - x^2 - y^2) * F(SH_CONSTS_F64[14]); end
        if i == 15; return                       z * (x^2 - y^2) * F(SH_CONSTS_F64[15]); end
        if i == 16; return                x * (x^2 - I(3) * y^2) * F(SH_CONSTS_F64[16]); end
        return zero(F)
    end
    # Reusing values like x^2 + y^2 does not make a performance difference
end

"""
    color_to_sh_coeffs_dc(color::SVector{3, F}) where F <: AbstractFloat

Convert an RGB color to its representation in the DC part (corresponding to
`sh_degree == 0`) of the spherical harmonics coefficients.
"""
function color_to_sh_coeffs_dc(color::SVector{3, F}) where F <: AbstractFloat
    @inbounds return (color .- F(0.5)) ./ F(SH_CONSTS_F64[1])
end

"""
    transposed_color_normalized_direction_jacobian(
        color, normalized_direction, sh_coeffs, ::Val{SH_DEGREE},
    ) where SH_DEGREE

Compute the transposed Jacobian of the function mapping the `normalized_direction`
into the corresponding `color` vector, using the spherical harmonics coefficients
`sh_coeffs`, which use the supplied spherical harmonics degree `SH_DEGREE`.

When `SH_DEGREE == 0` there is no viewpoint depencency, so we will return the appropriately
typed 0. In this case `normalized_direction` is not used, and can be set to `nothing`.
(We still use `color` to infer the output type. The rest of the parameters are also not
used.)
"""
function transposed_color_normalized_direction_jacobian(
    color, normalized_direction, sh_coeffs, ::Val{SH_DEGREE},
) where SH_DEGREE
    F = eltype(color)
    I = typeof(SH_DEGREE)
    SH_DEGREE == 0 && return zero(SMatrix{3, 3, F})

    partial_derivatives = ntuple(Val(3)) do c
        @inbounds(color[I(c)] <= 0) && return (zero(F), zero(F), zero(F))
        # color was clamped by the ReLU --> no contributions

        MAX_SH_ORDER = (SH_DEGREE + one(I)) ^ 2
        c_offset = (I(c) - one(I)) * MAX_SH_ORDER
        x, y, z = normalized_direction 
        
        # (SH_DEGREE >= 1)
        @inbounds dcc_dy = -F(SH_CONSTS_F64[2]) * sh_coeffs[I(2) + c_offset]
        @inbounds dcc_dz =  F(SH_CONSTS_F64[3]) * sh_coeffs[I(3) + c_offset]
        @inbounds dcc_dx = -F(SH_CONSTS_F64[4]) * sh_coeffs[I(4) + c_offset]

        @inbounds if SH_DEGREE >= 2
            let s = F(SH_CONSTS_F64[5]) * sh_coeffs[I(5) + c_offset]
                # Only fetch from memory once
                dcc_dx += y * s
                dcc_dy += x * s
            end
            let s = F(SH_CONSTS_F64[6]) * sh_coeffs[I(6) + c_offset]
                dcc_dy += z * s
                dcc_dz += y * s
            end
            let s = F(SH_CONSTS_F64[7]) * sh_coeffs[I(7) + c_offset]
                dcc_dx -= I(2) * x * s
                dcc_dy -= I(2) * y * s
                dcc_dz += I(4) * z * s
            end
            let s = F(SH_CONSTS_F64[8]) * sh_coeffs[I(8) + c_offset]
                dcc_dx += z * s
                dcc_dz += x * s
            end
            let s = F(SH_CONSTS_F64[9]) * sh_coeffs[I(9) + c_offset]
                dcc_dx += I(2) * x * s
                dcc_dy -= I(2) * y * s
            end

            if SH_DEGREE >= 3
                let s = F(SH_CONSTS_F64[10]) * sh_coeffs[I(10) + c_offset]
                    dcc_dx += I(6) * x * y * s
                    dcc_dy += I(3) * (x^2 - y^2) * s
                end
                let s = F(SH_CONSTS_F64[11]) * sh_coeffs[I(11) + c_offset]
                    dcc_dx += y * z * s
                    dcc_dy += x * z * s
                    dcc_dz += x * y * s
                end
                let s = F(SH_CONSTS_F64[12]) * sh_coeffs[I(12) + c_offset]
                    dcc_dx -= I(2) * x * y * s
                    dcc_dy += (I(4) * z^2 - x^2 - I(3) * y^2) * s
                    dcc_dz += I(8) * y * z * s
                end
                let s = F(SH_CONSTS_F64[13]) * sh_coeffs[I(13) + c_offset]
                    dcc_dx -= I(6) * x * z * s
                    dcc_dy -= I(6) * y * z * s
                    dcc_dz += I(3) * (I(2) * z^2 - x^2 - y^2) * s
                end
                let s = F(SH_CONSTS_F64[14]) * sh_coeffs[I(14) + c_offset]
                    dcc_dx += (I(4) * z^2 - I(3) * x^2 - y^2) * s
                    dcc_dy -= I(2) * x * y * s
                    dcc_dz += I(8) * x * z * s
                end
                let s = F(SH_CONSTS_F64[15]) * sh_coeffs[I(15) + c_offset]
                    dcc_dx += I(2) * x * z * s
                    dcc_dy -= I(2) * y * z * s
                    dcc_dz += (x^2 - y^2) * s
                end
                let s = F(SH_CONSTS_F64[16]) * sh_coeffs[I(15) + c_offset]
                    dcc_dx += I(3) * (x^2 - y^2) * s
                    dcc_dy -= I(6) * x * y * s
                end
            end
        end
        return (dcc_dx, dcc_dy, dcc_dz)
    end
    # partial_derivatives = (
    #   (∂c₁/∂x, ∂c₁/∂y, ∂c₁/∂z), 
    #   (∂c₂/∂x, ∂c₂/∂y, ∂c₂/∂z), 
    #   (∂c₃/∂x, ∂c₃/∂y, ∂c₃/∂z)
    # )
    # Its entries are the rows of the Jacobian. Therefore, it contains the transposed
    # Jacobian in column-major order
    return SMatrix{3, 3}(((partial_derivatives)...)...)
    # (On the CPU   reinterpret(SMatrix{3, 3, F, 9}, partial_derivatives)  also works,
    # but seeminly not on the GPU)
end