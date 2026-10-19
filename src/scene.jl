# TODO: docs
"""
    Scene{
        I <: Integer, F <: AbstractFloat,
        GV <: AbstractGaussiansVector{<:Any, I, F}, 
        CV <: AbstractVector{<:AbstractCamera{I, F}}
    }

Struct representing the combination of a GaussiansVector and a cameras vector.
"""
struct Scene{
    I <: Integer, F <: AbstractFloat,
    GV <: AbstractGaussiansVector{<:Any, I, F}, 
    CV <: AbstractVector{<:AbstractCamera{I, F}}
}
    gaussians::GV
    cameras::CV

    center::SVector{3, F}
    radius::F
    # Could (re)compute, but let's cache it to be slightly more efficient
end

"""
    Scene(gaussians::GV, cameras::CV) where {
        I <: Integer, F <: AbstractFloat,
        GV <: AbstractGaussiansVector{<:Any, I, F},
        CV <: AbstractVector{<:AbstractCamera{I, F}}
    }

Construct a `Scene` from its constituent vectors of Gaussians and cameras. 
"""
function Scene(gaussians::GV, cameras::CV) where {
    I <: Integer, F <: AbstractFloat,
    GV <: AbstractGaussiansVector{<:Any, I, F},
    CV <: AbstractVector{<:AbstractCamera{I, F}}
}
    center = _compute_scene_center(cameras)
    radius = _compute_scene_radius(cameras, center)
    return Scene{I, F, GV, CV}(gaussians, cameras, center, radius)
end

"""
    _compute_scene_center(cameras::AbstractVector{<:AbstractCamera})

Compute the center of the camera positions. This will serve as the center of a scene
with these cameras.
"""
_compute_scene_center(cameras::AbstractVector{<:AbstractCamera}) = 
    mean(get_position, cameras)

"""
    _compute_scene_radius(
        cameras::AbstractVector{<:AbstractCamera{<:Any, F}}, 
        scene_center,
        enlargement_factor = F(1.1)
    ) where F

Compute an enlarged radius of the scene as determined by the camera positions. By 'radius'
we mean the maximal distance from any camera to the `scene_center`. This parameter should
contain the 3D coordinates of the scene center. It can also be `nothing`, in which case we
use [`_compute_scene_center`](@ref).
"""
function _compute_scene_radius(
    cameras::AbstractVector{<:AbstractCamera{<:Any, F}}, 
    scene_center, 
    enlargement_factor = F(1.1)
) where F
    isnothing(scene_center) && ( scene_center = _compute_scene_center(cameras) )
    return enlargement_factor * maximum(cameras) do cam
        norm(get_position(cam) .- scene_center)
    end
end

@inline get_gaussians_vector(scene::Scene) = scene.gaussians
@inline get_cameras_vector(scene::Scene) = scene.cameras
@inline get_center(scene::Scene) = scene.center
@inline get_radius(scene::Scene) = scene.radius

function Base.show(
    io::IO, ::MIME"text/plain", 
    scene::Scene{I, F, <:AbstractGaussiansVector{D, I, F}}
) where {D, I, F}
    gv = scene.gaussians
    nb_g = length(gv)
    cv = scene.cameras
    nb_c = length(cv)
    print(io, 
        "Scene consisting of\n",
        "  $nb_g Gaussian$(_s_when_not_1(nb_g)) ($(typeof_noparams(gv)){$D, $I, $F})\n",
        "  and $nb_c camera$(_s_when_not_1(nb_c)) ($(typeof(cv)))"
    )
end


# TODO: Constructor taking the 3DGS style base directory (--> RenderOnly?)
#       Maybe also a COLMAP version? --> MutableTrainable?

"""
    create_test_scene(;
        sh_degree = 0, two_cameras = false, 
        array_domain = GPU, int_type = Int32, float_type = Float32
    )

Create a simple test scene consisting of a `MutableTrainableGaussiansVector` sort of
resembling the Julia logo. It consists of four Gaussians: [1]: a red one, [2]: a white
one, [3]: a green one, and [4]: a purple one. The white Gaussian is behind the camera(s), so
should not be rendered. The other ones are arranged as in the Julia logo, but overlapping
and of non-circular ellipsoidal shapes. In order of ascending depth we have white, green,
red, purple. The spherical harmonics degree is `sh_degree` (which will be converted to
`int_type` and can be 0, 1, 2 or 3). When `sh_degree > 0` the viewpoint dependent components
will still be set to ±0.01 (i.e. small, but non zero), where the sign alternates for each
Gaussian.

As cameras vector we use either a singleton of a simple `320 x 240` pinhole camera with 90°
horizontal field of view (`!two_cameras`), or this camera and an identical camera translated
over one unit in the x-direction (`two_cameras`). Note that in the former situation the
scene radius is zero. The world coordinates and camera coordinates are the same.

The keyword arguments determine the parameter types of the `MutableTrainableGaussiansVector`
and `PinholeCamera`.
"""
function create_test_scene(;
    sh_degree = 0, two_cameras = false, 
    array_domain = GPU, int_type = Int32, float_type = Float32
)
    D = array_domain; I = int_type; F = float_type
    
    w = I(320); h = I(240)
    f = fov_deg_to_focal_length(F(90), w)
    camera = PinholeCamera(
        w, h,
        f,
        one(SMatrix{3, 3, F}), zero(SVector{3, F})
    )
    K_inv = inv(get_intrinsic_matrix(camera))

    means2d = F[
        w/3    w/2 - 0.1   w/2 + 0.1   2w/3    
        2h/3   h/2 + 0.1   h/3         2h/3    
    ]
    # Break the symmetry to ensure predictable gradients
    # red, white, green, purple
    depths = F[1, -1, 2, 3]

    means = K_inv * [means2d .* depths'; depths']
    opacity_features = logit.(F[0.75, 0.99, 0.5, 0.99])
    scaling_features = log.(F[
        0.25   10   0.5   1.
        0.25   20   0.5   0.75
        0.25   30   0.5   0.5
    ])
    rotation_features = F[
        sqrt(2)/2   1.   1.   sqrt(3)
        0.          0.   0.   1.
        0.          0.   0.   0.
        sqrt(2)/2   0.   0.   0.
    ]
    # 90° around z-axis, no rotation (x2), 60° around x-axis
    # The last one is explicitly not normalized
    max_sh_order = (sh_degree + 1) ^ 2
    sh_coeffs = [
        color_to_sh_coeffs_dc(@SVector F[0.796, 0.235, 0.2  ])'; 
            fill(F(0.01), max_sh_order - 1, 3);;;
        color_to_sh_coeffs_dc(@SVector F[1.,    1.,    1.   ])'; 
            fill(F(-0.01), max_sh_order - 1, 3);;;
        color_to_sh_coeffs_dc(@SVector F[0.22,  0.596, 0.149])'; 
            fill(F(0.01), max_sh_order - 1, 3);;;
        color_to_sh_coeffs_dc(@SVector F[0.584, 0.345, 0.698])'; 
            fill(F(+0.01), max_sh_order - 1, 3)
    ]

    gaussians = MutableTrainableGaussiansVector(
        D(means),
        D(opacity_features),
        D(scaling_features),
        D(rotation_features),
        D(sh_coeffs);
        int_type
    )

    cameras = if two_cameras
        SVector{2}(camera, move(camera, SVector{3, F}(1, 0, 0)))
    else
        SVector{1}(camera)
    end
    return Scene(gaussians, cameras)
end