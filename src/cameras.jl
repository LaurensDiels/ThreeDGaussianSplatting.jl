"""
    AbstractCamera{
        I <: Integer, F <: AbstractFloat
    }

Abstract supertype for cameras. Represents integers using `I` and floats using `F`.
Concrete child types should implement the methods `get_width`, `get_height` and 
`project_with_depth` (where this method should be efficient). The projection produces
continuous pixel coordinates.
"""
abstract type AbstractCamera{
    I <: Integer, F <: AbstractFloat
} end

@inline int_type(::AbstractCamera{I}) where I = I
@inline float_type(::AbstractCamera{<:Any, F}) where F = F
# It would be safer to use these functions, rather than where constructs,
# in case we ever change the parameter structure. 
# TODO (though low priority)

"""
    PinholeCamera{
        I <: Integer, 
        F <: AbstractFloat, 
    } <: AbstractCamera{I, F}

A pinhole camera defined by two focal lengths, a skew (typically 0), a principal point
and near and far clipping planes. Does not have any distortion.

---

    PinholeCamera(
        width::I, height::I,
        focal_length_x::F, focal_length_y::F,
        skew::F,
        principal_point_x::F, principal_point_y::F,
        near_z::F, far_z::F
        camera_to_world_rotation::SMatrix{3, 3, F},
        camera_to_world_translation::SVector{3, F},
    ) where {
        I <: Integer,
        F <: AbstractFloat,
    }

Construct a `PinholeCamera` from the screen dimensions, the intrinsic components
(in `PixelCoordinates`), near and far clipping planes, and the pose.

The near clipping value is exclusive, the far one inclusive.
"""
struct PinholeCamera{
    I <: Integer, 
    F <: AbstractFloat,
} <: AbstractCamera{I, F}

    width::I
    height::I
    
    focal_length_x::F
    focal_length_y::F
    skew::F
    principal_point_x::F
    principal_point_y::F
    # More compact than storing the entire 3 x 3 intrinsic matrix

    near_z::F
    far_z::F

    camera_to_world_rotation::SMatrix{3, 3, F, 9}
    camera_to_world_translation::SVector{3, F}

    projection_matrix_3x4::SMatrix{3, 4, F, 12}
    # Cached for efficiency

    function PinholeCamera(
        width::I, height::I,
        focal_length_x::F, focal_length_y::F,
        skew::F,
        principal_point_x::F, principal_point_y::F,
        near_z::F, far_z::F,
        camera_to_world_rotation::SMatrix{3, 3, F},
        camera_to_world_translation::SVector{3, F}
    ) where {
        I <: Integer,
        F <: AbstractFloat
    }
        new{I, F}(
            width, height,
            focal_length_x, focal_length_y,
            skew,
            principal_point_x, principal_point_y,
            near_z::F, far_z::F,
            camera_to_world_rotation, camera_to_world_translation,

            _get_projection_matrix_3x4(
                _create_intrinsic_matrix(
                    focal_length_x, focal_length_y, 
                    skew, 
                    principal_point_x, principal_point_y
                ),
                _get_world_to_camera_transformation_3x4(
                    camera_to_world_rotation, camera_to_world_translation
                )
            )
        )
    end
end
# Note that this is always isbits (for reasonable I, F)

"""
    _create_intrinsic_matrix(
        focal_length_x::F, focal_length_y,
        skew,
        principal_point_x, principal_point_y
    ) where F

Construct the normalized 3x3 intrinsic `SMatrix`, given the provided components in px.
"""
function _create_intrinsic_matrix(
    focal_length_x::F, focal_length_y,
    skew,
    principal_point_x, principal_point_y
) where F

    return @SMatrix [
        focal_length_x  skew                principal_point_x
        0               focal_length_y    principal_point_y
        0               0                   1
    ]
end


"""
    _get_world_to_camera_transformation_3x4(
        camera_to_world_rotation,
        camera_to_world_translation
    )

Combine the provided camera to world rotation and translation into a world to camera 3x4
transformation matrix.
"""
function _get_world_to_camera_transformation_3x4(
    camera_to_world_rotation,
    camera_to_world_translation
)

    world_to_camera_rotation, world_to_camera_translation = 
        _switch_camera_world(camera_to_world_rotation, camera_to_world_translation)

    return [world_to_camera_rotation  world_to_camera_translation]
end

"""
    _switch_camera_world(rotation, translation)

Switch between camera to world and world to camera. Return a new rotation and translation.
"""
function _switch_camera_world(rotation, translation)
    R = rotation'
    t = -R * translation

    return R, t
end

"""
     _get_projection_matrix_3x4(
        intrinsic_matrix,
        world_to_camera_transformation_3x4
    )

Combine the provided 3x4 intrinsic matrix and 3x4 world to camera transformation matrix
into the 3x4 projection matrix. Uses the same 2D continuous image coordinates as
`intrinsic_matrix`.
"""
function _get_projection_matrix_3x4(
    intrinsic_matrix,
    world_to_camera_transformation_3x4
)
    return intrinsic_matrix * world_to_camera_transformation_3x4
end


"""
    CameraWorldOrder

Abstract supertype of `CameraToWorld` and `WorldToCamera`.
"""
abstract type CameraWorldOrder end

"""
    CameraToWorld

For camera to world coordinate transformations.
"""
struct CameraToWorld <: CameraWorldOrder end

"""
    WorldToCamera

For world to camera coordinate transformations (poses).
"""
struct WorldToCamera <: CameraWorldOrder end

"""
    PinholeCamera(
        width, height,
        focal_length_x, focal_length_y,
        skew,
        principal_point_x, principal_point_y,
        near_z, far_z,
        rotation, translation;
        rotation_camera_world_order::Type{<:CameraWorldOrder} = CameraToWorld,
        translation_camera_world_order::Type{<:CameraWorldOrder} = 
            rotation_camera_world_order
    )

Construct a `PinholeCamera` from the screen dimensions, the intrinsic parameters for pixel
units, near (exclusive: z > near) and far (inclusive: z <= far) clipping planes, and 
extrinsics (`camera_to_world` or `world_to_camera`).
"""
function PinholeCamera(
    width, height,
    focal_length_x, focal_length_y,
    skew,
    principal_point_x, principal_point_y,
    near_z, far_z,
    rotation, translation;
    rotation_camera_world_order::Type{<:CameraWorldOrder} = CameraToWorld,
    translation_camera_world_order::Type{<:CameraWorldOrder} = rotation_camera_world_order
)

    R_c2w, t_c2w = _convert_to_world_to_camera(
        rotation, translation,
        rotation_camera_world_order, translation_camera_world_order
    )

    I = promote_type(typeof(width), typeof(height))
    F = promote_type(
        typeof(focal_length_x), typeof(focal_length_y), typeof(skew),
        typeof(principal_point_x), typeof(principal_point_y),
        typeof(near_z), typeof(far_z),
        eltype(rotation), eltype(translation)
    )
    
    I <: Integer || throw(ArgumentError(
        "A `PinholeCamera`'s `width` and `height` need to have an integer type! \
         Received $I as promotion type"
    ))
    F <: AbstractFloat || throw(ArgumentError(
        "A `PinholeCamera`'s continuous parameters (such as the focal length, or its
        position) must be floats. Received $F as promotion type"
    ))
    
    return PinholeCamera(
        I(width), I(height),
        F(focal_length_x), F(focal_length_y),
        F(skew),
        F(principal_point_x), F(principal_point_y),
        F(near_z), F(far_z),
        convert(SMatrix{3, 3, F}, R_c2w), convert(SVector{3, F}, t_c2w),
    )
    # Make sure we call the internal constructor. If the types were not correct, we would
    # end up calling ourselves, resulting in a stack overflow.
end


"""
    _convert_to_world_to_camera(
        rotation, translation,
        ::Type{rotation_camera_world_order},
        ::Type{translation_camera_world_order}
    ) where {
        rotation_camera_world_order <: CameraWorldOrder, 
        translation_camera_world_order <: CameraWorldOrder
    }

Convert the supplied 3x3 `rotation` matrix and `length`-3 `translation` vector in
camera to world, or world to camera order, as determined by `rotation_camera_world_order`
and `translation_camera_world_order`, into their equivalents in world to camera order.
Returns (rotation_world_to_camera, translation_world_to_camera).
"""
function _convert_to_world_to_camera(
    rotation, translation,
    ::Type{rotation_camera_world_order},
    ::Type{translation_camera_world_order}
) where {
    rotation_camera_world_order <: CameraWorldOrder, 
    translation_camera_world_order <: CameraWorldOrder
}
    if rotation_camera_world_order === CameraToWorld
        R_c2w = rotation
    elseif rotation_camera_world_order === WorldToCamera
        R_c2w = rotation'
    # There should not be any third option, but if there somehow would be, we will throw
    # an error as R_c2w won't be defined.
    end

    if translation_camera_world_order === CameraToWorld
        t_c2w = translation
    elseif translation_camera_world_order === WorldToCamera
        t_c2w = -R_c2w * translation
    end

    return R_c2w, t_c2w
end

"""
    PinholeCamera(
        width, height,
        intrinsic_matrix,
        near_z, far_z,
        rotation, translation;
        rotation_camera_world_order::Type{<:CameraWorldOrder} = CameraToWorld,
        translation_camera_world_order::Type{<:CameraWorldOrder} = 
            rotation_camera_world_order
    )

Construct a `PinholeCamera` from the screen dimensions, the 3x3 intrinsic matrix
(using px), near (exclusive: z > near) and far (inclusive: z <= far) clipping 
planes, and extrinsics (`camera_to_world` or `world_to_camera`),.
"""
function PinholeCamera(
    width, height,
    intrinsic_matrix,
    near_z, far_z,
    rotation, translation;
    rotation_camera_world_order::Type{<:CameraWorldOrder} = CameraToWorld,
    translation_camera_world_order::Type{<:CameraWorldOrder} = rotation_camera_world_order
)

    R_c2w, t_c2w = _convert_to_world_to_camera(
        rotation, translation,
        rotation_camera_world_order, translation_camera_world_order
    )

    return PinholeCamera(
        width, height,
        intrinsic_matrix[1, 1], intrinsic_matrix[2, 2],
        intrinsic_matrix[1, 2],
        intrinsic_matrix[1, 3], intrinsic_matrix[2, 3],
        near_z, far_z,
        R_c2w, t_c2w
    )
end

const DEFAULT_NEAR_CLIPPING_PLANE_F64 = 0.2
const DEFAULT_FAR_CLIPPING_PLANE_F64 = Inf64

"""
    PinholeCamera(
        width, height,
        focal_length,
        rotation, translation;
        rotation_camera_world_order::Type{<:CameraWorldOrder} = CameraToWorld,
        translation_camera_world_order::Type{<:CameraWorldOrder} = 
            rotation_camera_world_order
    )

Construct a `PinholeCamera` from the screen dimensions, the `focal_length` (the same for
both axes) and extrinsics (`camera_to_world` or `world_to_camera`). We assume that there is
no skew and that the principle point is at the center of the image plane. For the near and
far clipping planes we use the standard 3DGS values.
"""
function PinholeCamera(
    width, height,
    focal_length,
    rotation, translation;
    rotation_camera_world_order::Type{<:CameraWorldOrder} = CameraToWorld,
    translation_camera_world_order::Type{<:CameraWorldOrder} = rotation_camera_world_order
)
    F = eltype(rotation)
    return PinholeCamera(
        width, height,
        focal_length, focal_length,
        zero(F),
        width / F(2), height / F(2),
        F(DEFAULT_NEAR_CLIPPING_PLANE_F64), F(DEFAULT_FAR_CLIPPING_PLANE_F64),
        rotation, translation;
        rotation_camera_world_order, translation_camera_world_order
    )
end


"""
    load_cameras_from_json(
        json_path;
        int_type::Type{<:Integer} = Int32,
        float_type::Type{<:AbstractFloat} = Float32,
        rescale_factor = one(float_type),
        forced_width = nothing,
        forced_height = nothing
    )

Load in a 3DGS cameras.json file, or one produced using [`save_json`](@ref), returning a
`Vector{PinholeCamera{int_type, float_type}}.` The entries do not contain references to
ground truth images, so can only be used for rendering, not training. See
[`load_cameras_with_images_from_json`](@ref) if training is desired.

Optionally rescales the image planes, so that the new dimensions are the original ones
multiplied by `rescale_factor` (and rounded). Alternatively (and with higher priority),
you can also forcibly rescale the camera width and height using `forced_width` and
`forced_height`. When set to `nothing` we do not rescale or force the resolution.
"""
function load_cameras_from_json(
    json_path;
    int_type::Type{<:Integer} = Int32,
    float_type::Type{<:AbstractFloat} = Float32,
    rescale_factor = nothing,
    forced_width = nothing,
    forced_height = nothing
)
    return map(JSON.parsefile(json_path)) do json_cam
        json_camera_entry_to_pinhole_camera(
            json_cam,
            int_type, float_type,
            rescale_factor, forced_width, forced_height
        )
    end
end

"""
    json_camera_entry_to_pinhole_camera(json_cam;
        int_type::Type{<:Integer} = Int32,
        float_type::Type{<:AbstractFloat} = Float32,
        rescale_factor = one(float_type),
        forced_width = nothing,
        forced_height = nothing
    )

Convert a single json camera entry to a PinholeCamera. See [`load_cameras_from_json`](@ref).
"""
function json_camera_entry_to_pinhole_camera(json_cam,
    int_type, float_type,
    rescale_factor, forced_width, forced_height
)
    I = int_type; F = float_type

    width = if isnothing(forced_width)
        if isnothing(rescale_factor)  
            # (or we could in this case set rescale_factor = one(F))
            I(json_cam.width)
        else
            round(I, json_cam.width * rescale_factor)
        end
    else
        I(forced_width)
    end
    height = if isnothing(forced_height)
        if isnothing(rescale_factor)
            I(json_cam.height)
        else
            round(I, json_cam.height * rescale_factor)
        end
    else
        I(forced_height)
    end

    scale_x = width / F(json_cam.width)
    # (Note that when relying on rescale_factor, the effective scaling factor might
    #  be slightly different from it, due to rounding)
    scale_y = height / F(json_cam.height)

    focal_length_x = F(json_cam.fx) * scale_x
    focal_length_y = F(json_cam.fy) * scale_y

    skew = if :skew in keys(json_cam)
        # Not present in original 3DGS, always 0. But we do include it in save_json
        skew = F(json_cam.skew) * scale_x
    else  
        skew = zero(F)
    end
    # (Or: skew = get(json_cam, :skew, zero(F)) * scale_x  )
    ppx = if :ppx in keys(json_cam)
        F(json_cam.ppx) * scale_x
    else
        F(width) / 2  # centered principal point
    end
    ppy = if :ppy in keys(json_cam)
        F(json_cam.ppy) * scale_x
    else
        F(height) / 2
    end
    near_z = get(json_cam, :near_z, F(DEFAULT_NEAR_CLIPPING_PLANE_F64))
    far_z  = get(json_cam, :far_z,  F(DEFAULT_FAR_CLIPPING_PLANE_F64))

    R_c2w = SMatrix{3, 3, F}(stack(json_cam.rotation, dims=1))
    # json_cam.rotation contains the camera to world rotation as a vector of rows
    t_c2w = SVector{3, F}(json_cam.position)

    return PinholeCamera(
        width, height,
        focal_length_x, focal_length_y,
        skew,
        ppx, ppy,             
        near_z, far_z,
        R_c2w, t_c2w,
    )
end

"""
    save_json(path, cams::AbstractVector{<:PinholeCamera})

Save a vector of `PinholeCamera`s to a file specified by the `path`. The result can be read
back using [`load_cameras_from_json`](@ref).
"""
save_json(path, cams::AbstractVector{<:PinholeCamera}) = 
    JSON.json(path, cams; pretty = true)

function JSON.lower(cam::PinholeCamera)
# Converts a PinholeCamera to a namedtuple representation
    return (;
        width    = get_width(cam),
        height   = get_height(cam),
        fx       = get_focal_length_x(cam),
        fy       = get_focal_length_y(cam),
        skew     = get_skew(cam),
        ppx      = get_principal_point_x(cam),
        ppy      = get_principal_point_y(cam),
        position = get_position(cam),
        rotation = get_camera_to_world_rotation(cam)'  
        # ' for row-major, becomes vector or rows
    )
end


"""
    fov_to_focal_length(fov_rad, axis_size)

Convert the supplied (horizontal or vertical) field of view (actually angle of view) in
radians to the corresponding focal length, given the `axis_size` (width or height). The
produced focal length will use the same units as the `axis_size` (typically px).

Inverse of [`focal_length_to_fov`](@ref).
"""
@inline fov_to_focal_length(fov_rad, axis_size) = axis_size / (2 * tan(fov_rad / 2))

"""
    fov_deg_to_focal_length(fov_deg, axis_size)

Like [`fov_to_focal_length`](@ref) but with the FOV in degrees.
"""
@inline fov_deg_to_focal_length(fov_deg, axis_size) = 
    fov_to_focal_length(deg2rad(fov_deg), axis_size)

"""
    focal_length_to_fov(focal_length, axis_size)

Convert the supplied `focal_length` into the corresponding (horizontal or vertical) field of
view (actually angle of view), given the `axis_size` (width or height). The same units
(typically px) should be used for `focal_length` and `axis_size`.

Inverse of [`fov_to_focal_length`](@ref).
"""
@inline focal_length_to_fov(focal_length, axis_size) = 
    2 * atan(axis_size / (2 * focal_length))

"""
    focal_length_to_fov_deg(focal_length, axis_size)

Like [`focal_length_to_fov`](@ref) but with the FOV in degrees.
"""
@inline focal_length_to_fov_deg(focal_length, axis_size) = 
    rad2deg(focal_length_to_fov(focal_length, axis_size))

@inline get_width(cam::PinholeCamera) = cam.width
@inline get_height(cam::PinholeCamera) = cam.height
@inline Base.size(cam::PinholeCamera) = (cam.height, cam.width)
# following the standard matrix indexing convention

@inline get_focal_length_x(cam::PinholeCamera) = cam.focal_length_x
@inline get_fov_x(cam::PinholeCamera) = 
    focal_length_to_fov(get_focal_length_x(cam), get_width(cam))
@inline get_fov_x_deg(cam::PinholeCamera) = rad2deg(get_fov_x(cam))
@inline get_focal_length_y(cam::PinholeCamera) = cam.focal_length_y
@inline get_fov_y(cam::PinholeCamera) = 
    focal_length_to_fov(get_focal_length_y(cam), get_height(cam))
@inline get_fov_y_deg(cam::PinholeCamera) = rad2deg(get_fov_y(cam))
@inline get_skew(cam::PinholeCamera) = cam.skew
@inline get_principal_point_x(cam::PinholeCamera) = cam.principal_point_x
@inline get_principal_point_y(cam::PinholeCamera) = cam.principal_point_y
@inline get_principal_point(cam::PinholeCamera) = 
    SA[get_principal_point_x(cam), get_principal_point_y(cam)]
@inline get_intrinsic_matrix(cam::PinholeCamera) = 
    _create_intrinsic_matrix(
        get_focal_length_x(cam), get_focal_length_y(cam),
        get_skew(cam),
        get_principal_point_x(cam), get_principal_point_y(cam)
    )

@inline get_near_z(cam::PinholeCamera) = cam.near_z
get_near_clipping_plane = get_near_z
@inline get_far_z(cam::PinholeCamera) = cam.far_z
get_far_clipping_plane = get_far_z

@inline get_camera_to_world_rotation(cam::PinholeCamera) = cam.camera_to_world_rotation
@inline get_camera_to_world_translation(cam::PinholeCamera) = 
    cam.camera_to_world_translation
@inline get_position(cam::PinholeCamera) = get_camera_to_world_translation(cam)

@inline get_camera_to_world_transformation_3x4(cam::PinholeCamera) =
    [get_camera_to_world_rotation(cam)  get_camera_to_world_translation(cam)]
@inline get_camera_to_world_transformation_4x4(cam::PinholeCamera) = 
    vcat(get_camera_to_world_transformation_3x4(cam), SA[0 0 0 1])

@inline get_world_to_camera_rotation(cam::PinholeCamera) = 
    transpose(get_camera_to_world_rotation(cam))
@inline get_world_to_camera_translation(cam::PinholeCamera) = 
    -get_world_to_camera_rotation(cam) * get_camera_to_world_translation(cam)
@inline get_world_to_camera_transformation_3x4(cam::PinholeCamera) = 
    _get_world_to_camera_transformation_3x4(
        get_camera_to_world_rotation(cam),
        get_camera_to_world_translation(cam)
    )
@inline get_world_to_camera_transformation_4x4(cam::PinholeCamera) = 
    vcat(get_world_to_camera_transformation_3x4(cam), SA[0 0 0 1])


@inline get_projection_matrix_3x4(cam::PinholeCamera) = 
    cam.projection_matrix_3x4

"""
    move(cam::PinholeCamera{<:Any, F}, xyz)

Move this `PinholeCamera` using coordinates `xyz` in its local reference system (x: right,
y: down, z: forward). Typically `xyz` will be a `StaticVector{3, F}`, though we do not
enforce this in the signature. When a longer vector is provided, we only take the first
three entries. Shorter vectors will lead to unexpected behavior and crashes.

For example, `xyz == (0., 0., 1.)` will move the camera forward over one unit.

Returns a new `PinholeCamera`.
"""
@inline function move(cam::PinholeCamera{<:Any, F}, xyz) where F
    R = get_camera_to_world_rotation(cam)
    @inbounds return PinholeCamera(
        get_width(cam), get_height(cam),
        get_focal_length_x(cam), get_focal_length_y(cam),
        get_skew(cam),
        get_principal_point_x(cam), get_principal_point_y(cam),
        get_near_z(cam), get_far_z(cam),
        R,
        get_camera_to_world_translation(cam) .+ R * SVector{3, F}(xyz[1], xyz[2], xyz[3])
        # Note that xyz contains the coordinates of the origin of the new camera in the old
        # camera system. Therefore, this is a new to old translation. Then, as 4x4 matrices,
        # new_camera_to_world = old_camera_to_world * new_camera_to_old_camera
    )
end

"""
    rotate(cam::PinholeCamera{<:Any, F}, R::StaticMatrix{3, 3, F})

Rotate this `PinholeCamera` by the provided rotation in its local reference system. For
`R` you can in particular use Rotations.jl's `Rotation{3}`, including `AngleAxis` and
`QuatRotation`. For example, `R == AngleAxis(rad2deg(30), 0, 1, 0)` rotates the camera 30°
to the right.
"""
@inline function rotate(cam::PinholeCamera{<:Any, F}, R) where F
    return PinholeCamera(
        get_width(cam), get_height(cam),
        get_focal_length_x(cam), get_focal_length_y(cam),
        get_skew(cam),
        get_principal_point_x(cam), get_principal_point_y(cam),
        get_near_z(cam), get_far_z(cam),
        get_camera_to_world_rotation(cam) * R,
        get_camera_to_world_translation(cam)
    )
end

# TODO: add tests
# TODO: add setters for focal length(s), resolution, ...

"""
    get_depth(point, cam::PinholeCamera)

Return the depth of the provided `3D` point for this camera.
"""
@inline function get_depth(point, cam::PinholeCamera)
    P = get_projection_matrix_3x4(cam)
    return dot(P[3, SOneTo(3)], point) .+ P[3, 4]
end

"""
    check_depth(depth::AbstractFloat, cam::PinholeCamera)

Check whether the provided `depth` lies between the near and far clipping planes of this
camera.
"""
@inline function check_depth(depth::AbstractFloat, cam::PinholeCamera)
    return get_near_z(cam) < depth <= get_far_z(cam)
end

"""
    check_depth(point::AbstractVector, cam::PinholeCamera)

Check whether the provided 3D `point`'s lies between the near and far clipping planes of
this camera.
"""
@inline function check_depth(point::AbstractVector, cam::PinholeCamera)
    return check_depth(get_depth(point, cam), cam)
end

"""
    project_with_depth(
        point,
        cam::PinholeCamera
    )
    -> Tuple{SVector{2}, <:AbstractFloat}

Project the 3D `point` into 2D continuous pixel coordinates for this `PinholeCamera`.
Also return the depth of the point.
"""
@inline function project_with_depth( 
    point, 
    cam::PinholeCamera
)
    P = get_projection_matrix_3x4(cam)
    proj_homogeneous = P[SOneTo(3), SOneTo(3)] * point .+ P[SOneTo(3), 4]
    depth = proj_homogeneous[3]
    return proj_homogeneous[SOneTo(2)] ./ depth, depth
end

"""
    project(point, cam::AbstractCamera)

Project the 3D `point` into continuous 2D pixel coordinates (without returning the depth of
the point, see [`project_with_depth`](@ref)).
"""
@inline function project(point, cam::AbstractCamera)
    return project_with_depth(point, cam)[1]
end

"""
    get_linearized_projection_3d(point, cam::AbstractCamera)

Return the 2 x 3 Jacobian at `point` of the map transforming z 3D point `pt` into the 2D
`project(pt, cam)`. 
"""
@inline function get_linearized_projection_3d(point, cam::AbstractCamera)
    proj_point, depth = project_with_depth(point, cam)
    return get_linearized_projection_2d(proj_point, depth, cam)
end

"""
    get_linearized_projection_2d(
        proj_point, 
        depth, 
        cam::AbstractCamera{<:Any, F}
    ) where F

Like [`get_linearized_projection_3d`](@ref), but the 3D point where we take the Jacobian is
not defined via its 2D projected `proj_point` and `depth`. 
"""
@inline function get_linearized_projection_2d(
    proj_point, 
    depth, 
    cam::AbstractCamera{<:Any, F}
) where F
    idepth = inv(depth)
    return ScaledSMatrix(
        idepth,
        SMatrixWithZeros(
            SVector{4}(one(F), one(F), -proj_point[1], -proj_point[2]),
            Tuple{2, 3},
            Val(((1, 2), (2, 1)))
        )
    ) * get_projection_matrix_3x4(cam)[SOneTo(3), SOneTo(3)]
    # project = g ∘ f with f: x ↦ P[1:3, 1:3] * x + P[1:3, 4] (with J(f) = P[1:3, 1:3])
    # and g(a, b, c) = (a / c, b / c)
    # The first matrix is
    # SA[idepth  zero(F)  -proj_point[1] * idepth
    #    zero(F) idepth   -proj_point[2] * idepth]
end
# TODO: export?

function Base.show(io::IO, ::MIME"text/plain", cam::PinholeCamera{I, F}) where {I, F}
    print(io, 
        "PinholeCamera{$I, $F}:", "\n",
        "  size (width × height): ", get_width(cam), " × ", get_height(cam), "\n",
        "  intrinsic matrix: ", get_intrinsic_matrix(cam), "\n",
        "  near clipping plane: ", get_near_clipping_plane(cam), "\n",
        "  far clipping plane: ", get_far_clipping_plane(cam), "\n",
        "  camera to world rotation: ", get_camera_to_world_rotation(cam), "\n",
        "  camera to world translation: ", get_camera_to_world_translation(cam), "\n",
        "    (a.k.a. position)", "\n",
        "    --> world to camera rotation: ", get_world_to_camera_rotation(cam), "\n",
        "        world to camera translation: ", get_world_to_camera_translation(cam)
    )
end

# Note: transform to clipping space, with
# [-1, 1]^3 and positive z convention would be
# [ 2 f_x / w   0           2 c_x / w - 1       0
#               2 f_y / h   2 c_y / h - 1       0 
#               0           (f + n) / (f - n)   - 2 f n / (f - n)
#   0           0           1                   0]




"""
    AbstractImageLoader{D <: CPUAccessibleDomain, F <: AbstractFloat}

Abstract type for loading an image. Concrete structs should implement [`get_image`](@ref),
which returns an image with `eltype` `F` and `array_domain` `D`. Additionally, we also
expect a [`_should_free_after_get_image`](@ref) method.
"""
abstract type AbstractImageLoader{D <: CPUAccessibleDomain, F <: AbstractFloat} end

"""
    get_image(il::AbstractImageLoader{D, F}) where {D, F}

Return the image with `eltype` `F` and `array_domain` `D` that this ImageLoader loads.
"""
function get_image end

"""
    _should_free_after_get_image(::Type{<:AbstractImageLoader})

Indicate whether one should (or is allowed to) call `maybe_unsafe_free!` on the image
produces from [`get_image`](@ref). This is the case when we return, but do not internally
store, a GPU version.
"""
@inline _should_free_after_get_image(::Type{<:AbstractImageLoader}) = false
# Structs should implement the method when not

"""
    with_image(f, il::IL) where IL <: AbstractImageLoader

Call `f` on the result of [`get_image`](@ref) and `maybe_unsafe_free!` the result
afterwards, if [`_should_free_after_get_image`](@ref).
"""
@inline function with_image(f, il::IL) where IL <: AbstractImageLoader
    img = get_image(il)
    ret = f(img)
    _should_free_after_get_image(IL) && maybe_unsafe_free!(img)
    return ret
end
# (@inline as this is potentially a short-running, relatively frequently used method, so we
#  want to avoid overhead)

"""
    load_image(::Type{D}, ::Type{F}, path)

Load an image from disk, where the file is located at `path`. The image will have
`array_domain` `D` and `eltype` `F` (e.g. `Float32`, not `RGB{N0f8}`).
"""
load_image(::Type{D}, ::Type{F}, path) where {
    D <: CPUAccessibleDomain, F <: AbstractFloat
} = D(array_convert(F, channelview(load(path))))

"""
    LazyImageLoader{
        D <: CPUAccessibleDomain, F <: AbstractFloat
    } <: AbstractImageLoader{D, F}

`AbstractImageLoader` which loads from disk every time. When `D <: GPU` it might be prudent
to call `maybe_unsafe_free!` on the produced image from [`get_image`](@ref) when you are
done using it. Or equivalently, use [`with_image`](@ref) instead.

Uses the least memory possible, but might be slower due to repeated loading.

---

    LazyImageLoader{
        D <: CPUAccessibleDomain, F <: AbstractFloat
    }(
        path::String
    )

Construct a `LazyImageLoader{D, F}` from the supplied `path`
"""
struct LazyImageLoader{
    D <: CPUAccessibleDomain, F <: AbstractFloat
} <: AbstractImageLoader{D, F}
    path::String
end
# (Note: not isbits, so cannot be used in kernels, but that's fine)

"""
    LazyImageLoader(::Type{D}, ::Type{F}, path)

Construct a `LazyImageLoader{D, F}` from the supplied `path`.
"""
LazyImageLoader(::Type{D}, ::Type{F}, path) where {D, F} = LazyImageLoader{D, F}(path)

get_image(lil::LazyImageLoader{D, F}) where {D, F} = load_image(D, F, lil.path)
@inline _should_free_after_get_image(::Type{<:LazyImageLoader{<:GPU}}) = true

"""
    LazyCachedImageLoader{
        D <: CPUAccessibleDomain, F <: AbstractFloat, A <: AbstractArray{F}
    } <: AbstractImageLoader{D, F}

`AbstractImageLoader` which defers loading the image from disk until the first
[`get_image`](@ref) call. The image is (if necessary) converted to `array_domain` `D` and
stored, so that the next `get_image` call is faster.

This uses more memory than `LazyImageLoader` but should be faster. Using `D == CPU`
you can trade some performance for less VRAM usage (when we later upload to the GPU 
just in time).
    
---

    LazyCachedImageLoader{D, F, A}(
        path::String, image_ref::Base.RefValue{A}
    ) where {D, F, A <: AbstractArray{F}}

Construct a `LazyCachedImageLoader` from its parameters and fields. Internal constructor
which performs an `array_domain` compatibility check. Should not be used directly.
"""
struct LazyCachedImageLoader{
    D <: CPUAccessibleDomain, F <: AbstractFloat, A <: AbstractArray{F}
} <: AbstractImageLoader{D, F}
    path::String
    image_ref::Base.RefValue{A}

    function LazyCachedImageLoader{D, F, A}(
        path::String, image_ref::Base.RefValue{A}
    ) where {D <: CPUAccessibleDomain, F, A <: AbstractArray{F}}
        the_D = D == GPU ? GPU{CUDACore.DeviceMemory} : D
        # With GPU get_image would work, but the comparison below might fail
        DA = array_domain(A)
        the_D === DA || throw(ArgumentError(
            "A `LazyCachedImageLoader`'s `ArrayDomain` and its stored image ('s type) \
            `ArrayDomain` must be the same! Received: $the_D and $DA"
        ))
        return new{the_D, F, A}(path, image_ref)
    end
end

"""
    LazyCachedImageLoader(::Type{A}, path) where {
        F <: AbstractFloat, A <: AbstractArray{F}
    }

Construct a `LazyCachedImageLoader` from the storage array type `A`, and image `path`. 
"""
function LazyCachedImageLoader(::Type{A}, path) where {
    F <: AbstractFloat, A <: AbstractArray{F}
}
    return LazyCachedImageLoader{array_domain(A), F, A}(path, Base.RefValue{A}())
end

"""
    LazyCachedImageLoader(
        ::Type{D}, ::Type{F}, ndims::Int, path
    ) where {
        D <: CPUAccessibleDomain, F <: AbstractFloat
    }

Construct a `LazyCachedImageLoader` from the image `array_domain` `D`, the image `eltype`
`F`, the number of image array dimensions `ndims` (usually 3: channels, vertical,
horizontal), and the image `path`. 
"""
function LazyCachedImageLoader(
    ::Type{D}, ::Type{F}, ndims::Int, path
) where {
    D <: CPUAccessibleDomain, F <: AbstractFloat
}
    if D <: GPU
        M = if D == GPU
            CUDACore.DeviceMemory
        else
            D.parameters[1]
        end
        A = CuArray{F, ndims, M}
    else
        A = Array{F, ndims}
    end
    return LazyCachedImageLoader(A, path)
end

# Already documented by virtual AbstractImageLoader version and the `LazyCachedImageLoader`
# documentation
@inline function get_image(
    lcil::LazyCachedImageLoader{D, F}
) where {D, F} 
    if isassigned(lcil.image_ref)
        return lcil.image_ref[]
    else
        image = load_image(D, F, lcil.path)
        lcil.image_ref[] = image
        return image
    end
end

# _should_free_after_get_image can remain at false

"""
    maybe_unsafe_free!(lcil::LazyCachedImageLoader{<:GPU}

Call `maybe_unsafe_free!` on the cached image (if any). 
"""
function maybe_unsafe_free!(lcil::LazyCachedImageLoader{<:GPU})
    if isassigned(lcil.image_ref)
        maybe_unsafe_free!(lcil.image_ref[])
    end
end


"""
    EagerImageLoader{
        D <: CPUAccessibleDomain,
        F <: AbstractFloat, A <: AbstractArray{F}
    } <: AbstractImageLoader{D, F}

`AbstractImageLoader` which immediately loads the image from disk in the constructors.
Apart from this, behaves like a `LazyCachedImageLoader`.

---

    EagerImageLoader{D, F, A}(
        path::String, image::A
    ) where {
        D <: CPUAccessibleDomain, F <: AbstractFloat, A <: AbstractArray{F}
    }

Construct a `EagerImageLoader` from its parameters and fields. Internal constructor
which performs an `array_domain` compatibility check. Should not be used directly.
"""
struct EagerImageLoader{
    D <: CPUAccessibleDomain, F <: AbstractFloat, A <: AbstractArray{F}
} <: AbstractImageLoader{D, F}
    path::String
    image::A

    function EagerImageLoader{D, F, A}(
        path::String, image::A
    ) where {
        D <: CPUAccessibleDomain, F <: AbstractFloat, A <: AbstractArray{F}
    }
        the_D = D == GPU ? GPU{CUDACore.DeviceMemory} : D
        D_image = array_domain(image)
        the_D === D_image || throw(ArgumentError(
            "An `EagerImageLoader`'s `ArrayDomain` and its image's `ArrayDomain` must be \
            the same! Received: $the_Dand $D_image"
        ))
        return new{the_D, F, A}(path, image)
    end
end

"""
    EagerImageLoader(
        path::String, image::A
    ) where {
        F <: AbstractFloat, A <: AbstractArray{F}
    }

Construct a `EagerImageLoader` from the image `path` and `image` array itself.  
"""
function EagerImageLoader(
    path::String, image::A
) where {
    F <: AbstractFloat, A <: AbstractArray{F}
}
    return EagerImageLoader{array_domain(A), F, A}(path, image)
end

"""
    EagerImageLoader(
        ::Type{D}, ::Type{F}, path
    ) where {
        D <: CPUAccessibleDomain, F <: AbstractFloat
    }

Construct a `EagerImageLoader` from the image `array_domain` `D`, the image `eltype` `F`,
and the image `path`.
"""
function EagerImageLoader(::Type{D}, ::Type{F}, path) where {
    D <: CPUAccessibleDomain, F <: AbstractFloat
}
    image = load_image(D, F, path)
    return EagerImageLoader(path, image)
end

@inline get_image(eil::EagerImageLoader) = eil.image
# Again, should never free (until we're completely done, in which case you can use the
# maybe_unsafe_free! below)

"""
    maybe_unsafe_free!(eil::EagerImageLoader{<:GPU})

Call `maybe_unsafe_free!` on the loaded image.
"""
maybe_unsafe_free!(eil::EagerImageLoader{<:GPU}) = maybe_unsafe_free!(eil.image)

"""
    CameraWithImage{
        D <: ArrayDomain, I <: Integer, F <: AbstractFloat, 
        C <: AbstractCamera{I, F}, 
        L <: AbstractImageLoader{D, F}
    }

Struct consisting of a `camera::C` and `image_loader::L`. This represents a training camera,
including ground truth image.
"""
struct CameraWithImage{
    D <: ArrayDomain, I <: Integer, F <: AbstractFloat, 
    C <: AbstractCamera{I, F}, 
    L <: AbstractImageLoader{D, F}
}
    camera::C
    image_loader::L
end
# Not isbits; must extract the camera for render!

@inline get_camera(cwi::CameraWithImage) = cwi.camera

"""
    get_image(cwi::CameraWithImage)

Call [`get_image`](@ref) on the image loader.
"""
@inline get_image(cwi::CameraWithImage) = get_image(cwi.image_loader)

"""
    with_image(cwi::CameraWithImage)

Call [`with_image`](@ref) on the image loader.
"""
@inline with_image(f, cwi::CameraWithImage) = with_image(f, cwi.image_loader)

"""
    load_cameras_with_images_from_json(
        json_path;
        image_loader_type::Type{<:AbstractImageLoader} = LazyCachedImageLoader,
        image_domain::Type{<:CPUAccessibleDomain} = CPU,
        float_type::Type{<:AbstractFloat} = Float32,
        int_type::Type{<:Integer} = Int32,
    )

Load in our custom training cameras json file, returning a 
`Vector{CameraWithImage{
    PinholeCamera{int_type, float_type}, 
    image_loader_type{image_domain, float_type}}.`

`image_loader_type` should be the struct name without parameters. When these are supplied,
they will be ignored, in favor of `image_domain` and `float_type`.
By default we use `LazyCachedImageLoader{CPU, Float32}`, meaning
after the first `get_image` calls we store the images in Float32 format on the CPU. They can
then later by uploaded to the GPU when needed, one by one. This assumes we have plenty of
RAM, but limited VRAM.
"""
function load_cameras_with_images_from_json(
    json_path;
    image_loader_type::Type{<:AbstractImageLoader} = LazyCachedImageLoader,
    image_domain::Type{<:CPUAccessibleDomain} = CPU,
    float_type::Type{<:AbstractFloat} = Float32,
    int_type::Type{<:Integer} = Int32,
)
    D = image_domain; I = int_type; F = float_type

    return map(JSON.parsefile(json_path)) do json_cam
        cam = json_camera_entry_to_pinhole_camera(json_cam, I, F, nothing, nothing, nothing)

        il = if image_loader_type <: LazyImageLoader
            LazyImageLoader(D, F, json_cam.path)
        elseif image_loader_type <: LazyCachedImageLoader
            LazyCachedImageLoader(D, F, 3, json_cam.path)
            # Assumed to be an RGB image
        elseif image_loader_type <: EagerImageLoader
            EagerImageLoader(D, F, json_cam.path)
        else
            throw(ArgumentError(
                "The only supported `image_loader_type`s are `LazyImageLoader`, \
                 `LazyCachedImageLoader` and `EagerImageLoader`! \
                 Received: $image_loader_type"))
        end
        return CameraWithImage(cam, il)
    end
end

"""
    save_json(path, cwis::AbstractVector{<:CameraWithImage})

Save a vector of `CameraWithImage`s to a file specified by the `path`. The result can be
read back using [`load_cameras_with_images_from_json`](@ref).
"""
save_json(path, cwis::AbstractVector{<:CameraWithImage}) = 
    JSON.json(path, cwis; pretty = true)
# (Except for the documentation (and types) this is identical to the other save_json)

function JSON.lower(cwi::CameraWithImage{<:Any, <:Any, <:Any, <:PinholeCamera})
    return (;
        JSON.lower(cwi.camera)...,
        path = cwi.image_loader.path
    )
end