# Immutable only: resize creates a new one, possibly sharing memory with the old
# TODO: why not just (@im_)mutable? (use resize_memory then)

# TODO: also make RGB image optional, but require at least one of the three
#   When using a GUI and showing e.g. depth, we do not need to render the RGB image
"""
    RenderOutputs{
        D <: ArrayDomain,
        I <: Integer,
        F <: AbstractFloat,
        RGB <: AbstractArray{F, 3},
        TRANSMITTANCE <: Union{Nothing, <:AbstractMatrix{F}},
        DEPTH <: Union{Nothing, <:AbstractMatrix{F}},
        LAST_IDS <: Union{Nothing, <:AbstractMatrix{I}}
    }

Struct specifying which outputs to render. Will also contain these outputs (or `nothing`
if we do not use them). The outputs are
- the RGB render (always included)
- the transmittance image (optional)
- the depth image (optional)
- an 'image' containing for each pixel the index of the last contributing sorted duplicated
    filtered Gaussian index (optional).
"""
struct RenderOutputs{
    D <: ArrayDomain,
    I <: Integer,
    F <: AbstractFloat,
    # Always contains RGB image, others are optional
    RGB <: AbstractArray{F, 3},
    TRANSMITTANCE <: Union{Nothing, <:AbstractMatrix{F}},
    DEPTH <: Union{Nothing, <:AbstractMatrix{F}},
    LAST_IDS <: Union{Nothing, <:AbstractMatrix{I}}
}
    rgb_image::RGB                          # 3 x h x w
    transmittance_image::TRANSMITTANCE      #     h x w
    depth_image::DEPTH                      #     h x w
    last_contributing_sdfgids::LAST_IDS     #     h x w

    """
        RenderOutputs(
            rgb_image::RGB,  
            transmittance_image::TRANSMITTANCE,
            depth_image::DEPTH,
            last_contributing_sdfgids::LAST_IDS;
            int_type = Int32
        ) where {
            I <: Integer,
            F <: AbstractFloat,
            RGB <: AbstractArray{F, 3},
            TRANSMITTANCE <: Union{Nothing, <:AbstractMatrix{F}},
            DEPTH <: Union{Nothing, <:AbstractMatrix{F}},
            LAST_IDS <: Union{Nothing, <:AbstractMatrix{I}}
        }

    Construct a `RenderOutput` from the supplied arrays or `nothing`s. The keyword argument
    `int_type` (defaulting to `Int32`) is only used when `LAST_IDS` is `Nothing`, hence
    contains no integer type.
    """
    function RenderOutputs(
        rgb_image::RGB,  
        transmittance_image::TRANSMITTANCE,
        depth_image::DEPTH,
        last_contributing_sdfgids::LAST_IDS;
        int_type = Int32
    ) where {
        I <: Integer,
        F <: AbstractFloat,
        RGB <: AbstractArray{F, 3},
        TRANSMITTANCE <: Union{Nothing, <:AbstractMatrix{F}},
        DEPTH <: Union{Nothing, <:AbstractMatrix{F}},
        LAST_IDS <: Union{Nothing, <:AbstractMatrix{I}}
    }
        arrays = filter(!isnothing,
            (rgb_image, transmittance_image, depth_image, last_contributing_sdfgids)
        )
        array_domains = map(array_domain, arrays)
        allequal(array_domains) || throw(ArgumentError(
            "All (non-`nothing`) images in a `RenderOutputs` need to have the same \
             `ArrayDomain`. Received $(join(array_domains, ' ', " and "))."
        ))
        D = first(array_domains)

        last_two_axis_sizes = map(arrays) do a
            size(a)[end-1:end]
        end
        allequal(last_two_axis_sizes) || throw(DimensionMismatch(
            "All (non-`nothing`) images in a `RenderOutputs` need to have the same spatial \
             `size`. Received $(join(last_two_axis_sizes, ' ', " and "))."
        ))

        size(rgb_image, 1) == 3 || throw(DimensionMismatch(
            "The first axis of `rgb_image` should have `size` `3`! \
             Received $(size(rgb_image, 1))."
        ))

        if LAST_IDS === Nothing
            the_I = int_type
        else
            the_I = I
        end

        return new{D, the_I, F, RGB, TRANSMITTANCE, DEPTH, LAST_IDS}(
            rgb_image, transmittance_image, depth_image, last_contributing_sdfgids
        )
    end
end

Adapt.@adapt_structure RenderOutputs

"""
    RenderOutputs(
        width, height;
        array_domain::Type{<:ArrayDomain} = GPU{CUDA.DeviceMemory},
        int_type::Type{<:Integer} = Int32,
        float_type::Type{<:AbstractFloat} = Float32,
        include_transmittance = true,
        include_depth = true,
        include_last_contributing_indices = true,
    )

Create a new `RenderOutputs` with the provided `width`, `height`, `array_domain` (default: 
`GPU`), `int_type` (default: `Int32`, regardless of the types of `width` and `height`),
`float_type` (default: `Float32`).

Apart from the always present rendered RGB image, the selected render outputs will be
included (all by default).
"""
function RenderOutputs(
    width, height;
    array_domain::Type{<:ArrayDomain} = GPU{CUDA.DeviceMemory},
    int_type::Type{<:Integer} = Int32,
    float_type::Type{<:AbstractFloat} = Float32,
    include_transmittance = true,
    include_depth = true,
    include_last_contributing_indices = true,
)
    D = array_domain; I = int_type; F = float_type
    return RenderOutputs(
        create_array(D, F, (3, height, width)),
        include_transmittance             ? create_array(D, F, (height, width)) : nothing,
        include_depth                     ? create_array(D, F, (height, width)) : nothing,
        include_last_contributing_indices ? create_array(D, I, (height, width)) : nothing;
        int_type
    )
end

"""
    RenderOutputs(
        cam::AbstractCamera{I, F};
        array_domain::Type{<:ArrayDomain} = GPU{CUDA.DeviceMemory},
        include_transmittance = true,
        include_depth = true,
        include_last_contributing_indices = true
    ) where {I, F}

Like the `width`, `height` constructor, but now extracting this information, as well as the
`int_type` and `float_type` from the provided camera.
"""
function RenderOutputs(
    cam::AbstractCamera{I, F};
    array_domain::Type{<:ArrayDomain} = GPU{CUDA.DeviceMemory},
    include_transmittance = true,
    include_depth = true,
    include_last_contributing_indices = true
) where {I, F}
    return RenderOutputs(
        get_width(cam), get_height(cam);
        array_domain, int_type = I, float_type = F,
        include_transmittance, include_depth, include_last_contributing_indices
    )
end



@inline Base.size(ro::RenderOutputs) = size(ro.rgb_image)[2:3]
@inline get_width(ro::RenderOutputs) = size(ro.rgb_image)[3]
@inline get_height(ro::RenderOutputs) = size(ro.rgb_image)[2]

@inline has_rgb_image(::RenderOutputs) = true
@inline get_rgb_image(ro::RenderOutputs) = ro.rgb_image
@inline has_transmittance_image(ro::RenderOutputs) = typeof(ro).parameters[5] !== Nothing
# Could also dispatch on the fifth type, i.e.
#   ...(::RenderOutputs) = true
# and
#   ...(::RenderOutputs{..., Nothing}) = false  
# but this is quite complicated as the former implicitly contains all the relations between
# types, which our second method should also include to be more specific (i.e. not just 
# <: Any)
# Also, isnothing(ro.transmittance_image) gets computed at compile-time, so that is an
# alternative.
@inline get_transmittance_image(ro::RenderOutputs) = ro.transmittance_image
@inline has_depth_image(ro::RenderOutputs) = typeof(ro).parameters[6] !== Nothing
@inline get_depth_image(ro::RenderOutputs) = ro.depth_image
@inline has_last_contributing_sdfgids(ro::RenderOutputs) = 
    typeof(ro).parameters[7] !== Nothing
@inline get_last_contributing_sdfgids(ro::RenderOutputs) = ro.last_contributing_sdfgids


"""
    maybe_unsafe_free!(::RenderOutputs)

Call `maybe_unsafe_free!` on outputs. When the `ArrayDomain <: GPU` we then `unsafe_free!`
all non-`nothing` outputs. If not, nothing happens.
"""
maybe_unsafe_free!(::RenderOutputs) = nothing
function maybe_unsafe_free!(ro::RenderOutputs{<:GPU})
    maybe_unsafe_free!(get_rgb_image(ro))
    maybe_unsafe_free!(get_transmittance_image(ro))
    maybe_unsafe_free!(get_depth_image(ro))
    maybe_unsafe_free!(get_last_contributing_sdfgids(ro))
    return nothing
end

"""
    resize!(ro::RenderOutputs{D, I, F}, new_size; free = D <: GPU) where {D, I, F}

Create a _new_ `RenderOutputs` with the provided size. If this new size is smaller than the
old, we will reuse the memory from the old `RenderOutputs`. Consequently, the old one
should no longer be used, hence the `!` (but if `!free` it is technically still possible).

When the new size is the same as the old, we just return `ro` again (and do not free
anything).

# Arguments
- `ro`: The input `RenderOutputs`.
- `new_size`: The `size` (height, width) of the output `RenderOutputs`.
- `free`: Whether the memory of the input `RenderOutputs` should be marked as freeable.
    Defaults to `true` if the array domain is `GPU`, and `false` otherwise (in which case
    we cannot free anything anyways).
    When set to `true` the input `RenderOutputs` becomes unusable. (When `new_size` is
    smaller than the old size, the part of the memory we reuse is of course still
    accessible). 
"""
function resize!(ro::RenderOutputs{D, I, F}, new_size; free = D <: GPU) where {D, I, F}
    size(ro) == new_size && return ro

    new_rgb_image = resize_memory(ro.rgb_image, (3, new_size...); free)
    new_transmittance_image = if has_transmittance_image(ro)
        resize_memory(ro.transmittance_image, new_size; free)
    else
        nothing
    end
    new_depth_image = if has_depth_image(ro)
        resize_memory(ro.depth_image, new_size; free)
    else
        nothing
    end
    new_last_contributing_sdfgids = if has_last_contributing_sdfgids(ro)
        resize_memory(ro.last_contributing_sdfgids, new_size; free)
    else
        nothing
    end

    return RenderOutputs(
        new_rgb_image,
        new_transmittance_image,
        new_depth_image,
        new_last_contributing_sdfgids
    )
end

"""
    resize!(ro::RenderOutputs{D}, cam::AbstractCamera; free = D <: GPU) where {D}

Like the `new_size` version, but deriving this from the provided camera.
"""
function resize!(ro::RenderOutputs{D}, cam::AbstractCamera; free = D <: GPU) where {D}
    return resize!(ro, (get_height(cam), get_width(cam)), free)
end


function Base.show(
    io::IO, 
    ::MIME"text/plain", 
    ro::RenderOutputs{D, I, F}
) where {D, I, F}
    print(io, 
        "RenderOutputs{$D, $I, $F} of size ", size(ro), ", with \n",
        "  an rgb image \n",
        "  ", has_transmittance_image(ro) ? "a " : "no ", "transmittance image \n",
        "  ", has_depth_image(ro) ? "a " : "no ", "depth image \n",
        "  ", has_last_contributing_sdfgids(ro) ? "" : "no ", "last contributing indices"
    )
end



"""
    Tiling{
        I <: Integer, 
        TILE_WIDTH,
        TILE_HEIGHT,
    }

Struct containing tiling information:
- The width of each tile (of type `I`, using pixel coordinates).
- The height of each tile.
- The number of tiles (of type `I`), along the horizontal axis (derived from the screen
    size).
- The number of tiles, along the vertical axis (derived from the screen size).
"""
struct Tiling{
    I <: Integer,  
    TILE_WIDTH, 
    TILE_HEIGHT,
}
    nb_tiles_x::I
    nb_tiles_y::I
end

@inline get_nb_tiles_x(t::Tiling) = t.nb_tiles_x
@inline get_nb_tiles_y(t::Tiling) = t.nb_tiles_y
@inline get_nb_tiles(t::Tiling) = get_nb_tiles_x(t) * get_nb_tiles_y(t)

@inline get_tile_width(t::Tiling) = typeof(t).parameters[2]
@inline get_tile_height(t::Tiling) = typeof(t).parameters[3]
@inline get_pxs_per_tile(t::Tiling) = get_tile_width(t) * get_tile_height(t)


"""
    Tiling(
        screen_width, screen_height,
        tile_width, tile_height;
        int_type::Type{<:Integer} = typeof(screen_width)
    )

Create a `Tiling{I, ...}` from the provided screen and tile dimensions in pixels. For `I`
we use the provided `int_type`, defaulting to the type of `screen_width`.
"""
function Tiling(
    screen_width, screen_height,
    tile_width, tile_height;
    int_type::Type{<:Integer} = typeof(screen_width)
)
    I = int_type    
    return Tiling{
        I, I(tile_width), I(tile_height), 
    }(
        cld(I(screen_width), I(tile_width)), 
        cld(I(screen_height), I(tile_height))
    )
end

"""
    Tiling(
        camera::AbstractCamera{I}, 
        tile_width, tile_height
    ) where I

Construct a `Tiling{I, ...}` from the provided camera, and the tile dimensions in pixels.
"""
function Tiling(
    camera::AbstractCamera{I}, 
    tile_width, tile_height
) where I
    return Tiling(
        get_width(camera), get_height(camera), 
        I(tile_width), I(tile_height))
end

"""
    Tiling(camera::AbstractCamera{I}) where I

Construct a `Tiling{I, ...}` from the provided camera and default (heuristic) values
for the tile dimensions.
"""
function Tiling(camera::AbstractCamera{I}) where I
    screen_width = get_width(camera)
    screen_height = get_height(camera)
    nb_px = screen_width * screen_height
    tile_width = I(16)
    tile_height = nb_px > 1_500_000 ? I(32) : I(16)
    return Tiling(screen_width, screen_height, tile_width, tile_height)
end

# The function below are trivial. We just add them to ensure we don't accidentally mix up
# conventions. We use y-contiguous order (like column-major).

"""
    tile_ids_to_linear_idx(t::Tiling{I}, x_idx, y_idx) where I

Convert a 2D tile index into a linear index. This uses x-contiguous order.
(Empirically this works faster than the column-major y-contiguous order, possibly
because 3D Gaussians are more naturally horizontally sorted than vertical.)
"""
@inline function tile_ids_to_linear_idx(t::Tiling{I}, x_idx, y_idx) where I
    return  _tile_x_idx_and_y_contribution_to_linear_idx( 
        x_idx,
        _tile_y_idx_to_linear_idx_contribution(t, y_idx))
end

"""
    _tile_y_idx_to_linear_idx_contribution(t::Tiling{I}, y_idx) where I

Convert the y-part of a 2D tile index into its contribution into the linear index.
"""
@inline function _tile_y_idx_to_linear_idx_contribution(t::Tiling{I}, y_idx) where I
    return (y_idx - one(I)) * get_nb_tiles_x(t)
end

"""
    _tile_x_idx_and_y_contribution_to_linear_idx(x_idx, y_contribution)

Combine the x-part of a 2D tile index and the y-contribution into the linear tile index into 
its the complete linear index.
"""
@inline function _tile_x_idx_and_y_contribution_to_linear_idx(x_idx, y_contribution)
    return x_idx + y_contribution
end

"""
    linear_tile_idx_to_tile_ids(t::Tiling, idx)

Convert a linear tile index to a 2D index `(x_idx, y_idx)`.
"""
@inline function linear_tile_idx_to_tile_ids(t::Tiling, idx)
    return reverse(fldmod1(idx, get_nb_tiles_x(t)))
end


"""
    point_to_tile_idx(point, t::Tiling{I}) where I

Find the 2D discrete tile index of the tile containing the provided 2D point (in continuous
or discrete pixel coordinates). The top-left screen pixel lies in tile index (1, 1).
"""
@inline function point_to_tile_idx(point, t::Tiling{I}) where I
    # Tile i (1-indexed) is ((i - 1) * tile_width, i * tile_width]
    return ceil.(I, point ./ SVector{2}(get_tile_width(t), get_tile_height(t)))
end



"""
    tile_idx_to_rectangle(t::Tiling{I}, tile_x, tile_y) where I

Convert the tile with index `(tile_x, tile_y)` into the corresponding `Rectangle{I}`.
The left and top side are exclusive, the bottom and right inclusive. The indices `tile_x`
and `tile_y` are expected to contain integer values.
(Fractional indices are also possible, but have less meaning, as this is not the point of
tiling.)
"""
function tile_idx_to_rectangle end

@inline function tile_idx_to_rectangle(t::Tiling{I}, tile_x, tile_y) where I
    tile_w = get_tile_width(t); tile_h = get_tile_height(t)
    right = tile_x * tile_w  # (inclusive)
    bottom = tile_y * tile_h
    return Rectangle{F}(
        right - tile_w,    # (exclusive)
        bottom - tile_h,  # i.e. (tile_y - one(I)) * tile_h
        right,
        bottom
        )
end

"""
    rectangle_to_tile_ids(
        r::Rectangle, t::Tiling{I}, 
        ::Val{clmp} = Val(true)
    ) where {I, clmp}

Find the `Rectangle{I}` of tile indices containing the `Rectangle` `r`. When `clmp` (by
default) we will clamp these indices to the boundaries. E.g. non-positive indices will be
set to 1. We assume that `r` uses the same coordinate system as `t`.
"""
@inline function rectangle_to_tile_ids(
    r::Rectangle, t::Tiling{I}, 
    ::Val{clmp} = Val(true)
) where {I, clmp}
    tl_tile = point_to_tile_idx(get_top_left(r), t)
    br_tile = point_to_tile_idx(get_bottom_right(r), t)
    if clmp
        tl_tile = max.(tl_tile, (I(1), I(1)))
        br_tile = min.(br_tile, (get_nb_tiles_x(t), get_nb_tiles_y(t)))
    end
    return Rectangle(tl_tile, br_tile)  # {I}
end

"""
    BoundaryClippingMode

Abstract supertype for strategies to deal with 3D Gaussians whose 2D projection lies too
far out of the image plane.
"""
abstract type BoundaryClippingMode end

"""
    CovarianceBoundaryClamp <: BoundaryClippingMode

`BoundaryClippingMode` where for the purpose of computing the 2D covariance matrix we clamp
the 2D mean to an extended image plane. This is the approach used in 3DGS's reference
implementation. It is more stable for training.
"""
struct CovarianceBoundaryClamp <: BoundaryClippingMode end

"""
    BoundaryFilter <: BoundaryClippingMode

`BoundaryClippingMode` where we filter out Gaussians if the 2D mean falls too far out of the
image plane. Faster, but worse for training: novel views (outside of the training set)
will have reduced quality.
"""
struct BoundaryFilter <: BoundaryClippingMode end

# cf. 3DGS
const DEFAULT_MIN_ALPHA_F64 = 1 / 255
const DEFAULT_MAX_ALPHA_F64 = 0.99              
const DEFAULT_MIN_TRANSMITTANCE_F64 = 0.0001
# all inclusive

const DEFAULT_DILATION_PX_F64 = 0.3
const DEFAULT_CLIPPING_MARGIN_NC_F64 = 0.3
const DEFAULT_MIN_AA_FACTOR_F64 = 0.0005

"""
    RenderOptions{
        I <: Integer, F <: AbstractFloat, 
        SH_DEGREE, 
        BOUNDARY_CLIPPING_MODE <: BoundaryClippingMode
    }

Struct containing both static (spherical harmonics degree, boundary clipping mode) and
runtime (background_color, antialiasing, minimal and maximal alpha, minimal transmittance,
dilation terms (in continuous pixel coordinates), clipping values (in the same coordinate
system)) options for rendering.
"""
struct RenderOptions{
    I <: Integer, F <: AbstractFloat, 
    SH_DEGREE,  # of type I
    BOUNDARY_CLIPPING_MODE <: BoundaryClippingMode
}
    # Put options which we do not intent to change (much) as static parameters,
    # only when we expect performance improvements (e.g. when used in loop boundaries).
    background_color::SVector{3, F}
    # TODO: would we not similarly need a background depth?
    
    use_antialiasing::Bool
    antialiasing_min_factor::F

    min_alpha::F  # as opacity >= alpha, this also serves as min_opacity
    max_alpha::F
    min_transmittance::F

    dilation_x::F
    dilation_y::F

    clip_x_min::F
    clip_x_max::F
    clip_y_min::F
    clip_y_max::F
end

"""
    RenderOptions(
        camera::AbstractCamera{I, F}, 
        sh_degree,
        boundary_clipping_mode::Type{<:BoundaryClippingMode} = CovarianceBoundaryClamp;
        background_color = zero(SVector{3, F}),
        use_antialiasing = false
    ) where {I, F}

Construct a `RenderOptions` from a `camera`, spherical harmonics degree, background color,
and the choice of whether to use (3DGS-style) antialiasing (which is more limited than
Mip-Splatting), and default values. 
"""
function RenderOptions(
    camera::AbstractCamera{I, F}, 
    sh_degree,
    boundary_clipping_mode::Type{<:BoundaryClippingMode} = CovarianceBoundaryClamp;
    background_color = zero(SVector{3, F}),
    use_antialiasing = false
) where {I, F}
    CLIPPING_MARGIN = F(DEFAULT_CLIPPING_MARGIN_NC_F64)
    width = get_width(camera); height = get_height(camera)
    n2p_x = create_coordinate_converter(NormalizedCoordinates, PixelCoordinates, F, width)
    n2p_y = create_coordinate_converter(NormalizedCoordinates, PixelCoordinates, F, height)
    return RenderOptions{
        I, F, I(sh_degree), boundary_clipping_mode
    }( 
        background_color,
        use_antialiasing,
        F(DEFAULT_MIN_AA_FACTOR_F64),
        F(DEFAULT_MIN_ALPHA_F64), 
        F(DEFAULT_MAX_ALPHA_F64),
        F(DEFAULT_MIN_TRANSMITTANCE_F64),
        F(DEFAULT_DILATION_PX_F64),
        F(DEFAULT_DILATION_PX_F64),
        n2p_x(-1 - CLIPPING_MARGIN), 
        n2p_x(1 + CLIPPING_MARGIN),
        n2p_y(-1 - CLIPPING_MARGIN),
        n2p_y(1 + CLIPPING_MARGIN)
    )
end

@inline int_type(::RenderOptions{I}) where I = I
@inline float_type(::RenderOptions{<:Any, F}) where F = F

@inline get_sh_degree(::RenderOptions{<:Any, <:Any, SH_DEGREE}) where SH_DEGREE = SH_DEGREE
@inline get_boundary_clipping_mode(ro::RenderOptions) = typeof(ro).parameters[4]

@inline get_min_alpha(ro::RenderOptions) = ro.min_alpha
@inline get_min_transmittance(ro::RenderOptions) = ro.min_transmittance
@inline get_max_alpha(ro::RenderOptions) = ro.max_alpha
@inline get_dilation_x(ro::RenderOptions) = ro.dilation_x
@inline get_dilation_y(ro::RenderOptions) = ro.dilation_y
@inline get_dilation(ro::RenderOptions) = SA[get_dilation_x(ro), get_dilation_y(ro)]
@inline get_background_color(ro::RenderOptions) = ro.background_color
@inline get_use_antialiasing(ro::RenderOptions) = ro.use_antialiasing
@inline get_antialiasing_min_factor(ro::RenderOptions) = ro.antialiasing_min_factor

"""
    set_sh_degree(ro::RenderOptions{I, F}, sh_degree) where {I, F}

Return a new `RenderOptions` with `get_sh_degree(...) == I(sh_degree)`.
"""
function set_sh_degree(ro::RenderOptions{I, F}, sh_degree) where {I, F}
    _check_sh_degree(sh_degree)
    return RenderOptions{I, F, I(sh_degree), get_boundary_clipping_mode(ro)}(
        (getfield(ro, fn) for fn in fieldnames(typeof(ro)))...
    )
end

"""
    lies_on_extended_screen(xy, ro::RenderOptions)

Check whether the 2D point `xy` (which must support indices [1] and [2]) lies on the
extended screen, as determined by the clipping values in `ro`.
"""
@inline function lies_on_extended_screen(xy, ro::RenderOptions)
    @inbounds return ro.clip_x_min < xy[1] <= ro.clip_x_max && 
                     ro.clip_y_min < xy[2] <= ro.clip_y_max
end

"""
    clamp_to_extended_screen(xy, ro::RenderOptions)

Clamp the 2D point `xy` to the extended screen, determined by the clipping values in `ro`.
"""
@inline function clamp_to_extended_screen(xy, ro::RenderOptions)
    return clamp.(xy, (ro.clip_x_min, ro.clip_y_min), (ro.clip_x_max, ro.clip_y_max))
end

const INVALID_LAST_CONTRIBUTING_SDFGIDX_I64 = 0

function render_void!(
    outputs::RenderOutputs{D, I, F}, 
    options::RenderOptions
) where {D, I, F}
    fill!(
        reinterpret(reshape, SVector{3, F}, get_rgb_image(outputs)),  # Matrix{SVector{3}}
        get_background_color(options))
    
    has_transmittance_image(outputs) && fill!(get_transmittance_image(outputs), one(F))
    has_depth_image(outputs) && fill!(get_depth_image(outputs), F(Inf))
    has_last_contributing_sdfgids(outputs) && 
        fill!(
            get_last_contributing_sdfgids(outputs), 
            I(INVALID_LAST_CONTRIBUTING_SDFGIDX_I64))
end

# Like for TrainableGaussiansVector, the documentation will be bundled later
# TODO: include nb_overlapped_tiles (maybe also cumsum; everything of length nb_fgaussians)
#   -> f2dg print: check if == 0: if so: invalid
@im_mutable_with_adapt(
    struct Filtered2DGaussiansVector{
        D <: ArrayDomain, I <: Integer, F <: AbstractFloat,
        GV <: AbstractGaussiansVector{D, I, F},
        FTG <: AbstractVector{I},
        Z <: AbstractVector{F},
        M2OICC <: AbstractMatrix{F},
        ITR <: AbstractArray{I, 3}
    }
        length::I           
        # Since the number of filtered Gaussians will constantly change, depending
        # on the render camera, we overallocate up to the upper bound of the number of
        # Gaussians. This means our arrays will have a larger capacity than the current
        # needed length.
        
        all_3d_gaussians::GV
        fgidx_to_gidx::FTG                                      # n (the capacity)

        depths::Z                                               # n
        means2d_opacities_inverse_covariances_colors::M2OICC    # 9 x n
        # Combining these noticeably helps for the tile rendering kernel

        intersected_tile_ranges::ITR                            # 2 x 2 x n
        # first axis: x/y, second: bounding box min/max, 
        # (Note that in memory we then use the order x_min, y_min, x_max, y_max)

        function Filtered2DGaussiansVector(
            len::I,
            all_3d_gaussians::GV, fgidx_to_gidx::FTG,
            depths::Z, means2d_opacities_inverse_covariances_colors::M2OICC,
            intersected_tile_ranges::ITR
        ) where {
            D <: ArrayDomain, I <: Integer, F <: AbstractFloat, 
            GV <: AbstractGaussiansVector{D, I, F},
            FTG <: AbstractVector{I},
            Z <: AbstractVector{F},
            M2OICC <: AbstractMatrix{F},
            ITR <: AbstractArray{I, 3},
        }
            type_str = "Filtered2DGaussiansVector"
            arrays = (
                fgidx_to_gidx, 
                depths, means2d_opacities_inverse_covariances_colors, 
                intersected_tile_ranges
            )
            D2 = _get_common_array_domain(type_str, arrays)
            D2 === D || throw(ArgumentError(
                "The arrays of a `$type_str` need to have the same `ArrayDomain` as the
                 parent `GaussiansVector`! Received: $D2 and $D."))
            capacity = _check_same_last_axis_size(type_str, arrays)
            _check_length_capacity(type_str, len, capacity)
      
            size(means2d_opacities_inverse_covariances_colors, 1) == 9 || 
                throw(DimensionMismatch(
                    "Expected 9 for the first axis length of \
                     `means2d_opacities_inverse_covariances_colors` (where 3 = 2 (means2d) \
                     + 1 (opacity) + 3 (lower triangle of inverse covariance matrix) + 3 \
                     (color))! \
                     Received: $(size(means2d_opacities_inverse_covariances_colors, 1))!"))
            size(intersected_tile_ranges)[1:2] == (2, 2) || throw(DimensionMismatch(
                "Expected (2, 2) for the `size` of `intersected_tile_ranges` along its \
                 first two axes! Received: $(size(intersected_tile_ranges)[1:2])!"))

            return new{D, I, F, GV, FTG, Z, M2OICC, ITR}(
                len,
                all_3d_gaussians, fgidx_to_gidx, 
                depths, means2d_opacities_inverse_covariances_colors, 
                intersected_tile_ranges
            )
        end
    end
)


"""
    _check_length_capacity(type_str, length, capacity)

Check that `0 <= length <= capacity` and throw an error if not. When `isnothing(capacity)`
we will skip the checks involving it.
"""
function _check_length_capacity(type_str, length, capacity)
    0 <= length || throw(DomainError(length, 
        "The `length` of a$(_n_when_starts_with_vowel(type_str)) `$type_str` must be \
         non-negative!"))
    # The DomainError already specifies the received value
    isnothing(capacity) || 0 <= capacity || throw(DomainError(capacity, 
        "The `capacity` of a$(_n_when_starts_with_vowel(type_str)) `$type_str` must be \
         non-negative!"))
    isnothing(capacity) || length <= capacity || throw(DimensionMismatch(
        "The `length` of a$(_n_when_starts_with_vowel(type_str)) `$type_str` may not \
         exceed its `capacity`! Received: $length > $capacity."))
end
# These days we not not explicitly store the capacity, but just retrieve it from the
# array sizes. In practice we then do not really need the capacity checks.

"""
    _check_length_capacity(type_str, length)

Alias for `_check_length_capacity(type_str, length, nothing)`.
"""
function _check_length(type_str, length)
    _check_length_capacity(type_str, length, nothing)
end


"""
    AbstractFiltered2DGaussiansVector{
        D <: ArrayDomain, I <: Integer, F <: AbstractFloat,
        GV <: AbstractGaussiansVector{D, I, F},
        FTG <: AbstractVector{I},
        Z <: AbstractVector{F},
        M2OICC <: AbstractMatrix{F},
        ITR <: AbstractArray{I, 3}
    }

Abstract supertype of [`MutableFiltered2DGaussiansVector`](@ref) and
[`ImmutableFiltered2DGaussiansVector`](@ref).
"""
AbstractFiltered2DGaussiansVector

"""
    ImmutableFiltered2DGaussiansVector{
        D <: ArrayDomain, I <: Integer, F <: AbstractFloat,
        GV <: AbstractGaussiansVector{D, I, F},
        FTG <: AbstractVector{I},
        Z <: AbstractVector{F},
        M2OICC <: AbstractMatrix{F},
        ITR <: AbstractArray{I, 3}
    } <: AbstractFiltered2DGaussiansVector{D, I, F, GV, FTG, Z, M2OICC, ITR}

Immutable version of [`MutableFiltered2DGaussiansVector`](@ref). Only useful within kernels,
as we do want to be able to change `length` in `render!`.
"""
ImmutableFiltered2DGaussiansVector

# Use length == capacity
@replace(
    X => (MutableFiltered2DGaussiansVector, ImmutableFiltered2DGaussiansVector),
    function X(
        all_3d_gaussians::GV, fgidx_to_gidx::FTG,
        depths::Z, means2d_opacities_inverse_covariances_colors::M2OICC,
        intersected_tile_ranges::ITR
    ) where {
        D <: ArrayDomain, I <: Integer, F <: AbstractFloat,
        GV <: AbstractGaussiansVector{D, I, F},
        FTG <: AbstractVector{I},
        Z <: AbstractVector{F},
        M2OICC <: AbstractMatrix{F},
        ITR <: AbstractArray{I, 3}
    }
        len = I(length(depths))
        return X(
            len,
            all_3d_gaussians, fgidx_to_gidx, 
            depths, means2d_opacities_inverse_covariances_colors, 
            intersected_tile_ranges
        )
    end
)

@replace(
    X => (MutableFiltered2DGaussiansVector, ImmutableFiltered2DGaussiansVector),
    function X(gv::AbstractGaussiansVector{D, I, F}) where {D, I, F}
        n = length(gv)
        return X(
            zero(I),
            gv, create_array(D, I, n),
            create_array(D, F, n), create_array(D, F, (9, n)),
            create_array(D, I, (2, 2, n))
        )
    end
)


"""
    MutableFiltered2DGaussiansVector{
        D <: ArrayDomain, I <: Integer, F <: AbstractFloat,
        GV <: AbstractGaussiansVector{D, I, F},
        FTG <: AbstractVector{I},
        Z <: AbstractVector{F},
        M2OICC <: AbstractMatrix{F},
        ITR <: AbstractArray{I, 3}
    } <: AbstractFiltered2DGaussiansVector{D, I, F, GV, FTG, Z, M2OICC, ITR}

Mutable struct representing projected 2D Gaussians (with means in continuous pixel
coordinates) after filtering. It contains the necessary information on these to splat them
for rendering.

The arrays here have a capacity potentially larger than the number of 2D Gaussians we
actually currently store (`length`). This avoid having to reallocate for every render call.
Broadcasting is supported, where the elements are [`Gaussian2D`](@ref)s.

Correctly filling in a Filtered2DGaussiansVector is complicated. Therefore, the content
will be created in multiple steps during the rendering pass. Consequently, the values might
not make sense in the mean time.

---

    MutableFiltered2DGaussiansVector(
        len::I,
        all_3d_gaussians::GV, fgidx_to_gidx::FTG,
        depths::Z, means2d_opacities_inverse_covariances_colors::M2OICC,
        intersected_tile_ranges::ITR
    ) where {
        D <: ArrayDomain, I <: Integer, F <: AbstractFloat,
        GV <: AbstractGaussiansVector{D, I, F},
        FTG <: AbstractVector{I},
        Z <: AbstractVector{F},
        M2OICC <: AbstractMatrix{F},
        ITR <: AbstractArray{I, 3}
    }

Construct a `MutableFiltered2DGaussiansVector` from its fields. Basic inner constructor
performing `size` and `ArrayDomain` checks.

---

    MutableFiltered2DGaussiansVector(
        all_3d_gaussians::GV, fgidx_to_gidx::FTG,
        depths::Z, means2d_opacities_inverse_covariances_colors::M2OICC,
        intersected_tile_ranges::ITR
    ) where {
        D <: ArrayDomain, I <: Integer, F <: AbstractFloat,
        GV <: AbstractGaussiansVector{D, I, F},
        FTG <: AbstractVector{I},
        Z <: AbstractVector{F},
        M2OICC <: AbstractMatrix{F},
        ITR <: AbstractArray{I, 3}
    }

Construct a `MutableFiltered2DGaussiansVector` from its arrays. Its `length` will be
correspond to the full array sizes.

---

    MutableFiltered2DGaussiansVector(gv::AbstractGaussiansVector{D, I, F}) where {D, I, F}
    
Construct an empty (`length == 0`) `MutableFiltered2DGaussiansVector` from a 3D Gaussians
vector. The produced MutableFiltered2DGaussiansVector's array values will be garbage.
"""
MutableFiltered2DGaussiansVector

# Don't provide a
#   Filtered2DGaussiansVector(gv::AbstractFiltered2DGaussiansVector)
# producing the mutable or immutable variant depending on ismutabletype(gv)
# as we really want MutableFiltered2DGaussiansVector in render!


# Similar to AbstractGaussiansVector and Gaussians:
@inline array_domain(::AbstractFiltered2DGaussiansVector{D}) where D = D
@inline int_type(::AbstractFiltered2DGaussiansVector{<:Any, I}) where I = I
@inline float_type(::AbstractFiltered2DGaussiansVector{<:Any, <:Any, F}) where F = F

@inline Base.length(f2dgv::AbstractFiltered2DGaussiansVector) = f2dgv.length
@inline function set_length!(f2dgv::MutableFiltered2DGaussiansVector, len)
    _check_length_capacity("MutableFiltered2DGaussiansVector", len, capacity(f2dgv))
    f2dgv.length = len
end
@inline set_nb_fgaussians!(f2dgv::MutableFiltered2DGaussiansVector, nb_fgaussians) = 
    set_length!(f2dgv, nb_fgaussians)
@inline capacity(f2dgv::AbstractFiltered2DGaussiansVector{<:Any, I}) where I = 
    I(length(f2dgv.depths))
@inline Base.parent(f2dgv::AbstractFiltered2DGaussiansVector) = f2dgv.all_3d_gaussians
@inline function Base.empty!(f2dgv::MutableFiltered2DGaussiansVector{<:Any, I}) where I
    f2dgv.length = zero(I)
    return f2dgv
end

function Base.show(
    io::IO, ::MIME"text/plain", 
    f2dgv::AbstractFiltered2DGaussiansVector{D, I, F}
) where {D, I, F}
    print(io, 
        "$(typeof_noparams(f2dgv)){$D, $I, $F}\n",
        "  length: ", length(f2dgv), " (out of ", length(parent(f2dgv)), 
            " 3D Gaussian$(_s_when_not_1(length(parent(f2dgv)))))\n",
        "  capacity: ", capacity(f2dgv), "\n",
    )
end

@inline Base.size(f2dgv::AbstractFiltered2DGaussiansVector) = (length(f2dgv),)
@inline Base.ndims(::Type{<:AbstractFiltered2DGaussiansVector}) = 1
@inline Base.firstindex(::AbstractFiltered2DGaussiansVector{<:Any, I}) where I = one(I)
@inline Base.lastindex(f2dgv::AbstractFiltered2DGaussiansVector) = length(f2dgv)

# slice
Base.@propagate_inbounds function Base.getindex(f2dgv::AbstractFiltered2DGaussiansVector, i)
    F2DGV = typeof(f2dgv)
    # We need to be a bit careful about the new length and capacity. For example,
    # if length(f2dgv) == 8 and capacity(f2dgv) == 16, then for i == 4:12 or the BitVector
    # equivalent, f2dgv[i] has length 4 and capacity 8.
    new_length = int_type(f2dgv)(
        _length_capacity_slice_count(length(f2dgv), capacity(f2dgv), i))
    remove_parameters(F2DGV)(
        new_length,
        f2dgv.all_3d_gaussians,  # (not an AbstractArray)
        (indexlastaxis(_length_normalized_getfield(f2dgv, fn, i), i) 
         for (fn, ft) in zip(fieldnames(F2DGV), fieldtypes(F2DGV))
         if ft <: AbstractArray  # Skip .length and .all_3d_gaussians
        )...
        # In particular, we slice fgidx_to_gidx. Then the fgids change, but the gids (i.e.
        # indices into the unchanged all_3d_gaussians) remain the same, as you'd want.
    )
end

"""
    _length_normalized_getfield(x, fn, i)

Return a `length`-normalized version of `getfield(x, fn)` compatible with slicing
by `i`. In particular, for logical indexing we need the size of the returned field to be
compatible with `length(i)`.
"""
_length_normalized_getfield(x, fn, _) = getfield(x, fn)
_length_normalized_getfield(x, fn, i::AbstractVector{Bool}) = 
    viewlastaxis(getfield(x, fn), 1:length(i))
    # Works both when length(i) == length(x) and when length(i) == capacity(gv)

"""
    _length_capacity_slice_count(len, cap, I)

Return `count(<=(len), (1:cap)[I])`, if possible without materializing or traversing the
array. We assume that `len <= cap`. `I` is allowed to exceed `cap`.
"""
_length_capacity_slice_count(len, cap, I) = count(<=(len), (1:cap)[I])  # fallback
_length_capacity_slice_count(len, _, I::OrdinalRange{<:Integer}) = 
    length(intersect(1:len, I))  # e.g. I == 2:4:10
_length_capacity_slice_count(len, _, I::AbstractVector{<:Integer}) = 
    count(<=(len), I)            # e.g. I == [2, 3, 2, 1]
_length_capacity_slice_count(len, _, I::AbstractVector{Bool}) = 
    count(view(I, 1:len))        # e.g. I == BitVector([0, 1, 1, 0])

Base.@propagate_inbounds function Base.view(f2dgv::AbstractFiltered2DGaussiansVector, i)
    F2DGV = typeof(f2dgv)
    new_length = int_type(f2dgv)(
        _length_capacity_slice_count(length(f2dgv), capacity(f2dgv), i))
    remove_parameters(F2DGV)(
        new_length,
        f2dgv.all_3d_gaussians,
        (viewlastaxis(_length_normalized_getfield(f2dgv, fn, i), i) 
         for (fn, ft) in zip(fieldnames(F2DGV), fieldtypes(F2DGV))
         if ft <: AbstractArray
        )...
    )
end


"""
    maybe_unsafe_free!
        f2dgv::F2DGV
    ) where {D, F2DGV <: AbstractFiltered2DGaussiansVector{D}}

Call `maybe_unsafe_free!` on all arrays in the Filtered2DGaussiansVector. If D <: GPU
these will then be `unsafe_free!`d. If not, nothing happens.

We will never call `unsafe_free!` on the parent `GaussiansVector`.
"""
function maybe_unsafe_free!(f2dgv::F2DGV) where F2DGV <: AbstractFiltered2DGaussiansVector
    for (fn, ft) in zip(fieldnames(F2DGV), fieldtypes(F2DGV))
        ft <: AbstractArray && maybe_unsafe_free!(getfield(f2dgv, fn))
    end
    return nothing
end


"""
    set_nb_gaussians!(mf2dgv::MutableFiltered2DGaussiansVector, nb_gaussians)

Unless this was already within `capacity`, resize the arrays to the new `nb_gaussians`.

!!! We do not alter `all_3d_gaussians`. In the intended use-case of `set_nb_gaussians!` in
adaptive density control, the adjustment of the MutableTrainableGaussiansVector will need
to be performed separately. 
"""
function set_nb_gaussians!(mf2dgv::MutableFiltered2DGaussiansVector, nb_gaussians)
    nb_gaussians <= capacity(mf2dgv) && return
    
    for fn in fieldnames(MutableFiltered2DGaussiansVector)
        fn in (:length, :all_3d_gaussians) && continue
        array = getfield(mf2dgv, fn)
        setfield!(mf2dgv, fn, resize_memory(array, (size(array)[1:end-1]..., nb_gaussians)))
    end
    mf2dgv.length = 0  # All values become invalid, so we have 0 valid filtered Gaussians
    return nothing
end


# TODO: Reference2DGaussianPrimitive <: Abstract2DGaussianPrimitive <: Abstract2DGaussianFunction
# TODO: Explicit2DGaussianFunction (or Primitive) <: ...?
"""
    Gaussian2D{
        D <: ArrayDomain, 
        I <: Integer, 
        F <: AbstractFloat,
        F2DGV <: AbstractFiltered2DGaussiansVector{D, I, F}
    }

A 2D Gaussian primitive, projection of a 3D Gaussian. The coordinates of the 2D mean are in
continuous pixel coordinates. `Gaussian2D`s are the elements of Filtered2DGaussiansVectors.

!!! A `Gaussian2D` is defined by its index into the Filtered2DGaussiansVector. If the latter
is mutated, the index and hence the `Gaussian2D` might become invalid. 
"""
struct Gaussian2D{
    D <: ArrayDomain,
    I <: Integer,
    F <: AbstractFloat,
    F2DGV <: AbstractFiltered2DGaussiansVector{D, I, F}
}
    f2dgv::F2DGV
    fgidx::I

    @inline function Gaussian2D(f2dgv::F2DGV, fgidx::Integer) where {
        D <: ArrayDomain, I <: Integer, F <: AbstractFloat,
        F2DGV <: AbstractFiltered2DGaussiansVector{D, I, F}
    }
        @boundscheck 1 <= fgidx <= length(f2dgv) || throw(ArgumentError(
            "The provided Gaussian2D index $fgidx is out of bounds (1:$(length(f2dgv)))!"))
        return new{D, I, F, F2DGV}(f2dgv, fgidx)
    end

    # For Device: no string interpolation
    @inline function Gaussian2D(f2dgv::F2DGV, fgidx::Integer) where {
        I <: Integer, F <: AbstractFloat,
        F2DGV <: AbstractFiltered2DGaussiansVector{Device, I, F}
    }
        @boundscheck 1 <= fgidx <= length(f2dgv) || throw(ArgumentError(
            "The provided Gaussian2D index is out of bounds!"))
        return new{Device, I, F, F2DGV}(f2dgv, fgidx)
    end
end

Base.@propagate_inbounds Base.getindex(
    f2dgv::AbstractFiltered2DGaussiansVector, fgidx::Integer) = Gaussian2D(f2dgv, fgidx)
Base.@propagate_inbounds Base.first(f2dgv::AbstractFiltered2DGaussiansVector) = f2dgv[1]
Base.@propagate_inbounds Base.last(f2dgv::AbstractFiltered2DGaussiansVector) = 
    f2dgv[length(f2dgv)]

@inline Base.eltype(::F2DGV) where {
    D, I, F, F2DGV <: AbstractFiltered2DGaussiansVector{D, I, F}
} = Gaussian2D{D, I, F, F2DGV}
@inline Base.broadcastable(f2dgv::AbstractFiltered2DGaussiansVector) = f2dgv
@inline Base.BroadcastStyle(::Type{F2DGV}) where 
    {M, F2DGV <: AbstractFiltered2DGaussiansVector{GPU{M}}
} = CUDACore.CuArrayStyle{1, M}()
function Base.iterate(
    f2dgv::AbstractFiltered2DGaussiansVector{<:Any, I}, 
    state::I = one(I)
) where I
    state <= length(f2dgv) || return nothing
    return (@inbounds f2dgv[state], state + one(I))
end

@inline array_domain(::Gaussian2D{D}) where D = D
@inline int_type(::Gaussian2D{<:Any, I}) where I = I
@inline float_type(::Gaussian2D{<:Any, <:Any, F}) where F = F

@inline Base.parent(g::Gaussian2D) = g.f2dgv
@inline get_fgidx(g::Gaussian2D) = g.fgidx
@inline get_gidx(g::Gaussian2D) = @inbounds g.f2dgv.fgidx_to_gidx[get_fgidx(g)]

@inline get_mean_view(g::Gaussian2D) = @inbounds view(
    g.f2dgv.means2d_opacities_inverse_covariances_colors, SOneTo(2), g.fgidx)
@inline get_mean_sv(g::Gaussian2D) = SVector{2}(get_mean_view(g))

"""
    set_mean!(g::Gaussian2D, x, y)

Set the mean of the 2D Gaussian `g` to `(x, y)` (in continuous pixel coordinates). 
"""
@inline function set_mean!(g::Gaussian2D, x, y)
    mv = get_mean_view(g)
    @inbounds begin mv[1] = x; mv[2] = y end
    return nothing
end
@inline set_mean!(g::Gaussian2D, xy) = set_mean!(g, xy...)

@inline get_opacity_view(g::Gaussian2D) = 
    @inbounds view(g.f2dgv.means2d_opacities_inverse_covariances_colors, 
                   SUnitRange(3, 3), g.fgidx)
@inline get_opacity(g::Gaussian2D) = 
    @inbounds g.f2dgv.means2d_opacities_inverse_covariances_colors[3, g.fgidx]  
    # Like for 3D Gaussians, will throw scalar indexing" error outside of kernels
@inline function set_opacity!(g::Gaussian2D, opacity)
    @inbounds g.f2dgv.means2d_opacities_inverse_covariances_colors[3, g.fgidx] = opacity
    return nothing
end

@inline get_depth_view(g::Gaussian2D) = @inbounds view(g.f2dgv.depths, g.fgidx:g.fgidx)
@inline get_depth(g::Gaussian2D) = @inbounds g.f2dgv.depths[g.fgidx]
@inline function set_depth!(g::Gaussian2D, depth) 
    @inbounds g.f2dgv.depths[g.fgidx] = depth
    return nothing
end

@inline get_inverse_covariance_lower_view(g::Gaussian2D) = 
    @inbounds view(g.f2dgv.means2d_opacities_inverse_covariances_colors, 
                   SUnitRange(4, 6), g.fgidx)
@inline get_inverse_covariance_lower_sv(g::Gaussian2D) = 
    SVector{3}(get_inverse_covariance_lower_view(g))
@inline function set_inverse_covariance_lower!(g::Gaussian2D, ic11, ic21, ic22)
    icv = get_inverse_covariance_lower_view(g)
    @inbounds begin icv[1] = ic11; icv[2] = ic21; icv[3] = ic22 end
    return nothing
end
@inline function set_inverse_covariance_lower!(g::Gaussian2D, ic11, ic21, _, ic22)
    icv = get_inverse_covariance_lower_view(g)
    @inbounds begin icv[1] = ic11; icv[2] = ic21; icv[3] = ic22 end
    return nothing
end
@inline set_inverse_covariance_lower!(g::Gaussian2D, ic) = 
    set_inverse_covariance_lower!(g, ic...)
# In particular, SVector{3}, SVector{4} and SMatrix{2, 2}

"""
    set_inverse_covariance_lower!(g::Gaussian2D, ...)

Set the lower triangular part of this 2D Gaussian. The second argument can either consist
of the relevant three entries ([1, 1], [2, 1], [2, 2]), individually or grouped as an
`SVector{3}` (or any type of static length 3), or it can consist of the entire 2x2 matrix
(or any type of static length 4), as 4 individual entries in column major order, as an
`SVector{4}` thereof, or as an `SMatrix{2, 2}`. In the 4 entries case we take the [2, 1]
entry and do not check symmetry.
"""
set_inverse_covariance_lower!

@inline get_inverse_covariance_sm(g::Gaussian2D) = 
    SMatrix(SymmetricSMatrix(get_inverse_covariance_lower_sv(g)))
@inline function get_inverse_covariance_determinant(g::Gaussian2D)
    icov = get_inverse_covariance_lower_view(g)
    return @inbounds icov[1] * icov[3] - icov[2] ^ 2
end

"""
    _get_bbox_cts_px_cov(opacity, x, y, cov11, cov22, min_alpha::F) where F

Get the tight axis-aligned bounding box for a 2D Gaussian ellipse level set at
alpha = `min_alpha`, where the 2D Gaussian has the supplied `opacity`, mean at `(x, y)`
and supplied diagonal entries of the (non-inverted) covariance matrix. The covariance matrix
and `x`, `y` need to use the same image coordinate system, i.e. continuous pixel
coordinates, which will also be the output coordinate system.
"""
@inline function _get_bbox_cts_px_cov(opacity, x, y, cov11, cov22, min_alpha::F) where F
    opacity >= min_alpha || return EmptyRectangle{F}()

    gamma = 2 * log(opacity / min_alpha)
    x_radius = @inbounds sqrt(gamma * cov11)
    y_radius = @inbounds sqrt(gamma * cov22)
    return Rectangle(
        x - x_radius, y - y_radius,
        x + x_radius, y + y_radius
    )
    # Found by converting opacity * Gaussian((u, v); (x, y), cov) > min_alpha
    # into a quadratic equation (in either u or v) and then checking when the
    # discriminant is non-negative.
end

"""
    _get_bbox_cts_px_icov(opacity, x, y, icov11, icov21, icov22, min_alpha::F) where F

Get the tight axis-aligned bounding box for a 2D Gaussian ellipse level set at
alpha = `min_alpha`, where the 2D Gaussian has the supplied `opacity`, mean at `(x, y)`
and supplied lower triangle entries of the inverted covariance matrix. The latter needs to
use the same image coordinate system as `x` and `y`, i.e. continuous pixel coordinates,
which will also be the output coordinate system.

!!! We require but do not check that `opacity` is at least `min_alpha`.
"""
@inline function _get_bbox_cts_px_icov(
    opacity, x, y, icov11, icov21, icov22, min_alpha::F
) where F
    opacity >= min_alpha || return EmptyRectangle{F}()
    
    idet = icov11 * icov22 - icov21 ^ 2
    gamma = 2 * log(opacity / min_alpha) / idet  # (Not the same gamma as for cov)
    x_radius = @inbounds sqrt(gamma * icov22)
    y_radius = @inbounds sqrt(gamma * icov11)
    return Rectangle(
        x - x_radius, y - y_radius,
        x + x_radius, y + y_radius
    )
end

"""
    get_bbox_cts_px(g::Gaussian2D, min_alpha)

Return a tight axis-aligned bounding box for this 2D Gaussian, for the alpha-level set
determined by `min_alpha`. It uses the same continuous pixel coordinates as
the 2D Gaussian itself.

!!! We assume that `g`'s opacity is at least `min_alpha`. This should have been realized in
the filtering step.
"""
@inline function get_bbox_cts_px(g::Gaussian2D, min_alpha)
    return _get_bbox_icov(
        get_opacity(g), 
        get_mean_sv(g)..., 
        get_covariance_lower_sv(g)..., 
        min_alpha)
end

"""
    _get_bbox_discrete_px(
        cts_bbox::AbstractRectangle{F},
        width::I, height::I,
    ) where {I, F}

Convert the axis-aligned bounding box in (floating point `F`) continuous pixel coordinates
into the corresponding `AbstractRectangle{I}` bounding box for discrete screen pixels.
`width` and `height` determine the screen size in pixels.
"""
@inline function _get_bbox_discrete_px(
    cts_bbox::Rectangle{F},
    width::I, height::I,
) where {I, F}

    cts_left_px = get_left(cts_bbox)
    cts_top_px = get_top(cts_bbox)
    cts_right_px = get_right(cts_bbox)
    cts_bottom_px = get_bottom(cts_bbox)
    # Horizontal pixel k (discrete) consists of continuous (k - 1, k] pixel coordinates.
    # Therefore, the bbox left lies in pixel (column) k = ceil(left).
    # k-1__________k
    #   |          |
    #   |    . left|
    #   |          |
    # k-1__________k
    # The pixel's (continuous) center is k - 0.5. When the bbox left lies to the right of
    # this center (as pictured), then pixel k does not get a contribution.
    # So if left <= ceil(left) - 0.5, use ceil(left),
    # and if left > ceil(left) - 0.5, use ceil(left) + 1.
    # E.g. k = ceil(left) = 3 (2 < left <= 3): 
    #   (2 <) left <= 2.5 --> 3
    #   2.5 < left (< 3)  --> 4
    # We can then use ceil(left + 0.5) (which except at the middle is round(left) + 1).
    # For right the situation is reversed:
    #   if right >= ceil(right) - 0.5, use ceil(right);
    #   if right <  ceil(right) - 0.5, use ceil(right) - 1.
    # E.g. k = ceil(right) = 3 (2 < right <= 3):
    #    (2 <) right < 2.5 --> 2
    #   2.5 <= right <= 3  --> 3
    # Here we can then use round(right) (or ceil(right - 0.5), except at the middle).

    # Finally, we need to clamp to the screen boundaries.
    return Rectangle{I}(
        max(one(I), ceil(I, cts_left_px + F(0.5))),
        max(one(I), ceil(I, cts_top_px + F(0.5))),
        min(width, round(I, cts_right_px)),
        min(height, round(I, cts_bottom_px))
    )
end

@inline _get_bbox_discrete_px(::EmptyRectangle{F}, ::I, ::I) where {I, F} = 
    EmptyRectangle{I}()

"""
    get_bbox_discrete_pixels(
        g::Gaussian2D,
        width, height,
        min_alpha
    )

Return a tight axis-aligned bounding box of those discrete pixels for which the 2D
Gaussian's alpha-value might be at least `min_alpha`. The screen size in pixels is given by
`width` and `height`.
"""
@inline function get_bbox_discrete_px(
    g::Gaussian2D,
    width, height,
    min_alpha
)
    return _get_bbox_discrete_px(get_bbox_cts_px(g, min_alpha), width, height)
end

@inline get_color_view(g::Gaussian2D) = 
    @inbounds view(g.f2dgv.means2d_opacities_inverse_covariances_colors, 
                   SUnitRange(7, 9), g.fgidx)
@inline get_color_sv(g::Gaussian2D) = SVector{3}(get_color_view(g))
@inline function set_color!(g::Gaussian2D, rgb) 
    cv = get_color_view(g)
    @inbounds begin cv[1] = rgb[1]; cv[2] = rgb[2]; cv[3] = rgb[3] end
    return nothing
end
@inline set_color!(gaussian::Gaussian2D, r, g, b) = set_color!(gaussian, SVector(r, g, b))

"""
    get_color_rgb(g::Gaussian2D)

Return the `RGB` color (with the same floating point `eltype` as `g`'s parent
Filtered2DGaussiansVector) of this 2D Gaussian. We clamp the values to [0, 1].
"""
@inline get_color_rgb(g::Gaussian2D) = RGB(clamp.(get_color_sv(g), 0, 1)...)
@inline set_color!(g::Gaussian2D, rgb::RGB) = 
    set_color!(g, red(rgb), green(rgb), blue(rgb))

@inline get_intersected_tile_range_view(g::Gaussian2D) = 
    @inbounds view(g.f2dgv.intersected_tile_ranges, SOneTo(2), SOneTo(2), g.fgidx)
@inline get_intersected_tile_range_sv(g::Gaussian2D) = 
    SVector{4}(get_intersected_tile_range_view(g))   
@inline get_intersected_tile_range_sm(g::Gaussian2D) = 
    SMatrix{2, 2}(get_intersected_tile_range_view(g))
@inline get_intersected_tile_range_rect(g::Gaussian2D) = 
    Rectangle(get_intersected_tile_range_sv(g)...)
    # Because both use left (xmin), top (ymin), right (xmax), bottom (ymax) order
@inline function set_intersected_tile_range!(
    g::Gaussian2D, 
    tile_x_min, tile_y_min, tile_x_max, tile_y_max
)
    itrv = get_intersected_tile_range_view(g)
    @inbounds begin 
        itrv[1] = tile_x_min; itrv[2] = tile_y_min; 
        itrv[3] = tile_x_max; itrv[4] = tile_y_max 
    end
    return nothing
end
@inline set_intersected_tile_range!(g::Gaussian2D, tile_range::SVector{4}) =
    set_intersected_tile_range!(g, tile_range...)
@inline set_intersected_tile_range!(g::Gaussian2D, tile_range::SMatrix{2, 2}) =
    set_intersected_tile_range!(g, tile_range...)
@inline set_intersected_tile_range!(g::Gaussian2D, bbox::Rectangle{I}) where I <: Integer =
    set_intersected_tile_range!(g, 
        get_left(bbox), get_top(bbox), get_right(bbox), get_bottom(bbox))


@inline get_3d_gaussian(g::Gaussian2D) = 
    @inbounds g.f2dgv.all_3d_gaussians[get_gidx(g)]

"""
    get_value(g::Gaussian2D{<:Any, <:Any, F}, u, v)

Return the 2D Gaussian function value at the continuous pixel coordinates `(u, v)`.
"""
@inline function get_value_cts_px(g::Gaussian2D{<:Any, <:Any, F}, u, v) where F
    p, q, r = get_inverse_covariance_lower_sv(g)
    x, y = get_mean_sv(g)
    du = u - x; dv = v - y
    return get_opacity(g) * exp(F(-0.5) * (p * du^2 + r * dv^2) - q * du * dv)
end

"""
    get_value_discrete_px(
        g::Gaussian2D{<:Any, <:Any, F}, 
        u, v, 
    ) where F

Return the 2D Gaussian function value at the discrete pixel coordinates `(u, v)`. I.e. we
evaluate the function at the (continuous) pixel center.
"""
@inline function get_value_discrete_px(
    g::Gaussian2D{<:Any, <:Any, F}, 
    u, v, 
) where F

    return get_value_cts_px(
        g,
        u - F(0.5),  # -0.5 to go to the pixel center
        v - F(0.5),
    )
end

function Base.show(io::IO, ::MIME"text/plain", g::Gaussian2D)
    CUDA.@allowscalar begin
        tile_rect = get_intersected_tile_range_rect(g)
        print(io, 
            "Filtered 2D Gaussian ", g.fgidx, " out of ", length(g.f2dgv), 
                " corresponding to 3D Gaussian ", get_gidx(g), "\n",
            "  2D mean: ", get_mean_view(g), "\n",
            "  depth: ", get_depth(g), "\n",
            "  opacity: ", get_opacity(g), "\n",
            "  inverse covariance matrix: ", get_inverse_covariance_sm(g), "\n",
            "  color: ", get_color_sv(g), "\n",
            "  intersected_tile_range (x × y):",
                " [$(get_left(tile_rect)), $(get_right(tile_rect))] × \
                    [$(get_top(tile_rect)), $(get_bottom(tile_rect))]"
        )
    end
end


# TODO: split into filtered, duplicated, sorted duplicated?
@im_mutable_with_adapt struct RenderBuffers{
    D <: ArrayDomain, I <: Integer, F <: AbstractFloat, 
    KG <: AbstractVector{Bool}, KGCS <: AbstractVector{I},
    F2DGV <: AbstractFiltered2DGaussiansVector{D, I, F},
    OT <: AbstractVector{I}, OTCS <: AbstractVector{I},
    DC <: AbstractVector{Int64},   # Always 64-bit
    DTF <: AbstractVector{Int32},  # Always 32-bit # TODO: why bother with I?
    SDC <: AbstractVector{Int64}, STF <: AbstractVector{Int32},
    SB <: AbstractVector{UInt8},
    TES <: AbstractVector{I},
}

    keep_gaussians::KG                  # n
    #   Indicates for every 3D gaussians whether we keep it
    keep_gaussians_cs::KGCS             # n
    #   Cumsum of the former
    f2dgv::F2DGV                        # capacity n
    #   Full Filtered2DGaussiansVector, constructed using the above

    f2dgs_nb_overlapped_tiles::OT       # n
    #   Contains for every filtered 2D Gaussian how many tiles they overlap
    f2dgs_nb_overlapped_tiles_cs::OTCS  # n
    
    nb_dfgaussians::I                   # l (<= m), cf. length (<= capacity)
    df2dgs_codes::DC                    # m (after set_nb_dfgaussians!) 
    #   Codes consisting of tile_index and depth, for each duplicated filtered 2D Gaussian
    #   Each filtered Gaussian contributes one per intersected tile
    #   In filtered Gaussian order
    dfgidx_to_fgidx::DTF                # m
    #   Maps a duplicated filtered Gaussian index to the corresponding filtered Gaussian
    #   index
    sdf2dgs_codes::SDC                  # m
    #   Sorted version of df2dgs_codes, in lexicographic (tile_index, depth) order
    sdfgidx_to_fgidx::STF               # m

    sorting_buffer::SB                  # k ≈ 2 l * sizeof(Int64) = 16 l
    #   Buffer required for cub's SortPairs

    tiles_end_sdfgidx::TES              # nb_tiles_x * nb_tiles_y
    #   End (boundary) indices of the tiles in the implicit sirted duplicated filtered 2D
    #   Gaussians vector. I.e. a tile with linear index tidx corresponds to sdfgids 
    #   [tiles_end_sdfgidx[tidx - 1] + 1, tiles_end_sdfgidx[tidx]].


    function RenderBuffers(
        keep_gaussians::KG, keep_gaussians_cs::KGCS,
        f2dgv::F2DGV,
        f2dgs_nb_overlapped_tiles::OT, f2dgs_nb_overlapped_tiles_cs::OTCS,
        nb_dfgaussians::I,
        df2dgs_codes::DC, dfgidx_to_fgidx::DTF,
        sdf2dgs_codes::SDC, sdfgidx_to_fgidx::STF,
        sorting_buffer::SB,
        tiles_end_sdfgidx::TES,
    ) where {
        D <: ArrayDomain, I <: Integer, F <: AbstractFloat, 
        KG <: AbstractVector{Bool}, KGCS <: AbstractVector{I},
        F2DGV <: AbstractFiltered2DGaussiansVector{D, I, F},
        OT <: AbstractVector{I}, OTCS <: AbstractVector{I},
        DC <: AbstractVector{Int64}, DTF <: AbstractVector{Int32},
        SDC <: AbstractVector{Int64}, STF <: AbstractVector{Int32},
        SB <: AbstractVector{UInt8},
        TES <: AbstractVector{I},
    }

        type_str = "RenderBuffers"
        f2dgs_arrays = (
            keep_gaussians, keep_gaussians_cs,
            f2dgs_nb_overlapped_tiles, f2dgs_nb_overlapped_tiles_cs, 
        )  # though not f2dgv itself
        df2dgs_arrays = (
            df2dgs_codes, dfgidx_to_fgidx,
            sdf2dgs_codes, sdfgidx_to_fgidx
        )
        arrays = (f2dgs_arrays..., df2dgs_arrays..., sorting_buffer, tiles_end_sdfgidx)
            
        D2 = _get_common_array_domain(type_str, arrays)
        D2 === D || throw(ArgumentError(
            "The arrays and `Filtered2DGaussiansVector` in `$type_str` need to have the \
             same `ArrayDomain`. Received: $D2 and $D."))
        _check_same_last_axis_size(
            type_str, f2dgs_arrays, capacity(f2dgv), "filtered Gaussian arrays")
        _check_same_last_axis_size(
            type_str, df2dgs_arrays, nothing, "duplicated filtered Gaussian arrays")
        dfcapacity = length(df2dgs_codes)
        dfcapacity >= nb_dfgaussians || throw(ArgumentError(
            "The capacity for the duplicated filtered Gaussian arrays must be at least the \
             number of actual duplicated filtered Gaussians. Received: \
             $dfcapacity < $nb_dfgaussians."))

        return new{
            D, I, F, 
            KG, KGCS,
            F2DGV,
            OT, OTCS,
            DC, DTF, SDC, STF, SB,
            TES
        }(
            keep_gaussians, keep_gaussians_cs,
            f2dgv,
            f2dgs_nb_overlapped_tiles, f2dgs_nb_overlapped_tiles_cs,
            nb_dfgaussians,
            df2dgs_codes, dfgidx_to_fgidx, sdf2dgs_codes, sdfgidx_to_fgidx,
            sorting_buffer,
            tiles_end_sdfgidx
        )
    end
end

function MutableRenderBuffers(
    gv::AbstractGaussiansVector{D, I}, t::Tiling,
) where {
    D, I
}
    n = length(gv)  # our capacity
    return MutableRenderBuffers(
        create_array(D, Bool, n),              # keep_gaussians
        create_array(D, I, n),                 # keep_gaussians_cs
        MutableFiltered2DGaussiansVector(gv),  # f2dgv
        create_array(D, I, n),                 # f2dgs_nb_overlapped_tiles
        create_array(D, I, n),                 # f2dgs_nb_overlapped_tiles_cs

        # We have no idea about the number of duplicated filtered Gaussians in advance.
        # (Could be higher than n, could be lower.) So just initialize at 0.
        zero(I),                                # nb_dfgaussians
        create_array(D, Int64, 0),              # df2dgs_codes
        create_array(D, Int32, 0),              # dfgidx_to_fgidx
        create_array(D, Int64, 0),              # sdf2dgs_codes
        create_array(D, Int32, 0),              # sdfgidx_to_fgidx
        create_array(D, UInt8, 0),              # sorting_buffer

        create_array(D, I, get_nb_tiles(t))     # tiles_end_sdfgidx
    )
end


"""
    AbstractRenderBuffers{
        D <: ArrayDomain, I <: Integer, F <: AbstractFloat, 
        KG <: AbstractVector{Bool}, KGCS <: AbstractVector{I},
        F2DGV <: AbstractFiltered2DGaussiansVector{D, I, F},
        OT <: AbstractVector{I}, OTCS <: AbstractVector{I},
        DC <: AbstractVector{Int64}, DTF <: AbstractVector{Int32},
        SDC <: AbstractVector{Int64}, STF <: AbstractVector{Int32},
        SB <: AbstractVector{UInt8},
        TES <: AbstractVector{I},
    }

Abstract supertype for [`MutableRenderBuffers`](@ref) and [`ImmutableRenderBuffers`](@ref).
"""
AbstractRenderBuffers

"""
    ImmutableRenderBuffers{
        D <: ArrayDomain, I <: Integer, F <: AbstractFloat, 
        KG <: AbstractVector{Bool}, KGCS <: AbstractVector{I},
        FG2DV <: AbstractFiltered2DGaussiansVector{D, I, F},
        OT <: AbstractVector{I}, OTCS <: AbstractVector{I},
        DC <: AbstractVector{Int64}, DTF <: AbstractVector{Int32},
        SDC <: AbstractVector{Int64}, STF <: AbstractVector{Int32},
        SB <: AbstractVector{UInt8},
        TES <: AbstractVector{I},
    } <: AbstractRenderBuffers{
        D, I, F, KG, KGCS, F2DGV, OT, OTCS, DC, DTF, SDC, STF, SB, TES}

Immutable version of [`MutableRenderBuffers`](@ref). Only for use in kernels (though
this can easily be avoided by only supplying the relevant buffers for each kernel).
"""
ImmutableRenderBuffers

"""
    MutableRenderBuffers{
        D <: ArrayDomain, I <: Integer, F <: AbstractFloat, 
        KG <: AbstractVector{Bool}, KGCS <: AbstractVector{I},
        F2DGV <: AbstractFiltered2DGaussiansVector{D, I, F},
        OT <: AbstractVector{I}, OTCS <: AbstractVector{I},
        DC <: AbstractVector{Int64}, DTF <: AbstractVector{Int32},
        SDC <: AbstractVector{Int64}, STF <: AbstractVector{Int32},
        SB <: AbstractVector{UInt8},
        TES <: AbstractVector{I}
    } <: AbstractRenderBuffers{
        D, I, F, KG, KGCS, F2DGV, OT, OTCS, TDE, DC, DTF, SDC, STF, SB, TES} 

Mutable struct used to group some render buffers. These allow memory to be reused
between subsequent renders.

---

    MutableRenderBuffers(
        keep_gaussians::KG, keep_gaussians_cs::KGCS,
        f2dgv::F2DGV,
        f2dgs_nb_overlapped_tiles::OT, f2dgs_nb_overlapped_tiles_cs::OTCS,
        nb_dfgaussians::I,
        df2dgs_codes::DC, dfgidx_to_fgidx::DTF,
        sdf2dgs_codes::SDC, sdfgidx_to_fgidx::STF,
        sorting_buffer::SB,
        tiles_end_sdfgidx::TES,
    ) where {
        D <: ArrayDomain, I <: Integer, F <: AbstractFloat, 
        KG <: AbstractVector{Bool}, KGCS <: AbstractVector{I},
        F2DGV <: AbstractFiltered2DGaussiansVector{D, I, F},
        OT <: AbstractVector{I}, OTCS <: AbstractVector{I},
        DC <: AbstractVector{Int64}, DTF <: AbstractVector{Int32},
        SDC <: AbstractVector{Int64}, STF <: AbstractVector{Int32},
        SB <: AbstractVector{UInt8},
        TES <: AbstractVector{I}
    }

Construct a `MutableRenderBuffers` from the individual buffers.

---

    MutableRenderBuffers(
        gv::AbstractGaussiansVector{D, I}, t::Tiling
    ) where {
        D, I
    }

Use the 3D Gaussians vector and tiling to construct a `MutableRenderBuffers`,
which will be filled with garbage.
"""
MutableRenderBuffers

@inline array_domain(::AbstractRenderBuffers{D}) where D = D
@inline int_type(::AbstractRenderBuffers{<:Any, I}) where I = I
@inline float_type(::AbstractRenderBuffers{<:Any, <:Any, F}) where F = F

"""
    set_nb_dfgaussians!(
        rb::MutableRenderBuffers, nb_dfgaussians; 
        free = true
    )

Set the number of duplicated filtered Gaussians in the `MutableRenderBuffers` to the
supplied value. This potentially requires growing the associated vectors, which requires
allocated new memory. `free` determines if we `maybe_unsafe_free!` the old. Defaults to
`true`.

"""
function set_nb_dfgaussians!(
    rb::MutableRenderBuffers, nb_dfgaussians; 
    free = true
)
    foreach((:df2dgs_codes, :dfgidx_to_fgidx, :sdf2dgs_codes, :sdfgidx_to_fgidx)) do fn
        setfield!(
            rb, fn, 
            resize_memory(getfield(rb, fn), nb_dfgaussians; free))
    end
    rb.nb_dfgaussians = nb_dfgaussians
    # TODO: should we not just always expand?, using capacity vs length?
end

@inline get_nb_dfgaussians(rb::AbstractRenderBuffers) = rb.nb_dfgaussians

"""
    maybe_unsafe_free!(rb::AbstractRenderBuffers)

Call `maybe_unsafe_free!` on all arrays (or groups thereof) in the RenderBuffers.
"""
function maybe_unsafe_free!(rb::RB) where RB <: AbstractRenderBuffers{<:GPU}
    # Note: using where {D <: GPU, I, RB <: AbstractRenderBuffers{D, I}} for some reason
    # causes the general method above to take precedence, unless we repeat the I <: Integer
    # from the RenderBuffers definition
    for (fn, ft) in zip(fieldnames(RB), fieldtypes(RB))
        ft <: AbstractArray && maybe_unsafe_free!(getfield(rb, fn))
    end
    maybe_unsafe_free!(rb.f2dgv)
    rb.nb_dfgaussians = zero(int_type(rb))
    return nothing
end

function Base.show(
    io::IO, 
    ::MIME"text/plain", 
    rb::AbstractRenderBuffers{D, I, F}
) where {D, I, F}
    nb_f = length(rb.f2dgv)
    nb_d = get_nb_dfgaussians(rb)
    print(io, 
        "$(typeof_noparams(rb)){$D, $I, $F} for $nb_f filtered 2D \
            Gaussian$(_s_when_not_1(nb_f)) \
         and $nb_d duplicated (filtered 2D) Gaussian$(_s_when_not_1(nb_d))"
    )
end

"""
    set_nb_gaussians!(mrb::MutableRenderBuffers, nb_gaussians)

Resize the per-Gaussian (capacity) buffers to the new `nb_gaussians`. 
"""
function set_nb_gaussians!(mrb::MutableRenderBuffers, nb_gaussians)
    mrb.keep_gaussians = resize_memory(mrb.keep_gaussians, nb_gaussians)
    mrb.keep_gaussians_cs = resize_memory(mrb.keep_gaussians_cs, nb_gaussians)
    set_nb_gaussians!(mrb.f2dgv, nb_gaussians)
    mrb.f2dgs_nb_overlapped_tiles = resize_memory(
        mrb.f2dgs_nb_overlapped_tiles, nb_gaussians)
    mrb.f2dgs_nb_overlapped_tiles_cs = resize_memory(
        mrb.f2dgs_nb_overlapped_tiles_cs, nb_gaussians)
end

# TODO: retile!, resize!