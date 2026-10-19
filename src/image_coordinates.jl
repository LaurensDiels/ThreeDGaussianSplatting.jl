# In recent versions we always use pixel coordinates, so these methods are not used heavily.
# Some hyperparameters are still more naturally expressed in normalized coordinates, though.

"Abstract supertype for continuous 2D image coordinates."
abstract type AbstractImageCoordinates end
"""Concrete `AbstractImageCoordinates` with valid continuous image extent
``[0, \\text{width}) \\times [0, \\text{height})]``."""
struct PixelCoordinates <: AbstractImageCoordinates end
"""Concrete `AbstractImageCoordinates` with valid (continuous) image extent ``(-1, 1]^2``
(horizontal, vertical). Like 2D normalized device coordinates (NDC)."""
struct NormalizedCoordinates <: AbstractImageCoordinates end

"Abstract supertype for converters between different image coordinates."
abstract type AbstractCoordinateConverter{F} end

"""
    PixelToPixelConverter{F} <: AbstractCoordinateConverter{F}

To convert from PixelCoordinates (using some type) to PixelCoordinates (using F).

---

    PixelToPixelConverter(::Type{F}) where F = new{F}()
    
Construct a PixelToPixelConverter{F}.

---

    PixelToPixelConverter(::Type{F}, _) where F = new{F}()
    
Construct a PixelToPixelConverter{F}. The second parameter is for consistency
with e.g. [`PixelToNormalizedConverter`](@ref).
"""
struct PixelToPixelConverter{F} <: AbstractCoordinateConverter{F}
    PixelToPixelConverter(::Type{F}) where F = new{F}()
    PixelToPixelConverter(::Type{F}, _) where F = new{F}()
end

"""
    PixelToNormalizedConverter{F} <: AbstractCoordinateConverter{F}

To convert from PixelCoordinates (using some type) to NormalizedCoordinates (using F).

---

    PixelToNormalizedConverter(::Type{F}, size) where F

Construct a PixelToNormalizedConverter{F} for a screen of size `size` along the
currently considered axis (i.e. the width or height), in pixels.

"""
struct PixelToNormalizedConverter{F} <: AbstractCoordinateConverter{F}
    # x -> 2/s * x - 1
    factor::F

    PixelToNormalizedConverter(::Type{F}, size) where F = new{F}(2 / F(size))
end

"""
    NormalizedToPixelConverter{F} <: AbstractCoordinateConverter{F}

To convert from NormalizedCoordinates (using some type) to PixelCoordinates (using F).

---

    NormalizedToPixelConverter(::Type{F}, size) where F

Construct a NormalizedToPixelConverter{F} for a screen of size `size` along the
currently considered axis (i.e. the width or height), in pixels.
"""
struct NormalizedToPixelConverter{F} <: AbstractCoordinateConverter{F}
    # x -> (x + 1) * s/2
    factor::F

    NormalizedToPixelConverter(::Type{F}, size) where F = new{F}(F(size) / 2)
end

"""
    NormalizedToNormalizedConverter{F} <: AbstractCoordinateConverter{F}

To convert from NormalizedCoordinates (using some type) to NormalizedCoordinates (using F).

---

    NormalizedToNormalizedConverter(::Type{F}) where F = new{F}()

Construct a NormalizedToNormalizedConverter{F}.

---

    NormalizedToNormalizedConverter(::Type{F}) where F = new{F}()

Construct a NormalizedToNormalizedConverter{F}. The second parameter is for consistency
with e.g. [`NormalizedToPixelConverter`](@ref).
"""
struct NormalizedToNormalizedConverter{F} <: AbstractCoordinateConverter{F}
    NormalizedToNormalizedConverter(::Type{F}) where F = new{F}()
    NormalizedToNormalizedConverter(::Type{F}, _) where F = new{F}()
end
# Alternatively, we could use a parametrized struct
#   CoordinateConverter{FROM, TO, F}
# but then the fields need to be the same for all FROM and TO.

"""
    create_coordinate_converter(
    ::Type{<:AbstractImageCoordinates}, ::Type{AbstractImageCoordinates},
    ::Type{F}, size
) where F

Create the appropriate `AbstractCoordinateConverter{F}` from the provided image
coordinate systems, and axis `size` (width or height) in pixels.
"""
function create_coordinate_converter end

@inline create_coordinate_converter(
    ::Type{PixelCoordinates}, ::Type{PixelCoordinates},
    ::Type{F}, size
) where F = PixelToPixelConverter(F, size)
@inline create_coordinate_converter(
    ::Type{PixelCoordinates}, ::Type{NormalizedCoordinates},
    ::Type{F}, size
) where F = PixelToNormalizedConverter(F, size)
@inline create_coordinate_converter(
    ::Type{NormalizedCoordinates}, ::Type{PixelCoordinates},
    ::Type{F}, size
) where F = NormalizedToPixelConverter(F, size)
@inline create_coordinate_converter(
    ::Type{NormalizedCoordinates}, ::Type{NormalizedCoordinates},
    ::Type{F}, size
) where F = NormalizedToNormalizedConverter(F, size)


@inline convert_coordinate(::PixelToPixelConverter{F}, coordinate) where F = F(coordinate)
@inline convert_coordinate(p2n::PixelToNormalizedConverter{F}, coordinate) where F = 
    F(p2n.factor * coordinate) - one(F)
@inline convert_coordinate(n2p::NormalizedToPixelConverter{F}, coordinate) where F = 
    F(n2p.factor * (coordinate + one(F)))
@inline convert_coordinate(::NormalizedToNormalizedConverter{F}, coordinate) where F = 
    F(coordinate)

@inline (cc::AbstractCoordinateConverter)(coordinate) = convert_coordinate(cc, coordinate)


@inline convert_length(::PixelToPixelConverter{F}, length) where F = F(length)
@inline convert_length(p2n::PixelToNormalizedConverter{F}, length) where F = 
    F(p2n.factor * length)
@inline convert_length(n2p::NormalizedToPixelConverter{F}, length) where F = 
    F(n2p.factor * length)
@inline convert_length(::NormalizedToNormalizedConverter{F}, length) where F = 
    F(length)


@inline convert_square(::PixelToPixelConverter{F}, sq) where F = F(sq)
@inline convert_square(p2n::PixelToNormalizedConverter{F}, sq) where F = 
    F(p2n.factor ^ 2 * sq)
@inline convert_square(n2p::NormalizedToPixelConverter{F}, sq) where F = 
    F(n2p.factor ^ 2 * sq)
@inline convert_square(::NormalizedToNormalizedConverter{F}, sq) where F = 
    F(sq)