# Our GUI struct is based mostly on GlfwOpenGLBackend.jl. In particular, we split its
# renderloop method

# TODO: help widget, containing a.o. the controls

OGLBackend = Base.get_extension(ig, :GlfwOpenGLBackend)

"""
    AbstractGUIImage{D <: ArrayDomain, I <: Integer}

Abstract type to represent images we can display in our GUI.
"""
abstract type AbstractGUIImage{D <: ArrayDomain, I <: Integer} end

"""
    CPUGUIImage{
        I <: Integer, A <: AbstractArray{UInt8, 3}
    } <: AbstractGUIImage{CPU, I}

CPU AbstractGUIImage backed by a 3D array.

---

    CPUGUIImage(
        array::AbstractArray{UInt8, 3}, texture_ref::lib.ImTextureRef; 
        int_type::Type{<:Integer} = Int32
    )

Construct a `CPUGUIImage` from a 3D array and CImGui texture reference.
"""
mutable struct CPUGUIImage{
    I <: Integer, A <: AbstractArray{UInt8, 3}
} <: AbstractGUIImage{CPU, I}
    array::A
    tex_ref::lib.ImTextureRef

    function CPUGUIImage(
        array::AbstractArray{UInt8, 3}, texture_ref::lib.ImTextureRef; 
        int_type::Type{<:Integer} = Int32
    )
        D = array_domain(array)
        D == CPU || throw(ArgumentError(
            "A `CPUGUIImage's `array` should reside on the CPU! Received: $D"))
        return new{int_type, typeof(array)}(array, texture_ref)
    end
end

"""
    GPUGUIImage{
        M <: CUDACore.AbstractMemory,
        I <: Integer
    } <: AbstractGUIImage{GPU{M}, I}

GPU AbstractGUIImages backed by an OpenGL PBO (Pixel Buffer Object).
"""
mutable struct GPUGUIImage{
    M <: CUDACore.AbstractMemory,
    I <: Integer
} <: AbstractGUIImage{GPU{M}, I}
    width::I
    height::I
    pbo::GL.GLuint
    cu_resource::CUDACore.CUgraphicsResource
    tex_ref::lib.ImTextureRef
end

@inline get_width(c::CPUGUIImage{I}) where I = I(size(c.array, 2))
@inline get_height(c::CPUGUIImage{I}) where I = I(size(c.array, 3))
@inline get_width(g::GPUGUIImage) = g.width
@inline get_height(g::GPUGUIImage) = g.height

"""
    create_gui_image(::Type{D}, width::I, height::I) where {D <: ArrayDomain, I}

Construct a GPU/GPUGUIImage (containing garbage) from the provided dimensions.
"""
function create_gui_image(::Type{CPU}, width::I, height::I) where I
    tex_ref = ig.create_image_texture(width, height)
    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
    # Our image will be tightly packed, without padding for alignment
    return CPUGUIImage(create_array(CPU, UInt8, (3, width, height)), tex_ref; int_type = I)
end

function create_gui_image(::Type{GPU{M}}, width::I, height::I) where {M, I}
    tex_ref = ig.create_image_texture(width, height)
    pbo, cu_resource = initialize_pbo(width, height)
    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)  
    return GPUGUIImage{M, I}(width, height, pbo, cu_resource, tex_ref)
end

# Based on https://discourse.julialang.org/t/cuarray-glmakie/52461/10
"""
    initialize_pbo(width, height)

Create an OpenGL Pixel Buffer Object (PBO) for RGB images of the provided dimensions.
Returns a tuple consisting of the PBO itself, and a `CUgraphicsResource` we can interact
with from CUDA.jl by wrapping it to a `CuArray`. 
"""
function initialize_pbo(width, height)
    
    pbo_ref = Ref{GL.GLuint}()
    GL.glGenBuffers(1, pbo_ref)  
    # Declare our intent to create 1 buffer: pbo
    pbo = pbo_ref[]
    GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, pbo)  
    # Subsequent glBuffer operations (i.e. the following line) will refer to pbo
    GL.glBufferData(
        GL.GL_PIXEL_UNPACK_BUFFER, 
        3 * width * height, C_NULL, GL.GL_DYNAMIC_COPY)  
    # Allocate memory (in bytes) for empty buffer
    GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)  
    # Stop referring to pbo (to avoid accidental manipulations)

    # Register PBO with CUDA
    cu_resource_ref = Ref{CUDACore.CUgraphicsResource}()
    CUDACore.cuGraphicsGLRegisterBuffer(
        cu_resource_ref, pbo, CUDACore.CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD)

    return pbo, cu_resource_ref[]
end

"""
    free_pbo!(pbo, cu_resource)

Free the provided PBO and CUgraphicsResource. Undoes the [`initialize_pbo`](@ref).
"""
function free_pbo!(pbo, cu_resource)
    CUDACore.cuGraphicsUnregisterResource(cu_resource)
    pbo_ref = Ref{GL.GLuint}(pbo)
    GL.glDeleteBuffers(1, pbo_ref)
end

"""
    resize!(::AbstractGUIImage{D, I}, width, height) where {D, I}

Resize the provided GUIImage to the new dimensions. This does not preserve the current
content of the buffers.

Noop if the new dimensions are the same as the old ones.
"""
function resize!(c::CPUGUIImage{I}, width, height) where I
    width == get_width(c) && height == get_height(c) && return

    c.array = resize_memory(c, (3, width, height))
    ig.destroy_image_texture(c.tex_ref)
    c.tex_ref = ig.create_image_texture(width, height)
end


function resize!(g::GPUGUIImage{I}, width, height) where I
    width == get_width(g) && height == get_height(g) && return

    g.width = width
    g.height = height
    ig.destroy_image_texture(g.tex_ref)
    free_pbo!(g.pbo, g.cu_resource)
    g.tex_ref = ig.create_image_texture(width, height)
    g.pbo, g.cu_resource = initialize_pbo(width, height)
end

"""
    set_gl_interpolation_mode(image::AbstractGUIImage, mode::Cuint)

Set the OpenGL interpolation mode for the provided GUIImage to the provided value
(e.g. GL.GL_LINEAR or GL.GL_NEAREST).
"""
function set_gl_interpolation_mode(image::AbstractGUIImage, mode::Cuint)
    GL.glBindTexture(GL.GL_TEXTURE_2D, image.tex_ref._TexID)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, mode)
    # upscaling
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, mode)
    # downscaling
    GL.glBindTexture(GL.GL_TEXTURE_2D, 0)  # unbind
end

"""
    GUIInterpolationMode::Cint
        gim_linear = 0
        gim_nearest = 1

Enum for the OpenGL interpolation modes we support.

---

    GUIInterpolationMode(::Cint)

Construct a `GUIInterpolationMode` from a valid underlying `Cint` (0 or 1).
"""
@enum GUIInterpolationMode::Cint gim_linear gim_nearest  # starts from 0

"""
    GUIInterpolationMode(GL_code::UInt32)

Construct a `GUIInterpolationMode` from a supported OpenGL mode code (`GL.GL_LINEAR` or 
`GL.GL_NEAREST`).
"""
function GUIInterpolationMode(GL_code::UInt32)
    GL_code == GL.GL_NEAREST && return gim_nearest
    return gim_linear  # default (for GL.GL_LINEAR)
end  # (Note: we now have two constructors: ::CInt (Int32) and ::UInt32)

"""
    to_GL(gim::GUIInterpolationMode)

Convert a `GUIInterpolationMode` to the corresponding OpenGL interpolation mode code.
"""
function to_GL(gim::GUIInterpolationMode)
    gim == gim_nearest && return GL.GL_NEAREST
    return GL.GL_LINEAR
end

"""
    to_string(gim::GUIInterpolationMode)

Convert a `GUIInterpolationMode` to a short string representation.
"""
function to_string(gim::GUIInterpolationMode)
    gim == gim_linear && return "Linear"
    gim == gim_nearest && return "Nearest"
    return "?"
end

"""
    Enum GUIBoundaryClippingMode::Cint:
    gbcm_clamp = 0
    gbcm_filter = 1

Enum for the GUI representation of the corresponding render option
[`BoundaryClippingMode`](@ref).

---

    GUIBoundaryClippingMode(::Cint)

Construct a `GUIBoundaryClippingMode` from a valid underlying `Cint` (0 or 1).
"""
@enum GUIBoundaryClippingMode::Cint gbcm_clamp gbcm_filter

"""
    GUIBoundaryClippingMode(gbcm::BoundaryClippingMode)

Construct a `GUIBoundaryClippingMode` from the corresponding render option
[`BoundaryClippingMode`](@ref) (struct).
"""
function GUIBoundaryClippingMode(gbcm::BoundaryClippingMode)
    gbcm isa BoundaryFilter && return gbcm_filter
    return gbcm_clamp
end

"""
    to_string(gbcm::GUIBoundaryClippingMode)

Convert a `GUIBoundaryClippingMode` to a short string representation.
"""
function to_string(gbcm::GUIBoundaryClippingMode)
    gbcm == gbcm_clamp && return "Clamp"
    gbcm == gbcm_filter && return "Filter"
    return "?"
end

"""
    to_render_options_boundary_clipping_mode(gbcm::GUIBoundaryClippingMode)

Convert a `GUIBoundaryClippingMode` to the corresponding render option
[`BoundaryClippingMode`](@ref) (struct).
"""
function to_render_options_boundary_clipping_mode(gbcm::GUIBoundaryClippingMode)
    gbcm == gbcm_filter && return BoundaryFilter
    return CovarianceBoundaryClamp
end

"""
    Enum GUIRenderMode::Cint:
        grm_rgb = 0
        grm_transmittance = 1
        grm_depth = 2

Enum for the GUI render mode (corresponding to `RenderOutputs`).

---

    GUIRenderMode(::Cint)

Construct a `GUIRenderMode` from a valid underlying `Cint` (0, 1 or 2).
"""
@enum GUIRenderMode::Cint grm_rgb grm_transmittance grm_depth

"""
    to_string(grm::GUIRenderMode)

Convert a `GUIRenderMode` to a short string representation.
"""
function to_string(grm::GUIRenderMode)
    grm == grm_rgb && return "RGB"
    grm == grm_transmittance && return "Transmittance"
    grm == grm_depth && return "Depth"
    return "?"
end

"""
    create_render_outputs(
        grm::GUIRenderMode, render_width, render_height,
        ::Type{D}, ::Type{I}, ::Type{F}
    ) where {D <: ArrayDomain, I <: Integer, F <: AbstractFloat}

Create a `MutableRenderOutputs{D, I, F}` corresponding to our `GUIRenderMode`, for theµ
provided render resolution.
"""
function create_render_outputs(
    grm::GUIRenderMode, render_width, render_height,
    ::Type{D}, ::Type{I}, ::Type{F}
) where {D <: ArrayDomain, I <: Integer, F <: AbstractFloat}
    if grm == grm_transmittance
        return MutableRenderOutputs(
            render_width, render_height; 
            array_domain = D, int_type = I, float_type = F,
            include_rgb = false,
            include_transmittance = true,
            include_depth = false,
            include_last_contributing_indices = false
        )
    elseif grm == grm_depth
        return MutableRenderOutputs(
            render_width, render_height; 
            array_domain = D, int_type = I, float_type = F,
            include_rgb = false,
            include_transmittance = D == CPU,
            include_depth = true,
            include_last_contributing_indices = false
        )
    else  # grm_rgb
        return MutableRenderOutputs(
            render_width, render_height; 
            array_domain = D, int_type = I, float_type = F,
            include_rgb = true,
            include_transmittance = D == CPU,
            include_depth = false,
            include_last_contributing_indices = false
        )
    end
end

has_rgb_image(grm::GUIRenderMode, _) = grm == grm_rgb
has_transmittance_image(grm::GUIRenderMode, ::Type{D}) where D <: ArrayDomain = 
    grm == grm_transmittance || D == CPU 
has_depth_image(grm::GUIRenderMode, _) = grm == grm_depth


# on_gpu is known at parse time (i.e. a literal or cosnt)
"""
    maybe_timed(on_gpu, should_time, ex)

Macro to conditionally time the provided expression `ex`, depending on the runtime value
of `should_time`. `on_gpu` should be known at parse time (i.e. a literal or `const` `true`
or `false`), indicating whether we should use `@timed` or `CUDA.@timed` (with
synchronization).
"""
macro maybe_timed(on_gpu, should_time, ex)
    timed_macro = on_gpu ? 
        GlobalRef(Base, Symbol("@timed")) : 
        GlobalRef(CUDA, Symbol("@timed")) 
    # (Note that :(@timed)) is an Expr (with an invalid macro call), not a Symbol)
    timed_ex = Expr(:macrocall, timed_macro, __source__, esc(ex))
    return quote
        if $(esc(should_time))
            $timed_ex.time
        else
            $(esc(ex))
            0.
        end
    end
end

# Implement update!, close!, has_closed, is_running
"""
    AbstractGUI

Abstract supertype for our GUIs. Concrete subtypes should implement `update!`, `close!`,
`has_closed` and `is_running`.
"""
abstract type AbstractGUI end

"""
    GUI{
        D <: ArrayDomain,
        I <: Integer, 
        F <: AbstractFloat,
        GV <: AbstractGaussiansVector{D, I, F},
        RB <: AbstractRenderBuffers{D, I, F},
        GUI_IMG <: AbstractGUIImage{D, I}
    } <: AbstractGUI

Concrete GUI struct. The CPU version is very slow.
"""
mutable struct GUI{
    D <: ArrayDomain,
    I <: Integer, 
    F <: AbstractFloat,
    GV <: AbstractGaussiansVector{D, I, F},
    RB <: AbstractRenderBuffers{D, I, F},
    GUI_IMG <: AbstractGUIImage{D, I}
} <: AbstractGUI

    gaussians::GV

    # Render camera parameters
    camera_fov_x_deg::F
    camera_fov_y_deg::F
    camera_skew_angle_deg::F  # 0°: no skew 
    camera_principal_point_x_ndc::F
    camera_principal_point_y_ndc::F
    camera_near_z::F
    camera_far_z::F

    # Our camera manipulation transformations are most naturally described using the new
    # axes in the old reference system, i.e. we have a transformation from new to old. As we
    # want to end up with new to world or world to new, it then makes sense to use . to
    # world:
    #   new to world = old to world * new to old
    # (Cf. move(::PinholeCamera) and rotate(::PinholeCamera))
    camera_to_world_rotation::SMatrix{3, 3, F, 9}
    camera_to_world_translation::SVector{3, F}

    image::GUI_IMG

    # Tiling parameters
    # (Not static: will incur dynamic dispatch)
    tile_width::I
    tile_height::I

    # Render options
    use_antialiasing::Bool 
    sh_degree::I
    boundary_clipping_mode::GUIBoundaryClippingMode
    clipping_min_x_ndc::F
    clipping_max_x_ndc::F
    clipping_min_y_ndc::F
    clipping_max_y_ndc::F
    render_min_alpha::F
    render_max_alpha::F
    render_min_transmittance::F
    render_dilation_x_px::F
    render_dilation_y_px::F
    background_color::Vector{Cfloat}  # of length 3


    render_outputs::MutableRenderOutputs  
    # Abstract type, as we can switch included outputs
    # Contains precisely one output (except for CPU, as we are forced to include
    # transmittance)
    render_mode::GUIRenderMode

    render_buffers::RB

    has_stopped::Bool

    last_timestep::Cdouble  # i.e. Float64
    translation_speed::F
    rotation_speed::F
    link_fovs::Bool
    # (Could alternatively use @cstatic within a widget to keep a 'local' variable alive for
    # the following frame)
    fix_aspect_ratio::Bool
    link_render_and_display_size::Bool
    display_interpolation_mode::GUIInterpolationMode

    visualization_min_depth::F
    visualization_max_depth::F

    time_rendering::Bool
    ewma_render_time::Float64  # Exponentially weighted moving average
    ewma_weight::Float64       # Weight for current average
    nb_timed_frames::Int       # For bias correction
    last_render_time::Float64
    render_time_history::Vector{Cfloat}  # Of the smoothed, bias-corrected timings
    # Cfloat to display using ig
    # TODO: circular?
    render_time_history_display_length::Int

    ctx::Ptr{ig.lib.ImGuiContext}
    wait_events::Bool

    window::GLFW.Window
    window_background_color::SVector{4, Cfloat}  # i.e. Float32
    vsync::Bool
end

"""
    GUI(
        gaussians::AbstractGaussiansVector{D, I, F};

        render_width = I(1280),
        render_height = I(720),
        camera_fov_x_deg = F(90.),
        camera_fov_y_deg = focal_length_to_fov_deg(
            fov_deg_to_focal_length(camera_fov_x_deg, render_width), render_height),
        camera_skew_angle_deg = F(0.),
        camera_principal_point_x_ndc = F(0.),
        camera_principal_point_y_ndc = F(0.),
        camera_near_z = F(DEFAULT_NEAR_CLIPPING_PLANE_F64),
        camera_far_z = F(DEFAULT_FAR_CLIPPING_PLANE_F64),
        camera_to_world_rotation = one(SMatrix{3, 3, F, 9}),
        camera_to_world_translation = zero(SVector{3, F}),

        tile_width = nothing,
        tile_height = nothing,

        use_antialiasing = false,
        sh_degree = get_sh_degree(gaussians),
        boundary_clipping_mode::GUIBoundaryClippingMode = gbcm_clamp,
        clipping_min_x_ndc = F(-1 - DEFAULT_CLIPPING_MARGIN_NC_F64),
        clipping_max_x_ndc = F(1 + DEFAULT_CLIPPING_MARGIN_NC_F64),
        clipping_min_y_ndc = F(-1 - DEFAULT_CLIPPING_MARGIN_NC_F64),
        clipping_max_y_ndc = F(1 + DEFAULT_CLIPPING_MARGIN_NC_F64),
        render_min_alpha = F(DEFAULT_MIN_ALPHA_F64),
        render_max_alpha = F(DEFAULT_MAX_ALPHA_F64),
        render_min_transmittance = F(DEFAULT_MIN_TRANSMITTANCE_F64),
        render_dilation_x_px = F(DEFAULT_DILATION_PX_F64),
        render_dilation_y_px = F(DEFAULT_DILATION_PX_F64),
        background_color = zero(SVector{3, F}),

        render_mode::GUIRenderMode = grm_rgb,

        translation_speed = F(1.),
        rotation_speed = F(0.5),
        link_fovs = true,
        fix_aspect_ratio = true,
        link_render_and_display_size = false,
        display_interpolation_mode::GUIInterpolationMode = gim_linear,

        visualization_min_depth = zero(F),
        visualization_max_depth = F(10.),

        time_rendering = true,
        ewma_weight = 0.99,
        render_time_history_display_length = 1000,

        window_background_color = SVector{4, Cfloat}(0.45, 0.55, 0.60, 1.00),
        window_width = 1280,
        window_height = 720,
        window_title = "ThreeDGaussianSplatting.jl",
        opengl_version = v"3.2",
        wait_events = false,
        vsync = true
    ) where {
        D <: ArrayDomain, I <: Integer, F <: AbstractFloat
    }

Construct a GUI from the provided `GaussiansVector` and keyword arguments. Many of these
kwargs can also be changed afterwards by interacting with the GUI.
"""
function GUI(
    gaussians::AbstractGaussiansVector{D, I, F};

    render_width = I(1280),  # TODO: restore from previous (cf. widgets)?
    render_height = I(720),
    camera_fov_x_deg = F(90.),
    camera_fov_y_deg = focal_length_to_fov_deg(
        fov_deg_to_focal_length(camera_fov_x_deg, render_width), render_height),
    camera_skew_angle_deg = F(0.),
    camera_principal_point_x_ndc = F(0.),
    camera_principal_point_y_ndc = F(0.),
    camera_near_z = F(DEFAULT_NEAR_CLIPPING_PLANE_F64),
    camera_far_z = F(DEFAULT_FAR_CLIPPING_PLANE_F64),
    camera_to_world_rotation = one(SMatrix{3, 3, F, 9}),
    camera_to_world_translation = zero(SVector{3, F}),

    tile_width = nothing,
    tile_height = nothing,

    use_antialiasing = false,
    sh_degree = get_sh_degree(gaussians),
    boundary_clipping_mode::GUIBoundaryClippingMode = gbcm_clamp,
    clipping_min_x_ndc = F(-1 - DEFAULT_CLIPPING_MARGIN_NC_F64),
    clipping_max_x_ndc = F(1 + DEFAULT_CLIPPING_MARGIN_NC_F64),
    clipping_min_y_ndc = F(-1 - DEFAULT_CLIPPING_MARGIN_NC_F64),
    clipping_max_y_ndc = F(1 + DEFAULT_CLIPPING_MARGIN_NC_F64),
    render_min_alpha = F(DEFAULT_MIN_ALPHA_F64),
    render_max_alpha = F(DEFAULT_MAX_ALPHA_F64),
    render_min_transmittance = F(DEFAULT_MIN_TRANSMITTANCE_F64),
    render_dilation_x_px = F(DEFAULT_DILATION_PX_F64),
    render_dilation_y_px = F(DEFAULT_DILATION_PX_F64),
    background_color = zero(SVector{3, F}),

    render_mode::GUIRenderMode = grm_rgb,

    translation_speed = F(1.),  # in world units / s,
    rotation_speed = F(0.5),    # in rad / s
    link_fovs = true,
    fix_aspect_ratio = true,
    link_render_and_display_size = false,
    display_interpolation_mode::GUIInterpolationMode = gim_linear,

    visualization_min_depth = zero(F),
    visualization_max_depth = F(10.),

    time_rendering = true,
    ewma_weight = 0.99,
    render_time_history_display_length = 1000,

    window_background_color = SVector{4, Cfloat}(0.45, 0.55, 0.60, 1.00),
    window_width = 1280,  # TODO: default to fullscreen (or compatible with render window?)
    window_height = 720,
    window_title = "ThreeDGaussianSplatting.jl",
    opengl_version = v"3.2",
    wait_events = false,
    vsync = true
) where {
    D <: ArrayDomain, I <: Integer, F <: AbstractFloat
}
    if OGLBackend.GLFW_VERSION >= v"3.4.4"
        # We leave thread-safety to the user
        GLFW.ENABLE_THREAD_ASSERTIONS[] = false
    end

    # Validate arguments
    if Sys.isapple() && opengl_version < v"3.2"
        throw(ArgumentError(
            "Only OpenGL 3.2+ is supported on OSX, but $(opengl_version) was requested"))
    end

    ig.set_backend(:GlfwOpenGL3)
    ctx = ig.CreateContext()

    glsl_version = OGLBackend.get_glsl_version(opengl_version)
    GLFW.WindowHint(GLFW.VISIBLE, true)
    GLFW.WindowHint(GLFW.DECORATED, true)
    GLFW.WindowHint(GLFW.CONTEXT_VERSION_MAJOR, opengl_version.major)
    GLFW.WindowHint(GLFW.CONTEXT_VERSION_MINOR, opengl_version.minor)

    if Sys.isapple()
        GLFW.WindowHint(GLFW.OPENGL_PROFILE, GLFW.OPENGL_CORE_PROFILE)
        GLFW.WindowHint(GLFW.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
    end

    # Create window
    window = GLFW.CreateWindow(window_width, window_height, window_title)
    OGLBackend._window = window 
    @assert window != C_NULL
    GLFW.MakeContextCurrent(window)
    if vsync
        GLFW.SwapInterval(1)
    else
        GLFW.SwapInterval(0)
    end

    # Setup Platform/Renderer bindings
    lib.ImGui_ImplGlfw_InitForOpenGL(Ptr{lib.GLFWwindow}(window.handle), true)
    lib.ImGui_ImplOpenGL3_Init("#version $(glsl_version)")

    image = create_gui_image(D, render_width, render_height)
    if isnothing(tile_width) || isnothing(tile_height)
        tile_width, tile_height = get_default_tile_size(render_width, render_height)
    end
    render_buffers = MutableRenderBuffers(
        gaussians, 
        Tiling(render_width, render_height, tile_width, tile_height; int_type = I)
    )

    render_outputs = create_render_outputs(
        render_mode, render_width, render_height, D, I, F)

    GUI(
        gaussians,

        F(camera_fov_x_deg), F(camera_fov_y_deg),
        F(camera_skew_angle_deg),
        F(camera_principal_point_x_ndc), F(camera_principal_point_y_ndc),
        F(camera_near_z), F(camera_far_z),
        convert(SMatrix{3, 3, F, 9}, camera_to_world_rotation),
        convert(SVector{3, F}, camera_to_world_translation),
        image,

        I(tile_width), I(tile_height),

        use_antialiasing,
        I(sh_degree),
        boundary_clipping_mode,
        F(clipping_min_x_ndc), F(clipping_max_x_ndc),
        F(clipping_min_y_ndc), F(clipping_max_y_ndc),
        F(render_min_alpha),
        F(render_max_alpha),
        F(render_min_transmittance),
        F(render_dilation_x_px),
        F(render_dilation_y_px),
        convert(Vector{Cfloat}, view(background_color, 1:3)),

        render_outputs,
        render_mode,
        render_buffers,

        false,
        -1.,
        F(translation_speed), F(rotation_speed),
        link_fovs,
        fix_aspect_ratio,
        link_render_and_display_size,
        display_interpolation_mode,

        F(visualization_min_depth), F(visualization_max_depth),

        time_rendering,
        0.,
        Float64(ewma_weight),
        0,
        0.,
        Vector{Cfloat}(undef, 0),
        Int(render_time_history_display_length),

        ctx,
        wait_events,

        window,
        window_background_color,
        vsync
    )
end

@inline array_domain(::GUI{D}) where D = D
@inline int_type(::GUI{<:Any, I}) where I = I
@inline float_type(::GUI{<:Any, <:Any, F}) where F = F

@inline get_render_width(gui::GUI) = get_width(gui.image)
@inline get_render_height(gui::GUI) = get_height(gui.image)

@inline has_stopped(gui::GUI) = gui.has_stopped
@inline is_running(gui::GUI) = !has_stopped(gui)

"""
    get_camera(gui::GUI)

Return the current rendering `PinholeCamera` used in our GUI.
"""
function get_camera(gui::GUI)
    F = float_type(gui)
    width = get_render_width(gui)
    height = get_render_height(gui)
    ndc_to_px_x = NormalizedToPixelConverter(F, width)
    ndc_to_px_y = NormalizedToPixelConverter(F, height)
    fx = fov_deg_to_focal_length(gui.camera_fov_x_deg, width)
    fy = fov_deg_to_focal_length(gui.camera_fov_y_deg, height)

    return PinholeCamera(
        width, height,
        fx, fy,
        fy * tand(gui.camera_skew_angle_deg),  # inverse of get_skew_angle(::PinholeCamera)
        ndc_to_px_x(gui.camera_principal_point_x_ndc),
        ndc_to_px_y(gui.camera_principal_point_y_ndc),
        gui.camera_near_z, gui.camera_far_z,
        gui.camera_to_world_rotation, gui.camera_to_world_translation
    )
end

"""
    set_camera!(gui::GUI, camera::PinholeCamera)

Set the parameters for the rendering camera used by our GUI.
"""
function set_camera!(gui::GUI, camera::PinholeCamera)
    F = float_type(gui)
    width = get_width(camera)
    height = get_height(camera)
    px_to_ndc_x = PixelToNormalizedConverter(F, width)
    px_to_ndc_y = PixelToNormalizedConverter(F, height)
    
    set_render_resolution!(gui, width, height)
    gui.camera_fov_x_deg = get_fov_x_deg(camera)
    gui.camera_fov_y_deg = get_fov_y_deg(camera)
    gui.camera_skew_angle_deg = get_skew_angle(camera)
    gui.camera_principal_point_x_ndc = px_to_ndc_x(get_principal_point_x(camera))
    gui.camera_principal_point_y_ndc = px_to_ndc_y(get_principal_point_y(camera))
    gui.camera_near_z = get_near_z(camera)
    gui.camera_far_z = get_far_z(camera)
    gui.camera_to_world_rotation = get_camera_to_world_rotation(camera)
    gui.camera_to_world_translation = get_camera_to_world_translation(camera)
end

"""
    get_tiling(gui::GUI)

Return the `Tiling` we use for rendering.
"""
function get_tiling(gui::GUI)
    return Tiling(get_render_width(gui), get_render_height(gui), 
                  gui.tile_width, gui.tile_height)
end

"""
    get_render_options(gui::GUI)

Return the `RenderOptions` we use for rendering in our GUI.
"""
function get_render_options(gui::GUI)
    return RenderOptions(
        get_render_width(gui), get_render_height(gui),
        gui.sh_degree, 
        to_render_options_boundary_clipping_mode(gui.boundary_clipping_mode);
        background_color = SVector{3}(gui.background_color),
        use_antialiasing = gui.use_antialiasing,
        min_alpha = gui.render_min_alpha,
        max_alpha = gui.render_max_alpha,
        min_transmittance = gui.render_min_transmittance,
        dilation_x_px = gui.render_dilation_x_px,
        dilation_y_px = gui.render_dilation_y_px,
        clip_x_min_nc = gui.clipping_min_x_ndc,
        clip_x_max_nc = gui.clipping_max_x_ndc,
        clip_y_min_nc = gui.clipping_min_y_ndc,
        clip_y_max_nc = gui.clipping_max_y_ndc,
        int_type = int_type(gui), float_type = float_type(gui)
    )
end

"""
    set_render_resolution!(gui::GUI, width, height)

Set the rendering resolution of our GUI to the provided dimensions. In other words,
afterwards `get_render_width` and `get_render_height` will return `width` and `height`.
"""
function set_render_resolution!(gui::GUI, width, height)
    resize!(gui.image, width, height)  # Automatically adjusts get_render_width
    resize!(gui.render_outputs, (height, width))
    set_nb_tiles!(gui.render_buffers, get_nb_tiles(get_tiling(gui)))  # Using the new tiling
end

"""
    set_gl_interpolation_mode!(gui::GUI, mode::GUIInterpolationMode)

Set the OpenGL interpolation mode used by the GUI for displaying images at a different
resolution than they were rendered.
"""
function set_gl_interpolation_mode!(gui::GUI, mode::GUIInterpolationMode)
    gui.display_interpolation_mode == mode && return
    gui.display_interpolation_mode = mode
    set_gl_interpolation_mode(gui.image, to_GL(mode))
end

"""
    render_image!(gui::GUI{<:Any, I, F}, display_size) where {I, F}

Call the [`render!`](@ref) function and convert the output to the format expected by
OpenGL. The rendered image will be displayed at `display_size`, which might differ from
the rendering resolution.
"""
function render_image!(gui::GUI{CPU, <:Any, F}, display_size) where F
    gui.last_render_time = @maybe_timed false gui.time_rendering render!(
        gui.render_outputs,
        gui.gaussians,
        get_camera(gui),
        get_tiling(gui),
        get_render_options(gui),
        gui.render_buffers
    )
    w = get_render_width(gui); h = get_render_height(gui)
    if has_rgb_image(gui.render_outputs)
        gui.image.array .= round.(
            UInt8, 
            min.(
                permutedims(get_rgb_image(gui.render_outputs), (1, 3, 2)),
                one(F)
            ) .* F(255))
        # c x h x w F -> c x w x h UInt8
    elseif has_transmittance_image(gui.render_outputs)  # (and not RGB)
        # h x w F ([0, 1]) -> c x w x h UInt8
        gui.image.array .= round.(
            UInt8, 
            reshape(get_transmittance_image(gui.render_outputs), 1, h, w) .* F(255))
        # Note: will apply the round computation 3 * h * w times, but does not materialize
        # anything. Alternatively,
        #    reshape(
        #       round.(UInt8, get_transmittance_image(gui.render_outputs) .* F(255)), 
        #       1, h, w
        #   )
        # only performs h * w computations, but materializes the array.
        # TODO: use some other lazy construction, or just loop
    elseif has_depth_image(gui.render_outputs)
        # h x w F ([0, ...)) -> c x w x h UInt8
        gui.image.array .= round.(
            UInt8, 
            (one(F) .- 
              clamp01.((reshape(get_transmittance_image(gui.render_outputs), 1, h, w) .- 
                 gui.visualization_min_depth) ./ 
                   (gui.visualization_max_depth - gui.visualization_min_depth)
                       )) .* F(255))
    end

    ig.update_image_texture(
        gui.image.tex_ref, gui.image.array, get_width(gui.image), get_height(gui.image);
        format = GL.GL_RGB, type = GL.GL_UNSIGNED_BYTE)
    ig.Image(gui.image.tex_ref, display_size)
    return nothing
end

function render_image!(gui::GUI{<:GPU, I, F}, display_size) where {I, F}
    # First map and wrap the pbo to a CuArray
    CUDACore.cuGraphicsMapResources(1, [gui.image.cu_resource], CUDA.stream())
    # TODO: needs to be a Vector?
    
    ptr_ref = Ref{CUDACore.CUdeviceptr}()
    nb_bytes_ref = Ref{Csize_t}()
    CUDACore.cuGraphicsResourceGetMappedPointer_v2(
        ptr_ref, nb_bytes_ref, gui.image.cu_resource)
    # Get mapped pointer and buffer size
    # We don't actually need nb_bytes, though, as this is just 
    # 3 * gui.image.height * gui.image.width * sizeof(UInt8),
    # with the latter factor being 1

    ptr = reinterpret(CuPtr{UInt8}, ptr_ref[])
    wrapped_pbo = unsafe_wrap(
        CuArray, ptr, 
        (3, Int(get_width(gui.image)), Int(get_height(gui.image))))
    # ! width x height; unsafe_wrap's dims must be NTuple{⋅, Int}

    # Now render our 3DGS model and write it to wrapped_pbo, in the appropriate format
     # TODO: render! (tile kernel) with direct UInt8 output (clamped), in 3 x h x w order
     # by using a different sink (RenderOutputs)
    gui.last_render_time = @maybe_timed true gui.time_rendering render!(
        gui.render_outputs,
        gui.gaussians,
        get_camera(gui),
        get_tiling(gui),
        get_render_options(gui),
        gui.render_buffers
    )

    # Convert from floating point (F) [3 x] height x width format to 
    # UInt8 3 x width x height
    threads = (32, 8)
        blocks = (cld(get_width(gui.image), threads[1]), cld(get_height(gui.image), threads[2]))
    if has_rgb_image(gui.render_outputs)
        @cuda threads blocks convert_rgb_to_gl_img_kernel!(
            wrapped_pbo, get_rgb_image(gui.render_outputs), I, F)
    elseif has_transmittance_image(gui.render_outputs)
        @cuda threads blocks convert_transmittance_to_gl_img_kernel!(
            wrapped_pbo, get_transmittance_image(gui.render_outputs), I, F)
    elseif has_depth_image(gui.render_outputs)
         @cuda threads blocks convert_depth_to_gl_img_kernel!(
            wrapped_pbo, get_depth_image(gui.render_outputs), 
            gui.visualization_min_depth, gui.visualization_max_depth,
            I, F)
    end

    # Unwrap (the mapping and unmapping every time is necessary for synchronization)
    CUDACore.cuGraphicsUnmapResources(1, [gui.image.cu_resource], CUDA.stream())

    # Update the texture
    GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, gui.image.pbo)
    ig.update_image_texture(
        gui.image.tex_ref, C_NULL, get_width(gui.image), get_height(gui.image);
        format = GL.GL_RGB, type = GL.GL_UNSIGNED_BYTE)
    # C_NULL as we don't pass data directly, but use the bound buffer
    GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)

    ig.Image(gui.image.tex_ref, display_size)

    return nothing
end

"""
    convert_rgb_to_gl_img_kernel!(cwh_ui8, chw_f, ::Type{I}, ::Type{F}) where {I, F}

Kernel for converting an RGB image in floating point (F) channels x height x width
(where channels == 3) format to the UInt8 channels x width x height format expected by
OpenGL. For index manipulations we use integer type `I`.

We use 2D blocks and grids. Threads correspond to pixels (with all channels).

!!! The floating point values must be non-negative. (They are allowed to exceed 1.)
"""
function convert_rgb_to_gl_img_kernel!(cwh_ui8, chw_f, ::Type{I}, ::Type{F}) where {I, F}
    # TODO: shared memory, cf. efficient transpose?
    w = I(size(cwh_ui8, 2))  # == I(size(chw_f, 3))
    h = I(size(cwh_ui8, 3))  # == I(size(chw_f, 2))
    k = threadIdx().y + (blockIdx().y - one(I)) * blockDim().y
    k_stride = blockDim().y * gridDim().y
    j_stride = blockDim().x * gridDim().x
    while k <= h
        j = threadIdx().x + (blockIdx().x - one(I)) * blockDim().x
        while j <= w
            i = one(I)
            @inbounds while i <= I(3)
                f_val = chw_f[i, k, j]
                ui8_val = f_val <= one(F) ? round(UInt8, f_val * F(255)) : UInt8(255) 
                cwh_ui8[i, j, k] = ui8_val
                i += one(I)
            end
            j += j_stride
        end
        k += k_stride
    end
    return
end

"""
    convert_transmittance_to_gl_img_kernel!(
        cwh_ui8, hw_f, ::Type{I}, ::Type{F}
    ) where {I, F}

Kernel for converting a transmittance image in floating point (F) height x width format to
the UInt8 channels x width x height format (with channels == 3) expected by OpenGL. For
index manipulations we use integer type `I`.

We use 2D blocks and grids. Threads correspond to pixels.

!!! The floating point values must lie in [0, 1].
"""
function convert_transmittance_to_gl_img_kernel!(
    cwh_ui8, hw_f, ::Type{I}, ::Type{F}
) where {I, F}
    w = I(size(cwh_ui8, 2))  # == I(size(hw_f, 2))
    h = I(size(cwh_ui8, 3))  # == I(size(hw_f, 1))
    k = threadIdx().y + (blockIdx().y - one(I)) * blockDim().y
    k_stride = blockDim().y * gridDim().y
    j_stride = blockDim().x * gridDim().x
    while k <= h
        j = threadIdx().x + (blockIdx().x - one(I)) * blockDim().x
        @inbounds while j <= w
            ui_val = round(UInt8, hw_f[k, j] * F(255))
            i = one(I)
            while i <= I(3)
                cwh_ui8[i, j, k] = ui_val
                i += one(I)
            end
            j += j_stride
        end
        k += k_stride
    end
    return
end

"""
    convert_depth_to_gl_img_kernel!(
        cwh_ui8, hw_f, min_f, max_f, ::Type{I}, ::Type{F}
    ) where {I, F}

Kernel for converting a depth image in floating point (F) height x width format to
the UInt8 channels x width x height format (with channels == 3) expected by OpenGL. We
additionally linearly rescale and clamp the floating point values to the range
`[min_f, max_f]`, where `min_f` (i.e. closeby depths) will correspond to white and `min_f`
(far away) to black. For index manipulations we use integer type `I`.

We use 2D blocks and grids. Threads correspond to pixels.

!!! The minimal value `min_f` must be smaller than the maximal value `max_f`.
"""
function convert_depth_to_gl_img_kernel!(
    cwh_ui8, hw_f, min_f, max_f, ::Type{I}, ::Type{F}
) where {I, F}
    w = I(size(cwh_ui8, 2))
    h = I(size(cwh_ui8, 3))
    k = threadIdx().y + (blockIdx().y - one(I)) * blockDim().y
    k_stride = blockDim().y * gridDim().y
    j_stride = blockDim().x * gridDim().x
    while k <= h
        j = threadIdx().x + (blockIdx().x - one(I)) * blockDim().x
        @inbounds while j <= w
            ui_val = round(
                UInt8, 
                (one(F) - clamp01(
                    (hw_f[k, j] - min_f) / (max_f - min_f))
                ) * F(255)
            )
            # min_f ↦ 1 (white for close by)
            # max_f ↦ 0 (black for far away)
            i = one(I)
            while i <= I(3)
                cwh_ui8[i, j, k] = ui_val
                i += one(I)
            end
            j += j_stride
        end
        k += k_stride
    end
    return
end

"""
    update!(gui::GUI)

Update the GUI. This method contains the core CImGUI steps and will call
[`update_custom!`](@ref) for e.g. our custom rendering.
"""
function update!(gui::GUI)
    gui.has_stopped && return
    if GLFW.WindowShouldClose(gui.window)
        close!(gui)
        gui.has_stopped = true
        return
    end

    # Polling/waiting for events blocks the whole thread and there's
    # nothing we can do about it. But, if there are no callbacks
    # registered on the window we can safely enter a GC-safe region to
    # allow the GC to run in parallel, which will help prevent GC pauses
    # that cause the GUI to stutter.
    have_callbacks = any(!isnothing, GLFW.callbacks(gui.window))
    if have_callbacks
        gui.wait_events ? GLFW.WaitEvents() : GLFW.PollEvents()
    else
        try
            @ccall jl_gc_safe_enter()::Int8
            gui.wait_events ? GLFW.WaitEvents() : GLFW.PollEvents()
        finally
            @ccall jl_gc_safe_leave()::Int8
        end
    end

    # Start the Dear ImGui frame
    lib.ImGui_ImplOpenGL3_NewFrame()
    lib.ImGui_ImplGlfw_NewFrame()
    ig.NewFrame()

    # Prepare the stuff to show via CImGui
    update_custom!(gui)
    
    # CImGui rendering
    ig.Render()
    GLFW.MakeContextCurrent(gui.window)

    display_w, display_h = GLFW.GetFramebufferSize(gui.window)
    GL.glViewport(0, 0, display_w, display_h)
    GL.glClearColor(gui.window_background_color...)
    GL.glClear(GL.GL_COLOR_BUFFER_BIT)
    lib.ImGui_ImplOpenGL3_RenderDrawData(Ptr{Cint}(ig.GetDrawData()))

    GLFW.MakeContextCurrent(gui.window)

    # This is when what we've just rendered actually gets
    # displayed. With the swap interval set to 1 it will synchronize
    # with the display refresh rate to prevent screen tearing, which
    # means this may block for a non-negligible amount of time. Hence we
    # again enter a GC-safe region so as not to block the
    # GC. SwapBuffers() does not call any callbacks so this should
    # always be safe.
    try
        @ccall jl_gc_safe_enter()::Int8
        GLFW.SwapBuffers(gui.window)
    finally
        @ccall jl_gc_safe_leave()::Int8
    end

    if (unsafe_load(ig.GetIO().ConfigFlags) & lib.ImGuiConfigFlags_ViewportsEnable) == 
            lib.ImGuiConfigFlags_ViewportsEnable
        backup_current_context = GLFW.GetCurrentContext()
        lib.igUpdatePlatformWindows()
        lib.igRenderPlatformWindowsDefault(C_NULL, C_NULL)
        GLFW.MakeContextCurrent(backup_current_context)
    end

    yield()
end

"""
    update_custom!(gui)

Handle the image rendering, interaction and information presentation.
"""
function update_custom!(gui)
    # In case of training (where the number of Gaussians is not constant)
    set_nb_gaussians!(gui.render_buffers, length(gui.gaussians))

    settings_window!(gui)
    # Before the render_window, as set_render_resolution! leads to artefacts on the
    # previously rendered image, for some reason
    render_window!(gui)

    info_window!(gui)

    handle_keyboard_input!(gui)
end

"""
    render_window!(gui::GUI)

Render an image and present it in its own window.
"""
function render_window!(gui::GUI)
    I = int_type(gui); F = float_type(gui)
    if ig.Begin("Renderer")
        # For the display size of the image we want to use all available place in the
        # window, possibly preserving the aspect ratio
        window_size = ig.GetContentRegionAvail()  # ig.ImVec2
        image_width = get_width(gui.image); image_height = get_height(gui.image)
        if gui.fix_aspect_ratio
            image_aspect_ratio = image_width / F(image_height)
            display_width = round(I, 
                min(window_size.x, window_size.y * image_aspect_ratio))
            display_height = round(I, display_width / image_aspect_ratio)
        else
            display_width = round(I, window_size.x)
            display_height = round(I, window_size.y)
        end
        if gui.link_render_and_display_size
            set_render_resolution!(gui, display_width, display_height)
        end
        render_image!(gui, ig.ImVec2(display_width, display_height))
    end
    ig.End()
end

"""
    setting_window!(gui::GUI)

Show a window for controlling many settings like the camera intrinsics.
"""
function settings_window!(gui::GUI)
    D = array_domain(gui); I = int_type(gui); F = float_type(gui)

    # TODO: incorporate GT images or poses thereof
    #   gui.cameras_with_images / nothing?
    if ig.Begin("Settings")
        slider_width = Cfloat(100)
        line_height = ig.GetFrameHeight()
        checkbox_width = line_height  # as it's a square

        if ig.TreeNodeEx("Camera settings", ig.ImGuiTreeNodeFlags_DefaultOpen)
            if ig.TreeNode("Movement")
                ig.SetNextItemWidth(slider_width)
                @c ig.DragFloat("Translation speed (world units / s)", &gui.translation_speed, 
                    Cfloat(0.01), Cfloat(0), Cfloat(Inf), "%.2f")
                # (Instead of @c, we could pass a Ref(Value))
                ig.SetNextItemWidth(slider_width)
                @c ig.DragFloat("Rotation speed (rad / s)", &gui.rotation_speed, 
                    Cfloat(0.01), Cfloat(0), Cfloat(Inf), "%.2f")    
                ig.TreePop()
            end

            if ig.TreeNode("Resolution")
                ig.SetNextItemWidth(slider_width)
                ig.BeginDisabled(gui.link_render_and_display_size)
                # In that case, you should resize the display window to control the render size
                render_width = Cint(get_render_width(gui))
                render_height = Cint(get_render_height(gui))
                if gui.fix_aspect_ratio
                    aspect_ratio = render_width / render_height
                end
                @c ig.DragInt("Render width (px)", &render_width, 
                    Cint(1), Cint(1), typemax(Cint))
                # Label, input/output variable, slider speed (value / px), min value, max value
                if gui.fix_aspect_ratio
                    render_height = round(I, render_width / aspect_ratio)
                end

                ig.SetNextItemWidth(slider_width)
                @c ig.DragInt("Render height (px)", &render_height, 
                    Cint(1), Cint(1), typemax(Cint))
                if gui.fix_aspect_ratio
                    render_width = round(I, render_height * aspect_ratio)
                end
                ig.EndDisabled()
                
                # Put right-aligned checkbox for fix_aspect_ratio
                ig.SetCursorPosX(ig.GetCursorPosX() + slider_width - checkbox_width)
                @c ig.Checkbox("Fix aspect ratio", &gui.fix_aspect_ratio)

                ig.SetCursorPosX(ig.GetCursorPosX() + slider_width - checkbox_width)
                @c ig.Checkbox("Link render and display size", 
                    &gui.link_render_and_display_size)

                if !gui.link_render_and_display_size
                    interpolation_options = collect(
                        to_string.(instances(GUIInterpolationMode)))
                    # (Must be a Vector{String}, not e.g. a Tuple)
                    ig.SetNextItemWidth(slider_width)
                    selected_interpolation_index = Cint(gui.display_interpolation_mode)
                    @c ig.Combo("Display interpolation", &selected_interpolation_index, 
                        interpolation_options)
                    set_gl_interpolation_mode!(
                        gui, 
                        GUIInterpolationMode(selected_interpolation_index))
                end            

                if render_width >= 1 && render_height >= 1
                    set_render_resolution!(gui, render_width, render_height)
                else
                    println(
                        "GUI: Rejected invalid resolution settings: \
                         render width and height must be at least 1!")
                end
                ig.TreePop()
            end  # Resolution

            if ig.TreeNode("Other")
                ig.SetNextItemWidth(slider_width)
                fov_x_deg = Cfloat(gui.camera_fov_x_deg)
                # Could use gui.camera_fov_x_deg directly, but by double-clicking one can
                # set arbitrary values, which might crash the rendering. So delay adjusting
                # our internal settings
                fov_y_deg = Cfloat(gui.camera_fov_y_deg)
                @c ig.DragFloat("Horizontal FOV (°)", &fov_x_deg, 
                Cfloat(0.1), Cfloat(0.1), Cfloat(179.9), "%.1f")
                if gui.link_fovs
                    # Use same focal length for x and y
                    fov_y_deg = focal_length_to_fov_deg(
                        fov_deg_to_focal_length(fov_x_deg, get_render_width(gui)), 
                        get_render_height(gui))
                end

                ig.SetNextItemWidth(slider_width)
                @c ig.DragFloat("Vertical FOV (°)", &fov_y_deg, 
                    Cfloat(0.1), Cfloat(0.1), Cfloat(179.9), "%.1f")

                if gui.link_fovs
                    fov_x_deg = focal_length_to_fov_deg(
                        fov_deg_to_focal_length(fov_y_deg, get_render_height(gui)), 
                        get_render_width(gui))
                end
                if 0 < fov_x_deg < 180 && 0 <= fov_y_deg <= 180
                    gui.camera_fov_x_deg = fov_x_deg
                    gui.camera_fov_y_deg = fov_y_deg
                else
                    println(
                        "GUI: Rejected FOV settings: \
                          FOVs must lie in the range (0°, 180°)!"
                    )
                end

                ig.SetCursorPosX(ig.GetCursorPosX() + slider_width - checkbox_width)
                @c ig.Checkbox("Link FOVs", &gui.link_fovs)


                ig.SetNextItemWidth(slider_width)
                skew_angle_deg = Cfloat(gui.camera_skew_angle_deg)
                @c ig.DragFloat("Skew angle (°)", &skew_angle_deg, 
                    Cfloat(0.1), Cfloat(-89.9), Cfloat(89.9), "%.1f")
                if -90 < skew_angle_deg < 90
                    gui.camera_skew_angle_deg = skew_angle_deg 
                else
                    println(
                        "GUI: Rejected skew angle: \
                         must lie in the range (-90°, 90°)!"
                    )
                end

                ig.SetNextItemWidth(slider_width)
                @c ig.DragFloat("Principal point - x (NDC)", &gui.camera_principal_point_x_ndc, 
                    Cfloat(0.001), Cfloat(-Inf), Cfloat(Inf), "%.3f")  
                    # (allowed to go outside of screen)
                ig.SetNextItemWidth(slider_width)
                @c ig.DragFloat("Principal point - y (NDC)", &gui.camera_principal_point_y_ndc, 
                    Cfloat(0.001), Cfloat(-Inf), Cfloat(Inf), "%.3f")

                ig.SetNextItemWidth(slider_width)
                near_z = Cfloat(gui.camera_near_z)
                @c ig.DragFloat("Near clipping plane (world units)", &near_z, 
                    Cfloat(0.01), Cfloat(0.01), Cfloat(Inf), "%.2f")
                # Disallow zero (or less), as we divide by depth
                if near_z > 0
                    gui.camera_near_z = near_z 
                else
                    println(
                        "GUI: Rejected near clipping plane: \
                         must be strictly positive!"
                    )
                end

                ig.SetNextItemWidth(slider_width)
                @c ig.DragFloat("Far clipping plane (world units)", &gui.camera_far_z, 
                    Cfloat(0.01), Cfloat(0), Cfloat(Inf), "%.2f")
                # Non-positive far clipping plane is fine: we'll just not render any
                # Gaussians

                ig.TreePop()
            end  # Other

            ig.TreePop()
        end  # Camera settings

        if ig.TreeNode("Tiling")
            tw = gui.tile_width; th = gui.tile_height
            # Don't allow ig to change the real settings directly to avoid crashes on
            # invalid values
            ig.SetNextItemWidth(slider_width)
            @c ig.InputInt("Tile width", &tw)
            ig.SetNextItemWidth(slider_width)
            @c ig.InputInt("Tile height", &th)

            if 1 <= tw && 1 <= th
                if tw * th <= 1024
                    gui.tile_width = tw
                    gui.tile_height = th
                else
                    println(
                        "GUI: Rejected invalid tile settings: \
                         a tile can contain at most 1024 pixels!")
                end
            else
                println(
                    "GUI: Rejected invalid tile settings: \
                     width and height must be at least 1!")
            end

            ig.TreePop()
        end

        if ig.TreeNode("Render options")

            if @c ig.Checkbox("Vsync", &gui.vsync)
                # If anything has changed
                if gui.vsync
                    GLFW.SwapInterval(1)
                else
                    GLFW.SwapInterval(0)
                end
            end

            selected_render_mode_index = Cint(gui.render_mode)
            render_mode_options = collect(to_string.(instances(GUIRenderMode)))
            ig.SetNextItemWidth(slider_width)
            @c ig.Combo("Render mode", &selected_render_mode_index, render_mode_options)

            new_render_mode = GUIRenderMode(selected_render_mode_index)
            if new_render_mode != gui.render_mode
                maybe_unsafe_free!(gui.render_outputs)
                gui.render_outputs = create_render_outputs(
                    new_render_mode, get_render_width(gui), get_render_height(gui),
                    D, I, F
                )
                gui.render_mode = new_render_mode
                # (Note that on the CPU when going from RGB to transmittance, we still need
                #  a new RenderOutputs even though we were already rendering the
                #  transmittance, to signal to render! we no now longer need the RGB output.
                #  In principle we could reuse the memory, though.)
            end

            if gui.render_mode == grm_rgb
                sh_deg = Cint(gui.sh_degree)
                gaussians_sh_deg = get_sh_degree(gui.gaussians)
                ig.SetNextItemWidth(slider_width)
                @c ig.SliderInt("SH degree", &sh_deg,  Cint(0), Cint(gaussians_sh_deg))
                if 0 <= sh_deg <= gaussians_sh_deg
                    gui.sh_degree = sh_deg
                else
                    println(
                        "GUI: Rejected invalid SH degree value: \
                        must lie inclusively between 0 and the 3DGS model's SH degree of \
                        $(gaussians_sh_deg)!")
                end
            end

            if gui.render_mode == grm_depth
                min_depth = Cfloat(gui.visualization_min_depth)
                max_depth = Cfloat(gui.visualization_max_depth)
                ig.SetNextItemWidth(slider_width)
                @c ig.DragFloat("Minimal depth", &min_depth, 
                    Cfloat(0.01), Cfloat(0), max_depth, "%.2f")
                ig.SetNextItemWidth(slider_width)
                @c ig.DragFloat("Maximal depth", &max_depth, 
                    Cfloat(0.01), min_depth, Cfloat(Inf), "%.2f")

                if min_depth < max_depth
                    gui.visualization_min_depth = min_depth
                    gui.visualization_max_depth = max_depth
                else
                    println(
                        "GUI: Rejected depth map visualization settings: \
                            the maximal value must be larger than the minimal value!")
                end

                # Right-aligned auto button
                auto_button_width = slider_width / 2
                ig.SetCursorPosX(ig.GetCursorPosX() + slider_width - auto_button_width)
                ig.SetNextItemWidth(auto_button_width)
                if ig.Button("Auto##Depth", ig.ImVec2(auto_button_width, line_height))  
                # ##Depth, in case we end up with multiple "Auto" buttons
                    depths_cpu = CPU(get_depth_image(gui.render_outputs))
                    gui.visualization_min_depth = quantile(vec(depths_cpu), F(0.01))
                    gui.visualization_max_depth = quantile(vec(depths_cpu), F(0.99))
                    # Not particularly efficient, but CuVector does not support
                    # quantile. We should not be pushing this button too often, so 
                    # should be fine.
                end
            end

            @c ig.Checkbox("Antialiasing", &gui.use_antialiasing)

            selected_boundary_index = Cint(gui.boundary_clipping_mode)
            boundary_clipping_options = collect(
                to_string.(instances(GUIBoundaryClippingMode)))
            ig.SetNextItemWidth(slider_width)
            @c ig.Combo("Boundary clipping mode", &selected_boundary_index, 
                boundary_clipping_options)
            gui.boundary_clipping_mode = GUIBoundaryClippingMode(selected_boundary_index)
            
            min_alpha = Cfloat(gui.render_min_alpha)
            ig.SetNextItemWidth(slider_width)
            @c ig.DragFloat("Minimum alpha threshold", &min_alpha, 
                Cfloat(0.0001), Cfloat(0.0001), Cfloat(Inf), "%.4f")
            if min_alpha > 0
                gui.render_min_alpha = min_alpha
            else
                println(
                    "GUI: Rejected invalid minimum alpha threshold: \
                     must be strictly positive (to have bounded 2D Gaussian bounding \
                     boxes)!")
            end

            max_alpha = Cfloat(gui.render_max_alpha)
            ig.SetNextItemWidth(slider_width)
            @c ig.DragFloat("Maximum alpha clamp", &max_alpha, 
                Cfloat(0.0001), Cfloat(0), Cfloat(Inf), "%.4f")
            if 0 <= max_alpha <= 1
                gui.render_max_alpha = max_alpha
            else
                println(
                    "GUI: Rejected invalid maximum alpha clamp value: \
                     must lie in [0, 1]!")
            end

            min_transmittance = Cfloat(gui.render_min_transmittance)
            ig.SetNextItemWidth(slider_width)
            @c ig.DragFloat("Minimum transmittance break", &min_transmittance, 
                Cfloat(0.0001), Cfloat(0), Cfloat(1), "%.4f")
            if 0 <= min_transmittance <= 1
                gui.render_min_transmittance = min_transmittance
            else
                println(
                    "GUI: Rejected invalid minimum transmittance render break value: \
                     must lie in [0, 1]!")
            end

            dilation_x = Cfloat(gui.render_dilation_x_px)
            ig.SetNextItemWidth(slider_width)
            @c ig.DragFloat("Dilation - x (px)", &dilation_x, 
                Cfloat(0.0001), Cfloat(0), Cfloat(Inf), "%.4f")
            if dilation_x >= 0
                gui.render_dilation_x_px = dilation_x
            else
                println(
                    "GUI: Rejected invalid x-dilation value: \
                     must be non-negative!")
            end

            dilation_y = Cfloat(gui.render_dilation_y_px)
            ig.SetNextItemWidth(slider_width)
            @c ig.DragFloat("Dilation - y (px)", &dilation_y, 
                Cfloat(0.0001), Cfloat(0), Cfloat(Inf), "%.4f")
            if dilation_y >= 0
                gui.render_dilation_y_px = dilation_y
            else
                println(
                    "GUI: Rejected invalid y-dilation value: \
                     must be non-negative!")
            end


            @cstatic link_clippings = false begin
                if !link_clippings
                    ig.SetNextItemWidth(slider_width)
                    @c ig.DragFloat("Clipping min x (NDC)", &gui.clipping_min_x_ndc, 
                        Cfloat(0.001), Cfloat(-Inf), Cfloat(Inf), "%.3f")

                    ig.SetNextItemWidth(slider_width)
                    @c ig.DragFloat("Clipping max x (NDC)", &gui.clipping_max_x_ndc, 
                        Cfloat(0.001), Cfloat(-Inf), Cfloat(Inf), "%.3f")
                    # (Fine if below min x)

                    ig.SetNextItemWidth(slider_width)
                    @c ig.DragFloat("Clipping min y (NDC)", &gui.clipping_min_y_ndc, 
                        Cfloat(0.001), Cfloat(-Inf), Cfloat(Inf), "%.3f")

                    ig.SetNextItemWidth(slider_width)
                    @c ig.DragFloat("Clipping max y (NDC)", &gui.clipping_max_y_ndc, 
                        Cfloat(0.001), Cfloat(-Inf), Cfloat(Inf), "%.3f")
                else
                    margin = gui.clipping_max_x_ndc - one(F)
                    ig.SetNextItemWidth(slider_width)
                    @c ig.DragFloat("Clipping margin (NDC)", &margin, 
                        Cfloat(0.001), Cfloat(-Inf), Cfloat(Inf), "%.3f")

                    gui.clipping_min_x_ndc = -one(F) - margin
                    gui.clipping_max_x_ndc = one(F) + margin
                    gui.clipping_min_y_ndc = -one(F) - margin
                    gui.clipping_max_y_ndc = one(F) + margin
                end

                ig.SetCursorPosX(ig.GetCursorPosX() + slider_width - checkbox_width)
                @c ig.Checkbox("Link clipping margins", &link_clippings)

            end

            ig.Dummy(Cfloat(0), line_height / 2)  # Some whitespace
            ig.Text("Background color:")
            color_preview_width = slider_width / 2  # Should be < slider_width
            color_preview_height = 2 * line_height
            ig.ColorButton(
                "##BackgroundColorShow",  # Hidden label: don't show
                ig.ImVec4(
                    gui.background_color[1], 
                    gui.background_color[2], 
                    gui.background_color[3], 
                    Cfloat(1)),
                ig.ImGuiColorEditFlags_NoTooltip,
                ig.ImVec2(color_preview_width, color_preview_height)
            )
            ig.ColorPicker3("##BackgroundColorSelect", gui.background_color, 
                ig.ImGuiColorEditFlags_PickerHueWheel | 
                    ig.ImGuiColorEditFlags_NoSidePreview)

            ig.TreePop()
        end

    end
    ig.End()
end

"""
    info_window!(gui::GUI)

Show a window presenting non-directly interactive information, such as the render time and
camera pose.
"""
function info_window!(gui::GUI)
    # TODO: Integrate with Trainer
    #   Store trainer / nothing in GUI?
    #   --> backward time, loss, training iteration nb, ...

    if ig.Begin("Info")
        line_height = ig.GetFrameHeight()
        if @c ig.Checkbox("Time rendering", &gui.time_rendering)
            gui.nb_timed_frames = 0
            gui.ewma_render_time = 0.
            # For when we go from not timing to timing, but no issues for the other
            # direction

            empty!(gui.render_time_history)
        end
        
        if gui.time_rendering && gui.last_render_time != 0
            gui.nb_timed_frames += 1
            gui.ewma_render_time = 
                     gui.ewma_weight  * gui.ewma_render_time + 
                (1 - gui.ewma_weight) * gui.last_render_time
            corrected_render_time = 
                gui.ewma_render_time / (1 - gui.ewma_weight ^ gui.nb_timed_frames)
            fps = round(Int, 1. / corrected_render_time) 
            ig.Text(
                @sprintf(
                    "Render time:  %0.2f ms  (%d fps)",
                    corrected_render_time * 1000, fps)
            )
            # s -> ms

            push!(gui.render_time_history, corrected_render_time)
            history_to_show = if length(gui.render_time_history) >= 
                                    gui.render_time_history_display_length
                @view(gui.render_time_history[
                        end - gui.render_time_history_display_length + 1 : end])
            else
                gui.render_time_history
            end            
            range_min, range_max = quantile(history_to_show, (0.05, 0.95))
            ig.PlotLines(
                "History", 
                history_to_show, length(history_to_show), 
                0, C_NULL,   # offset, overlay
                range_min, range_max)
            ig.Text(@sprintf("Range:  %8.4f   - %8.4f ms", 
                    Cfloat(range_min * 1000), 
                    Cfloat(range_max * 1000)))
        end

        # Gaussian counts
        ig.Dummy(Cfloat(0), line_height)
        nb_gaussians_strs = format.(
            (length(gui.gaussians), 
                get_nb_fgaussians(gui.render_buffers), 
                get_nb_dfgaussians(gui.render_buffers)),
            commas = true
        )
        max_length = maximum(length.(nb_gaussians_strs))
        ig.Text("Gaussians:     " * lpad(nb_gaussians_strs[1], max_length))
        ig.Text("  Filtered:    " * lpad(nb_gaussians_strs[2], max_length))
        ig.Text("  Duplicated:  " * lpad(nb_gaussians_strs[3], max_length))

        # Camera matrices
        ig.Dummy(Cfloat(0), line_height)
        ig.Text("Camera intrinsic matrix:")
        ig.Text(matrix_to_pretty_string(get_intrinsic_matrix(get_camera(gui))))
        ig.Dummy(Cfloat(0), line_height / 4)
        ig.Text("Camera to world (pose):")
        ig.Text(matrix_to_pretty_string(
            get_camera_to_world_transformation_3x4(get_camera(gui))))
        ig.Dummy(Cfloat(0), line_height / 4)
        ig.Text("World to camera:")
        ig.Text(matrix_to_pretty_string(
            get_world_to_camera_transformation_3x4(get_camera(gui))))
    end
    ig.End()
end

"""
    matrix_to_pretty_string(matrix; indent = 2, spacing = 2, precision = 4)

Convert a floating point matrix to a pretty string representation where all entries are
nicely aligned at their decimal point. All values will have `precision` digits after this
decimal point. (For integer matrices we would pad at the front with zeros.) Entries will be
spaced apart according to the `spacing`. The first entries of each row will start with
the provided `indent` as spacing.
"""
function matrix_to_pretty_string(matrix; indent = 2, spacing = 2, precision = 4)
    string_matrix = format.(matrix; positivespace = true, precision)
    max_length = maximum(length, string_matrix)
    string_matrix = map(string_matrix) do s
        lpad(s, max_length)
    end
    join(
        map(
            join.(eachrow(string_matrix), " " ^ spacing)
        ) do s
            " " ^ indent * s
        end, 
        '\n'
    )
end

"""
    handle_keyboard_input!(gui::GUI) 

Handle keyboard input for our GUI, allowing to move the camera around.
"""
function handle_keyboard_input!(gui::GUI) 
    F = float_type(gui)

    current_timestamp = ig.GetTime()  # in s
    if gui.last_timestep < 0
        # Skip input on the first frame
        gui.last_timestep = current_timestamp
        return
    else
        time_diff = F(current_timestamp - gui.last_timestep)
        gui.last_timestep = current_timestamp
    end
    
    translation_step = gui.translation_speed * time_diff
    rotation_step    = gui.rotation_speed * time_diff

    # TODO: rebindable
    # Translation
    if GLFW.GetKey(gui.window, GLFW.KEY_W) == GLFW.PRESS  # forward
        gui.camera_to_world_translation = gui.camera_to_world_translation .+ 
            gui.camera_to_world_rotation * SVector{3, F}(0, 0, translation_step)
    end
    # GLFW keys use scan codes, not key codes like CImGUI. This allows for keyboard
    # layout agnostic input handling. For example, KEY_Q it the top-left letter key on
    # the keyboard, regardless of whether it is actually a Q (on qwerty) or e.g. an A
    # (on azerty)
    if GLFW.GetKey(gui.window, GLFW.KEY_S) == GLFW.PRESS  # backward
        gui.camera_to_world_translation = gui.camera_to_world_translation .+ 
            gui.camera_to_world_rotation * SVector{3, F}(0, 0, -translation_step)
        
    end
    if GLFW.GetKey(gui.window, GLFW.KEY_A) == GLFW.PRESS  # left
       gui.camera_to_world_translation = gui.camera_to_world_translation .+ 
            gui.camera_to_world_rotation * SVector{3, F}(-translation_step, 0, 0)
    end
    if GLFW.GetKey(gui.window, GLFW.KEY_D) == GLFW.PRESS  # right
        gui.camera_to_world_translation = gui.camera_to_world_translation .+ 
            gui.camera_to_world_rotation * SVector{3, F}(translation_step, 0, 0)
    end
    if GLFW.GetKey(gui.window, GLFW.KEY_LEFT_SHIFT) == GLFW.PRESS || 
        GLFW.GetKey(gui.window, GLFW.KEY_E) == GLFW.PRESS  # up
        gui.camera_to_world_translation = gui.camera_to_world_translation .+ 
            gui.camera_to_world_rotation * SVector{3, F}(0, -translation_step, 0)
    end
    if GLFW.GetKey(gui.window, GLFW.KEY_LEFT_CONTROL) == GLFW.PRESS || 
        GLFW.GetKey(gui.window, GLFW.KEY_Q) == GLFW.PRESS  # down
        gui.camera_to_world_translation = gui.camera_to_world_translation .+ 
            gui.camera_to_world_rotation * SVector{3, F}(0, translation_step, 0)
    end

    if GLFW.GetKey(gui.window, GLFW.KEY_J) == GLFW.PRESS  # pan left
        gui.camera_to_world_rotation = gui.camera_to_world_rotation * 
            AngleAxis(-rotation_step, zero(F), one(F), zero(F))
    end
    if GLFW.GetKey(gui.window, GLFW.KEY_L) == GLFW.PRESS  # pan right
        gui.camera_to_world_rotation = gui.camera_to_world_rotation * 
            AngleAxis(rotation_step, zero(F), one(F), zero(F))
    end
    if GLFW.GetKey(gui.window, GLFW.KEY_I) == GLFW.PRESS  # tilt up
        gui.camera_to_world_rotation = gui.camera_to_world_rotation *  
            AngleAxis(rotation_step, one(F), zero(F), zero(F))
    end
    if GLFW.GetKey(gui.window, GLFW.KEY_K) == GLFW.PRESS  # tilt down
        gui.camera_to_world_rotation = gui.camera_to_world_rotation *  
            AngleAxis(-rotation_step, one(F), zero(F), zero(F))
    end
    if GLFW.GetKey(gui.window, GLFW.KEY_U) == GLFW.PRESS  # roll left
        gui.camera_to_world_rotation = gui.camera_to_world_rotation * 
            AngleAxis(-rotation_step, zero(F), zero(F), one(F))
    end
    if GLFW.GetKey(gui.window, GLFW.KEY_O) == GLFW.PRESS  # roll right
       gui.camera_to_world_rotation = gui.camera_to_world_rotation * 
            AngleAxis(rotation_step, zero(F), zero(F), one(F))
    end
end

"""
    close!(gui::GUI)

Close the GUI. Should only be called when we are done (`has_stopped`). If so, we will
clean everything up. If `!has_stopped`, we just return without doing anything.
"""
function close!(gui::GUI)
    has_stopped(gui) && return nothing

    GLFW.SetWindowShouldClose(gui.window, true)
    lib.ImGui_ImplOpenGL3_Shutdown()
    lib.ImGui_ImplGlfw_Shutdown()
    ig.DestroyContext(gui.ctx)
    GLFW.DestroyWindow(gui.window)

    maybe_unsafe_free!(gui.render_outputs)
    maybe_unsafe_free!(gui.render_buffers)

    return nothing
end

"""
    NoGUI <: AbstractGUI

Dummy AbstractGUI which simply does nothing.
"""
struct NoGUI <: AbstractGUI end

NoGUI(args...; kwargs...) = NoGUI()
set_camera!(::NoGUI, _) = nothing
@inline has_stopped(::AbstractGUI) = false
@inline is_running(::AbstractGUI) = false
@inline update!(::AbstractGUI) = nothing
@inline close!(::AbstractGUI) = nothing
