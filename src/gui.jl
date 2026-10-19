# Our GUI struct is based mostly on GlfwOpenGLBackend.jl. In particular, we split its
# renderloop method

# TODO: docs!
# TODO: decompose, expose and recompose all rendering (e.g. antialiasing) and camera arguments
# TODO: help widget, containing a.o. the controls

OGLBackend = Base.get_extension(ig, :GlfwOpenGLBackend)

abstract type AbstractGUIImage{D <: ArrayDomain, I <: Integer} end

struct CPUGUIImage{
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

struct GPUGUIImage{
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

# Implement update!, close!, has_closed, is_running
abstract type AbstractGUI end

# CPU is very slow
mutable struct GUI{
    D <: ArrayDomain,
    I <: Integer, 
    F <: AbstractFloat,
    GV <: AbstractGaussiansVector{D, I, F},
    C <: AbstractCamera{I, F},
    T <: Tiling{I},
    ROP <: RenderOptions{I, F},
    ROUT <: RenderOutputs{D, I, F},
    RB <: AbstractRenderBuffers{D, I, F},
    GUI_IMG <: AbstractGUIImage{D, I}
} <: AbstractGUI
    gaussians::GV
    camera::C            # TODO: disassemble into parts, including (render) resolution, fovs?
    # Alternative, make this mutable somehow (MArrays, switching to SArrays on device)
    tiling::T            # TODO: allow changing tile size; could store params, recreate every render
    render_options::ROP  # TODO: allow changing SH_DEGREE
    render_outputs::ROUT
    render_buffers::RB

    has_stopped::Bool
    last_timestep::Cdouble  # i.e. Float64

    ctx::Ptr{ig.lib.ImGuiContext}
    wait_events::Bool

    window::GLFW.Window
    window_width::I
    window_height::I
    window_title::String
    window_background_color::SVector{4, Cfloat}  # i.e. Float32

    image::GUI_IMG
end


function GUI(
    gaussians::AbstractGaussiansVector{D, I, F},
    camera::AbstractCamera{I, F},
    tiling::Tiling{I},  # TODO: create here?
    render_options::RenderOptions{I, F},
    render_buffers::AbstractRenderBuffers{D, I, F};  #TODO: nothing -> create own
    window_background_color = SVector{4, Cfloat}(0.45, 0.55, 0.60, 1.00),
    window_width = 1280,  # TODO: default to fullscreen (or compatible with render window?)
    window_height = 720,
    window_title = "ThreeDGaussianSplatting.jl",
    image_width = nothing,  #TODO
    image_height = nothing,
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
    end

    # Setup Platform/Renderer bindings
    lib.ImGui_ImplGlfw_InitForOpenGL(Ptr{lib.GLFWwindow}(window.handle), true)
    lib.ImGui_ImplOpenGL3_Init("#version $(glsl_version)")

    isnothing(image_width)  && ( image_width  = get_width(camera) )
    isnothing(image_height) && ( image_height = get_height(camera) )

    image = create_gui_image(D, image_width, image_height)
    # TODO: detach render size (camera) from display size?
    #   (if not, we don't need to store both)

    render_outputs = RenderOutputs(
        image_width, image_height; 
        array_domain = D, int_type = I, float_type = F,
        include_transmittance = D == CPU,
        include_depth = false,
        include_last_contributing_indices = false)
    # TODO: This is fine to start, and we should always only render one (if possible; not indices),
    #  but how to switch to depth?

    GUI(
        gaussians,
        camera,
        tiling,
        render_options,
        render_outputs,
        render_buffers,

        false,
        Float64(-1),
        ctx,
        wait_events,

        window,
        I(window_width), I(window_height),
        window_title,
        window_background_color,

        image 
    )
end

@inline has_stopped(gui::GUI) = gui.has_stopped
@inline is_running(gui::GUI) = !has_stopped(gui)

# Based on https://discourse.julialang.org/t/cuarray-glmakie/52461/10
function initialize_pbo(width, height)
    # Create an OpenGL Pixel Buffer Object (PBO)
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

function render_image!(gui::GUI{CPU, <:Any, F}, display_size) where F
    render!(
        gui.render_outputs,
        gui.gaussians,
        gui.camera,
        gui.tiling,
        gui.render_options,
        gui.render_buffers
    )
    gui.image.array .= round.(
        UInt8, 
        min.(
            permutedims(get_rgb_image(gui.render_outputs), (1, 3, 2)),
            one(F)
        ) .* F(255))
    # c x h x w Float32 -> c x w x h UInt8

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
     # TODO: render! (tile kernel) with UInt8 output (clamped), in 3 x h x w order
     # TODO: what about depth --> needs min_depth, max_depth (from ig sliders?)
    render!(
        gui.render_outputs,
        gui.gaussians,
        gui.camera,
        gui.tiling,
        gui.render_options,
        gui.render_buffers
    )
    threads = (32, 8)
    blocks = (cld(get_width(gui.image), threads[1]), cld(get_height(gui.image), threads[2]))
    @cuda threads blocks convert_to_gl_img_kernel!(
        wrapped_pbo, get_rgb_image(gui.render_outputs), I, F)

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

function convert_to_gl_img_kernel!(cwh_ui8, chw_f, ::Type{I}, ::Type{F}) where {I, F}
    # Must be chw_f >= 0 !
    # TODO: shared memory, cf. efficient transpose?
    w = I(size(cwh_ui8, 2))
    h = I(size(cwh_ui8, 3))
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

function update_custom!(gui::GUI{<:Any, I, F}) where {I, F}
    if ig.Begin("Renderer")
        # For the display size of the image we want to use all available place inthe window,
        # but preserve the aspect ratio
        window_size = ig.GetContentRegionAvail()  # ig.ImVec2
        image_width = get_width(gui.image); image_height = get_height(gui.image)
        image_aspect_ratio = image_width / F(image_height)
        display_width = round(I, min(window_size.x, window_size.y * image_aspect_ratio))
        display_height = round(I, display_width / image_aspect_ratio)
        render_image!(gui, ig.ImVec2(display_width, display_height))
        # TODO: option to switch between linear interpolation, nearest neighbor etc.
        #=
        GL.glBindTexture(GL.GL_TEXTURE_2D, gui_image.tex_ref)
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D,
            GL.GL_TEXTURE_MAG_FILTER,
            GL.GL_NEAREST,
        )
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        =#
    end
    ig.End()

    # TODO: windows for information
    #   camera
    #   render time
    #   other times (if applicable)
    #   loss (if applicable)
    # TODO: window for settings
    #   camera (render resolution, principal point, fovs, ...)
    #   speeds
    #   sh_degree
    #       (with max from gaussians)
    #   background color
    #   depth min max
    #       (maybe checkbox for auto)

    handle_keyboard_input!(gui)
end

function handle_keyboard_input!(gui::GUI{<:Any, <:Any, F}) where F
    current_timestamp = ig.GetTime()  # in s
    if gui.last_timestep < 0
        # Skip input on the first frame
        gui.last_timestep = current_timestamp
        return
    else
        time_diff = F(current_timestamp - gui.last_timestep)
        gui.last_timestep = current_timestamp
    end
    

    # TODO: adjustable: gui. ... (fields), sliders
    movement_speed = F(1.)   # world units / s; translation
    rotation_speed = F(0.5)  # rad / s
    translation_step = movement_speed * time_diff
    rotation_step    = rotation_speed * time_diff



    # TODO: rebindable
    # Translation
    if GLFW.GetKey(gui.window, GLFW.KEY_W) == GLFW.PRESS  # forward
        gui.camera = move(gui.camera, SVector{3, F}(0, 0, translation_step))
    end
    # GLFW keys use scan codes, not key codes like CImGUI. This allows for keyboard
    # layout agnostic input handling. For example, KEY_Q it the top-left letter key on
    # the keyboard, regardless of whether it is actually a Q (on qwerty) or e.g. an A
    # (on azerty)
    if GLFW.GetKey(gui.window, GLFW.KEY_S) == GLFW.PRESS  # backward
        gui.camera = move(gui.camera, SVector{3, F}(0, 0, -translation_step))
    end
    if GLFW.GetKey(gui.window, GLFW.KEY_A) == GLFW.PRESS  # left
        gui.camera = move(gui.camera, SVector{3, F}(-translation_step, 0, 0))
    end
    if GLFW.GetKey(gui.window, GLFW.KEY_D) == GLFW.PRESS  # right
        gui.camera = move(gui.camera, SVector{3, F}(translation_step, 0, 0))
    end
    if GLFW.GetKey(gui.window, GLFW.KEY_LEFT_SHIFT) == GLFW.PRESS  # up
        gui.camera = move(gui.camera, SVector{3, F}(0, -translation_step, 0))
    end
    if GLFW.GetKey(gui.window, GLFW.KEY_LEFT_CONTROL) == GLFW.PRESS  # down
        gui.camera = move(gui.camera, SVector{3, F}(0, translation_step, 0))
    end

    if GLFW.GetKey(gui.window, GLFW.KEY_J) == GLFW.PRESS  # pan left
        gui.camera = rotate(gui.camera, 
            AngleAxis(-rotation_step, zero(F), one(F), zero(F)))
    end
    if GLFW.GetKey(gui.window, GLFW.KEY_L) == GLFW.PRESS  # pan right
        gui.camera = rotate(gui.camera, 
            AngleAxis(rotation_step, zero(F), one(F), zero(F)))
    end
    if GLFW.GetKey(gui.window, GLFW.KEY_I) == GLFW.PRESS  # tilt up
        gui.camera = rotate(gui.camera, 
            AngleAxis(rotation_step, one(F), zero(F), zero(F)))
    end
    if GLFW.GetKey(gui.window, GLFW.KEY_K) == GLFW.PRESS  # tilt down
        gui.camera = rotate(gui.camera, 
            AngleAxis(-rotation_step, one(F), zero(F), zero(F)))
    end
    if GLFW.GetKey(gui.window, GLFW.KEY_U) == GLFW.PRESS  # roll left
        gui.camera = rotate(gui.camera, 
            AngleAxis(-rotation_step, zero(F), zero(F), one(F)))
    end
    if GLFW.GetKey(gui.window, GLFW.KEY_O) == GLFW.PRESS  # roll right
        gui.camera = rotate(gui.camera, 
            AngleAxis(rotation_step, zero(F), zero(F), one(F)))
    end
end

function close!(gui::GUI)
    has_stopped(gui) && return nothing

    GLFW.SetWindowShouldClose(gui.window, true)
    lib.ImGui_ImplOpenGL3_Shutdown()
    lib.ImGui_ImplGlfw_Shutdown()
    ig.DestroyContext(gui.ctx)
    GLFW.DestroyWindow(gui.window)

    return nothing
end


# TODO: render frame time, losses, more interaction

struct NoGUI <: AbstractGUI end
NoGUI(args...; kwargs...) = NoGUI()
@inline has_stopped(::AbstractGUI) = false
@inline is_running(::AbstractGUI) = false
@inline update!(::AbstractGUI) = nothing
@inline close!(::AbstractGUI) = nothing
