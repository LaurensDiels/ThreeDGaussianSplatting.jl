"""
    render(path; iterations=nothing)

Use an interactive GUI to visualize renders from the supplied 3D Gaussian splatting model
path.

# Argument
`path`: Either the .ply file of the 3D Gaussians model to load, or the output directory from
    training, i.e. containing ./point_cloud/iteration_.../point_cloud.ply. In the second
    case we by default use the model with highest number of training iterations.

# Keyword argument
`iterations`: Only relevant for the output directory case for `path`. In this case providing
    this parameter will force to use the model with the specified number of training
    iterations.
"""
function render(path; iterations=nothing)
    D = GPU; I = Int32; F = Float32 
    if isdir(path)  # 3DGS output dir
        # Take point_cloud/iteration_n/point_cloud.ply for largest/provided n
        n = if isnothing(iterations)
            save_iterations_dirs = readdir(joinpath(path, "point_cloud"))
            save_iterations = map(save_iterations_dirs) do dir
                parse(Int, dir[length("iteration_") + 1 : end])
            end
            maximum(save_iterations)
        else
            iterations
        end
        gaussians_path = joinpath(path, "point_cloud", "iteration_$n", "point_cloud.ply")
        camera = first(load_cameras_from_json(
            joinpath(path, "cameras.json");
            int_type = I, float_type = F))

    else  # .ply
        gaussians_path = path

        # Default camera
        w = I(1280); h = I(720)
        camera = PinholeCamera(
            w, h,
            fov_deg_to_focal_length(F(90), w),
            one(SMatrix{3, 3, F}), zero(SVector{3, F})
        )
    end

    gaussians = RenderOnlyGaussiansVector( 
        gaussians_path;
        array_domain = D, int_type = I, float_type = F)

    render_options = RenderOptions(camera, get_sh_degree(gaussians))
    tiling = Tiling(camera)
    render_buffers = MutableRenderBuffers(gaussians, tiling)

    gui = GUI(
        gaussians,
        camera,
        tiling,
        render_options,
        render_buffers
    )
    while !has_stopped(gui)
        update!(gui)
    end

    maybe_unsafe_free!(render_buffers)
    maybe_unsafe_free!(gaussians)
end