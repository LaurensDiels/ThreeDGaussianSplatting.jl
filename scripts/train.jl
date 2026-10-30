"""
    train(
        input_dir, output_dir; 
        show_gui = true, 
        evaluation_iterations = 1000, save_iterations = (7000, 30_000),
        validation_ratio = 0.,
        random_validation_cameras = false
    )

Train a 3D Gaussian splatting model from provided images.

# Arguments
- `input_dir`: Either a directory containing equally sized images of the same scene, or
    the undistorted output directory from COLMAP obtained using [`process_colmap`](@ref).
    (In case of an image directory, we will call this method here ourselves.)
- `output_dir`: The directory where we will write the output to, following the same
    general file structure as the reference implementation.

# Keyword arguments
- `show_gui`: Whether to visualize the training process using a GUI. Defaults to `true`.
- `evaluation_iterations`: Either an `AbstractVector` of training iteration numbers after
    which we will compute training and validation losses, or a single `Integer`, in which
    case we will compute the losses every `evaluation_iterations` iterations.
- `save_iterations`: Similar, but saving the trained Gaussians model.
- `validation_ratio`: The ratio of validation cameras to total cameras. Defaults to 0,
    meaning we do not set apart a validation set. Use e.g. 20% to set aside 1/5 of the 
    cameras for validation. Note that this is approximate, as the number of training cameras
    might not be divisible by five.
- `random_validation_cameras`: When set to `false` (by default) the validation cameras
    will be equidistant in terms of camera index. When set to `true` we use a random
    selection.
"""
function train(
    input_dir, output_dir; 
    show_gui = true, 
    evaluation_iterations = 1000, save_iterations = (7000, 30_000),
    validation_ratio = 0.,
    random_validation_cameras = false
)
    D = GPU; I = Int32; F = Float32

    if !isdir(joinpath(input_dir, "sparse-txt"))
        input_dir = process_colmap(input_dir)
    end
    # input_dir should now be the undistorted COLMAP dir

    cameras_with_images, gaussians = 
        processed_colmap_to_cameras_with_images_and_gaussians_model(input_dir)
    mkpath(output_dir)
    save_json(joinpath(output_dir, "cameras.json"), cameras_with_images)

    nb_cameras = length(cameras_with_images)
    nb_validation_cameras = floor(Int, nb_cameras * validation_ratio)
    # Floor, as training cameras are more important
    validation_camera_ids = if random_validation_cameras
         randperm(nb_cameras)[1:nb_validation_cameras]
    else
        if nb_validation_cameras == 0
            1:0  # empty
        else
            step = cld(nb_cameras, nb_validation_cameras)
            step:step:nb_cameras
        end
    end
    training_camera_ids = setdiff(1:length(cameras_with_images), validation_camera_ids)
    training_cameras_with_images = view(cameras_with_images, training_camera_ids)
    validation_cameras_with_images = view(cameras_with_images, validation_camera_ids)
    training_cameras = map(get_camera, training_cameras_with_images)

    training_scene = Scene(gaussians, training_cameras)  # TODO: or cameras_with_images?

    first_camera = first(training_cameras)
    render_options = RenderOptions(
        first_camera, get_sh_degree(gaussians),
    )  # TODO: expose options boundary_clipping_mode, background_color and use_antialiasing
    tiling = Tiling(first_camera)
    training_options = TrainingOptions(training_scene)
    optimizer = MutableAdamOptimizer(get_gaussians_vector(training_scene), training_options)

    trainer = Trainer(gaussians, training_cameras_with_images,  #TODO: just scene?
        render_options, tiling, training_options, optimizer)
    # TODO: maybe: Scene{WithGT}, Scene{WithoutGT}


    gui = if show_gui
        GUI(
            gaussians,
            first_camera,
            tiling,
            render_options,          # might be better decoupled  # TODO: stuck at sh deg 0
            #   --> will be automatically fixed when we use destructured options in GUI
            trainer.render_buffers;  # fine to couple (memory reuse)
            vsync = false
        )
    else
        NoGUI()
    end

    function compute_losses(rendered_image, gt_image)
        mse_ = mapreduce(
            (a, b) -> (a - b)^2, +, rendered_image, gt_image
        ) / length(gt_image)
        mae_ = mapreduce(
            (a, b) -> abs(a - b), +, rendered_image, gt_image
        ) / length(gt_image)
        dssim_ = dssim(rendered_image, gt_image)
        return mse_, mae_, dssim_
    end

    # Use the trainer's buffers
    # (In principle this method then mutates, but since we are making this totally opaque
    # by using variables from the surrounding scope, and because it really does not matter,
    # don't add the !)
    function maybe_eval_losses()
        should_eval_losses() || return

        iteration_nb = trainer.iteration_nb
        println("Losses after iteration $(iteration_nb):")
        for (type, cwis) in (
                ("Training", training_cameras_with_images),
                ("Validation", validation_cameras_with_images)
            )
            mse_sum = zero(F)
            mae_sum = zero(F)
            dssim_sum = zero(F)
            for cwi in cwis
                gt_image = with_image(cwi) do gt
                    copyto!(trainer.gt_image_buffer, gt)
                    return trainer.gt_image_buffer
                end
                render!(
                    trainer.render_outputs,
                    gaussians, get_camera(cwi),
                    tiling, trainer.render_options,  # for sh degree increase
                    trainer.render_buffers
                ) # TODO: render_options with just max sh degree? (allow 3 when get_sh_degree == 0?)
                rendered_image = get_rgb_image(trainer.render_outputs)
                mse_, mae_, dssim_ = compute_losses(rendered_image, gt_image)
                mse_sum += mse_; mae_sum += mae_; dssim_sum += dssim_
            end
            if !isempty(cwis)
                println("  $type:")
                println("    MSE (L2): ", mse_sum / length(cwis))
                println("    MAE (L1): ", mae_sum / length(cwis))
                println("    DSSIM:    ", dssim_sum / length(cwis))
            end
        end
    end

    should_eval_losses() = should_eval_losses(trainer.iteration_nb, evaluation_iterations)
    should_eval_losses(iteration_nb, eval_iterations::Integer) = 
        iteration_nb % eval_iterations == 0
    should_eval_losses(iteration_nb, eval_iterations) = iteration_nb in eval_iterations

    function maybe_save()
        should_save() || return
        save_dir = joinpath(output_dir, "point_cloud", "iteration_$(trainer.iteration_nb)")
        mkpath(save_dir)
        save_path = joinpath(save_dir, "point_cloud.ply")
        # TODO: this is not a point cloud, so why call it that?
        save_to_ply(save_path, gaussians)
        println("Saved Gaussians model after $(trainer.iteration_nb) iterations to \
                 $save_path")
    end

    should_save() = should_save(trainer.iteration_nb, save_iterations)
    should_save(iteration_nb, save_iterations::Integer) = 
        iteration_nb % save_iterations == 0
    should_save(iteration_nb, save_iterations) = iteration_nb in save_iterations

    try
        while trainer.iteration_nb <= get_nb_iterations(training_options)
            maybe_eval_losses()
            update!(gui)
            training_step!(trainer) 
            maybe_save()
        end
        close!(gui)
    catch e
        if e isa InterruptException
            println("Stopping training")
        else
            rethrow(e)
        end
    finally
        close!(gui)
    end
    maybe_unsafe_free!(trainer)
    maybe_unsafe_free!(optimizer)
    maybe_unsafe_free!(gaussians)
end


"""
    process_colmap(images_dir)

Run COLMAP SfM on the provided directory containing images. We assume there is only one
camera (model) is used. The output files will be written inside a sibling directory
"colmap" of `images_dir`.

We return the directory containing the SfM results in .txt form.
"""  # TODO: use .bin, compatibility with reference 3DGS
function process_colmap(images_dir)
    colmap_output_dir = normpath(joinpath(images_dir, "..", "colmap"))
    mkpath(colmap_output_dir)
    distorted_dir = joinpath(colmap_output_dir, "distorted")
    mkpath(distorted_dir)
    db_path = joinpath(distorted_dir, "database.db")

    colmap_base_cmd_args = if !Sys.iswindows()
        [get(ENV, "TDGS_COLMAP", "colmap")]
        # Default assumption: on PATH.
    else
        if "TDGS_COLMAP" in keys(ENV)
            [ENV["TDGS_COLMAP"]]
        else
            ["cmd.exe", "/c", "colmap"]
            # Windows can be annoying and not properly resolve it without cmd
            # We cannot use "cmd.exe /c colmap" as this entire part is then interpreted
            # as the executable
        end
    end
    cmd_args = [
        colmap_base_cmd_args...,
        "feature_extractor",
        "--database_path", db_path,
        "--image_path", images_dir,
        "--ImageReader.camera_model", "OPENCV", 
        "--ImageReader.single_camera", "1",
        "--FeatureExtraction.use_gpu", "1"
    ]
    run(Cmd(cmd_args))

    cmd_args = [
        colmap_base_cmd_args...,
        "exhaustive_matcher",
        "--database_path", db_path,
        "--FeatureMatching.use_gpu", "1"
    ]
    run(Cmd(cmd_args))

    cmd_args = [
        colmap_base_cmd_args...,
        "mapper",
        "--database_path", db_path,
        "--image_path", images_dir,
        "--output_path", distorted_dir,
        "--Mapper.ba_global_function_tolerance=0.000001",
        "--Mapper.ba_refine_focal_length", "1",
        "--Mapper.ba_refine_principal_point", "1",
        "--Mapper.ba_refine_extra_params", "1"
    ]
    # (We can handle a non-centered principal point)
    run(Cmd(cmd_args))

    # There might now be multiple models: 0, 1, etc. We will use the one with the biggest
    # points3D.bin file.
    model_id = 0
    best_model_id = -1
    largest_points3D_size = -1
    handled_all_models = false
    while !handled_all_models
        model_dir = joinpath(distorted_dir, string(model_id))
        if !isdir(model_dir)
            handled_all_models = true
            continue  # i.e. break
        end

        points3D_size = filesize(joinpath(model_dir, "points3D.bin"))
        if points3D_size > largest_points3D_size
            largest_points3D_size = points3D_size
            best_model_id = model_id
        end

        model_id += 1
    end

    # model_id now contains the number of valid models
    if model_id >= 2
        @info "COLMAP created multiple models. The best one has id $best_model_id."
    end
    distorted_dir = joinpath(distorted_dir, string(best_model_id))

    undistorted_dir = joinpath(colmap_output_dir, "undistorted")
    mkpath(undistorted_dir)
    cmd_args = [
        colmap_base_cmd_args...,
        "image_undistorter",
        "--image_path", images_dir,
        "--input_path", distorted_dir,
        "--output_path", undistorted_dir,
        "--output_type", "COLMAP"
    ]
    run(Cmd(cmd_args))
    undistorted_sparse_dir = joinpath(undistorted_dir, "sparse")

    undistorted_sparse_txt_dir = joinpath(undistorted_dir, "sparse-txt")
    mkpath(undistorted_sparse_txt_dir)
    cmd_args = [
        colmap_base_cmd_args...,
        "model_converter",
        "--input_path", undistorted_sparse_dir,
        "--output_path", undistorted_sparse_txt_dir,
        "--output_type", "TXT"
    ]
    run(Cmd(cmd_args))
    return undistorted_dir
end

"""
    processed_colmap_to_cameras_with_images_and_gaussians_model(undistorted_dir)

Read in the output data from COLMAP to produce a `Vector{CameraWithImage}` and
`MutableTrainableGaussiansVector` initialized from the sparse point cloud.
"""
function processed_colmap_to_cameras_with_images_and_gaussians_model(undistorted_dir)
    D = GPU; DIL = CPU; I = Int32; F = Float32
    # DIL: ArrayDomain for the ImageLoader

    undistorted_sparse_txt_dir = joinpath(undistorted_dir, "sparse-txt")
    undistorted_images_dir = joinpath(undistorted_dir, "images")

    # cameras.txt: extract intrinsic matrix
    cam_info_line = split(last(eachline(
        joinpath(undistorted_sparse_txt_dir, "cameras.txt"))))
    undistorted_width, undistorted_height = parse.(I, cam_info_line[3:4])
    # (skip camera_id (1), model_type (PINHOLE))
    fx, fy, cx, cy = parse.(F, cam_info_line[5:end])

    # images.txt: extract poses
    iter = Iterators.map(
        t -> t[2],  # t = (index, line) 
        Iterators.filter(
            t -> t[1] >= 5 && isodd(t[1]),
            enumerate(
                eachline(joinpath(undistorted_sparse_txt_dir, "images.txt"))
            )
        )
    )
    # The first four lines are comments with some metadata
    # Even lines after that contain the (2D) keypoint coordinates for each image
    cameras_with_images_vector = map(iter) do line
        line_split = split(line)
        image_filename = line_split[10]

        qr, qx, qy, qz, tx, ty, tz = parse.(F, line_split[2:8])  # qr = qw
        R_w2c = quaternion_to_rotation_matrix(SA[qr, qx, qy, qz])
        t_w2c = @SVector [tx, ty, tz]
        
        return CameraWithImage(
            PinholeCamera(
                undistorted_width, undistorted_height,
                fx, fy,
                zero(F),
                cx, cy,
                F(DEFAULT_NEAR_CLIPPING_PLANE_F64), F(DEFAULT_FAR_CLIPPING_PLANE_F64),
                R_w2c, t_w2c;
                rotation_camera_world_order = WorldToCamera,
                translation_camera_world_order = WorldToCamera
            ),
            EagerImageLoader(  # TODO: allow for other ImageLoaders
                DIL, F, joinpath(undistorted_images_dir, image_filename),
            )
        )
    end

    # points3D.txt: Sparse point cloud
    points3d_txt_path = joinpath(undistorted_sparse_txt_dir, "points3D.txt")
    nb_points_line = Iterators.nth(
        eachline(points3d_txt_path),
        3
    )  # "# Number of points: ..., ..."
    nb_points = parse(Int, first(split(nb_points_line, ','))[21:end])
    points = Matrix{F}(undef, 3, nb_points)
    colors = Matrix{F}(undef, 3, nb_points)
    for (idx, line) = enumerate(eachline(points3d_txt_path))
        idx >= 4 || continue  # starts with comments about the format
        line_split = split(line)
        pt_idx = idx - 3
        points[:, pt_idx] .= parse.(F, line_split[2:4])
        colors[:, pt_idx] .= parse.(UInt8, line_split[5:7]) ./ F(255)
    end
    gaussians_vector = MutableTrainableGaussiansVector(
        points, colors;
        array_domain = D, int_type = I, float_type = F
    )

    return cameras_with_images_vector, gaussians_vector
end