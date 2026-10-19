@testset "Cameras" begin
    # General constructor
    cam_w2c = PinholeCamera(
        640, 320,
        fov_deg_to_focal_length(90., 640),
        fov_deg_to_focal_length(60., 320),
        0.,
        320., 160.,
        0., Inf64,
        SA[0.  1.  0.
           -1. 0.  0.
           0.  0. -1.],
        SA[1., 2., 3.];
        rotation_camera_world_order = WorldToCamera,
        # translation_camera_world_order is then the same, if we don't explicitly specify
    )

    # Internal, camera to world only
    cam_c2w = PinholeCamera(
        640, 320,
        fov_deg_to_focal_length(90., 640),
        fov_deg_to_focal_length(60., 320),
        0.,
        320., 160.,
        0., Inf64,
        SA[0. -1.  0.
           1.  0.  0.
           0.  0. -1.],
        SA[2., -1., 3.];
    )

    @test get_camera_to_world_transformation_4x4(cam_w2c) ≈ 
        get_camera_to_world_transformation_4x4(cam_c2w)
    @test get_world_to_camera_transformation_4x4(cam_w2c) ≈ 
        get_world_to_camera_transformation_4x4(cam_c2w) 

    
    cam = cam_c2w
    @test get_camera_to_world_rotation(cam) ≈
            SA[
                0. -1.  0.
                1.  0.  0.
                0.  0. -1.
            ] ≈
            transpose(get_world_to_camera_rotation(cam)) ≈
            inv(get_world_to_camera_rotation(cam))

    @test get_position(cam) ≈ SA[2., -1., 3.]
    @test get_fov_x_deg(cam) ≈ 90
    @test get_fov_y_deg(cam) ≈ 60

    pt = SA[2., -1., 0.]
    proj_pt, depth = project_with_depth(pt, cam)
    @test proj_pt ≈ SA[320., 160.]
    @test depth ≈ 3.
    @test check_depth(depth, cam)

    pt = SA[2., -1., 4.]
    proj_pt, depth = project_with_depth(pt, cam)
    @test proj_pt ≈ SA[320., 160.]
    @test depth ≈ -1.
    @test !check_depth(depth, cam)

    pt = SA[2. - sqrt(3)/3, -2., 2.]
    @test get_world_to_camera_transformation_3x4(cam) * [pt; 1] ≈ 
        SA[-1., sqrt(3)/3, 1.]  # in viewspace, on the side of the view frustum
    # Found by reasoning on the field of view: at depth z=1 the frustrum edge is
    # x = ±tan(fov_x/2), y = ±tan(fov_y/2)
    # Reasoning on the focal length:
    @test -1 ≈ -get_width(cam) / (2 * get_focal_length_x(cam))
    # A depth-step of focal_length_x x-px units brings the side to (-)width/2,
    # so a depth-step of 1 yields the above
    @test sqrt(3)/3 ≈ get_height(cam) / (2 * get_focal_length_y(cam))
    @test project(pt, cam) ≈ SA[0., 320.]
end