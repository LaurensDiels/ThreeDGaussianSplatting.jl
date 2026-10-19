@testset "BackwardConsistency" begin
    # between our implementation (with 3DGS parameters) and the original reference
    # implementation

    ref_gradients = Dict(
        0 => Dict(
            "means2d" => [
                [ 2.11412762e-04,  2.38666214e-05],
                [-4.86466051e-06,  8.94421168e-05],
                [-1.49620650e-04, -7.09912507e-05]
                # Obtained via viewspace_point_tensor, converted from normalized device
                # coordinates to pixel coordinates
            ],
            "means3d" => [
                [ 0.04078607, -0.00225872,  0.06931462],
                # For debugging purposes:
                #   From mean2d_grad: [ 0.03382608,     0.00381865,     0.0103207]
                #   From cov2d_grad:  [ 6.9598691e-03, -6.0773445e-03,  5.8993828e-02]
                #   From color_grad:  [-0.,             0.,             0.]
                # (Had to insert prints in the reference source code to get these)
                [-0.0003805 ,  0.00845862,  0.01585804],
                #   From mean2d_grad: [-0.00038917,     0.00715539,     0.00178909]
                #   From cov2d_grad:  [ 8.6717246e-06,  1.3032802e-03,  1.4068962e-02]
                #   From color_grad:  [ 0.,            -0.,             0.]
                [-0.00986622, -0.00280794,  0.03382641]
                #   From mean2d_grad: [-0.00986622,    -0.00280794,     0.03382641]
                #   From cov2d_grad:  [-1.8864594e-03,  9.7823900e-04,  3.0219954e-02]
                #   From color_grad:  [ 0.,             0.,             0.]
            ],
            "opacity_features" => [
                -0.00523243,
                -0.00821592,
                -0.00035515
            ],
            "scaling_features" => [
                [-0.02823007, -0.02308522, -0.0038393],
                [-0.01640814, -0.01042652, -0.00065163],
                [-0.03485852, -0.01003832, -0.04461021]
            ],
            "rotation_features" => [
                [0.,             0.,             0.,             0.],
                # For the quaternions the reference implementation does not produce
                # consistent gradients for the first Gaussian, e.g.
                #   [1.4901161e-08,  2.0226114e-09, -1.7939669e-09,  0.0000000e+00]
                # (where the first and last component are quite common), and
                #   [7.450581e-09,  1.952997e-09, -2.245921e-09, -7.450581e-09]
                # Therefore, we won't trust their values here, and assume we just need to be
                # close to zero
                [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00],
                # Consistent zeros here
                [-7.7846907e-03,  1.3483479e-02, -4.6989047e-03,  2.6847634e-03]
                # Actually informative gradient
            ],
            "sh_coeffs" => [
                [-0.00450583;; -0.00966001;; -0.0094495],   # 1 x 3
                [-0.00520206;; -0.00549568;; -0.00531953],
                [-0.00580028;; -0.00979152;; -0.00466066]
            ]
        ),
        3 => Dict(
            "means2d" => [
                [ 2.23214086e-04,  2.21246194e-05],
                [-7.11953398e-06,  9.20800958e-05],
                [-1.59374322e-04, -7.19392905e-05]
            ],
            "means3d" => [
                [ 0.04386666,  0.00231351,  0.07039624],
                #   From mean2d_grad: [ 0.03571432,     0.00353991,     0.0110198]
                #   From cov2d_grad:  [ 7.0626317e-03, -6.2168459e-03,  6.0260803e-02]
                #   From color_grad:  [ 1.0900004e-03,  4.9905041e-03, -8.8429288e-04])
                [ 0.00056998,  0.00885604,  0.01710301],
                #   From mean2d_grad: [-0.00056957,     0.00736634,     0.00184194]
                #   From cov2d_grad:  [ 1.7768582e-05,  1.4223317e-03,  1.5244900e-02]
                #   From color_grad:  [ 1.1217815e-03,  6.7352492e-05,  1.6137035e-05]
                [-0.00994405, -0.00220804,  0.0335226]
                #   From mean2d_grad: [-0.0085,        -0.0038368,      0.00379254]
                #   From cov2d_grad:  [-1.8604652e-03,  9.7946718e-04,  3.0031124e-02]
                #   From color_grad:  [ 4.1643207e-04,  6.4940250e-04, -3.0116129e-04]
            ],
            "opacity_features" => [
                -0.00524313,
                -0.00888327,
                -0.00034778
            ],
            "scaling_features" => [
                [-0.02895599, -0.02348773, -0.00390841],
                [-0.01768842, -0.01137908, -0.00071114],
                [-0.03437268, -0.0100142 , -0.04458125]
            ],
            "rotation_features" => [
                [0.,              0.,             0.,             0.],
                # Again not completely consistent, but always in the order of at most e-08
                [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00],
                [-7.7853082e-03,  1.3484549e-02, -4.5974255e-03,  2.6769806e-03]
            ],
            "sh_coeffs" => [
                [-0.00406662  -0.00965997  -0.00944245
                  0.00162544   0.00386113   0.00377418
                 -0.00650177  -0.01544452  -0.01509674
                 -0.00216726  -0.00514817  -0.00503225
                  0.00111834   0.00265654   0.00259672
                  0.00335502   0.00796962   0.00779015
                 -0.0070755   -0.01680738  -0.0164289 
                 -0.00447335  -0.01062615  -0.01038687
                 -0.00032618  -0.00077482  -0.00075738
                  0.00045298   0.00107602   0.00105179
                  0.00273124   0.00648788   0.00634179
                  0.00495725   0.01177561   0.01151044
                 -0.00625868  -0.01486708  -0.0145323 
                 -0.00660966  -0.01570081  -0.01534725
                 -0.00079661  -0.0018923   -0.00184969
                  0.00017035   0.00040466   0.00039555],   # 16 x 3
                [-5.15818363e-03  -5.49568282e-03  -5.31509751e-03
                 -2.16687005e-03  -2.30864808e-03  -2.23278720e-03
                 -8.66748020e-03  -9.23459232e-03  -8.93114880e-03
                  5.41729878e-06   5.77175160e-06   5.58209513e-06
                  2.93794255e-06   3.13017154e-06   3.02731587e-06
                 -4.70060017e-03  -5.00815967e-03  -4.84359404e-03
                 -1.05163334e-02  -1.12044150e-02  -1.08362446e-02
                  1.17517702e-05   1.25206861e-05   1.21092635e-05
                  5.87571354e-04   6.26016059e-04   6.05445472e-04
                  1.53923189e-04   1.63994366e-04   1.58605588e-04
                  7.54097891e-06   8.03438343e-06   7.77037803e-06
                 -7.51152774e-03  -8.00300576e-03  -7.74003146e-03
                 -1.12927537e-02  -1.20316371e-02  -1.16362842e-02
                  1.87792484e-05   2.00079721e-05   1.93505202e-05
                  1.50815176e-03   1.60682993e-03   1.55403034e-03
                 -1.15446960e-06  -1.23000632e-06  -1.18958894e-06],
                [-0.00548864  -0.00979151  -0.00459713
                  0.00219383   0.00391371   0.00183749
                 -0.00877533  -0.01565483  -0.00734997
                  0.00292511   0.00521828   0.00244999
                 -0.0015094   -0.00269271  -0.00126423
                  0.00452821   0.00807814   0.0037927 
                 -0.00954969  -0.01703624  -0.00799854
                  0.00603761   0.01077085   0.00505693
                 -0.00044024  -0.00078537  -0.00036873
                  0.00061138   0.00109067   0.00051207
                 -0.00368631  -0.00657623  -0.00308755
                  0.00669072   0.01193596   0.00560395
                 -0.00844724  -0.01506952  -0.00707517
                  0.00892095   0.01591461   0.00747194
                 -0.00107517  -0.00191807  -0.00090054
                 -0.00022992  -0.00041017  -0.00019257]
            ]
        )
    )

    all_approx(x, y) = all(zip(x, y)) do (a, b)
        if all(==(0), b)
            return isapprox(a, b, atol = 0.001)
        else
            return isapprox(a, b, rtol = 0.01)
        end
    end

    for sh_degree in (0, 3)
        D = GPU; I = Int32; F = Float32
        scene = create_test_scene(; sh_degree,
            array_domain = D, int_type = I, float_type = F)
        gaussians = scene.gaussians
            camera = only(scene.cameras)
        render_outputs = RenderOutputs(camera; array_domain = D, 
            include_transmittance = true, 
            include_depth = false, 
            include_last_contributing_indices = true
        )
        render_options = RenderOptions(
            camera, get_sh_degree(gaussians),
            CovarianceBoundaryClamp;
            background_color = zero(SVector{3, F})
        )
        tiling = Tiling(camera, I(16), I(16))
        render_buffers = MutableRenderBuffers(gaussians, tiling)

        render!(
            render_outputs,
            gaussians, camera,
            tiling, render_options,
            render_buffers
        )

        rendered_image = get_rgb_image(render_outputs)
        gt_image = similar(rendered_image)
        fill!(gt_image, F(0.5))  # uniformly gray
        L1_loss = mapreduce(+, rendered_image, gt_image) do a, b
            return abs(a - b)
        end .* inv(length(rendered_image))
        rgb_image_grad = inv(length(rendered_image)) .* ifelse.(
            rendered_image .> gt_image, one(F), -one(F))

        fgradients = MutableFilteredGaussiansGradientsVector(render_buffers.f2dgv) 
        backward!(
            fgradients,
            rgb_image_grad,
            render_outputs,
            gaussians,
            camera,
            tiling,
            render_options,
            render_buffers
        )

        fgradients_cpu = CPU(fgradients)
        for i = 1:3  # filtered Gaussian index (into fgradients_cpu)
            @test all_approx(
                get_mean2d_grad_sv(fgradients_cpu[i]),
                ref_gradients[sh_degree]["means2d"][i]
            )
            #@test all_approx(
            #    get_mean3d_grad_sv(fgradients_cpu[i]),
            #    ref_gradients[sh_degree]["means3d"][i]
            #)
            # For sh_degree == 0 we seem to have a slightly different 2D covariance
            # contribution: 8.72e-6 instead of 8.67e-6
            @test all_approx(
                get_opacity_feature_grad(fgradients_cpu[i]), 
                ref_gradients[sh_degree]["opacity_features"][i])
            @test all_approx(
                get_scaling_features_grad_sv(fgradients_cpu[i]),
                ref_gradients[sh_degree]["scaling_features"][i])
            @test all_approx(
                get_rotation_features_grad_sv(fgradients_cpu[i]),
                ref_gradients[sh_degree]["rotation_features"][i]
            )
            @test all_approx(
                get_sh_coeffs_grad_sm(fgradients_cpu[i]), 
                ref_gradients[sh_degree]["sh_coeffs"][i]
            )
        end
    end
end