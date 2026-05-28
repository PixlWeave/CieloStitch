# SPDX-License-Identifier: MIT

import numpy as np

from cielostitch_core.core.apap import APAPEstimator, APAPRegistrationResult
from cielostitch_core.core.warper import Warper


def test_project_image_noop_when_projection_disabled() -> None:
    warper = Warper(projection_mode="native")
    image = np.arange(64, dtype=np.uint16).reshape(8, 8)

    projected = warper.project_image(image)

    assert np.array_equal(projected, image)


def test_project_image_legacy_none_alias_maps_to_native() -> None:
    warper = Warper(projection_mode="none")

    assert warper.projection_mode == "native"
    assert warper.is_projection_enabled() is False


def test_project_image_cylindrical_preserves_shape_dtype() -> None:
    warper = Warper(
        projection_mode="cylindrical",
        focal_length_mm=24.0,
        sensor_width_mm=36.0,
        sensor_height_mm=24.0,
        camera_angle_deg=0.0,
    )
    image = np.zeros((32, 48), dtype=np.uint16)
    image[:, 20:28] = 1000

    projected = warper.project_image(image)

    assert projected.shape == image.shape
    assert projected.dtype == image.dtype
    assert np.isfinite(projected).all()


def test_project_image_cylindrical_handles_quarter_turn_axis_swap() -> None:
    warper = Warper(
        projection_mode="cylindrical",
        focal_length_mm=24.0,
        sensor_width_mm=36.0,
        sensor_height_mm=24.0,
        camera_angle_deg=90.0,
    )
    image = np.zeros((24, 40), dtype=np.uint16)
    image[8:16, :] = 500

    projected = warper.project_image(image)

    assert projected.shape == image.shape
    assert projected.dtype == image.dtype
    assert np.isfinite(projected).all()


def test_project_image_cylindrical_accepts_float16_input() -> None:
    warper = Warper(
        projection_mode="cylindrical",
        focal_length_mm=24.0,
        sensor_width_mm=36.0,
        sensor_height_mm=24.0,
        camera_angle_deg=0.0,
    )
    image = np.zeros((24, 40), dtype=np.float16)
    image[:, 12:28] = np.float16(0.5)

    projected = warper.project_image(image)

    assert projected.shape == image.shape
    assert projected.dtype == image.dtype
    assert np.isfinite(projected.astype(np.float32)).all()


def test_project_image_cylindrical_clamps_invalid_edge_angles() -> None:
    warper = Warper(
        projection_mode="cylindrical",
        focal_length_mm=4.0,
        sensor_width_mm=36.0,
        sensor_height_mm=24.0,
        camera_angle_deg=0.0,
    )
    image = np.zeros((24, 120), dtype=np.uint16)
    image[:, 54:66] = 2000

    projected = warper.project_image(image)

    assert projected.shape == image.shape
    assert projected.dtype == image.dtype
    assert np.isfinite(projected.astype(np.float32)).all()
    assert np.all(projected[:, 0] == 0)
    assert np.all(projected[:, -1] == 0)


def test_warp_integer_translation_preserves_black_rgb_pixels_in_mask() -> None:
    warper = Warper(mask_close_px=0)
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[1, 1] = np.array([0, 0, 0], dtype=np.uint8)
    image[1, 2] = np.array([10, 20, 30], dtype=np.uint8)
    H = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    warped, mask = warper.warp(image, H, output_shape=(4, 5))

    assert warped.shape == (4, 5, 3)
    assert mask.shape == (4, 5)
    assert mask[1, 2] == 1
    assert mask[1, 3] == 1
    assert np.array_equal(warped[1, 2], np.array([0, 0, 0], dtype=np.uint8))
    assert np.array_equal(warped[1, 3], np.array([10, 20, 30], dtype=np.uint8))


def test_warp_apap_falls_back_to_bootstrap_homography() -> None:
    # When local_warp_ready=False and no dense remap, the fallback must apply the
    # homography AND account for the ROI origin offset.  With identity H and
    # roi_origin=(0,0) the output should be pixel-identical to the input.
    warper = Warper(mask_close_px=0)
    image = np.zeros((8, 8), dtype=np.uint16)
    image[2:6, 2:6] = 100
    registration = APAPRegistrationResult(
        homography=np.eye(3, dtype=np.float64),
        inlier_mask=np.ones(4, dtype=bool),
        diagnostics={"local_warp_ready": False},
    )

    warped, mask = warper.warp_apap(image, registration, image.shape, roi_origin=(0.0, 0.0))

    assert np.array_equal(warped, image)
    assert mask.shape == image.shape
    assert np.count_nonzero(mask) == np.count_nonzero(image)


def test_warp_apap_fallback_respects_roi_origin_offset() -> None:
    # When the ROI starts at (ox, oy) in canvas space the fallback path must
    # apply T(-ox,-oy) @ H so that canvas pixels are looked up correctly from
    # the source image.  A pure-translation H that shifts by (+ox, +oy) combined
    # with roi_origin=(ox, oy) should produce the identity mapping into the output
    # buffer (net shift is zero).
    warper = Warper(mask_close_px=0)
    image = np.zeros((10, 10), dtype=np.uint16)
    image[3:7, 3:7] = 200
    ox, oy = 5.0, 3.0
    # H maps panel → canvas with translation (+ox, +oy)
    H = np.array([[1, 0, ox], [0, 1, oy], [0, 0, 1]], dtype=np.float64)
    registration = APAPRegistrationResult(
        homography=H,
        inlier_mask=np.ones(4, dtype=bool),
        diagnostics={"local_warp_ready": False},
    )

    warped, mask = warper.warp_apap(image, registration, image.shape, roi_origin=(ox, oy))

    # T(-ox,-oy) @ H = identity → warped must equal image
    assert np.array_equal(warped, image)
    assert np.count_nonzero(mask) == np.count_nonzero(image)


def test_warp_apap_uses_dense_remap_when_available() -> None:
    warper = Warper(mask_close_px=0)
    image = np.arange(25, dtype=np.uint16).reshape(5, 5)
    remap_x, remap_y = np.meshgrid(
        np.arange(5, dtype=np.float32),
        np.arange(5, dtype=np.float32),
    )
    registration = APAPRegistrationResult(
        homography=np.eye(3, dtype=np.float64),
        inlier_mask=np.ones(4, dtype=bool),
        remap_x=remap_x,
        remap_y=remap_y,
        diagnostics={"local_warp_ready": True},
    )

    warped, mask = warper.warp_apap(image, registration, image.shape, roi_origin=(-10.0, -20.0))

    assert np.array_equal(warped, image)
    assert np.all(mask == 1)


def test_warp_apap_closes_invalid_dense_remap_holes_with_nearest_valid_fill() -> None:
    warper = Warper(mask_close_px=1)
    image = np.full((5, 5), 100, dtype=np.uint16)
    remap_x, remap_y = np.meshgrid(
        np.arange(5, dtype=np.float32),
        np.arange(5, dtype=np.float32),
    )
    remap_x[2, 2] = -1.0
    remap_y[2, 2] = -1.0
    registration = APAPRegistrationResult(
        homography=np.eye(3, dtype=np.float64),
        inlier_mask=np.ones(4, dtype=bool),
        remap_x=remap_x,
        remap_y=remap_y,
        diagnostics={"local_warp_ready": True},
    )

    warped, mask = warper.warp_apap(image, registration, image.shape)

    assert mask[2, 2] == 1
    assert warped[2, 2] == 100


def test_warp_apap_prefers_mesh_homographies_when_available() -> None:
    warper = Warper(mask_close_px=0)
    image = np.arange(25, dtype=np.uint16).reshape(5, 5)
    mesh_x = np.arange(5, dtype=np.float64)
    mesh_y = np.arange(5, dtype=np.float64)
    mesh_h = np.repeat(np.eye(3, dtype=np.float64)[None, None, :, :], 25, axis=0).reshape(5, 5, 3, 3)
    registration = APAPRegistrationResult(
        homography=np.eye(3, dtype=np.float64),
        inlier_mask=np.ones(4, dtype=bool),
        mesh_x=mesh_x,
        mesh_y=mesh_y,
        mesh_homographies=mesh_h,
        diagnostics={"local_warp_ready": True},
    )

    warped, mask = warper.warp_apap(image, registration, image.shape)

    assert np.array_equal(warped, image)
    assert np.all(mask == 1)


def test_warp_apap_identity_mesh_remains_correct_after_canvas_offset() -> None:
    warper = Warper(mask_close_px=0)
    estimator = APAPEstimator()
    image = np.arange(25, dtype=np.uint16).reshape(5, 5)
    mesh_x = np.arange(5, dtype=np.float64)
    mesh_y = np.arange(5, dtype=np.float64)
    mesh_h = np.repeat(np.eye(3, dtype=np.float64)[None, None, :, :], 25, axis=0).reshape(5, 5, 3, 3)
    registration = APAPRegistrationResult(
        homography=np.eye(3, dtype=np.float64),
        inlier_mask=np.ones(4, dtype=bool),
        mesh_x=mesh_x,
        mesh_y=mesh_y,
        mesh_homographies=mesh_h,
        diagnostics={"local_warp_ready": True},
    )
    registration = estimator.with_canvas_offset(registration, 10.0, 20.0)

    warped, mask = warper.warp_apap(image, registration, image.shape, roi_origin=(10.0, 20.0))

    assert np.array_equal(warped, image)
    assert np.all(mask == 1)


def test_warp_apap_skips_singular_mesh_nodes_without_failing() -> None:
    warper = Warper(mask_close_px=0)
    image = np.arange(25, dtype=np.uint16).reshape(5, 5)
    mesh_x = np.arange(5, dtype=np.float64)
    mesh_y = np.arange(5, dtype=np.float64)
    mesh_h = np.repeat(np.eye(3, dtype=np.float64)[None, None, :, :], 25, axis=0).reshape(5, 5, 3, 3)
    mesh_h[2, 2] = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
    registration = APAPRegistrationResult(
        homography=np.eye(3, dtype=np.float64),
        inlier_mask=np.ones(4, dtype=bool),
        mesh_x=mesh_x,
        mesh_y=mesh_y,
        mesh_homographies=mesh_h,
        diagnostics={"local_warp_ready": True},
    )

    warped, mask = warper.warp_apap(image, registration, image.shape)

    assert np.count_nonzero(mask) > 0
    assert warped.shape == image.shape


def test_warp_apap_builds_nontrivial_remap_from_support_residuals() -> None:
    warper = Warper(mask_close_px=0)
    image = np.zeros((6, 6), dtype=np.uint16)
    image[:, 2] = 100
    registration = APAPRegistrationResult(
        homography=np.eye(3, dtype=np.float64),
        inlier_mask=np.ones(4, dtype=bool),
        support_source_points=np.array([[2.0, 1.0], [2.0, 4.0]], dtype=np.float64),
        support_destination_points=np.array([[3.0, 1.0], [3.0, 4.0]], dtype=np.float64),
        support_residuals=np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float64),
        diagnostics={
            "local_warp_ready": True,
            "mesh_cols": 3,
            "mesh_rows": 3,
            "kernel_sigma_px": 8.0,
            "min_local_support": 2,
        },
    )

    warped, mask = warper.warp_apap(image, registration, image.shape)

    assert mask.shape == image.shape
    assert np.count_nonzero(mask) > 0
    assert not np.array_equal(warped, image)


def test_warp_apap_ignores_distant_supports_for_local_nodes() -> None:
    warper = Warper(mask_close_px=0)
    image = np.zeros((10, 10), dtype=np.uint16)
    image[:, 1] = 100
    image[:, 8] = 200
    registration = APAPRegistrationResult(
        homography=np.eye(3, dtype=np.float64),
        inlier_mask=np.ones(4, dtype=bool),
        support_source_points=np.array([[1.0, 2.0], [1.0, 7.0], [8.0, 2.0], [8.0, 7.0]], dtype=np.float64),
        support_destination_points=np.array([[2.0, 2.0], [2.0, 7.0], [8.0, 2.0], [8.0, 7.0]], dtype=np.float64),
        support_residuals=np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float64),
        diagnostics={
            "local_warp_ready": True,
            "mesh_cols": 5,
            "mesh_rows": 4,
            "kernel_sigma_px": 1.2,
            "min_local_support": 2,
        },
    )

    warped, _mask = warper.warp_apap(image, registration, image.shape)

    left_mass = int(np.sum(warped[:, :4]))
    right_mass = int(np.sum(warped[:, 6:]))
    assert left_mass > 0
    assert right_mass > 0
    assert left_mass != right_mass


def test_warp_apap_skips_local_remap_when_local_warp_not_ready() -> None:
    warper = Warper(mask_close_px=0)
    image = np.zeros((6, 6), dtype=np.uint16)
    image[:, 2] = 100
    registration = APAPRegistrationResult(
        homography=np.eye(3, dtype=np.float64),
        inlier_mask=np.ones(4, dtype=bool),
        support_source_points=np.array([[2.0, 1.0], [2.0, 4.0]], dtype=np.float64),
        support_destination_points=np.array([[3.0, 1.0], [3.0, 4.0]], dtype=np.float64),
        support_residuals=np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float64),
        diagnostics={"local_warp_ready": False, "mesh_cols": 3, "mesh_rows": 3, "kernel_sigma_px": 8.0},
    )

    warped, mask = warper.warp_apap(image, registration, image.shape)

    assert np.array_equal(warped, image)
    assert np.count_nonzero(mask) == np.count_nonzero(image)
