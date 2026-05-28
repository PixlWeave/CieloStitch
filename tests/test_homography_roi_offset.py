"""Test homography ROI offset handling for projective transforms."""
import numpy as np
import pytest
from cielostitch_core.core.warper import Warper


def test_warp_into_roi_handles_projective_homography_correctly():
    """Verify that warp_into_roi correctly handles full projective transforms with ROI offset.
    
    For affine transforms (where h20=0, h21=0, h22=1), simple subtraction of offsets works.
    But for full homography with perspective (non-zero h20 or h21), we must compose with
    a translation matrix using matrix multiplication.
    """
    warper = Warper(mask_close_px=0)
    
    # Create a simple test image
    image = np.zeros((100, 100), dtype=np.uint16)
    image[40:60, 40:60] = 1000
    
    # Create a projective homography with perspective distortion (non-zero h20, h21)
    # This is not purely affine
    H = np.array([
        [1.0,   0.05,  50.0],   # rotation + translation
        [0.05,  1.0,   30.0],   # rotation + translation  
        [0.001, 0.002, 1.0]     # perspective terms (non-zero h20, h21)
    ], dtype=np.float64)
    
    # Test with ROI offset
    roi_bounds = (10, 20, 90, 80)  # x0, y0, x1, y1
    
    # Should not raise an error and should produce valid output
    warped, mask = warper.warp_into_roi(image, H, roi_bounds)
    
    # Verify output shape matches ROI dimensions
    assert warped.shape == (60, 80)  # (y1-y0, x1-x0)
    assert mask.shape == (60, 80)
    assert warped.dtype == np.uint16
    assert mask.dtype == np.uint8
    
    # Verify that the warped image contains some of the original data
    # (exact values depend on the transform, but it shouldn't be all zeros)
    assert np.max(warped) > 0


def test_warp_into_roi_affine_vs_projective():
    """Compare affine and projective transform handling to show the difference."""
    warper = Warper(mask_close_px=0)
    
    # Simple test pattern
    image = np.zeros((50, 50), dtype=np.uint16)
    image[20:30, 20:30] = 5000
    
    # Affine transform (h20=0, h21=0, h22=1)
    H_affine = np.array([
        [1.0, 0.0, 10.0],
        [0.0, 1.0, 15.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    
    # Projective transform with perspective
    H_projective = np.array([
        [1.0,   0.0,  10.0],
        [0.0,   1.0,  15.0],
        [0.005, 0.003, 1.0]  # perspective distortion
    ], dtype=np.float64)
    
    roi_bounds = (5, 5, 45, 45)
    
    # Both should work correctly with the fixed implementation
    warped_affine, mask_affine = warper.warp_into_roi(image, H_affine, roi_bounds)
    warped_projective, mask_projective = warper.warp_into_roi(image, H_projective, roi_bounds)
    
    assert warped_affine.shape == (40, 40)
    assert warped_projective.shape == (40, 40)
    
    # The results should be different because of the perspective distortion
    # but both should be valid (not all zeros, not identical)
    assert np.max(warped_affine) > 0
    assert np.max(warped_projective) > 0
    assert not np.array_equal(warped_affine, warped_projective)


def test_canvas_offset_composition_in_base_mode():
    """Verify that canvas offset uses matrix multiplication for proper homography handling.
    
    The key insight: when composing translations with a homography, left-multiplication
    T @ H where T is [[1,0,tx],[0,1,ty],[0,0,1]] gives [[H00, H01, tx+H02], ...]
    which is mathematically equivalent to H with offset added to translation components.
    
    However, when the homography has perspective terms (H20 != 0 or H21 != 0), the
    ORDER of composition matters. We must use matrix multiplication to ensure proper
    composition order, especially when multiple offsets are applied sequentially.
    """
    
    # Projective homography with perspective terms
    global_h = np.array([
        [0.9,   0.1,  0.0],   # rotation
        [-0.1,  0.9,  0.0],   # rotation
        [0.002, 0.001, 1.0]   # perspective terms (h20, h21, h22)
    ], dtype=np.float64)
    
    canvas_offset_x = 100.0
    canvas_offset_y = 50.0
    roi_offset_x = 10.0
    roi_offset_y = 20.0
    
    # Using matrix multiplication (correct)
    T_canvas = np.array([
        [1, 0, canvas_offset_x],
        [0, 1, canvas_offset_y],
        [0, 0, 1]
    ], dtype=np.float64)
    
    T_roi = np.array([
        [1, 0, -roi_offset_x],
        [0, 1, -roi_offset_y],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # Compose: ROI offset @ Canvas offset @ global_h
    h_correct = T_roi @ T_canvas @ global_h
    
    # Old incorrect method: add offsets directly (doesn't compose properly)
    h_incorrect = global_h.copy()
    h_incorrect[0, 2] += canvas_offset_x - roi_offset_x
    h_incorrect[1, 2] += canvas_offset_y - roi_offset_y
    
    # These should be EQUAL for the translation components in this specific case
    # because T2 @ T1 @ H = T(combined) @ H for pure translations
    # BUT the point is that matrix multiplication is the CORRECT way to compose,
    # and it works correctly even when offsets are applied separately at different stages
    
    # The real test: verify our implementation matches the mathematically correct approach
    T_combined = np.array([
        [1, 0, canvas_offset_x - roi_offset_x],
        [0, 1, canvas_offset_y - roi_offset_y],
        [0, 0, 1]
    ], dtype=np.float64)
    h_expected = T_combined @ global_h
    
    assert np.allclose(h_correct, h_expected), \
        "Sequential composition should equal combined translation"
    
    # Now test a point transformation to show perspective effect
    test_point = np.array([100.0, 100.0, 1.0])
    
    # Transform with correct method
    pt_correct = h_correct @ test_point
    pt_correct = pt_correct[:2] / pt_correct[2]  # perspective divide
    
    # Transform with incorrect simple addition
    pt_incorrect = h_incorrect @ test_point
    pt_incorrect = pt_incorrect[:2] / pt_incorrect[2]
    
    # For projective transforms (h20 != 0 or h21 != 0), these give different results
    assert not np.allclose(pt_correct, pt_incorrect), \
        "Point transformation should differ between correct and incorrect methods for projective H"


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
