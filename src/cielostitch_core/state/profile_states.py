# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 Debasish Saha
#
# Part of the CieloStitch open-source core.
#
# Licensed under the MIT License.
# See the LICENSE file for details.

import logging
from typing import Any, Dict, List, Tuple, ClassVar, Set
from dataclasses import dataclass, fields as dataclass_fields

logger = logging.getLogger(__name__)

AFieldDef = Dict[str, Any]
AFieldDefs = List[AFieldDef]
ASectionDef = Tuple[str, AFieldDefs] | Tuple[str, AFieldDefs, dict]
ASectionDefs = List[ASectionDef]
AFieldVals = Dict[str, Any]

from .profiles import (
    SolarProfile,
    LunarProfile,
    LandscapeProfile,
    GeneralProfile,
    NightscapeProfile,
    MilkyWayProfile,
    SolarHAlphaProfile,
    PanoramaProfile,
)

from ..config.constants import (
    IMAGE_PROFILES, GRID_MODES, UNKNOWN_OVERLAP_MODES,
    SCAN_ORDERS, START_CORNERS, ALTERNATING, BLEND_TYPES,
    MOUNT_PRECISIONS, GAIN_COMPENSATION_OPTIONS, SIMPLE_GAIN_METHODS,
    SEEING_CONDITIONS, SEAMLESS_OPTIONS, MAX_MULTIBAND_LEVELS,
    PROJECTION_MODES, TRANSFORM_MODES, BUNDLE_ADJUSTMENT_MODES, FPX_MODES,
    SESSION_GRID_COLS_MIN, SESSION_GRID_COLS_MAX, EDGE_AWARE_TYPES,
    SIMPLE_ENGINE_MODES,
)


PROFILE_CLASSES: Dict[str, Any] = {
    "solar": SolarProfile,
    "solar-h-alpha": SolarHAlphaProfile,
    "lunar": LunarProfile,
    "landscape": LandscapeProfile,
    "nightscape": NightscapeProfile,
    "milky-way": MilkyWayProfile,
    "panorama": PanoramaProfile,
    "general": GeneralProfile,
}

simple_engine: ASectionDef = \
    (
        "Simple Engine, session.svg",
        [
            {
                "name": "simple_transform_mode", "label": "Simple transform", "type": "enum",
                "choices": SIMPLE_ENGINE_MODES,
            },
            {
                "name": "simple_confidence_threshold", "label": "Confidence threshold", "type": "double",
                "min": 0.0, "max": 1.0, "step": 0.05, "decimals": 2,
            },
            {
                "name": "simple_registration_resol_mp", "label": "Registration res. (MP)", "type": "double",
                "min": -1.0, "max": 50.0, "step": 0.1, "decimals": 1,
            },
            {
                "name": "simple_seam_estimation_resol_mp", "label": "Seam res. (MP)", "type": "double",
                "min": -1.0, "max": 50.0, "step": 0.1, "decimals": 1,
            },
            {
                "name": "simple_compositing_resol_mp", "label": "Compositing res. (MP)", "type": "double",
                "min": -1.0, "max": 100.0, "step": 0.1, "decimals": 1,
            },
            {
                "name": "simple_wave_correction", "label": "Wave correction", "type": "bool",
            },
            {
                "name": "simple_scans_retry", "label": "Retry with SCANS", "type": "bool",
                "depends_on": {"field": "simple_transform_mode", "value": "scans", "negate": True},
            },
            {
                "name": "simple_engine_timeout_sec", "label": "Timeout (s)", "type": "int",
                "min": 0, "max": 86400, "step": 30,
            },
        ],
        {"depends_on": {"context": "engine", "value": "simple"}},
    )

grid: ASectionDef = \
    (
        "Capture Grid, session.svg",
        [
            {
                "name": "grid_cols", "label": "Columns", "type": "int",
                "min": SESSION_GRID_COLS_MIN, "max": SESSION_GRID_COLS_MAX,
            },
            {
                "name": "scan_order", "label": "Scan order", "type": "enum",
                "choices": SCAN_ORDERS,
                "depends_on": {"field": "stitch_mode", "values": GRID_MODES},
            },
            {
                "name": "start_corner", "label": "Start corner", "type": "enum",
                "choices": START_CORNERS,
                "depends_on": {"field": "stitch_mode", "values": GRID_MODES},
            },
            {
                "name": "alternating", "label": "Alternating", "type": "enum",
                "choices": ALTERNATING,
                "depends_on": {"field": "stitch_mode", "values": GRID_MODES},
            },
            {
                "name": "scan_icon_label", "label": "Scan pattern", "type": "thumb",
                "depends_on": {"field": "stitch_mode", "values": GRID_MODES},
            },
            {
                "name": "overlap_x_pct", "label": "Overlap X%", "type": "double", "min": 0.00,
                "max": 100.00, "step": 5.00, "decimals": 2,
                "depends_on": {"field": "stitch_mode", "value": "zero-overlap", "negate": True},
            },
            {
                "name": "overlap_y_pct", "label": "Overlap Y%", "type": "double", "min": 0.00,
                "max": 100.00, "step": 5.00, "decimals": 2,
                "depends_on": {"field": "stitch_mode", "value": "zero-overlap", "negate": True},
            },
            {
                "name": "staged_matching_mode", "label": "Staged mode", "type": "enum",
                "choices": ["auto", "disabled", "manual"],
                "depends_on": {
                    "all": [
                        {"field": "stitch_mode", "value": "freeform"},
                        {"context": "developer_mode", "value": True},
                    ],
                },
            },
            {
                "name": "refs_per_stage", "label": "Refs per stage", "type": "int",
                "min": 4, "max": 8, "step": 1,
                "depends_on": {
                    "all": [
                        {"field": "staged_matching_mode", "value": "manual"},
                        {"field": "stitch_mode", "value": "freeform"},
                        {"context": "developer_mode", "value": True},
                    ],
                },
            },
        ],
        # {"depends_on": {"context": "engine", "value": "simple", "negate": True}},
    )

session: ASectionDef = \
    (
        "Capture Session, session.svg",
        [
            {
                "name": "mount_precision", "label": "Mount precision", "type": "enum",
                "choices": MOUNT_PRECISIONS,
            },
            {
                "name": "seeing", "label": "Seeing condition", "type": "enum",
                "choices": SEEING_CONDITIONS,
            },
            {
                "name": "projection_mode", "label": "🌐 Projection mode", "type": "enum",
                "choices": PROJECTION_MODES,
            },
            {
                "name": "fpx_mode", "label": "Lens info source", "type": "enum",
                "choices": FPX_MODES,
                # "depends_on": {"field": "projection_mode", "value": "native", "negate": True},
            },
            {
                "name": "focal_length_mm", "label": "Focal length (mm)", "type": "double",
                "min": 0.0, "max": 6000.0, "step": 0.1, "decimals": 2,
                "depends_on": {"all": [
                    # {"field": "projection_mode", "value": "native", "negate": True},
                    {"field": "fpx_mode", "value": "camera"},
                ]},
            },
            {
                "name": "sensor_width_mm", "label": "Sensor width (mm)", "type": "double",
                "min": 0.0, "max": 100.0, "step": 0.1, "decimals": 2,
                "depends_on": {"all": [
                    # {"field": "projection_mode", "value": "native", "negate": True},
                    {"field": "fpx_mode", "value": "camera"},
                ]},
            },
            {
                "name": "sensor_height_mm", "label": "Sensor height (mm)", "type": "double",
                "min": 0.0, "max": 100.0, "step": 0.1, "decimals": 2,
                "depends_on": {"all": [
                    # {"field": "projection_mode", "value": "native", "negate": True},
                    {"field": "fpx_mode", "value": "camera"},
                ]},
            },
            {
                "name": "hfov_deg", "label": "Horiz. FOV (deg)", "type": "double",
                "min": 0.0, "max": 179.9, "step": 0.5, "decimals": 1,
                "depends_on": {"all": [
                    # {"field": "projection_mode", "value": "native", "negate": True},
                    {"field": "fpx_mode", "value": "fov"},
                ]},
            },
            {
                "name": "fpx_factor", "label": "fpx factor (0.01–120)", "type": "double",
                "min": 0.01, "max": 120, "step": 0.05, "decimals": 2,
                "depends_on": {"all": [
                    # {"field": "projection_mode", "value": "native", "negate": True},
                    {"field": "fpx_mode", "value": "factor"},
                ]},
            },
            {
                "name": "camera_angle_deg", "label": "Camera angle (deg)", "type": "double",
                "min": -360.0, "max": 360.0, "step": 1.0, "decimals": 1,
                # "depends_on": {"field": "projection_mode", "value": "native", "negate": True},
            },
        ],
        {"depends_on": {"all":[
            {"field": "stitch_mode", "value": "zero-overlap", "negate": True},
            {"field": "stitch_mode", "value": "auto", "negate": True},
            # {"context": "engine", "value": "simple", "negate": True},
        ],},},
    )

blending: ASectionDef = \
    (
        "Blending, profile.svg",
        [
            {
                "name": "blend_type", "label": "🔀 Blend type", "type": "enum",
                "choices": BLEND_TYPES,
            },
            {
                "name": "seamless_quality", "label": "Quality vs Speed", "type": "enum",
                "choices": SEAMLESS_OPTIONS,
                "depends_on": {"field": "blend_type", "values": ["seamless"]},
            },
            {
                "name": "multiband_levels", "label": "Multiband levels", "type": "int",
                "min": 1, "max": MAX_MULTIBAND_LEVELS, "step": 1,
                "depends_on": {"field": "blend_type", "values": ["multiband"]},
            },
            {
                "name": "blend_feather_px", "label": "Feather radius (px)", "type": "int",
                "min": 0, "max": 300, "step": 5,
                "depends_on": {"field": "blend_type", "values": ["feather"]},
            },
            {
                "name": "edge_aware_smoothing", "label": "Edge-aware blend",
                "type": "bool",
                "depends_on": {
                    "all": [
                        {"context": "advanced_mode", "value": True},
                        {"field": "blend_type", "values": EDGE_AWARE_TYPES},
                    ],
                }
            },
            {
                "name": "adaptive_mb_risk_boost_threshold", "label": "MB risk boost thres",
                "type": "double", "min": 0.00, "max": 1.00,
                "step": 0.05, "decimals": 2,
                "depends_on": {
                    "all": [
                        {"context": "advanced_mode", "value": True},
                        {"field": "blend_type", "value": "adaptive-multiband"},
                        {"context": "developer_mode", "value": True},
                    ],
                }
            },
            {
                "name": "adaptive_mb_low_risk_threshold", "label": "MB low-risk thres",
                "type": "double", "min": 0.00, "max": 1.00,
                "step": 0.05, "decimals": 2,
                "depends_on": {
                    "all": [
                        {"context": "advanced_mode", "value": True},
                        {"field": "blend_type", "value": "adaptive-multiband"},
                        {"context": "developer_mode", "value": True},
                    ],
                }
            },
            {
                "name": "adaptive_mb_max_boost", "label": "MB max level boost",
                "type": "int", "min": 0, "max": 2, "step": 1,
                "depends_on": {
                    "all": [
                        {"context": "advanced_mode", "value": True},
                        {"field": "blend_type", "value": "adaptive-multiband"},
                        {"context": "developer_mode", "value": True},
                    ],
                }
            },
            {
                "name": "ghost_guard_enabled", "label": "Ghosting guard",
                "type": "bool",
                "depends_on": {
                    "all": [
                        {"context": "advanced_mode", "value": True},
                        {"field": "blend_type", "values": ["feather", "adaptive-feather", "multiband", "adaptive-multiband", "seamless"]},
                    ],
                }
            },
            {
                "name": "ghost_guard_risk_threshold", "label": "Ghost risk threshold",
                "type": "double", "min": 0.00, "max": 1.00,
                "step": 0.05, "decimals": 2,
                "depends_on": {
                    "all": [
                        {"context": "advanced_mode", "value": True},
                        {"field": "blend_type", "values": ["feather", "adaptive-feather", "multiband", "adaptive-multiband", "seamless"]},
                        {"field": "ghost_guard_enabled", "value": True},
                        {"context": "developer_mode", "value": True},
                    ],
                }
            },
            {
                "name": "ghost_guard_feather_px", "label": "Ghost seam feather (px)",
                "type": "int", "min": 5, "max": 200, "step": 1,
                "depends_on": {
                    "all": [
                        {"context": "advanced_mode", "value": True},
                        {"field": "blend_type", "values": ["feather", "adaptive-feather", "multiband", "adaptive-multiband", "seamless"]},
                        {"field": "ghost_guard_enabled", "value": True},
                        {"context": "developer_mode", "value": True},
                    ],
                }
            },
            {
                "name": "photometric_min_overlap_px", "label": "Blend min overlap (px²)",
                "type": "int", "min": 0, "max": 1_000_000, "step": 500,
                "depends_on": {
                    "all": [
                        {"context": "advanced_mode", "value": True},
                        {"field": "blend_type", "value": "none", "negate": True},
                        {"context": "developer_mode", "value": True},
                    ],
                }
            },
            {
                "name": "photometric_full_confidence_px", "label": "Blend full overlap (px²)",
                "type": "int", "min": 0, "max": 2_000_000, "step": 5_000,
                "depends_on": {
                    "all": [
                        {"context": "advanced_mode", "value": True},
                        {"field": "blend_type", "value": "none", "negate": True},
                        {"context": "developer_mode", "value": True},
                    ],
                }
            },
        ],
        {"depends_on": {"all":[
            {"field": "stitch_mode", "value": "zero-overlap", "negate": True},
            {"field": "stitch_mode", "value": "auto", "negate": True},
            # {"context": "engine", "value": "simple", "negate": True},
        ],},},
    )

gain: ASectionDef = \
    (
        "Gain Compensation, profile.svg",
        [
            {
                "name": "gain_compensation", "label": "Gain compensation", "type": "enum",
                "choices": GAIN_COMPENSATION_OPTIONS,
            },
            {
                "name": "local_gain_tile_size", "label": "Local gain tile size", "type": "int",
                "min": 16, "max": 512, "step": 16,
                "depends_on": {
                    "all": [
                        {"field": "gain_compensation", "value": "local"},
                        {"context": "developer_mode", "value": True},
                    ],
                },
            },
            {
                "name": "gain_method", "label": "Simple gain method",
                "type": "enum", "choices": SIMPLE_GAIN_METHODS,
                "depends_on": {"field": "gain_compensation", "value": "simple"},
            },
            {
                "name": "lum_only_gain", "label": "Luminance only gain",
                "type": "bool",
                "depends_on": {
                    "any": [
                        {"field": "gain_compensation", "value": "simple"},
                        {"field": "blend_type", "value": "seamless"},
                    ],
                },
            },
            {
                "name": "histogram_matching", "label": "Histogram matching",
                "type": "bool",
                "depends_on": {"field": "gain_compensation", "value": "simple"},
            },
        ],
        {"depends_on": {"all":[
            {"field": "stitch_mode", "value": "zero-overlap", "negate": True},
            {"field": "stitch_mode", "value": "auto", "negate": True},
            # {"context": "engine", "value": "cielo"},
        ],},},
    )
# advanced

feature_detection: ASectionDef = \
    (
        "Feature Detection, profile.svg",
        [
            {
                "name": "detector_max_points", "label": "Max points/image", "type": "int",
                "min": 0, "max": 30000, "step": 500,
                "depends_on": {"field": "stitch_mode", "values": UNKNOWN_OVERLAP_MODES},
            },
            {
                "name": "detector_feature_sensitivity", "label": "Feature sensitivity", "type": "double",
                "min": 0.005, "max": 0.080, "step": 0.005, "decimals": 3,
                "depends_on": {"field": "stitch_mode", "values": UNKNOWN_OVERLAP_MODES},
            },
            {
                "name": "detector_downscale", "label": "Analyze smaller copy", "type": "double",
                "min": 0.30, "max": 1.0, "step": 0.10, "decimals": 2,
                "depends_on": {"field": "stitch_mode", "values": UNKNOWN_OVERLAP_MODES},
            },
        ],
        {"depends_on": {"all": [
            {"field": "stitch_mode", "value": "auto", "negate": True},
            # {"context": "engine", "value": "simple", "negate": True},
        ]}},
    )

feature_matching: ASectionDef = \
    (
        "Feature Matching, profile.svg",
        [
            {
                "name": "ratio_test", "label": "Match strictness", "type": "double", "min": 0.1, "max": 1.0,
                "step": 0.01, "decimals": 3,
                "depends_on": {"field": "stitch_mode", "values": UNKNOWN_OVERLAP_MODES},
            },
            {
                "name": "ransac_thresh", "label": "Alignment tolerance", "type": "double", "min": 0.5, "max": 9.5,
                "step": 0.1, "decimals": 2,
                "depends_on": {"field": "stitch_mode", "values": UNKNOWN_OVERLAP_MODES},
            },
            {
                "name": "min_inlier_ratio", "label": "Match quality thresh", "type": "double", "min": 0.01, "max": 0.4,
                "step": 0.01, "decimals": 3,
                "depends_on": {"field": "stitch_mode", "values": UNKNOWN_OVERLAP_MODES},
            },
            {
                "name": "min_inliers", "label": "Min good matches", "type": "int", "min": 4, "max": 64, "step": 1,
                "depends_on": {"field": "stitch_mode", "values": UNKNOWN_OVERLAP_MODES},
            },
            # {
            #     "name": "transform_mode", "label": "🔄 Transform mode", "type": "enum",
            #     "choices": TRANSFORM_MODES,
            #     # "help": "Controls the geometric model for feature matching: 'auto' (profile default),
            #     # 'affine' (rotation/scale), 'homography' (perspective, best for wide/pano), or 'translation' (shift only).",
            #     "depends_on": {"field": "stitch_mode", "values": UNKNOWN_OVERLAP_MODES},
            # },
        ],
        {"depends_on": {"all": [
            {"field": "stitch_mode", "value": "auto", "negate": True},
            # {"context": "engine", "value": "simple", "negate": True},
        ]}},
    )

transform: ASectionDef = \
    (
        "Transformation, profile.svg",
        [
            {
                "name": "transform_mode", "label": "🔄 Transform mode", "type": "enum",
                "choices": TRANSFORM_MODES,
                # "help": "Controls the geometric model for feature matching: 'auto' (profile default),
                # 'affine' (rotation/scale), 'homography' (perspective, best for wide/pano), or 'translation' (shift only).",
                "depends_on": {"field": "stitch_mode", "values": UNKNOWN_OVERLAP_MODES},
            },
        ],
        {"depends_on": {"all": [
            {"field": "stitch_mode", "value": "auto", "negate": True},
            # {"context": "engine", "value": "simple", "negate": True},
        ]}},
    )

grid_fallback: ASectionDef = \
    (
        "Grid Fallback, session.svg",
        [
            {
                "name": "bundle_adjustment_mode", "label": "Bundle adjustment", "type": "enum",
                "choices": BUNDLE_ADJUSTMENT_MODES,
                "depends_on": {"field": "stitch_mode", "values": UNKNOWN_OVERLAP_MODES},
            },
            {
                "name": "enable_grid_phase_fallback", "label": "Phase fallback", "type": "bool",
                "depends_on": {"field": "stitch_mode", "values": UNKNOWN_OVERLAP_MODES},
            },
            {
                "name": "enable_grid_shift_guard", "label": "Phase shift guard", "type": "bool",
                "depends_on": {"field": "stitch_mode", "values": UNKNOWN_OVERLAP_MODES},
            },
            {
                "name": "enable_grid_nominal_fallback", "label": "Nominal fallback", "type": "bool",
                "depends_on": {
                    "all": [
                        {"field": "stitch_mode", "values": UNKNOWN_OVERLAP_MODES},
                        {"context": "developer_mode", "value": True},
                    ],
                },
            },
        ],
        {"depends_on": {"all": [
            {"field": "stitch_mode", "value": "auto", "negate": True},
            # {"context": "engine", "value": "simple", "negate": True},
        ]}},
    )


def _all_basic_field_groups() -> ASectionDefs:
    return [simple_engine, blending, grid, session, transform]


def _all_advanced_field_groups() -> ASectionDefs:
    return [gain, feature_detection, feature_matching, grid_fallback]


def _profile_field_groups() -> ASectionDefs:  # x 8 profiles # for data
    return [blending, feature_detection, feature_matching]


@dataclass
class ProfileStates:

    # Class-level attributes for shared state
    initialized: ClassVar[bool] = False
    BAS_FIELD_GROUPS: ClassVar[ASectionDefs] = []
    ADV_FIELD_GROUPS: ClassVar[ASectionDefs] = []
    PROFILE_FIELD_GROUPS: ClassVar[ASectionDefs] = []
    PROFILE_FIELD_NAMES_UI: ClassVar[List[str]] = []
    PROFILE_FIELD_NAMES_ALL: ClassVar[List[str]] = []
    PROFILE_DEFAULT_VALUES: ClassVar[Dict[str, AFieldVals]] = {}
    PROFILE_MODIFIED_VALUES: ClassVar[Dict[str, AFieldVals]] = {}
    PROFILES: ClassVar[List[str]] = []

    @classmethod
    def prepare(cls) -> None:
        if cls.initialized:
            return
        cls.initialized = True

        # Build unified field_groups (same for all profiles)
        cls.BAS_FIELD_GROUPS = _all_basic_field_groups()
        cls.ADV_FIELD_GROUPS = _all_advanced_field_groups()
        cls.PROFILE_FIELD_GROUPS = _profile_field_groups()
        cls.BAS_DEPS = cls.get_dependencies(cls.BAS_FIELD_GROUPS)
        cls.ADV_DEPS = cls.get_dependencies(cls.ADV_FIELD_GROUPS)
        # print(cls.BAS_DEPS)
        # print(cls.ADV_DEPS)

        _profile_defaults: Dict[str, AFieldVals] = {}

        for profile_name in IMAGE_PROFILES:
            profile_cls = PROFILE_CLASSES.get(profile_name)
            if profile_cls is None:
                continue
            instance = profile_cls()

            # Extract profile attributes from the instance
            # Note: dataclass_fields() only works if fields have type annotations.
            # These profiles have plain class attributes without annotations, so use vars() instead.
            profile_dict = {}

            try:
                # dataclass_fields() only returns annotated fields
                fields_list = dataclass_fields(instance)
                # logger.debug(f"  [DEBUG] {profile_name} has {len(fields_list)} annotated dataclass fields")
                for field in fields_list:
                    profile_dict[field.name] = getattr(instance, field.name, None)
            except TypeError:
                pass

            # Also get non-annotated class attributes (the actual values we need!)
            # Use vars() to get all instance attributes
            instance_vars = vars(instance) if hasattr(instance, '__dict__') else {}

            # And get class attributes from the class itself
            class_vars = {
                name: getattr(profile_cls, name)
                for name in dir(profile_cls)
                if not name.startswith('_')
            }

            # Combine: class attributes first (defaults), then instance attributes (overrides)
            profile_dict.update(class_vars)
            profile_dict.update(instance_vars)

            # Filter out methods and other non-data attributes
            profile_dict = {
                k: v for k, v in profile_dict.items()
                if not callable(v) and not k.startswith('_')
            }

            _profile_defaults[profile_name] = profile_dict
            # logger.debug(f"[ProfileStates.prepare] {profile_name}: found {len(_profile_defaults[profile_name])} attributes")

        # collect UI-backed profile field names
        field_names = []
        tuple_of_sections = cls.PROFILE_FIELD_GROUPS
        for section in tuple_of_sections:
            list_of_dict = section[1]
            for dict_of_fields in list_of_dict:
                # Extend the list with the keys from the dictionary
                field_names.append(dict_of_fields['name'])
        # Store the list of field names in the dictionary
        cls.PROFILE_FIELD_NAMES_UI = field_names

        # collect all known profile attribute names from discovered defaults
        all_field_names = sorted({
            field_name
            for profile_values in _profile_defaults.values()
            for field_name in profile_values.keys()
        })
        cls.PROFILE_FIELD_NAMES_ALL = all_field_names

        cls.PROFILE_DEFAULT_VALUES = _profile_defaults
        cls.PROFILE_MODIFIED_VALUES = {k: dict(v) for k, v in cls.PROFILE_DEFAULT_VALUES.items()}
        cls.PROFILES = list(cls.PROFILE_DEFAULT_VALUES.keys()) or []

    @classmethod
    def get_dependencies(cls, field_groups) -> Dict[str, List[str]]:
        """
        Returns a dictionary mapping controlling fields to their dependent fields.
        
        Args:
            field_groups: ASectionDefs (list of tuples with field definitions)
        
        Returns:
            Dict[str, List[str]] where key is controlling field, value is list of dependent fields
        """
        def collect_dependency_fields(dependency_spec: Dict[str, Any]) -> Set[str]:
            fields: Set[str] = set()

            if not isinstance(dependency_spec, dict):
                return fields

            _field_name = dependency_spec.get('field')
            if _field_name:
                fields.add(_field_name)

            for key in ('any', 'all'):
                items = dependency_spec.get(key, [])
                if items:
                    for item in items:
                        fields.update(collect_dependency_fields(item))

            return fields

        dependencies: Dict[str, List[str]] = {}

        for section in field_groups:
            field_list = section[1]
            for field_dict in field_list:
                dep_spec = field_dict.get('depends_on')
                if not dep_spec:
                    continue

                dependent_name = field_dict.get('name')
                if not dependent_name:
                    continue

                controller_fields = collect_dependency_fields(dep_spec)
                for field_name in controller_fields:
                    if field_name in ['profile', 'stitch_mode']:
                        continue
                    if field_name not in dependencies:
                        dependencies[field_name] = []
                    if dependent_name not in dependencies[field_name]:
                        dependencies[field_name].append(dependent_name)
        # print(dependencies)
        return dependencies


if not ProfileStates.initialized:
    ProfileStates.prepare()