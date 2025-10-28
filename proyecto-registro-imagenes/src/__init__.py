"""
Proyecto: Registro de Imágenes y Medición del Mundo Real
Universidad Nacional de Colombia - Visión por Computador

Módulos principales:
- feature_detection: Detección de características (SIFT, ORB, AKAZE)
- matching: Emparejamiento de descriptores
- registration: Registro y fusión de imágenes
- measurement: Herramientas de medición métrica
- utils: Funciones auxiliares
"""

__version__ = "1.0.0"
__author__ = "David A. Londoño"
__email__ = "Universidad Nacional de Colombia"

from .feature_detection import (
    FeatureDetector,
    detect_sift_features,
    detect_orb_features,
    detect_akaze_features,
    compare_detectors,
    visualize_keypoints
)

from .matching import (
    match_features,
    match_flann,
    match_bruteforce,
    apply_ratio_test,
    visualize_matches,
    get_matched_points
)

from .registration import (
    estimate_homography,
    warp_image,
    register_images,
    blend_images,
    stitch_multiple_images
)

from .measurement import MeasurementTool

from .utils import (
    generate_synthetic_image,
    compute_registration_metrics,
    visualize_registration,
    create_checkerboard_comparison,
    save_results,
    load_images_from_directory
)

__all__ = [
    # Feature Detection
    'FeatureDetector',
    'detect_sift_features',
    'detect_orb_features',
    'detect_akaze_features',
    'compare_detectors',
    'visualize_keypoints',
    
    # Matching
    'match_features',
    'match_flann',
    'match_bruteforce',
    'apply_ratio_test',
    'visualize_matches',
    'get_matched_points',
    
    # Registration
    'estimate_homography',
    'warp_image',
    'register_images',
    'blend_images',
    'stitch_multiple_images',
    
    # Measurement
    'MeasurementTool',
    
    # Utils
    'generate_synthetic_image',
    'compute_registration_metrics',
    'visualize_registration',
    'create_checkerboard_comparison',
    'save_results',
    'load_images_from_directory',
]
