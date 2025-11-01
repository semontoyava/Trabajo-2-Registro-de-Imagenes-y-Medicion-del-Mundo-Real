import os
import sys
import urllib.request
import tarfile
from pathlib import Path
import cv2
import numpy as np
import json
from typing import List, Tuple, Dict

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from feature_detection import detect_sift_features
from matching import match_features
from registration import estimate_homography, warp_image
from utils import generate_synthetic_image, compute_registration_metrics, visualize_registration, save_results

# Path para guardar resultados
RESULTS_DIR = Path("proyecto-registro-imagenes/results/synthetic_validation")

def main():
    print("\n" + "="*80)
    print("VALIDACIÓN PIPELINE CON IMÁGENES SINTÉTICAS")
    print("="*80)

    # Crear imagen base artificial
    base_img = np.zeros((400, 400), dtype=np.uint8)
    cv2.circle(base_img, (200, 200), 80, 255, -1)
    cv2.line(base_img, (50, 50), (350, 350), 255, 3)
    cv2.rectangle(base_img, (100, 300), (300, 350), 255, -1)
    cv2.putText(base_img, "TEST", (160, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.3, 255, 2)

    # Generar imagen transformada + homografía real
    img_transformed, H_true = generate_synthetic_image(
        base_img,
        rotation=40,
        translation=(40, -20),
        scale=1.1
    )

    # Detectar características
    kp1, des1 = detect_sift_features(base_img)
    kp2, des2 = detect_sift_features(img_transformed)

    # Emparejar keypoints
    matches = match_features(des1, des2, method="flann")

    # Registrar coincidencias
    H_est, mask = estimate_homography(kp1, kp2, matches)
    H_est = H_est.astype(np.float32)
    img_registered = warp_image(img_transformed, H_est, base_img.shape)

    # Comparar con ground truth
    metrics = compute_registration_metrics(H_true, H_est, image_shape=base_img.shape)

    print("\nMÉTRICAS SINTÉTICAS:")
    print(json.dumps(metrics, indent=2))

    # Guardado
    synthetic_dir = RESULTS_DIR 
    synthetic_dir.mkdir(parents=True, exist_ok=True)

    visualize_registration(
        base_img,
        img_transformed,
        img_registered,
        save_path=str(synthetic_dir / "synthetic.png")
    )

    save_results(
        {
            "original": base_img,
            "transformed": img_transformed,
            "registered": img_registered,
            "H_true": H_true.tolist(),
            "H_est": H_est.tolist(),
            **metrics
        },
        output_dir=str(synthetic_dir),
        #prefix="synthetic"
    )

    return 0

if __name__ == "__main__":
    sys.exit(main())