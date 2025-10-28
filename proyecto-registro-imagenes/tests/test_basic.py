"""
test_basic.py

Pruebas unitarias básicas para el proyecto de registro de imágenes.

Universidad Nacional de Colombia
Visión por Computador - Trabajo 2: Registro de Imágenes
Autor: David A. Londoño
Fecha: Octubre 2025
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
import cv2
from pathlib import Path


class TestFeatureDetection:
    """Pruebas para el módulo de detección de características"""
    
    def test_import_feature_detection(self):
        """Verifica que el módulo se puede importar"""
        from feature_detection import detect_sift_features
        assert callable(detect_sift_features)
    
    def test_sift_detection(self):
        """Prueba detección SIFT en imagen sintética"""
        from feature_detection import detect_sift_features
        
        # Crear imagen de prueba
        img = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
        cv2.rectangle(img, (100, 100), (300, 300), (255, 255, 255), -1)
        
        # Detectar características
        kp, desc = detect_sift_features(img)
        
        assert len(kp) > 0, "Deberían detectarse keypoints"
        assert desc is not None, "Descriptores no deberían ser None"
        assert desc.shape[0] == len(kp), "Número de descriptores debe igualar keypoints"
    
    def test_orb_detection(self):
        """Prueba detección ORB"""
        from feature_detection import detect_orb_features
        
        img = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
        kp, desc = detect_orb_features(img)
        
        assert len(kp) > 0
        assert desc is not None


class TestMatching:
    """Pruebas para el módulo de emparejamiento"""
    
    def test_import_matching(self):
        """Verifica importación del módulo"""
        from matching import match_features
        assert callable(match_features)
    
    def test_flann_matching(self):
        """Prueba emparejamiento FLANN"""
        from feature_detection import detect_sift_features
        from matching import match_features
        
        # Crear dos imágenes similares
        img1 = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
        img2 = img1.copy()
        
        # Detectar características
        kp1, desc1 = detect_sift_features(img1)
        kp2, desc2 = detect_sift_features(img2)
        
        # Emparejar
        matches = match_features(desc1, desc2, method='flann')
        
        # Con imágenes idénticas, debería haber muchos matches
        assert len(matches) > 0, "Deberían encontrarse matches"


class TestRegistration:
    """Pruebas para el módulo de registro"""
    
    def test_import_registration(self):
        """Verifica importación"""
        from registration import estimate_homography
        assert callable(estimate_homography)
    
    def test_synthetic_registration(self):
        """Prueba registro con imagen sintética conocida"""
        from feature_detection import detect_sift_features
        from matching import match_features
        from registration import estimate_homography
        from utils import generate_synthetic_image
        
        # Crear imagen base
        img_base = np.random.randint(50, 200, (400, 400, 3), dtype=np.uint8)
        cv2.rectangle(img_base, (100, 100), (300, 300), (255, 255, 255), -1)
        cv2.circle(img_base, (200, 200), 50, (0, 0, 255), -1)
        
        # Generar transformada
        img_transformed, H_true = generate_synthetic_image(
            img_base,
            rotation=15,
            translation=(20, 20),
            scale=1.0
        )
        
        # Detectar y emparejar
        kp1, desc1 = detect_sift_features(img_base)
        kp2, desc2 = detect_sift_features(img_transformed)
        matches = match_features(desc1, desc2, method='flann')
        
        # Estimar homografía
        H_estimated, mask = estimate_homography(kp1, kp2, matches)
        
        assert H_estimated is not None, "Debería estimarse homografía"
        assert mask is not None, "Debería haber máscara de inliers"


class TestMeasurement:
    """Pruebas para el módulo de medición"""
    
    def test_import_measurement(self):
        """Verifica importación"""
        from measurement import MeasurementTool
        assert MeasurementTool is not None
    
    def test_measurement_tool_creation(self):
        """Prueba creación de herramienta de medición"""
        from measurement import MeasurementTool
        
        img = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
        tool = MeasurementTool(img)
        
        assert tool.image is not None
        assert tool.scale_pixels_per_cm is None  # No calibrada aún
    
    def test_calibration(self):
        """Prueba calibración de escala"""
        from measurement import MeasurementTool
        
        img = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
        tool = MeasurementTool(img)
        
        # Calibración manual (sin interfaz)
        tool.points = [(100, 100), (200, 100)]  # 100 píxeles
        scale = tool.calibrate_scale(
            reference_object="Test Object",
            real_dimension_cm=50.0,
            interactive=False
        )
        
        assert scale is not None
        assert tool.scale_pixels_per_cm == 2.0  # 100px / 50cm = 2 px/cm


class TestUtils:
    """Pruebas para utilidades"""
    
    def test_generate_synthetic_image(self):
        """Prueba generación de imagen sintética"""
        from utils import generate_synthetic_image
        
        img = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
        img_transformed, H = generate_synthetic_image(
            img,
            rotation=30,
            translation=(50, 50),
            scale=1.2
        )
        
        assert img_transformed.shape == img.shape
        assert H.shape == (3, 3)
    
    def test_compute_registration_metrics(self):
        """Prueba cálculo de métricas"""
        from utils import compute_registration_metrics
        
        # Matrices de prueba
        H_true = np.eye(3)
        H_estimated = np.eye(3) + np.random.randn(3, 3) * 0.01
        
        metrics = compute_registration_metrics(H_true, H_estimated)
        
        assert 'rmse' in metrics
        assert 'mean_error' in metrics
        assert 'angular_error' in metrics


class TestIntegration:
    """Pruebas de integración"""
    
    def test_full_pipeline_synthetic(self):
        """Prueba pipeline completo con imagen sintética"""
        from feature_detection import detect_sift_features
        from matching import match_features
        from registration import estimate_homography
        from utils import generate_synthetic_image, compute_registration_metrics
        
        # Crear imagen base
        img_base = np.random.randint(50, 200, (400, 400, 3), dtype=np.uint8)
        for i in range(5):
            x, y = np.random.randint(50, 350, 2)
            cv2.circle(img_base, (x, y), 20, (255, 255, 255), -1)
        
        # Generar transformada
        img_transformed, H_true = generate_synthetic_image(
            img_base, rotation=20, translation=(30, 30), scale=1.0
        )
        
        # Pipeline completo
        kp1, desc1 = detect_sift_features(img_base)
        kp2, desc2 = detect_sift_features(img_transformed)
        matches = match_features(desc1, desc2, method='flann')
        H_estimated, _ = estimate_homography(kp1, kp2, matches)
        
        if H_estimated is not None:
            metrics = compute_registration_metrics(H_true, H_estimated)
            
            # Verificar que el error es razonable
            assert metrics['rmse'] < 50, "RMSE debería ser menor a 50 píxeles"
            print(f"✓ Pipeline completo: RMSE = {metrics['rmse']:.2f} px")


def test_project_structure():
    """Verifica que la estructura del proyecto existe"""
    base_dir = Path(__file__).parent.parent
    
    required_dirs = [
        'src',
        'notebooks',
        'data/original',
        'data/synthetic',
        'results/figures',
        'results/measurements',
        'tests'
    ]
    
    for dir_path in required_dirs:
        full_path = base_dir / dir_path
        assert full_path.exists(), f"Directorio {dir_path} no existe"
    
    print("✓ Estructura del proyecto verificada")


def test_required_files():
    """Verifica que los archivos principales existen"""
    base_dir = Path(__file__).parent.parent
    
    required_files = [
        'README.md',
        'requirements.txt',
        'src/feature_detection.py',
        'src/matching.py',
        'src/registration.py',
        'src/measurement.py',
        'src/utils.py'
    ]
    
    for file_path in required_files:
        full_path = base_dir / file_path
        assert full_path.exists(), f"Archivo {file_path} no existe"
    
    print("✓ Archivos principales verificados")


if __name__ == "__main__":
    # Ejecutar pruebas básicas sin pytest
    print("\n" + "="*60)
    print("EJECUTANDO PRUEBAS BÁSICAS")
    print("="*60)
    
    try:
        # Pruebas de estructura
        test_project_structure()
        test_required_files()
        
        # Pruebas de módulos
        print("\nProbando módulos...")
        
        test_obj = TestFeatureDetection()
        test_obj.test_import_feature_detection()
        print("✓ Feature Detection importado")
        
        test_obj2 = TestMatching()
        test_obj2.test_import_matching()
        print("✓ Matching importado")
        
        test_obj3 = TestRegistration()
        test_obj3.test_import_registration()
        print("✓ Registration importado")
        
        test_obj4 = TestMeasurement()
        test_obj4.test_import_measurement()
        print("✓ Measurement importado")
        
        print("\n" + "="*60)
        print("✓ TODAS LAS PRUEBAS BÁSICAS PASARON")
        print("="*60)
        print("\nPara ejecutar todas las pruebas con pytest:")
        print("  pytest tests/test_basic.py -v")
        
    except Exception as e:
        print(f"\n✗ Error en pruebas: {e}")
        import traceback
        traceback.print_exc()
