"""
process_comedor.py

Script para procesar las imágenes del comedor y crear una vista panorámica.
Parte 2 del Trabajo: Registro de Imágenes Reales

Universidad Nacional de Colombia
Visión por Computador - Trabajo 2: Registro de Imágenes
Autor: David A. Londoño
Fecha: Octubre 2025

Objetos de referencia para calibración:
- Cuadro Virgen de Guadalupe: 117 cm de altura
- Mesa: 161.1 cm de ancho
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from feature_detection import detect_sift_features, detect_orb_features, compare_detectors
from matching import match_features, compute_match_statistics
from registration import estimate_homography, warp_image, register_images
from utils import visualize_registration, save_results

# Configuración
IMAGES_DIR = Path("data/original")
RESULTS_DIR = Path("results/figures")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Objetos de referencia para calibración (Parte 3)
REFERENCE_OBJECTS = {
    "cuadro_virgen": {"dimension": 117.0, "unit": "cm", "type": "altura"},
    "mesa": {"dimension": 161.1, "unit": "cm", "type": "ancho"}
}


def load_comedor_images() -> List[Tuple[np.ndarray, str]]:
    """
    Carga las imágenes del comedor desde el directorio.
    
    Returns:
        List[Tuple[np.ndarray, str]]: Lista de tuplas (imagen, nombre_archivo)
    """
    print("\n" + "="*80)
    print("CARGANDO IMÁGENES DEL COMEDOR")
    print("="*80 + "\n")
    
    images = []
    image_files = sorted([f for f in IMAGES_DIR.glob("*.jpg")] + 
                        [f for f in IMAGES_DIR.glob("*.png")])
    
    if not image_files:
        print(f"[ERROR] No se encontraron imágenes en {IMAGES_DIR}")
        print("Por favor, coloque las imágenes del comedor en esta carpeta.")
        return images
    
    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is not None:
            images.append((img, img_path.name))
            print(f"✓ Cargada: {img_path.name} - {img.shape}")
        else:
            print(f"✗ Error al cargar: {img_path.name}")
    
    print(f"\nTotal de imágenes cargadas: {len(images)}")
    return images


def detect_features_both_methods(images: List[Tuple[np.ndarray, str]]) -> Dict:
    """
    Detecta características usando SIFT y ORB en todas las imágenes.
    
    Args:
        images: Lista de tuplas (imagen, nombre)
        
    Returns:
        Dict: Diccionario con características detectadas por cada método
    """
    print("\n" + "="*80)
    print("DETECCIÓN DE CARACTERÍSTICAS")
    print("="*80 + "\n")
    
    results = {
        'sift': [],
        'orb': []
    }
    
    for i, (img, name) in enumerate(images, 1):
        print(f"\nImagen {i}: {name}")
        
        # SIFT
        print("  Detectando con SIFT...")
        kp_sift, desc_sift = detect_sift_features(img)
        results['sift'].append((kp_sift, desc_sift, name))
        print(f"    ✓ SIFT: {len(kp_sift)} keypoints")
        
        # ORB
        print("  Detectando con ORB...")
        kp_orb, desc_orb = detect_orb_features(img)
        results['orb'].append((kp_orb, desc_orb, name))
        print(f"    ✓ ORB: {len(kp_orb)} keypoints")
    
    return results


def register_image_pair(img1: np.ndarray, img2: np.ndarray,
                        kp1: List, desc1: np.ndarray,
                        kp2: List, desc2: np.ndarray,
                        name1: str, name2: str,
                        method: str = 'sift') -> Tuple[np.ndarray, Dict]:
    """
    Registra un par de imágenes.
    
    Args:
        img1, img2: Imágenes a registrar
        kp1, kp2: Keypoints
        desc1, desc2: Descriptores
        name1, name2: Nombres de las imágenes
        method: Método usado ('sift' o 'orb')
        
    Returns:
        Tuple[np.ndarray, Dict]: Imagen registrada y estadísticas
    """
    print(f"\n--- Registrando: {name1} + {name2} ({method.upper()}) ---")
    
    # Emparejar características
    print("1. Emparejando características...")
    matches = match_features(desc1, desc2, 
                            method='flann' if method == 'sift' else 'bf_hamming',
                            ratio_test=0.75)
    print(f"   Matches encontrados: {len(matches)}")
    
    if len(matches) < 10:
        print("   [ADVERTENCIA] Muy pocos matches encontrados")
        return None, {}
    
    # Estimar homografía
    print("2. Estimando homografía con RANSAC...")
    H, mask = estimate_homography(kp1, kp2, matches)
    
    if H is None:
        print("   [ERROR] No se pudo estimar la homografía")
        return None, {}
    
    inliers = np.sum(mask)
    inlier_ratio = inliers / len(matches) * 100
    print(f"   Inliers: {inliers}/{len(matches)} ({inlier_ratio:.1f}%)")
    
    # Aplicar transformación
    print("3. Aplicando transformación...")
    h, w = img1.shape[:2]
    img_registered = warp_image(img2, H, (h, w))
    
    # Estadísticas
    stats = {
        'matches': len(matches),
        'inliers': int(inliers),
        'inlier_ratio': float(inlier_ratio),
        'homography': H.tolist()
    }
    
    # Visualizar y guardar
    output_path = RESULTS_DIR / f"registration_{method}_{name1}_{name2}.png"
    visualize_registration(img1, img2, img_registered, 
                          save_path=str(output_path))
    print(f"   ✓ Visualización guardada: {output_path}")
    
    return img_registered, stats


def create_panorama(images: List[Tuple[np.ndarray, str]], 
                    features: Dict,
                    method: str = 'sift') -> Tuple[np.ndarray, Dict]:
    """
    Crea un panorama fusionando todas las imágenes.
    
    Args:
        images: Lista de tuplas (imagen, nombre)
        features: Características detectadas
        method: Método a usar ('sift' o 'orb')
        
    Returns:
        Tuple[np.ndarray, Dict]: Panorama final y estadísticas
    """
    print("\n" + "="*80)
    print(f"CREANDO PANORAMA CON {method.upper()}")
    print("="*80 + "\n")
    
    if len(images) < 2:
        print("[ERROR] Se necesitan al menos 2 imágenes")
        return None, {}
    
    # Usar imagen central como referencia
    ref_idx = len(images) // 2
    panorama = images[ref_idx][0].copy()
    stats_all = []
    
    # Registrar imágenes adyacentes
    feature_list = features[method]
    
    # Registrar hacia la izquierda
    for i in range(ref_idx - 1, -1, -1):
        kp1, desc1, name1 = feature_list[i]
        kp2, desc2, name2 = feature_list[i + 1]
        img1 = images[i][0]
        
        registered, stats = register_image_pair(
            panorama, img1, 
            feature_list[i + 1][0], feature_list[i + 1][1],
            kp1, desc1,
            name2, name1, method
        )
        
        if registered is not None:
            # Fusionar con blending simple
            panorama = cv2.addWeighted(panorama, 0.5, registered, 0.5, 0)
            stats_all.append(stats)
    
    # Registrar hacia la derecha
    for i in range(ref_idx + 1, len(images)):
        kp1, desc1, name1 = feature_list[i - 1]
        kp2, desc2, name2 = feature_list[i]
        img2 = images[i][0]
        
        registered, stats = register_image_pair(
            panorama, img2,
            kp1, desc1,
            kp2, desc2,
            name1, name2, method
        )
        
        if registered is not None:
            panorama = cv2.addWeighted(panorama, 0.5, registered, 0.5, 0)
            stats_all.append(stats)
    
    # Guardar panorama
    output_path = RESULTS_DIR / f"panorama_{method}.jpg"
    cv2.imwrite(str(output_path), panorama)
    print(f"\n✓ Panorama guardado: {output_path}")
    print(f"  Tamaño: {panorama.shape}")
    
    return panorama, {'registrations': stats_all}


def save_results_json(results: Dict, filename: str = "comedor_results.json"):
    """Guarda los resultados en formato JSON."""
    output_path = RESULTS_DIR / filename
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Resultados guardados: {output_path}")


def main():
    """Función principal."""
    print("\n" + "="*80)
    print("PROCESAMIENTO DE IMÁGENES DEL COMEDOR")
    print("Universidad Nacional de Colombia - Visión por Computador")
    print("Trabajo 2: Registro de Imágenes y Medición del Mundo Real")
    print("="*80)
    
    # 1. Cargar imágenes
    images = load_comedor_images()
    if len(images) < 2:
        print("\n[ERROR] Se necesitan al menos 2 imágenes para continuar.")
        return
    
    # 2. Detectar características con ambos métodos
    features = detect_features_both_methods(images)
    
    # 3. Crear panoramas con SIFT y ORB
    panorama_sift, stats_sift = create_panorama(images, features, 'sift')
    panorama_orb, stats_orb = create_panorama(images, features, 'orb')
    
    # 4. Guardar resultados
    results = {
        'num_images': len(images),
        'image_names': [name for _, name in images],
        'reference_objects': REFERENCE_OBJECTS,
        'sift_results': stats_sift,
        'orb_results': stats_orb,
        'notes': [
            'Panorama creado exitosamente',
            'Siguiente paso: Calibración y medición (Parte 3)',
            'Usar objetos de referencia para establecer escala métrica'
        ]
    }
    
    save_results_json(results)
    
    # Resumen final
    print("\n" + "="*80)
    print("PROCESAMIENTO COMPLETADO")
    print("="*80)
    print(f"\n✓ Imágenes procesadas: {len(images)}")
    print(f"✓ Panoramas generados: 2 (SIFT y ORB)")
    print(f"✓ Resultados en: {RESULTS_DIR}")
    print("\nPróximos pasos:")
    print("  1. Revisar los panoramas generados")
    print("  2. Seleccionar el mejor panorama (SIFT o ORB)")
    print("  3. Ejecutar la herramienta de calibración y medición (Parte 3)")
    print("\nObjetos de referencia para calibración:")
    for obj, info in REFERENCE_OBJECTS.items():
        print(f"  - {obj}: {info['dimension']} {info['unit']} ({info['type']})")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
