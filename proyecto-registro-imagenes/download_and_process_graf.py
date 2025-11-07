"""
download_and_process_graf.py

Script para descargar y procesar el dataset Graf de Oxford VGG.
Este dataset contiene imágenes con transformaciones de viewpoint conocidas,
ideal para validar algoritmos de registro.

Universidad Nacional de Colombia
Visión por Computador - Trabajo 2: Registro de Imágenes
Autor: David A. Londoño
Fecha: Octubre 2025

Dataset: https://www.robots.ox.ac.uk/~vgg/research/affine/
Reference: Mikolajczyk, K. & Schmid, C. (2005). "A Performance Evaluation of Local Descriptors"
"""

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

from feature_detection import detect_sift_features, detect_orb_features, compare_detectors
from matching import match_features, compute_match_statistics
from registration import estimate_homography, warp_image
from utils import compute_registration_metrics, visualize_registration, save_results

# Configuración
DATASET_URL = "https://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/graf.tar.gz"
DATA_DIR = Path("data/graf_dataset")
RESULTS_DIR = Path("results/figures")


def download_dataset(url: str, output_dir: Path) -> Path:
    """
    Descarga el archivo tar.gz del dataset Graf.
    
    Args:
        url (str): URL del dataset
        output_dir (Path): Directorio donde guardar el archivo
        
    Returns:
        Path: Ruta del archivo descargado
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = url.split('/')[-1]
    output_path = output_dir / filename
    
    if output_path.exists():
        print(f"✓ Dataset ya existe: {output_path}")
        return output_path
    
    print(f"\n{'='*80}")
    print(f"DESCARGANDO DATASET GRAF")
    print(f"{'='*80}")
    print(f"URL: {url}")
    print(f"Destino: {output_path}")
    
    try:
        # Descargar con barra de progreso
        def reporthook(blocknum, blocksize, totalsize):
            readsofar = blocknum * blocksize
            if totalsize > 0:
                percent = readsofar * 100 / totalsize
                s = f"\r{percent:5.1f}% {readsofar:,} / {totalsize:,} bytes"
                sys.stderr.write(s)
                if readsofar >= totalsize:
                    sys.stderr.write("\n")
            else:
                sys.stderr.write(f"\rDescargado {readsofar:,} bytes\n")
        
        urllib.request.urlretrieve(url, output_path, reporthook)
        print(f"✓ Descarga completada: {output_path.stat().st_size:,} bytes")
        
    except Exception as e:
        print(f"✗ Error en la descarga: {e}")
        if output_path.exists():
            output_path.unlink()
        raise
    
    return output_path


def extract_dataset(tar_path: Path, extract_dir: Path) -> Path:
    """
    Extrae el archivo tar.gz descargado.
    
    Args:
        tar_path (Path): Ruta del archivo tar.gz
        extract_dir (Path): Directorio donde extraer
        
    Returns:
        Path: Directorio con las imágenes extraídas
    """
    print(f"\n{'='*80}")
    print(f"EXTRAYENDO DATASET")
    print(f"{'='*80}")
    print(f"Archivo: {tar_path}")
    print(f"Destino: {extract_dir}")
    
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        with tarfile.open(tar_path, 'r:gz') as tar:
            # Listar contenido
            members = tar.getmembers()
            print(f"\nArchivos en el dataset: {len(members)}")
            
            # Extraer
            tar.extractall(path=extract_dir)
            print(f"✓ Extracción completada")
            
            # Mostrar archivos extraídos
            extracted_files = list(extract_dir.rglob('*'))
            image_files = [f for f in extracted_files if f.suffix.lower() in ['.ppm', '.pgm', '.png', '.jpg']]
            print(f"\nImágenes encontradas: {len(image_files)}")
            for img_file in sorted(image_files)[:10]:  # Mostrar primeras 10
                print(f"  - {img_file.name}")
            
    except Exception as e:
        print(f"✗ Error en la extracción: {e}")
        raise
    
    return extract_dir


def load_graf_images(dataset_dir: Path, n_images: int = 3) -> List[Tuple[str, np.ndarray]]:
    """
    Carga las primeras n imágenes del dataset Graf.
    
    El dataset Graf contiene imágenes con diferentes viewpoints (img1.ppm, img2.ppm, etc.)
    
    Args:
        dataset_dir (Path): Directorio con las imágenes
        n_images (int): Número de imágenes a cargar (por defecto 3)
        
    Returns:
        List[Tuple[str, np.ndarray]]: Lista de (nombre, imagen)
    """
    print(f"\n{'='*80}")
    print(f"CARGANDO IMÁGENES DEL DATASET")
    print(f"{'='*80}")
    
    images = []
    
    # Buscar imágenes en formato img1.ppm, img2.ppm, etc.
    for i in range(1, n_images + 1):
        # Probar diferentes extensiones
        for ext in ['.ppm', '.pgm', '.png', '.jpg']:
            img_path = dataset_dir / f"img{i}{ext}"
            
            # Buscar también en subdirectorios
            if not img_path.exists():
                found_paths = list(dataset_dir.rglob(f"img{i}{ext}"))
                if found_paths:
                    img_path = found_paths[0]
            
            if img_path.exists():
                img = cv2.imread(str(img_path))
                if img is not None:
                    images.append((img_path.name, img))
                    print(f"✓ Cargada: {img_path.name} - {img.shape[1]}x{img.shape[0]} píxeles")
                    break
        else:
            print(f"⚠ No se encontró img{i} en el dataset")
    
    if len(images) == 0:
        print("\n⚠ No se encontraron imágenes. Listando contenido del directorio:")
        all_files = list(dataset_dir.rglob('*'))
        for f in sorted(all_files)[:20]:
            print(f"  {f}")
    
    print(f"\n✓ Total imágenes cargadas: {len(images)}")
    return images


def process_image_pair(img1: np.ndarray, img2: np.ndarray,
                       name1: str, name2: str,
                       detector_type: str = 'sift') -> Dict:
    """
    Procesa un par de imágenes: detecta características, empareja y registra.
    
    Args:
        img1 (np.ndarray): Primera imagen (referencia)
        img2 (np.ndarray): Segunda imagen (a registrar)
        name1 (str): Nombre de la primera imagen
        name2 (str): Nombre de la segunda imagen
        detector_type (str): Tipo de detector ('sift', 'orb', 'akaze')
        
    Returns:
        Dict: Diccionario con resultados del procesamiento
    """
    print(f"\n{'='*80}")
    print(f"PROCESANDO PAR: {name1} <-> {name2}")
    print(f"{'='*80}")
    
    results = {
        'pair': f"{name1} - {name2}",
        'detector': detector_type,
        'img1_name': name1,
        'img2_name': name2
    }
    
    # 1. Detectar características
    print(f"\n1. Detectando características con {detector_type.upper()}...")
    
    if detector_type == 'sift':
        kp1, desc1 = detect_sift_features(img1)
        kp2, desc2 = detect_sift_features(img2)
    elif detector_type == 'orb':
        kp1, desc1 = detect_orb_features(img1)
        kp2, desc2 = detect_orb_features(img2)
    else:
        from feature_detection import detect_akaze_features
        kp1, desc1 = detect_akaze_features(img1)
        kp2, desc2 = detect_akaze_features(img2)
    
    results['keypoints_img1'] = len(kp1)
    results['keypoints_img2'] = len(kp2)
    print(f"   Imagen 1: {len(kp1)} keypoints")
    print(f"   Imagen 2: {len(kp2)} keypoints")
    
    # 2. Emparejar características
    print(f"\n2. Emparejando características...")
    method = 'flann' if detector_type == 'sift' else 'bf_hamming'
    matches = match_features(desc1, desc2, method=method, ratio_test=0.75)
    
    results['matches'] = len(matches)
    print(f"   Matches encontrados: {len(matches)}")
    
    if len(matches) < 4:
        print("   ⚠ Insuficientes matches para registro")
        results['success'] = False
        return results
    
    # Estadísticas de matches
    match_stats = compute_match_statistics(matches)
    results['match_mean_distance'] = float(match_stats['mean_distance'])
    results['match_std_distance'] = float(match_stats['std_distance'])
    
    # 3. Estimar homografía
    print(f"\n3. Estimando homografía con RANSAC...")
    H, mask = estimate_homography(kp1, kp2, matches, ransac_threshold=5.0)
    
    if H is None:
        print("   ✗ No se pudo estimar homografía")
        results['success'] = False
        return results
    
    inliers = int(np.sum(mask)) if mask is not None else 0
    inlier_ratio = inliers / len(matches) * 100
    results['inliers'] = inliers
    results['inlier_ratio'] = float(inlier_ratio)
    
    print(f"   Inliers: {inliers}/{len(matches)} ({inlier_ratio:.1f}%)")
    
    # 4. Aplicar registro
    print(f"\n4. Aplicando transformación...")
    img2_registered = warp_image(img2, H, img1.shape[:2])
    
    results['registered_image'] = img2_registered
    results['homography'] = H.tolist()
    results['success'] = True
    
    print(f"   ✓ Imagen registrada: {img2_registered.shape}")
    
    # 5. Calcular métricas (si hay ground truth disponible)
    # En el dataset Graf, las transformaciones son conocidas pero no tenemos
    # las matrices de homografía exactas, así que medimos calidad visual
    
    # Calcular similitud estructural en región de overlap
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2_reg = cv2.cvtColor(img2_registered, cv2.COLOR_BGR2GRAY)
    
    # Crear máscara de región válida
    mask_valid = (gray2_reg > 0).astype(np.uint8)
    
    # Calcular diferencia promedio en región válida
    diff = cv2.absdiff(gray1, gray2_reg)
    mean_diff = np.mean(diff[mask_valid > 0]) if np.any(mask_valid) else 0
    
    results['mean_pixel_difference'] = float(mean_diff)
    print(f"   Diferencia promedio de píxeles: {mean_diff:.2f}")
    
    return results


def save_visualization(img1: np.ndarray, img2: np.ndarray, 
                       img_registered: np.ndarray,
                       name1: str, name2: str,
                       output_dir: Path):
    """
    Guarda visualización comparativa del registro.
    
    Args:
        img1 (np.ndarray): Imagen referencia
        img2 (np.ndarray): Imagen original
        img_registered (np.ndarray): Imagen registrada
        name1 (str): Nombre imagen 1
        name2 (str): Nombre imagen 2
        output_dir (Path): Directorio de salida
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Guardar visualización comparativa
    pair_name = f"{name1.split('.')[0]}_{name2.split('.')[0]}"
    vis_path = output_dir / f"registration_{pair_name}.png"
    
    visualize_registration(img1, img2, img_registered, save_path=str(vis_path))
    
    # Guardar imagen registrada individual
    reg_path = output_dir / f"registered_{pair_name}.jpg"
    cv2.imwrite(str(reg_path), img_registered)
    
    print(f"   ✓ Visualizaciones guardadas:")
    print(f"      - {vis_path.name}")
    print(f"      - {reg_path.name}")


def main():
    """
    Función principal: descarga, extrae y procesa el dataset Graf.
    """
    print("\n" + "="*80)
    print("DATASET GRAF - VALIDACIÓN DE REGISTRO DE IMÁGENES")
    print("Universidad Nacional de Colombia - Visión por Computador")
    print("="*80)
    
    # 1. Descargar dataset
    try:
        tar_path = download_dataset(DATASET_URL, DATA_DIR)
    except Exception as e:
        print(f"\n✗ Error descargando dataset: {e}")
        print("\nPor favor, descargue manualmente desde:")
        print(DATASET_URL)
        return 1
    
    # 2. Extraer dataset
    try:
        extract_dir = extract_dataset(tar_path, DATA_DIR)
    except Exception as e:
        print(f"\n✗ Error extrayendo dataset: {e}")
        return 1
    
    # 3. Cargar primeras 3 imágenes
    images = load_graf_images(DATA_DIR, n_images=3)
    
    if len(images) < 2:
        print("\n✗ Error: Se necesitan al menos 2 imágenes para el registro")
        return 1
    
    # 4. Procesar pares de imágenes
    all_results = []
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Usar la primera imagen como referencia
    name_ref, img_ref = images[0]
    print(f"\n{'='*80}")
    print(f"IMAGEN DE REFERENCIA: {name_ref}")
    print(f"{'='*80}")
    
    # Registrar las demás imágenes contra la referencia
    for i in range(1, len(images)):
        name_i, img_i = images[i]
        
        # Procesar con SIFT
        result = process_image_pair(img_ref, img_i, name_ref, name_i, detector_type='sift')
        
        if result['success']:
            # Guardar visualizaciones
            save_visualization(
                img_ref, img_i, result['registered_image'],
                name_ref, name_i,
                RESULTS_DIR
            )
            
            # Remover imagen del resultado para JSON (no serializable)
            result_copy = result.copy()
            result_copy.pop('registered_image', None)
            all_results.append(result_copy)
    
    # 5. Guardar métricas en JSON
    metrics_path = RESULTS_DIR / 'graf_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("RESUMEN DE RESULTADOS")
    print(f"{'='*80}")
    
    for result in all_results:
        if result['success']:
            print(f"\nPar: {result['pair']}")
            print(f"  Detector: {result['detector'].upper()}")
            print(f"  Keypoints: {result['keypoints_img1']} + {result['keypoints_img2']}")
            print(f"  Matches: {result['matches']}")
            print(f"  Inliers: {result['inliers']} ({result['inlier_ratio']:.1f}%)")
            print(f"  Diferencia píxel promedio: {result['mean_pixel_difference']:.2f}")
    
    print(f"\n{'='*80}")
    print("PROCESO COMPLETADO")
    print(f"{'='*80}")
    print(f"\n✓ Métricas guardadas en: {metrics_path}")
    print(f"✓ Visualizaciones en: {RESULTS_DIR}")
    print(f"✓ Dataset en: {DATA_DIR}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
