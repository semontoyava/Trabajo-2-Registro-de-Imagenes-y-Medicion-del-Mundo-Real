"""
registration.py

Módulo para registro de imágenes mediante estimación de homografía y warping.
Implementa RANSAC para estimación robusta y funciones de fusión de imágenes.

Universidad Nacional de Colombia
Visión por Computador - Trabajo 2: Registro de Imágenes
Autor: David A. Londoño
Fecha: Octubre 2025
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def estimate_homography(keypoints1: List[cv2.KeyPoint],
                       keypoints2: List[cv2.KeyPoint],
                       matches: List[cv2.DMatch],
                       ransac_threshold: float = 5.0,
                       confidence: float = 0.99,
                       max_iters: int = 2000) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Estima la matriz de homografía entre dos conjuntos de puntos usando RANSAC.
    
    Args:
        keypoints1 (List[cv2.KeyPoint]): Keypoints de la imagen origen
        keypoints2 (List[cv2.KeyPoint]): Keypoints de la imagen destino
        matches (List[cv2.DMatch]): Matches entre keypoints
        ransac_threshold (float): Umbral de error para considerar inliers (píxeles)
        confidence (float): Nivel de confianza deseado (0-1)
        max_iters (int): Número máximo de iteraciones RANSAC
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - Matriz de homografía 3x3 (o None si falla)
            - Máscara de inliers (o None si falla)
            
    References:
        Fischler & Bolles (1981). "Random Sample Consensus"
    """
    if len(matches) < 4:
        logger.warning("Se necesitan al menos 4 matches para estimar homografía")
        return None, None
    
    # Extraer puntos emparejados
    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # Estimar homografía con RANSAC
    H, mask = cv2.findHomography(
        pts1, pts2,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold,
        confidence=confidence,
        maxIters=max_iters
    )
    
    if H is not None:
        inliers = np.sum(mask)
        inlier_ratio = inliers / len(matches) * 100
        logger.info(f"Homografía estimada: {inliers}/{len(matches)} inliers ({inlier_ratio:.1f}%)")
    else:
        logger.warning("No se pudo estimar homografía")
    
    return H, mask


def warp_image(img: np.ndarray, H: np.ndarray, 
              output_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Aplica transformación de perspectiva (warp) a una imagen usando homografía.
    
    Args:
        img (np.ndarray): Imagen a transformar
        H (np.ndarray): Matriz de homografía 3x3
        output_shape (Tuple[int, int], optional): (height, width) de salida
            Si None, usa las dimensiones de la imagen de entrada
            
    Returns:
        np.ndarray: Imagen transformada
    """
    if output_shape is None:
        height, width = img.shape[:2]
    else:
        height, width = output_shape
    
    # Aplicar transformación de perspectiva
    warped = cv2.warpPerspective(img, H, (width, height))
    
    return warped


def compute_panorama_size(img1_shape: Tuple[int, int],
                         img2_shape: Tuple[int, int],
                         H: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Calcula el tamaño del panorama y la matriz de traslación necesaria.
    
    Args:
        img1_shape (Tuple[int, int]): (height, width) de la imagen 1
        img2_shape (Tuple[int, int]): (height, width) de la imagen 2
        H (np.ndarray): Homografía que transforma img2 al sistema de img1
        
    Returns:
        Tuple[np.ndarray, Tuple[int, int]]:
            - Matriz de traslación ajustada
            - Dimensiones del panorama (width, height)
    """
    h1, w1 = img1_shape[:2]
    h2, w2 = img2_shape[:2]
    
    # Esquinas de las imágenes
    corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    
    # Transformar esquinas de img2
    corners2_transformed = cv2.perspectiveTransform(corners2, H)
    
    # Combinar todas las esquinas
    all_corners = np.concatenate((corners1, corners2_transformed), axis=0)
    
    # Encontrar límites del panorama
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    
    # Matriz de traslación para ajustar coordenadas
    translation = np.array([
        [1, 0, -x_min],
        [0, 1, -y_min],
        [0, 0, 1]
    ])
    
    # Dimensiones del panorama
    panorama_width = x_max - x_min
    panorama_height = y_max - y_min
    
    logger.info(f"Dimensiones del panorama: {panorama_width}x{panorama_height}")
    
    return translation, (panorama_width, panorama_height)


def register_images(img1: np.ndarray, img2: np.ndarray,
                   keypoints1: List[cv2.KeyPoint],
                   keypoints2: List[cv2.KeyPoint],
                   matches: List[cv2.DMatch],
                   ransac_threshold: float = 5.0) -> np.ndarray:
    """
    Registra dos imágenes creando un panorama básico.
    
    Args:
        img1 (np.ndarray): Primera imagen (referencia)
        img2 (np.ndarray): Segunda imagen (a transformar)
        keypoints1 (List[cv2.KeyPoint]): Keypoints de img1
        keypoints2 (List[cv2.KeyPoint]): Keypoints de img2
        matches (List[cv2.DMatch]): Matches entre keypoints
        ransac_threshold (float): Umbral RANSAC
        
    Returns:
        np.ndarray: Imagen panorámica registrada
    """
    # Estimar homografía
    H, mask = estimate_homography(keypoints1, keypoints2, matches, ransac_threshold)
    
    if H is None:
        logger.error("No se pudo registrar las imágenes")
        return img1
    
    # Calcular tamaño del panorama
    translation, (pano_width, pano_height) = compute_panorama_size(
        img1.shape, img2.shape, H
    )
    
    # Ajustar homografía con traslación
    H_adjusted = translation @ H
    
    # Transformar img2
    img2_warped = cv2.warpPerspective(img2, H_adjusted, (pano_width, pano_height))
    
    # Transformar img1 (solo traslación)
    img1_warped = cv2.warpPerspective(img1, translation, (pano_width, pano_height))
    
    # Combinar imágenes (fusión simple)
    panorama = np.where(img1_warped > 0, img1_warped, img2_warped)
    
    return panorama


def blend_images(img1: np.ndarray, img2: np.ndarray, 
                panorama: np.ndarray,
                blend_method: str = 'linear') -> np.ndarray:
    """
    Fusiona dos imágenes en el panorama con blending suave.
    
    Args:
        img1 (np.ndarray): Primera imagen
        img2 (np.ndarray): Segunda imagen
        panorama (np.ndarray): Panorama sin blending
        blend_method (str): Método de blending ('linear', 'multiband')
        
    Returns:
        np.ndarray: Panorama con blending suave
    """
    if blend_method == 'linear':
        return linear_blend(panorama)
    elif blend_method == 'multiband':
        return multiband_blend(img1, img2, panorama)
    else:
        logger.warning(f"Método '{blend_method}' no implementado, usando linear")
        return linear_blend(panorama)


def linear_blend(panorama: np.ndarray) -> np.ndarray:
    """
    Aplica blending lineal simple al panorama.
    
    Args:
        panorama (np.ndarray): Panorama a procesar
        
    Returns:
        np.ndarray: Panorama con blending lineal
    """
    # Por simplicidad, retornar el panorama sin modificar
    # En una implementación completa, se calcularían máscaras de distancia
    return panorama


def multiband_blend(img1: np.ndarray, img2: np.ndarray, 
                   panorama: np.ndarray, num_bands: int = 4) -> np.ndarray:
    """
    Aplica multi-band blending para fusión suave.
    
    Args:
        img1 (np.ndarray): Primera imagen
        img2 (np.ndarray): Segunda imagen
        panorama (np.ndarray): Panorama base
        num_bands (int): Número de bandas de frecuencia
        
    Returns:
        np.ndarray: Panorama con multi-band blending
        
    References:
        Burt & Adelson (1983). "A Multiresolution Spline With Application to Image Mosaics"
    """
    # Implementación simplificada - retornar panorama base
    # Una implementación completa requeriría pirámides Laplacianas
    logger.warning("Multi-band blending no implementado completamente, usando panorama base")
    return panorama


def compute_homography_error(H_true: np.ndarray, H_estimated: np.ndarray,
                            points: np.ndarray) -> Dict[str, float]:
    """
    Calcula el error entre homografía verdadera y estimada.
    
    Args:
        H_true (np.ndarray): Matriz de homografía verdadera (3x3)
        H_estimated (np.ndarray): Matriz de homografía estimada (3x3)
        points (np.ndarray): Puntos de prueba (Nx2)
        
    Returns:
        Dict[str, float]: Métricas de error (RMSE, mean, max, etc.)
    """
    # Transformar puntos con ambas homografías
    points_homo = np.hstack([points, np.ones((len(points), 1))])
    
    pts_true = (H_true @ points_homo.T).T
    pts_true = pts_true[:, :2] / pts_true[:, 2:]
    
    pts_est = (H_estimated @ points_homo.T).T
    pts_est = pts_est[:, :2] / pts_est[:, 2:]
    
    # Calcular diferencias
    errors = np.linalg.norm(pts_true - pts_est, axis=1)
    
    metrics = {
        'rmse': np.sqrt(np.mean(errors**2)),
        'mean_error': np.mean(errors),
        'median_error': np.median(errors),
        'max_error': np.max(errors),
        'std_error': np.std(errors)
    }
    
    return metrics


def decompose_homography(H: np.ndarray) -> Dict[str, float]:
    """
    Descompone la homografía en transformaciones básicas.
    
    Args:
        H (np.ndarray): Matriz de homografía 3x3
        
    Returns:
        Dict[str, float]: Componentes de la transformación
    """
    # Normalizar
    H_norm = H / H[2, 2]
    
    # Extraer componentes aproximados
    tx = H_norm[0, 2]
    ty = H_norm[1, 2]
    
    # Escala aproximada
    scale_x = np.sqrt(H_norm[0, 0]**2 + H_norm[1, 0]**2)
    scale_y = np.sqrt(H_norm[0, 1]**2 + H_norm[1, 1]**2)
    
    # Rotación aproximada (en grados)
    rotation = np.arctan2(H_norm[1, 0], H_norm[0, 0]) * 180 / np.pi
    
    components = {
        'translation_x': tx,
        'translation_y': ty,
        'scale_x': scale_x,
        'scale_y': scale_y,
        'rotation_deg': rotation
    }
    
    return components


def stitch_multiple_images(images: List[np.ndarray],
                          keypoints_list: List[List[cv2.KeyPoint]],
                          descriptors_list: List[np.ndarray],
                          match_method: str = 'flann') -> np.ndarray:
    """
    Fusiona múltiples imágenes en un panorama.
    
    Args:
        images (List[np.ndarray]): Lista de imágenes a fusionar
        keypoints_list (List[List[cv2.KeyPoint]]): Keypoints de cada imagen
        descriptors_list (List[np.ndarray]): Descriptores de cada imagen
        match_method (str): Método de emparejamiento
        
    Returns:
        np.ndarray: Panorama completo
    """
    if len(images) < 2:
        logger.warning("Se necesitan al menos 2 imágenes")
        return images[0] if len(images) == 1 else np.array([])
    
    # Comenzar con la primera imagen como base
    panorama = images[0]
    
    # Fusionar secuencialmente
    for i in range(1, len(images)):
        logger.info(f"Fusionando imagen {i+1}/{len(images)}")
        
        # Importar matching aquí para evitar dependencia circular
        from matching import match_features
        
        # Emparejar con la imagen anterior
        matches = match_features(
            descriptors_list[i-1],
            descriptors_list[i],
            method=match_method
        )
        
        # Registrar
        panorama = register_images(
            panorama,
            images[i],
            keypoints_list[i-1],
            keypoints_list[i],
            matches
        )
    
    logger.info("Panorama completado")
    return panorama


if __name__ == "__main__":
    # Ejemplo de uso
    import sys
    from feature_detection import detect_sift_features
    from matching import match_features
    
    if len(sys.argv) < 3:
        print("Uso: python registration.py <imagen1> <imagen2>")
        sys.exit(1)
    
    # Cargar imágenes
    img1 = cv2.imread(sys.argv[1])
    img2 = cv2.imread(sys.argv[2])
    
    if img1 is None or img2 is None:
        print("Error: No se pudieron cargar las imágenes")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("REGISTRO DE IMÁGENES")
    print("="*60)
    
    # Detectar características
    print("\n1. Detectando características...")
    kp1, desc1 = detect_sift_features(img1)
    kp2, desc2 = detect_sift_features(img2)
    
    # Emparejar
    print("\n2. Emparejando características...")
    matches = match_features(desc1, desc2, method='flann')
    
    # Registrar
    print("\n3. Registrando imágenes...")
    panorama = register_images(img1, img2, kp1, kp2, matches)
    
    # Guardar
    output_path = "panorama_registered.jpg"
    cv2.imwrite(output_path, panorama)
    print(f"\n✓ Panorama guardado en: {output_path}")
    print(f"  Dimensiones: {panorama.shape[1]}x{panorama.shape[0]}")
    
    print("\n" + "="*60)
