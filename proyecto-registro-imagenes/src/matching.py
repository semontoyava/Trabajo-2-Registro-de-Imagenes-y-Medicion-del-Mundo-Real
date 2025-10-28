"""
matching.py

Módulo para emparejamiento de características entre imágenes.
Implementa FLANN, BFMatcher y filtrado con Lowe ratio test.

Universidad Nacional de Colombia
Visión por Computador - Trabajo 2: Registro de Imágenes
Autor: David A. Londoño
Fecha: Octubre 2025
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def match_features(descriptors1: np.ndarray, descriptors2: np.ndarray,
                  method: str = 'flann', ratio_test: float = 0.75,
                  cross_check: bool = False) -> List[cv2.DMatch]:
    """
    Empareja descriptores entre dos imágenes usando diferentes métodos.
    
    Args:
        descriptors1 (np.ndarray): Descriptores de la primera imagen
        descriptors2 (np.ndarray): Descriptores de la segunda imagen
        method (str): Método de emparejamiento ('flann', 'bf', 'bf_hamming')
        ratio_test (float): Umbral para Lowe's ratio test (0.7-0.8 típico)
        cross_check (bool): Activar cross-checking en BFMatcher
        
    Returns:
        List[cv2.DMatch]: Lista de matches filtrados
        
    References:
        Lowe, D. G. (2004). Ratio test for distinctive matches.
    """
    if descriptors1 is None or descriptors2 is None:
        logger.warning("Descriptores vacíos, retornando lista vacía de matches")
        return []
    
    if len(descriptors1) == 0 or len(descriptors2) == 0:
        logger.warning("No hay descriptores para emparejar")
        return []
    
    # Seleccionar método de emparejamiento
    if method == 'flann':
        matches = match_flann(descriptors1, descriptors2, ratio_test)
    elif method == 'bf':
        matches = match_bruteforce(descriptors1, descriptors2, 
                                   norm_type=cv2.NORM_L2,
                                   ratio_test=ratio_test,
                                   cross_check=cross_check)
    elif method == 'bf_hamming':
        matches = match_bruteforce(descriptors1, descriptors2,
                                   norm_type=cv2.NORM_HAMMING,
                                   ratio_test=ratio_test,
                                   cross_check=cross_check)
    else:
        raise ValueError(f"Método '{method}' no soportado. Use 'flann', 'bf' o 'bf_hamming'")
    
    logger.info(f"Emparejamiento completado: {len(matches)} matches ({method})")
    
    return matches


def match_flann(descriptors1: np.ndarray, descriptors2: np.ndarray,
               ratio_test: float = 0.75) -> List[cv2.DMatch]:
    """
    Empareja descriptores usando FLANN (Fast Library for Approximate Nearest Neighbors).
    
    FLANN es más rápido que BFMatcher para grandes conjuntos de descriptores.
    Recomendado para descriptores SIFT y SURF (punto flotante).
    
    Args:
        descriptors1 (np.ndarray): Descriptores de la primera imagen
        descriptors2 (np.ndarray): Descriptores de la segunda imagen
        ratio_test (float): Umbral para Lowe's ratio test
        
    Returns:
        List[cv2.DMatch]: Lista de buenos matches
    """
    # Configurar parámetros FLANN
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # Mayor número = más preciso pero más lento
    
    # Crear matcher FLANN
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # Convertir a float32 si es necesario
    if descriptors1.dtype != np.float32:
        descriptors1 = descriptors1.astype(np.float32)
    if descriptors2.dtype != np.float32:
        descriptors2 = descriptors2.astype(np.float32)
    
    # Encontrar los 2 mejores matches para cada descriptor
    knn_matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    
    # Aplicar Lowe's ratio test
    good_matches = apply_ratio_test(knn_matches, ratio_test)
    
    return good_matches


def match_bruteforce(descriptors1: np.ndarray, descriptors2: np.ndarray,
                    norm_type: int = cv2.NORM_L2,
                    ratio_test: float = 0.75,
                    cross_check: bool = False) -> List[cv2.DMatch]:
    """
    Empareja descriptores usando Brute-Force Matcher.
    
    Args:
        descriptors1 (np.ndarray): Descriptores de la primera imagen
        descriptors2 (np.ndarray): Descriptores de la segunda imagen
        norm_type (int): Tipo de norma (cv2.NORM_L2 para SIFT, cv2.NORM_HAMMING para ORB)
        ratio_test (float): Umbral para Lowe's ratio test
        cross_check (bool): Si True, solo retorna matches consistentes bidireccionales
        
    Returns:
        List[cv2.DMatch]: Lista de buenos matches
    """
    # Crear BFMatcher
    bf = cv2.BFMatcher(norm_type, crossCheck=cross_check)
    
    if cross_check:
        # Con cross_check, match() retorna solo los mejores matches consistentes
        matches = bf.match(descriptors1, descriptors2)
        # Ordenar por distancia
        matches = sorted(matches, key=lambda x: x.distance)
    else:
        # Sin cross_check, usar knnMatch y aplicar ratio test
        knn_matches = bf.knnMatch(descriptors1, descriptors2, k=2)
        matches = apply_ratio_test(knn_matches, ratio_test)
    
    return matches


def apply_ratio_test(knn_matches: List[List[cv2.DMatch]], 
                    ratio: float = 0.75) -> List[cv2.DMatch]:
    """
    Aplica el Lowe's ratio test para filtrar matches ambiguos.
    
    Un match es bueno si la distancia al mejor vecino es significativamente menor
    que la distancia al segundo mejor vecino.
    
    Args:
        knn_matches (List[List[cv2.DMatch]]): Matches k-NN (típicamente k=2)
        ratio (float): Umbral de ratio (0.7-0.8 típico, menor = más restrictivo)
        
    Returns:
        List[cv2.DMatch]: Matches que pasan el test
        
    References:
        Lowe, D. G. (2004). "Distinctive Image Features from Scale-Invariant Keypoints"
    """
    good_matches = []
    
    for match_pair in knn_matches:
        # Verificar que hay al menos 2 matches
        if len(match_pair) >= 2:
            best_match, second_match = match_pair[0], match_pair[1]
            # Aplicar ratio test
            if best_match.distance < ratio * second_match.distance:
                good_matches.append(best_match)
        elif len(match_pair) == 1:
            # Solo hay un match, aceptarlo (caso raro)
            good_matches.append(match_pair[0])
    
    logger.info(f"Ratio test: {len(knn_matches)} -> {len(good_matches)} matches")
    
    return good_matches


def filter_matches_by_distance(matches: List[cv2.DMatch], 
                               max_distance: Optional[float] = None,
                               percentile: float = 75.0) -> List[cv2.DMatch]:
    """
    Filtra matches por distancia, manteniendo solo los de menor distancia.
    
    Args:
        matches (List[cv2.DMatch]): Lista de matches
        max_distance (float, optional): Distancia máxima permitida
        percentile (float): Percentil de distancia para filtrar (si max_distance=None)
        
    Returns:
        List[cv2.DMatch]: Matches filtrados
    """
    if len(matches) == 0:
        return matches
    
    # Calcular distancias
    distances = [m.distance for m in matches]
    
    if max_distance is None:
        max_distance = np.percentile(distances, percentile)
    
    # Filtrar
    filtered_matches = [m for m in matches if m.distance <= max_distance]
    
    logger.info(f"Filtrado por distancia: {len(matches)} -> {len(filtered_matches)} matches "
               f"(umbral={max_distance:.2f})")
    
    return filtered_matches


def get_matched_points(keypoints1: List[cv2.KeyPoint], 
                       keypoints2: List[cv2.KeyPoint],
                       matches: List[cv2.DMatch]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extrae las coordenadas de los keypoints emparejados.
    
    Args:
        keypoints1 (List[cv2.KeyPoint]): Keypoints de la imagen 1
        keypoints2 (List[cv2.KeyPoint]): Keypoints de la imagen 2
        matches (List[cv2.DMatch]): Lista de matches
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Arrays de puntos emparejados (Nx2)
            - points1: Coordenadas en imagen 1
            - points2: Coordenadas en imagen 2
    """
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])
    
    return points1, points2


def visualize_matches(img1: np.ndarray, keypoints1: List[cv2.KeyPoint],
                     img2: np.ndarray, keypoints2: List[cv2.KeyPoint],
                     matches: List[cv2.DMatch],
                     max_matches: int = 100,
                     match_color: Tuple[int, int, int] = (0, 255, 0),
                     single_point_color: Tuple[int, int, int] = (255, 0, 0),
                     flags: int = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                     ) -> np.ndarray:
    """
    Visualiza los matches entre dos imágenes.
    
    Args:
        img1 (np.ndarray): Primera imagen
        keypoints1 (List[cv2.KeyPoint]): Keypoints de la imagen 1
        img2 (np.ndarray): Segunda imagen
        keypoints2 (List[cv2.KeyPoint]): Keypoints de la imagen 2
        matches (List[cv2.DMatch]): Lista de matches
        max_matches (int): Número máximo de matches a visualizar
        match_color (Tuple): Color para las líneas de match (BGR)
        single_point_color (Tuple): Color para keypoints sin match (BGR)
        flags (int): Flags de OpenCV para el dibujo
        
    Returns:
        np.ndarray: Imagen con matches dibujados
    """
    # Limitar número de matches para visualización
    matches_to_draw = matches[:max_matches] if len(matches) > max_matches else matches
    
    # Dibujar matches
    img_matches = cv2.drawMatches(
        img1, keypoints1,
        img2, keypoints2,
        matches_to_draw, None,
        matchColor=match_color,
        singlePointColor=single_point_color,
        flags=flags
    )
    
    # Añadir texto informativo
    text = f"Matches: {len(matches)} (mostrando {len(matches_to_draw)})"
    cv2.putText(img_matches, text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return img_matches


def compute_match_statistics(matches: List[cv2.DMatch]) -> Dict[str, float]:
    """
    Calcula estadísticas sobre los matches.
    
    Args:
        matches (List[cv2.DMatch]): Lista de matches
        
    Returns:
        Dict[str, float]: Diccionario con estadísticas
    """
    if len(matches) == 0:
        return {
            'num_matches': 0,
            'mean_distance': 0.0,
            'std_distance': 0.0,
            'min_distance': 0.0,
            'max_distance': 0.0,
            'median_distance': 0.0
        }
    
    distances = np.array([m.distance for m in matches])
    
    stats = {
        'num_matches': len(matches),
        'mean_distance': np.mean(distances),
        'std_distance': np.std(distances),
        'min_distance': np.min(distances),
        'max_distance': np.max(distances),
        'median_distance': np.median(distances)
    }
    
    return stats


def match_multiple_images(descriptors_list: List[np.ndarray],
                         method: str = 'flann',
                         ratio_test: float = 0.75) -> Dict[Tuple[int, int], List[cv2.DMatch]]:
    """
    Empareja características entre múltiples imágenes (todos los pares).
    
    Args:
        descriptors_list (List[np.ndarray]): Lista de arrays de descriptores
        method (str): Método de emparejamiento
        ratio_test (float): Umbral para ratio test
        
    Returns:
        Dict[Tuple[int, int], List[cv2.DMatch]]: Diccionario con matches para cada par
            Key: (idx_img1, idx_img2)
            Value: Lista de matches
    """
    n_images = len(descriptors_list)
    all_matches = {}
    
    for i in range(n_images):
        for j in range(i + 1, n_images):
            matches = match_features(
                descriptors_list[i],
                descriptors_list[j],
                method=method,
                ratio_test=ratio_test
            )
            all_matches[(i, j)] = matches
            logger.info(f"Par ({i}, {j}): {len(matches)} matches")
    
    return all_matches


if __name__ == "__main__":
    # Ejemplo de uso
    import sys
    from feature_detection import detect_sift_features
    
    if len(sys.argv) < 3:
        print("Uso: python matching.py <imagen1> <imagen2>")
        sys.exit(1)
    
    # Cargar imágenes
    img1_path, img2_path = sys.argv[1], sys.argv[2]
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        print("Error: No se pudieron cargar las imágenes")
        sys.exit(1)
    
    print(f"\nEmparejando: {img1_path} <-> {img2_path}")
    
    # Detectar características
    print("\nDetectando características con SIFT...")
    kp1, desc1 = detect_sift_features(img1)
    kp2, desc2 = detect_sift_features(img2)
    
    # Emparejar con diferentes métodos
    print("\n" + "="*60)
    print("COMPARACIÓN DE MÉTODOS DE EMPAREJAMIENTO")
    print("="*60)
    
    methods = ['flann', 'bf']
    
    for method in methods:
        print(f"\n{method.upper()}:")
        matches = match_features(desc1, desc2, method=method, ratio_test=0.75)
        
        # Estadísticas
        stats = compute_match_statistics(matches)
        print(f"  Matches: {stats['num_matches']}")
        print(f"  Distancia promedio: {stats['mean_distance']:.2f}")
        print(f"  Distancia std: {stats['std_distance']:.2f}")
        
        # Visualizar
        img_matches = visualize_matches(img1, kp1, img2, kp2, matches, max_matches=50)
        output_path = f"matches_{method}.jpg"
        cv2.imwrite(output_path, img_matches)
        print(f"  Guardado en: {output_path}")
    
    print("\n" + "="*60)
