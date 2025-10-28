"""
feature_detection.py

Módulo para detección de características en imágenes usando diferentes detectores.
Implementa SIFT, ORB y AKAZE para la extracción de keypoints y descriptores.

Universidad Nacional de Colombia
Visión por Computador - Trabajo 2: Registro de Imágenes
Autor: David A. Londoño
Fecha: Octubre 2025
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureDetector:
    """
    Clase base para detección de características en imágenes.
    
    Attributes:
        detector_type (str): Tipo de detector ('sift', 'orb', 'akaze')
        detector: Objeto detector de OpenCV
    """
    
    def __init__(self, detector_type: str = 'sift'):
        """
        Inicializa el detector de características.
        
        Args:
            detector_type (str): Tipo de detector a utilizar.
                Opciones: 'sift', 'orb', 'akaze'
        """
        self.detector_type = detector_type.lower()
        self.detector = self._create_detector()
        
    def _create_detector(self):
        """
        Crea el detector correspondiente según el tipo especificado.
        
        Returns:
            Objeto detector de OpenCV
            
        Raises:
            ValueError: Si el tipo de detector no es válido
        """
        if self.detector_type == 'sift':
            return cv2.SIFT_create()
        elif self.detector_type == 'orb':
            return cv2.ORB_create(nfeatures=5000)
        elif self.detector_type == 'akaze':
            return cv2.AKAZE_create()
        else:
            raise ValueError(f"Detector tipo '{self.detector_type}' no soportado. "
                           f"Use 'sift', 'orb' o 'akaze'.")
    
    def detect_and_compute(self, image: np.ndarray, mask: Optional[np.ndarray] = None
                          ) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        Detecta keypoints y calcula descriptores en la imagen.
        
        Args:
            image (np.ndarray): Imagen de entrada (color o escala de grises)
            mask (np.ndarray, optional): Máscara para limitar la región de detección
            
        Returns:
            Tuple[List[cv2.KeyPoint], np.ndarray]: 
                - Lista de keypoints detectados
                - Array de descriptores (NxD donde N=número de keypoints)
        """
        # Convertir a escala de grises si es necesario
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Detectar y computar
        keypoints, descriptors = self.detector.detectAndCompute(gray, mask)
        
        logger.info(f"Detectados {len(keypoints)} keypoints con {self.detector_type.upper()}")
        
        return keypoints, descriptors


def detect_sift_features(image: np.ndarray, n_features: int = 0, 
                        contrast_threshold: float = 0.04,
                        edge_threshold: float = 10.0,
                        sigma: float = 1.6) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """
    Detecta características usando el algoritmo SIFT (Scale-Invariant Feature Transform).
    
    SIFT es invariante a escala, rotación e iluminación, ideal para registro robusto.
    
    Args:
        image (np.ndarray): Imagen de entrada
        n_features (int): Número máximo de características a retener (0 = todas)
        contrast_threshold (float): Umbral de contraste para filtrar keypoints débiles
        edge_threshold (float): Umbral para filtrar keypoints en bordes
        sigma (float): Sigma del Gaussiano aplicado a la imagen inicial
        
    Returns:
        Tuple[List[cv2.KeyPoint], np.ndarray]: Keypoints y descriptores SIFT
        
    References:
        Lowe, D. G. (2004). "Distinctive Image Features from Scale-Invariant Keypoints"
    """
    # Crear detector SIFT con parámetros personalizados
    sift = cv2.SIFT_create(
        nfeatures=n_features,
        contrastThreshold=contrast_threshold,
        edgeThreshold=edge_threshold,
        sigma=sigma
    )
    
    # Convertir a escala de grises
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Detectar y computar
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    logger.info(f"SIFT: Detectados {len(keypoints)} keypoints")
    
    return keypoints, descriptors


def detect_orb_features(image: np.ndarray, n_features: int = 5000,
                       scale_factor: float = 1.2, n_levels: int = 8,
                       edge_threshold: int = 31) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """
    Detecta características usando el algoritmo ORB (Oriented FAST and Rotated BRIEF).
    
    ORB es una alternativa rápida y libre de patentes a SIFT/SURF.
    
    Args:
        image (np.ndarray): Imagen de entrada
        n_features (int): Número máximo de características a retener
        scale_factor (float): Factor de escala entre niveles de pirámide
        n_levels (int): Número de niveles en la pirámide
        edge_threshold (int): Tamaño del borde donde no se detectan características
        
    Returns:
        Tuple[List[cv2.KeyPoint], np.ndarray]: Keypoints y descriptores ORB
        
    References:
        Rublee, E. et al. (2011). "ORB: An efficient alternative to SIFT or SURF"
    """
    # Crear detector ORB con parámetros personalizados
    orb = cv2.ORB_create(
        nfeatures=n_features,
        scaleFactor=scale_factor,
        nlevels=n_levels,
        edgeThreshold=edge_threshold
    )
    
    # Convertir a escala de grises
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Detectar y computar
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    
    logger.info(f"ORB: Detectados {len(keypoints)} keypoints")
    
    return keypoints, descriptors


def detect_akaze_features(image: np.ndarray, descriptor_type: int = cv2.AKAZE_DESCRIPTOR_MLDB,
                         descriptor_size: int = 0, descriptor_channels: int = 3,
                         threshold: float = 0.001) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """
    Detecta características usando el algoritmo AKAZE (Accelerated-KAZE).
    
    AKAZE es más rápido que KAZE y proporciona características robustas.
    
    Args:
        image (np.ndarray): Imagen de entrada
        descriptor_type (int): Tipo de descriptor AKAZE
        descriptor_size (int): Tamaño del descriptor (0 = full size)
        descriptor_channels (int): Número de canales del descriptor
        threshold (float): Umbral de detección
        
    Returns:
        Tuple[List[cv2.KeyPoint], np.ndarray]: Keypoints y descriptores AKAZE
        
    References:
        Alcantarilla, P. F. et al. (2012). "KAZE Features"
    """
    # Crear detector AKAZE con parámetros personalizados
    akaze = cv2.AKAZE_create(
        descriptor_type=descriptor_type,
        descriptor_size=descriptor_size,
        descriptor_channels=descriptor_channels,
        threshold=threshold
    )
    
    # Convertir a escala de grises
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Detectar y computar
    keypoints, descriptors = akaze.detectAndCompute(gray, None)
    
    logger.info(f"AKAZE: Detectados {len(keypoints)} keypoints")
    
    return keypoints, descriptors


def compare_detectors(image: np.ndarray, detectors: Optional[List[str]] = None
                     ) -> Dict[str, Tuple[List[cv2.KeyPoint], np.ndarray]]:
    """
    Compara múltiples detectores de características en la misma imagen.
    
    Args:
        image (np.ndarray): Imagen de entrada
        detectors (List[str], optional): Lista de detectores a comparar.
            Por defecto: ['sift', 'orb', 'akaze']
    
    Returns:
        Dict[str, Tuple]: Diccionario con resultados de cada detector
            {detector_name: (keypoints, descriptors)}
    """
    if detectors is None:
        detectors = ['sift', 'orb', 'akaze']
    
    results = {}
    
    for detector_name in detectors:
        try:
            detector = FeatureDetector(detector_name)
            kp, desc = detector.detect_and_compute(image)
            results[detector_name] = (kp, desc)
            logger.info(f"{detector_name.upper()}: {len(kp)} keypoints detectados")
        except Exception as e:
            logger.error(f"Error con detector {detector_name}: {e}")
            results[detector_name] = ([], None)
    
    return results


def visualize_keypoints(image: np.ndarray, keypoints: List[cv2.KeyPoint],
                       color: Tuple[int, int, int] = (0, 255, 0),
                       flags: int = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
                       ) -> np.ndarray:
    """
    Visualiza los keypoints detectados sobre la imagen.
    
    Args:
        image (np.ndarray): Imagen base
        keypoints (List[cv2.KeyPoint]): Lista de keypoints a visualizar
        color (Tuple[int, int, int]): Color para dibujar los keypoints (BGR)
        flags (int): Flags de OpenCV para el dibujo
        
    Returns:
        np.ndarray: Imagen con keypoints dibujados
    """
    img_with_keypoints = cv2.drawKeypoints(
        image, keypoints, None, color=color, flags=flags
    )
    
    return img_with_keypoints


def filter_keypoints_by_response(keypoints: List[cv2.KeyPoint], 
                                 descriptors: np.ndarray,
                                 top_k: int = 1000) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """
    Filtra los keypoints manteniendo solo los top-k con mayor response.
    
    Args:
        keypoints (List[cv2.KeyPoint]): Lista de keypoints
        descriptors (np.ndarray): Descriptores correspondientes
        top_k (int): Número de keypoints a mantener
        
    Returns:
        Tuple[List[cv2.KeyPoint], np.ndarray]: Keypoints y descriptores filtrados
    """
    if len(keypoints) <= top_k:
        return keypoints, descriptors
    
    # Ordenar por response (descendente)
    sorted_indices = sorted(range(len(keypoints)), 
                          key=lambda i: keypoints[i].response, 
                          reverse=True)
    
    # Mantener top-k
    top_indices = sorted_indices[:top_k]
    
    filtered_kp = [keypoints[i] for i in top_indices]
    filtered_desc = descriptors[top_indices] if descriptors is not None else None
    
    logger.info(f"Filtrados {len(keypoints)} -> {len(filtered_kp)} keypoints")
    
    return filtered_kp, filtered_desc


if __name__ == "__main__":
    # Ejemplo de uso
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python feature_detection.py <ruta_imagen>")
        sys.exit(1)
    
    # Cargar imagen
    img_path = sys.argv[1]
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"Error: No se pudo cargar la imagen {img_path}")
        sys.exit(1)
    
    print(f"\nAnalizando imagen: {img_path}")
    print(f"Dimensiones: {img.shape}")
    
    # Comparar detectores
    print("\n" + "="*60)
    print("COMPARACIÓN DE DETECTORES")
    print("="*60)
    
    results = compare_detectors(img)
    
    # Visualizar resultados
    for detector_name, (kp, desc) in results.items():
        if len(kp) > 0:
            img_with_kp = visualize_keypoints(img, kp)
            output_path = f"keypoints_{detector_name}.jpg"
            cv2.imwrite(output_path, img_with_kp)
            print(f"\n{detector_name.upper()}:")
            print(f"  - Keypoints: {len(kp)}")
            print(f"  - Descriptor shape: {desc.shape if desc is not None else 'None'}")
            print(f"  - Guardado en: {output_path}")
    
    print("\n" + "="*60)
