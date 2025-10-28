"""
utils.py

Módulo de utilidades para el proyecto de registro de imágenes.
Incluye funciones de visualización, generación de imágenes sintéticas,
cálculo de métricas y otras utilidades.

Universidad Nacional de Colombia
Visión por Computador - Trabajo 2: Registro de Imágenes
Autor: David A. Londoño
Fecha: Octubre 2025
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_synthetic_image(image: np.ndarray,
                            rotation: float = 0.0,
                            translation: Tuple[float, float] = (0, 0),
                            scale: float = 1.0,
                            add_noise: bool = False,
                            noise_std: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera una imagen sintética aplicando transformaciones conocidas.
    
    Útil para validar algoritmos de registro con ground truth conocido.
    
    Args:
        image (np.ndarray): Imagen base
        rotation (float): Ángulo de rotación en grados
        translation (Tuple[float, float]): Traslación (tx, ty) en píxeles
        scale (float): Factor de escala
        add_noise (bool): Si True, añade ruido Gaussiano
        noise_std (float): Desviación estándar del ruido
        
    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - Imagen transformada
            - Matriz de transformación 3x3
    """
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    
    # Crear matriz de transformación
    # 1. Rotación y escala alrededor del centro
    M_rot_scale = cv2.getRotationMatrix2D(center, rotation, scale)
    
    # 2. Añadir traslación
    M_rot_scale[0, 2] += translation[0]
    M_rot_scale[1, 2] += translation[1]
    
    # Convertir a matriz 3x3 homogénea
    M_homo = np.vstack([M_rot_scale, [0, 0, 1]])
    
    # Aplicar transformación
    img_transformed = cv2.warpAffine(image, M_rot_scale, (width, height))
    
    # Añadir ruido si se solicita
    if add_noise:
        noise = np.random.normal(0, noise_std, img_transformed.shape).astype(np.uint8)
        img_transformed = cv2.add(img_transformed, noise)
    
    logger.info(f"Imagen sintética generada: rot={rotation}°, trans={translation}, scale={scale}")
    
    return img_transformed, M_homo


def compute_registration_metrics(H_true: np.ndarray, H_estimated: np.ndarray,
                                 test_points: Optional[np.ndarray] = None,
                                 image_shape: Optional[Tuple[int, int]] = None
                                 ) -> Dict[str, float]:
    """
    Calcula métricas de error entre homografía verdadera y estimada.
    
    Args:
        H_true (np.ndarray): Matriz de homografía verdadera (3x3)
        H_estimated (np.ndarray): Matriz de homografía estimada (3x3)
        test_points (np.ndarray, optional): Puntos de prueba (Nx2)
        image_shape (Tuple[int, int], optional): (height, width) para generar puntos
        
    Returns:
        Dict[str, float]: Métricas de error
    """
    # Generar puntos de prueba si no se proporcionan
    if test_points is None:
        if image_shape is None:
            # Usar puntos por defecto
            test_points = np.array([
                [0, 0], [100, 0], [0, 100], [100, 100],
                [50, 50], [200, 200], [300, 150]
            ], dtype=np.float32)
        else:
            h, w = image_shape
            # Generar grid de puntos
            x = np.linspace(0, w, 10)
            y = np.linspace(0, h, 10)
            xx, yy = np.meshgrid(x, y)
            test_points = np.column_stack([xx.ravel(), yy.ravel()])
    
    # Convertir a coordenadas homogéneas
    points_homo = np.hstack([test_points, np.ones((len(test_points), 1))])
    
    # Aplicar transformaciones
    pts_true = (H_true @ points_homo.T).T
    pts_true = pts_true[:, :2] / pts_true[:, 2:]
    
    pts_est = (H_estimated @ points_homo.T).T
    pts_est = pts_est[:, :2] / pts_est[:, 2:]
    
    # Calcular errores
    errors = np.linalg.norm(pts_true - pts_est, axis=1)
    
    # Calcular error angular (para transformaciones afines)
    try:
        # Extraer ángulo de rotación de las matrices
        angle_true = np.arctan2(H_true[1, 0], H_true[0, 0]) * 180 / np.pi
        angle_est = np.arctan2(H_estimated[1, 0], H_estimated[0, 0]) * 180 / np.pi
        angular_error = abs(angle_true - angle_est)
    except:
        angular_error = 0.0
    
    metrics = {
        'rmse': float(np.sqrt(np.mean(errors**2))),
        'mean_error': float(np.mean(errors)),
        'median_error': float(np.median(errors)),
        'max_error': float(np.max(errors)),
        'std_error': float(np.std(errors)),
        'angular_error': float(angular_error),
        'num_points': len(test_points)
    }
    
    return metrics


def visualize_registration(img_original: np.ndarray,
                          img_transformed: np.ndarray,
                          img_registered: np.ndarray,
                          save_path: Optional[str] = None):
    """
    Visualiza el proceso de registro: original, transformada y registrada.
    
    Args:
        img_original (np.ndarray): Imagen original
        img_transformed (np.ndarray): Imagen transformada (antes de registro)
        img_registered (np.ndarray): Imagen después del registro
        save_path (str, optional): Ruta para guardar la figura
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Convertir BGR a RGB para matplotlib
    if len(img_original.shape) == 3:
        img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
        img_transformed = cv2.cvtColor(img_transformed, cv2.COLOR_BGR2RGB)
        img_registered = cv2.cvtColor(img_registered, cv2.COLOR_BGR2RGB)
    
    axes[0].imshow(img_original, cmap='gray' if len(img_original.shape) == 2 else None)
    axes[0].set_title('Original', fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(img_transformed, cmap='gray' if len(img_transformed.shape) == 2 else None)
    axes[1].set_title('Transformada', fontsize=14)
    axes[1].axis('off')
    
    axes[2].imshow(img_registered, cmap='gray' if len(img_registered.shape) == 2 else None)
    axes[2].set_title('Registrada', fontsize=14)
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Visualización guardada en: {save_path}")
    
    plt.show()


def visualize_matches_comparison(img1: np.ndarray, img2: np.ndarray,
                                 matches_dict: Dict[str, Tuple],
                                 save_path: Optional[str] = None):
    """
    Compara visualmente diferentes métodos de emparejamiento.
    
    Args:
        img1 (np.ndarray): Primera imagen
        img2 (np.ndarray): Segunda imagen
        matches_dict (Dict[str, Tuple]): Diccionario con resultados de diferentes métodos
            {method_name: (keypoints1, keypoints2, matches)}
        save_path (str, optional): Ruta para guardar la figura
    """
    n_methods = len(matches_dict)
    fig, axes = plt.subplots(n_methods, 1, figsize=(15, 5 * n_methods))
    
    if n_methods == 1:
        axes = [axes]
    
    for idx, (method_name, (kp1, kp2, matches)) in enumerate(matches_dict.items()):
        # Dibujar matches
        img_matches = cv2.drawMatches(
            img1, kp1, img2, kp2, matches[:50],  # Mostrar solo 50 para claridad
            None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        # Convertir BGR a RGB
        img_matches = cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB)
        
        axes[idx].imshow(img_matches)
        axes[idx].set_title(f'{method_name}: {len(matches)} matches', fontsize=14)
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Comparación guardada en: {save_path}")
    
    plt.show()


def plot_metrics_table(metrics_list: List[Dict[str, any]],
                      method_names: List[str],
                      save_path: Optional[str] = None):
    """
    Muestra una tabla de métricas para diferentes métodos.
    
    Args:
        metrics_list (List[Dict]): Lista de diccionarios con métricas
        method_names (List[str]): Nombres de los métodos
        save_path (str, optional): Ruta para guardar la figura
    """
    fig, ax = plt.subplots(figsize=(10, len(metrics_list) * 0.5 + 2))
    ax.axis('tight')
    ax.axis('off')
    
    # Preparar datos para la tabla
    columns = ['Método', 'RMSE', 'Error Medio', 'Error Angular (°)']
    data = []
    
    for method, metrics in zip(method_names, metrics_list):
        row = [
            method,
            f"{metrics.get('rmse', 0):.2f}",
            f"{metrics.get('mean_error', 0):.2f}",
            f"{metrics.get('angular_error', 0):.2f}"
        ]
        data.append(row)
    
    table = ax.table(cellText=data, colLabels=columns, loc='center',
                    cellLoc='center', colWidths=[0.3, 0.2, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Estilo de la tabla
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Comparación de Métricas de Registro', fontsize=14, pad=20)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Tabla guardada en: {save_path}")
    
    plt.show()


def create_checkerboard_comparison(img1: np.ndarray, img2: np.ndarray,
                                   block_size: int = 50) -> np.ndarray:
    """
    Crea una visualización en tablero de ajedrez para comparar dos imágenes.
    
    Args:
        img1 (np.ndarray): Primera imagen
        img2 (np.ndarray): Segunda imagen (debe tener el mismo tamaño)
        block_size (int): Tamaño de los bloques del tablero
        
    Returns:
        np.ndarray: Imagen con visualización de tablero
    """
    h, w = img1.shape[:2]
    
    # Crear máscara de tablero
    mask = np.zeros((h, w), dtype=np.uint8)
    
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            if ((i // block_size) + (j // block_size)) % 2 == 0:
                mask[i:i+block_size, j:j+block_size] = 255
    
    # Expandir máscara a 3 canales si es necesario
    if len(img1.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # Combinar imágenes
    checkerboard = np.where(mask > 0, img1, img2)
    
    return checkerboard


def blend_images_alpha(img1: np.ndarray, img2: np.ndarray, 
                       alpha: float = 0.5) -> np.ndarray:
    """
    Fusiona dos imágenes con alpha blending.
    
    Args:
        img1 (np.ndarray): Primera imagen
        img2 (np.ndarray): Segunda imagen
        alpha (float): Peso de la primera imagen (0-1)
        
    Returns:
        np.ndarray: Imagen fusionada
    """
    return cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)


def save_results(results: Dict, output_dir: str, prefix: str = "result"):
    """
    Guarda resultados en formato estructurado.
    
    Args:
        results (Dict): Diccionario con resultados
        output_dir (str): Directorio de salida
        prefix (str): Prefijo para los archivos
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Guardar imágenes
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            if len(value.shape) in [2, 3]:  # Es una imagen
                filepath = output_path / f"{prefix}_{key}.jpg"
                cv2.imwrite(str(filepath), value)
                logger.info(f"Guardado: {filepath}")
    
    # Guardar métricas en JSON
    import json
    metrics = {k: v for k, v in results.items() if not isinstance(v, np.ndarray)}
    
    if metrics:
        filepath = output_path / f"{prefix}_metrics.json"
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Métricas guardadas: {filepath}")


def load_images_from_directory(directory: str, 
                               extensions: List[str] = ['.jpg', '.jpeg', '.png']
                               ) -> List[Tuple[str, np.ndarray]]:
    """
    Carga todas las imágenes de un directorio.
    
    Args:
        directory (str): Ruta del directorio
        extensions (List[str]): Extensiones de archivo a buscar
        
    Returns:
        List[Tuple[str, np.ndarray]]: Lista de (nombre_archivo, imagen)
    """
    dir_path = Path(directory)
    images = []
    
    for ext in extensions:
        for img_path in sorted(dir_path.glob(f'*{ext}')):
            img = cv2.imread(str(img_path))
            if img is not None:
                images.append((img_path.name, img))
                logger.info(f"Cargada: {img_path.name} ({img.shape})")
    
    return images


def resize_images_to_match(img1: np.ndarray, img2: np.ndarray,
                           scale_factor: Optional[float] = None
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Redimensiona imágenes para que tengan dimensiones compatibles.
    
    Args:
        img1 (np.ndarray): Primera imagen
        img2 (np.ndarray): Segunda imagen
        scale_factor (float, optional): Factor de escala fijo
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Imágenes redimensionadas
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    if scale_factor:
        new_h1, new_w1 = int(h1 * scale_factor), int(w1 * scale_factor)
        new_h2, new_w2 = int(h2 * scale_factor), int(w2 * scale_factor)
    else:
        # Usar el tamaño promedio
        avg_h = (h1 + h2) // 2
        avg_w = (w1 + w2) // 2
        new_h1, new_w1 = avg_h, avg_w
        new_h2, new_w2 = avg_h, avg_w
    
    img1_resized = cv2.resize(img1, (new_w1, new_h1))
    img2_resized = cv2.resize(img2, (new_w2, new_h2))
    
    logger.info(f"Redimensionado: {img1.shape} -> {img1_resized.shape}, "
               f"{img2.shape} -> {img2_resized.shape}")
    
    return img1_resized, img2_resized


def create_gif_from_images(image_list: List[np.ndarray], 
                           output_path: str,
                           duration: int = 500):
    """
    Crea un GIF animado a partir de una lista de imágenes.
    
    Args:
        image_list (List[np.ndarray]): Lista de imágenes
        output_path (str): Ruta del archivo GIF de salida
        duration (int): Duración de cada frame en milisegundos
    """
    try:
        from PIL import Image
        
        # Convertir imágenes de BGR a RGB
        pil_images = []
        for img in image_list:
            if len(img.shape) == 3:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = img
            pil_images.append(Image.fromarray(img_rgb))
        
        # Guardar como GIF
        pil_images[0].save(
            output_path,
            save_all=True,
            append_images=pil_images[1:],
            duration=duration,
            loop=0
        )
        
        logger.info(f"GIF creado: {output_path}")
        
    except ImportError:
        logger.error("Pillow no está instalado. Instale con: pip install Pillow")


if __name__ == "__main__":
    # Ejemplo de uso
    print("\n" + "="*60)
    print("MÓDULO DE UTILIDADES - EJEMPLOS")
    print("="*60)
    
    # Crear imagen sintética de prueba
    test_img = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
    
    print("\n1. Generando imagen sintética...")
    transformed_img, transform_matrix = generate_synthetic_image(
        test_img,
        rotation=30,
        translation=(50, 50),
        scale=1.2
    )
    print(f"   Matriz de transformación:\n{transform_matrix}")
    
    print("\n2. Calculando métricas de registro...")
    H_estimated = transform_matrix + np.random.randn(3, 3) * 0.1
    metrics = compute_registration_metrics(transform_matrix, H_estimated)
    print(f"   RMSE: {metrics['rmse']:.2f}")
    print(f"   Error angular: {metrics['angular_error']:.2f}°")
    
    print("\n" + "="*60)
