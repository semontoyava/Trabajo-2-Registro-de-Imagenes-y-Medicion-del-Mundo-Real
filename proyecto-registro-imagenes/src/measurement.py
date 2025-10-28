"""
measurement.py

Módulo para medición interactiva de distancias en imágenes con calibración métrica.
Permite calibrar la escala usando objetos de referencia y medir dimensiones reales.

Universidad Nacional de Colombia
Visión por Computador - Trabajo 2: Registro de Imágenes
Autor: David A. Londoño
Fecha: Octubre 2025
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import json
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MeasurementTool:
    """
    Herramienta interactiva para medición de distancias en imágenes.
    
    Permite calibrar la escala usando objetos de referencia conocidos
    y medir dimensiones de objetos en la imagen.
    
    Attributes:
        image (np.ndarray): Imagen sobre la que se realizan mediciones
        scale_pixels_per_cm (float): Factor de escala (píxeles por cm)
        measurements (List[Dict]): Historial de mediciones realizadas
        points (List[Tuple]): Puntos seleccionados para medición actual
    """
    
    def __init__(self, image: np.ndarray):
        """
        Inicializa la herramienta de medición.
        
        Args:
            image (np.ndarray): Imagen para realizar mediciones
        """
        self.image = image.copy()
        self.scale_pixels_per_cm = None
        self.measurements = []
        self.points = []
        self.calibration_data = {}
        self.window_name = "Herramienta de Medición"
        
    def calibrate_scale(self, reference_object: str, 
                       real_dimension_cm: float,
                       interactive: bool = True) -> float:
        """
        Calibra la escala métrica usando un objeto de referencia.
        
        Args:
            reference_object (str): Nombre del objeto de referencia
            real_dimension_cm (float): Dimensión real conocida en cm
            interactive (bool): Si True, permite selección interactiva de puntos
            
        Returns:
            float: Factor de escala (píxeles por cm)
        """
        logger.info(f"Calibrando escala con: {reference_object} ({real_dimension_cm} cm)")
        
        if interactive:
            print(f"\n{'='*60}")
            print(f"CALIBRACIÓN DE ESCALA")
            print(f"{'='*60}")
            print(f"Objeto de referencia: {reference_object}")
            print(f"Dimensión real: {real_dimension_cm} cm")
            print(f"\nInstrucciones:")
            print(f"  1. Haga clic en el punto inicial de la medida")
            print(f"  2. Haga clic en el punto final de la medida")
            print(f"  3. Presione 'q' para confirmar")
            print(f"  4. Presione 'r' para reiniciar")
            print(f"{'='*60}\n")
            
            # Seleccionar puntos interactivamente
            pixel_distance = self._select_distance_interactive()
            
            if pixel_distance is None:
                logger.error("Calibración cancelada")
                return None
        else:
            # Usar puntos previamente seleccionados
            if len(self.points) < 2:
                logger.error("Se necesitan 2 puntos para calibrar")
                return None
            pixel_distance = np.linalg.norm(
                np.array(self.points[0]) - np.array(self.points[1])
            )
        
        # Calcular escala
        self.scale_pixels_per_cm = pixel_distance / real_dimension_cm
        
        # Guardar datos de calibración
        self.calibration_data = {
            'reference_object': reference_object,
            'real_dimension_cm': real_dimension_cm,
            'pixel_distance': float(pixel_distance),
            'scale_pixels_per_cm': float(self.scale_pixels_per_cm),
            'points': [tuple(p) for p in self.points[:2]]
        }
        
        logger.info(f"✓ Escala calibrada: {self.scale_pixels_per_cm:.2f} píxeles/cm")
        logger.info(f"  ({1/self.scale_pixels_per_cm:.4f} cm/píxel)")
        
        return self.scale_pixels_per_cm
    
    def _select_distance_interactive(self) -> Optional[float]:
        """
        Permite seleccionar dos puntos interactivamente para medir distancia.
        
        Returns:
            float: Distancia en píxeles entre los puntos (o None si se cancela)
        """
        self.points = []
        img_display = self.image.copy()
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal img_display
            
            if event == cv2.EVENT_LBUTTONDOWN:
                self.points.append((x, y))
                
                # Dibujar punto
                cv2.circle(img_display, (x, y), 5, (0, 255, 0), -1)
                
                # Si hay dos puntos, dibujar línea
                if len(self.points) == 2:
                    cv2.line(img_display, self.points[0], self.points[1], 
                            (0, 255, 0), 2)
                    
                    # Calcular y mostrar distancia en píxeles
                    dist = np.linalg.norm(
                        np.array(self.points[0]) - np.array(self.points[1])
                    )
                    text = f"{dist:.1f} px"
                    mid_point = (
                        (self.points[0][0] + self.points[1][0]) // 2,
                        (self.points[0][1] + self.points[1][1]) // 2
                    )
                    cv2.putText(img_display, text, mid_point,
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                cv2.imshow(self.window_name, img_display)
        
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, mouse_callback)
        cv2.imshow(self.window_name, img_display)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):  # Confirmar
                if len(self.points) >= 2:
                    break
                else:
                    print("⚠ Seleccione al menos 2 puntos")
            
            elif key == ord('r'):  # Reiniciar
                self.points = []
                img_display = self.image.copy()
                cv2.imshow(self.window_name, img_display)
            
            elif key == 27:  # ESC - Cancelar
                cv2.destroyWindow(self.window_name)
                return None
        
        cv2.destroyWindow(self.window_name)
        
        if len(self.points) >= 2:
            dist = np.linalg.norm(
                np.array(self.points[0]) - np.array(self.points[1])
            )
            return dist
        
        return None
    
    def measure_distance(self, object_name: str, 
                        interactive: bool = True) -> Optional[Dict]:
        """
        Mide la distancia de un objeto en la imagen.
        
        Args:
            object_name (str): Nombre del objeto a medir
            interactive (bool): Si True, permite selección interactiva
            
        Returns:
            Dict: Diccionario con información de la medición
        """
        if self.scale_pixels_per_cm is None:
            logger.error("Primero debe calibrar la escala con calibrate_scale()")
            return None
        
        logger.info(f"Midiendo: {object_name}")
        
        if interactive:
            print(f"\n{'='*60}")
            print(f"MEDICIÓN: {object_name}")
            print(f"{'='*60}")
            print(f"Instrucciones:")
            print(f"  - Haga clic en dos puntos para medir la distancia")
            print(f"  - Presione 'q' para confirmar")
            print(f"  - Presione 'r' para reiniciar")
            print(f"{'='*60}\n")
            
            pixel_distance = self._select_distance_interactive()
            
            if pixel_distance is None:
                return None
        else:
            if len(self.points) < 2:
                logger.error("Se necesitan 2 puntos")
                return None
            pixel_distance = np.linalg.norm(
                np.array(self.points[0]) - np.array(self.points[1])
            )
        
        # Convertir a cm
        distance_cm = pixel_distance / self.scale_pixels_per_cm
        
        # Crear registro de medición
        measurement = {
            'object_name': object_name,
            'distance_pixels': float(pixel_distance),
            'distance_cm': float(distance_cm),
            'points': [tuple(p) for p in self.points[:2]],
            'scale_used': float(self.scale_pixels_per_cm)
        }
        
        self.measurements.append(measurement)
        
        logger.info(f"✓ {object_name}: {distance_cm:.2f} cm ({pixel_distance:.1f} px)")
        
        return measurement
    
    def measure_interactive(self):
        """
        Modo interactivo para realizar múltiples mediciones.
        """
        if self.scale_pixels_per_cm is None:
            print("⚠ Error: Primero debe calibrar la escala")
            return
        
        print(f"\n{'='*60}")
        print("MODO INTERACTIVO DE MEDICIÓN")
        print(f"{'='*60}")
        print(f"Escala actual: {self.scale_pixels_per_cm:.2f} píxeles/cm")
        print(f"\nInstrucciones:")
        print(f"  1. Ingrese el nombre del objeto a medir")
        print(f"  2. Seleccione dos puntos en la imagen")
        print(f"  3. Repita para múltiples objetos")
        print(f"  4. Ingrese 'q' como nombre para salir")
        print(f"{'='*60}\n")
        
        while True:
            object_name = input("\nNombre del objeto (o 'q' para salir): ").strip()
            
            if object_name.lower() == 'q':
                break
            
            if not object_name:
                print("⚠ Por favor ingrese un nombre válido")
                continue
            
            measurement = self.measure_distance(object_name, interactive=True)
            
            if measurement:
                print(f"✓ Medición guardada: {measurement['distance_cm']:.2f} cm")
        
        print(f"\n✓ Total de mediciones realizadas: {len(self.measurements)}")
    
    def measure_objects(self, object_names: List[str]) -> List[Dict]:
        """
        Mide múltiples objetos de forma secuencial.
        
        Args:
            object_names (List[str]): Lista de nombres de objetos a medir
            
        Returns:
            List[Dict]: Lista de mediciones realizadas
        """
        results = []
        
        for obj_name in object_names:
            measurement = self.measure_distance(obj_name, interactive=True)
            if measurement:
                results.append(measurement)
        
        return results
    
    def compute_uncertainty(self, n_samples: int = 5) -> Dict[str, float]:
        """
        Estima la incertidumbre de medición mediante muestreo repetido.
        
        Args:
            n_samples (int): Número de mediciones de muestra para calcular varianza
            
        Returns:
            Dict[str, float]: Estadísticas de incertidumbre
        """
        if len(self.measurements) < 2:
            logger.warning("Se necesitan al menos 2 mediciones para calcular incertidumbre")
            return {'mean': 0.0, 'std': 0.0, 'variance': 0.0}
        
        # Calcular estadísticas de las mediciones realizadas
        distances_cm = [m['distance_cm'] for m in self.measurements]
        
        uncertainty = {
            'mean': float(np.mean(distances_cm)),
            'std': float(np.std(distances_cm)),
            'variance': float(np.var(distances_cm)),
            'min': float(np.min(distances_cm)),
            'max': float(np.max(distances_cm)),
            'n_measurements': len(distances_cm)
        }
        
        logger.info(f"Incertidumbre: ±{uncertainty['std']:.2f} cm")
        
        return uncertainty
    
    def save_measurements(self, filepath: str):
        """
        Guarda las mediciones en un archivo JSON.
        
        Args:
            filepath (str): Ruta del archivo de salida
        """
        data = {
            'calibration': self.calibration_data,
            'measurements': self.measurements,
            'uncertainty': self.compute_uncertainty() if len(self.measurements) > 1 else None
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Mediciones guardadas en: {filepath}")
    
    def load_measurements(self, filepath: str):
        """
        Carga mediciones desde un archivo JSON.
        
        Args:
            filepath (str): Ruta del archivo a cargar
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.calibration_data = data.get('calibration', {})
        self.measurements = data.get('measurements', [])
        
        if 'scale_pixels_per_cm' in self.calibration_data:
            self.scale_pixels_per_cm = self.calibration_data['scale_pixels_per_cm']
        
        logger.info(f"✓ Mediciones cargadas desde: {filepath}")
        logger.info(f"  - {len(self.measurements)} mediciones")
    
    def visualize_measurements(self, save_path: Optional[str] = None) -> np.ndarray:
        """
        Visualiza todas las mediciones realizadas sobre la imagen.
        
        Args:
            save_path (str, optional): Ruta para guardar la visualización
            
        Returns:
            np.ndarray: Imagen con mediciones dibujadas
        """
        img_vis = self.image.copy()
        
        # Dibujar calibración
        if 'points' in self.calibration_data:
            pts = self.calibration_data['points']
            cv2.line(img_vis, pts[0], pts[1], (255, 0, 0), 3)
            cv2.circle(img_vis, pts[0], 7, (255, 0, 0), -1)
            cv2.circle(img_vis, pts[1], 7, (255, 0, 0), -1)
            
            text = f"Ref: {self.calibration_data['real_dimension_cm']:.1f} cm"
            cv2.putText(img_vis, text, 
                       (pts[0][0], pts[0][1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Dibujar mediciones
        for i, measurement in enumerate(self.measurements):
            pts = measurement['points']
            color = tuple(np.random.randint(0, 255, 3).tolist())
            
            cv2.line(img_vis, pts[0], pts[1], color, 2)
            cv2.circle(img_vis, pts[0], 5, color, -1)
            cv2.circle(img_vis, pts[1], 5, color, -1)
            
            # Texto con nombre y medida
            text = f"{measurement['object_name']}: {measurement['distance_cm']:.1f} cm"
            mid_point = (
                (pts[0][0] + pts[1][0]) // 2,
                (pts[0][1] + pts[1][1]) // 2 - 10
            )
            cv2.putText(img_vis, text, mid_point,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        if save_path:
            cv2.imwrite(save_path, img_vis)
            logger.info(f"✓ Visualización guardada en: {save_path}")
        
        return img_vis
    
    def generate_report(self) -> str:
        """
        Genera un reporte textual de las mediciones.
        
        Returns:
            str: Reporte formateado
        """
        report = []
        report.append("="*60)
        report.append("REPORTE DE MEDICIONES")
        report.append("="*60)
        
        # Calibración
        if self.calibration_data:
            report.append("\nCALIBRACIÓN:")
            report.append(f"  Objeto de referencia: {self.calibration_data['reference_object']}")
            report.append(f"  Dimensión real: {self.calibration_data['real_dimension_cm']:.2f} cm")
            report.append(f"  Escala: {self.scale_pixels_per_cm:.2f} píxeles/cm")
            report.append(f"  Resolución: {1/self.scale_pixels_per_cm:.4f} cm/píxel")
        
        # Mediciones
        report.append(f"\nMEDICIONES ({len(self.measurements)}):")
        for i, m in enumerate(self.measurements, 1):
            report.append(f"  {i}. {m['object_name']}: {m['distance_cm']:.2f} cm")
        
        # Estadísticas
        if len(self.measurements) > 1:
            uncertainty = self.compute_uncertainty()
            report.append(f"\nESTADÍSTICAS:")
            report.append(f"  Media: {uncertainty['mean']:.2f} cm")
            report.append(f"  Desviación estándar: ±{uncertainty['std']:.2f} cm")
            report.append(f"  Rango: [{uncertainty['min']:.2f}, {uncertainty['max']:.2f}] cm")
        
        report.append("="*60)
        
        return "\n".join(report)


if __name__ == "__main__":
    # Ejemplo de uso
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python measurement.py <imagen>")
        sys.exit(1)
    
    # Cargar imagen
    img = cv2.imread(sys.argv[1])
    
    if img is None:
        print(f"Error: No se pudo cargar la imagen {sys.argv[1]}")
        sys.exit(1)
    
    # Crear herramienta
    tool = MeasurementTool(img)
    
    # Calibrar con objeto de referencia
    print("Calibrando con cuadro de la Virgen de Guadalupe (117 cm)...")
    tool.calibrate_scale(
        reference_object="Cuadro Virgen de Guadalupe",
        real_dimension_cm=117.0,
        interactive=True
    )
    
    # Mediciones interactivas
    if tool.scale_pixels_per_cm:
        tool.measure_interactive()
        
        # Mostrar reporte
        print("\n" + tool.generate_report())
        
        # Guardar resultados
        tool.save_measurements("mediciones.json")
        tool.visualize_measurements("mediciones_visualizacion.jpg")
