"""
measure_comedor.py

Herramienta interactiva para calibración y medición de objetos en el panorama.
Parte 3 del Trabajo: Calibración y Medición del Mundo Real

Universidad Nacional de Colombia
Visión por Computador - Trabajo 2: Registro de Imágenes
Autor: David A. Londoño
Fecha: Octubre 2025

Objetos de referencia:
- Cuadro Virgen de Guadalupe: 117 cm de altura
- Mesa: 161.1 cm de ancho

Mediciones requeridas:
- Ancho del cuadro
- Largo de la mesa
- 3 elementos adicionales (ventanas, sillas, plantas, etc.)
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt

# Configuración
RESULTS_DIR = Path("results/comedor_registration")
MEASUREMENTS_DIR = Path("results/measurements")
MEASUREMENTS_DIR.mkdir(parents=True, exist_ok=True)

# Objetos de referencia conocidos
REFERENCE_OBJECTS = {
    "cuadro_altura": 117.0,  # cm
    "mesa_ancho": 161.1      # cm
}


class InteractiveMeasurementTool:
    """
    Herramienta interactiva para medir distancias en imágenes.
    Permite calibrar la escala usando objetos de referencia y luego
    medir otros objetos en la escena.
    """
    
    def __init__(self, image: np.ndarray, window_name: str = "Medición Interactiva"):
        """
        Inicializa la herramienta de medición.
        
        Args:
            image: Imagen a medir (BGR)
            window_name: Nombre de la ventana
        """
        self.image = image.copy()
        self.display_image = image.copy()
        self.window_name = window_name
        
        # Estado de medición
        self.points = []
        self.measurements = []
        self.scale_factor = None  # píxeles por cm
        self.calibrated = False
        
        # Configuración visual
        self.point_color = (0, 255, 0)  # Verde
        self.line_color = (255, 0, 0)   # Azul
        self.text_color = (0, 255, 255) # Amarillo
        self.point_radius = 5
        self.line_thickness = 2
        
    def mouse_callback(self, event, x, y, flags, param):
        """Callback para eventos del mouse."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            
            # Dibujar punto
            cv2.circle(self.display_image, (x, y), self.point_radius, 
                      self.point_color, -1)
            
            # Si tenemos 2 puntos, dibujar línea
            if len(self.points) == 2:
                self.draw_measurement()
            
            cv2.imshow(self.window_name, self.display_image)
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Clic derecho: reiniciar medición actual
            self.points = []
            self.display_image = self.image.copy()
            self.redraw_measurements()
            cv2.imshow(self.window_name, self.display_image)
    
    def draw_measurement(self):
        """Dibuja la línea de medición entre dos puntos."""
        if len(self.points) != 2:
            return
        
        p1, p2 = self.points
        
        # Dibujar línea
        cv2.line(self.display_image, p1, p2, self.line_color, self.line_thickness)
        
        # Calcular distancia en píxeles
        dist_pixels = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        
        # Convertir a cm si está calibrado
        if self.calibrated and self.scale_factor:
            dist_cm = dist_pixels / self.scale_factor
            text = f"{dist_cm:.1f} cm ({dist_pixels:.0f} px)"
        else:
            text = f"{dist_pixels:.0f} px"
        
        # Posición del texto (punto medio)
        text_pos = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2 - 10)
        
        # Dibujar texto con fondo
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Rectángulo de fondo
        cv2.rectangle(self.display_image, 
                     (text_pos[0] - 5, text_pos[1] - text_h - 5),
                     (text_pos[0] + text_w + 5, text_pos[1] + 5),
                     (0, 0, 0), -1)
        
        # Texto
        cv2.putText(self.display_image, text, text_pos, font, font_scale,
                   self.text_color, thickness)
    
    def redraw_measurements(self):
        """Redibuja todas las mediciones guardadas."""
        for measurement in self.measurements:
            p1, p2 = measurement['points']
            cv2.circle(self.display_image, p1, self.point_radius, self.point_color, -1)
            cv2.circle(self.display_image, p2, self.point_radius, self.point_color, -1)
            cv2.line(self.display_image, p1, p2, self.line_color, self.line_thickness)
            
            # Texto
            dist_cm = measurement['distance_cm']
            dist_px = measurement['distance_pixels']
            text = f"{dist_cm:.1f} cm ({dist_px:.0f} px)"
            text_pos = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2 - 10)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
            
            cv2.rectangle(self.display_image,
                         (text_pos[0] - 5, text_pos[1] - text_h - 5),
                         (text_pos[0] + text_w + 5, text_pos[1] + 5),
                         (0, 0, 0), -1)
            
            cv2.putText(self.display_image, text, text_pos, font, font_scale,
                       self.text_color, thickness)
    
    def calibrate(self, reference_distance_cm: float):
        """
        Calibra la escala usando una distancia conocida.
        
        Args:
            reference_distance_cm: Distancia real en cm
        """
        if len(self.points) != 2:
            print("[ERROR] Necesitas marcar 2 puntos para calibrar")
            return False
        
        p1, p2 = self.points
        dist_pixels = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        
        self.scale_factor = dist_pixels / reference_distance_cm
        self.calibrated = True
        
        # Guardar medición de calibración
        self.measurements.append({
            'name': 'Calibración',
            'points': (p1, p2),
            'distance_pixels': dist_pixels,
            'distance_cm': reference_distance_cm,
            'is_calibration': True
        })
        
        print(f"✓ Calibración exitosa!")
        print(f"  Escala: {self.scale_factor:.2f} píxeles/cm")
        print(f"  Distancia: {dist_pixels:.0f} px = {reference_distance_cm} cm")
        
        self.points = []
        self.display_image = self.image.copy()
        self.redraw_measurements()
        
        return True
    
    def measure(self, object_name: str):
        """
        Registra una medición.
        
        Args:
            object_name: Nombre del objeto medido
        """
        if not self.calibrated:
            print("[ERROR] Primero debes calibrar la escala")
            return False
        
        if len(self.points) != 2:
            print("[ERROR] Necesitas marcar 2 puntos para medir")
            return False
        
        p1, p2 = self.points
        dist_pixels = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        dist_cm = dist_pixels / self.scale_factor
        
        # Calcular incertidumbre (aproximación: ±2 píxeles)
        uncertainty_px = 2.0
        uncertainty_cm = uncertainty_px / self.scale_factor
        uncertainty_percent = (uncertainty_cm / dist_cm) * 100
        
        self.measurements.append({
            'name': object_name,
            'points': (p1, p2),
            'distance_pixels': dist_pixels,
            'distance_cm': dist_cm,
            'uncertainty_cm': uncertainty_cm,
            'uncertainty_percent': uncertainty_percent,
            'is_calibration': False
        })
        
        print(f"✓ Medición registrada: {object_name}")
        print(f"  {dist_cm:.1f} ± {uncertainty_cm:.1f} cm ({uncertainty_percent:.1f}%)")
        
        self.points = []
        self.display_image = self.image.copy()
        self.redraw_measurements()
        
        return True
    
    def run(self):
        """Ejecuta la interfaz interactiva."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print("\n" + "="*80)
        print("HERRAMIENTA DE MEDICIÓN INTERACTIVA")
        print("="*80)
        print("\nInstrucciones:")
        print("  • Clic izquierdo: Marcar punto")
        print("  • Clic derecho: Cancelar medición actual")
        print("  • ESC: Salir y guardar")
        print("  • C: Calibrar con distancia conocida")
        print("  • M: Medir objeto")
        print("  • R: Reiniciar todo")
        print("  • S: Guardar imagen con mediciones")
        print("\nProcedimiento:")
        print("  1. Marca 2 puntos en un objeto de referencia conocido")
        print("  2. Presiona 'C' para calibrar")
        print("  3. Marca 2 puntos en objetos a medir")
        print("  4. Presiona 'M' para registrar medición")
        print("="*80 + "\n")
        
        cv2.imshow(self.window_name, self.display_image)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            
            elif key == ord('c') or key == ord('C'):
                # Calibrar
                if len(self.points) == 2:
                    print("\nIngresa la distancia real en cm:")
                    try:
                        dist = float(input("  Distancia (cm): "))
                        self.calibrate(dist)
                        cv2.imshow(self.window_name, self.display_image)
                    except ValueError:
                        print("[ERROR] Distancia inválida")
                else:
                    print("[ERROR] Marca 2 puntos primero")
            
            elif key == ord('m') or key == ord('M'):
                # Medir
                if len(self.points) == 2:
                    print("\nIngresa el nombre del objeto:")
                    name = input("  Nombre: ")
                    self.measure(name)
                    cv2.imshow(self.window_name, self.display_image)
                else:
                    print("[ERROR] Marca 2 puntos primero")
            
            elif key == ord('r') or key == ord('R'):
                # Reiniciar todo
                self.points = []
                self.measurements = []
                self.calibrated = False
                self.scale_factor = None
                self.display_image = self.image.copy()
                cv2.imshow(self.window_name, self.display_image)
                print("\n[INFO] Reiniciado")
            
            elif key == ord('s') or key == ord('S'):
                # Guardar imagen
                output_path = MEASUREMENTS_DIR / "mediciones_anotadas.jpg"
                cv2.imwrite(str(output_path), self.display_image)
                print(f"\n✓ Imagen guardada: {output_path}")
        
        cv2.destroyAllWindows()
        return self.measurements


def save_measurements_json(measurements: List[Dict], filename: str = "measurements.json"):
    """Guarda las mediciones en formato JSON."""
    output_path = MEASUREMENTS_DIR / filename
    
    # Convertir puntos a listas (no tuplas) para JSON
    measurements_serializable = []
    for m in measurements:
        m_copy = m.copy()
        m_copy['points'] = [list(p) for p in m['points']]
        measurements_serializable.append(m_copy)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(measurements_serializable, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Mediciones guardadas: {output_path}")


def generate_measurement_report(measurements: List[Dict]):
    """Genera un reporte de las mediciones."""
    report_path = MEASUREMENTS_DIR / "reporte_mediciones.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("REPORTE DE MEDICIONES - COMEDOR\n")
        f.write("Universidad Nacional de Colombia - Visión por Computador\n")
        f.write("="*80 + "\n\n")
        
        # Calibración
        f.write("CALIBRACIÓN:\n")
        f.write("-" * 40 + "\n")
        for m in measurements:
            if m.get('is_calibration', False):
                f.write(f"  Objeto de referencia: {m['name']}\n")
                f.write(f"  Distancia real: {m['distance_cm']:.1f} cm\n")
                f.write(f"  Distancia en píxeles: {m['distance_pixels']:.0f} px\n")
                scale = m['distance_pixels'] / m['distance_cm']
                f.write(f"  Escala: {scale:.2f} píxeles/cm\n")
        
        f.write("\n")
        
        # Mediciones
        f.write("MEDICIONES:\n")
        f.write("-" * 40 + "\n")
        for i, m in enumerate(measurements, 1):
            if not m.get('is_calibration', False):
                f.write(f"{i}. {m['name']}:\n")
                f.write(f"   Distancia: {m['distance_cm']:.1f} ± {m['uncertainty_cm']:.1f} cm\n")
                f.write(f"   Incertidumbre: {m['uncertainty_percent']:.1f}%\n")
                f.write(f"   Píxeles: {m['distance_pixels']:.0f} px\n")
                f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("ANÁLISIS DE INCERTIDUMBRE:\n")
        f.write("-" * 40 + "\n")
        f.write("Fuentes de error:\n")
        f.write("  • Error de marcación de puntos: ±2 píxeles\n")
        f.write("  • Distorsión de perspectiva: Variable\n")
        f.write("  • Error de calibración: Propagado a todas las mediciones\n")
        f.write("\n")
        f.write("Recomendaciones:\n")
        f.write("  • Usar objetos de referencia en el mismo plano que el objeto a medir\n")
        f.write("  • Marcar puntos con precisión (zoom si es necesario)\n")
        f.write("  • Realizar múltiples mediciones y promediar\n")
        f.write("="*80 + "\n")
    
    print(f"✓ Reporte generado: {report_path}")


def main():
    """Función principal."""
    print("\n" + "="*80)
    print("CALIBRACIÓN Y MEDICIÓN - COMEDOR")
    print("Universidad Nacional de Colombia - Visión por Computador")
    print("Trabajo 2: Registro de Imágenes - Parte 3")
    print("="*80)
    
    # Buscar panorama SIFT o ORB
    panorama_path = None
    for method in ['sift', 'orb']:
        path = RESULTS_DIR / f"panorama_{method}.jpg"
        if path.exists():
            panorama_path = path
            print(f"\n✓ Encontrado panorama: {panorama_path}")
            break
    
    if panorama_path is None:
        print("\n[ERROR] No se encontró ningún panorama.")
        print("Por favor, ejecuta primero 'python process_comedor.py'")
        return
    
    # Cargar panorama
    panorama = cv2.imread(str(panorama_path))
    if panorama is None:
        print(f"[ERROR] No se pudo cargar: {panorama_path}")
        return
    
    print(f"  Tamaño: {panorama.shape}")
    
    # Mostrar objetos de referencia
    print("\n" + "="*80)
    print("OBJETOS DE REFERENCIA CONOCIDOS:")
    print("="*80)
    print("  1. Cuadro de la Virgen de Guadalupe: 117 cm de altura")
    print("  2. Mesa: 161.1 cm de ancho")
    print("\nUSA UNO DE ESTOS OBJETOS PARA CALIBRAR\n")
    
    # Herramienta interactiva
    tool = InteractiveMeasurementTool(panorama, "Medición Comedor - UNAL")
    measurements = tool.run()
    
    if not measurements:
        print("\n[INFO] No se realizaron mediciones")
        return
    
    # Guardar resultados
    print("\n" + "="*80)
    print("GUARDANDO RESULTADOS")
    print("="*80 + "\n")
    
    save_measurements_json(measurements)
    generate_measurement_report(measurements)
    
    # Resumen
    print("\n" + "="*80)
    print("RESUMEN DE MEDICIONES")
    print("="*80)
    print(f"\nTotal de mediciones: {len([m for m in measurements if not m.get('is_calibration', False)])}")
    print(f"\nResultados guardados en: {MEASUREMENTS_DIR}")
    print("\nArchivos generados:")
    print("  • measurements.json - Datos en formato JSON")
    print("  • reporte_mediciones.txt - Reporte completo")
    print("  • mediciones_anotadas.jpg - Imagen con anotaciones")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
