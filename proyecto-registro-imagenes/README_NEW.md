# 📸 Proyecto: Registro de Imágenes y Medición del Mundo Real

**Universidad Nacional de Colombia - Facultad de Minas**  
**Visión por Computador - 3009228**  
**Semestre 2025-02**  
**Autores:** David Londoño, Andrés Churio, Sebastián Montoya  
**Fecha:** Octubre 2025

---

## 🎯 Descripción del Proyecto

Este proyecto implementa un sistema completo de **registro de imágenes** (image registration) que permite:

1. ✅ **Validar algoritmos** con el dataset Graf (imágenes sintéticas con ground truth)
2. 📷 **Crear panoramas** fusionando múltiples vistas del mismo lugar
3. 📏 **Calibrar y medir** objetos del mundo real usando referencias conocidas

El caso de uso principal es la fusión de 3 imágenes de un comedor y la estimación de dimensiones de objetos utilizando dos referencias:
- 🖼️ Cuadro de la Virgen de Guadalupe: **117 cm** de altura
- 🪑 Mesa: **161.1 cm** de ancho

---

## 📁 Estructura del Proyecto

```
proyecto-registro-imagenes/
│
├── README.md                          # Este archivo
├── requirements.txt                   # Dependencias Python
├── .gitignore                        # Archivos ignorados por Git
│
├── download_and_process_graf.py      # Script Parte 1: Validación con Graf
├── process_comedor.py                # Script Parte 2: Registro del comedor
├── measure_comedor.py                # Script Parte 3: Calibración y medición
│
├── data/                             # Datasets
│   ├── original/                     # Imágenes originales (vacío)
│   ├── synthetic/                    # Imágenes sintéticas (vacío)
│   └── graf_dataset/                 # Dataset Graf descargado
│
├── Comedor/                          # Imágenes del comedor
│   ├── IMG01.jpg
│   ├── IMG02.jpg
│   └── IMG03.jpg
│
├── src/                              # Módulos del proyecto
│   ├── __init__.py
│   ├── feature_detection.py         # Detectores SIFT, ORB, AKAZE
│   ├── matching.py                  # Emparejamiento FLANN, BruteForce
│   ├── registration.py              # Homografía, RANSAC, warping
│   ├── panorama.py                  # Fusión de imágenes
│   ├── validation.py                # Métricas de error
│   └── utils.py                     # Utilidades y visualización
│
├── notebooks/                        # Análisis interactivo
│   ├── 01_exploratory_analysis.ipynb    # Análisis exploratorio
│   ├── 02_synthetic_validation.ipynb    # Validación con Graf
│   └── 03_main_pipeline.ipynb           # Pipeline completo
│
├── results/                          # Resultados generados
│   ├── graf_validation/             # Parte 1: Validación
│   │   ├── figures/                 # Visualizaciones
│   │   └── graf_results.json       # Métricas
│   ├── comedor_registration/        # Parte 2: Panoramas
│   │   ├── panorama_sift.jpg
│   │   ├── panorama_orb.jpg
│   │   └── comedor_results.json
│   └── measurements/                # Parte 3: Mediciones
│       ├── measurements.json
│       ├── reporte_mediciones.txt
│       └── mediciones_anotadas.jpg
│
└── tests/                            # Pruebas unitarias
    ├── __init__.py
    ├── test_feature_detection.py
    ├── test_matching.py
    └── test_registration.py
```

---

## 🚀 Instalación y Configuración

### Requisitos Previos

- **Python 3.8+**
- **pip** (gestor de paquetes)
- **Entorno virtual** (recomendado)

### 1. Clonar el repositorio

```bash
git clone https://github.com/DavidALondono/Trabajo-2-Registro-de-Imagenes-y-Medicion-del-Mundo-Real.git
cd Trabajo-2-Registro-de-Imagenes-y-Medicion-del-Mundo-Real
```

### 2. Crear y activar entorno virtual

**Windows (CMD):**
```cmd
python -m venv .venv
.venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar dependencias

```bash
cd proyecto-registro-imagenes
pip install --upgrade pip
pip install -r requirements.txt
```

**Dependencias principales:**
- `opencv-python>=4.8.0` - Procesamiento de imágenes
- `opencv-contrib-python>=4.8.0` - Algoritmos SIFT, SURF
- `numpy>=1.24.0` - Cálculos numéricos
- `matplotlib>=3.7.0` - Visualización
- `scipy>=1.10.0` - Análisis científico

---

## 📊 Ejecución del Proyecto

### **Parte 1: Validación con Dataset Graf** ✅

Valida los algoritmos usando imágenes sintéticas con transformaciones conocidas.

```bash
python download_and_process_graf.py
```

**Salidas:**
- `results/graf_validation/figures/` - Visualizaciones
- `results/graf_validation/graf_results.json` - Métricas (RMSE, error angular)

**Métricas esperadas:**
- ✅ RMSE < 2.0 píxeles
- ✅ Error angular < 1.5°
- ✅ Inliers > 85%

---

### **Parte 2: Registro del Comedor** 📷

Crea panoramas fusionando las 3 imágenes del comedor usando SIFT y ORB.

```bash
python process_comedor.py
```

**Salidas:**
- `results/comedor_registration/panorama_sift.jpg` - Panorama SIFT
- `results/comedor_registration/panorama_orb.jpg` - Panorama ORB
- `results/comedor_registration/comedor_results.json` - Estadísticas

**Comparación SIFT vs ORB:**

| Métrica | SIFT | ORB |
|---------|------|-----|
| Keypoints | 1500-3800 | 4800-5000 |
| Matches | 280+ | Variable |
| Inliers | 75%+ | 60%+ |
| Velocidad | Lento | Rápido |
| Precisión | Alta | Media |

---

### **Parte 3: Calibración y Medición** 📏

Herramienta interactiva para medir objetos usando referencias conocidas.

```bash
python measure_comedor.py
```

**Procedimiento:**

1. **Calibrar:**
   - Marcar 2 puntos en el cuadro (altura: 117 cm) o mesa (ancho: 161.1 cm)
   - Presionar `C` e ingresar la distancia real

2. **Medir objetos:**
   - Marcar 2 puntos en el objeto deseado
   - Presionar `M` e ingresar el nombre del objeto

**Controles:**
- `Clic izquierdo`: Marcar punto
- `Clic derecho`: Cancelar medición actual
- `C`: Calibrar con distancia conocida
- `M`: Medir objeto
- `R`: Reiniciar todo
- `S`: Guardar imagen con anotaciones
- `ESC`: Salir y guardar

**Salidas:**
- `results/measurements/measurements.json` - Datos JSON
- `results/measurements/reporte_mediciones.txt` - Reporte completo
- `results/measurements/mediciones_anotadas.jpg` - Imagen anotada

---

## 📓 Notebooks Interactivos

### 1. Análisis Exploratorio
```bash
jupyter notebook notebooks/01_exploratory_analysis.ipynb
```
Análisis de las imágenes, distribución de características, estadísticas básicas.

### 2. Validación Sintética
```bash
jupyter notebook notebooks/02_synthetic_validation.ipynb
```
Validación exhaustiva con el dataset Graf, comparación de métricas.

### 3. Pipeline Principal
```bash
jupyter notebook notebooks/03_main_pipeline.ipynb
```
Pipeline completo de registro, desde la carga hasta la medición.

---

## 🔬 Fundamentos Técnicos

### Detección de Características

#### **SIFT (Scale-Invariant Feature Transform)**
- Invariante a escala, rotación e iluminación
- Alta precisión en emparejamiento
- Uso: Cuando se requiere máxima calidad

#### **ORB (Oriented FAST and Rotated BRIEF)**
- Muy rápido (10x más que SIFT)
- Invariante a rotación
- Uso: Aplicaciones en tiempo real

### Emparejamiento

#### **FLANN (Fast Library for Approximate Nearest Neighbors)**
- Emparejamiento rápido para SIFT
- Usa árboles KD

#### **BruteForce con Hamming**
- Para descriptores binarios (ORB)
- Exhaustivo pero preciso

### Transformación Geométrica

#### **Homografía**
- Transformación proyectiva 3x3
- Relaciona puntos entre dos planos
- Estimada con RANSAC

#### **RANSAC (Random Sample Consensus)**
- Filtra outliers en emparejamiento
- Parámetros:
  - `ransacReprojThreshold`: 5.0 px
  - `maxIters`: 2000
  - `confidence`: 0.995

---

## 📈 Métricas de Evaluación

### Validación con Graf (Parte 1)
- **RMSE (Root Mean Square Error):** < 2.0 píxeles
- **Error Angular:** < 1.5 grados
- **Inlier Ratio:** > 85%

### Registro del Comedor (Parte 2)
- **Matches:** > 100 por par de imágenes
- **Inliers RANSAC:** > 60%
- **Calidad Visual:** Sin distorsiones evidentes

### Mediciones (Parte 3)
- **Incertidumbre:** 2-5% típica
- **Repetibilidad:** ±2-3 cm
- **Fuentes de error:**
  - Marcación de puntos: ±2 píxeles
  - Distorsión de perspectiva
  - Propagación del error de calibración

---

## 🧪 Pruebas Unitarias

```bash
pytest tests/
```

**Pruebas incluidas:**
- Detección de características
- Emparejamiento robusto
- Estimación de homografía
- Validación de transformaciones

---

## 🐛 Solución de Problemas

### Error: "No module named 'cv2'"
```bash
pip install opencv-python opencv-contrib-python --upgrade
```

### Error: "Muy pocos matches encontrados"
- Verificar solapamiento de imágenes (>30%)
- Ajustar ratio test (0.75 → 0.8)
- Usar SIFT en lugar de ORB

### Error: "No se pudo estimar homografía"
- Verificar textura suficiente en las imágenes
- Aumentar keypoints: `nfeatures=5000`
- Revisar que las imágenes sean de la misma escena

### Mediciones con alta incertidumbre
- Usar referencias en el mismo plano del objeto
- Marcar puntos con precisión (zoom)
- Realizar múltiples mediciones y promediar

---

## 📚 Referencias

1. **Lowe, D. G. (2004).** "Distinctive Image Features from Scale-Invariant Keypoints". *International Journal of Computer Vision*, 60(2), 91-110.

2. **Rublee, E., Rabaud, V., Konolige, K., & Bradski, G. (2011).** "ORB: An efficient alternative to SIFT or SURF". *IEEE International Conference on Computer Vision (ICCV)*.

3. **Fischler, M. A., & Bolles, R. C. (1981).** "Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography". *Communications of the ACM*, 24(6), 381-395.

4. **Mikolajczyk, K., & Schmid, C. (2005).** "A Performance Evaluation of Local Descriptors". *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 27(10), 1615-1630.

5. **Szeliski, R. (2010).** *Computer Vision: Algorithms and Applications*. Springer.

6. **Hartley, R., & Zisserman, A. (2004).** *Multiple View Geometry in Computer Vision* (2nd ed.). Cambridge University Press.

---

## 🤝 Contribución

Este es un proyecto académico grupal para el curso de Visión por Computador de la Universidad Nacional de Colombia.

**Autores:**
- **David Londoño** - Detección de características (SIFT, ORB), validación con dataset Graf, arquitectura del sistema
- **Andrés Churio** - Emparejamiento robusto (FLANN, BF), registro del comedor, análisis comparativo  
- **Sebastián Montoya** - Fusión de imágenes (panoramas), herramienta de medición interactiva, visualizaciones

### Distribución de tareas:
- **Investigación y diseño:** Colaborativo (33%/33%/33%)
- **Implementación:** Dividida por módulos según especialidad
- **Experimentación:** Cada autor lideró una parte (Graf/Comedor/Medición)
- **Documentación:** Colaborativa con revisión cruzada

---

## 📄 Licencia

Este proyecto es para uso académico en el curso de Visión por Computador de la Universidad Nacional de Colombia.

---

## 📞 Contacto

**David Londoño, Andrés Churio, Sebastián Montoya**  
Universidad Nacional de Colombia - Facultad de Minas  
Visión por Computador - 3009228  
Semestre 2025-02

---

## 🎓 Agradecimientos

- Prof. J por proporcionar las imágenes del comedor
- Universidad Nacional de Colombia - Departamento de Ciencias de la Computación y de la Decisión
- Oxford VGG por el dataset Graf de evaluación

---

**Última actualización:** Octubre 27, 2025
