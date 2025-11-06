# 📸 Proyecto: Registro de Imágenes y Medición del Mundo Real

**Universidad Nacional de Colombia - Facultad de Minas**  
**Visión por Computador - 3009228**  
**Semestre 2025-02**  
**Autores:** David Londoño, Andrés Churio, Sebastián Montoya  
**Fecha:** Octubre 2025  

**Fecha:** Octubre 2025---



---## 👥 Autores



## 🎯 Descripción del Proyecto- **David A. Londoño** - Universidad Nacional de Colombia



Este proyecto implementa un sistema completo de **registro de imágenes** (image registration) que permite:---



1. ✅ **Validar algoritmos** con el dataset Graf o una imágen sintética con ground truth ## 🎯 Objetivo del Proyecto

2. 📷 **Crear panoramas** fusionando múltiples vistas del mismo lugar

3. 📏 **Calibrar y medir** objetos del mundo real usando referencias conocidasEste proyecto implementa un pipeline completo de **registro de imágenes** y **medición métrica** para reconstruir una vista panorámica del comedor de una casa a partir de múltiples fotografías con solapamiento. El objetivo es:



El caso de uso principal es la fusión de 3 imágenes de un comedor y la estimación de dimensiones de objetos utilizando dos referencias:1. **Validar** el algoritmo de registro mediante imágenes sintéticas con transformaciones conocidas.

- 🖼️ Cuadro de la Virgen de Guadalupe: **117 cm** de altura2. **Fusionar** imágenes reales aplicando detección de características, emparejamiento robusto y estimación de homografía.

- 🪑 Mesa: **161.1 cm** de ancho3. **Calibrar** la escala métrica usando objetos de referencia conocidos (cuadro de la Virgen de Guadalupe: 117 cm, mesa: 161.1 cm).

4. **Medir** dimensiones de objetos arbitrarios en la escena fusionada con una herramienta interactiva.

---

---

## 📁 Estructura del Proyecto

## 📋 Descripción General del Pipeline

```

proyecto-registro-imagenes/El pipeline metodológico del proyecto se divide en tres partes principales:

│

├── README.md                          # Este archivo### **Parte 1: Validación con Imágenes Sintéticas**

├── requirements.txt                   # Dependencias Python

├── .gitignore                        # Archivos ignorados por Git```

│Imagen Base → Aplicar Transformaciones → Detección de Características

├── download_and_process_graf.py      # Script Parte 1: Validación con Graf                (Rotación, Traslación,      (SIFT, ORB, AKAZE)

├── process_comedor.py                # Script Parte 2: Registro del comedor                 Escala)                            ↓

├── measure_comedor.py                # Script Parte 3: Calibración y medición                                            Emparejamiento Robusto

│                                            (FLANN + Lowe Ratio Test)

├── data/                             # Datasets                                                    ↓

│   ├── original/                     # Imágenes originales (vacío)                                            Estimar Homografía (RANSAC)

│   ├── synthetic/                    # Imágenes sintéticas (vacío)                                                    ↓

│   └── graf_dataset/                 # Dataset Graf descargado                                            Aplicar Registro

│                                                    ↓

├── Comedor/                          # Imágenes del comedor                                            Calcular Métricas de Error

│   ├── IMG01.jpg                                            (RMSE, Error Angular)

│   ├── IMG02.jpg```

│   └── IMG03.jpg

│**Objetivos:**

├── src/                              # Módulos del proyecto- Generar imágenes transformadas con parámetros conocidos

│   ├── __init__.py- Recuperar las transformaciones mediante el algoritmo de registro

│   ├── feature_detection.py         # Detectores SIFT, ORB, AKAZE- Evaluar precisión con métricas cuantitativas (RMSE, error angular)

│   ├── matching.py                  # Emparejamiento FLANN, BruteForce- Visualizar resultados antes/después del registro

│   ├── registration.py              # Homografía, RANSAC, warping

│   ├── panorama.py                  # Fusión de imágenes### **Parte 2: Registro de Imágenes Reales**

│   ├── validation.py                # Métricas de error

│   └── utils.py                     # Utilidades y visualización```

│Múltiples Imágenes → Detección de Características → Emparejamiento

├── notebooks/                        # Análisis interactivo  del Comedor            (SIFT, ORB, AKAZE)        (FLANN/BFMatcher)

│   ├── 01_exploratory_analysis.ipynb    # Análisis exploratorio                                                           ↓

│   ├── 02_synthetic_validation.ipynb    # Validación con Graf                                                    Filtrado por Ratio Test

│   └── 03_main_pipeline.ipynb           # Pipeline completo                                                           ↓

│                                                    Estimación de Homografía

├── results/                          # Resultados generados                                                    (RANSAC para outliers)

│   ├── graf_validation/             # Parte 1: Validación                                                           ↓

│   │   ├── figures/                 # Visualizaciones                                                    Warping + Blending

│   │   └── graf_results.json       # Métricas                                                    (Multi-band blending)

│   ├── comedor_registration/        # Parte 2: Panoramas                                                           ↓

│   │   ├── panorama_sift.jpg                                                    Imagen Panorámica

│   │   ├── panorama_orb.jpg```

│   │   └── comedor_results.json

│   └── measurements/                # Parte 3: Mediciones**Objetivos:**

│       ├── measurements.json- Implementar al menos 2 detectores de características (SIFT, ORB/AKAZE)

│       ├── reporte_mediciones.txt- Emparejar características con métodos robustos

│       └── mediciones_anotadas.jpg- Estimar homografía con RANSAC para eliminar outliers

│- Fusionar imágenes con blending suave

└── tests/                            # Pruebas unitarias

    ├── __init__.py### **Parte 3: Calibración y Medición**

    ├── test_feature_detection.py

    ├── test_matching.py```

    └── test_registration.pyImagen Fusionada → Identificar Objetos de Referencia → Calcular Escala

```                   (Cuadro: 117 cm, Mesa: 161.1 cm)    (píxeles → cm)

                                                              ↓

---                                                    Herramienta Interactiva

                                                    (Clics del mouse)

## 🚀 Instalación y Configuración                                                              ↓

                                                    Medir Objetos Adicionales

### Requisitos Previos                                                    (ventana, silla, planta)

                                                              ↓

- **Python 3.8+**                                                    Estimar Incertidumbre

- **pip** (gestor de paquetes)                                                    (Varianza, Error promedio)

- **Entorno virtual** (recomendado)```



### 1. Clonar el repositorio**Objetivos:**

- Establecer escala métrica usando dimensiones conocidas

```bash- Crear herramienta interactiva para medición con clics

git clone https://github.com/DavidALondono/Trabajo-2-Registro-de-Imagenes-y-Medicion-del-Mundo-Real.git- Estimar dimensiones de 3+ objetos adicionales

cd Trabajo-2-Registro-de-Imagenes-y-Medicion-del-Mundo-Real- Calcular incertidumbre de medición

```

---

### 2. Crear y activar entorno virtual

## 🚀 Instalación

**Windows (CMD):**

```cmd### Requisitos Previos

python -m venv .venv #py -m venv .venv

.venv\Scripts\activate- Python 3.8 o superior

```- pip (gestor de paquetes de Python)

- Git (opcional, para clonar el repositorio)

**Linux/Mac:**

```bash### Configuración en Windows

python3 -m venv .venv

source .venv/bin/activateEjecute los siguientes comandos en el terminal:

```

```bash

### 3. Instalar dependencias# 1. Navegar al directorio del proyecto

cd proyecto-registro-imagenes

```bash

cd proyecto-registro-imagenes# 2. Crear entorno virtual

pip install --upgrade pippython -m venv .venv

pip install -r requirements.txt

```# 3. Activar entorno virtual

.venv\Scripts\activate

**Dependencias principales:**

- `opencv-python>=4.8.0` - Procesamiento de imágenes# 4. Instalar dependencias

- `opencv-contrib-python>=4.8.0` - Algoritmos SIFT, SURFsetup

- `numpy>=1.24.0` - Cálculos numéricos```

- `matplotlib>=3.7.0` - Visualización

- `scipy>=1.10.0` - Análisis científico### Configuración en macOS y Linux

- `scikit-image>=0.20.0` - Procesamiento adicional

Ejecute los siguientes comandos en el terminal:

---

```bash

## 📊 Ejecución del Proyecto# 1. Navegar al directorio del proyecto

cd proyecto-registro-imagenes

### **Parte 1: Validación con Dataset Graf** ✅

# 2. Crear entorno virtual

Valida los algoritmos usando imágenes sintéticas con transformaciones conocidas.python3 -m venv .venv



```bash# 3. Activar entorno virtual

python download_and_process_graf.pysource .venv/bin/activate

```

# 4. Dar permisos de ejecución y ejecutar script

**Salidas:**chmod +x setup.sh

- `results/graf_validation/figures/` - Visualizacionessource setup.sh

- `results/graf_validation/graf_results.json` - Métricas (RMSE, error angular)```



**Métricas esperadas:**### Scripts de Configuración

- ✅ RMSE < 2.0 píxeles

- ✅ Error angular < 1.5°Los siguientes scripts están disponibles para instalar las dependencias:

- ✅ Inliers > 85%

- **`setup.bat`** (Windows): Instala pip actualizado y todas las dependencias

---- **`setup.sh`** (macOS/Linux): Instala pip actualizado y todas las dependencias  

- **`setup.py`**: Configuración de setuptools con todas las dependencias del proyecto

### **Parte 2: Registro del Comedor** 📷

**Contenido de setup.bat:**

Crea panoramas fusionando las 3 imágenes del comedor usando SIFT y ORB.```batch

@echo off

```bash

python process_comedor.pypython -m pip install --upgrade pip

```python -m pip install -r requirements.txt

```

**Salidas:**

- `results/comedor_registration/panorama_sift.jpg` - Panorama SIFT**Contenido de setup.sh:**

- `results/comedor_registration/panorama_orb.jpg` - Panorama ORB```bash

- `results/comedor_registration/comedor_results.json` - Estadísticas#!/bin/bash



**Comparación SIFT vs ORB:**python -m pip install --upgrade pip

python -m pip install -r requirements.txt

| Métrica | SIFT | ORB |```

|---------|------|-----|

| Keypoints | 1500-3800 | 4800-5000 |### Verificación de la Instalación

| Matches | 280+ | Variable |

| Inliers | 75%+ | 60%+ |Después de completar la instalación, verifica que todo funciona correctamente:

| Velocidad | Lento | Rápido |

| Precisión | Alta | Media |```bash

# Verificar que el entorno virtual está activo

---# Deberías ver (.venv) al inicio de tu línea de comando



### **Parte 3: Calibración y Medición** 📏# Verificar instalación de OpenCV

python -c "import cv2; print(f'OpenCV {cv2.__version__} instalado correctamente')"

Herramienta interactiva para medir objetos usando referencias conocidas.

# Verificar módulos principales

```bashpython -c "from src import feature_detection, matching, registration, measurement, utils; print('✓ Todos los módulos importados correctamente')"

python measure_comedor.py```

```

### Desactivar el Entorno Virtual

**Procedimiento:**

Cuando termines de trabajar en el proyecto:

1. **Calibrar:**

   - Marcar 2 puntos en el cuadro (altura: 117 cm) o mesa (ancho: 161.1 cm)```bash

   - Presionar `C` e ingresar la distancia realdeactivate

```

2. **Medir objetos:**

   - Marcar 2 puntos en el objeto deseado### Problemas Comunes

   - Presionar `M` e ingresar el nombre del objeto

**Problema:** `python: command not found`  

**Controles:****Solución:** Instala Python desde [python.org](https://www.python.org/) o usa `python3` en lugar de `python`

- `Clic izquierdo`: Marcar punto

- `Clic derecho`: Cancelar medición actual**Problema:** `pip: command not found`  

- `C`: Calibrar con distancia conocida**Solución:** Instala pip ejecutando `python -m ensurepip --upgrade`

- `M`: Medir objeto

- `R`: Reiniciar todo**Problema:** OpenCV no se instala correctamente  

- `S`: Guardar imagen con anotaciones**Solución:** Reinstala con `pip install --force-reinstall opencv-python opencv-contrib-python`

- `ESC`: Salir y guardar

**Problema:** Error al crear el entorno virtual  

**Salidas:****Solución:** Asegúrate de tener Python 3.8 o superior instalado con `python --version`

- `results/measurements/measurements.json` - Datos JSON

- `results/measurements/reporte_mediciones.txt` - Reporte completo---

- `results/measurements/mediciones_anotadas.jpg` - Imagen anotada

## 📦 Estructura del Proyecto

---

```

## 📓 Notebooks Interactivosproyecto-registro-imagenes/

├── README.md                           # Este archivo

### 1. Análisis Exploratorio├── requirements.txt                    # Dependencias del proyecto

```bash├── data/

jupyter notebook notebooks/01_exploratory_analysis.ipynb│   ├── original/                       # Imágenes reales del comedor

```│   └── synthetic/                      # Imágenes sintéticas para validación

Análisis de las imágenes, distribución de características, estadísticas básicas.├── src/

│   ├── feature_detection.py           # Detección de características (SIFT, ORB, AKAZE)

### 2. Validación Sintética│   ├── matching.py                    # Emparejamiento de descriptores

```bash│   ├── registration.py                # Cálculo de homografía y warping

jupyter notebook notebooks/02_synthetic_validation.ipynb│   ├── measurement.py                 # Herramienta de medición interactiva

```│   └── utils.py                       # Funciones auxiliares

Validación exhaustiva con el dataset Graf, comparación de métricas.├── notebooks/

│   ├── 01_exploratory_analysis.ipynb  # Análisis exploratorio de datos

### 3. Pipeline Principal│   ├── 02_synthetic_validation.ipynb  # Validación con imágenes sintéticas

```bash│   └── 03_main_pipeline.ipynb         # Pipeline completo de registro

jupyter notebook notebooks/03_main_pipeline.ipynb├── results/

```│   ├── figures/                       # Visualizaciones y gráficos

Pipeline completo de registro, desde la carga hasta la medición.│   └── measurements/                  # Resultados de mediciones

└── tests/

---    └── test_basic.py                  # Pruebas unitarias básicas

```

## 🔬 Fundamentos Técnicos

---

### Detección de Características

## 💻 Ejemplos de Ejecución

#### **SIFT (Scale-Invariant Feature Transform)**

- Invariante a escala, rotación e iluminación### 1. Validación con Imágenes Sintéticas

- Alta precisión en emparejamiento

- Uso: Cuando se requiere máxima calidad```python

from src.utils import generate_synthetic_image, visualize_registration

#### **ORB (Oriented FAST and Rotated BRIEF)**from src.feature_detection import detect_sift_features

- Muy rápido (10x más que SIFT)from src.matching import match_features

- Invariante a rotaciónfrom src.registration import estimate_homography, warp_image

- Uso: Aplicaciones en tiempo realfrom src.utils import compute_registration_metrics



### Emparejamiento# Generar imagen sintética transformada

img_base = cv2.imread('data/original/base.jpg')

#### **FLANN (Fast Library for Approximate Nearest Neighbors)**img_transformed, true_matrix = generate_synthetic_image(

- Emparejamiento rápido para SIFT    img_base, 

- Usa árboles KD    rotation=30, 

    translation=(50, 30), 

#### **BruteForce con Hamming**    scale=1.2

- Para descriptores binarios (ORB))

- Exhaustivo pero preciso

# Detectar características

### Transformación Geométricakp1, desc1 = detect_sift_features(img_base)

kp2, desc2 = detect_sift_features(img_transformed)

#### **Homografía**

- Transformación proyectiva 3x3# Emparejar características

- Relaciona puntos entre dos planosmatches = match_features(desc1, desc2, method='flann')

- Estimada con RANSAC

# Estimar homografía

#### **RANSAC (Random Sample Consensus)**H, mask = estimate_homography(kp1, kp2, matches)

- Filtra outliers en emparejamiento

- Parámetros:# Aplicar registro

  - `ransacReprojThreshold`: 5.0 pximg_registered = warp_image(img_transformed, H, img_base.shape)

  - `maxIters`: 2000

  - `confidence`: 0.995# Calcular métricas

metrics = compute_registration_metrics(true_matrix, H)

---print(f"RMSE: {metrics['rmse']:.2f}, Error Angular: {metrics['angular_error']:.2f}°")



## 📈 Métricas de Evaluación# Visualizar resultados

visualize_registration(img_base, img_transformed, img_registered)

### Validación con Graf (Parte 1)```

- **RMSE (Root Mean Square Error):** < 2.0 píxeles

- **Error Angular:** < 1.5 grados### 2. Registro de Imágenes Reales

- **Inlier Ratio:** > 85%

```python

### Registro del Comedor (Parte 2)from src.feature_detection import detect_sift_features, detect_orb_features

- **Matches:** > 100 por par de imágenesfrom src.matching import match_features

- **Inliers RANSAC:** > 60%from src.registration import register_images, blend_images

- **Calidad Visual:** Sin distorsiones evidentes

# Cargar imágenes

### Mediciones (Parte 3)img1 = cv2.imread('data/original/comedor_1.jpg')

- **Incertidumbre:** 2-5% típicaimg2 = cv2.imread('data/original/comedor_2.jpg')

- **Repetibilidad:** ±2-3 cm

- **Fuentes de error:**# Detectar características con SIFT

  - Marcación de puntos: ±2 píxeleskp1, desc1 = detect_sift_features(img1)

  - Distorsión de perspectivakp2, desc2 = detect_sift_features(img2)

  - Propagación del error de calibración

# Emparejar características

---matches = match_features(desc1, desc2, method='flann', ratio_test=0.75)



## 🧪 Pruebas Unitarias# Registrar y fusionar imágenes

panorama = register_images(img1, img2, kp1, kp2, matches)

```bashpanorama_blended = blend_images(img1, img2, panorama)

pytest tests/

```# Guardar resultado

cv2.imwrite('results/figures/panorama.jpg', panorama_blended)

**Pruebas incluidas:**```

- Detección de características

- Emparejamiento robusto### 3. Calibración y Medición

- Estimación de homografía

- Validación de transformaciones```python

from src.measurement import MeasurementTool

---

# Cargar imagen fusionada

## 🐛 Solución de Problemaspanorama = cv2.imread('results/figures/panorama.jpg')



### Error: "No module named 'cv2'"# Crear herramienta de medición

```bashtool = MeasurementTool(panorama)

pip install opencv-python opencv-contrib-python --upgrade

```# Calibrar escala con objeto de referencia

# Cuadro de la Virgen de Guadalupe: 117 cm de altura

### Error: "Muy pocos matches encontrados"tool.calibrate_scale(

- Verificar solapamiento de imágenes (>30%)    reference_object='Cuadro Virgen de Guadalupe',

- Ajustar ratio test (0.75 → 0.8)    real_dimension_cm=117.0

- Usar SIFT en lugar de ORB)



### Error: "No se pudo estimar homografía"# Modo interactivo: medir objetos con clics

- Verificar textura suficiente en las imágenestool.measure_interactive()

- Aumentar keypoints: `nfeatures=5000`

- Revisar que las imágenes sean de la misma escena# Medir objetos específicos

distances = tool.measure_objects([

### Mediciones con alta incertidumbre    'ventana',

- Usar referencias en el mismo plano del objeto    'silla',

- Marcar puntos con precisión (zoom)    'planta'

- Realizar múltiples mediciones y promediar])



---# Calcular incertidumbre

uncertainty = tool.compute_uncertainty()

## 📚 Referenciasprint(f"Incertidumbre promedio: {uncertainty['mean']:.2f} cm")



1. **Lowe, D. G. (2004).** "Distinctive Image Features from Scale-Invariant Keypoints". *International Journal of Computer Vision*, 60(2), 91-110.# Guardar resultados

tool.save_measurements('results/measurements/mediciones.json')

2. **Rublee, E., Rabaud, V., Konolige, K., & Bradski, G. (2011).** "ORB: An efficient alternative to SIFT or SURF". *IEEE International Conference on Computer Vision (ICCV)*.```



3. **Fischler, M. A., & Bolles, R. C. (1981).** "Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography". *Communications of the ACM*, 24(6), 381-395.### 4. Ejecución Completa con Notebooks



4. **Mikolajczyk, K., & Schmid, C. (2005).** "A Performance Evaluation of Local Descriptors". *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 27(10), 1615-1630.Para una ejecución interactiva y detallada, utiliza los notebooks Jupyter:



5. **Szeliski, R. (2010).** *Computer Vision: Algorithms and Applications*. Springer.```bash

# Iniciar Jupyter

6. **Hartley, R., & Zisserman, A. (2004).** *Multiple View Geometry in Computer Vision* (2nd ed.). Cambridge University Press.jupyter notebook



---# Abrir notebooks en orden:

# 1. notebooks/01_exploratory_analysis.ipynb

## 🤝 Contribución# 2. notebooks/02_synthetic_validation.ipynb

# 3. notebooks/03_main_pipeline.ipynb

Este es un proyecto académico individual para el curso de Visión por Computador de la Universidad Nacional de Colombia.```



**Autor:** David A. Londoño  ---

**Contribución:** 100%

## 📊 Resultados Esperados

### Tareas realizadas:

- Implementación completa del pipeline de registro### Parte 1: Validación con Imágenes Sintéticas

- Validación con dataset Graf

- Herramienta de medición interactiva| Transformación | RMSE (píxeles) | Error Angular (°) | Tiempo (s) |

- Documentación y análisis|----------------|----------------|-------------------|------------|

- Pruebas y optimización| Rotación 15°   | < 1.0          | < 0.5             | ~0.5       |

| Traslación 50px| < 0.5          | < 0.1             | ~0.4       |

---| Escala 1.2x    | < 1.5          | < 1.0             | ~0.6       |

| Combinada      | < 2.0          | < 1.5             | ~0.8       |

## 📄 Licencia

### Parte 2: Registro de Imágenes Reales

Este proyecto es para uso académico en el curso de Visión por Computador de la Universidad Nacional de Colombia.

- **Características detectadas:** 1000-5000 por imagen (SIFT)

---- **Matches robustos:** 200-800 después de ratio test y RANSAC

- **Inliers RANSAC:** > 80% de matches

## 📞 Contacto- **Calidad visual:** Fusión suave sin artefactos visibles



**David A. Londoño**  ### Parte 3: Calibración y Medición

Universidad Nacional de Colombia - Facultad de Minas  

Visión por Computador - 3009228  | Objeto                    | Dimensión Real | Medición Estimada | Error (%) |

Semestre 2025-02|---------------------------|----------------|-------------------|-----------|

| Cuadro Virgen (altura)    | 117.0 cm       | 117.0 cm (ref)    | 0.0       |

---| Mesa (ancho)              | 161.1 cm       | 161.1 cm (ref)    | 0.0       |

| Ventana (altura)          | ~ 180 cm       | 178.5 ± 3.2 cm    | 0.8       |

## 🎓 Agradecimientos| Silla (altura)            | ~ 90 cm        | 88.7 ± 2.5 cm     | 1.4       |

| Planta (altura)           | ~ 45 cm        | 44.2 ± 1.8 cm     | 1.8       |

- Prof. J por proporcionar las imágenes del comedor

- Universidad Nacional de Colombia - Departamento de Ciencias de la Computación y de la Decisión**Incertidumbre promedio:** ±2.5 cm

- Oxford VGG por el dataset Graf de evaluación

---

---

## 🧪 Pruebas Unitarias

**Última actualización:** Octubre 27, 2025

Ejecutar pruebas:

```bash
# Ejecutar todas las pruebas
python -m pytest tests/

# Ejecutar pruebas con cobertura
python -m pytest tests/ --cov=src --cov-report=html
```

---

## 📚 Referencias Académicas

1. **Lowe, D. G. (2004)**. "Distinctive Image Features from Scale-Invariant Keypoints". *International Journal of Computer Vision*, 60(2), 91-110.  
   DOI: [10.1023/B:VISI.0000029664.99615.94](https://doi.org/10.1023/B:VISI.0000029664.99615.94)

2. **Rublee, E., Rabaud, V., Konolige, K., & Bradski, G. (2011)**. "ORB: An efficient alternative to SIFT or SURF". *IEEE International Conference on Computer Vision (ICCV)*, 2564-2571.  
   DOI: [10.1109/ICCV.2011.6126544](https://doi.org/10.1109/ICCV.2011.6126544)

3. **Fischler, M. A., & Bolles, R. C. (1981)**. "Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography". *Communications of the ACM*, 24(6), 381-395.  
   DOI: [10.1145/358669.358692](https://doi.org/10.1145/358669.358692)

4. **Szeliski, R. (2010)**. *Computer Vision: Algorithms and Applications*. Springer. Chapter 9: Image Stitching.  
   ISBN: 978-1-84882-935-0

5. **Brown, M., & Lowe, D. G. (2007)**. "Automatic Panoramic Image Stitching using Invariant Features". *International Journal of Computer Vision*, 74(1), 59-73.  
   DOI: [10.1007/s11263-006-0002-3](https://doi.org/10.1007/s11263-006-0002-3)

6. **Bradski, G., & Kaehler, A. (2008)**. *Learning OpenCV: Computer Vision with the OpenCV Library*. O'Reilly Media.  
   ISBN: 978-0-596-51613-0

7. **Alcantarilla, P. F., Bartoli, A., & Davison, A. J. (2012)**. "KAZE Features". *European Conference on Computer Vision (ECCV)*, 214-227.  
   DOI: [10.1007/978-3-642-33783-3_16](https://doi.org/10.1007/978-3-642-33783-3_16)

---

## 📝 Blog Técnico

Para una descripción detallada del proceso, resultados y análisis, consulta el blog técnico del proyecto:

**🔗 [Blog Técnico del Proyecto](https://github.com/DavidALondono/Trabajo-2-Registro-de-Imagenes-y-Medicion-del-Mundo-Real/wiki)**

*(Pendiente de publicación)*

---

## 📄 Licencia

Este proyecto es parte del trabajo académico para el curso de Visión por Computador de la Universidad Nacional de Colombia.

---

## 📧 Contacto

Para preguntas o comentarios sobre el proyecto:

- **Autor:** David A. Londoño
- **Institución:** Universidad Nacional de Colombia
- **Curso:** Visión por Computador

---

## 🙏 Agradecimientos

- Profesores del curso de Visión por Computador - Universidad Nacional de Colombia
- Comunidad de OpenCV por la documentación y ejemplos
- Autores de las referencias académicas citadas

---

**Última actualización:** Octubre 2025
