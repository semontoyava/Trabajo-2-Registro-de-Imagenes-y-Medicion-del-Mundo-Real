# 📝 Guía de Publicación del Reporte Técnico

Este documento explica cómo publicar el reporte técnico como blog post en diferentes plataformas.

---

## 📋 Reporte Técnico

El reporte completo está en: **`REPORTE_TECNICO.md`**

Contiene:
- ✅ Introducción con contexto y motivación
- ✅ Marco teórico completo (SIFT, ORB, RANSAC, Homografía)
- ✅ Metodología detallada con diagramas de flujo
- ✅ Experimentos y resultados (Graf + Comedor)
- ✅ Análisis y discusión (comparación, errores, mejoras)
- ✅ Conclusiones
- ✅ 10 referencias académicas
- ✅ Análisis de contribución individual (100%)

---

## 🌐 Opciones de Publicación

### **Opción 1: GitHub Pages** (Recomendado) ⭐

**Ventajas:**
- Gratis, fácil, integrado con GitHub
- Markdown nativo (no necesita conversión)
- Versionado automático

**Pasos:**

1. **Activar GitHub Pages:**
   ```bash
   # En GitHub: Settings → Pages → Source: main branch
   ```

2. **Crear index.md:**
   ```bash
   cp REPORTE_TECNICO.md docs/index.md
   git add docs/index.md
   git commit -m "Add blog post"
   git push
   ```

3. **Acceder:**
   ```
   https://davidalondono.github.io/Trabajo-2-Registro-de-Imagenes-y-Medicion-del-Mundo-Real/
   ```

**Configuración adicional (opcional):**
```yaml
# Crear _config.yml
theme: jekyll-theme-cayman
title: Registro de Imágenes - UNAL
description: Trabajo de Visión por Computador
```

---

### **Opción 2: RPubs** (Para usuarios de R)

**Ventajas:**
- Especializado en contenido técnico/científico
- Fácil compartir con comunidad académica

**Pasos:**

1. **Instalar R y RStudio** (si no lo tienes)

2. **Crear R Markdown:**
   ```r
   # En RStudio: File → New File → R Markdown
   # Copiar contenido de REPORTE_TECNICO.md
   ```

3. **Publicar:**
   ```r
   # Botón "Publish" en RStudio
   # Seleccionar RPubs
   # Crear cuenta (gratis)
   ```

4. **URL:**
   ```
   https://rpubs.com/davidalondono/registro-imagenes-unal
   ```

---

### **Opción 3: Medium** (Mayor audiencia)

**Ventajas:**
- Plataforma popular, gran audiencia
- Buen diseño automático

**Pasos:**

1. **Convertir Markdown a HTML:**
   ```bash
   python generate_blog_post.py
   ```
   Genera: `REPORTE_TECNICO.html`

2. **Importar a Medium:**
   - Ir a: https://medium.com/new-story
   - Clic en "..." → "Import a story"
   - Subir `REPORTE_TECNICO.html`

3. **Ajustar formato:**
   - Agregar imágenes de resultados
   - Revisar código (Medium formatea automáticamente)
   - Añadir tags: `Computer Vision`, `Image Registration`, `Python`, `OpenCV`

4. **Publicar:**
   - Clic en "Publish"
   - Elegir audiencia y distribución

---

### **Opción 4: Observable** (Para contenido interactivo)

**Ventajas:**
- Notebooks interactivos en JavaScript
- Visualizaciones dinámicas

**Pasos:**

1. **Crear cuenta:** https://observablehq.com/

2. **Crear notebook:**
   - Clic en "New notebook"
   - Importar datos JSON de resultados

3. **Agregar celdas:**
   ```javascript
   // Cargar resultados
   graf_results = FileAttachment("graf_results.json").json()
   
   // Visualizar RMSE vs ángulo
   Plot.plot({
     marks: [
       Plot.line(graf_results, {x: "angle", y: "rmse"}),
       Plot.dot(graf_results, {x: "angle", y: "rmse"})
     ]
   })
   ```

4. **Publicar:**
   - Clic en "Publish"
   - Compartir URL

---

### **Opción 5: Blog Personal** (WordPress, Hugo, etc.)

**WordPress:**
```bash
# Instalar plugin Markdown
# Copiar contenido de REPORTE_TECNICO.md
# Publicar
```

**Hugo (Static Site Generator):**
```bash
# Instalar Hugo
hugo new site mi-blog
cd mi-blog

# Crear post
hugo new posts/registro-imagenes.md
# Copiar contenido de REPORTE_TECNICO.md

# Generar sitio
hugo

# Deploy a Netlify/Vercel (gratis)
```

---

## 📊 Incluir Figuras

Para todas las plataformas, necesitas incluir las imágenes de resultados:

```bash
# Copiar figuras a carpeta pública
cp results/graf_validation/figures/* docs/images/
cp results/comedor_registration/*.jpg docs/images/
cp results/measurements/*.jpg docs/images/
```

**Actualizar referencias en Markdown:**
```markdown
![Panorama SIFT](images/panorama_sift.jpg)
```

---

## ✅ Checklist de Publicación

Antes de publicar, verifica:

- [ ] Todas las secciones completas (intro, teoría, métodos, resultados, análisis, conclusiones, referencias)
- [ ] Imágenes incluidas y con buena resolución
- [ ] Tablas formateadas correctamente
- [ ] Código con syntax highlighting
- [ ] Referencias citadas correctamente
- [ ] Sin errores de ortografía/gramática
- [ ] Metadata correcta (título, autor, fecha)
- [ ] Enlaces funcionando
- [ ] Licencia especificada (si es GitHub Pages)

---

## 🎯 Recomendación Final

**Para este trabajo académico, recomiendo:**

1. **GitHub Pages** como plataforma principal
   - Integrado con el repositorio
   - Fácil de actualizar
   - Profesional

2. **Medium** como publicación secundaria
   - Mayor visibilidad
   - Comunidad técnica activa
   - Portfolio personal

**Pasos sugeridos:**

```bash
# 1. Configurar GitHub Pages
git checkout -b gh-pages
mkdir docs
cp REPORTE_TECNICO.md docs/index.md
cp -r results docs/
git add docs/
git commit -m "Add blog post for GitHub Pages"
git push origin gh-pages

# 2. Activar en GitHub: Settings → Pages → Source: gh-pages/docs

# 3. Publicar también en Medium para mayor alcance
python generate_blog_post.py
# Importar REPORTE_TECNICO.html a Medium
```

---

## 📚 Recursos Adicionales

**Markdown:**
- https://guides.github.com/features/mastering-markdown/

**GitHub Pages:**
- https://pages.github.com/

**Jekyll (para GitHub Pages):**
- https://jekyllrb.com/

**Medium:**
- https://help.medium.com/hc/en-us

---

**¿Dudas?** Contacta al profesor o consulta la documentación de cada plataforma.
