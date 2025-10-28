"""
generate_blog_post.py

Genera versión HTML del reporte técnico para publicación en RPubs/GitHub Pages.

Uso:
    python generate_blog_post.py
"""

import markdown
from pathlib import Path

def generate_html_blog():
    """Convierte el reporte Markdown a HTML estilizado."""
    
    # Leer reporte markdown
    md_path = Path("REPORTE_TECNICO.md")
    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convertir a HTML
    html_content = markdown.markdown(
        md_content,
        extensions=[
            'extra',
            'codehilite',
            'toc',
            'tables',
            'fenced_code',
            'sane_lists'
        ]
    )
    
    # Template HTML con estilos
    html_template = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Registro de Imágenes y Medición del Mundo Real - UNAL</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.8;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-radius: 8px;
        }
        
        h1 {
            color: #1a1a1a;
            font-size: 2.5em;
            margin-bottom: 20px;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }
        
        h2 {
            color: #2c3e50;
            font-size: 2em;
            margin-top: 40px;
            margin-bottom: 20px;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 10px;
        }
        
        h3 {
            color: #34495e;
            font-size: 1.5em;
            margin-top: 30px;
            margin-bottom: 15px;
        }
        
        h4 {
            color: #546e7a;
            font-size: 1.2em;
            margin-top: 20px;
            margin-bottom: 10px;
        }
        
        p {
            margin-bottom: 15px;
            text-align: justify;
        }
        
        ul, ol {
            margin-left: 30px;
            margin-bottom: 15px;
        }
        
        li {
            margin-bottom: 8px;
        }
        
        code {
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            color: #c7254e;
        }
        
        pre {
            background: #2d2d2d;
            color: #f8f8f2;
            padding: 20px;
            border-radius: 5px;
            overflow-x: auto;
            margin-bottom: 20px;
        }
        
        pre code {
            background: none;
            color: #f8f8f2;
            padding: 0;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        th {
            background: #4CAF50;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }
        
        td {
            padding: 10px;
            border-bottom: 1px solid #e0e0e0;
        }
        
        tr:nth-child(even) {
            background: #f9f9f9;
        }
        
        tr:hover {
            background: #f5f5f5;
        }
        
        blockquote {
            border-left: 4px solid #4CAF50;
            padding-left: 20px;
            margin: 20px 0;
            color: #666;
            font-style: italic;
        }
        
        .metadata {
            background: #e8f5e9;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 30px;
        }
        
        .metadata p {
            margin-bottom: 5px;
        }
        
        .toc {
            background: #f5f5f5;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 30px;
        }
        
        .toc h2 {
            margin-top: 0;
            font-size: 1.5em;
        }
        
        .footer {
            margin-top: 50px;
            padding-top: 20px;
            border-top: 2px solid #e0e0e0;
            text-align: center;
            color: #666;
        }
        
        .badge {
            display: inline-block;
            background: #4CAF50;
            color: white;
            padding: 5px 10px;
            border-radius: 3px;
            font-size: 0.9em;
            margin: 5px 5px 5px 0;
        }
        
        .alert {
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 5px;
        }
        
        .alert-success {
            background: #d4edda;
            border-left: 4px solid #28a745;
            color: #155724;
        }
        
        .alert-info {
            background: #d1ecf1;
            border-left: 4px solid #17a2b8;
            color: #0c5460;
        }
        
        .alert-warning {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            color: #856404;
        }
        
        img {
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            margin: 20px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 20px;
            }
            
            h1 {
                font-size: 2em;
            }
            
            h2 {
                font-size: 1.5em;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        {content}
        
        <div class="footer">
            <p><strong>David A. Londoño</strong></p>
            <p>Universidad Nacional de Colombia - Facultad de Minas</p>
            <p>Visión por Computador - 3009228</p>
            <p>Octubre 2025</p>
        </div>
    </div>
</body>
</html>
"""
    
    # Generar HTML final
    final_html = html_template.format(content=html_content)
    
    # Guardar
    output_path = Path("REPORTE_TECNICO.html")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_html)
    
    print(f"✓ Reporte HTML generado: {output_path}")
    print("\nPara publicar:")
    print("  • RPubs: https://rpubs.com/ (subir HTML)")
    print("  • GitHub Pages: Commit y push al repo")
    print("  • Medium: Copiar contenido del HTML")
    
    return output_path


if __name__ == "__main__":
    generate_html_blog()
