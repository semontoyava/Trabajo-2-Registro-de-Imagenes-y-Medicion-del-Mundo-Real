@echo off
echo Limpiando archivos duplicados y carpetas innecesarias...
echo.

REM Eliminar setup duplicados DENTRO de proyecto-registro-imagenes
echo [1/7] Eliminando setup.bat duplicado de proyecto-registro-imagenes...
if exist "proyecto-registro-imagenes\setup.bat" (
    del /q "proyecto-registro-imagenes\setup.bat"
    echo   ✓ Eliminado
) else (
    echo   - No existe
)

echo [2/7] Eliminando setup.sh duplicado de proyecto-registro-imagenes...
if exist "proyecto-registro-imagenes\setup.sh" (
    del /q "proyecto-registro-imagenes\setup.sh"
    echo   ✓ Eliminado
)

echo [3/7] Eliminando setup.py duplicado de proyecto-registro-imagenes...
if exist "proyecto-registro-imagenes\setup.py" (
    del /q "proyecto-registro-imagenes\setup.py"
    echo   ✓ Eliminado
)

REM Eliminar carpeta duplicada proyecto-registro-imagenes/proyecto-registro-imagenes
echo [4/7] Eliminando carpeta duplicada proyecto-registro-imagenes/proyecto-registro-imagenes...
if exist "proyecto-registro-imagenes\proyecto-registro-imagenes" (
    rmdir /s /q "proyecto-registro-imagenes\proyecto-registro-imagenes"
    echo   ✓ Eliminada
)

REM Eliminar carpeta results duplicada
echo [5/7] Eliminando carpeta results duplicada de proyecto-registro-imagenes...
if exist "proyecto-registro-imagenes\results" (
    rmdir /s /q "proyecto-registro-imagenes\results"
    echo   ✓ Eliminada
)

REM Eliminar carpeta data duplicada de proyecto-registro-imagenes (si existe)
echo [6/7] Eliminando carpeta data duplicada de proyecto-registro-imagenes...
if exist "proyecto-registro-imagenes\data" (
    rmdir /s /q "proyecto-registro-imagenes\data"
    echo   ✓ Eliminada
)

REM Eliminar archivos innecesarios
echo [7/7] Eliminando archivos markdown duplicados...
if exist "proyecto-registro-imagenes\README.md" (
    del /q "proyecto-registro-imagenes\README.md"
    echo   ✓ README.md eliminado
)
if exist "proyecto-registro-imagenes\QUICKSTART.md" (
    del /q "proyecto-registro-imagenes\QUICKSTART.md"
    echo   ✓ QUICKSTART.md eliminado
)

echo.
echo ===============================================
echo Limpieza completada
echo ===============================================
echo.
echo Estructura final:
echo   - Archivos de configuracion: directorio raiz
echo   - Codigo fuente: proyecto-registro-imagenes/src/
echo   - Notebooks: proyecto-registro-imagenes/notebooks/
echo   - Scripts: proyecto-registro-imagenes/*.py
echo.
pause
