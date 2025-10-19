# predictor/__init__.py
"""
Paquete predictor - Agentes de predicción financiera
"""

import os
import sys

# Añadir el directorio padre al path para que todos los módulos
# del predictor puedan importar correctamente
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Añadir el directorio del predictor al path
predictor_dir = os.path.dirname(__file__)
if predictor_dir not in sys.path:
    sys.path.insert(0, predictor_dir)

# Importar componentes principales del módulo
try:
    from predictor.main_predictor import PredictorOrchestrator
    __all__ = ['PredictorOrchestrator']
except ImportError as e:
    print(f"⚠️ Error importando PredictorOrchestrator: {e}")
    __all__ = []
