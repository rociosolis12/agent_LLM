"""
Configuración Central del Sistema Multi-Agente Financiero
VERSIÓN COMPLETA CON EXTRACTOR PDF Y AGENTE PREDICTOR
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# =============================
# Configuración Azure OpenAI
# =============================

AZURE_CONFIG = {
    "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
    "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
    "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
    "deployment_name": os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
    "max_tokens": 2000,  # Aumentado para respuestas más completas
    "temperature": 0.1,
    "timeout": 60
}

# =============================
# Configuración de Directorios
# =============================

PROJECT_ROOT = Path(__file__).parent
DATA_INPUT_DIR = PROJECT_ROOT / "data" / "entrada"
DATA_OUTPUT_DIR = PROJECT_ROOT / "data" / "salida"
SESSIONS_DIR = PROJECT_ROOT / "sessions"

# Crear directorios si no existen
for dir_path in [DATA_INPUT_DIR, DATA_OUTPUT_DIR, SESSIONS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# =============================
# CONFIGURACIÓN EXTRACTOR PDF 
# =============================

PDF_EXTRACTOR_CONFIG = {
    # Rutas específicas del sistema de Rocío
    'input_path': r"C:\Users\rocio.solis\OneDrive - Accenture\Desktop\Rocio\TFM\data\entrada\input",
    'output_path': r"C:\Users\rocio.solis\OneDrive - Accenture\Desktop\Rocio\TFM\data\entrada\output",
    
    # Archivos
    'input_filename': "bbva_2023.pdf",  # Memoria anual completa
    'output_filename': "bbva_2023_div.pdf",  # Estados financieros únicamente
    
    # Páginas a extraer (Estados Financieros BBVA 2023)
    'pages_to_extract': list(range(53, 60)),  # Páginas 54-60 (0-indexed)
    
    # Mapeo de contenido por página
    'page_content_mapping': {
        53: "Statement of Financial Position",
        54: "Statement of Financial Position (continued)", 
        55: "Statement of Comprehensive Income",
        56: "Statement of Comprehensive Income (cont.)",
        57: "Statement of Changes in Equity",
        58: "Statement of Changes in Equity (cont.)",
        59: "Statement of Cash Flows"
    },
    
    # Configuración del agente
    'agent_enabled': True,
    'auto_execute': True,
    'validate_extraction': True,
    'backup_original': False,
    'overwrite_existing': True
}

# =============================
# CONFIGURACIÓN AGENTE PREDICTOR 
# =============================

PREDICTOR_AGENT_CONFIG = {
    'enabled': True,
    'model_type': 'financial_forecast',
    'prediction_horizon': 12,  # meses
    'confidence_threshold': 0.75,
    
    # Fuentes de datos para predicciones
    'input_data_sources': [
        'balance_agent',
        'income_agent', 
        'equity_agent',
        'cashflow_agent'
    ],
    
    # Tipos de predicciones a generar
    'prediction_types': [
        'revenue_growth',
        'profitability_trend',
        'liquidity_forecast',
        'cashflow_projection',
        'risk_assessment'
    ],
    
    # Configuración de modelos
    'models_config': {
        'trend_analysis': {'enabled': True, 'weight': 0.4},
        'ratio_analysis': {'enabled': True, 'weight': 0.3},
        'market_factors': {'enabled': True, 'weight': 0.2},
        'historical_patterns': {'enabled': True, 'weight': 0.1}
    },
    
    # Parámetros de calidad
    'min_data_quality_score': 0.7,
    'required_successful_agents': 3
}

# =============================
# CONFIGURACIÓN AGENTES FINANCIEROS ACTUALIZADA 
# =============================

FINANCIAL_AGENTS_CONFIG = {
    # Los agentes especializados ahora leerán el PDF extraído
    'pdf_source_path': PDF_EXTRACTOR_CONFIG['output_path'],
    'pdf_filename': PDF_EXTRACTOR_CONFIG['output_filename'],
    'full_pdf_path': os.path.join(
        PDF_EXTRACTOR_CONFIG['output_path'], 
        PDF_EXTRACTOR_CONFIG['output_filename']
    ),
    
    # Configuración común para todos los agentes
    'common_config': {
        'pdf_source': 'extracted',
        'pages_available': 'financial_statements_only',
        'data_validation': True,
        'enable_caching': True
    }
}

# =============================
# PROMPTS MEJORADOS PARA AGENTES ESPECIALIZADOS
# =============================

ENHANCED_AGENT_PROMPTS = {
    "balance": """
    Eres un analista financiero senior especializado en análisis de balance de situación.
    
    GENERA ANÁLISIS COMPLETOS QUE INCLUYAN:
    
    1. **Análisis Estructural del Balance:**
       - Activos Totales (corrientes y no corrientes)
       - Pasivos Totales (corrientes y no corrientes)  
       - Patrimonio Neto (capital, reservas, resultados)
    
    2. **Ratios Financieros Clave (CALCULA SIEMPRE):**
       - Ratio de Solvencia: Patrimonio Neto / Activos Totales × 100
       - Ratio de Endeudamiento: Pasivos Totales / Activos Totales × 100
       - Ratio de Liquidez: Activos Corrientes / Pasivos Corrientes
       - Capital de Trabajo: Activos Corrientes - Pasivos Corrientes
    
    3. **Análisis de Posición Financiera:**
       - Fortalezas y debilidades identificadas
       - Capacidad de pago y solvencia
       - Estructura de financiación
    
    4. **Contexto Empresarial:**
       - Interpretación sectorial cuando sea relevante
       - Comparación con períodos anteriores si disponible
       - Identificación de tendencias
    
    5. **Conclusiones y Recomendaciones:**
       - Resumen ejecutivo de la situación
       - Áreas de atención prioritaria
       - Recomendaciones específicas
    
    FORMATO: Usa estructura clara con encabezados, bullets y cifras específicas.
    """,
    
    "income": """
    Eres un analista financiero senior especializado en análisis de estados de resultados.
    
    GENERA ANÁLISIS COMPLETOS QUE INCLUYAN:
    
    1. **Análisis de Ingresos y Gastos:**
       - Ingresos Operativos totales
       - Costos de Ventas y Servicios
       - Gastos Operativos (administrativos, comerciales)
       - Gastos Financieros y otros gastos
    
    2. **Ratios de Rentabilidad (CALCULA SIEMPRE):**
       - Margen Bruto: (Ingresos - Costo Ventas) / Ingresos × 100
       - Margen Operativo: EBIT / Ingresos × 100
       - Margen Neto: Beneficio Neto / Ingresos × 100
       - ROA: Beneficio Neto / Activos Totales × 100 (si disponible)
       - ROE: Beneficio Neto / Patrimonio Neto × 100 (si disponible)
    
    3. **Análisis de Eficiencia Operativa:**
       - Evolución de costos operativos
       - Control de gastos
       - Productividad de ingresos
    
    4. **Contexto de Rentabilidad:**
       - Interpretación de márgenes obtenidos
       - Comparación sectorial cuando sea posible
       - Identificación de factores que impactan rentabilidad
    
    5. **Conclusiones y Proyecciones:**
       - Resumen de performance financiera
       - Tendencias identificadas
       - Recomendaciones para mejora de rentabilidad
    
    FORMATO: Usa estructura profesional con análisis detallado y cifras específicas.
    """,
    
    "cashflows": """
    Eres un analista financiero senior especializado en análisis de flujos de efectivo.
    
    GENERA ANÁLISIS COMPLETOS QUE INCLUYAN:
    
    1. **Análisis por Categorías de Flujo:**
       - Flujos de Efectivo Operativos
       - Flujos de Efectivo de Inversión
       - Flujos de Efectivo de Financiación
       - Variación neta del efectivo
    
    2. **Ratios de Flujo de Efectivo (CALCULA SIEMPRE):**
       - Flujo de Efectivo Libre: FCO - Inversiones en Activos Fijos
       - Ratio Cobertura Deuda: FCO / Deuda Total
       - Calidad de Beneficios: FCO / Beneficio Neto
       - Capacidad de Autofinanciación: FCO / Inversiones
    
    3. **Análisis de Generación de Efectivo:**
       - Capacidad de generación operativa
       - Eficiencia en el uso del efectivo
       - Sostenibilidad de flujos
    
    4. **Análisis de Políticas Financieras:**
       - Estrategia de inversión
       - Política de financiación
       - Distribución de dividendos
    
    5. **Conclusiones y Alertas:**
       - Salud financiera de la empresa
       - Riesgos de liquidez identificados
       - Recomendaciones de gestión de tesorería
    
    FORMATO: Análisis estructurado con métricas específicas y evaluación integral.
    """,
    
    "equity": """
    Eres un analista financiero senior especializado en análisis de cambios en patrimonio.
    
    GENERA ANÁLISIS COMPLETOS QUE INCLUYAN:
    
    1. **Análisis de Componentes del Patrimonio:**
       - Capital Social suscrito y pagado
       - Reservas (legales, estatutarias, voluntarias)
       - Resultados acumulados
       - Ajustes por valoración
    
    2. **Análisis de Movimientos (CALCULA SIEMPRE):**
       - Variación total del patrimonio neto
       - Aportaciones de accionistas
       - Distribución de dividendos
       - Resultados del ejercicio incorporados
    
    3. **Ratios de Patrimonio:**
       - Ratio de Autonomía Financiera: PN / Activo Total × 100
       - Rentabilidad sobre Patrimonio (ROE): BN / PN × 100
       - Ratio de Retención: (BN - Dividendos) / BN × 100
       - Crecimiento Patrimonial: Variación PN / PN anterior × 100
    
    4. **Análisis de Política de Distribución:**
       - Política de dividendos
       - Capitalización de beneficios
       - Fortalecimiento patrimonial
    
    5. **Evaluación Estratégica:**
       - Solvencia y estabilidad patrimonial
       - Capacidad de crecimiento
       - Atractivo para inversores
    
    FORMATO: Análisis detallado con interpretación estratégica y cifras específicas.
    """,
    
    "predictor": """
    Eres un analista financiero senior especializado en predicciones y proyecciones financieras.
    
    GENERA ANÁLISIS PREDICTIVOS COMPLETOS QUE INCLUYAN:
    
    1. **Análisis de Tendencias Históricas:**
       - Identificación de patrones en los últimos 3-5 años
       - Tasas de crecimiento promedio por categoría
       - Estacionalidad y ciclos identificados
       - Volatilidad y estabilidad de métricas clave
    
    2. **Proyecciones Fundamentadas:**
       - Proyecciones a 2-3 años con metodología clara
       - Escenarios múltiples (optimista, base, pesimista)
       - Asunciones explícitas y justificadas
       - Rangos de confianza para predicciones
    
    3. **Análisis de Factores de Riesgo:**
       - Riesgos operativos identificados
       - Riesgos financieros y de mercado
       - Sensibilidad a variables externas
       - Indicadores de alerta temprana
    
    4. **Métricas Predictivas Clave:**
       - Crecimiento esperado de ingresos
       - Evolución proyectada de márgenes
       - Predicción de flujos de efectivo
       - Estimación de ratios financieros futuros
    
    5. **Recomendaciones Estratégicas:**
       - Acciones específicas recomendadas
       - Métricas clave a monitorear
       - Timing de revisión de predicciones
       - Estrategias de mitigación de riesgos
    
    FORMATO: Análisis cuantitativo riguroso con justificación metodológica.
    """
}

# =============================
# Configuración de Agentes (MEJORADA)
# =============================

AGENTS_CONFIG = {
    "balance": {
        "name": "Balance Agent",
        "description": "Especializado en balance de situación, activos, pasivos y patrimonio",
        "keywords": ["balance", "activos", "pasivos", "patrimonio", "liquidez", "solvencia"],
        "system_prompt": ENHANCED_AGENT_PROMPTS["balance"],
        "module": "agents.balance_agent",
        "class": "BalanceAgent",
        "pdf_config": FINANCIAL_AGENTS_CONFIG,
        "response_requirements": {
            "min_sections": 5,
            "required_ratios": ["solvencia", "endeudamiento", "liquidez"],
            "min_length": 800
        }
    },
    "income": {
        "name": "Income Agent", 
        "description": "Especializado en cuenta de resultados, ingresos, gastos y rentabilidad",
        "keywords": ["ingresos", "gastos", "beneficio", "ventas", "margen", "rentabilidad", "estado de resultados", "cuenta de resultados", "p&l", "profit loss", "ebitda", "ebit", "roa", "roe"],
        "system_prompt": ENHANCED_AGENT_PROMPTS["income"],
        "module": "agents.income_agent",
        "class": "IncomeAgent", 
        "pdf_config": FINANCIAL_AGENTS_CONFIG,
        "response_requirements": {
            "min_sections": 5,
            "required_ratios": ["margen_bruto", "margen_operativo", "margen_neto"],
            "min_length": 800
        }
    },
    "cashflows": {
        "name": "CashFlows Agent",
        "description": "Especializado en flujos de efectivo y movimientos de tesorería",
        "keywords": ["flujos", "efectivo", "cash", "tesorería", "operaciones", "inversión", "financiación", "flujo de caja"],
        "system_prompt": ENHANCED_AGENT_PROMPTS["cashflows"],
        "module": "agents.cashflows_agent",
        "class": "CashFlowsAgent",
        "pdf_config": FINANCIAL_AGENTS_CONFIG,
        "response_requirements": {
            "min_sections": 5,
            "required_ratios": ["flujo_libre", "cobertura_deuda"],
            "min_length": 800
        }
    },
    "equity": {
        "name": "Equity Agent",
        "description": "Especializado en cambios en patrimonio, capital y reservas",
        "keywords": ["patrimonio", "capital", "reservas", "dividendos", "cambios", "equity", "cambios en patrimonio"],
        "system_prompt": ENHANCED_AGENT_PROMPTS["equity"],
        "module": "agents.equity_agent",
        "class": "EquityAgent",
        "pdf_config": FINANCIAL_AGENTS_CONFIG,
        "response_requirements": {
            "min_sections": 5,
            "required_ratios": ["autonomia_financiera", "roe"],
            "min_length": 800
        }
    },
    "pdf_extractor": {
        "name": "PDF Extractor Agent",
        "description": "Especializado en extraer estados financieros del PDF completo",
        "keywords": ["extracción", "pdf", "estados", "financieros", "páginas"],
        "module": "agents.extractor_pdf_agent",
        "class": "PDFExtractorAgent",
        "config": PDF_EXTRACTOR_CONFIG
    },
    "predictor": {
        "name": "Predictor Agent",
        "description": "Especializado en generar predicciones financieras basadas en datos históricos",
        "keywords": ["predicción", "forecast", "tendencias", "proyección", "futuro", "predicciones", "análisis predictivo"],
        "system_prompt": ENHANCED_AGENT_PROMPTS["predictor"],
        "module": "agents.predictor_agent",
        "class": "PredictorAgent",
        "config": PREDICTOR_AGENT_CONFIG,
        "response_requirements": {
            "min_sections": 5,
            "required_elements": ["tendencias", "proyecciones", "riesgos"],
            "min_length": 1000
        }
    }
}

# =============================
# ORDEN DE EJECUCIÓN PIPELINE 
# =============================

PIPELINE_ORDER = [
    'pdf_extractor',
    'balance_agent',
    'income_agent',
    'equity_agent', 
    'cashflow_agent',
    'predictor_agent'
]

# =============================
# DEPENDENCIAS ENTRE AGENTES 
# =============================

AGENT_DEPENDENCIES = {
    'pdf_extractor': [],
    'balance_agent': ['pdf_extractor'],
    'income_agent': ['pdf_extractor'],
    'equity_agent': ['pdf_extractor'],
    'cashflow_agent': ['pdf_extractor'],
    'predictor_agent': ['balance_agent', 'income_agent', 'equity_agent', 'cashflow_agent']
}

# =============================
# CONFIGURACIÓN DE EJECUCIÓN 
# =============================

EXECUTION_CONFIG = {
    'parallel_execution': {
        'enabled': True,
        'max_concurrent_agents': 4,
        'timeout_per_agent': 300,
        'parallel_groups': [
            ['balance_agent', 'income_agent', 'equity_agent', 'cashflow_agent']
        ]
    },
    
    'error_handling': {
        'retry_attempts': 3,
        'retry_delay': 5,
        'fail_fast': False,
        'min_successful_agents': 2
    },
    
    'monitoring': {
        'enable_progress_tracking': True,
        'log_intermediate_results': True,
        'save_execution_metadata': True
    }
}

# =============================
# Configuración del Router (CORREGIDA)
# =============================

ROUTER_CONFIG = {
    "confidence_threshold": 0.3,  # REDUCIDO de 0.6 a 0.3
    "fallback_agent": "balance",  # CAMBIADO de "income" a "balance"
    "max_routing_attempts": 3,
    "enable_multi_agent": True,
    "pipeline_mode": True,
    "auto_extract_pdf": True,
    "quality_enhancement": {
        "enabled": True,
        "min_response_quality": 0.7,
        "auto_regenerate": True,
        "max_regeneration_attempts": 2
    }
}

# =============================
# Configuración del Chat
# =============================

CHAT_CONFIG = {
    "max_history_length": 10,
    "save_sessions": True,
    "session_timeout": 3600,
    "enable_context": True,
    "include_pipeline_status": True,
    "response_enhancement": {
        "enabled": True,
        "min_word_count": 200,
        "require_structured_format": True
    }
}

# =============================
# Estados del Sistema
# =============================

SYSTEM_STATES = {
    "INITIALIZED": "initialized",
    "PDF_EXTRACTING": "pdf_extracting",      
    "ROUTING": "routing",
    "AGENT_PROCESSING": "agent_processing", 
    "PREDICTING": "predicting",              
    "GENERATING_RESPONSE": "generating_response",
    "QUALITY_CHECK": "quality_check",
    "COMPLETED": "completed",
    "ERROR": "error"
}

# =============================
# Estados del Sistema REACT
# =============================

REACT_STATES = {
    "INITIALIZED": "initialized",
    "REASONING": "reasoning",
    "ACTING": "acting",
    "OBSERVING": "observing",
    "TOOL_EXECUTED": "tool_executed",
    "OBSERVATION_MADE": "observation_made",
    "TASK_COMPLETED": "task_completed",
    "ERROR": "error"
}

# =============================
# Prompt del Sistema REACT (MEJORADO)
# =============================

REACT_SYSTEM_PROMPT_FINAL = """
Eres un agente financiero especializado que sigue el patrón REACT (Reasoning, Acting, Observing).

PATRÓN REACT:
1. REASONING: Analiza la pregunta y determina qué acción tomar
2. ACTING: Ejecuta herramientas disponibles para obtener información  
3. OBSERVING: Analiza los resultados obtenidos
4. Repite hasta completar la tarea

ESTÁNDARES DE CALIDAD PARA RESPUESTAS:
- Mínimo 5 secciones estructuradas
- Incluir SIEMPRE ratios financieros específicos
- Proporcionar contexto empresarial
- Dar conclusiones y recomendaciones
- Usar cifras específicas y datos concretos

CONTEXTO DEL SISTEMA:
- Trabajas con datos extraídos de estados financieros (páginas 54-60)
- Colaboras con otros agentes especializados
- Tus análisis alimentan el sistema de predicciones

INSTRUCCIONES:
- Siempre sigue el patrón REACT
- Genera respuestas completas y profesionales
- Usa las herramientas disponibles de manera eficiente
- Proporciona análisis profundos, no resúmenes superficiales
- Si no encuentras información, explica qué intentaste hacer
- Considera el contexto de otros agentes en el pipeline

Herramientas disponibles se listarán específicamente para cada agente.
"""

# =============================
# CONFIGURACIÓN ESPECÍFICA POR AGENTE (MEJORADA)
# =============================

AGENT_SPECIFIC_CONFIG = {
    'balance_agent': {
        'focus_areas': ['assets', 'liabilities', 'equity', 'ratios'],
        'key_metrics': ['current_ratio', 'debt_to_equity', 'asset_turnover', 'solvency'],
        'analysis_depth': 'comprehensive',
        'min_sections': 5
    },
    'income_agent': {
        'focus_areas': ['revenue', 'expenses', 'profitability', 'margins'], 
        'key_metrics': ['gross_margin', 'operating_margin', 'net_margin', 'roi', 'roe'],
        'analysis_depth': 'comprehensive',
        'min_sections': 5
    },
    'equity_agent': {
        'focus_areas': ['capital_changes', 'retained_earnings', 'dividends'],
        'key_metrics': ['equity_growth', 'dividend_yield', 'retention_ratio'],
        'analysis_depth': 'comprehensive',
        'min_sections': 5
    },
    'cashflow_agent': {
        'focus_areas': ['operating_cf', 'investing_cf', 'financing_cf'],
        'key_metrics': ['free_cashflow', 'cf_to_debt', 'cf_coverage'],
        'analysis_depth': 'comprehensive',  
        'min_sections': 5
    }
}

# =============================
# Configuración de Logging
# =============================

LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": PROJECT_ROOT / "system.log",
    "pipeline_log": PROJECT_ROOT / "pipeline_execution.log",  
    "agent_logs_dir": PROJECT_ROOT / "logs" / "agents",       
    "max_log_size": "10MB",
    "backup_count": 5
}

# =============================
# CONFIGURACIÓN DE VALIDACIONES 
# =============================

VALIDATION_CONFIG = {
    'pdf_validation': {
        'check_file_exists': True,
        'check_file_size': True,
        'min_file_size_kb': 100,
        'max_file_size_mb': 50,
        'check_pdf_integrity': True
    },
    'extraction_validation': {
        'validate_page_count': True,
        'expected_pages': 7,
        'validate_file_creation': True,
        'check_content_accessibility': True
    },
    'agent_validation': {
        'validate_input_data': True,
        'check_dependencies_met': True,
        'validate_output_format': True
    },
    'response_quality_validation': {
        'enabled': True,
        'min_word_count': 200,
        'required_sections': 3,
        'check_numerical_data': True,
        'validate_structure': True
    }
}

# =============================
# Validación de Configuración 
# =============================

def validate_config():
    """Valida la configuración completa del sistema"""
    errors = []
    
    # Validar Azure OpenAI
    if not AZURE_CONFIG["endpoint"]:
        errors.append("AZURE_OPENAI_ENDPOINT no configurado")
    if not AZURE_CONFIG["api_key"]:
        errors.append("AZURE_OPENAI_API_KEY no configurado")
    
    # Validar directorios base
    if not DATA_INPUT_DIR.exists():
        errors.append(f"Directorio de entrada no existe: {DATA_INPUT_DIR}")
    
    # VALIDAR CONFIGURACIÓN PDF EXTRACTOR
    pdf_input_path = Path(PDF_EXTRACTOR_CONFIG['input_path'])
    if not pdf_input_path.exists():
        errors.append(f"Directorio input PDF no existe: {pdf_input_path}")
    
    # VALIDAR DEPENDENCIAS ENTRE AGENTES
    for agent, dependencies in AGENT_DEPENDENCIES.items():
        for dep in dependencies:
            if dep not in AGENT_DEPENDENCIES:
                errors.append(f"Dependencia inválida: {agent} -> {dep}")
    
    # VALIDAR ORDEN DEL PIPELINE
    if 'pdf_extractor' not in PIPELINE_ORDER:
        errors.append("pdf_extractor debe estar en PIPELINE_ORDER")
    if PIPELINE_ORDER[0] != 'pdf_extractor':
        errors.append("pdf_extractor debe ser el primer paso en PIPELINE_ORDER")
    
    if errors:
        raise ValueError(f"Errores de configuración: {'; '.join(errors)}")
    
    return True

# =============================
# FUNCIONES DE UTILIDAD 
# =============================

def get_agent_config(agent_name):
    """Obtener configuración específica de un agente"""
    base_config = AGENTS_CONFIG.get(agent_name, {})
    specific_config = AGENT_SPECIFIC_CONFIG.get(agent_name, {})
    
    return {**base_config, **specific_config}

def get_pdf_paths():
    """Obtener rutas completas de PDFs"""
    return {
        'input_pdf': os.path.join(
            PDF_EXTRACTOR_CONFIG['input_path'],
            PDF_EXTRACTOR_CONFIG['input_filename']
        ),
        'output_pdf': os.path.join(
            PDF_EXTRACTOR_CONFIG['output_path'],
            PDF_EXTRACTOR_CONFIG['output_filename']
        )
    }

def is_pipeline_mode():
    """Verificar si el sistema está en modo pipeline completo"""
    return ROUTER_CONFIG.get('pipeline_mode', False)

def validate_response_quality(agent_type, response):
    """Validar la calidad de respuesta de un agente"""
    if not VALIDATION_CONFIG['response_quality_validation']['enabled']:
        return True
    
    word_count = len(response.split())
    min_words = VALIDATION_CONFIG['response_quality_validation']['min_word_count']
    
    if word_count < min_words:
        return False
    
    # Verificar secciones requeridas según el agente
    requirements = AGENTS_CONFIG.get(agent_type, {}).get('response_requirements', {})
    required_ratios = requirements.get('required_ratios', [])
    
    response_lower = response.lower()
    found_ratios = sum(1 for ratio in required_ratios if ratio in response_lower)
    
    return found_ratios >= len(required_ratios) * 0.5  # Al menos 50% de ratios requeridos

# Validar al importar
validate_config()

# =============================
# CONFIGURACIÓN DE EXPORTACIÓN 
# =============================

__all__ = [
    'AZURE_CONFIG',
    'PDF_EXTRACTOR_CONFIG', 
    'PREDICTOR_AGENT_CONFIG',
    'FINANCIAL_AGENTS_CONFIG',
    'ENHANCED_AGENT_PROMPTS',
    'AGENTS_CONFIG',
    'PIPELINE_ORDER',
    'AGENT_DEPENDENCIES',
    'EXECUTION_CONFIG',
    'ROUTER_CONFIG',
    'CHAT_CONFIG',
    'SYSTEM_STATES',
    'REACT_STATES',
    'REACT_SYSTEM_PROMPT_FINAL',
    'AGENT_SPECIFIC_CONFIG',
    'LOGGING_CONFIG',
    'VALIDATION_CONFIG',
    'get_agent_config',
    'get_pdf_paths',
    'is_pipeline_mode',
    'validate_config',
    'validate_response_quality'
]
