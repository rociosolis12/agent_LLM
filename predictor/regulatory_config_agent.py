"""
regulatory_config_agent.py - Agente de Configuración Dinámica
Obtiene umbrales regulatorios y parámetros técnicos contextuales
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from openai import AzureOpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RegulatoryConfigAgent:
    """
    Agente que obtiene configuración dinámica basada en:
    - Regulación Basel III/IV vigente
    - Jurisdicción del banco
    - Tipo de institución (G-SIB, regional, etc.)
    - Mejores prácticas de la industria
    """
    
    def __init__(self, azure_endpoint=None, azure_api_key=None):
        """Inicializa cliente Azure OpenAI"""
        self.client = AzureOpenAI(
            azure_endpoint=azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=azure_api_key or os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-02-15-preview"
        )
        self.config_cache = {}
    
    
    def get_regulatory_thresholds(
        self, 
        bank_symbol: str,
        jurisdiction: str = "EU",
        bank_type: str = "regional",
        reference_date: str = None
    ) -> Dict[str, Any]:
        """
        Obtiene umbrales regulatorios contextuales vía LLM
        
        Args:
            bank_symbol: Símbolo del banco (ej: "GARAN.IS")
            jurisdiction: Jurisdicción regulatoria ("EU", "US", "CH", "UK", "TR")  
            bank_type: Tipo de banco ("G-SIB", "regional", "international")
            reference_date: Fecha de referencia (default: hoy)
        
        Returns:
            Dict con umbrales regulatorios aplicables
        """
        if reference_date is None:
            reference_date = datetime.now().strftime("%Y-%m-%d")
        
        cache_key = f"{bank_symbol}_{jurisdiction}_{bank_type}_{reference_date}"
        
        if cache_key in self.config_cache:
            logger.info("✅ Usando configuración en caché")
            return self.config_cache[cache_key]
        
        logger.info(f"🔍 Obteniendo umbrales regulatorios para {bank_symbol}...")
        
        prompt = self._build_regulatory_prompt(
            bank_symbol, jurisdiction, bank_type, reference_date
        )
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": """Eres un experto en regulación bancaria Basel III/IV. 
                        Proporciona umbrales regulatorios EXACTOS vigentes según la normativa actual.
                        IMPORTANTE: Responde SOLO con JSON válido, sin texto adicional."""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.0,  # Determinístico
                response_format={"type": "json_object"}
            )
            
            config = json.loads(response.choices[0].message.content)
            
            # Valida estructura
            validated_config = self._validate_and_enrich_config(config, jurisdiction)
            
            # Cachea resultado
            self.config_cache[cache_key] = validated_config
            
            logger.info("✅ Configuración regulatoria obtenida exitosamente")
            return validated_config
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo configuración: {e}")
            return self._get_fallback_config(jurisdiction)
    
    
    def _build_regulatory_prompt(
        self, 
        bank_symbol: str, 
        jurisdiction: str,
        bank_type: str,
        reference_date: str
    ) -> str:
        """Construye prompt para obtener umbrales regulatorios"""
        
        return f"""
        Proporciona los umbrales regulatorios aplicables para análisis financiero bancario con estos parámetros:
        
        CONTEXTO:
        - Banco: {bank_symbol}
        - Jurisdicción: {jurisdiction}
        - Tipo de institución: {bank_type}
        - Fecha de referencia: {reference_date}
        - Marco regulatorio: Basel III/IV
        
        UMBRALES REQUERIDOS (responde en formato JSON):
        
        {{
            "capital_ratios": {{
                "cet1_minimum": <valor_numerico>,
                "cet1_with_buffers": <valor_numerico>,
                "tier1_minimum": <valor_numerico>,
                "total_capital_minimum": <valor_numerico>,
                "leverage_ratio_minimum": <valor_numerico>
            }},
            "liquidity_ratios": {{
                "lcr_minimum": <valor_numerico>,
                "nsfr_minimum": <valor_numerico>
            }},
            "performance_benchmarks": {{
                "roa_healthy": <valor_numerico>,
                "roa_warning": <valor_numerico>,
                "roe_healthy": <valor_numerico>,
                "roe_warning": <valor_numerico>
            }},
            "risk_thresholds": {{
                "solvency_critical": <valor_numerico>,
                "solvency_warning": <valor_numerico>,
                "liquidity_critical": <valor_numerico>,
                "liquidity_warning": <valor_numerico>
            }},
            "metadata": {{
                "regulation_framework": "<nombre_marco>",
                "effective_date": "<fecha>",
                "source": "<fuente_regulatoria>",
                "notes": "<observaciones_relevantes>"
            }}
        }}
        
        INSTRUCCIONES:
        1. Usa valores EXACTOS de la regulación vigente en {reference_date}
        2. Si el banco es G-SIB, aplica surcharges adicionales
        3. Para jurisdicción EU: considera CRR II/CRD V
        4. Para US: considera Dodd-Frank y Fed requirements
        5. Para TR (Turquía): considera regulación BRSA (Basel III turco)  # ← AÑADE ESTA LÍNEA
        6. Para filiales internacionales: aplica regulación local, no de matriz  # ← AÑADE ESTA LÍNEA
        7. Proporciona valores numéricos en porcentaje (ej: 4.5 para 4.5%)
        8. Incluye fuentes regulatorias específicas en metadata
        
        Responde ÚNICAMENTE con el JSON, sin texto adicional.
        """
    
    
    def _validate_and_enrich_config(
        self, 
        config: Dict, 
        jurisdiction: str
    ) -> Dict[str, Any]:
        """
        Valida y enriquece la configuración obtenida
        """
        # Estructura esperada
        required_keys = ['capital_ratios', 'liquidity_ratios', 
                        'performance_benchmarks', 'risk_thresholds']
        
        for key in required_keys:
            if key not in config:
                logger.warning(f"⚠️ Clave faltante: {key}. Usando fallback.")
                config[key] = self._get_fallback_config(jurisdiction)[key]
        
        # Valida rangos razonables
        config = self._validate_ranges(config)
        
        # Añade timestamp
        config['metadata']['retrieval_timestamp'] = datetime.now().isoformat()
        
        return config
    
    
    def _validate_ranges(self, config: Dict) -> Dict:
        """Valida que los umbrales estén en rangos razonables"""
        
        validations = {
            'capital_ratios.cet1_minimum': (3.0, 6.0),
            'capital_ratios.leverage_ratio_minimum': (2.0, 5.0),
            'liquidity_ratios.lcr_minimum': (80.0, 120.0),
            'liquidity_ratios.nsfr_minimum': (80.0, 120.0)
        }
        
        for path, (min_val, max_val) in validations.items():
            keys = path.split('.')
            value = config
            for key in keys:
                value = value.get(key, None)
                if value is None:
                    break
            
            if value is not None and not (min_val <= value <= max_val):
                logger.warning(
                    f"⚠️ Valor fuera de rango para {path}: {value}. "
                    f"Esperado entre {min_val} y {max_val}"
                )
        
        return config
    
    
    def _get_fallback_config(self, jurisdiction: str) -> Dict[str, Any]:
        """
        Configuración de respaldo basada en Basel III/IV estándar
        Fuentes: BIS, ECB, Fed
        """
        logger.warning("⚠️ Usando configuración de respaldo (Basel III estándar)")
        
        base_config = {
            "capital_ratios": {
                "cet1_minimum": 4.5,          # Basel III mínimo
                "cet1_with_buffers": 10.5,    # Con conservation + countercyclical
                "tier1_minimum": 6.0,          # Basel III Tier 1
                "total_capital_minimum": 8.0,  # Basel III total
                "leverage_ratio_minimum": 3.0  # Basel III leverage
            },
            "liquidity_ratios": {
                "lcr_minimum": 100.0,          # LCR estándar desde 2019
                "nsfr_minimum": 100.0          # NSFR estándar
            },
            "performance_benchmarks": {
                "roa_healthy": 0.8,            # Industry benchmark
                "roa_warning": 0.0,            # Pérdidas
                "roe_healthy": 10.0,           # Industry benchmark
                "roe_warning": 5.0             # Underperformance
            },
            "risk_thresholds": {
                "solvency_critical": 4.5,      # Por debajo de CET1 mínimo
                "solvency_warning": 8.0,       # Por debajo de total capital
                "liquidity_critical": 80.0,    # LCR crítico
                "liquidity_warning": 100.0     # En el límite
            },
            "metadata": {
                "regulation_framework": "Basel III/IV Standard",
                "effective_date": "2019-01-01",
                "source": "BIS - Basel Committee on Banking Supervision",
                "notes": "Configuración de respaldo - valores estándar conservadores",
                "is_fallback": True
            }
        }
        
        # Ajustes por jurisdicción
        if jurisdiction == "EU":
            base_config["capital_ratios"]["cet1_with_buffers"] = 11.0  # CRR II
            base_config["metadata"]["regulation_framework"] = "CRR II / CRD V"
            base_config["metadata"]["source"] = "European Banking Authority (EBA)"
        
        elif jurisdiction == "US":
            base_config["capital_ratios"]["cet1_with_buffers"] = 11.0  # Dodd-Frank
            base_config["metadata"]["regulation_framework"] = "Dodd-Frank / Fed Rules"
            base_config["metadata"]["source"] = "Federal Reserve Board"
        
        elif jurisdiction == "TR":  
            base_config["capital_ratios"]["total_capital_minimum"] = 12.0
            base_config["capital_ratios"]["cet1_with_buffers"] = 10.0
            base_config["liquidity_ratios"]["lcr_minimum"] = 100.0
            base_config["metadata"]["regulation_framework"] = "Turkish Basel III"
            base_config["metadata"]["source"] = "BRSA (Turkish Banking Regulator)"
            base_config["metadata"]["notes"] = "Turkish subsidiaries including Garanti BBVA"
    
        
        return base_config
    
    
    def get_ml_configuration(
        self, 
        data_characteristics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Obtiene configuración ML dinámica basada en características de datos
        
        Args:
            data_characteristics: Dict con:
                - num_periods: Número de períodos históricos
                - frequency: Frecuencia de datos ('monthly', 'quarterly', 'annual')
                - num_metrics: Número de métricas a predecir
        
        Returns:
            Dict con configuración ML óptima
        """
        logger.info("🤖 Obteniendo configuración ML dinámica...")
        
        num_periods = data_characteristics.get('num_periods', 0)
        frequency = data_characteristics.get('frequency', 'monthly')
        
        ml_config = {
            "use_prophet": False,
            "use_xgboost": False,
            "use_ensemble": False,
            "prophet_config": {},
            "xgboost_config": {},
            "recommendations": []
        }
        
        # Prophet: Requiere mínimo 18 períodos para datos mensuales
        prophet_minimum = {
            'monthly': 18,
            'quarterly': 8,
            'annual': 3
        }
        
        if num_periods >= prophet_minimum.get(frequency, 18):
            ml_config["use_prophet"] = True
            ml_config["prophet_config"] = {
                "seasonality_mode": "multiplicative" if num_periods >= 24 else "additive",
                "changepoint_prior_scale": 0.05,
                "seasonality_prior_scale": 10.0
            }
            ml_config["recommendations"].append(
                f"Prophet habilitado: {num_periods} períodos suficientes para {frequency} data"
            )
        else:
            ml_config["recommendations"].append(
                f"Prophet deshabilitado: Se requieren al menos {prophet_minimum.get(frequency, 18)} "
                f"períodos para datos {frequency}, disponibles: {num_periods}"
            )
        
        # XGBoost: Más flexible, pero mejor con 12+ períodos
        if num_periods >= 12:
            ml_config["use_xgboost"] = True
            ml_config["xgboost_config"] = {
                "max_depth": min(6, max(3, num_periods // 10)),
                "n_estimators": min(100, num_periods * 5),
                "learning_rate": 0.1,
                "min_child_weight": max(1, num_periods // 20)
            }
            ml_config["recommendations"].append(
                f"XGBoost habilitado con parámetros adaptados a {num_periods} períodos"
            )
        else:
            ml_config["recommendations"].append(
                f"XGBoost limitado: Se recomiendan al menos 12 períodos, disponibles: {num_periods}"
            )
        
        # Ensemble si ambos modelos están disponibles
        if ml_config["use_prophet"] and ml_config["use_xgboost"]:
            ml_config["use_ensemble"] = True
            ml_config["recommendations"].append(
                "Ensemble habilitado: Combinando Prophet + XGBoost para mayor robustez"
            )
        
        logger.info(f"✅ Configuración ML generada: {ml_config['recommendations']}")
        return ml_config


# Integración con HybridPredictorAgent
class DynamicHybridPredictorAgent:
    """
    Wrapper del HybridPredictorAgent que inyecta configuración dinámica
    """
    
    def __init__(self, bank_symbol="GARAN.IS", alpha_vantage_key=None):
        from predictor.hybrid_predictor_agent_old import HybridPredictorAgent
        
        # Inicializa agente base
        self.base_agent = HybridPredictorAgent(bank_symbol, alpha_vantage_key)
        
        # Inicializa agente de configuración
        self.config_agent = RegulatoryConfigAgent()
        
        # Configuración dinámica
        self.regulatory_config = None
        self.ml_config = None
    
    
    def run_complete_hybrid_analysis(
        self, 
        agent_results: Dict[str, Any],
        bank_symbol: str = "GARAN.IS",
        jurisdiction: str = "TR",
        bank_type: str = "international"
    ) -> Dict[str, Any]:
        """
        Ejecuta análisis con configuración dinámica inyectada
        """
        logger.info("🔧 Obteniendo configuración dinámica...")
        
        # PASO 1: Obtiene configuración regulatoria
        self.regulatory_config = self.config_agent.get_regulatory_thresholds(
            bank_symbol=bank_symbol,
            jurisdiction=jurisdiction,
            bank_type=bank_type
        )
        
        # PASO 2: Inyecta configuración en el agente base
        self._inject_regulatory_config()
        
        # PASO 3: Ejecuta pipeline normal (sin modificaciones)
        results = self.base_agent.run_complete_hybrid_analysis(
            agent_results=agent_results,
            bank_symbol=bank_symbol
        )
        
        # PASO 4: Añade metadata de configuración dinámica
        results['dynamic_configuration'] = {
            'regulatory_config': self.regulatory_config,
            'ml_config': self.ml_config
        }
        
        return results
    
    
    def _inject_regulatory_config(self):
        """
        Inyecta configuración dinámica en métodos del agente base
        SIN modificar el código original
        """
        # Monkey-patch del método _identify_risk_factors
        original_identify_risk = self.base_agent._identify_risk_factors
        regulatory_config = self.regulatory_config
        
        def dynamic_identify_risk_factors(extracted_data: Dict) -> list:
            """Versión dinámica con umbrales de configuración"""
            risk_factors = []
            
            thresholds = regulatory_config['risk_thresholds']
            
            solvency = extracted_data.get('solvency_ratio')
            if solvency is not None:
                if solvency < thresholds['solvency_critical']:
                    risk_factors.append(
                        f"Ratio de solvencia ({solvency:.2f}%) por debajo del mínimo "
                        f"regulatorio ({thresholds['solvency_critical']}%)"
                    )
                elif solvency < thresholds['solvency_warning']:
                    risk_factors.append(
                        f"Ratio de solvencia ({solvency:.2f}%) en zona de advertencia"
                    )
            
            roa = extracted_data.get('roa')
            if roa is not None:
                if roa < regulatory_config['performance_benchmarks']['roa_warning']:
                    risk_factors.append(
                        f"ROA ({roa:.2f}%) por debajo del benchmark de la industria"
                    )
            
            liquidity = extracted_data.get('liquidity_ratio')
            if liquidity is not None:
                if liquidity < thresholds['liquidity_critical']:
                    risk_factors.append(
                        f"Ratio de liquidez ({liquidity:.2f}) en nivel crítico"
                    )
            
            return risk_factors
        
        # Reemplaza el método
        self.base_agent._identify_risk_factors = dynamic_identify_risk_factors


# Función de testing
def test_dynamic_configuration():
    """
    Test del sistema de configuración dinámica para Garanti BBVA
    """
    print("🧪 Testing Dynamic Configuration System for Garanti BBVA...")
    
    config_agent = RegulatoryConfigAgent()
    
    # ================================================================
    # Test 1: Configuración regulatoria para Garanti BBVA (Turquía)
    # ================================================================
    print("\n" + "="*70)
    print("📊 TEST 1: Garanti BBVA (Turkey) - Regulatory Thresholds")
    print("="*70)
    
    garanti_config = config_agent.get_regulatory_thresholds(
        bank_symbol="GARAN.IS",        # ← Garanti BBVA en Borsa Istanbul
        jurisdiction="TR",              # ← Turquía
        bank_type="international"       # ← Filial internacional (no G-SIB)
    )
    
    print(json.dumps(garanti_config, indent=2, ensure_ascii=False))
    
    # Muestra resumen
    print("\n📋 Resumen de Umbrales Aplicables:")
    print(f"  Marco regulatorio: {garanti_config['metadata']['regulation_framework']}")
    print(f"  Fuente: {garanti_config['metadata']['source']}")
    print(f"  CET1 mínimo: {garanti_config['capital_ratios']['cet1_minimum']}%")
    print(f"  Total capital mínimo: {garanti_config['capital_ratios']['total_capital_minimum']}%")
    print(f"  LCR mínimo: {garanti_config['liquidity_ratios']['lcr_minimum']}%")
    
    # ================================================================
    # Test 2: Configuración ML
    # ================================================================
    print("\n" + "="*70)
    print("🤖 TEST 2: ML Configuration (Prophet + XGBoost)")
    print("="*70)
    
    ml_config = config_agent.get_ml_configuration({
        'num_periods': 24,          # 24 meses de datos históricos
        'frequency': 'monthly',      # Datos mensuales
        'num_metrics': 4             # ROA, ROE, Solvencia, Liquidez
    })
    
    print(json.dumps(ml_config, indent=2, ensure_ascii=False))
    
    print("\n📋 Modelos ML Habilitados:")
    print(f"  Prophet: {'✅ Habilitado' if ml_config['use_prophet'] else '❌ Deshabilitado'}")
    print(f"  XGBoost: {'✅ Habilitado' if ml_config['use_xgboost'] else '❌ Deshabilitado'}")
    print(f"  Ensemble: {'✅ Habilitado' if ml_config['use_ensemble'] else '❌ Deshabilitado'}")
    
    # ================================================================
    # Test 3 (Opcional): Comparación con BBVA matriz
    # ================================================================
    print("\n" + "="*70)
    print("📊 TEST 3 (Comparación): BBVA Matriz (Spain/EU, G-SIB)")
    print("="*70)
    
    bbva_config = config_agent.get_regulatory_thresholds(
        bank_symbol="BBVA.MC",
        jurisdiction="EU",
        bank_type="G-SIB"
    )
    
    print(f"\n  BBVA (matriz):")
    print(f"    CET1 con buffers: {bbva_config['capital_ratios']['cet1_with_buffers']}%")
    print(f"    Marco: {bbva_config['metadata']['regulation_framework']}")
    
    print(f"\n  Garanti BBVA (filial):")
    print(f"    CET1 con buffers: {garanti_config['capital_ratios']['cet1_with_buffers']}%")
    print(f"    Marco: {garanti_config['metadata']['regulation_framework']}")
    
    print("\n  ℹ️ Nota: Diferentes reguladores aplican umbrales distintos")
    print("     BBVA matriz → ECB/EBA (Unión Europea)")
    print("     Garanti BBVA → BRSA (Regulador turco)")
    
    print("\n" + "="*70)
    print("✅ Tests completados exitosamente!")
    print("="*70)


if __name__ == "__main__":
    test_dynamic_configuration()


