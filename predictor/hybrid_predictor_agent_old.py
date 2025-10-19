"""
hybrid_predictor_agent.py - Agente Predictor Híbrido
Combina extracción LLM + Machine Learning avanzado
"""

import os
import json
import sys
import logging
import asyncio
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime

# Añadir directorio padre al path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Imports
from predictor_agent import PredictorAgent
from update_predictor_agent import EvolutionaryPredictorAgent
from regulatory_config_agent import RegulatoryConfigAgent  


#from balance_agent import BalanceAgent
#from income_agent import IncomeAgent
#from cashflow_agent import CashFlowAgent
#from equity_agent import EquityAgent


from config import (
    PREDICTOR_AGENT_CONFIG, 
    FINANCIAL_AGENTS_CONFIG,
    DATA_OUTPUT_DIR
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridPredictorAgent:
    """
    Agente Predictor Híbrido: LLM + Machine Learning
    
    Configurado para Garanti BBVA (Türkiye Garanti Bankası A.Ş.)
    - Filial turca de BBVA
    - Regulación: BRSA (Banking Regulation and Supervision Agency)
    - Jurisdicción: Turquía (TR)
    
    Pipeline de 2 fases:
    1. Extracción LLM: Procesa estados financieros y genera análisis cualitativo
    2. Predicción ML: Genera forecasting cuantitativo con Prophet/XGBoost
    """
    
    def __init__(
        self, 
        bank_symbol="GARAN.IS",             
        alpha_vantage_key=None,
        jurisdiction="TR",                   
        bank_type="international",          
        parent_bank="BBVA",                 
        use_regulatory_config=True,
        data_lake_df=None,
    ):
        """
        Inicializa agente híbrido para Garanti BBVA
        
        Args:
            bank_symbol: GARAN.IS (Borsa Istanbul)
            alpha_vantage_key: API key para datos de mercado
            jurisdiction: TR (Turquía - regulación BRSA)
            bank_type: international (parte de grupo BBVA global)
            parent_bank: BBVA (matriz española)
            use_regulatory_config: Si True, usa umbrales dinámicos BRSA vía LLM
        """
        logger.info("="*70)
        logger.info("🚀 Inicializando HybridPredictorAgent")
        logger.info(f"   Banco: Garanti BBVA (Türkiye Garanti Bankası A.Ş.)")
        logger.info(f"   Ticker: {bank_symbol}")
        logger.info(f"   Matriz: {parent_bank}")
        logger.info(f"   Jurisdicción: {jurisdiction} (BRSA)")
        logger.info(f"   Tipo: {bank_type}")
        logger.info("="*70)
        
        # Agente LLM para extracción
        self.llm_agent = PredictorAgent()
        
        # Agente ML para predicciones
        self.ml_agent = EvolutionaryPredictorAgent(
            alpha_vantage_key=alpha_vantage_key
        )
        
        # Agente Regulatorio para umbrales BRSA (NUEVO)
        self.regulatory_agent = RegulatoryConfigAgent() if use_regulatory_config else None
        
        # Configuración
        self.bank_symbol = bank_symbol
        self.jurisdiction = jurisdiction
        self.bank_type = bank_type
        self.parent_bank = parent_bank
        self.output_dir = DATA_OUTPUT_DIR
        self.use_regulatory_config = use_regulatory_config
        self.data_lake_df = data_lake_df

        # Almacenamiento de resultados
        self.results = {
            'llm_extraction': {},
            'ml_predictions': {},
            'hybrid_analysis': {},
            'regulatory_config': {}  
        }
        
        # Configuración regulatoria dinámica BRSA (NUEVO)
        
        self.regulatory_config = None
        if self.use_regulatory_config:
            self._load_regulatory_config()
    
        
        logger.info("✅ HybridPredictorAgent inicializado correctamente")
        logger.info("")
    
    
    def _load_regulatory_config(self):
        """
        Carga configuración regulatoria dinámica de BRSA (Turquía)
        
        Obtiene vía LLM:
        - Umbrales de capital BRSA (más estrictos que Basel III)
        - Ratios de liquidez específicos de Turquía
        - Benchmarks de rentabilidad para banca turca
        """
        try:
            logger.info(f" Obteniendo configuración regulatoria BRSA para {self.bank_symbol}...")
            
            self.regulatory_config = self.regulatory_agent.get_regulatory_thresholds(
                bank_symbol=self.bank_symbol,
                jurisdiction=self.jurisdiction,
                bank_type=self.bank_type
            )
            
            self.results['regulatory_config'] = self.regulatory_config
            
            metadata = self.regulatory_config.get('metadata', {})
            capital_ratios = self.regulatory_config.get('capital_ratios', {})
            liquidity_ratios = self.regulatory_config.get('liquidity_ratios', {})
            
            logger.info(f"Configuración regulatoria BRSA cargada:")
            logger.info(f"   • Marco: {metadata.get('regulation_framework', 'Basel III/IV')}")
            logger.info(f"   • Jurisdicción: {metadata.get('jurisdiction', self.jurisdiction)}")
            logger.info(f"   • CET1 mínimo: {capital_ratios.get('cet1_minimum', 'N/A')}%")
            logger.info(f"   • Capital total: {capital_ratios.get('total_capital_minimum', 'N/A')}%")
            logger.info(f"   • LCR mínimo: {liquidity_ratios.get('lcr_minimum', 'N/A')}%")
            logger.info("")
            
        except Exception as e:
            logger.warning(f"⚠️ Error cargando configuración BRSA: {e}")
            logger.warning("Usando configuración de respaldo (Basel III + ajustes BRSA)")
            self.regulatory_config = self._get_fallback_regulatory_config()
            self.results['regulatory_config'] = self.regulatory_config

    
    
    def _get_fallback_regulatory_config(self) -> Dict[str, Any]:
        """
        🆕 NUEVO: Configuración de respaldo si falla regulatory_agent
        
        Valores basados en regulación BRSA conocida:
        - BRSA exige capital mínimo más alto que Basel III
        - Turquía tiene requisitos de liquidez más estrictos
        """
        logger.info("⚙️ Usando configuración de respaldo BRSA (hardcoded)")
        
        return {
            'capital_ratios': {
                'cet1_minimum': 4.5,           # Basel III base
                'tier1_minimum': 6.0,          # Basel III
                'total_capital_minimum': 10.0,  # BRSA exige >8% Basel III
                'leverage_ratio_minimum': 3.0
            },
            'liquidity_ratios': {
                'lcr_minimum': 100.0,          # LCR estándar
                'nsfr_minimum': 100.0
            },
            'performance_benchmarks': {
                'roa_healthy': 1.2,            # Banca turca promedio ~1.5%
                'roa_warning': 0.5,
                'roe_healthy': 12.0,           # Banca turca promedio 15-18%
                'roe_warning': 8.0,
                'nim_healthy': 3.5,            # Net Interest Margin
                'cost_income_healthy': 40.0    # Cost-to-Income Ratio
            },
            'risk_thresholds': {
                'solvency_critical': 6.0,      # Por debajo Tier 1
                'solvency_warning': 10.0,      # BRSA mínimo
                'liquidity_critical': 80.0,
                'liquidity_warning': 100.0,
                'npl_ratio_warning': 3.0,      # Non-Performing Loans
                'npl_ratio_critical': 5.0
            },
            'metadata': {
                'regulation_framework': 'BRSA + Basel III (Fallback)',
                'jurisdiction': 'Turkey (TR)',
                'source': 'Hardcoded fallback configuration',
                'is_fallback': True,
                'last_updated': datetime.now().isoformat()
            }
        }
    
    
    async def extract_financial_data_from_agents(self, agent_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        FASE 1: Extracción de datos usando predictor_agent
        
        Extrae datos de:
        - Balance Agent: Activos, pasivos, patrimonio
        - Income Agent: Ingresos, gastos, beneficio neto
        - Cash Flow Agent: Flujos operativos, inversión, financiación
        - Equity Agent: Capital social, reservas
        
        Returns:
            Dict con datos extraídos y análisis cualitativo
        """
        logger.info("📊 FASE 1: Extracción LLM iniciada...")
        
        try:
            # Prepara estructura para extract_and_map_data()
            financial_data = {
                'agents_results': agent_results,
                'structured_for_predictor': {},
                'bank_info': {
                    'symbol': self.bank_symbol,
                    'name': 'Garanti BBVA',
                    'jurisdiction': self.jurisdiction,
                    'parent': self.parent_bank
                }
            }
            
            # Usa extract_and_map_data del PredictorAgent
            extracted_data = await self.llm_agent._extract_and_map_data(financial_data)
            
            # Evalúa completitud
            completeness = self.llm_agent._assess_data_completeness(extracted_data)

            # Obtener score con manejo de ambos formatos
            score = completeness.get('completeness_score', completeness.get('completeness_percentage', 0))
            if isinstance(score, (int, float)):
                if score <= 1:
                    logger.info(f"✅ Datos extraídos - Completitud: {score:.1%}")
                else:
                    logger.info(f"✅ Datos extraídos - Completitud: {score:.1f}%")
            
            # Genera análisis cualitativo
            qualitative_analysis = self._generate_llm_analysis(extracted_data, completeness)
            
            self.results['llm_extraction'] = {
                'extracted_data': extracted_data,
                'completeness': completeness,
                'qualitative_analysis': qualitative_analysis,
                'timestamp': datetime.now().isoformat(),
                'bank_context': {
                    'symbol': self.bank_symbol,
                    'jurisdiction': self.jurisdiction,
                    'parent_bank': self.parent_bank
                }
            }
            
            return self.results['llm_extraction']
            
        except Exception as e:
            logger.error(f"❌ Error en extracción LLM: {e}")
            raise
    
    
    def _generate_llm_analysis(
        self, 
        extracted_data: Dict, 
        completeness: Dict
    ) -> Dict[str, Any]:
        """
        🔄 MODIFICADO: Genera análisis cualitativo usando configuración BRSA
        """
        logger.info("🤖 Generando análisis cualitativo con LLM...")
        
        try:
            analysis = {}
            
            # Obtener completeness en formato correcto
            completeness_pct = completeness.get('completeness_percentage', completeness.get('completeness_score', 0))
            if completeness_pct <= 1:
                completeness_pct = completeness_pct * 100
            
            # Análisis de riesgo usando umbrales BRSA
            if completeness_pct >= 50:
                risk_score = self._calculate_risk_score_local(extracted_data)
                analysis['risk_assessment'] = {
                    'score': risk_score,
                    'level': self._get_risk_level_local(risk_score),
                    'factors': self._identify_risk_factors(extracted_data),
                    'regulatory_framework': self.regulatory_config['metadata']['regulation_framework']
                }
            
            # Análisis de escenarios
            analysis['scenario_analysis'] = self._generate_scenario_analysis(extracted_data)
            
            # Contexto cualitativo del LLM
            analysis['llm_insights'] = self._call_azure_openai_for_insights(extracted_data)
            
            logger.info("✅ Análisis cualitativo completado")
            return analysis
            
        except Exception as e:
            logger.warning(f"⚠️ Error en análisis LLM: {e}")
            return {'error': str(e)}
    
    
    def _generate_scenario_analysis(self, extracted_data: Dict) -> Dict[str, Any]:
        """
        🆕 NUEVO: Genera escenarios cuantitativos basados en datos reales
        """
        scenarios = {}
        
        # Obtener métricas clave
        roa = extracted_data.get('roa', None)
        roe = extracted_data.get('roe', None)
        solvency = extracted_data.get('ratio_solvencia', None)
        
        if roa is not None:
            scenarios['base_case'] = {
                'description': f'Escenario base con ROA actual de {roa:.2f}%',
                'roa_forecast': roa,
                'assumptions': 'Continuidad operativa sin cambios significativos'
            }
            
            scenarios['optimistic'] = {
                'description': 'Escenario optimista con mejora en eficiencia',
                'roa_forecast': roa * 1.15,  # +15%
                'assumptions': 'Mejora 15% en eficiencia operativa, reducción NPL ratio'
            }
            
            scenarios['pessimistic'] = {
                'description': 'Escenario pesimista con deterioro económico',
                'roa_forecast': roa * 0.85,  # -15%
                'assumptions': 'Deterioro económico en Turquía, aumento NPL ratio'
            }
        else:
            scenarios = {
                'note': 'Escenarios cuantitativos requieren datos de ROA'
            }
        
        return scenarios
    
    
    def _calculate_risk_score_local(self, data: Dict) -> float:
        """
        🔄 MODIFICADO: Calcula score de riesgo usando umbrales BRSA
        
        Sin valores default optimistas - solo evalúa si hay datos
        """
        score = 0.0
        
        if self.regulatory_config:
            thresholds = self.regulatory_config['risk_thresholds']
            benchmarks = self.regulatory_config['performance_benchmarks']
            capital = self.regulatory_config['capital_ratios']
        else:
            # Fallback
            thresholds = {'solvency_warning': 10.0, 'liquidity_critical': 1.0}
            benchmarks = {'roa_warning': 0.5, 'roe_warning': 8.0}
            capital = {'total_capital_minimum': 10.0}
        
        # ROA por debajo del umbral de advertencia
        if 'roa' in data:
            if data['roa'] < benchmarks.get('roa_warning', 0.5):
                score += 20
        
        # Ratio de capital por debajo del mínimo BRSA
        if 'ratio_solvencia' in data:
            min_capital = capital.get('total_capital_minimum', 10.0)
            if data['ratio_solvencia'] < min_capital:
                score += 30
        
        # Liquidez crítica
        if 'liquidez' in data:
            if data['liquidez'] < thresholds.get('liquidity_critical', 1.0):
                score += 25
        
        # ROE por debajo del umbral
        if 'roe' in data:
            if data['roe'] < benchmarks.get('roe_warning', 8.0):
                score += 15
        
        # NPL Ratio alto (específico banca turca)
        if 'npl_ratio' in data:
            if data['npl_ratio'] > thresholds.get('npl_ratio_warning', 3.0):
                score += 10
        
        return min(score, 100)
    
    
    def _get_risk_level_local(self, score: float) -> str:
        """
        Determina nivel de riesgo basado en score
        """
        if score >= 70:
            return 'HIGH'
        elif score >= 40:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    
    def _identify_risk_factors(self, extracted_data: Dict) -> List[str]:
        """
        🔄 MODIFICADO: Identifica factores de riesgo específicos BRSA
        
        SIN valores default optimistas - solo evalúa datos reales
        """
        risk_factors = []
        
        if self.regulatory_config:
            capital = self.regulatory_config['capital_ratios']
            liquidity = self.regulatory_config['liquidity_ratios']
            thresholds = self.regulatory_config['risk_thresholds']
            framework = self.regulatory_config['metadata']['regulation_framework']
        else:
            capital = {'total_capital_minimum': 10.0}
            liquidity = {'lcr_minimum': 100.0}
            thresholds = {'npl_ratio_warning': 3.0}
            framework = 'Basel III + BRSA'
        
        # Verificar ratio de capital (SIN default)
        if 'solvency_ratio' in extracted_data:
            solvency = extracted_data['solvency_ratio']
            min_required = capital.get('total_capital_minimum', 10.0)
            
            if solvency < min_required:
                risk_factors.append(
                    f"⚠️ Ratio de capital ({solvency:.2f}%) por debajo del mínimo "
                    f"BRSA ({min_required}%) según {framework}"
                )
        
        # Verificar ROA (SIN default)
        if 'roa' in extracted_data:
            if extracted_data['roa'] < 0:
                risk_factors.append("⚠️ ROA negativo indica pérdidas operativas")
            elif extracted_data['roa'] < 0.5:
                risk_factors.append(f"⚠️ ROA ({extracted_data['roa']:.2f}%) por debajo de niveles saludables para banca turca")
        
        # Verificar liquidez (SIN default)
        if 'liquidity_ratio' in extracted_data:
            liquidity_ratio = extracted_data['liquidity_ratio']
            if liquidity_ratio < 1:
                risk_factors.append(
                    f"🔴 Ratio de liquidez ({liquidity_ratio:.2f}) CRÍTICO - "
                    f"incapacidad para cubrir pasivos corrientes"
                )
        
        # Verificar LCR (SIN default)
        if 'lcr' in extracted_data:
            lcr = extracted_data['lcr']
            min_lcr = liquidity.get('lcr_minimum', 100.0)
            if lcr < min_lcr:
                risk_factors.append(
                    f"⚠️ LCR ({lcr:.1f}%) por debajo del mínimo BRSA ({min_lcr}%)"
                )
        
        # NPL Ratio (específico banca turca)
        if 'npl_ratio' in extracted_data:
            npl = extracted_data['npl_ratio']
            warning_level = thresholds.get('npl_ratio_warning', 3.0)
            if npl > warning_level:
                risk_factors.append(
                    f"⚠️ NPL Ratio ({npl:.2f}%) elevado - deterioro calidad crediticia"
                )
        
        return risk_factors
    
    
    def _call_azure_openai_for_insights(
        self, 
        extracted_data: Dict
    ) -> Dict[str, str]:
        """
        Llama a Azure OpenAI para insights contextuales sobre Garanti BBVA
        """
        try:
            prompt = self._prepare_llm_prompt(extracted_data)
            
            if hasattr(self.llm_agent, 'client'):
                response = self.llm_agent.client.chat.completions.create(
                    model=PREDICTOR_AGENT_CONFIG.get('model', 'gpt-4o'),
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Eres un analista financiero experto en banca turca y regulación BRSA. "
                                "Especializate en analizar Garanti BBVA (Türkiye Garanti Bankası), "
                                "una filial de BBVA España operando en Turquía bajo regulación BRSA."
                            )
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.3
                )
                
                insights = response.choices[0].message.content
                
                return {
                    'narrative_analysis': insights,
                    'model_used': PREDICTOR_AGENT_CONFIG.get('model'),
                    'context': 'Garanti BBVA (Turkey) under BRSA regulation'
                }
            else:
                return {'narrative_analysis': 'LLM no disponible'}
                
        except Exception as e:
            logger.warning(f"⚠️ Error llamando a Azure OpenAI: {e}")
            return {'narrative_analysis': f'Error: {e}'}
    
    
    def _prepare_llm_prompt(self, extracted_data: Dict) -> str:
        """
        Prepara prompt contextual para el LLM
        """
        prompt = f"""
        Analiza los siguientes datos financieros bancarios y proporciona insights cualitativos:
        
        DATOS EXTRAÍDOS:
        {json.dumps(extracted_data, indent=2)}
        
        Proporciona un análisis que incluya:
        1. Principales fortalezas financieras identificadas
        2. Áreas de preocupación o riesgo
        3. Tendencias observables en los ratios clave
        4. Recomendaciones estratégicas de alto nivel
        
        Sé conciso y enfócate en insights accionables.
        """
        return prompt
    
    
    def prepare_time_series_data(
        self, 
        extracted_data: Dict
    ) -> pd.DataFrame:
        """
        Convierte datos extraídos en series temporales para ML
        
        Args:
            extracted_data: Datos del agente LLM
        
        Returns:
            DataFrame con series temporales indexadas por fecha
        """
        logger.info("🔄 Preparando series temporales para ML...")
        
        try:
            # Busca datos históricos en los resultados de agentes
            time_series_data = []
            
            # Extrae series temporales de cada métrica
            metrics_to_extract = [
                'total_assets', 'total_liabilities', 'total_equity',
                'net_income', 'operating_cash_flow', 'roa', 'roe',
                'solvency_ratio', 'liquidity_ratio'
            ]
            
            # Lee datos históricos de los CSV generados por otros agentes
            historical_data = self._load_historical_data_from_csvs()
            
            if historical_data is not None and len(historical_data) > 0:
                logger.info(f"✅ Series temporales preparadas: {len(historical_data)} períodos")
                return historical_data
            else:
                # Si no hay datos históricos, intenta construirlos desde extracted_data
                logger.warning("⚠️ Datos históricos limitados, usando datos actuales")
                return self._construct_minimal_time_series(extracted_data)
                
        except Exception as e:
            logger.error(f"❌ Error preparando series temporales: {e}")
            raise
    
    def _ensure_date_column(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'date' not in df.columns:
            for cand in ['fecha', 'period', 'periodo', 'report_date', 'time']:
                if cand in df.columns:
                    df = df.rename(columns={cand: 'date'})
                    break
        if 'date' not in df.columns:
            raise ValueError("No se encontró una columna de fecha en el CSV")
        df['date'] = pd.to_datetime(df['date'])
        return df

    def _load_historical_data_from_csvs(self) -> Optional[pd.DataFrame]:
        """Carga datos históricos desde los CSVs generados por agentes"""
        import glob
        
        try:
            logger.info(" Buscando archivos CSV de agentes financieros...")
            
            # PATRONES FLEXIBLES (Recomendado)
            csv_patterns = {
                'balance': '*balance*data.csv',
                'income': '*income*data.csv',
                'cashflow': '*cashflow*data.csv',
                'equity': '*equity*data.csv'
            }
            
            dfs = []
            files_found = []
            
            for agent_name, pattern in csv_patterns.items():
                # Buscar archivos que coincidan con el patrón
                matching_files = glob.glob(os.path.join(self.output_dir, pattern))
                
                if matching_files:
                    # Usar el archivo más reciente si hay varios
                    filepath = max(matching_files, key=os.path.getmtime)
                    filename = os.path.basename(filepath)
                    
                    logger.info(f" {agent_name}: {filename}")
                    files_found.append(filename)
                    
                    try:
                        df = pd.read_csv(filepath)
                        df = self._ensure_date_column(df)
                        
                        # Añadir columna de origen
                        df['source'] = agent_name
                        dfs.append(df)
                        
                    except Exception as e:
                        logger.error(f" Error leyendo {filename}: {e}")
                else:
                    logger.warning(f" {agent_name}: No se encontró archivo (patrón: {pattern})")
            
            if not dfs:
                logger.warning(" No se encontraron CSVs históricos")
                logger.info("ℹ Se usarán solo datos actuales para predicción")
                return None
            
            logger.info(f" Total archivos cargados: {len(files_found)}")
            
            # Combinar todos los DataFrames
            combined_df = pd.concat(dfs, axis=0, ignore_index=True)
            combined_df = combined_df.sort_values('date')
            
            logger.info(f" Datos históricos cargados: {len(combined_df)} registros")
            return combined_df
            
        except Exception as e:
            logger.error(f" Error cargando CSVs: {e}")
            return None

    
    def _calculate_financial_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula ratios financieros clave
        """
        try:
            if 'net_income' in df.columns and 'total_assets' in df.columns:
                df['ROA'] = (df['net_income'] / df['total_assets']) * 100
            
            if 'net_income' in df.columns and 'total_equity' in df.columns:
                df['ROE'] = (df['net_income'] / df['total_equity']) * 100
            
            if 'total_equity' in df.columns and 'total_assets' in df.columns:
                df['ratio_solvencia'] = (df['total_equity'] / df['total_assets']) * 100
            
            if 'current_assets' in df.columns and 'current_liabilities' in df.columns:
                df['liquidez'] = df['current_assets'] / df['current_liabilities']
            
            return df
            
        except Exception as e:
            logger.warning(f"⚠️ Error calculando ratios: {e}")
            return df
    
    
    def _construct_minimal_time_series(self, extracted_data: Dict) -> pd.DataFrame:
        """Construye serie mínima con columna 'date' y 1 punto temporal."""
        numeric_pairs = {k: v for k, v in extracted_data.items() if isinstance(v, (int, float, float))}
        df = pd.DataFrame([numeric_pairs])
        df['date'] = pd.Timestamp.now().normalize()
        df = df.set_index('date').sort_index()
        logger.warning("Solo 1 período disponible - predicciones ML limitadas")
        return df
    
    
    def generate_ml_predictions(
        self, 
        time_series_data: pd.DataFrame,
        metrics_to_predict: List[str] = None
    ) -> Dict[str, Any]:
        """
        FASE 2: Genera predicciones ML con Prophet + XGBoost
        
        Args:
            time_series_data: DataFrame con series temporales
            metrics_to_predict: Lista de métricas a predecir
        
        Returns:
            Dict con predicciones ML completas
        """
        logger.info("🚀 FASE 2: Predicción ML iniciada...")
        
        if metrics_to_predict is None:
            metrics_to_predict = ['ROA', 'ratio_solvencia', 'liquidez', 'net_income']
        
        # Filtra métricas disponibles en los datos
        available_metrics = [
            m for m in metrics_to_predict 
            if m in time_series_data.columns
        ]
        
        if len(available_metrics) == 0:
            logger.error("❌ No hay métricas disponibles para predecir")
            return {
                'error': 'No metrics available',
                'predictions': {}
            }
        
        logger.info(f"📊 Prediciendo {len(available_metrics)} métricas: {available_metrics}")
        
        try:
            # Ejecuta predicciones ML
            ml_results = self.ml_agent.predict_financial_metrics(
                financial_data=time_series_data,
                bank_symbol=self.bank_symbol,
                metrics=available_metrics
            )
            
            self.results['ml_predictions'] = {
                'predictions': ml_results,
                'metrics_predicted': available_metrics,
                'bank_symbol': self.bank_symbol,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"✅ Predicciones ML completadas: {len(ml_results)} métricas")
            return self.results['ml_predictions']
            
        except Exception as e:
            logger.error(f"❌ Error en predicciones ML: {e}")
            return {
                'error': str(e),
                'predictions': {}
            }
    
    
    def synthesize_hybrid_analysis(self) -> Dict[str, Any]:
        """
        Sintetiza resultados de ambas fases en análisis híbrido final
        
        Combina:
        - Análisis cualitativo del LLM
        - Predicciones cuantitativas del ML
        - Recomendaciones integradas
        """
        logger.info("🎯 Sintetizando análisis híbrido...")
        
        try:
            hybrid_analysis = {
                'metadata': {
                    'analysis_date': datetime.now().isoformat(),
                    'bank_symbol': self.bank_symbol,
                    'methodology': 'Hybrid LLM + ML (Prophet + XGBoost)'
                },
                
                # Análisis cualitativo del LLM
                'qualitative_insights': self.results['llm_extraction'].get(
                    'qualitative_analysis', {}
                ),
                
                # Predicciones cuantitativas del ML
                'quantitative_forecasts': self._format_ml_forecasts(),
                
                # Análisis integrado
                'integrated_recommendations': self._generate_integrated_recommendations(),
                
                # Métricas de calidad
                'quality_metrics': {
                    'data_completeness': self.results['llm_extraction'].get(
                        'completeness', {}
                    ).get('completeness_percentage', 0),
                    'ml_models_used': ['Prophet', 'XGBoost', 'Ensemble'],
                    'confidence_level': self._calculate_confidence_level()
                }
            }
            
            self.results['hybrid_analysis'] = hybrid_analysis
            
            logger.info("✅ Análisis híbrido completado")
            return hybrid_analysis
            
        except Exception as e:
            logger.error(f"❌ Error sintetizando análisis: {e}")
            return {'error': str(e)}
    
    
    def _format_ml_forecasts(self) -> Dict[str, Any]:
        """
        Formatea predicciones ML en estructura legible
        """
        ml_preds = self.results['ml_predictions'].get('predictions', {})
        
        formatted = {}
        
        for metric, results in ml_preds.items():
            ensemble = results.get('ensemble', {})
            
            formatted[metric] = {
                'forecast_periods': len(ensemble.get('predictions', [])),
                'ensemble_predictions': ensemble.get('predictions', []),
                'confidence_intervals': {
                    'lower': ensemble.get('lower_bound', []),
                    'upper': ensemble.get('upper_bound', [])
                },
                'model_weights': ensemble.get('weights', {}),
                'recommendation': results.get('recommendation', 'ensemble')
            }
        
        return formatted
    
    
    def _generate_integrated_recommendations(self) -> Dict[str, Any]:
        """
        Genera recomendaciones integrando LLM + ML
        """
        recommendations = {
            'strategic': [],
            'tactical': [],
            'risk_mitigation': []
        }
        
        # Extrae insights del LLM
        llm_insights = self.results['llm_extraction'].get(
            'qualitative_analysis', {}
        ).get('llm_insights', {})
        
        # Extrae tendencias del ML
        ml_trends = self._analyze_ml_trends()
        
        # Combina ambos enfoques
        
        # Recomendaciones estratégicas (del LLM)
        if 'narrative_analysis' in llm_insights:
            recommendations['strategic'].append({
                'source': 'LLM Analysis',
                'insight': llm_insights['narrative_analysis'][:500]  # Primeras 500 chars
            })
        
        # Recomendaciones tácticas (del ML)
        for metric, trend in ml_trends.items():
            if trend['direction'] == 'declining':
                recommendations['tactical'].append({
                    'source': 'ML Forecast',
                    'metric': metric,
                    'insight': f"Se proyecta declive en {metric}. Considerar acciones correctivas.",
                    'forecast_trend': trend
                })
            elif trend['direction'] == 'improving':
                recommendations['tactical'].append({
                    'source': 'ML Forecast',
                    'metric': metric,
                    'insight': f"Tendencia positiva en {metric}. Mantener estrategia actual.",
                    'forecast_trend': trend
                })
        
        # Recomendaciones de mitigación de riesgo
        risk_assessment = self.results['llm_extraction'].get(
            'qualitative_analysis', {}
        ).get('risk_assessment', {})
        
        if risk_assessment.get('level') == 'high':
            recommendations['risk_mitigation'].append({
                'priority': 'HIGH',
                'insight': 'Score de riesgo elevado detectado. Revisar factores de riesgo.',
                'risk_factors': risk_assessment.get('factors', [])
            })
        
        return recommendations
    
    
    def _analyze_ml_trends(self) -> Dict[str, Dict]:
        """
        Analiza tendencias en las predicciones ML
        """
        trends = {}
        ml_forecasts = self._format_ml_forecasts()
        
        for metric, forecast in ml_forecasts.items():
            predictions = forecast.get('ensemble_predictions', [])
            
            if len(predictions) >= 2:
                # Calcula tendencia
                first_pred = predictions[0]
                last_pred = predictions[-1]
                change_pct = ((last_pred - first_pred) / first_pred) * 100
                
                trends[metric] = {
                    'direction': 'improving' if change_pct > 0 else 'declining',
                    'change_percentage': change_pct,
                    'forecast_range': {
                        'min': min(predictions),
                        'max': max(predictions)
                    }
                }
        
        return trends
    
    
    def _calculate_confidence_level(self) -> str:
        """
        Calcula nivel de confianza global del análisis
        """
        completeness = self.results['llm_extraction'].get(
            'completeness', {}
        ).get('completeness_percentage', 0)
        
        num_ml_predictions = len(
            self.results['ml_predictions'].get('predictions', {})
        )
        
        # Lógica de confianza
        if completeness >= 80 and num_ml_predictions >= 3:
            return 'HIGH'
        elif completeness >= 50 and num_ml_predictions >= 2:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    
    async def run_complete_hybrid_analysis(
        self, 
        agent_results: Dict[str, Any],
        bank_symbol: str = "GARAN.IS"
    ) -> Dict[str, Any]:
        """
        MÉTODO PRINCIPAL: Ejecuta análisis híbrido completo
        
        Args:
            agent_results: Resultados de los 4 agentes especializados
            bank_symbol: Símbolo del banco para datos externos
        
        Returns:
            Dict con análisis híbrido completo
        """
        logger.info("="*60)
        logger.info(" INICIANDO ANÁLISIS HÍBRIDO COMPLETO")
        logger.info("="*60)
        
        self.bank_symbol = bank_symbol
        
        # FASE 1: Extracción LLM (con await porque es async)
        llm_results = await self.extract_financial_data_from_agents(agent_results)
        
        # Prepara series temporales
        time_series_data = self.prepare_time_series_data(
            llm_results['extracted_data']
        )
        
        # FASE 2: Predicción ML (solo si hay suficientes datos)
        if len(time_series_data) >= 8:
            ml_results = self.generate_ml_predictions(time_series_data)
        else:
            logger.warning(
                f" Datos insuficientes para ML ({len(time_series_data)} períodos). "
                "Se requieren al menos 8 períodos."
            )
            # Usa solo análisis LLM
            self.results['ml_predictions'] = {
                'predictions': {},
                'note': 'Insufficient data for ML predictions'
            }
        
        # Síntesis híbrida
        hybrid_analysis = self.synthesize_hybrid_analysis()
        
        # Guarda resultados
        self._save_hybrid_results()
        
        logger.info("="*60)
        logger.info(" ANÁLISIS HÍBRIDO COMPLETADO EXITOSAMENTE")
        logger.info("="*60)
        
        return {
            'llm_extraction': llm_results,
            'ml_predictions': self.results['ml_predictions'],
            'hybrid_analysis': hybrid_analysis
        }

    
    def _save_hybrid_results(self):
        """
        Guarda resultados híbridos en JSON
        """
        try:
            output_file = os.path.join(
                self.output_dir, 
                f'hybrid_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            )
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            
            logger.info(f" Resultados guardados en: {output_file}")
            
        except Exception as e:
            logger.warning(f" Error guardando resultados: {e}")

async def test_hybrid_predictor():
    """Test del sistema híbrido"""
    print("Testing HybridPredictorAgent...")
    
    # Inicializar agente
    hybrid_agent = HybridPredictorAgent(bank_symbol='GARAN.IS')
    
    # ESTRUCTURA CORRECTA que coincide con lo que espera PredictorAgent
    mock_agent_results = {
        'balance': {
            'success': True,
            'agent': 'balance',
            'data': {
                'specific_answer': json.dumps({
                    'total_assets': 1500000,
                    'total_liabilities': 1200000,
                    'total_equity': 300000,
                    'current_assets': 800000,
                    'current_liabilities': 400000
                })
            }
        },
        'income': {
            'success': True,
            'agent': 'income',
            'data': {
                'specific_answer': json.dumps({
                    'net_income': 50000,
                    'total_revenue': 200000,
                    'operating_income': 75000
                })
            }
        },
        'cashflows': {
            'success': True,
            'agent': 'cashflows',
            'data': {
                'specific_answer': json.dumps({
                    'operating_cash_flow': 60000,
                    'investing_cash_flow': -20000,
                    'financing_cash_flow': -10000
                })
            }
        },
        'equity': {
            'success': True,
            'agent': 'equity',
            'data': {
                'specific_answer': json.dumps({
                    'shareholders_equity': 300000,
                    'share_capital': 200000,
                'retained_earnings': 100000
                })
            }
        }
    }
    
    # Ejecutar análisis completo
    results = await hybrid_agent.run_complete_hybrid_analysis(
        agent_results=mock_agent_results,
        bank_symbol='GARAN.IS'
    )
    
    print("\n RESULTADOS:")
    print(f"LLM Extraction: {len(results['llm_extraction'])} keys")
    print(f"ML Predictions: {len(results['ml_predictions'].get('predictions', {}))} metrics")
    print(f"Hybrid Analysis: {results['hybrid_analysis'].get('metadata', {})}")
    
    print("\n Test completado!")
    return results



# Ejecutar con asyncio
if __name__ == "__main__":
    asyncio.run(test_hybrid_predictor())
