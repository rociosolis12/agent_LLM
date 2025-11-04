import os
import json
import logging
import pandas as pd
from datetime import datetime
from update_predictor_agent import EvolutionaryPredictorAgent
from validation_module import WalkForwardValidator
from regulatory_config_agent import RegulatoryConfigAgent
from predictor_agent import PredictorAgent
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridPredictorAgent:
    def __init__(self,
                 bank_symbol="GARAN.IS",
                 alpha_vantage_key=None,
                 jurisdiction="TR",
                 parent_bank="BBVA",
                 use_regulatory_config=True,
                 data_dir="./data_outputs",
                 output_dir="./data_outputs"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.bank_symbol = bank_symbol
        self.jurisdiction = jurisdiction
        self.parent_bank = parent_bank
        self.alpha_vantage_key = alpha_vantage_key
        self.results = {}
        # Core modules
        self.evo_agent = EvolutionaryPredictorAgent(alpha_vantage_key=alpha_vantage_key)
        self.validator = WalkForwardValidator(self.evo_agent)
        self.llm_agent = PredictorAgent()
        self.regulatory_agent = RegulatoryConfigAgent() if use_regulatory_config else None
        self.use_regulatory_config = use_regulatory_config
        self.regulatory_config = None
        if self.use_regulatory_config:
            self._load_regulatory_config()

    def _load_regulatory_config(self):
        try:
            logger.info(f"Obteniendo configuración BRSA...")
            self.regulatory_config = self.regulatory_agent.get_regulatory_thresholds(
                bank_symbol=self.bank_symbol,
                jurisdiction=self.jurisdiction,
                bank_type='international')
            self.results['regulatory_config'] = self.regulatory_config
        except Exception as e:
            logger.warning(f"Error config BRSA: {e}")
            self.regulatory_config = self._get_fallback_regulatory_config()
            self.results['regulatory_config'] = self.regulatory_config

    def _get_fallback_regulatory_config(self):
        logger.info("Usando configuración BRSA hardcoded")
        return {
            # ... (igual a la lógica ya en tu código actual)
        }

    def load_ml_predictions(self):
        csv_path = os.path.join(self.data_dir, "evolutionary_predictions.csv")
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)
        else:
            logger.info("No existe CSV con predicciones. Ejecutando predictor evolutivo...")
            self.evo_agent.run_predictor(bank_symbol=self.bank_symbol)
            return pd.read_csv(csv_path)

    def validate_ml_predictions(self, df_ml):
        # Validar las principales métricas financieras
        metricas_validar = ["ROA", "ratio_solvencia", "liquidez"]
        validaciones = {}
        for metrica in metricas_validar:
            if metrica in df_ml.columns:
                resultado_val = self.validator.walkforwardvalidate(df_ml[metrica], metrica, nsplits=8, mintrainsize=8)
                validaciones[metrica] = resultado_val
                logger.info(f"Validación {metrica}: MAE={resultado_val['mae']:.4f}, R2={resultado_val['r2']:.4f}")
        return validaciones

    def synthesize_hybrid_analysis(self, llm_qualitative, ml_predictions, validation_results):
        # Integra recomendaciones, riesgos y síntesis en resultado final.
        analysis = {
            'metadata': {
                'analysis_date': datetime.now().isoformat(),
                'bank_symbol': self.bank_symbol,
                'methodology': 'Hybrid LLM + ML (Prophet + XGBoost)'
            },
            'regulatory_config': self.regulatory_config,
            'qualitative_insights': llm_qualitative,
            'ml_predictions': ml_predictions.to_dict(orient='records'),
            'validation_results': validation_results,
            'integrated_recommendations': self._generate_integrated_recommendations(llm_qualitative, ml_predictions, validation_results)
        }
        self.results['hybrid_analysis'] = analysis
        return analysis

    def _generate_integrated_recommendations(self, llm_qualitative, ml_predictions, validation_results):
        # Aquí combinas insights LLM, tendencias ML, alertas regulatorias y validaciones
        recommendations = {
            'strategic': [],
            'tactical': [],
            'risk_mitigation': []
        }
        # Ejemplo: Estrategias desde LLM
        if llm_qualitative and 'narrative_analysis' in llm_qualitative:
            recommendations['strategic'].append({
                'source': 'LLM',
                'text': llm_qualitative['narrative_analysis']
            })
        # Ejemplo: Tendencias ML
        for metrica, val in validation_results.items():
            if val['mae'] > 0.5:
                recommendations['risk_mitigation'].append({
                    'metric': metrica,
                    'alert': 'Elevado error, revisar modelo o datos históricos'
                })
            else:
                recommendations['tactical'].append({
                    'metric': metrica,
                    'recommendation': 'Evolución estable, mantener estrategia actual'
                })
        # Ejemplo: Factores regulatorios
        if self.regulatory_config:
            recommendations['strategic'].append({
                'source': 'Regulatory',
                'text': f"CET1 mínimo: {self.regulatory_config['capital_ratios']['cet1_minimum']}%"
            })
        return recommendations

    async def run_full_analysis(self, agent_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Ejecuta análisis híbrido completo integrando datos de agentes
        
        Args:
            agent_results: Diccionario con resultados de balance, income, cashflows, equity
            
        Returns:
            Diccionario con análisis completo híbrido
        """
        logger.info("=" * 80)
        logger.info("🔬 EJECUTANDO ANÁLISIS HÍBRIDO COMPLETO")
        logger.info("=" * 80)
        
        try:
            # ✅ PASO 1: Procesar insights cualitativos de agentes
            logger.info("PASO 1/4: Procesando insights cualitativos de agentes...")
            if agent_results:
                llm_qualitative = await self._process_agent_insights(agent_results)
                logger.info(f"   ✅ Insights procesados: {len(llm_qualitative)} categorías")
            else:
                logger.warning("   ⚠️ No hay agent_results, usando insights vacíos")
                llm_qualitative = {
                    'balance_insights': {},
                    'income_insights': {},
                    'cashflow_insights': {},
                    'equity_insights': {}
                }
            
            # ✅ PASO 2: Cargar predicciones ML
            logger.info("PASO 2/4: Cargando predicciones ML...")
            try:
                ml_preds = self.load_ml_predictions()
                logger.info(f"   ✅ Predicciones cargadas: {len(ml_preds)} registros")
            except Exception as e:
                logger.error(f"   ❌ Error cargando predicciones: {e}")
                ml_preds = pd.DataFrame()
            
            # ✅ PASO 3: Validar predicciones ML
            logger.info("PASO 3/4: Validando predicciones ML...")
            try:
                validation_results = self.validate_ml_predictions(ml_preds)
                logger.info(f"   ✅ Validación completada: {len(validation_results)} métricas")
            except Exception as e:
                logger.warning(f"   ⚠️ Error en validación: {e}")
                validation_results = {}
            
            # ✅ PASO 4: Síntesis final e integración
            logger.info("PASO 4/4: Sintetizando análisis híbrido...")
            analysis = self.synthesize_hybrid_analysis(llm_qualitative, ml_preds, validation_results)
            
            # ✅ Exportar resultados
            logger.info("💾 Exportando resultados...")
            os.makedirs(self.output_dir, exist_ok=True)
            
            output_filename = f"hybrid_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            output_path = os.path.join(self.output_dir, output_filename)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            
            logger.info(f" Resultados exportados: {output_path}")
            logger.info("=" * 80)
            logger.info(" ANÁLISIS HÍBRIDO COMPLETADO EXITOSAMENTE")
            logger.info("=" * 80)
            
            return analysis
            
        except Exception as e:
            logger.error(f" Error crítico en análisis híbrido: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    async def _process_agent_insights(
        self, 
        agent_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Procesa insights cualitativos de los agentes financieros
        
        Args:
            agent_results: Diccionario con resultados de agentes
            
        Returns:
            Diccionario con insights estructurados por categoría
        """
        logger.info("🧠 Procesando insights cualitativos de agentes...")
        
        insights = {
            'balance_insights': {},
            'income_insights': {},
            'cashflow_insights': {},
            'equity_insights': {}
        }
        
        # Balance Agent Insights
        if 'balance' in agent_results and agent_results['balance']:
            balance = agent_results['balance']
            logger.info("   • Balance Agent: extrayendo insights...")
            insights['balance_insights'] = {
                'cet1_ratio': balance.get('cet1_ratio', 'N/A'),
                'solvency_ratio': balance.get('solvency_ratio', 'N/A'),
                'liquidity_ratio': balance.get('liquidity_ratio', 'N/A'),
                'key_findings': balance.get('analysis', {}).get('findings', []),
                'status': balance.get('analysis', {}).get('status', 'N/A')
            }
            logger.info(f"       Extraídos: {len(insights['balance_insights']['key_findings'])} findings")
        
        #  Income Agent Insights
        if 'income' in agent_results and agent_results['income']:
            income = agent_results['income']
            logger.info("   • Income Agent: extrayendo insights...")
            insights['income_insights'] = {
                'roa': income.get('roa', 'N/A'),
                'roe': income.get('roe', 'N/A'),
                'net_interest_margin': income.get('net_interest_margin', 'N/A'),
                'profitability_trend': income.get('profitability_trend', 'stable'),
                'key_findings': income.get('analysis', {}).get('findings', []),
                'status': income.get('analysis', {}).get('status', 'N/A')
            }
            logger.info(f"       Extraídos: {len(insights['income_insights']['key_findings'])} findings")
        
        # CashFlows Agent Insights
        if 'cashflows' in agent_results and agent_results['cashflows']:
            cashflows = agent_results['cashflows']
            logger.info("   • CashFlows Agent: extrayendo insights...")
            insights['cashflow_insights'] = {
                'liquidity_status': cashflows.get('liquidity_status', 'N/A'),
                'operating_efficiency': cashflows.get('operating_efficiency', 'N/A'),
                'cash_conversion': cashflows.get('cash_conversion_ratio', 'N/A'),
                'key_findings': cashflows.get('analysis', {}).get('findings', []),
                'status': cashflows.get('analysis', {}).get('status', 'N/A')
            }
            logger.info(f"      Extraídos: {len(insights['cashflow_insights']['key_findings'])} findings")
        
        #  Equity Agent Insights
        if 'equity' in agent_results and agent_results['equity']:
            equity = agent_results['equity']
            logger.info("   • Equity Agent: extrayendo insights...")
            insights['equity_insights'] = {
                'dividend_policy': equity.get('dividend_policy', 'N/A'),
                'capital_structure': equity.get('capital_structure', 'N/A'),
                'book_value_per_share': equity.get('book_value_per_share', 'N/A'),
                'key_findings': equity.get('analysis', {}).get('findings', []),
                'status': equity.get('analysis', {}).get('status', 'N/A')
            }
            logger.info(f"       Extraídos: {len(insights['equity_insights']['key_findings'])} findings")
        
        logger.info(f" Insights procesados: {len(insights)} categorías")
        return insights

    def _generate_integrated_recommendations(
        self,
        llm_qualitative: Dict[str, Any],
        ml_predictions: pd.DataFrame,
        validation_results: Dict[str, Any]
    ) -> Dict[str, list]:
        """
        Genera recomendaciones integradas desde múltiples fuentes:
        1. Insights cualitativos de agentes
        2. Predicciones ML
        3. Resultados de validación
        4. Configuración regulatoria
        
        Args:
            llm_qualitative: Insights de agentes
            ml_predictions: DataFrame con predicciones ML
            validation_results: Diccionario con métricas de validación
            
        Returns:
            Diccionario con recomendaciones categorizadas
        """
        logger.info("💡 Generando recomendaciones integradas...")
        
        recommendations = {
            'strategic': [],
            'tactical': [],
            'risk_mitigation': []
        }
        
        #  1. RECOMENDACIONES DESDE BALANCE AGENT
        if llm_qualitative.get('balance_insights'):
            balance = llm_qualitative['balance_insights']
            
            for finding in balance.get('key_findings', []):
                recommendations['strategic'].append({
                    'source': 'Balance Agent',
                    'insight': finding,
                    'priority': 'high',
                    'category': 'solvency'
                })
            
            # Recomendación basada en CET1
            cet1 = balance.get('cet1_ratio', None)
            if cet1 and self.regulatory_config:
                min_cet1 = self.regulatory_config.get('capital_ratios', {}).get('cet1_minimum', 4.5)
                if float(cet1) < min_cet1:
                    recommendations['risk_mitigation'].append({
                        'source': 'Balance Agent',
                        'insight': f'CET1 ratio ({cet1}%) below regulatory minimum ({min_cet1}%)',
                        'priority': 'critical',
                        'action': 'Increase capital immediately'
                    })
        
        #  2. RECOMENDACIONES DESDE INCOME AGENT
        if llm_qualitative.get('income_insights'):
            income = llm_qualitative['income_insights']
            
            for finding in income.get('key_findings', []):
                recommendations['tactical'].append({
                    'source': 'Income Agent',
                    'insight': finding,
                    'priority': 'medium',
                    'category': 'profitability'
                })
            
            # Recomendación basada en ROA
            roa = income.get('roa', None)
            if roa:
                roa_float = float(roa)
                if roa_float < 0.5:
                    recommendations['risk_mitigation'].append({
                        'source': 'Income Agent',
                        'insight': f'ROA ({roa}%) is critically low',
                        'priority': 'high',
                        'action': 'Improve operational efficiency'
                    })
        
        #  3. RECOMENDACIONES DESDE CASHFLOWS AGENT
        if llm_qualitative.get('cashflow_insights'):
            cashflows = llm_qualitative['cashflow_insights']
            
            for finding in cashflows.get('key_findings', []):
                recommendations['risk_mitigation'].append({
                    'source': 'CashFlows Agent',
                    'insight': finding,
                    'priority': 'medium',
                    'category': 'liquidity'
                })
        
        #  4. RECOMENDACIONES DESDE EQUITY AGENT
        if llm_qualitative.get('equity_insights'):
            equity = llm_qualitative['equity_insights']
            
            for finding in equity.get('key_findings', []):
                recommendations['strategic'].append({
                    'source': 'Equity Agent',
                    'insight': finding,
                    'priority': 'low',
                    'category': 'capital_structure'
                })
        
        #  5. RECOMENDACIONES DESDE VALIDACIÓN ML
        if validation_results:
            for metric, val in validation_results.items():
                mae = val.get('mae', 0) if isinstance(val, dict) else val
                
                if mae > 0.1:  # Umbral de error alto
                    recommendations['risk_mitigation'].append({
                        'source': 'ML Validation',
                        'insight': f'{metric} prediction error is high (MAE: {mae:.4f})',
                        'priority': 'medium',
                        'action': 'Review model or add more historical data'
                    })
        
        #  6. RECOMENDACIONES REGULATORIAS
        if self.regulatory_config:
            recommendations['strategic'].append({
                'source': 'Regulatory',
                'insight': f"CET1 minimum: {self.regulatory_config.get('capital_ratios', {}).get('cet1_minimum', 4.5)}%",
                'priority': 'critical',
                'category': 'compliance'
            })
        
        #  LOG RESUMEN
        total_recs = sum(len(v) for v in recommendations.values())
        logger.info(f" Recomendaciones generadas: {total_recs} total")
        logger.info(f"   Strategic: {len(recommendations['strategic'])}")
        logger.info(f"   Tactical: {len(recommendations['tactical'])}")
        logger.info(f"   Risk Mitigation: {len(recommendations['risk_mitigation'])}")
        
        return recommendations


if __name__ == "__main__":
    import asyncio
    # Supón que agent_results se genera fuera (de agentes especializados o entrada manual)
    asyncio.run(HybridPredictorAgent(bank_symbol="GARAN.IS").run_full_analysis(agent_results=None))
