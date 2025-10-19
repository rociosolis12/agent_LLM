import os
import json
import logging
import pandas as pd
from datetime import datetime
from update_predictor_agent import EvolutionaryPredictorAgent
from validation_module import WalkForwardValidator
from regulatory_config_agent import RegulatoryConfigAgent
from predictor_agent import PredictorAgent

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

    async def run_full_analysis(self, agent_results=None):
        logger.info("==== HÍBRIDO - Análisis completo ====")
        # LLM extraction & qualitative - si tienes agent_results
        llm_qualitative = {}
        if agent_results:
            # Extraer y estructurar datos cualitativos de los agentes
            llm_qualitative = {
                'balance_insights': agent_results.get('balance', {}) if agent_results else {},
                'income_insights': agent_results.get('income', {}) if agent_results else {},
                'cashflow_insights': agent_results.get('cashflows', {}) if agent_results else {},
                'equity_insights': agent_results.get('equity', {}) if agent_results else {},
                'timestamp': datetime.now().isoformat()
            }
        # ML: carga o genera predicciones robustas
        ml_preds = self.load_ml_predictions()
        # Validación
        validation_results = self.validate_ml_predictions(ml_preds)
        # Síntesis final
        analysis = self.synthesize_hybrid_analysis(llm_qualitative, ml_preds, validation_results)
        # Exporta resultados
        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, f"hybrid_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"), 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        logger.info(f"Resultados exportados a {self.output_dir}")
        return analysis

if __name__ == "__main__":
    import asyncio
    # Supón que agent_results se genera fuera (de agentes especializados o entrada manual)
    asyncio.run(HybridPredictorAgent(bank_symbol="GARAN.IS").run_full_analysis(agent_results=None))
