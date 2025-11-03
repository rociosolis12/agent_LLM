from prophet import Prophet
import yfinance as yf
import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
from xgboost import XGBRegressor
import warnings
import os
from datetime import datetime, timedelta
import logging
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any, Optional  


# Cargar .env desde el directorio raíz del proyecto
project_root = Path(__file__).parent.parent
env_path = project_root / ".env"
load_dotenv(env_path)
os.chdir(project_root)

if not env_path.exists():
    print(f"Archivo .env no encontrado en {env_path}")

# Cargar las API keys
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
AZURE_EMBEDDING_MODEL = os.getenv("AZURE_EMBEDDING_MODEL", "text-embedding-3-small")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvolutionaryPredictorAgent:
    def __init__(self, alpha_vantage_key=None):
        """
        Predictor Agent Evolucionado con Prophet + XGBoost + APIs externas
        """
        # Alpha Vantage API (opcional - funciona sin ella)
        self.alpha_vantage_key = alpha_vantage_key or os.getenv('ALPHA_VANTAGE_KEY', 'demo')

        # Almacenamiento de modelos
        self.models = {
            'prophet': {},
            'xgboost': {},
            'ensemble': {}
        }

        # Configuración por defecto
        self.default_config = {
            'prophet_weight': 0.7,
            'xgboost_weight': 0.3,
            'validation_splits': 8,
            'prediction_periods': 4
        }

        logger.info("EvolutionaryPredictorAgent inicializado correctamente")

    def fetch_external_data(self, bank_symbol="GARAN.IS", period="5y"):

        """Obtiene datos externos de mercado financiero"""
        try:
            logger.info(f" Obteniendo datos externos para {bank_symbol}")
            ticker = yf.Ticker(bank_symbol)
            historical = ticker.history(period=period)
            if historical.empty:
                logger.warning(" No se pudieron obtener datos de Yahoo Finance")
                return self._generate_mock_market_data()

            historical['returns'] = historical['Close'].pct_change()
            historical['volatility'] = historical['returns'].rolling(30).std()
            historical['sma_20'] = historical['Close'].rolling(20).mean()
            historical['rsi'] = self._calculate_rsi(historical['Close'])

            external_data = {
                'stock_price': historical['Close'].resample('Q').last(),
                'volume': historical['Volume'].resample('Q').mean(),
                'volatility': historical['volatility'].resample('Q').mean(),
                'returns': historical['returns'].resample('Q').sum(),
                'rsi': historical['rsi'].resample('Q').last()
            }

            logger.info(" Datos externos obtenidos exitosamente")
            logger.info(f"NaNs en la serie external_stock_price: {external_data['stock_price'].isna().sum()}")
            logger.info(f"Primeros valores: {external_data['stock_price'].head(10)}")
            return external_data

        except Exception as e:
            logger.warning(f"Error obteniendo datos externos: {e}")
            return self._generate_mock_market_data()

    def _calculate_rsi(self, prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _generate_mock_market_data(self):
        """Genera datos simulados si las APIs fallan"""
        logger.info(" Generando datos de mercado simulados")
        periods = 20
        dates = pd.date_range('2020-01-01', periods=periods, freq='Q')
        np.random.seed(42)
        base_price = 100
        returns = np.random.normal(0.02, 0.15, periods)
        prices = [base_price]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        return {
            'stock_price': pd.Series(prices[1:], index=dates),
            'volume': pd.Series(np.random.uniform(1_000_000, 5_000_000, periods), index=dates),
            'volatility': pd.Series(np.abs(returns), index=dates),
            'returns': pd.Series(returns, index=dates),
            'rsi': pd.Series(np.random.uniform(30, 70, periods), index=dates)
        }

    def prepare_data_for_prophet(self, financial_series, external_data=None, metric_name="metric", frequency='Y'):
        df = pd.DataFrame({
            'ds': financial_series.index,
            'y': financial_series.values
        })
        df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)
        
        # NUEVO: Validación para datos anuales
        if frequency == 'Y' and len(df) < 5:
            logger.warning(f"⚠️ Solo {len(df)} años de datos para {metric_name}")
            return df  # Prophet funcionará pero con alta incertidumbre
        
        # Integración externa adaptada a frecuencia anual
        if external_data is not None:
            for key, series in external_data.items():
                ext_df = pd.DataFrame({
                    'ds': pd.to_datetime(series.index).tz_localize(None),
                    f'external_{key}': series.values
                })
                # Resample a anual si viene en otra frecuencia
                ext_df = ext_df.set_index('ds').resample('Y').last().reset_index()
                df = pd.merge(df, ext_df, on='ds', how='left')
                df[f'external_{key}'] = df[f'external_{key}'].ffill().bfill().fillna(0)
        
        return df

    def prophet_prediction(self, df, metric_name, periods=4, frequency='Y'):
        try:
            # Ajustar configuración Prophet para datos anuales
            model = Prophet(
                growth='linear',
                yearly_seasonality=False,  
                changepoint_prior_scale=0.1,  
                interval_width=0.95
            )
            
            ext_cols = [c for c in df.columns if c.startswith('external_')]
            for col in ext_cols:
                model.add_regressor(col)
            
            model.fit(df)
            
            # Crear futuro con frecuencia anual
            future = model.make_future_dataframe(periods=periods, freq='Y')
            
            # Propagar regresores externos
            for col in ext_cols:
                future[col] = list(df[col]) + [df[col].iloc[-1]] * periods
            
            forecast = model.predict(future)
            preds = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
            
            return {
                'model': model,
                'forecast': forecast,
                'predictions': preds['yhat'].tolist(),
                'lower_bound': preds['yhat_lower'].tolist(),
                'upper_bound': preds['yhat_upper'].tolist()
            }
        except Exception as e:
            logger.error(f" Error Prophet con datos anuales: {e}")
            return self._fallback_prediction(df, periods)

    def _fallback_prediction(self, df, periods):
        x = np.arange(len(df))
        coeffs = np.polyfit(x, df['y'], 1)
        future_x = np.arange(len(df), len(df) + periods)
        preds = np.polyval(coeffs, future_x)
        return {'predictions': preds.tolist(),
                'lower_bound': (preds * 0.95).tolist(),
                'upper_bound': (preds * 1.05).tolist()}

    def create_features_for_xgboost(self, df):
        df_feat = df.copy()
        df_feat['lag_1'] = df_feat['y'].shift(1)
        df_feat['lag_4'] = df_feat['y'].shift(4)
        df_feat['trend'] = np.arange(len(df_feat))
        df_feat['year'] = df_feat['ds'].dt.year
        df_feat = df_feat.dropna()
        return df_feat

    def xgboost_prediction(self, df, metric_name, periods=4):
        try:
            df_feat = self.create_features_for_xgboost(df)
            if len(df_feat) < 8:
                return self._simple_xgb_prediction(df, periods)
            X = df_feat.drop(columns=['ds', 'y'])
            y = df_feat['y']
            model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1,
                                 subsample=0.8, random_state=42)
            model.fit(X, y)
            preds = model.predict(X.tail(periods))
            return {'model': model, 'predictions': preds.tolist()}
        except Exception as e:
            logger.error(f" Error XGBoost: {e}")
            return self._simple_xgb_prediction(df, periods)

    def _simple_xgb_prediction(self, df, periods):
        last_val = df['y'].iloc[-1]
        trend = (df['y'].iloc[-1] - df['y'].iloc[0]) / len(df)
        preds = [last_val + (trend * (i+1)) for i in range(periods)]
        return {'model': None, 'predictions': preds}

    def ensemble_prediction(self, prophet_results, xgb_results):
        pw, xw = self.default_config['prophet_weight'], self.default_config['xgboost_weight']
        p_preds, x_preds = prophet_results['predictions'], xgb_results['predictions']
        combined = [(p * pw + x * xw) for p, x in zip(p_preds, x_preds)]
        lowers = [(p * pw * 0.95 + x * xw * 0.95) for p, x in zip(p_preds, x_preds)]
        uppers = [(p * pw * 1.05 + x * xw * 1.05) for p, x in zip(p_preds, x_preds)]
        return {'predictions': combined, 'lower_bound': lowers, 'upper_bound': uppers}

    def export_predictions_to_csv(self, all_results, filename="evolutionary_predictions.csv"):
        """Exporta resultados para el HybridPredictorAgent"""
        output_dir = os.getenv("DATA_OUTPUT_DIR", "./data_outputs")
        os.makedirs(output_dir, exist_ok=True)
        all_rows = []
        for metric, res in all_results.items():
            preds = res['ensemble']['predictions']
            for i, val in enumerate(preds):
                all_rows.append({
                    'metric': metric,
                    'timestep': i + 1,
                    'prediction': val,
                    'lower': res['ensemble']['lower_bound'][i],
                    'upper': res['ensemble']['upper_bound'][i],
                    'date_generated': datetime.now().strftime("%Y-%m-%d")
                })
        df_out = pd.DataFrame(all_rows)
        path = os.path.join(output_dir, filename)
        df_out.to_csv(path, index=False)
        logger.info(f" Predicciones exportadas correctamente en {path}")
        return path

    def predict_financial_metrics(
        self, 
        financial_data: Optional[pd.DataFrame] = None,
        agent_results: Optional[Dict[str, Any]] = None,
        bank_symbol: str = "GARAN.IS", 
        metrics: list = None,
        periods: int = 4
    ):
        """
        Genera predicciones financieras desde DataFrame o directamente desde agentes
        
        Args:
            financial_data: DataFrame con datos históricos (opcional)
            agent_results: Diccionario con resultados de agentes (opcional)
            bank_symbol: Símbolo del banco para datos externos
            metrics: Lista de métricas a predecir
            periods: Períodos futuros a predecir
            
        Returns:
            Diccionario con predicciones por métrica
        """
        logger.info(f"🔮 Generando predicciones para {periods} períodos...")
        
        # PRIORIDAD 1: Usar datos de agentes si están disponibles
        if agent_results is not None:
            logger.info("✅ Usando datos de agentes financieros")
            financial_data = self.extract_financial_data_from_agents(agent_results)
            
        # PRIORIDAD 2: Usar DataFrame proporcionado
        elif financial_data is not None and not financial_data.empty:
            logger.info("✅ Usando DataFrame proporcionado")
            financial_data = financial_data.copy()
            
        # PRIORIDAD 3: Generar datos mock (fallback)
        else:
            logger.warning("⚠️ No hay datos disponibles, generando datos mock")
            dates = pd.date_range('2019-01-01', periods=5, freq='Y')
            financial_data = pd.DataFrame({
                'ROA': 1.2 + 0.1 * np.sin(np.arange(5)) + np.random.normal(0, 0.05, 5),
                'ratio_solvencia': 12.5 + 0.5 * np.sin(np.arange(5)) + np.random.normal(0, 0.2, 5)
            }, index=dates)
        
        # Validar que hay datos
        if financial_data.empty:
            logger.error("❌ No se pudieron obtener datos financieros")
            return {}
        
        # Determinar métricas a predecir
        if metrics is None:
            metrics = [col for col in financial_data.columns if col not in ['periodo', 'date', 'ds']]
            logger.info(f"📊 Métricas detectadas automáticamente: {metrics}")
        
        # Obtener datos externos de mercado
        external_data = self.fetch_external_data(bank_symbol)
        
        # Generar predicciones para cada métrica
        results = {}
        for metric in metrics:
            if metric not in financial_data.columns:
                logger.warning(f"⚠️ Métrica {metric} no encontrada en datos")
                continue
            
            series = financial_data[metric].dropna()
            
            # Validar mínimo de datos
            if len(series) < 3:
                logger.warning(f"⚠️ Datos insuficientes para {metric}: {len(series)} < 3")
                continue
            
            logger.info(f"  ➤ Prediciendo: {metric} ({len(series)} datos históricos)")
            
            # Preparar datos para Prophet
            df = self.prepare_data_for_prophet(series, external_data, metric, frequency='Y')
            
            # Predicción Prophet
            prop = self.prophet_prediction(df, metric, periods=periods, frequency='Y')
            
            # Predicción XGBoost
            xgb = self.xgboost_prediction(df, metric, periods=periods)
            
            # Ensemble (promedio ponderado)
            ens = self.ensemble_prediction(prop, xgb)
            
            results[metric] = {
                'prophet': prop, 
                'xgboost': xgb, 
                'ensemble': ens
            }
        
        # Exportar predicciones si hay resultados
        if results:
            self.export_predictions_to_csv(results)
            logger.info(f" Predicciones generadas para {len(results)} métricas")
        else:
            logger.warning(" No se generaron predicciones")
        
        return results

    def extract_financial_data_from_agents(self, agent_results: Dict[str, Any]) -> pd.DataFrame:
        """
        Extrae series temporales desde los agentes financieros mejorados
        
        Args:
            agent_results: Diccionario con resultados de balance, income, cashflows, equity
            
        Returns:
            DataFrame con series temporales por año
        """
        logger.info(" Extrayendo datos financieros de agentes...")
        
        financial_data = {}
        
        # 1. Balance Agent - Ratios de solvencia y liquidez
        if 'balance' in agent_results:
            balance = agent_results['balance']
            financial_data.update(self._extract_balance_metrics(balance))
            
        # 2. Income Agent - ROA, ROE, márgenes
        if 'income' in agent_results:
            income = agent_results['income']
            financial_data.update(self._extract_income_metrics(income))
            
        # 3. CashFlows Agent - Liquidez operativa
        if 'cashflows' in agent_results:
            cashflows = agent_results['cashflows']
            financial_data.update(self._extract_cashflow_metrics(cashflows))
            
        # 4. Equity Agent - Capitalización y dividendos
        if 'equity' in agent_results:
            equity = agent_results['equity']
            financial_data.update(self._extract_equity_metrics(equity))
        
        # Convertir a DataFrame
        if not financial_data:
            logger.warning(" No se extrajeron datos de los agentes")
            return pd.DataFrame()
            
        df = pd.DataFrame(financial_data)
        
        # Asegurar columna de periodo si no existe
        if 'periodo' not in df.columns and len(df) > 0:
            current_year = datetime.now().year
            df['periodo'] = range(current_year - len(df) + 1, current_year + 1)
            
        logger.info(f" Datos extraídos: {len(df)} periodos, {len(df.columns)} métricas")
        return df
    
    def _extract_balance_metrics(self, balance_data: Dict) -> Dict:
        """Extrae métricas del Balance Agent"""
        metrics = {}
        
        # Buscar ratios de solvencia históricos (prueba diferentes keys)
        if 'solvency_ratios' in balance_data:
            metrics['ratio_solvencia'] = balance_data['solvency_ratios']
        elif 'capital_ratio' in balance_data:
            metrics['ratio_solvencia'] = balance_data['capital_ratio']
        elif 'solvency_ratio' in balance_data:
            metrics['ratio_solvencia'] = balance_data['solvency_ratio']
            
        # Liquidez
        if 'liquidity_ratio' in balance_data:
            metrics['liquidez'] = balance_data['liquidity_ratio']
        elif 'lcr' in balance_data:
            metrics['liquidez'] = balance_data['lcr']
        elif 'liquidity' in balance_data:
            metrics['liquidez'] = balance_data['liquidity']
            
        # CET1 (capital tier 1)
        if 'cet1_ratio' in balance_data:
            metrics['cet1'] = balance_data['cet1_ratio']
        elif 'cet1' in balance_data:
            metrics['cet1'] = balance_data['cet1']
            
        logger.info(f"  ✓ Balance: {len(metrics)} métricas extraídas")
        return metrics
    
    def _extract_income_metrics(self, income_data: Dict) -> Dict:
        """Extrae métricas del Income Agent"""
        metrics = {}
        
        # ROA
        if 'roa' in income_data:
            metrics['ROA'] = income_data['roa']
        elif 'return_on_assets' in income_data:
            metrics['ROA'] = income_data['return_on_assets']
            
        # ROE
        if 'roe' in income_data:
            metrics['ROE'] = income_data['roe']
        elif 'return_on_equity' in income_data:
            metrics['ROE'] = income_data['return_on_equity']
            
        # Margen de interés neto
        if 'nim' in income_data:
            metrics['margen_interes'] = income_data['nim']
        elif 'net_interest_margin' in income_data:
            metrics['margen_interes'] = income_data['net_interest_margin']
        elif 'interest_margin' in income_data:
            metrics['margen_interes'] = income_data['interest_margin']
            
        logger.info(f"  ✓ Income: {len(metrics)} métricas extraídas")
        return metrics
    
    def _extract_cashflow_metrics(self, cashflow_data: Dict) -> Dict:
        """Extrae métricas del CashFlows Agent"""
        metrics = {}
        
        # Flujo de caja operativo
        if 'operating_cashflow' in cashflow_data:
            metrics['flujo_operativo'] = cashflow_data['operating_cashflow']
        elif 'operating_cash_flow' in cashflow_data:
            metrics['flujo_operativo'] = cashflow_data['operating_cash_flow']
            
        # Ratio de conversión de efectivo
        if 'cash_conversion_ratio' in cashflow_data:
            metrics['conversion_efectivo'] = cashflow_data['cash_conversion_ratio']
        elif 'cash_conversion' in cashflow_data:
            metrics['conversion_efectivo'] = cashflow_data['cash_conversion']
            
        logger.info(f"  ✓ CashFlows: {len(metrics)} métricas extraídas")
        return metrics
    
    def _extract_equity_metrics(self, equity_data: Dict) -> Dict:
        """Extrae métricas del Equity Agent"""
        metrics = {}
        
        # Ratio de pago de dividendos
        if 'dividend_payout_ratio' in equity_data:
            metrics['payout_ratio'] = equity_data['dividend_payout_ratio']
        elif 'payout_ratio' in equity_data:
            metrics['payout_ratio'] = equity_data['payout_ratio']
            
        # Book value per share
        if 'book_value_per_share' in equity_data:
            metrics['valor_libro_accion'] = equity_data['book_value_per_share']
        elif 'book_value' in equity_data:
            metrics['valor_libro_accion'] = equity_data['book_value']
            
        logger.info(f"  ✓ Equity: {len(metrics)} métricas extraídas")
        return metrics

# Test rápido
def test_evolutionary_predictor():
    logger.info(" Iniciando test del EvolutionaryPredictorAgent...")
    dates = pd.date_range('2020-01-01', periods=16, freq='Q')
    np.random.seed(42)
    financial_data = pd.DataFrame({
        'ROA': 1.2 + 0.1 * np.sin(np.arange(16)) + np.random.normal(0, 0.05, 16),
        'ratio_solvencia': 12.5 + 0.5 * np.sin(np.arange(16)) + np.random.normal(0, 0.2, 16)
    }, index=dates)
    agent = EvolutionaryPredictorAgent()
    results = agent.predict_financial_metrics(financial_data)
    logger.info(" Test completado")
    return results

if __name__ == "__main__":
    test_evolutionary_predictor()
