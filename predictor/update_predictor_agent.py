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

    def predict_financial_metrics(self, financial_data, bank_symbol="GARAN.IS", metrics=None):
        if metrics is None:
            metrics = ['ROA', 'ratio_solvencia', 'liquidez', 'beneficio_neto']

        external_data = self.fetch_external_data(bank_symbol)
        results = {}
        for metric in metrics:
            if metric not in financial_data.columns:
                continue
            series = financial_data[metric].dropna()
            if len(series) < 8:
                continue
            df = self.prepare_data_for_prophet(series, external_data, metric)
            prop = self.prophet_prediction(df, metric)
            xgb = self.xgboost_prediction(df, metric)
            ens = self.ensemble_prediction(prop, xgb)
            results[metric] = {'prophet': prop, 'xgboost': xgb, 'ensemble': ens}

        if results:
            self.export_predictions_to_csv(results)

        return results

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
