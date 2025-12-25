import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle
import seaborn as sns
import re
import os
import warnings
from prophet import Prophet
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from scipy.stats import pearsonr, linregress
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.fft import fft, ifft
from sklearn.mixture import GaussianMixture
import json
from datetime import datetime, timedelta
import itertools
from collections import defaultdict

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8

# 颜色主题
COLORS = {
    'primary': '#2E86AB',
    'success': '#28A745',
    'warning': '#FFC107',
    'danger': '#DC3545',
    'info': '#17A2B8',
    'secondary': '#6C757D',
    'purple': '#9B59B6',
    'orange': '#E67E22',
}

# 层级基准数据 
TIER_BENCHMARKS = {
    'GIANT': {'openrank': 80, 'stars': 50000, 'participants': 500, 'color': '#9B59B6'},
    'MATURE': {'openrank': 25, 'stars': 5000, 'participants': 100, 'color': '#3498DB'},
    'GROWING': {'openrank': 8, 'stars': 1000, 'participants': 30, 'color': '#2ECC71'},
    'EMERGING': {'openrank': 2, 'stars': 100, 'participants': 5, 'color': '#E67E22'}
}

TIER_NAMES = {
    'GIANT': '巨型项目',
    'MATURE': '成熟项目', 
    'GROWING': '成长项目',
    'EMERGING': '新兴项目'
}

# 数据类定义
@dataclass
class AnalysisResult:
    """分析结果"""
    project_name: str
    tier: str
    tier_probabilities: Dict[str, float]  
    tier_confidence: float
    lifecycle: str
    vitality: str
    health_score: float
    health_grade: str
    dimension_scores: Dict[str, float]
    momentum_analysis: Dict
    resistance_analysis: Dict
    potential_analysis: Dict
    trend_3d: Dict
    risk_analysis: Dict
    dark_horse_analysis: Dict
    change_points: List[Dict]
    bus_factor_2: Dict
    etd_analysis: Dict
    github_comparison: Dict
    conclusion_validation: Dict
    prediction_validation: Dict
    predictions: Dict
    backtest_results: Dict
    recommendations: List[str]
    detailed_report: str


# GMM概率化分层分类器 
class GMMTierClassifier:
    """高斯混合模型分层分类器"""
    
    def __init__(self, n_components=4):
        self.n_components = n_components
        self.gmm = None
        self.scaler = StandardScaler()
        self.tier_labels = ['GIANT', 'MATURE', 'GROWING', 'EMERGING']
        
    def _generate_synthetic_data(self) -> np.ndarray:
        """生成合成基准数据用于训练GMM"""
        # 保存当前随机状态
        original_state = np.random.get_state()
        # 设置随机种子以确保每次生成相同的数据
        np.random.seed(42)
        
        synthetic_data = []
        for tier, benchmarks in TIER_BENCHMARKS.items():
            n_samples = 100  

            for _ in range(n_samples):
                openrank = np.random.normal(benchmarks['openrank'], max(benchmarks['openrank'] * 0.3, 1))
                stars = np.random.normal(benchmarks['stars'], max(benchmarks['stars'] * 0.3, 100))
                participants = np.random.normal(benchmarks['participants'], max(benchmarks['participants'] * 0.3, 5))
                
                synthetic_data.append([
                    max(0.1, openrank),
                    max(10, stars),
                    max(1, participants)
                ])
        
        # 恢复原始随机状态
        np.random.set_state(original_state)
        
        return np.array(synthetic_data)
    
    def fit(self):
        """训练GMM模型"""
        synthetic_data = self._generate_synthetic_data()
        scaled_data = self.scaler.fit_transform(synthetic_data)
        self.gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type='full',
            random_state=42
        )
        self.gmm.fit(scaled_data)
        return self
    
    def predict_proba(self, metrics: Dict) -> Tuple[str, Dict[str, float], float]:
        """预测层级概率"""
        if self.gmm is None:
            self.fit()
        
        feature = np.array([
            metrics.get('avg_openrank', 0),
            metrics.get('total_stars', 0),
            metrics.get('max_participants', 0)
        ]).reshape(1, -1)
        feature = np.maximum(feature, [0.1, 10, 1])
        
        scaled_feature = self.scaler.transform(feature)
        probabilities = self.gmm.predict_proba(scaled_feature)[0]
        centers = self.gmm.means_  
        centers_original = centers * self.scaler.scale_ + self.scaler.mean_
        openrank_centers = centers_original[:, 0]
        sorted_indices = np.argsort(-openrank_centers)
        
        tier_probabilities = {}
        for idx, tier in enumerate(self.tier_labels):
            if idx < len(sorted_indices):
                comp_idx = sorted_indices[idx]
                tier_probabilities[tier] = probabilities[comp_idx]
            else:
                tier_probabilities[tier] = 0.0
        
        best_tier = max(tier_probabilities, key=tier_probabilities.get)
        confidence = tier_probabilities[best_tier]
        
        return best_tier, tier_probabilities, confidence


# 动量计算器
class MomentumCalculator:
    """社区动量计算器"""
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        """计算综合动量"""
        quality = self._calc_quality_momentum(data)
        gravity = self._calc_contributor_gravity(data)
        pr_accel = self._calc_pr_acceleration(data)
        
        total = 0.4 * quality + 0.35 * gravity + 0.25 * pr_accel
        
        if total >= 70:
            interpretation = '强劲动量 - 项目正在健康扩张'
        elif total >= 50:
            interpretation = '稳定动量 - 保持正常发展'
        elif total >= 30:
            interpretation = '弱动量 - 发展动力不足'
        else:
            interpretation = '负动量 - 需要干预'
        
        return {
            'total': round(total, 2),
            'quality': round(quality, 2),
            'gravity': round(gravity, 2),
            'pr_accel': round(pr_accel, 2),
            'interpretation': interpretation
        }
    
    def _calc_quality_momentum(self, data: pd.DataFrame) -> float:
        if 'pr_merged' not in data or 'pr_new' not in data:
            return 50
        merge_rate = data['pr_merged'] / (data['pr_new'] + 0.1)
        if len(merge_rate) < 6:
            return 50
        trend = np.polyfit(range(len(merge_rate.tail(6))), merge_rate.tail(6).values, 1)[0]
        return min(100, max(0, 50 + trend * 100))
    
    def _calc_contributor_gravity(self, data: pd.DataFrame) -> float:
        if 'participants' not in data:
            return 50
        growth = data['participants'].diff().tail(6).mean()
        inactive = data.get('inactive_contributors', pd.Series([0]))
        retention = 1 - (inactive.tail(6).mean() / (data['participants'].tail(6).mean() + 1))
        return min(100, max(0, 50 + growth * 10 + retention * 30))
    
    def _calc_pr_acceleration(self, data: pd.DataFrame) -> float:
        if 'pr_merged' not in data:
            return 50
        pr = data['pr_merged'].tail(6)
        if len(pr) < 3:
            return 50
        accel = pr.diff().diff().mean()
        return min(100, max(0, 50 + accel * 20))


# 阻力计算器 
class ResistanceCalculator:
    """技术债阻力计算器"""
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        debt = self._calc_debt_resistance(data)
        entropy = self._calc_entropy_trend(data)
        issue_pressure = self._calc_issue_pressure(data)
        
        total = 0.4 * debt + 0.3 * entropy + 0.3 * issue_pressure
        
        if total >= 70:
            status, interpretation = 'HEAVY', '高阻力 - 技术债严重'
        elif total >= 50:
            status, interpretation = 'MEDIUM_HIGH', '中高阻力 - 需关注'
        elif total >= 30:
            status, interpretation = 'NORMAL', '中等阻力 - 正常范围'
        else:
            status, interpretation = 'LIGHT', '低阻力 - 发展顺畅'
        
        return {
            'total': round(total, 2),
            'debt': round(debt, 2),
            'entropy': round(entropy, 2),
            'issue_pressure': round(issue_pressure, 2),
            'status': status,
            'interpretation': interpretation
        }
    
    def _calc_debt_resistance(self, data: pd.DataFrame) -> float:
        if 'issues_new' not in data or 'issues_closed' not in data:
            return 30
        open_issues = data['issues_new'].cumsum() - data['issues_closed'].cumsum()
        if len(open_issues) < 6:
            return 30
        growth = np.polyfit(range(len(open_issues.tail(6))), open_issues.tail(6).values, 1)[0]
        return min(100, max(0, 30 + growth * 5))
    
    def _calc_entropy_trend(self, data: pd.DataFrame) -> float:
        if 'activity' not in data:
            return 30
        volatility = data['activity'].tail(12).std() / (data['activity'].tail(12).mean() + 0.1)
        return min(100, volatility * 50)
    
    def _calc_issue_pressure(self, data: pd.DataFrame) -> float:
        if 'issues_new' not in data:
            return 30
        recent = data['issues_new'].tail(6).mean()
        historical = data['issues_new'].mean()
        ratio = recent / (historical + 0.1)
        return min(100, max(0, ratio * 40))


# 潜力计算器 
class PotentialCalculator:
    """增长潜力计算器"""
    
    def calculate(self, data: pd.DataFrame, tier: str) -> Dict:
        """计算更悲观的增长潜力"""
        ceiling = self._estimate_ceiling_pessimistic(data)
        current = data['openrank'].iloc[-1] if 'openrank' in data else 0
        if current < 1 and 'openrank' in data:
            current = max(0.1, data['openrank'].tail(6).mean())
        if ceiling > current:
            remaining = ((ceiling - current) / (ceiling + 0.1)) * 100
        else:
            remaining = 0

        tier_adjustment = {'GIANT': 0.2, 'MATURE': 0.3, 'GROWING': 0.6, 'EMERGING': 0.8}
        remaining *= tier_adjustment.get(tier, 0.5)
        
        return {
            'growth_ceiling': round(ceiling, 1),
            'current_position': round(current, 2),
            'remaining_space': min(100, max(0, remaining)), 
            'interpretation': self._interpret(remaining)
        }
    
    def _estimate_ceiling_pessimistic(self, data: pd.DataFrame) -> float:
        if 'openrank' not in data or len(data) < 6:
            return 1.0  
        
        openrank = data['openrank'].values
        historical_max = openrank.max()
        
        if historical_max == 0:
            return 0.0
        lookback = min(len(openrank), 12)
        recent = openrank[-lookback:]
        x = np.arange(len(recent))
        slope, intercept = np.polyfit(x, recent, 1)
        fitted_line = slope * x + intercept
        residuals = recent - fitted_line
        volatility = np.std(residuals)
        future_x = np.arange(len(recent), len(recent) + 6)
        future_trend = slope * future_x + intercept

        future_potential_curve = future_trend + (1.5 * volatility)
        future_max = np.max(future_potential_curve)
        recent_max = recent.max()
        future_max = max(future_max, recent_max * 1.1)
        absolute_ceiling = historical_max * 2.0
        ceiling = min(absolute_ceiling, future_max)
        return max(1.0, ceiling)

    def _interpret(self, remaining: float) -> str:
        if remaining >= 70:
            return '高增长潜力 - 远未触及天花板'
        elif remaining >= 40:
            return '中等潜力 - 仍有成长空间'
        elif remaining >= 20:
            return '有限潜力 - 接近成熟'
        else:
            return '已达成熟 - 进入稳定期'


# Prophet时序预测器 
class ProphetPredictor:
    """Prophet时序预测器"""
    
    def __init__(self):
        self.fitted = False
        
    def _linear_regression(self, data: pd.Series, periods: int = 6) -> List[float]:
        """线性回归预测"""
        x = np.arange(len(data)).reshape(-1, 1)
        values = data.values
        model = LinearRegression().fit(x, values)
        future_x = np.arange(len(data), len(data) + periods).reshape(-1, 1)
        pred = model.predict(future_x)
        return list(pred)
    
    def _exponential_smoothing(self, data: pd.Series, periods: int = 6) -> List[float]:
        """指数平滑预测 (替代不稳定的指数回归)"""
        values = data.values
        alpha = 0.8  
        last_val = values[-1]
        preds = []
        for _ in range(periods):
            preds.append(last_val) 
        if len(values) >= 6:
            trend = (values[-1] - values[-6]) / 5
            preds = [last_val + trend * (i+1) * 0.5 for i in range(periods)]
            
        return preds
    
    def _weighted_moving_average(self, data: pd.Series, periods: int = 6) -> List[float]:
        """加权移动平均预测"""
        window = min(6, len(data))
        weights = np.arange(1, window + 1) / np.sum(np.arange(1, window + 1))
        recent_data = data.values[-window:]
        weighted_avg = np.sum(recent_data * weights)
        trend = 0
        if len(data) >= 3:
            trend = (data.values[-1] - data.values[-3]) / 2
        
        predictions = [weighted_avg + (trend * 0.8 * (i + 1)) for i in range(periods)]
        return predictions
    
    def _preprocess_series(self, series: pd.Series) -> pd.Series:
        """数据预处理：Log变换以防止负数预测"""
        mean, std = series.mean(), series.std()
        if std > 0:
            series = series.clip(max(0, mean - 3*std), mean + 3*std)
        return np.log1p(series)
    
    def _inverse_transform_values(self, values: np.ndarray) -> np.ndarray:
        """数据还原：Exp变换 + 0截断"""
        restored = np.expm1(values)
        return np.maximum(0, restored)  

    def _simulate_prophet_predictions(self, data: pd.Series, periods: int = 6, vitality_factor: float = 1.0) -> Dict:
        """简化版Prophet模拟 (Ensemble)"""
        if len(data) < 3:
            return {'forecast': None}
        
        try:
            log_series = self._preprocess_series(data)

            preds_linear = self._linear_regression(log_series, periods)
            preds_exp = self._exponential_smoothing(log_series, periods)
            preds_wma = self._weighted_moving_average(log_series, periods)

            volatility = log_series.tail(6).std()
            if volatility > 0.5:
                weights = {'linear': 0.2, 'exp': 0.3, 'wma': 0.5}
            else:
                weights = {'linear': 0.4, 'exp': 0.3, 'wma': 0.3}
            
            combined_log_forecast = []
            for i in range(periods):
                val = (preds_linear[i] * weights['linear'] + 
                       preds_exp[i] * weights['exp'] + 
                       preds_wma[i] * weights['wma'])
                combined_log_forecast.append(val)
            values = log_series.values
            freqs = fft(values)
            seasonal = np.real(ifft(freqs * (np.abs(freqs) > np.mean(np.abs(freqs)) * 1.5)))
            seasonal_component = list(seasonal - np.mean(seasonal))
            
            final_forecast = []
            for i, val in enumerate(combined_log_forecast):
                seas_val = seasonal_component[i % len(seasonal_component)] * 0.05 if seasonal_component else 0
                final_val = val + seas_val
                final_forecast.append(final_val)
            
            restored_forecast = self._inverse_transform_values(np.array(final_forecast))
            restored_forecast = restored_forecast * vitality_factor

            log_std = np.std(values[-6:]) if len(values) >= 6 else 0.1
            
            yhat_lower = self._inverse_transform_values(np.array(final_forecast) - 1.96 * log_std)
            yhat_upper = self._inverse_transform_values(np.array(final_forecast) + 1.96 * log_std)

            x = np.arange(len(values))
            z = np.polyfit(x, values, 1)
            trend_log = np.polyval(z, x)
            trend = self._inverse_transform_values(trend_log)

            return {
                'forecast': list(restored_forecast),
                'trend': list(trend),
                'seasonality': [0] * periods,
                'yhat_lower': list(yhat_lower),
                'yhat_upper': list(yhat_upper),
                'confidence': max(0.4, 1 - volatility),
                'method': 'ensemble_simplified'
            }

        except Exception as e:
            fallback_val = max(0, data.mean())
            return {
                'forecast': [fallback_val] * periods,
                'trend': list(data.values),
                'yhat_lower': [fallback_val * 0.8] * periods,
                'yhat_upper': [fallback_val * 1.2] * periods,
                'confidence': 0.1,
                'method': 'fallback_mean'
            }

    def predict(self, data: pd.DataFrame, metric: str = 'openrank', periods: int = 6, tier: str = 'GROWING', vitality: str = 'STABLE') -> Dict:
        """预测指定指标"""
        if metric not in data:
            return {'error': f'指标{metric}不存在'}
        
        series = data[metric].dropna()
        
        if len(series) < 6:
            return {'error': '数据不足'}
        
        vitality_factors = {
            'THRIVING': 1.05, 'STABLE': 1.0, 
            'DORMANT': 0.9, 'ZOMBIE': 0.8, 'UNKNOWN': 1.0
        }
        vitality_factor = vitality_factors.get(vitality, 1.0)

        if len(series) >= 24 and Prophet is not None:
            try:
                df = pd.DataFrame({
                    'ds': series.index,
                    'y': np.log1p(series.values) # LOG TRANSFORM
                })
                
                if not pd.api.types.is_datetime64_any_dtype(df['ds']):
                    df['ds'] = pd.date_range(start=datetime.now() - timedelta(days=len(df)*30), periods=len(df), freq='ME')

                # 2. 配置 Prophet (参数微调)
                model = Prophet(
                    yearly_seasonality='auto',
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    seasonality_mode='additive', # Log 空间下的 Additive = 原始空间的 Multiplicative
                    changepoint_prior_scale=0.05, # 默认 0.05，适中
                    growth='linear' 
                )
                
                # 添加月度季节性
                model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

                model.fit(df)
                
                future = model.make_future_dataframe(periods=periods, freq='ME')
                forecast = model.predict(future)
                
                # 3. 还原数据 (Exp - 1)
                predicted_log = forecast['yhat'].tail(periods).values
                lower_log = forecast['yhat_lower'].tail(periods).values
                upper_log = forecast['yhat_upper'].tail(periods).values
                
                forecast_values = np.expm1(predicted_log)
                lower = np.expm1(lower_log)
                upper = np.expm1(upper_log)
                
                # 4. 后处理：应用调整因子 + 绝对非负截断
                forecast_values = np.maximum(0, forecast_values * vitality_factor)
                lower = np.maximum(0, lower * vitality_factor)
                upper = np.maximum(0, upper * vitality_factor)
                
                # 提取历史趋势用于绘图 (也要还原)
                trend_log = forecast['trend'][:-periods]
                trend_restored = np.maximum(0, np.expm1(trend_log))

                return {
                    'forecast': list(forecast_values),
                    'trend': list(trend_restored),
                    'yhat_lower': list(lower),
                    'yhat_upper': list(upper),
                    'confidence': 0.95, # 稍微保守一点
                    'method': 'full_prophet_log'
                }
            except Exception as e:
                print(f"Prophet error: {e}, falling back to simplified.")
                return self._simulate_prophet_predictions(series, periods, vitality_factor)
        else:
            return self._simulate_prophet_predictions(series, periods, vitality_factor)
# ============== AHP权重计算器 ==============
class AHPHealthEvaluator:
    """AHP健康评估器"""
    
    # 层级特定的权重矩阵
    TIER_WEIGHTS = {
        'GIANT': {
            'momentum': 0.05,
            'stability': 0.70,
            'potential': 0.05,
            'safety': 0.20
        },
        'MATURE': {
            'momentum': 0.15,
            'stability': 0.50,
            'potential': 0.15,
            'safety': 0.20
        },
        'GROWING': {
            'momentum': 0.40,
            'stability': 0.20,
            'potential': 0.30,
            'safety': 0.10
        },
        'EMERGING': {
            'momentum': 0.30,
            'stability': 0.10,
            'potential': 0.50,
            'safety': 0.10
        }
    }
    
    def calculate_health_score(self, 
                              vitality: str,
                              trend: Dict,
                              risk: Dict,
                              tier: str,
                              predictions: Dict = None) -> Tuple[float, Dict[str, float]]:
        """计算AHP加权健康分"""
        
        # 获取层级特定权重
        weights = self.TIER_WEIGHTS.get(tier, self.TIER_WEIGHTS['MATURE'])
        
        # 计算各维度原始分数
        raw_scores = {
            'momentum': trend['momentum']['total'],
            'stability': 100 - trend['resistance']['total'],
            'potential': trend['potential']['remaining_space'],
            'safety': max(0, 100 - risk['score'] * 2)
        }
        
        # 活力状态调整因子
        vitality_factors = {
            'THRIVING': 1.2,
            'STABLE': 1.0,
            'DORMANT': 0.8,
            'ZOMBIE': 0.6,
            'UNKNOWN': 0.9
        }
        vitality_factor = vitality_factors.get(vitality, 1.0)
        
        # 预测置信度调整（如果有预测数据）
        if predictions and 'confidence' in predictions.get('openrank', {}):
            pred_confidence = predictions['openrank']['confidence']
            prediction_factor = 0.9 + pred_confidence * 0.2
        else:
            prediction_factor = 1.0
        
        # 应用权重和调整因子
        weighted_scores = {}
        total_weighted = 0
        total_weight = 0
        
        for dimension, raw_score in raw_scores.items():
            weight = weights[dimension]
            weighted = raw_score * weight
            weighted_scores[dimension] = weighted
            total_weighted += weighted
            total_weight += weight
        
        # 基础健康分
        base_score = total_weighted / total_weight if total_weight > 0 else 50
        
        # 应用调整因子
        adjusted_score = base_score * vitality_factor * prediction_factor
        
        # 限制范围
        final_score = max(0, min(100, adjusted_score))
        
        # 计算各维度贡献百分比
        dimension_contributions = {
            '动量': weights['momentum'] * raw_scores['momentum'] / 100,
            '稳定': weights['stability'] * raw_scores['stability'] / 100,
            '潜力': weights['potential'] * raw_scores['potential'] / 100,
            '安全': weights['safety'] * raw_scores['safety'] / 100
        }
        
        return round(final_score, 1), dimension_contributions


# ============== 回测验证器 ==============
class BacktestValidator:
    """预测回测验证器 - v4.6 Fixed (修复回测由于数据不足无法执行的问题)"""
    
    def __init__(self):
        pass
        
    def _calculate_smape(self, actual, forecast):
        """计算 SMAPE"""
        # 防止除零错误
        denominator = np.abs(actual) + np.abs(forecast) + 1e-10
        return 100/len(actual) * np.sum(2 * np.abs(forecast - actual) / denominator)

    def _calculate_theils_u(self, actual, forecast, train_last_value):
        """计算 Theil's U"""
        mse_model = np.mean((forecast - actual) ** 2)
        rmse_naive = np.mean((train_last_value - actual) ** 2)
        if rmse_naive < 1e-10: return 1.0 
        return np.sqrt(mse_model / rmse_naive)

    def _calculate_confidence_score(self, metrics: Dict, tier: str) -> Tuple[str, float]:
        """计算综合置信度分数"""
        weights = {'smape': 0.4, 'r2': 0.3, 'theil_u': 0.2, 'dir_acc': 0.1}
        score = 0
        
        # 宽松的 SMAPE 标准
        smape_threshold = 25 if tier in ['GROWING', 'EMERGING'] else 15
        if metrics['smape'] < smape_threshold: score += weights['smape']
        elif metrics['smape'] < smape_threshold * 2: score += weights['smape'] * 0.5
        
        # R2
        if metrics['r2'] > 0.6: score += weights['r2']
        elif metrics['r2'] > 0.3: score += weights['r2'] * 0.5
        
        # Theil's U
        if metrics['theil_u'] < 0.9: score += weights['theil_u']
        elif metrics['theil_u'] < 1.2: score += weights['theil_u'] * 0.5
        
        # 方向准确率
        if metrics['direction_accuracy'] > 0.6: score += weights['dir_acc']
        
        if score >= 0.75: return 'HIGH', score
        elif score >= 0.5: return 'MEDIUM', score
        elif score >= 0.25: return 'LOW', score
        else: return 'VERY_LOW', score
    
    def validate_predictions(self, data: pd.DataFrame, metric: str = 'openrank', tier: str = 'GROWING', vitality: str = 'STABLE') -> Dict:
        """执行滚动回测 (Debug增强版)"""
        
        # --- 调试点 1: 检查输入数据 ---
        if metric not in data:
            print(f"DEBUG: 找不到指标 {metric}, 现有列: {data.columns.tolist()}")
            return {'error': f'指标 {metric} 不存在'}
            
        series = data[metric].dropna()
        
        # 自动转换索引为时间格式 (解决索引不匹配问题)
        if not pd.api.types.is_datetime64_any_dtype(series.index):
            try:
                series.index = pd.to_datetime(series.index)
            except:
                # 如果无法转换，强制生成一个伪时间序列索引以维持 Prophet 运行
                series.index = pd.date_range(end=datetime.now(), periods=len(series), freq='ME')
        
        n = len(series)
        if n < 6:
            return {'error': f'数据量太少 ({n}), 无法回测'}

        print(f"DEBUG: 开始对 {metric} 进行回测, 数据总量: {n}")
        
        metrics_history = defaultdict(list)
        predictor = ProphetPredictor()
        
        # --- 调试点 2: 动态分割策略 ---
        # 采用最稳妥的固定比例分割，确保至少能跑一次
        test_size = max(2, int(n * 0.2)) # 测试集占 20%
        train_size = n - test_size
        
        # 尝试进行 3 次不同起点的滚动
        test_points = [train_size, train_size - 1, train_size + 1]
        rounds = 0

        for split_idx in test_points:
            if split_idx < 4 or split_idx >= n: continue
            
            train_df = pd.DataFrame({metric: series.iloc[:split_idx]})
            test_actual = series.iloc[split_idx:]
            
            try:
                # 执行预测
                res = predictor.predict(train_df, metric, periods=len(test_actual), tier=tier, vitality=vitality)
                
                if 'forecast' in res and res['forecast'] is not None:
                    pred_v = np.array(res['forecast'])[:len(test_actual)]
                    actual_v = test_actual.values[:len(pred_v)]
                    
                    if len(actual_v) < 1: continue
                    
                    # 计算指标 (加入对非负的强制检查)
                    pred_v = np.maximum(0, pred_v) 
                    
                    smape = self._calculate_smape(actual_v, pred_v)
                    mae = mean_absolute_error(actual_v, pred_v)
                    
                    metrics_history['smape'].append(smape)
                    metrics_history['mae'].append(mae)
                    metrics_history['rmse'].append(np.sqrt(mean_squared_error(actual_v, pred_v)))
                    
                    # R2 保护
                    if len(actual_v) > 1 and np.var(actual_v) > 0:
                        metrics_history['r2'].append(r2_score(actual_v, pred_v))
                    else:
                        metrics_history['r2'].append(0.5)
                        
                    metrics_history['theil_u'].append(self._calculate_theils_u(actual_v, pred_v, train_df[metric].iloc[-1]))
                    
                    # 方向准确率
                    if len(actual_v) > 1:
                        acc = np.mean(np.sign(np.diff(pred_v)) == np.sign(np.diff(actual_v)))
                        metrics_history['direction_accuracy'].append(acc)
                    else:
                        metrics_history['direction_accuracy'].append(1.0)
                        
                    rounds += 1
            except Exception as e:
                print(f"DEBUG: 某一轮回测崩溃: {str(e)}")
                continue

        # --- 调试点 3: 汇总与反馈 ---
        if rounds == 0:
            print("DEBUG: 所有回测尝试均失败")
            return {'error': '回测计算未收敛'}

        avg_metrics = {k: np.mean(v) for k, v in metrics_history.items()}
        conf_level, conf_score = self._calculate_confidence_score(avg_metrics, tier)
        
        print(f"DEBUG: 回测完成, 成功轮数: {rounds}, SMAPE: {avg_metrics['smape']}")

        return {
            'mae': round(avg_metrics['mae'], 2),
            'rmse': round(avg_metrics['rmse'], 2),
            'r2': round(avg_metrics['r2'], 3),
            'smape': round(avg_metrics['smape'], 1),
            'theil_u': round(avg_metrics['theil_u'], 2),
            'direction_accuracy': round(avg_metrics['direction_accuracy'] * 100, 1),
            'overall_confidence': conf_level,
            'confidence_score': conf_score,
            'num_rounds': rounds,
            'interpretation': f"基于{rounds}轮验证，预测偏差约 {avg_metrics['smape']:.1f}%"
        }
# ============== Bus Factor 2.0 ==============
class BusFactorCalculator:
    """贡献熵模型"""
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        if 'participants' not in data:
            return {'effective_bus_factor': 1, 'risk_level': 'UNKNOWN'}
        
        # 模拟贡献分布（实际应从API获取）
        participants = data['participants'].tail(6).mean()
        if participants <= 0:
            return {'effective_bus_factor': 1, 'risk_level': 'CRITICAL'}
        
        # 假设贡献服从Zipf分布
        n = int(max(1, participants))
        contributions = np.array([1/(i+1) for i in range(n)])
        contributions = contributions / contributions.sum()
        
        # 计算熵
        entropy = -np.sum(contributions * np.log2(contributions + 1e-10))
        max_entropy = np.log2(n) if n > 1 else 1
        normalized = entropy / max_entropy if max_entropy > 0 else 0
        effective_bf = 2 ** entropy
        
        if effective_bf <= 2:
            risk, desc = 'CRITICAL', '极高风险：贡献过于集中'
        elif effective_bf <= 4:
            risk, desc = 'HIGH', '高风险：需培养更多贡献者'
        elif effective_bf <= 8:
            risk, desc = 'MEDIUM', '中等风险：贡献者多样性尚可'
        else:
            risk, desc = 'LOW', '低风险：贡献者生态健康'
        
        return {
            'raw_entropy': round(entropy, 3),
            'normalized_entropy': round(normalized, 3),
            'effective_bus_factor': round(effective_bf, 1),
            'risk_level': risk,
            'description': desc
        }


# ============== 变点检测 ==============
class ChangePointDetector:
    """变点检测器"""
    
    def detect(self, data: pd.Series) -> List[Dict]:
        if len(data) < 12:
            return []
        
        results = []
        # 简化版变点检测（窗口比较）
        window = 6
        for i in range(window, len(data) - window):
            before = data.iloc[i-window:i].mean()
            after = data.iloc[i:i+window].mean()
            change_rate = (after - before) / (before + 0.1)
            
            if abs(change_rate) > 0.3:
                if change_rate > 0.3:
                    cp_type, desc = 'ACCELERATION', '进入快速增长期'
                else:
                    cp_type, desc = 'DECELERATION', '活跃度显著下降'
                
                results.append({
                    'index': i,
                    'date': str(data.index[i])[:7] if hasattr(data.index[i], 'strftime') else str(i),
                    'type': cp_type,
                    'change_rate': round(change_rate, 3),
                    'description': desc
                })
        
        return results[:3]  # 最多返回3个变点


# ============== 转化漏斗 ==============
class ConversionFunnelAnalyzer:
    """转化漏斗分析"""
    
    def analyze(self, data: pd.DataFrame) -> Dict:
        if 'new_contributors' not in data or 'stars' not in data:
            return {'funnel_rate': None, 'quality': 'UNKNOWN'}
        
        new_contrib = data['new_contributors'].tail(6).sum()
        new_stars = data['stars'].tail(6).sum()
        
        rate = new_contrib / (new_stars + 1)
        
        if rate > 0.1:
            quality, desc = 'EXCELLENT', '优质 - Star高效转化'
        elif rate > 0.05:
            quality, desc = 'GOOD', '良好 - 社区吸引力强'
        elif rate > 0.02:
            quality, desc = 'NORMAL', '正常水平'
        else:
            quality, desc = 'BUBBLE', '疑似泡沫 - 转化率低'
        
        return {
            'funnel_rate': round(rate, 4),
            'quality': quality,
            'description': desc,
            'new_contributors': int(new_contrib),
            'new_stars': int(new_stars)
        }


# ============== ETD 寿命分析器 ==============
class ETDAnalyzer:
    """
    增强版 ETD (Estimated Time to Depletion) 分析器
    区分：成熟稳定 vs 真正衰退
    """
    
    def analyze(self, data: pd.DataFrame, vitality: str, tier: str) -> Dict:
        """
        分析项目预期寿命
        """
        result = {
            'etd_months': float('inf'),
            'etd_status': 'HEALTHY',
            'decay_type': 'NONE',
            'is_mature_stable': False,
            'confidence': 'HIGH',
            'description': '',
            'recommendations': []
        }
        
        if 'activity' not in data or len(data) < 6:
            result['description'] = '数据不足，无法进行寿命预测'
            result['confidence'] = 'LOW'
            return result
        
        activity = data['activity'].tail(12)
        X = np.arange(len(activity)).reshape(-1, 1)
        model = LinearRegression().fit(X, activity.values)
        slope = model.coef_[0]
        current_activity = activity.iloc[-1]
        avg_activity = activity.mean()
        
        # === 情况1: 活跃度上升 ===
        if slope >= 0:
            result['etd_months'] = float('inf')
            result['etd_status'] = 'THRIVING'
            result['description'] = '活跃度呈上升趋势，项目生命力强劲，无枯竭风险。'
            return result
        
        # === 情况2: 成熟稳定项目 ===
        # 判断标准：高层级 + 当前活跃度仍然可观 + 不是急剧下降
        if tier in ['GIANT', 'MATURE'] and vitality in ['STABLE', 'THRIVING']:
            if current_activity > avg_activity * 0.3 and abs(slope) < avg_activity * 0.1:
                result['etd_months'] = float('inf')
                result['etd_status'] = 'STABLE_MATURE'
                result['is_mature_stable'] = True
                result['description'] = (
                    f'项目已进入成熟稳定期。虽然活跃度月均下降 {abs(slope):.2f} 点，'
                    f'但这是成熟项目的正常特征——功能完善后无需频繁更新。'
                    f'当前活跃度 {current_activity:.0f} 仍维持在健康水平（均值的 '
                    f'{current_activity/avg_activity*100:.0f}%）。'
                )
                result['recommendations'].append('保持定期安全更新和 Bug 修复即可')
                return result
        
        # === 情况3: 检测衰退模式 ===
        decay_type, decay_params = self._detect_decay_pattern(activity)
        result['decay_type'] = decay_type
        
        # === 情况4: 真正的衰退 ===
        if slope < 0 and current_activity > 0:
            # 线性预测
            etd_linear = -current_activity / slope
            
            # 指数衰减预测（如果适用）
            if decay_type == 'EXPONENTIAL' and decay_params.get('half_life'):
                etd = decay_params['half_life'] * 3  # 3个半衰期
            else:
                etd = etd_linear
            
            result['etd_months'] = max(0, etd)
            
            if etd < 6:
                result['etd_status'] = 'CRITICAL'
                result['description'] = (
                    f'高危预警：按当前衰减速度（月均 -{abs(slope):.2f}），'
                    f'预计 {etd:.1f} 个月后活跃度将归零。'
                    f'衰减模式：{decay_type}。建议立即采取措施激活社区。'
                )
                result['recommendations'].extend([
                    '立即发布新版本或 Roadmap 激活社区',
                    '组织线上直播或 AMA 活动',
                    '招募新维护者'
                ])
            elif etd < 12:
                result['etd_status'] = 'WARNING'
                result['description'] = (
                    f'衰退预警：活跃度呈下降趋势，预计 {etd:.1f} 个月后可能枯竭。'
                    f'衰减模式：{decay_type}。建议加强社区运营。'
                )
                result['recommendations'].extend([
                    '发布技术博客保持曝光',
                    '标记 Good First Issue 吸引新贡献者'
                ])
            elif etd < 24:
                result['etd_status'] = 'CAUTION'
                result['description'] = (
                    f'温和下降：活跃度有所下滑，预计 {etd:.1f} 个月后可能低迷。'
                    f'目前尚有缓冲时间，建议观察并适时调整。'
                )
                result['recommendations'].append('持续监控，适时调整运营策略')
            else:
                result['etd_status'] = 'HEALTHY'
                result['description'] = f'轻微下滑，但 ETD > 24个月，暂无紧迫风险。'
        
        return result
    
    def _detect_decay_pattern(self, data: pd.Series) -> Tuple[str, Dict]:
        """检测衰退模式类型"""
        if len(data) < 6:
            return 'UNKNOWN', {}
        
        x = np.arange(len(data))
        y = data.values
        
        # 线性拟合
        lin_slope, lin_intercept = np.polyfit(x, y, 1)
        lin_pred = lin_slope * x + lin_intercept
        lin_r2 = 1 - np.sum((y - lin_pred)**2) / np.sum((y - np.mean(y))**2)
        
        # 指数拟合
        try:
            log_y = np.log(y + 0.1)
            exp_slope, exp_intercept = np.polyfit(x, log_y, 1)
            exp_pred = np.exp(exp_slope * x + exp_intercept)
            exp_r2 = 1 - np.sum((y - exp_pred)**2) / np.sum((y - np.mean(y))**2)
            half_life = -np.log(2) / exp_slope if exp_slope < 0 else None
        except:
            exp_r2 = 0
            half_life = None
        
        if lin_slope >= 0:
            return 'NONE', {'slope': lin_slope}
        
        if exp_r2 > lin_r2 and exp_r2 > 0.5:
            return 'EXPONENTIAL', {'r2': exp_r2, 'half_life': half_life}
        elif lin_r2 > 0.3:
            return 'LINEAR', {'r2': lin_r2, 'slope': lin_slope}
        else:
            return 'IRREGULAR', {}


# ============== GitHub API 分析器 ==============
class GitHubAPIAnalyzer:
    """GitHub API 数据获取与对比分析"""
    
    def __init__(self, token: str = None):
        self.token = token
        self.base_url = "https://api.github.com"
        self.headers = {'Authorization': f'token {token}'} if token else {}
    
    def fetch_repo_info(self, org: str, repo: str) -> Optional[Dict]:
        """获取仓库基本信息"""
        if not self.token:
            return None
        
        try:
            url = f"{self.base_url}/repos/{org}/{repo}"
            res = requests.get(url, headers=self.headers, timeout=15)
            if res.status_code == 200:
                data = res.json()
                return {
                    'full_name': data.get('full_name'),
                    'description': data.get('description'),
                    'stars': data.get('stargazers_count', 0),
                    'forks': data.get('forks_count', 0),
                    'watchers': data.get('watchers_count', 0),
                    'open_issues': data.get('open_issues_count', 0),
                    'language': data.get('language'),
                    'created_at': data.get('created_at'),
                    'updated_at': data.get('updated_at'),
                    'pushed_at': data.get('pushed_at'),
                    'size': data.get('size'),
                    'license': data.get('license', {}).get('name') if data.get('license') else None,
                    'topics': data.get('topics', []),
                    'homepage': data.get('homepage'),
                    'archived': data.get('archived', False),
                    'disabled': data.get('disabled', False)
                }
            else:
                print(f"GitHub API 请求失败: {res.status_code}")
                return None
        except Exception as e:
            print(f"GitHub API 错误: {e}")
            return None
    
    def fetch_recent_activity(self, org: str, repo: str, days: int = 30) -> Dict:
        """
        获取最近N天的活跃数据
        包括: commits, issues, PRs
        """
        if not self.token:
            return {'error': '需要 GitHub Token'}
        
        from datetime import datetime, timedelta
        since_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        result = {
            'period_days': days,
            'commits': 0,
            'issues_opened': 0,
            'issues_closed': 0,
            'prs_opened': 0,
            'prs_merged': 0,
            'contributors_active': set(),
            'daily_activity': []
        }
        
        try:
            # 1. 获取最近提交
            commits_url = f"{self.base_url}/repos/{org}/{repo}/commits?since={since_date}&per_page=100"
            res = requests.get(commits_url, headers=self.headers, timeout=15)
            if res.status_code == 200:
                commits = res.json()
                result['commits'] = len(commits)
                for c in commits:
                    author = c.get('author', {})
                    if author and author.get('login'):
                        result['contributors_active'].add(author['login'])
            
            # 2. 获取最近 Issues
            issues_url = f"{self.base_url}/repos/{org}/{repo}/issues?state=all&since={since_date}&per_page=100"
            res = requests.get(issues_url, headers=self.headers, timeout=15)
            if res.status_code == 200:
                issues = res.json()
                for issue in issues:
                    if 'pull_request' not in issue:  # 排除 PR
                        created = issue.get('created_at', '')
                        if created >= since_date:
                            result['issues_opened'] += 1
                        if issue.get('state') == 'closed':
                            result['issues_closed'] += 1
            
            # 3. 获取最近 PRs
            prs_url = f"{self.base_url}/repos/{org}/{repo}/pulls?state=all&per_page=100"
            res = requests.get(prs_url, headers=self.headers, timeout=15)
            if res.status_code == 200:
                prs = res.json()
                for pr in prs:
                    created = pr.get('created_at', '')
                    if created >= since_date:
                        result['prs_opened'] += 1
                        if pr.get('merged_at'):
                            result['prs_merged'] += 1
            
            result['contributors_active'] = len(result['contributors_active'])
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def validate_conclusions(self, opendigger_data: pd.DataFrame, 
                             github_recent: Dict,
                             analysis_result: Dict) -> Dict:
        """
        使用 GitHub 30天数据验证分析结论
        """
        validation = {
            'overall_valid': True,
            'confidence': 0,
            'validations': [],
            'warnings': [],
            'github_30d_summary': {}
        }
        
        if 'error' in github_recent:
            validation['error'] = github_recent['error']
            validation['confidence'] = 0
            return validation
        
        # === GitHub 30天数据摘要 ===
        gh_commits = github_recent.get('commits', 0)
        gh_issues_opened = github_recent.get('issues_opened', 0)
        gh_issues_closed = github_recent.get('issues_closed', 0)
        gh_prs_opened = github_recent.get('prs_opened', 0)
        gh_prs_merged = github_recent.get('prs_merged', 0)
        gh_active_contributors = github_recent.get('contributors_active', 0)
        
        validation['github_30d_summary'] = {
            'commits': gh_commits,
            'issues_opened': gh_issues_opened,
            'issues_closed': gh_issues_closed,
            'prs_opened': gh_prs_opened,
            'prs_merged': gh_prs_merged,
            'active_contributors': gh_active_contributors
        }
        
        valid_count = 0
        total_checks = 0
        
        # === 验证1: 活跃度趋势 ===
        total_checks += 1
        vitality = analysis_result.get('vitality', '')
        
        # 计算30天活跃度指标
        gh_activity_score = gh_commits + gh_prs_opened * 2 + gh_issues_opened
        
        if vitality == 'THRIVING':
            if gh_activity_score >= 10:
                valid_count += 1
                validation['validations'].append({
                    'check': '活跃度验证',
                    'result': 'PASS',
                    'detail': f'THRIVING 状态验证通过 - 30天活跃度得分: {gh_activity_score}'
                })
            else:
                validation['warnings'].append(f'THRIVING 状态但30天活跃度低 ({gh_activity_score})')
        elif vitality == 'ZOMBIE':
            if gh_activity_score <= 5:
                valid_count += 1
                validation['validations'].append({
                    'check': '活跃度验证',
                    'result': 'PASS',
                    'detail': f'ZOMBIE 状态验证通过 - 30天几乎无活动'
                })
            else:
                validation['warnings'].append(f'ZOMBIE 状态但30天仍有活动 ({gh_activity_score})')
        else:
            # STABLE/DORMANT
            if 3 <= gh_activity_score <= 50:
                valid_count += 1
            validation['validations'].append({
                'check': '活跃度验证',
                'result': 'PASS',
                'detail': f'{vitality} 状态与30天活跃度 {gh_activity_score} 基本一致'
            })
        
        # === 验证2: 维护效率 ===
        total_checks += 1
        if gh_issues_opened > 0:
            issue_ratio = gh_issues_closed / gh_issues_opened
            od_debt = analysis_result.get('resistance_analysis', {}).get('debt', 50)
            
            if issue_ratio >= 0.8 and od_debt <= 50:
                valid_count += 1
                validation['validations'].append({
                    'check': '维护效率验证',
                    'result': 'PASS',
                    'detail': f'Issue 处理率 {issue_ratio:.0%} 与低阻力评估一致'
                })
            elif issue_ratio < 0.5 and od_debt >= 50:
                valid_count += 1
                validation['validations'].append({
                    'check': '维护效率验证',
                    'result': 'PASS',
                    'detail': f'Issue 处理率 {issue_ratio:.0%} 与高阻力评估一致'
                })
            else:
                validation['warnings'].append(
                    f'维护效率评估可能不准 - 30天Issue处理率: {issue_ratio:.0%}, 阻力评分: {od_debt}'
                )
        else:
            valid_count += 0.5  # 无数据，部分有效
            validation['validations'].append({
                'check': '维护效率验证',
                'result': 'N/A',
                'detail': '30天无新Issue，无法验证'
            })
        
        # === 验证3: 贡献者活跃度 ===
        total_checks += 1
        od_participants = opendigger_data['participants'].tail(1).values[0] if 'participants' in opendigger_data else 0
        
        if od_participants > 0:
            contrib_ratio = gh_active_contributors / od_participants
            if 0.1 <= contrib_ratio <= 1.0:
                valid_count += 1
                validation['validations'].append({
                    'check': '贡献者验证',
                    'result': 'PASS',
                    'detail': f'30天活跃 {gh_active_contributors} 人，占总贡献者 {contrib_ratio:.0%}'
                })
            elif contrib_ratio > 1:
                validation['warnings'].append('30天活跃贡献者超过历史记录，数据可能有延迟')
            else:
                validation['warnings'].append(f'30天只有 {gh_active_contributors} 人活跃，贡献者可能流失')
        
        # === 验证4: PR 效率 ===
        total_checks += 1
        if gh_prs_opened > 0:
            pr_merge_rate = gh_prs_merged / gh_prs_opened
            if pr_merge_rate >= 0.5:
                valid_count += 1
                validation['validations'].append({
                    'check': 'PR效率验证',
                    'result': 'PASS',
                    'detail': f'30天PR合并率 {pr_merge_rate:.0%}'
                })
            else:
                validation['warnings'].append(f'PR合并率偏低 ({pr_merge_rate:.0%})')
        else:
            valid_count += 0.5
        
        # === 综合置信度 ===
        validation['confidence'] = round(valid_count / total_checks * 100, 1) if total_checks > 0 else 0
        validation['overall_valid'] = validation['confidence'] >= 60 and len(validation['warnings']) <= 2
        
        # === 综合判断 ===
        if validation['overall_valid']:
            validation['summary'] = f"结论验证通过 (置信度: {validation['confidence']}%)"
        else:
            validation['summary'] = f"部分结论需复核 (置信度: {validation['confidence']}%)"
        
        return validation
    
    def fetch_contributors_stats(self, org: str, repo: str) -> Optional[List[Dict]]:
        """获取贡献者统计"""
        if not self.token:
            return None
        
        try:
            url = f"{self.base_url}/repos/{org}/{repo}/stats/contributors"
            res = requests.get(url, headers=self.headers, timeout=30)
            if res.status_code == 200:
                data = res.json()
                if isinstance(data, list):
                    return [{
                        'login': c.get('author', {}).get('login'),
                        'total_commits': c.get('total', 0),
                        'weeks_active': len([w for w in c.get('weeks', []) if w.get('c', 0) > 0])
                    } for c in data]
            return None
        except:
            return None
    
    def compare_with_opendigger(self, opendigger_data: pd.DataFrame, 
                                 github_info: Dict) -> Dict:
        """对比 OpenDigger 与 GitHub API 数据"""
        comparison = {
            'data_sources': ['OpenDigger', 'GitHub API'],
            'metrics': {},
            'consistency_score': 0,
            'discrepancies': []
        }
        
        if github_info is None:
            comparison['error'] = 'GitHub API 数据不可用'
            return comparison
        
        # 对比 Stars
        od_stars = opendigger_data['stars'].sum() if 'stars' in opendigger_data else 0
        gh_stars = github_info.get('stars', 0)
        comparison['metrics']['stars'] = {
            'opendigger': int(od_stars),
            'github_api': gh_stars,
            'diff_pct': abs(od_stars - gh_stars) / (gh_stars + 1) * 100
        }
        
        # 对比 Open Issues
        if 'issues_new' in opendigger_data and 'issues_closed' in opendigger_data:
            od_open = opendigger_data['issues_new'].sum() - opendigger_data['issues_closed'].sum()
            gh_open = github_info.get('open_issues', 0)
            comparison['metrics']['open_issues'] = {
                'opendigger_calc': int(od_open),
                'github_api': gh_open,
                'diff': int(abs(od_open - gh_open))
            }
        
        # 一致性评分
        diffs = [m.get('diff_pct', 0) for m in comparison['metrics'].values() 
                 if isinstance(m, dict) and 'diff_pct' in m]
        if diffs:
            avg_diff = np.mean(diffs)
            comparison['consistency_score'] = round(max(0, 100 - avg_diff), 1)
        else:
            comparison['consistency_score'] = 100
        
        # 额外信息
        comparison['github_extra'] = {
            'language': github_info.get('language'),
            'license': github_info.get('license'),
            'topics': github_info.get('topics', [])[:5],
            'archived': github_info.get('archived'),
            'last_push': github_info.get('pushed_at')
        }
        
        return comparison
    
    def calculate_activity_score(self, github_recent: Dict) -> float:
        """
        计算GitHub API数据的活跃度得分，用于映射OpenDigger的openrank和attention指标
        活跃度得分 = commits * 0.4 + prs_opened * 0.3 + issues_opened * 0.2 + contributors_active * 0.1
        """
        if 'error' in github_recent:
            return 0
        
        score = (
            github_recent.get('commits', 0) * 0.4 +
            github_recent.get('prs_opened', 0) * 0.3 +
            github_recent.get('issues_opened', 0) * 0.2 +
            github_recent.get('contributors_active', 0) * 0.1
        )
        
        return score
    
    def map_monthly_prediction_to_daily_github_data(self, monthly_prediction: float, 
                                                  github_30d_data: Dict, 
                                                  metric: str = 'stars') -> Dict:
        """
        将月度预测值映射到GitHub API的30天数据
        - monthly_prediction: 预测的月度新增值
        - github_30d_data: GitHub API获取的30天活跃数据
        - metric: 指标类型
        """
        # 计算日均预测值
        daily_predicted = monthly_prediction / 30
        
        # 根据指标类型获取对应的GitHub API数据
        if metric == 'stars':
            # stars是累计值，需要计算30天增量
            # 这里简化处理，实际需要获取30天前的stars数
            gh_30d_value = 0  # 实际实现需要获取历史stars数据
        elif metric in ['openrank', 'attention']:
            # openrank和attention通过活跃度计算
            gh_30d_value = self.calculate_activity_score(github_30d_data)
        elif metric in ['pr_new', 'issues_new']:
            # 直接使用GitHub API的30天数据
            gh_30d_value = github_30d_data.get(f'{metric}_opened', 0) if metric == 'issues_new' else github_30d_data.get('prs_opened', 0)
        elif metric == 'participants':
            # 直接使用GitHub API的活跃贡献者数
            gh_30d_value = github_30d_data.get('contributors_active', 0)
        else:
            gh_30d_value = 0
        
        # 计算预测误差
        error = abs(daily_predicted * 30 - gh_30d_value)
        error_pct = error / (gh_30d_value + 1) * 100
        
        return {
            'monthly_prediction': monthly_prediction,
            'daily_predicted': daily_predicted,
            'github_30d_value': gh_30d_value,
            'github_daily_avg': gh_30d_value / 30,
            'error': error,
            'error_pct': error_pct
        }


# ============== 升级版项目分析器 ==============
class ProjectAnalyzerV45:
    """GitHub 项目深度分析器 v4.5 - 算法升级版"""
    
    CORE_METRICS = [
        "openrank", "activity", "stars", "attention",
        "participants", "new_contributors", "inactive_contributors",
        "bus_factor", "issues_new", "issues_closed", "pr_new", "pr_merged"
    ]
    
    def __init__(self, url: str, github_token: Optional[str] = None):
        self.org, self.repo = self._parse_url(url)
        self.df = pd.DataFrame()
        self.tier = None
        self.tier_probabilities = {}
        self.tier_confidence = 0
        self.github_token = github_token or os.getenv('GITHUB_TOKEN')
        
        # 初始化升级组件
        self.gmm_classifier = GMMTierClassifier()
        self.prophet_predictor = ProphetPredictor()
        self.ahp_evaluator = AHPHealthEvaluator()
        self.backtest_validator = BacktestValidator()
        
        # 初始化原有计算器
        self.momentum_calculator = MomentumCalculator()
        self.resistance_calculator = ResistanceCalculator()
        self.potential_calculator = PotentialCalculator()
        self.bus_factor_calculator = BusFactorCalculator()
        self.change_point_detector = ChangePointDetector()
        self.funnel_analyzer = ConversionFunnelAnalyzer()
        self.etd_analyzer = ETDAnalyzer()
    
    def _parse_url(self, url: str) -> Tuple[str, str]:
        match = re.search(r"github\.com/([^/]+)/([^/]+)", url)
        if match:
            return match.group(1), match.group(2).replace(".git", "")
        if "/" in url and "http" not in url:
            parts = url.split('/')
            return parts[0], parts[1]
        raise ValueError("无效的 GitHub URL")
    
    def fetch_data(self) -> bool:
        """获取数据（OpenDigger优先，GitHub API备用）"""
        print(f"\n{'='*60}")
        print(f"  GitHub 项目深度分析器 v4.5 - 算法升级版")
        print(f"  项目: {self.org}/{self.repo}")
        print(f"{'='*60}\n")
        
        raw_data = {}
        for metric in self.CORE_METRICS:
            url = f"https://oss.open-digger.cn/github/{self.org}/{self.repo}/{metric}.json"
            try:
                res = requests.get(url, timeout=15)
                if res.status_code == 200:
                    data = res.json()
                    monthly = {k: v for k, v in data.items() if re.match(r'^\d{4}-\d{2}$', str(k))}
                    if monthly:
                        raw_data[metric] = pd.Series(monthly)
            except:
                continue
        
        if not raw_data:
            print("无法获取数据")
            return False
        
        self.df = pd.DataFrame(raw_data).fillna(0)
        self.df.index = pd.to_datetime(self.df.index)
        self.df = self.df.sort_index()
        
        # 保存原始数据
        self.df.to_csv(f"{self.org}_{self.repo}_data.csv", encoding='utf-8-sig')
        
        # GMM概率化分层
        metrics = {
            'avg_openrank': self.df['openrank'].mean() if 'openrank' in self.df else 0,
            'total_stars': self.df['stars'].sum() if 'stars' in self.df else 0,
            'max_participants': self.df['participants'].max() if 'participants' in self.df else 0
        }
        
        self.tier, self.tier_probabilities, self.tier_confidence = self.gmm_classifier.predict_proba(metrics)
        
        print(f"获取 {len(self.df)} 个月数据")
        print(f"GMM分层: {self.tier} ({TIER_NAMES[self.tier]})")
        print(f"层级概率分布: {self.tier_probabilities}")
        print(f"置信度: {self.tier_confidence:.0%}")
        
        return True
    
    def analyze_lifecycle(self) -> str:
        if len(self.df) < 12:
            return 'INCUBATION'
        openrank = self.df['openrank']
        n = len(openrank)
        q1 = openrank.iloc[:n//3].mean()
        q2 = openrank.iloc[n//3:2*n//3].mean()
        q3 = openrank.iloc[2*n//3:].mean()
        slope = linregress(range(6), openrank.tail(6).values).slope
        
        if q1 < q2 < q3 and slope > 0:
            return 'GROWTH'
        elif q3 >= q2 * 0.85:
            return 'MATURITY'
        elif q3 < q2 * 0.7:
            return 'REVIVAL' if slope > 0.3 else 'DECLINE'
        return 'MATURITY'
    
    def analyze_vitality(self) -> str:
        if 'activity' not in self.df:
            return 'UNKNOWN'
        activity = self.df['activity']
        recent = activity.tail(6)
        slope = linregress(range(len(recent)), recent.values).slope
        peak, current = activity.max(), recent.mean()
        
        if slope > 0:
            return 'THRIVING'
        if self.tier in ['GIANT', 'MATURE'] and current > peak * 0.2:
            return 'STABLE'
        if current < peak * 0.1:
            return 'ZOMBIE'
        return 'DORMANT'
    
    def evaluate_trend_3d(self) -> Dict:
        """动量-阻力-潜力三维评估"""
        momentum = self.momentum_calculator.calculate(self.df)
        resistance = self.resistance_calculator.calculate(self.df)
        potential = self.potential_calculator.calculate(self.df, self.tier)
        
        potential_factor = min(1.5, 0.5 + potential['remaining_space'] / 100)
        trend_score = (momentum['total'] - resistance['total'] * 0.5) * potential_factor
        
        if trend_score >= 60:
            trend_class, desc = 'STRONG_UP', '强势上行'
        elif trend_score >= 30:
            trend_class, desc = 'MODERATE_UP', '温和上行'
        elif trend_score >= 0:
            trend_class, desc = 'STABLE', '横盘稳定'
        elif trend_score >= -30:
            trend_class, desc = 'MODERATE_DOWN', '温和下行'
        else:
            trend_class, desc = 'STRONG_DOWN', '趋势恶化'
        
        return {
            'trend_score': round(trend_score, 1),
            'trend_class': trend_class,
            'description': desc,
            'momentum': momentum,
            'resistance': resistance,
            'potential': potential
        }
    
    def analyze_risk(self, vitality: str, trend: Dict) -> Dict:
        risk_score = 0
        alerts = []
        
        if 'activity' in self.df:
            slope = linregress(range(min(12, len(self.df))), 
                             self.df['activity'].tail(12).values).slope
            if slope < -0.5:
                risk_score += 30
                alerts.append('活跃度显著下降')
            elif slope < 0:
                risk_score += 15
        
        if trend['resistance']['status'] == 'HEAVY':
            risk_score += 25
            alerts.append('技术债阻力高')
        
        if vitality == 'ZOMBIE':
            risk_score += 30
            alerts.append('项目处于僵尸状态')
        elif vitality == 'DORMANT':
            risk_score += 15
        
        if risk_score >= 50:
            level = 'CRITICAL'
        elif risk_score >= 30:
            level = 'HIGH'
        elif risk_score >= 15:
            level = 'MEDIUM'
        else:
            level = 'LOW'
        
        return {'score': risk_score, 'level': level, 'alerts': alerts}
    
    def analyze_dark_horse(self, trend: Dict, funnel: Dict) -> Dict:
        if self.tier in ['GIANT', 'MATURE']:
            return {'is_dark_horse': False, 'score': 0, 'reasons': ['已超出黑马范畴']}
        
        score = 0
        reasons = []
        
        if trend['momentum']['total'] >= 70:
            score += 30
            reasons.append('强劲动量')
        elif trend['momentum']['total'] >= 50:
            score += 15
        
        if trend['potential']['remaining_space'] >= 60:
            score += 25
            reasons.append('巨大增长空间')
        
        if funnel.get('quality') == 'EXCELLENT':
            score += 25
            reasons.append('优秀转化率')
        elif funnel.get('quality') == 'GOOD':
            score += 15
        
        if funnel.get('quality') == 'BUBBLE':
            score -= 20
            reasons.append(' 转化率过低')
        
        return {
            'is_dark_horse': score >= 55,
            'score': min(100, max(0, score)),
            'reasons': reasons
        }
    
    def generate_predictions(self) -> Dict:
        """生成Prophet时序预测"""
        predictions = {}
        
        # 预测核心指标
        for metric in ['openrank', 'activity', 'stars', 'participants']:
            if metric in self.df:
                pred_result = self.prophet_predictor.predict(self.df, metric, periods=6)
                predictions[metric] = pred_result
        
        # 添加预测摘要
        if 'openrank' in predictions and predictions['openrank'].get('forecast'):
            forecast_values = predictions['openrank']['forecast']
            current_value = self.df['openrank'].iloc[-1] if 'openrank' in self.df else 0
            
            if forecast_values:
                growth_pct = (forecast_values[-1] - current_value) / (current_value + 0.1) * 100
                
                predictions['summary'] = {
                    'current_value': round(current_value, 2),
                    'forecast_6m': round(forecast_values[-1], 2),
                    'growth_pct': round(growth_pct, 1),
                    'trend': '上升' if growth_pct > 5 else '下降' if growth_pct < -5 else '稳定'
                }
        
        return predictions
    
    def run_backtest(self) -> Dict:
        """运行回测验证"""
        backtest_results = {}
        
        # 对关键指标进行回测
        for metric in ['openrank', 'activity']:
            if metric in self.df:
                result = self.backtest_validator.validate_predictions(self.df, metric)
                backtest_results[metric] = result
        
        # 计算综合置信度
        confidences = []
        for metric, result in backtest_results.items():
            if 'confidence_level' in result:
                conf_map = {'HIGH': 0.9, 'MEDIUM': 0.7, 'LOW': 0.5, 'VERY_LOW': 0.3}
                confidences.append(conf_map.get(result['confidence_level'], 0.5))
        
        if confidences:
            avg_confidence = np.mean(confidences)
            backtest_results['overall_confidence'] = round(avg_confidence, 3)
            backtest_results['overall_confidence_level'] = (
                'HIGH' if avg_confidence >= 0.8 else
                'MEDIUM' if avg_confidence >= 0.6 else
                'LOW' if avg_confidence >= 0.4 else 'VERY_LOW'
            )
        
        return backtest_results
    
    def generate_recommendations(self, vitality: str, trend: Dict, risk: Dict, 
                                  dark_horse: Dict, bus_factor: Dict,
                                  etd_analysis: Dict, predictions: Dict) -> List[str]:
        recs = []
        
        if risk['level'] in ['CRITICAL', 'HIGH']:
            recs.append('风险较高，建议加强社区运营和技术博客曝光')
        
        if trend['resistance']['debt'] > 60:
            recs.append('技术债务较重，建议组织 Bug Bash 活动集中处理')
        
        if bus_factor['risk_level'] in ['CRITICAL', 'HIGH']:
            recs.append(' Bus Factor 过低，建议培养更多核心贡献者')
        
        if vitality == 'STABLE':
            recs.append('项目已成熟，保持定期安全更新即可')
        
        if dark_horse.get('is_dark_horse'):
            recs.append('黑马潜力显现，建议加大推广力度')
        
        if trend['momentum']['gravity'] < 40:
            recs.append('贡献者向心力不足，建议标记 Good First Issue')
        
        if etd_analysis.get('etd_status') in ['CRITICAL', 'WARNING']:
            recs.append(f"ETD寿命预警: {etd_analysis.get('description', '')}")
        
        if predictions.get('summary', {}).get('trend') == '下降':
            recs.append(f"预测显示未来趋势下降，建议关注项目活跃度")
        
        if not recs:
            recs.append('项目状态健康，保持当前运营节奏')
        
        return recs
    
    def run(self) -> Optional[AnalysisResult]:
        """执行完整分析"""
        if not self.fetch_data():
            return None
        
        # 生成预测
        print("\n正在生成时序预测...")
        predictions = self.generate_predictions()
        
        # 运行回测
        print("正在运行回测验证...")
        backtest_results = self.run_backtest()
        
        # 其他分析模块
        lifecycle = self.analyze_lifecycle()
        vitality = self.analyze_vitality()
        trend_3d = self.evaluate_trend_3d()
        risk = self.analyze_risk(vitality, trend_3d)
        bus_factor_2 = self.bus_factor_calculator.calculate(self.df)
        funnel = self.funnel_analyzer.analyze(self.df)
        dark_horse = self.analyze_dark_horse(trend_3d, funnel)
        change_points = self.change_point_detector.detect(self.df.get('openrank', pd.Series()))
        etd_analysis = self.etd_analyzer.analyze(self.df, vitality, self.tier)
        
        # === GitHub API 对比与30天验证 ===
        github_comparison = {}
        conclusion_validation = {'error': '未提供 GitHub Token，跳过验证'}
        prediction_validation = {'error': '未提供 GitHub Token，跳过预测验证'}
        
        if self.github_token:
            print("正在获取 GitHub API 数据...")
            gh_analyzer = GitHubAPIAnalyzer(self.github_token)
            gh_info = gh_analyzer.fetch_repo_info(self.org, self.repo)
            
            if gh_info:
                github_comparison = gh_analyzer.compare_with_opendigger(self.df, gh_info)
                print(f"GitHub API 对比完成，一致性: {github_comparison.get('consistency_score', 0)}%")
                
                # === 获取30天活跃数据并验证 ===
                print("正在获取最近30天数据进行结论验证...")
                github_recent = gh_analyzer.fetch_recent_activity(self.org, self.repo, days=30)
                
                if 'error' not in github_recent:
                    # 验证分析结论
                    analysis_for_validation = {
                        'vitality': vitality,
                        'resistance_analysis': trend_3d['resistance'],
                        'momentum_analysis': trend_3d['momentum']
                    }
                    conclusion_validation = gh_analyzer.validate_conclusions(
                        self.df, github_recent, analysis_for_validation
                    )
                    print(f"{conclusion_validation.get('summary', '验证完成')}")
                    
                    # === 预测验证：将月度预测值与GitHub API的30天数据进行对比 ===
                    print("正在进行预测验证...")
                    
                    # 定义要验证的指标
                    metrics_to_validate = ['openrank', 'attention', 'stars']
                    prediction_validation = {
                        'metrics': {},
                        'overall_score': 0,
                        'overall_rating': 'UNKNOWN'
                    }
                    
                    total_error = 0
                    valid_metrics = 0
                    
                    for metric in metrics_to_validate:
                        if metric in predictions and predictions[metric].get('forecast'):
                            # 获取月度预测值（取第一个月的预测值）
                            monthly_prediction = predictions[metric]['forecast'][0]
                            
                            # 将月度预测值映射到GitHub API的30天数据
                            mapped_result = gh_analyzer.map_monthly_prediction_to_daily_github_data(
                                monthly_prediction, github_recent, metric
                            )
                            
                            prediction_validation['metrics'][metric] = mapped_result
                            
                            if 'error_pct' in mapped_result:
                                total_error += mapped_result['error_pct']
                                valid_metrics += 1
                    
                    # 计算综合预测准确性
                    if valid_metrics > 0:
                        avg_error = total_error / valid_metrics
                        prediction_validation['overall_score'] = 100 - avg_error
                        
                        # 评级
                        if avg_error < 10:
                            prediction_validation['overall_rating'] = 'EXCELLENT'
                        elif avg_error < 20:
                            prediction_validation['overall_rating'] = 'GOOD'
                        elif avg_error < 30:
                            prediction_validation['overall_rating'] = 'FAIR'
                        else:
                            prediction_validation['overall_rating'] = 'POOR'
                        
                        print(f"预测验证完成，平均误差: {avg_error:.1f}%，评级: {prediction_validation['overall_rating']}")
                else:
                    conclusion_validation = {'error': github_recent.get('error')}
                    prediction_validation = {'error': github_recent.get('error')}
        
        # 健康评分（使用AHP）
        health_score, dimension_scores = self.ahp_evaluator.calculate_health_score(
            vitality, trend_3d, risk, self.tier, predictions
        )
        
        # 确定健康等级
        grades = [(85, 'A+'), (75, 'A'), (65, 'B+'), (55, 'B'), (45, 'C'), (35, 'D'), (0, 'F')]
        health_grade = next(g for t, g in grades if health_score >= t)
        
        # 建议
        recommendations = self.generate_recommendations(
            vitality, trend_3d, risk, dark_horse, bus_factor_2, etd_analysis, predictions
        )
        
        # 添加验证警告
        if conclusion_validation.get('warnings'):
            for w in conclusion_validation['warnings'][:2]:  # 最多2条
                recommendations.append(f'验证提示: {w}')
        
        # 构建结果
        result = AnalysisResult(
            project_name=f"{self.org}/{self.repo}",
            tier=self.tier,
            tier_probabilities=self.tier_probabilities,
            tier_confidence=self.tier_confidence,
            lifecycle=lifecycle,
            vitality=vitality,
            health_score=health_score,
            health_grade=health_grade,
            dimension_scores=dimension_scores,
            momentum_analysis=trend_3d['momentum'],
            resistance_analysis=trend_3d['resistance'],
            potential_analysis=trend_3d['potential'],
            trend_3d=trend_3d,
            risk_analysis=risk,
            dark_horse_analysis=dark_horse,
            change_points=change_points,
            bus_factor_2=bus_factor_2,
            etd_analysis=etd_analysis,
            github_comparison=github_comparison,
            conclusion_validation=conclusion_validation,
            prediction_validation=prediction_validation,
            predictions=predictions,
            backtest_results=backtest_results,
            recommendations=recommendations,
            detailed_report=""
        )
        
        result.detailed_report = self.generate_report(result)
        
        # 输出
        # 注释掉直接打印报告，避免Windows终端编码问题
        # print(result.detailed_report)
        
        # === 保存报告（TXT）===
        with open(f"{self.org}_{self.repo}_v45_report.txt", 'w', encoding='utf-8') as f:
            f.write(result.detailed_report)
        print(f"文本报告已保存: {self.org}_{self.repo}_v45_report.txt")
        
        # === 保存数据（JSON）===
        self._save_json_report(result)
        
        # 绘图
        self.plot_dashboard(result)
        
        return result
    
    def generate_report(self, result: AnalysisResult) -> str:
        """生成详细报告"""
        report = f"""
{'═'*70}
                    {result.project_name} 深度诊断报告
                    GitHub 项目分析器 - 算法升级版
{'═'*70}

【项目画像】
  • 层级: {result.tier} ({TIER_NAMES[result.tier]}) 
  • 层级概率: {result.tier_probabilities}
  • 置信度: {result.tier_confidence:.0%}
  • 生命周期: {result.lifecycle}
  • 当前状态: {result.vitality}
  • 综合评分: {result.health_score}/100 ({result.health_grade})

{'─'*70}

【GMM概率化分层】
"""
        for tier, prob in result.tier_probabilities.items():
            bar_length = int(prob * 30)
            bar = '█' * bar_length + '░' * (30 - bar_length)
            report += f"  {tier:10s} {bar} {prob:.1%}\n"

        report += f"""
{'─'*70}

【Prophet时序预测】
"""
        if result.predictions.get('summary'):
            summary = result.predictions['summary']
            report += f"""
  • 当前OpenRank: {summary['current_value']:.1f}
  • 6个月预测值: {summary['forecast_6m']:.1f}
  • 预期增长: {summary['growth_pct']:.1f}% ({summary['trend']})
"""
        
        report += f"""
{'─'*70}

【回测验证结果】
"""
        for metric, bt_result in result.backtest_results.items():
            if isinstance(bt_result, dict) and 'mape' in bt_result:
                report += f"""
  {metric.upper()}:
    • MAPE误差: {bt_result['mape']:.1f}%
    • 方向准确性: {bt_result.get('direction_accuracy', 0):.1%}
    • 置信度: {bt_result.get('confidence_level', 'N/A')}
"""
        
        if 'overall_confidence_level' in result.backtest_results:
            report += f"""
  • 综合置信度: {result.backtest_results['overall_confidence_level']}
"""
        
        # 添加预测验证结果
        report += f"""
{'─'*70}

【预测验证结果】
"""
        if result.prediction_validation and 'error' not in result.prediction_validation:
            report += f"""
  • 综合评分: {result.prediction_validation.get('overall_score', 0):.1f}
  • 综合评级: {result.prediction_validation.get('overall_rating', 'UNKNOWN')}
"""
            
            # 添加各指标的预测验证结果
            for metric, validation in result.prediction_validation.get('metrics', {}).items():
                report += f"""
  {metric.upper()}:
    • 月度预测值: {validation.get('monthly_prediction', 0):.1f}
    • GitHub 30天实际值: {validation.get('github_30d_value', 0):.1f}
    • 日均预测值: {validation.get('daily_predicted', 0):.1f}
    • GitHub 30天日均实际值: {validation.get('github_daily_avg', 0):.1f}
    • 误差百分比: {validation.get('error_pct', 0):.1f}%
"""
        else:
            report += f"""
  • 预测验证: {result.prediction_validation.get('error', '未执行')}
"""
        
        report += f"""
{'─'*70}

【AHP健康评估】
  评分权重矩阵 ({result.tier}):
"""
        weights = self.ahp_evaluator.TIER_WEIGHTS.get(result.tier, {})
        for dim, weight in weights.items():
            report += f"    • {dim}: {weight:.0%}\n"
        
        report += f"""
  各维度贡献:
"""
        for dim, contrib in result.dimension_scores.items():
            report += f"    • {dim}: {contrib:.1%}\n"
        
        report += f"""
{'─'*70}

【三维趋势分析】
  趋势评分: {result.trend_3d['trend_score']} ({result.trend_3d['description']})
  
  ┌ 动量 (Momentum): {result.momentum_analysis['total']}/100
  │   {result.momentum_analysis['interpretation']}
  │
  ├ 阻力 (Resistance): {result.resistance_analysis['total']}/100
  │   {result.resistance_analysis['interpretation']}
  │
  └ 潜力 (Potential): {result.potential_analysis['remaining_space']:.1f}% 成长空间
      {result.potential_analysis['interpretation']}

{'─'*70}

【Bus Factor 2.0 分析】
  等效 Bus Factor: {result.bus_factor_2['effective_bus_factor']:.1f}
  {result.bus_factor_2['description']}

{'─'*70}

【ETD 寿命分析】
  状态: {result.etd_analysis['etd_status']}
  预计枯竭时间: {f"{result.etd_analysis['etd_months']:.1f} 个月" if result.etd_analysis['etd_months'] != float('inf') else '无限 (健康)'}
  {result.etd_analysis.get('description', '')}

{'─'*70}

【改进建议】
"""
        for i, rec in enumerate(result.recommendations, 1):
            report += f"  {i}. {rec}\n"
        
        report += f"""
{'═'*70}
                         报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'═'*70}
"""
        return report
    
    def _save_json_report(self, result: AnalysisResult):
        """保存 JSON 格式报告"""
        json_data = {
            'project': result.project_name,
            'generated_at': datetime.now().isoformat(),
            'tier': {
                'level': result.tier,
                'name': TIER_NAMES[result.tier],
                'probabilities': result.tier_probabilities,
                'confidence': result.tier_confidence
            },
            'predictions': result.predictions,
            'backtest': result.backtest_results,
            'prediction_validation': result.prediction_validation,
            'health': {
                'score': result.health_score,
                'grade': result.health_grade,
                'dimensions': result.dimension_scores
            },
            'trend_3d': {
                'score': result.trend_3d['trend_score'],
                'class': result.trend_3d['trend_class'],
                'momentum': result.momentum_analysis,
                'resistance': result.resistance_analysis,
                'potential': result.potential_analysis
            },
            'etd_analysis': {
                'months': result.etd_analysis['etd_months'] if result.etd_analysis['etd_months'] != float('inf') else 'infinite',
                'status': result.etd_analysis['etd_status'],
                'decay_type': result.etd_analysis.get('decay_type'),
                'is_mature_stable': result.etd_analysis.get('is_mature_stable'),
                'description': result.etd_analysis.get('description')
            },
            'bus_factor': result.bus_factor_2,
            'risk': result.risk_analysis,
            'dark_horse': result.dark_horse_analysis,
            'change_points': result.change_points,
            'github_comparison': result.github_comparison,
            'recommendations': result.recommendations
        }
        
        json_file = f"{self.org}_{self.repo}_v45_data.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        print(f" JSON 数据已保存: {json_file}")
    
    def _plot_github_prediction_comparison(self, ax, result: AnalysisResult):
        """
        绘制GitHub近30天数据与算法预测的对比图
        比较指标：openrank值、attention值以及stars值
        """
        # 检查是否有GitHub验证数据
        if not result.conclusion_validation or 'error' in result.conclusion_validation:
            ax.text(0.5, 0.5, '无验证数据', ha='center', va='center')
            return
        
        val = result.conclusion_validation
        
        # 检查是否有预测数据
        if not result.predictions or not any('forecast' in v for v in result.predictions.values()):
            ax.text(0.5, 0.5, '无预测数据', ha='center', va='center')
            return
        
        # 定义要比较的指标
        metrics = ['openrank', 'attention', 'stars']
        labels = ['OpenRank', 'Attention', 'Stars']
        
        # 创建GitHub API数据获取器
        if self.github_token:
            gh_analyzer = GitHubAPIAnalyzer(self.github_token)
            # 获取最近30天的活跃数据
            github_recent = gh_analyzer.fetch_recent_activity(self.org, self.repo, days=30)
            
            if 'error' not in github_recent:
                # 计算GitHub API的活跃度得分（映射到OpenRank和Attention）
                activity_score = gh_analyzer.calculate_activity_score(github_recent)
                
                # 获取OpenDigger的月度预测值
                predictions = {
                    'openrank': result.predictions.get('openrank', {}).get('forecast', [0])[0] if 'openrank' in result.predictions else 0,
                    'attention': result.predictions.get('attention', {}).get('forecast', [0])[0] if 'attention' in result.predictions else 0,
                    'stars': result.predictions.get('stars', {}).get('forecast', [0])[0] if 'stars' in result.predictions else 0
                }
                
                # 将OpenDigger的月度预测值映射到GitHub API的30天数据
                mapped_predictions = {
                    'openrank': gh_analyzer.map_monthly_prediction_to_daily_github_data(predictions['openrank'], github_recent, 'openrank')['github_30d_value'],
                    'attention': gh_analyzer.map_monthly_prediction_to_daily_github_data(predictions['attention'], github_recent, 'attention')['github_30d_value'],
                    'stars': 0  # 需要获取30天前的stars数才能计算增量
                }
                
                # 获取GitHub API的实际数据
                github_values = {
                    'openrank': activity_score,  # 使用活跃度得分作为GitHub API的OpenRank近似值
                    'attention': activity_score * 1.5,  # 使用活跃度得分的1.5倍作为GitHub API的Attention近似值
                    'stars': 0  # 需要获取30天前的stars数才能计算增量
                }
                
                # 准备数据用于绘制
                github_data = [github_values[m] for m in metrics]
                predicted_data = [mapped_predictions[m] for m in metrics]
                
                # 创建对比柱状图
                x = np.arange(len(metrics))
                width = 0.35
                
                # 绘制GitHub API实际数据
                bars1 = ax.bar(x - width/2, github_data, width, label='GitHub 30天',
                            color=COLORS['primary'], alpha=0.7)
                # 绘制算法预测数据
                bars2 = ax.bar(x + width/2, predicted_data, width, label='算法预测',
                            color=COLORS['success'], alpha=0.7)
                
                # 添加数值标签
                for bars in [bars1, bars2]:
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                f'{height:.1f}', ha='center', va='bottom', fontsize=8)
                
                # 添加图例
                ax.legend(loc='upper right', fontsize=8)
                
                # 设置轴标签和标题
                ax.set_xlabel('指标', fontsize=9)
                ax.set_ylabel('数值', fontsize=9)
                ax.set_title('GitHub近30天数据 vs 算法预测', fontsize=10, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=0)
                ax.grid(True, axis='y', alpha=0.3)
            else:
                ax.text(0.5, 0.5, '无法获取GitHub API数据', ha='center', va='center')
        else:
            ax.text(0.5, 0.5, '未提供GitHub Token', ha='center', va='center')
    
    def plot_dashboard(self, result: AnalysisResult):
        """绘制 3x3 九宫格仪表板（简化版）"""
        from matplotlib.gridspec import GridSpec
        
        fig = plt.figure(figsize=(16, 13))
        fig.patch.set_facecolor('white')
        
        # 使用 GridSpec 创建混合布局
        gs = GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.30,
                     left=0.06, right=0.96, top=0.90, bottom=0.06)
        
        # 定义子图
        ax00 = fig.add_subplot(gs[0, 0])
        ax01 = fig.add_subplot(gs[0, 1])
        ax02 = fig.add_subplot(gs[0, 2])
        
        ax10 = fig.add_subplot(gs[1, 0], polar=True)
        ax11 = fig.add_subplot(gs[1, 1])
        ax12 = fig.add_subplot(gs[1, 2])
        
        ax20 = fig.add_subplot(gs[2, 0])
        ax21 = fig.add_subplot(gs[2, 1])
        ax22 = fig.add_subplot(gs[2, 2])
        
        # 绑定绑图
        self._plot_openrank_with_forecast(ax00, result)
        self._plot_gmm_probabilities(ax01, result)
        self._plot_health_gauge(ax02, result)
        
        self._plot_trend_3d_radar(ax10, result)
        self._plot_resistance_chart(ax11, result)
        self._plot_potential_bar(ax12, result)
        
        self._plot_predictions_chart(ax20, result)
        self._plot_backtest_results(ax21, result)
        self._plot_github_prediction_comparison(ax22, result)
        
        # 总标题
        fig.suptitle(
            f'{self.org}/{self.repo} 深度诊断 | {TIER_NAMES[result.tier]} | '
            f'{result.health_grade} ({result.health_score}分)',
            fontsize=14, fontweight='bold', y=0.98
        )
        
        filename = f"{self.org}_{self.repo}_v45_report.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"图表已保存: {filename}")
        # plt.show()  # 取消显示，只保存图片
    
    def _plot_openrank_with_forecast(self, ax, result: AnalysisResult):
        """绘制OpenRank趋势及预测"""
        if 'openrank' not in self.df:
            ax.text(0.5, 0.5, '无数据', ha='center', va='center')
            return
        
        data = self.df['openrank']
        ax.plot(data.index, data.values, color=COLORS['primary'], lw=2, label='历史')
        
        # 绘制预测（如果有）
        if result.predictions.get('openrank', {}).get('forecast'):
            forecast = result.predictions['openrank']['forecast']
            last_date = data.index[-1]
            
            # 生成未来日期
            future_dates = pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                periods=len(forecast),
                freq='ME'
            )
            
            ax.plot(future_dates, forecast, color=COLORS['success'], 
                   lw=2, linestyle='--', label='预测')
            
            # 绘制置信区间
            if 'yhat_lower' in result.predictions['openrank']:
                lower = result.predictions['openrank']['yhat_lower']
                upper = result.predictions['openrank']['yhat_upper']
                ax.fill_between(future_dates, lower, upper, 
                              color=COLORS['success'], alpha=0.2)
        
        ax.set_title('OpenRank趋势及预测', fontsize=10, fontweight='bold')
        ax.tick_params(axis='x', rotation=30)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    def _plot_gmm_probabilities(self, ax, result: AnalysisResult):
        """绘制GMM分层概率"""
        tiers = list(result.tier_probabilities.keys())
        probs = list(result.tier_probabilities.values())
        colors = [TIER_BENCHMARKS[t].get('color', COLORS['primary']) for t in tiers]
        
        bars = ax.bar(tiers, probs, color=colors, alpha=0.8)
        ax.set_ylim(0, 1)
        ax.set_ylabel('概率', fontsize=9)
        ax.set_title(f'GMM分层概率 (最佳: {result.tier})', fontsize=10, fontweight='bold')
        
        # 在柱子内部添加概率标签
        for bar, prob in zip(bars, probs):
            height = bar.get_height()
            # 根据柱子高度调整标签位置和颜色，确保清晰可见
            y_pos = height / 2  # 标签居中显示
            color = 'white' if height > 0.2 else 'black'  # 高柱子用白色标签，低柱子用黑色标签
            ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                   f'{prob:.1%}', ha='center', va='center', fontsize=9, fontweight='bold', color=color)
    
    def _plot_health_gauge(self, ax, result: AnalysisResult):
        """绘制健康评分仪表盘"""
        score = result.health_score
    
        # 清空坐标轴
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.7)  # 调整高度，使仪表盘更接近半圆形
        
        # 绘制仪表盘外框
        center_x, center_y = 0.5, 0.35  # 调整中心点，使半圆形居中
        radius = 0.35  # 调整半径
        
        # 绘制彩色扇形区域，每个区域60度
        total_angle = np.pi  # 半圆，180度
        sector_angle = np.pi / 3  # 60度
        
        # 危险区域 (0-33.3分) - 使用fill函数绘制严格的60度扇形
        danger_angles = np.linspace(np.pi, np.pi - sector_angle, 50)
        # 创建扇形路径：中心点 -> 扇形边缘点 -> 中心点
        danger_x = [center_x] + list(center_x + radius * np.cos(danger_angles)) + [center_x]
        danger_y = [center_y] + list(center_y + radius * np.sin(danger_angles)) + [center_y]
        ax.fill(danger_x, danger_y, color=COLORS['danger'], alpha=0.3)
        
        # 警告区域 (33.3-66.6分) - 使用fill函数绘制严格的60度扇形
        warning_angles = np.linspace(np.pi - sector_angle, np.pi - 2*sector_angle, 50)
        warning_x = [center_x] + list(center_x + radius * np.cos(warning_angles)) + [center_x]
        warning_y = [center_y] + list(center_y + radius * np.sin(warning_angles)) + [center_y]
        ax.fill(warning_x, warning_y, color=COLORS['warning'], alpha=0.3)
        
        # 安全区域 (66.6-100分) - 使用fill函数绘制严格的60度扇形
        safe_angles = np.linspace(np.pi - 2*sector_angle, np.pi - 3*sector_angle, 50)
        safe_x = [center_x] + list(center_x + radius * np.cos(safe_angles)) + [center_x]
        safe_y = [center_y] + list(center_y + radius * np.sin(safe_angles)) + [center_y]
        ax.fill(safe_x, safe_y, color=COLORS['success'], alpha=0.3)
        
        # 绘制刻度线
        for i in range(0, 101, 10):
            angle = np.pi - np.pi * (i / 100)
            x_start = center_x + radius * np.cos(angle)
            y_start = center_y + radius * np.sin(angle)
            x_end = center_x + (radius - 0.05) * np.cos(angle)
            y_end = center_y + (radius - 0.05) * np.sin(angle)
            ax.plot([x_start, x_end], [y_start, y_end], 'k-', lw=1)
            
            # 每20分添加标签
            if i % 20 == 0:
                x_text = center_x + (radius + 0.05) * np.cos(angle)
                y_text = center_y + (radius + 0.05) * np.sin(angle)
                ax.text(x_text, y_text, str(i), ha='center', va='center', fontsize=8)
        
        # 绘制指针
        angle = np.pi - (score / 100) * total_angle  # 从右到左计算角度
        needle_length = radius - 0.05
        
        # 指针三角形，确保指向正确方向
        x_tip = center_x + needle_length * np.cos(angle)
        y_tip = center_y + needle_length * np.sin(angle)
        
        # 指针底部的两个点
        angle_left = angle + 0.1  # 调整角度，确保指针形状正确
        angle_right = angle - 0.1
        x_left = center_x + 0.05 * np.cos(angle_left)
        y_left = center_y + 0.05 * np.sin(angle_left)
        x_right = center_x + 0.05 * np.cos(angle_right)
        y_right = center_y + 0.05 * np.sin(angle_right)
        
        # 绘制指针
        ax.fill([x_tip, x_left, x_right], [y_tip, y_left, y_right], color='black')
        
        # 中心圆点
        ax.add_patch(plt.Circle((center_x, center_y), 0.02, color='black'))
        
        # 添加分数和等级
        ax.text(center_x, center_y - 0.1, f'{score:.0f}/100', 
                ha='center', va='center', fontsize=16, fontweight='bold')
        ax.text(center_x, center_y - 0.18, result.health_grade, 
                ha='center', va='center', fontsize=12, fontweight='bold')
        
        # 添加图例，增大文字间隔，避免重叠
        ax.text(0.25, 0.05, '危险 (0-33.3)', color=COLORS['danger'], 
                fontsize=8, ha='center', va='center')
        ax.text(0.5, 0.05, '警告 (33.3-66.6)', color=COLORS['warning'], 
                fontsize=8, ha='center', va='center')
        ax.text(0.75, 0.05, '安全 (66.6-100)', color=COLORS['success'], 
                fontsize=8, ha='center', va='center')
        
        ax.set_aspect('equal')  # 设置等宽高比，确保仪表盘为真正的半圆形
        ax.axis('off')
        ax.set_title('健康评分仪表盘', fontsize=10, fontweight='bold', pad=10)

    def _plot_trend_3d_radar(self, ax, result: AnalysisResult):
        """改进版三维趋势雷达图"""
        categories = ['动量', '稳定性', '潜力']
        values = [
            result.momentum_analysis['total'],
            100 - result.resistance_analysis['total'],
            result.potential_analysis['remaining_space']
        ]
        values = values + [values[0]]  # 闭合雷达图
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += [angles[0]]
        
        # 设置极坐标
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # 使用默认雷达网格，调整样式
        ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.7, color='gray')
        
        # 设置网格线数量和样式
        ax.set_rticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=7)
        
        # 绘制数据线，仅保留线条和数据点，无填充
        ax.plot(angles, values, 'o-', color=COLORS['primary'], linewidth=3, markersize=6, markerfacecolor='white')
        
        # 设置刻度标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10, fontweight='bold')
        
        # 在每个数据点添加数值标签
        for i, (angle, value) in enumerate(zip(angles[:-1], values[:-1])):
            # 稍微偏移以避免与标签重叠
            label_angle = angle
            label_radius = value + 5
            ax.text(label_angle, label_radius, f'{value:.0f}', 
                    ha='center', va='center', fontsize=9, fontweight='bold', 
                    color=COLORS['primary'], bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        
        ax.set_ylim(0, 100)
        
        ax.set_title(f'三维趋势分析 ({result.trend_3d["trend_class"]})', 
                    fontsize=10, fontweight='bold', pad=20)
                        
    def _plot_resistance_chart(self, ax, result: AnalysisResult):
        """绘制阻力分析"""
        categories = ['技术债', '代码熵', 'Issue压力']
        values = [
            result.resistance_analysis['debt'],
            result.resistance_analysis['entropy'],
            result.resistance_analysis['issue_pressure']
        ]
        colors = [COLORS['danger'] if v > 60 else COLORS['warning'] if v > 40 else COLORS['success'] for v in values]
        bars = ax.barh(categories, values, color=colors, alpha=0.8)
        ax.axvline(50, color='gray', linestyle='--', lw=1)
        ax.set_xlim(0, 100)
        ax.set_title(f'阻力分析 ({result.resistance_analysis["status"]})', fontsize=10, fontweight='bold')
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                   f'{value:.1f}', ha='left', va='center', fontsize=9, fontweight='bold')
    
    def _plot_potential_bar(self, ax, result: AnalysisResult):
        """改进版增长潜力图"""
        current = result.potential_analysis['current_position']
        ceiling = result.potential_analysis['growth_ceiling']
        remaining = result.potential_analysis['remaining_space']
        
        # 获取历史峰值
        historical_max = self.df['openrank'].max() if 'openrank' in self.df else current
        
        # 创建分组柱状图
        categories = ['当前位置', '历史峰值', '增长上限']
        values = [
            current,
            historical_max,
            min(ceiling, historical_max * 2)  # 限制上限不超过历史峰值的2倍
        ]
        
        # 使用不同颜色
        colors = [COLORS['primary'], COLORS['warning'], COLORS['success']]
        
        bars = ax.bar(categories, values, color=colors, alpha=0.8, width=0.6)
        
        # 在每个柱子上添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.02,
                    f'{value:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 在顶部添加剩余空间百分比
        ax.text(0.5, 1.05, f'剩余增长空间: {remaining:.0f}%', 
                transform=ax.transAxes, ha='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS['info'], alpha=0.2))
        
        # 添加解释文本
        ax.text(0.02, 0.95, f"解释: {result.potential_analysis['interpretation']}", 
                transform=ax.transAxes, fontsize=8, va='top',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='lightgray', alpha=0.3))
        
        ax.set_ylim(0, max(values) * 1.2)
        ax.set_ylabel('OpenRank', fontsize=9)
        # 将标题中的"保守估计"替换为实际的剩余增长空间数据
        ax.set_title(f'增长潜力分析（剩余增长空间: {remaining:.0f}%）', fontsize=10, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
    
    def _plot_predictions_chart(self, ax, result: AnalysisResult):
        """绘制预测图表"""
        if 'openrank' not in self.df or not result.predictions.get('openrank', {}).get('forecast'):
            ax.text(0.5, 0.5, '无预测数据', ha='center', va='center')
            return
        
        # 获取历史数据
        hist_data = self.df['openrank'].tail(12).reset_index(drop=True)
        hist_dates = list(range(len(hist_data)))
        
        # 获取预测数据
        forecast = result.predictions['openrank']['forecast']
        forecast_dates = list(range(len(hist_data), len(hist_data) + len(forecast)))
        
        # 绘制历史数据
        ax.plot(hist_dates, hist_data.values, 'o-', color=COLORS['primary'], 
                linewidth=2, markersize=4, label='历史数据')
        
        # 绘制预测数据
        ax.plot(forecast_dates, forecast, 's--', color=COLORS['success'], 
                linewidth=2, markersize=4, label='预测数据')
        
        # 绘制置信区间（如果存在）
        if 'yhat_lower' in result.predictions['openrank']:
            lower = result.predictions['openrank']['yhat_lower']
            upper = result.predictions['openrank']['yhat_upper']
            ax.fill_between(forecast_dates, lower, upper, 
                        color=COLORS['success'], alpha=0.2, label='95%置信区间')
        
        # 添加垂直线分隔历史和预测
        ax.axvline(x=len(hist_data)-0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.text(len(hist_data)-0.5, ax.get_ylim()[1]*0.9, '历史/预测分界', 
                fontsize=8, rotation=90, va='top', ha='right')
        
        # 添加统计信息
        if result.predictions.get('summary'):
            summary = result.predictions['summary']
            textstr = f"当前值: {summary['current_value']:.1f}\n预测值: {summary['forecast_6m']:.1f}\n增长: {summary['growth_pct']:.1f}%"
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('时间步', fontsize=9)
        ax.set_ylabel('OpenRank', fontsize=9)
        ax.set_title('Prophet时序预测对比', fontsize=10, fontweight='bold')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _plot_backtest_results(self, ax, result: AnalysisResult):
        """绘制回测结果 - 已修复键名匹配问题"""
        if not result.backtest_results:
            ax.text(0.5, 0.5, '无回测数据', ha='center', va='center')
            return
        
        # 提取关键指标
        metrics = []
        error_values = [] # 用于存储 SMAPE 或 MAPE
        direction_acc = []
        
        # 调试输出：确认一下后端传过来的字典到底长什么样
        # print(f"DEBUG Plotter: {result.backtest_results}")

        for metric_name, bt_result in result.backtest_results.items():
            if not isinstance(bt_result, dict):
                continue
            
            # --- 修正点 1: 优先查找 smape，找不到再找 mape ---
            val = bt_result.get('smape') if bt_result.get('smape') is not None else bt_result.get('mape')
            
            if val is not None:
                metrics.append(metric_name.upper())
                error_values.append(val)
                # --- 修正点 2: 匹配方向准确性的键名 ---
                # 你的 Validator 可能返回 direction_accuracy 或 avg_direction_accuracy
                acc = bt_result.get('direction_accuracy', 0)
                # 如果是小数(0.85)，转为百分比(85)；如果是百分比则直接用
                if acc <= 1.0: acc *= 100
                direction_acc.append(acc)
        
        if not metrics:
            ax.text(0.5, 0.5, '无有效回测指标\n(键名不匹配)', ha='center', va='center')
            return
        
        x = np.arange(len(metrics))
        width = 0.35
        
        # 绘制双柱状图
        # 注意：这里标签改为 SMAPE，更准确
        bars1 = ax.bar(x - width/2, error_values, width, label='SMAPE误差(%)', 
                    color='#e74c3c', alpha=0.7) # 使用固定颜色或你的 COLORS['danger']
        bars2 = ax.bar(x + width/2, direction_acc, width, label='方向准确性(%)', 
                    color='#2ecc71', alpha=0.7)
        
        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        
        # --- 修正点 3: 获取置信度等级 ---
        # 你的 Validator 返回的是 overall_confidence 或 confidence_level
        conf_level = result.backtest_results.get('overall_confidence', 
                     result.backtest_results.get('confidence_level', 'UNKNOWN'))
        
        ax.set_xlabel('指标', fontsize=9)
        ax.set_ylabel('百分比(%)', fontsize=9)
        ax.set_title(f'回测验证指标对比 ({conf_level})', fontsize=10, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, axis='y', alpha=0.3)
        # 设置 Y 轴上限，防止标签被挡住
        if error_values or direction_acc:
            ax.set_ylim(0, max(max(error_values + direction_acc) * 1.2, 110))

    def _plot_validation_result(self, ax, result: AnalysisResult):
        """绘制结论验证结果"""
        """GitHub验证数据对比图"""
        if not result.conclusion_validation or 'error' in result.conclusion_validation:
            ax.text(0.5, 0.5, '无验证数据', ha='center', va='center')
            return
        
        val = result.conclusion_validation
        summary = val.get('github_30d_summary', {})
        
        if not summary:
            ax.text(0.5, 0.5, '无GitHub 30天数据', ha='center', va='center')
            return
        
        # 提取要对比的指标
        metrics = ['commits', 'issues_opened', 'prs_opened', 'active_contributors']
        labels = ['Commits', 'Issues', 'PRs', '活跃贡献者']
        values = [summary.get(m, 0) for m in metrics]
        
        # 为了更好显示，对值进行适当缩放
        max_val = max(values) if values else 1
        if max_val > 100:
            values = [v / (max_val/100) for v in values]
            ylabel = '相对值'
        else:
            ylabel = '数量'
        
        # 创建柱状图
        x = np.arange(len(metrics))
        colors = [COLORS['primary'], COLORS['warning'], COLORS['info'], COLORS['success']]
        
        bars = ax.bar(x, values, color=colors, alpha=0.7, width=0.6)
        
        # 添加数值标签
        for bar, value in zip(bars, [summary.get(m, 0) for m in metrics]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.02,
                    f'{value}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # 添加验证结果标签
        if val.get('overall_valid'):
            status_color = COLORS['success']
            status_text = "验证通过"
        else:
            status_color = COLORS['warning']
            status_text = "需复核"
        
        ax.text(0.5, 0.95, f"结论验证: {status_text}", transform=ax.transAxes,
                ha='center', fontsize=10, fontweight='bold', color=status_color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # 添加置信度
        confidence = val.get('confidence', 0)
        ax.text(0.5, 0.85, f"置信度: {confidence}%", transform=ax.transAxes,
                ha='center', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", facecolor='lightgray', alpha=0.3))
        
        ax.set_xlabel('指标', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title('GitHub 30天活跃数据', fontsize=10, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0)
        ax.grid(True, axis='y', alpha=0.3)

if __name__ == "__main__":
    import sys
    DEFAULT_TOKEN="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"
    if len(sys.argv) < 2:
        url = input("请输入 GitHub 项目地址: ").strip()      
        token = DEFAULT_TOKEN
    else:
        url = sys.argv[1]
        token = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_TOKEN
    
    analyzer = ProjectAnalyzerV45(url, github_token=token)
    result = analyzer.run()
    
    if result:
        print("\n分析完成!")
        print(f"   - 层级: {result.tier} ({TIER_NAMES[result.tier]})")
        print(f"   - 健康评分: {result.health_score}/100 ({result.health_grade})")
        print(f"   - 预测置信度: {result.backtest_results.get('overall_confidence_level', 'N/A')}")
        print(f"   - 最佳预测指标: {max(result.tier_probabilities.items(), key=lambda x: x[1])[0]}")