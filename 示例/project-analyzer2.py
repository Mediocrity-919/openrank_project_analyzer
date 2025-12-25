"""
GitHub 项目深度分析器 v6.0 - 数据流重构版
==========================================
核心重构：
1. OpenDigger 只用于趋势分析（月增量）
2. GitHub API 用于锚定现实（绝对快照）
3. Prophet 只预测趋势方向，不预测绝对值
4. 分离结构、时间、活跃状态，消除逻辑冲突
"""

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import re
import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from scipy.stats import pearsonr, linregress
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import json
from datetime import datetime, timedelta
import itertools
from collections import defaultdict

# ============== 显示设置 ==============
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8

# ============== 颜色主题 ==============
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

# ============== 层级基准数据 ==============
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

# ============== 数据类定义 ==============
@dataclass
class AnalysisResult:
    """分析结果"""
    project_name: str
    # 三层状态分离
    structural_tier: str  # GMM分类的结构层级
    temporal_state: str   # 时间趋势状态
    activity_state: str   # 活跃状态
    # 概率和置信度
    tier_probabilities: Dict[str, float]
    tier_confidence: float
    # 健康评分
    health_score: float
    health_grade: str
    dimension_scores: Dict[str, float]
    # 趋势分析
    trend_analysis: Dict
    # 风险评估
    risk_analysis: Dict
    # 特殊分析
    bus_factor_2: Dict
    etd_analysis: Dict
    dark_horse_analysis: Dict
    change_points: List[Dict]
    # 数据验证
    github_comparison: Dict
    conclusion_validation: Dict
    # 预测结果（只趋势）
    trend_predictions: Dict
    backtest_results: Dict
    # 建议
    recommendations: List[str]
    detailed_report: str


# ============== 核心数据拆分器 ==============
class DataReconciliation:
    """OpenDigger 与 GitHub 数据协调器"""
    
    @staticmethod
    def split_od_trend_and_gh_snapshot(od_df: pd.DataFrame, gh_info: Dict, col: str) -> Dict:
        """
        OpenDigger：趋势（月度变化）
        GitHub API：现实锚点（当前快照）
        
        返回值：
        - monthly：月度变化
        - cumulative：累积到当前（根据OpenDigger）
        - github_snapshot：GitHub当前值
        - reconciliation：两者一致性评估
        """
        if col not in od_df.columns:
            return {
                "monthly": pd.Series([]),
                "cumulative": pd.Series([]),
                "github_snapshot": gh_info.get(col, 0),
                "reconciliation": "数据缺失"
            }
        
        # OpenDigger数据：月度变化
        monthly = od_df[col].dropna().astype(float)
        
        # 累积变化（从OpenDigger看的总变化）
        cumulative = monthly.cumsum()
        
        # GitHub快照
        github_value = gh_info.get(col, None)
        
        # 协调评估
        if len(cumulative) > 0:
            od_current = cumulative.iloc[-1]
            reconciliation = {
                "od_current": float(od_current),
                "gh_current": float(github_value) if github_value else 0,
                "diff_pct": abs(od_current - (github_value or 0)) / ((github_value or 0) + 1) * 100
            }
        else:
            reconciliation = {"od_current": 0, "gh_current": 0, "diff_pct": 100}
        
        return {
            "monthly": monthly,
            "cumulative": cumulative,
            "github_snapshot": github_value,
            "reconciliation": reconciliation
        }
    
    @staticmethod
    def get_structural_metrics(od_df: pd.DataFrame, gh_info: Dict) -> Dict:
        """获取用于结构层级的指标（基于GitHub快照）"""
        # 优先使用GitHub快照，没有则用OpenDigger累积值
        return {
            'avg_openrank': od_df['openrank'].mean() if 'openrank' in od_df.columns else 0,
            'total_stars': gh_info.get('stars', 0) or od_df['stars'].sum() if 'stars' in od_df.columns else 0,
            'max_participants': od_df['participants'].max() if 'participants' in od_df.columns else 0
        }


# ============== GMM概率化分层分类器 ==============
class GMMTierClassifier:
    """高斯混合模型分层分类器"""
    
    def __init__(self, n_components=4):
        self.n_components = n_components
        self.gmm = None
        self.scaler = StandardScaler()
        self.tier_labels = ['GIANT', 'MATURE', 'GROWING', 'EMERGING']
        
    def _generate_synthetic_data(self) -> np.ndarray:
        """生成合成基准数据用于训练GMM"""
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
        
        # 处理零值
        feature = np.maximum(feature, [0.1, 10, 1])
        
        scaled_feature = self.scaler.transform(feature)
        probabilities = self.gmm.predict_proba(scaled_feature)[0]
        
        # 计算每个组件的中心点
        centers = self.gmm.means_
        centers_original = centers * self.scaler.scale_ + self.scaler.mean_
        openrank_centers = centers_original[:, 0]
        
        # 按openrank从大到小排序
        sorted_indices = np.argsort(-openrank_centers)
        
        # 分配标签
        tier_probabilities = {}
        for idx, tier in enumerate(self.tier_labels):
            if idx < len(sorted_indices):
                comp_idx = sorted_indices[idx]
                tier_probabilities[tier] = probabilities[comp_idx]
            else:
                tier_probabilities[tier] = 0.0
        
        # 确定主要层级
        best_tier = max(tier_probabilities, key=tier_probabilities.get)
        confidence = tier_probabilities[best_tier]
        
        return best_tier, tier_probabilities, confidence


# ============== 真实 Prophet 预测器 ==============
class ProphetTrendPredictor:
    """Prophet 趋势预测器（只预测方向）"""
    
    def __init__(self):
        try:
            from prophet import Prophet
            self.Prophet = Prophet
            self.available = True
        except ImportError:
            print("❌ 未安装 Prophet 库，使用简化版趋势预测")
            self.Prophet = None
            self.available = False
    
    def prophet_forecast_monthly_trend(self, series: pd.Series, periods: int = 6) -> Dict:
        """使用真实 Prophet 预测月度趋势（只方向）"""
        if not self.available or len(series) < 12:
            # 回退到简化预测
            return self._simple_trend_forecast(series, periods)
        
        try:
            df = series.reset_index()
            df.columns = ['ds', 'y']
            df['ds'] = pd.to_datetime(df['ds'])
            
            model = self.Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                changepoint_prior_scale=0.05  # 保守
            )
            model.fit(df)
            
            future = model.make_future_dataframe(periods=periods, freq='MS')
            forecast = model.predict(future)
            
            # 提取未来预测
            future_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
            
            # 趋势方向分析
            current = series.iloc[-1]
            future_values = future_forecast['yhat'].values
            future_avg = np.mean(future_values)
            
            direction = "上升" if future_avg > current else "下降" if future_avg < current else "平稳"
            direction_confidence = min(0.9, np.mean((future_forecast['yhat_upper'] - future_forecast['yhat_lower']) / 
                                                    (future_forecast['yhat'].abs() + 0.1)))
            
            return {
                'forecast': future_values.tolist(),
                'yhat_lower': future_forecast['yhat_lower'].values.tolist(),
                'yhat_upper': future_forecast['yhat_upper'].values.tolist(),
                'direction': direction,
                'direction_confidence': round(direction_confidence, 3),
                'current_value': float(current),
                'future_avg': float(future_avg),
                'is_prophet': True
            }
        except Exception as e:
            print(f"Prophet 预测失败: {e}")
            return self._simple_trend_forecast(series, periods)
    
    def _simple_trend_forecast(self, series: pd.Series, periods: int = 6) -> Dict:
        """简化趋势预测（当 Prophet 不可用时）"""
        if len(series) < 3:
            return {'error': '数据不足'}
        
        values = series.values
        # 计算最近3个月的趋势
        recent = values[-3:]
        x = np.arange(len(recent))
        slope, intercept = np.polyfit(x, recent, 1)
        
        # 未来预测
        future = [max(0, intercept + slope * i) for i in range(periods)]
        
        # 方向判断
        current = values[-1]
        future_avg = np.mean(future)
        direction = "上升" if future_avg > current * 1.05 else "下降" if future_avg < current * 0.95 else "平稳"
        
        # 简单置信度
        volatility = np.std(recent) / (np.mean(recent) + 0.1)
        confidence = max(0.1, 1 - volatility)
        
        return {
            'forecast': future,
            'direction': direction,
            'direction_confidence': round(confidence, 3),
            'current_value': float(current),
            'future_avg': float(future_avg),
            'is_prophet': False
        }


# ============== 时间状态分析器 ==============
class TemporalStateAnalyzer:
    """分析项目时间趋势状态"""
    
    def analyze(self, od_df: pd.DataFrame, metric: str = 'openrank') -> Dict:
        """分析时间趋势状态"""
        if metric not in od_df.columns or len(od_df) < 6:
            return {'state': 'INSUFFICIENT_DATA', 'confidence': 0, 'reason': '数据不足'}
        
        series = od_df[metric]
        
        # 1. 短期趋势（最近3个月）
        if len(series) >= 3:
            recent = series.tail(3).values
            short_term_slope = self._calculate_slope(recent)
        else:
            short_term_slope = 0
        
        # 2. 中期趋势（最近6个月）
        if len(series) >= 6:
            mid_term = series.tail(6).values
            mid_term_slope = self._calculate_slope(mid_term)
        else:
            mid_term_slope = 0
        
        # 3. 长期趋势（全部数据）
        long_term_slope = self._calculate_slope(series.values)
        
        # 4. 趋势稳定性
        volatility = series.tail(12).std() / (series.tail(12).mean() + 0.1) if len(series) >= 12 else 1
        
        # 状态判断
        if len(series) < 6:
            return {'state': 'INSUFFICIENT_DATA', 'confidence': 0, 'reason': '数据不足'}
        
        # 使用加权趋势：短期权重0.5，中期0.3，长期0.2
        weighted_slope = short_term_slope * 0.5 + mid_term_slope * 0.3 + long_term_slope * 0.2
        
        # 状态分类
        if weighted_slope > 0.1:
            state = 'GROWING'
            reason = f'加权趋势斜率: {weighted_slope:.3f}'
        elif weighted_slope < -0.1:
            state = 'DECLINING'
            reason = f'加权趋势斜率: {weighted_slope:.3f}'
        else:
            state = 'STABLE'
            reason = f'加权趋势斜率: {weighted_slope:.3f}'
        
        # 置信度计算
        if volatility < 0.2:
            confidence = 0.9
        elif volatility < 0.5:
            confidence = 0.7
        else:
            confidence = 0.5
        
        return {
            'state': state,
            'confidence': round(confidence, 2),
            'reason': reason,
            'short_term_slope': round(short_term_slope, 4),
            'mid_term_slope': round(mid_term_slope, 4),
            'long_term_slope': round(long_term_slope, 4),
            'volatility': round(volatility, 3)
        }
    
    def _calculate_slope(self, values: np.ndarray) -> float:
        """计算趋势斜率"""
        if len(values) < 2:
            return 0
        
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        
        # 标准化斜率（除以均值）
        mean_val = np.mean(values)
        if mean_val > 0:
            normalized_slope = slope / mean_val
        else:
            normalized_slope = slope
        
        return normalized_slope


# ============== 活跃状态分析器 ==============
class ActivityStateAnalyzer:
    """分析项目活跃状态"""
    
    def analyze(self, od_df: pd.DataFrame, gh_recent: Dict = None) -> Dict:
        """分析活跃状态"""
        # 1. 基于OpenDigger的活跃度
        od_activity = self._analyze_od_activity(od_df)
        
        # 2. 基于GitHub最近30天活跃度（如果有）
        gh_activity = self._analyze_gh_activity(gh_recent) if gh_recent else None
        
        # 3. 综合判断
        if gh_activity:
            # 两者结合
            if od_activity['state'] == 'THRIVING' and gh_activity['state'] == 'ACTIVE':
                state = 'THRIVING'
            elif od_activity['state'] == 'DORMANT' and gh_activity['state'] == 'INACTIVE':
                state = 'ZOMBIE'
            elif od_activity['state'] == 'STABLE' or gh_activity['state'] == 'ACTIVE':
                state = 'ACTIVE'
            else:
                state = 'DORMANT'
        else:
            state = od_activity['state']
        
        return {
            'state': state,
            'od_analysis': od_activity,
            'gh_analysis': gh_activity
        }
    
    def _analyze_od_activity(self, od_df: pd.DataFrame) -> Dict:
        """分析OpenDigger活跃度"""
        if 'activity' not in od_df.columns or len(od_df) < 3:
            return {'state': 'UNKNOWN', 'reason': '数据不足'}
        
        activity = od_df['activity']
        recent = activity.tail(3)
        avg_recent = recent.mean()
        historical_avg = activity.mean()
        
        if avg_recent > historical_avg * 1.2:
            state = 'THRIVING'
            reason = f'近期活跃度({avg_recent:.1f})高于历史({historical_avg:.1f})'
        elif avg_recent > historical_avg * 0.8:
            state = 'STABLE'
            reason = f'近期活跃度({avg_recent:.1f})接近历史({historical_avg:.1f})'
        elif avg_recent > historical_avg * 0.3:
            state = 'DORMANT'
            reason = f'近期活跃度({avg_recent:.1f})低于历史({historical_avg:.1f})'
        else:
            state = 'ZOMBIE'
            reason = f'近期活跃度({avg_recent:.1f})远低于历史({historical_avg:.1f})'
        
        return {'state': state, 'reason': reason, 'avg_recent': float(avg_recent), 'avg_historical': float(historical_avg)}
    
    def _analyze_gh_activity(self, gh_recent: Dict) -> Dict:
        """分析GitHub最近30天活跃度"""
        if 'error' in gh_recent:
            return None
        
        commits = gh_recent.get('commits', 0)
        prs = gh_recent.get('prs_opened', 0)
        issues = gh_recent.get('issues_opened', 0)
        
        total_activity = commits + prs * 2 + issues  # PR权重更高
        
        if total_activity >= 20:
            state = 'VERY_ACTIVE'
        elif total_activity >= 10:
            state = 'ACTIVE'
        elif total_activity >= 3:
            state = 'LOW_ACTIVITY'
        else:
            state = 'INACTIVE'
        
        return {
            'state': state,
            'total_activity': total_activity,
            'commits': commits,
            'prs': prs,
            'issues': issues
        }


# ============== 方向性回测验证器 ==============
class DirectionalBacktestValidator:
    """方向性回测验证器（只验证趋势方向）"""
    
    @staticmethod
    def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算方向准确性"""
        if len(y_true) < 2 or len(y_pred) < 2:
            return 0.0
        
        # 计算真实方向
        true_dir = np.sign(np.diff(y_true))
        
        # 计算预测方向
        pred_dir = np.sign(np.diff(y_pred))
        
        # 确保长度一致
        min_len = min(len(true_dir), len(pred_dir))
        true_dir = true_dir[:min_len]
        pred_dir = pred_dir[:min_len]
        
        # 计算匹配率
        matches = np.sum(true_dir == pred_dir)
        return matches / min_len
    
    def validate(self, data: pd.Series, test_ratio: float = 0.3) -> Dict:
        """回测验证"""
        if len(data) < 12:
            return {'error': '数据不足，需要至少12个月数据'}
        
        n_test = max(3, int(len(data) * test_ratio))
        n_train = len(data) - n_test
        
        if n_train < 6:
            return {'error': '训练数据不足'}
        
        # 划分训练集和测试集
        train_data = data.iloc[:n_train]
        test_data = data.iloc[n_train:]
        
        # 使用简化预测器进行预测
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        
        # 准备训练数据
        X_train = np.arange(len(train_data)).reshape(-1, 1)
        y_train = train_data.values
        
        model.fit(X_train, y_train)
        
        # 预测测试集
        X_test = np.arange(len(train_data), len(train_data) + len(test_data)).reshape(-1, 1)
        y_pred = model.predict(X_test)
        
        # 计算方向准确性
        dir_acc = self.directional_accuracy(test_data.values, y_pred)
        
        # 计算置信度
        if dir_acc > 0.8:
            confidence = 'HIGH'
        elif dir_acc > 0.6:
            confidence = 'MEDIUM'
        elif dir_acc > 0.5:
            confidence = 'LOW'
        else:
            confidence = 'VERY_LOW'
        
        return {
            'train_samples': n_train,
            'test_samples': n_test,
            'direction_accuracy': round(dir_acc, 3),
            'confidence': confidence,
            'actual_values': [float(v) for v in test_data.values],
            'predicted_values': [float(v) for v in y_pred]
        }


# ============== 新的趋势分析器 ==============
class TrendAnalyzerV6:
    """新版趋势分析器（分离动量、阻力、潜力）"""
    
    def analyze(self, od_df: pd.DataFrame, tier: str, gh_info: Dict = None) -> Dict:
        """分析趋势"""
        # 动量分析（基于月度变化）
        momentum = self._analyze_momentum(od_df)
        
        # 阻力分析（基于问题积压等）
        resistance = self._analyze_resistance(od_df)
        
        # 潜力分析（基于GitHub快照与基准对比）
        potential = self._analyze_potential(od_df, tier, gh_info)
        
        # 综合趋势评分
        trend_score = momentum['score'] * 0.4 - resistance['score'] * 0.3 + potential['score'] * 0.3
        
        if trend_score >= 60:
            trend_class, trend_desc = 'STRONG_UP', '强烈上升趋势'
        elif trend_score >= 30:
            trend_class, trend_desc = 'MODERATE_UP', '温和上升趋势'
        elif trend_score >= 0:
            trend_class, trend_desc = 'STABLE', '趋势稳定'
        elif trend_score >= -30:
            trend_class, trend_desc = 'MODERATE_DOWN', '温和下降趋势'
        else:
            trend_class, trend_desc = 'STRONG_DOWN', '强烈下降趋势'
        
        return {
            'trend_score': round(trend_score, 1),
            'trend_class': trend_class,
            'trend_description': trend_desc,
            'momentum': momentum,
            'resistance': resistance,
            'potential': potential
        }
    
    def _analyze_momentum(self, od_df: pd.DataFrame) -> Dict:
        """动量分析（基于月度变化趋势）"""
        # 参与度动量
        if 'participants' in od_df.columns and len(od_df) >= 6:
            participants_trend = linregress(range(6), od_df['participants'].tail(6).values).slope
            participants_momentum = min(100, max(0, 50 + participants_trend * 50))
        else:
            participants_momentum = 50
        
        # 活跃度动量
        if 'activity' in od_df.columns and len(od_df) >= 6:
            activity_trend = linregress(range(6), od_df['activity'].tail(6).values).slope
            activity_momentum = min(100, max(0, 50 + activity_trend * 30))
        else:
            activity_momentum = 50
        
        # 贡献者动量
        if 'new_contributors' in od_df.columns:
            recent_contributors = od_df['new_contributors'].tail(6).mean()
            historical_contributors = od_df['new_contributors'].mean()
            if historical_contributors > 0:
                contributor_ratio = recent_contributors / historical_contributors
                contributor_momentum = min(100, max(0, contributor_ratio * 50))
            else:
                contributor_momentum = recent_contributors * 10
        else:
            contributor_momentum = 50
        
        # 综合动量
        total = participants_momentum * 0.3 + activity_momentum * 0.4 + contributor_momentum * 0.3
        
        return {
            'score': round(total, 1),
            'participants_momentum': round(participants_momentum, 1),
            'activity_momentum': round(activity_momentum, 1),
            'contributor_momentum': round(contributor_momentum, 1),
            'description': self._momentum_description(total)
        }
    
    def _analyze_resistance(self, od_df: pd.DataFrame) -> Dict:
        """阻力分析"""
        # Issue积压
        if 'issues_new' in od_df.columns and 'issues_closed' in od_df.columns:
            recent_new = od_df['issues_new'].tail(6).sum()
            recent_closed = od_df['issues_closed'].tail(6).sum()
            if recent_new > 0:
                issue_ratio = recent_closed / recent_new
                issue_resistance = max(0, 100 - issue_ratio * 100)
            else:
                issue_resistance = 0
        else:
            issue_resistance = 50
        
        # PR合并效率
        if 'pr_new' in od_df.columns and 'pr_merged' in od_df.columns:
            recent_pr_new = od_df['pr_new'].tail(6).sum()
            recent_pr_merged = od_df['pr_merged'].tail(6).sum()
            if recent_pr_new > 0:
                pr_ratio = recent_pr_merged / recent_pr_new
                pr_resistance = max(0, 100 - pr_ratio * 100)
            else:
                pr_resistance = 0
        else:
            pr_resistance = 50
        
        # 贡献者流失
        if 'inactive_contributors' in od_df.columns and 'participants' in od_df.columns:
            recent_inactive = od_df['inactive_contributors'].tail(6).mean()
            recent_participants = od_df['participants'].tail(6).mean()
            if recent_participants > 0:
                churn_rate = recent_inactive / recent_participants
                churn_resistance = min(100, churn_rate * 200)
            else:
                churn_resistance = 0
        else:
            churn_resistance = 50
        
        # 综合阻力
        total = issue_resistance * 0.4 + pr_resistance * 0.3 + churn_resistance * 0.3
        
        return {
            'score': round(total, 1),
            'issue_resistance': round(issue_resistance, 1),
            'pr_resistance': round(pr_resistance, 1),
            'churn_resistance': round(churn_resistance, 1),
            'description': self._resistance_description(total)
        }
    
    def _analyze_potential(self, od_df: pd.DataFrame, tier: str, gh_info: Dict) -> Dict:
        """潜力分析（基于当前状态与层级基准的差距）"""
        # 获取当前状态
        if 'openrank' in od_df.columns:
            current_openrank = od_df['openrank'].iloc[-1]
        else:
            current_openrank = 0
        
        # 获取层级基准
        tier_benchmark = TIER_BENCHMARKS.get(tier, TIER_BENCHMARKS['EMERGING'])
        benchmark_openrank = tier_benchmark['openrank']
        
        # 计算与下一层级的差距
        tiers = ['EMERGING', 'GROWING', 'MATURE', 'GIANT']
        current_idx = tiers.index(tier) if tier in tiers else 0
        
        if current_idx < len(tiers) - 1:
            next_tier = tiers[current_idx + 1]
            next_benchmark = TIER_BENCHMARKS[next_tier]['openrank']
            gap_to_next = next_benchmark - current_openrank
            max_gap = next_benchmark - tier_benchmark['openrank']
            
            if max_gap > 0:
                potential_score = (gap_to_next / max_gap) * 100
            else:
                potential_score = 0
        else:
            # 已经是最高层级
            potential_score = 0
        
        # 限制范围
        potential_score = max(0, min(100, potential_score))
        
        return {
            'score': round(potential_score, 1),
            'current_openrank': round(current_openrank, 2),
            'tier_benchmark': benchmark_openrank,
            'description': self._potential_description(potential_score, tier)
        }
    
    def _momentum_description(self, score: float) -> str:
        if score >= 70:
            return '强劲增长动力'
        elif score >= 50:
            return '稳定发展动力'
        elif score >= 30:
            return '动力不足'
        else:
            return '增长停滞'
    
    def _resistance_description(self, score: float) -> str:
        if score >= 70:
            return '阻力较大，需关注'
        elif score >= 50:
            return '中等阻力'
        elif score >= 30:
            return '阻力较小'
        else:
            return '发展顺畅'
    
    def _potential_description(self, score: float, tier: str) -> str:
        if tier == 'GIANT':
            return '已达顶级规模'
        elif score >= 70:
            return f'高增长潜力，有较大空间达到{TIER_NAMES.get(self._get_next_tier(tier), "下一层级")}'
        elif score >= 40:
            return f'中等潜力，逐步向{TIER_NAMES.get(self._get_next_tier(tier), "下一层级")}发展'
        elif score >= 20:
            return f'有限潜力，接近当前层级上限'
        else:
            return f'已接近当前层级天花板'
    
    def _get_next_tier(self, current_tier: str) -> str:
        tiers = ['EMERGING', 'GROWING', 'MATURE', 'GIANT']
        current_idx = tiers.index(current_tier) if current_tier in tiers else 0
        if current_idx < len(tiers) - 1:
            return tiers[current_idx + 1]
        return current_tier


# ============== 新版 AHP 健康评估器 ==============
class AHPHealthEvaluatorV6:
    """新版AHP健康评估器（降噪权重）"""
    
    TIER_WEIGHTS = {
        'GIANT': {
            'momentum': 0.20,    # 降低动量权重
            'stability': 0.40,   # 提高稳定性权重
            'potential': 0.15,
            'safety': 0.25
        },
        'MATURE': {
            'momentum': 0.15,    # 降低动量权重
            'stability': 0.45,   # 提高稳定性权重
            'potential': 0.20,
            'safety': 0.20
        },
        'GROWING': {
            'momentum': 0.20,    # 从0.35降到0.20
            'stability': 0.25,
            'potential': 0.40,   # 从0.30提高到0.40
            'safety': 0.15       # 从0.10提高到0.15
        },
        'EMERGING': {
            'momentum': 0.25,    # 从0.30降到0.25
            'stability': 0.20,
            'potential': 0.45,   # 从0.40提高到0.45
            'safety': 0.10
        }
    }
    
    def calculate_health_score(self, 
                              trend_analysis: Dict,
                              temporal_state: Dict,
                              activity_state: Dict,
                              tier: str) -> Tuple[float, Dict[str, float]]:
        """计算健康分"""
        weights = self.TIER_WEIGHTS.get(tier, self.TIER_WEIGHTS['MATURE'])
        
        # 各维度原始分数
        raw_scores = {
            'momentum': trend_analysis['momentum']['score'],
            'stability': 100 - trend_analysis['resistance']['score'],
            'potential': trend_analysis['potential']['score'],
            'safety': self._calculate_safety_score(temporal_state, activity_state)
        }
        
        # 应用权重
        weighted_sum = 0
        for dim, score in raw_scores.items():
            weighted_sum += score * weights[dim]
        
        # 基础健康分
        base_score = weighted_sum
        
        # 根据时间状态微调
        temporal_factor = self._get_temporal_factor(temporal_state['state'])
        base_score *= temporal_factor
        
        # 根据活跃状态微调
        activity_factor = self._get_activity_factor(activity_state['state'])
        base_score *= activity_factor
        
        # 限制范围
        final_score = max(0, min(100, base_score))
        
        # 各维度贡献
        dimension_contributions = {}
        for dim, score in raw_scores.items():
            dimension_contributions[dim] = round(score * weights[dim] / 100, 3)
        
        return round(final_score, 1), dimension_contributions
    
    def _calculate_safety_score(self, temporal_state: Dict, activity_state: Dict) -> float:
        """计算安全分数"""
        score = 70  # 基础分
        
        # 时间状态调整
        if temporal_state['state'] == 'DECLINING':
            score -= 20
        elif temporal_state['state'] == 'INSUFFICIENT_DATA':
            score -= 10
        
        # 活跃状态调整
        if activity_state['state'] == 'ZOMBIE':
            score -= 30
        elif activity_state['state'] == 'DORMANT':
            score -= 15
        
        return max(0, min(100, score))
    
    def _get_temporal_factor(self, state: str) -> float:
        factors = {
            'GROWING': 1.1,
            'STABLE': 1.0,
            'DECLINING': 0.8,
            'INSUFFICIENT_DATA': 0.9
        }
        return factors.get(state, 1.0)
    
    def _get_activity_factor(self, state: str) -> float:
        factors = {
            'THRIVING': 1.2,
            'ACTIVE': 1.1,
            'STABLE': 1.0,
            'DORMANT': 0.8,
            'ZOMBIE': 0.6
        }
        return factors.get(state, 1.0)


# ============== GitHub API 分析器 ==============
class GitHubAPIAnalyzerV6:
    """新版GitHub API分析器"""
    
    def __init__(self, token: str = None):
        self.token = token
        self.base_url = "https://api.github.com"
        self.headers = {'Authorization': f'token {token}'} if token else {}
    
    def fetch_repo_info(self, org: str, repo: str) -> Optional[Dict]:
        """获取仓库基本信息（用于锚定）"""
        if not self.token:
            return None
        
        try:
            url = f"{self.base_url}/repos/{org}/{repo}"
            res = requests.get(url, headers=self.headers, timeout=15)
            if res.status_code == 200:
                data = res.json()
                return {
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
                    'archived': data.get('archived', False)
                }
            else:
                print(f"GitHub API 请求失败: {res.status_code}")
                return None
        except Exception as e:
            print(f"GitHub API 错误: {e}")
            return None
    
    def fetch_recent_activity(self, org: str, repo: str, days: int = 30) -> Dict:
        """获取最近N天的活跃数据"""
        if not self.token:
            return {'error': '需要 GitHub Token'}
        
        since_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        result = {
            'period_days': days,
            'commits': 0,
            'issues_opened': 0,
            'issues_closed': 0,
            'prs_opened': 0,
            'prs_merged': 0,
            'contributors_active': set()
        }
        
        try:
            # 获取最近提交
            commits_url = f"{self.base_url}/repos/{org}/{repo}/commits?since={since_date}&per_page=100"
            res = requests.get(commits_url, headers=self.headers, timeout=15)
            if res.status_code == 200:
                commits = res.json()
                result['commits'] = len(commits)
                for c in commits:
                    author = c.get('author', {})
                    if author and author.get('login'):
                        result['contributors_active'].add(author['login'])
            
            # 获取最近 Issues
            issues_url = f"{self.base_url}/repos/{org}/{repo}/issues?state=all&since={since_date}&per_page=100"
            res = requests.get(issues_url, headers=self.headers, timeout=15)
            if res.status_code == 200:
                issues = res.json()
                for issue in issues:
                    if 'pull_request' not in issue:
                        created = issue.get('created_at', '')
                        if created >= since_date:
                            result['issues_opened'] += 1
                        if issue.get('state') == 'closed':
                            result['issues_closed'] += 1
            
            # 获取最近 PRs
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
    
    def validate_conclusions(self, temporal_state: Dict, activity_state: Dict, gh_recent: Dict) -> Dict:
        """使用GitHub 30天数据验证结论"""
        if 'error' in gh_recent:
            return {'error': gh_recent['error']}
        
        validation = {
            'overall_valid': True,
            'confidence': 0,
            'validations': [],
            'warnings': []
        }
        
        # 活跃状态验证
        gh_activity = activity_state.get('gh_analysis', {})
        if gh_activity:
            od_state = activity_state['state']
            gh_state = gh_activity.get('state', 'UNKNOWN')
            
            if (od_state == 'THRIVING' and gh_state in ['VERY_ACTIVE', 'ACTIVE']) or \
               (od_state == 'ZOMBIE' and gh_state == 'INACTIVE') or \
               (od_state == 'DORMANT' and gh_state == 'LOW_ACTIVITY'):
                validation['validations'].append({
                    'check': '活跃状态',
                    'result': 'PASS',
                    'detail': f'OpenDigger状态({od_state})与GitHub状态({gh_state})一致'
                })
                validation['confidence'] += 25
            else:
                validation['warnings'].append(f'活跃状态不一致: OD={od_state}, GH={gh_state}')
        
        # 时间趋势验证（简化）
        temporal_state_val = temporal_state['state']
        gh_commits = gh_recent.get('commits', 0)
        
        if temporal_state_val == 'GROWING' and gh_commits >= 5:
            validation['validations'].append({
                'check': '增长趋势',
                'result': 'PASS',
                'detail': f'增长趋势与近期活动(commits={gh_commits})一致'
            })
            validation['confidence'] += 25
        elif temporal_state_val == 'DECLINING' and gh_commits <= 2:
            validation['validations'].append({
                'check': '衰退趋势',
                'result': 'PASS',
                'detail': f'衰退趋势与近期低活动(commits={gh_commits})一致'
            })
            validation['confidence'] += 25
        else:
            validation['validations'].append({
                'check': '趋势验证',
                'result': 'NEUTRAL',
                'detail': f'趋势{temporal_state_val}与近期活动(commits={gh_commits})无明显冲突'
            })
            validation['confidence'] += 15
        
        # 贡献者验证
        gh_contributors = gh_recent.get('contributors_active', 0)
        if gh_contributors >= 3:
            validation['validations'].append({
                'check': '贡献者活跃',
                'result': 'PASS',
                'detail': f'近期有{gh_contributors}名活跃贡献者'
            })
            validation['confidence'] += 25
        
        # PR效率验证
        pr_opened = gh_recent.get('prs_opened', 0)
        pr_merged = gh_recent.get('prs_merged', 0)
        if pr_opened > 0:
            merge_rate = pr_merged / pr_opened
            if merge_rate >= 0.5:
                validation['validations'].append({
                    'check': 'PR效率',
                    'result': 'PASS',
                    'detail': f'PR合并率{merge_rate:.0%}良好'
                })
                validation['confidence'] += 25
            else:
                validation['warnings'].append(f'PR合并率偏低: {merge_rate:.0%}')
        
        # 最终置信度
        validation['confidence'] = min(100, validation['confidence'])
        validation['overall_valid'] = validation['confidence'] >= 50 and len(validation['warnings']) <= 2
        
        return validation


# ============== 其他分析器（保持原样但优化） ==============
class BusFactorCalculatorV6:
    """Bus Factor 2.0"""
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        try:
            if 'participants' not in data.columns:
                return {'effective_bus_factor': 1, 'risk_level': 'UNKNOWN'}
            
            participants = data['participants'].tail(6).mean()
            if participants <= 0:
                return {'effective_bus_factor': 1, 'risk_level': 'CRITICAL'}
            
            n = int(max(1, participants))
            contributions = np.array([1/(i+1) for i in range(n)])
            contributions = contributions / contributions.sum()
            
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
        except Exception as e:
            return {'effective_bus_factor': 1, 'risk_level': 'UNKNOWN'}


class ETDAnalyzerV6:
    """ETD分析器（优化版）"""
    
    def analyze(self, data: pd.DataFrame, activity_state: str, tier: str) -> Dict:
        result = {
            'etd_months': float('inf'),
            'etd_status': 'HEALTHY',
            'is_mature_stable': False,
            'description': '',
            'recommendations': []
        }
        
        if 'activity' not in data.columns or len(data) < 6:
            result['description'] = '数据不足，无法进行寿命预测'
            return result
        
        try:
            activity = data['activity'].tail(12)
            slope, _ = np.polyfit(range(len(activity)), activity.values, 1)
            current = activity.iloc[-1]
            
            # 成熟稳定项目判断
            if tier in ['GIANT', 'MATURE'] and activity_state in ['STABLE', 'ACTIVE']:
                if current > activity.mean() * 0.3 and abs(slope) < current * 0.1:
                    result['etd_status'] = 'STABLE_MATURE'
                    result['is_mature_stable'] = True
                    result['description'] = '项目进入成熟稳定期，低活跃度是正常特征'
                    return result
            
            # 真正衰退判断
            if slope < 0 and current > 0:
                etd = -current / slope
                result['etd_months'] = max(0, etd)
                
                if etd < 6:
                    result['etd_status'] = 'CRITICAL'
                    result['description'] = f'高危：预计{etd:.1f}个月后活跃度归零'
                elif etd < 12:
                    result['etd_status'] = 'WARNING'
                    result['description'] = f'预警：预计{etd:.1f}个月后可能枯竭'
                elif etd < 24:
                    result['etd_status'] = 'CAUTION'
                    result['description'] = f'注意：预计{etd:.1f}个月后可能低迷'
                else:
                    result['etd_status'] = 'HEALTHY'
                    result['description'] = f'健康：ETD > 24个月，暂无风险'
            else:
                result['etd_status'] = 'THRIVING'
                result['description'] = '活跃度稳定或上升，无枯竭风险'
            
            return result
        except Exception as e:
            result['description'] = '分析过程出错'
            return result


# ============== 新版项目分析器 ==============
class ProjectAnalyzerV6:
    """GitHub 项目深度分析器 v6.0 - 数据流重构版"""
    
    CORE_METRICS = [
        "openrank", "activity", "stars", "attention",
        "participants", "new_contributors", "inactive_contributors",
        "bus_factor", "issues_new", "issues_closed", "pr_new", "pr_merged"
    ]
    
    def __init__(self, url: str, github_token: Optional[str] = None):
        self.org, self.repo = self._parse_url(url)
        self.od_df = pd.DataFrame()  # OpenDigger数据（月度变化）
        self.gh_info = {}            # GitHub快照数据
        self.gh_recent = {}          # GitHub近期活动
        
        self.structural_tier = None
        self.tier_probabilities = {}
        self.tier_confidence = 0
        
        self.github_token = github_token or os.getenv('GITHUB_TOKEN')
        
        # 初始化各分析器
        self.data_reconciliation = DataReconciliation()
        self.gmm_classifier = GMMTierClassifier()
        self.temporal_analyzer = TemporalStateAnalyzer()
        self.activity_analyzer = ActivityStateAnalyzer()
        self.trend_analyzer = TrendAnalyzerV6()
        self.ahp_evaluator = AHPHealthEvaluatorV6()
        self.prophet_predictor = ProphetTrendPredictor()
        self.backtest_validator = DirectionalBacktestValidator()
        self.bus_factor_calculator = BusFactorCalculatorV6()
        self.etd_analyzer = ETDAnalyzerV6()
        self.github_analyzer = GitHubAPIAnalyzerV6(self.github_token) if self.github_token else None
    
    def _parse_url(self, url: str) -> Tuple[str, str]:
        match = re.search(r"github\.com/([^/]+)/([^/]+)", url)
        if match:
            return match.group(1), match.group(2).replace(".git", "")
        if "/" in url and "http" not in url:
            parts = url.split('/')
            return parts[0], parts[1]
        raise ValueError("无效的 GitHub URL")
    
    def fetch_data(self) -> bool:
        """获取数据"""
        print(f"\n{'='*60}")
        print(f"  GitHub 项目深度分析器 v6.0 - 数据流重构版")
        print(f"  项目: {self.org}/{self.repo}")
        print(f"{'='*60}\n")
        
        # 1. 获取OpenDigger数据（月度变化）
        print("📊 正在获取 OpenDigger 数据（月度变化）...")
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
            print("❌ 无法获取 OpenDigger 数据")
            return False
        
        self.od_df = pd.DataFrame(raw_data).fillna(0)
        if len(self.od_df) == 0:
            print("❌ OpenDigger 数据为空")
            return False
        
        self.od_df.index = pd.to_datetime(self.od_df.index)
        self.od_df = self.od_df.sort_index()
        print(f"✅ 获取到 {len(self.od_df)} 个月度变化数据")
        
        # 2. 获取GitHub快照数据（用于锚定）
        if self.github_token and self.github_analyzer:
            print("🔗 正在获取 GitHub API 数据（当前快照）...")
            self.gh_info = self.github_analyzer.fetch_repo_info(self.org, self.repo)
            if self.gh_info:
                print("✅ GitHub 快照数据获取成功")
                
                # 获取最近30天活动
                print("📈 正在获取 GitHub 最近30天活动数据...")
                self.gh_recent = self.github_analyzer.fetch_recent_activity(self.org, self.repo, days=30)
                if 'error' not in self.gh_recent:
                    print(f"✅ 获取到 {self.gh_recent.get('commits', 0)} 次提交，{self.gh_recent.get('contributors_active', 0)} 名活跃贡献者")
            else:
                print("⚠️  GitHub API 数据获取失败，继续使用 OpenDigger 数据")
        else:
            print("⚠️  未提供 GitHub Token，跳过 GitHub 数据获取")
        
        # 3. GMM结构层级分类（基于GitHub快照）
        print("🏗️  正在分析项目结构层级...")
        structural_metrics = self.data_reconciliation.get_structural_metrics(self.od_df, self.gh_info)
        self.structural_tier, self.tier_probabilities, self.tier_confidence = self.gmm_classifier.predict_proba(structural_metrics)
        
        print(f"✅ 结构层级分析完成: {self.structural_tier} ({TIER_NAMES[self.structural_tier]})")
        print(f"   层级概率: {self.tier_probabilities}")
        print(f"   置信度: {self.tier_confidence:.0%}")
        
        return True
    
    def analyze(self) -> Optional[AnalysisResult]:
        """执行完整分析"""
        if not self.fetch_data():
            return None
        
        try:
            print("\n" + "="*60)
            print("🧠 开始深度分析...")
            print("="*60)
            
            # 1. 时间趋势状态分析
            print("⏰ 分析时间趋势状态...")
            temporal_state = self.temporal_analyzer.analyze(self.od_df, 'openrank')
            print(f"   时间状态: {temporal_state['state']} (置信度: {temporal_state['confidence']})")
            
            # 2. 活跃状态分析
            print("🔋 分析活跃状态...")
            activity_state = self.activity_analyzer.analyze(self.od_df, self.gh_recent)
            print(f"   活跃状态: {activity_state['state']}")
            
            # 3. 趋势分析
            print("📈 分析综合趋势...")
            trend_analysis = self.trend_analyzer.analyze(self.od_df, self.structural_tier, self.gh_info)
            print(f"   趋势评分: {trend_analysis['trend_score']} ({trend_analysis['trend_description']})")
            
            # 4. 健康评分
            print("❤️  计算健康评分...")
            health_score, dimension_scores = self.ahp_evaluator.calculate_health_score(
                trend_analysis, temporal_state, activity_state, self.structural_tier
            )
            
            # 健康等级
            grades = [(85, 'A+'), (75, 'A'), (65, 'B+'), (55, 'B'), (45, 'C'), (35, 'D'), (0, 'F')]
            health_grade = next(g for t, g in grades if health_score >= t)
            print(f"   健康评分: {health_score}/100 ({health_grade})")
            
            # 5. 趋势预测
            print("🔮 生成趋势预测...")
            trend_predictions = {}
            for metric in ['openrank', 'activity', 'participants']:
                if metric in self.od_df.columns and len(self.od_df[metric]) >= 6:
                    prediction = self.prophet_predictor.prophet_forecast_monthly_trend(
                        self.od_df[metric].dropna(), periods=6
                    )
                    if 'error' not in prediction:
                        trend_predictions[metric] = prediction
            
            # 6. 回测验证
            print("🔍 进行回测验证...")
            backtest_results = {}
            if 'openrank' in self.od_df.columns:
                backtest = self.backtest_validator.validate(self.od_df['openrank'].dropna())
                if 'error' not in backtest:
                    backtest_results['openrank'] = backtest
            
            # 7. 其他分析
            print("⚙️  进行专项分析...")
            bus_factor_2 = self.bus_factor_calculator.calculate(self.od_df)
            etd_analysis = self.etd_analyzer.analyze(self.od_df, activity_state['state'], self.structural_tier)
            
            # 8. GitHub验证
            print("✅ 进行GitHub数据验证...")
            github_comparison = {}
            conclusion_validation = {'error': '未提供GitHub Token，跳过验证'}
            
            if self.github_analyzer and self.gh_info:
                # 数据协调分析
                github_comparison = {
                    'stars': self.data_reconciliation.split_od_trend_and_gh_snapshot(
                        self.od_df, self.gh_info, 'stars'
                    )['reconciliation']
                }
                
                if self.gh_recent and 'error' not in self.gh_recent:
                    conclusion_validation = self.github_analyzer.validate_conclusions(
                        temporal_state, activity_state, self.gh_recent
                    )
            
            # 9. 风险分析
            risk_analysis = self._analyze_risk(temporal_state, activity_state, trend_analysis)
            
            # 10. 黑马分析
            dark_horse_analysis = self._analyze_dark_horse(trend_analysis, bus_factor_2)
            
            # 11. 变点检测
            change_points = []
            if 'openrank' in self.od_df.columns:
                change_points = self._detect_change_points(self.od_df['openrank'])
            
            # 12. 生成建议
            recommendations = self._generate_recommendations(
                self.structural_tier, temporal_state, activity_state,
                trend_analysis, risk_analysis, bus_factor_2, etd_analysis
            )
            
            # 构建结果
            result = AnalysisResult(
                project_name=f"{self.org}/{self.repo}",
                structural_tier=self.structural_tier,
                temporal_state=temporal_state['state'],
                activity_state=activity_state['state'],
                tier_probabilities=self.tier_probabilities,
                tier_confidence=self.tier_confidence,
                health_score=health_score,
                health_grade=health_grade,
                dimension_scores=dimension_scores,
                trend_analysis=trend_analysis,
                risk_analysis=risk_analysis,
                bus_factor_2=bus_factor_2,
                etd_analysis=etd_analysis,
                dark_horse_analysis=dark_horse_analysis,
                change_points=change_points,
                github_comparison=github_comparison,
                conclusion_validation=conclusion_validation,
                trend_predictions=trend_predictions,
                backtest_results=backtest_results,
                recommendations=recommendations,
                detailed_report=""
            )
            
            # 生成报告
            result.detailed_report = self.generate_report(result)
            
            # 输出结果
            print("\n" + "="*60)
            print("🎉 分析完成!")
            print("="*60)
            print(result.detailed_report)
            
            # 保存报告
            self._save_reports(result)
            
            # 生成图表
            self.plot_dashboard(result)
            
            return result
            
        except Exception as e:
            print(f"❌ 分析过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _analyze_risk(self, temporal_state: Dict, activity_state: Dict, trend_analysis: Dict) -> Dict:
        """风险分析"""
        risk_score = 0
        alerts = []
        
        # 时间状态风险
        if temporal_state['state'] == 'DECLINING':
            risk_score += 30
            alerts.append('时间趋势显示衰退')
        
        # 活跃状态风险
        if activity_state['state'] == 'ZOMBIE':
            risk_score += 40
            alerts.append('项目处于僵尸状态')
        elif activity_state['state'] == 'DORMANT':
            risk_score += 20
        
        # 趋势风险
        if trend_analysis['trend_class'] == 'STRONG_DOWN':
            risk_score += 30
            alerts.append('强烈下降趋势')
        elif trend_analysis['trend_class'] == 'MODERATE_DOWN':
            risk_score += 15
        
        # 阻力风险
        if trend_analysis['resistance']['score'] >= 70:
            risk_score += 25
            alerts.append('技术阻力较高')
        
        # 风险等级
        if risk_score >= 60:
            level = 'CRITICAL'
        elif risk_score >= 40:
            level = 'HIGH'
        elif risk_score >= 20:
            level = 'MEDIUM'
        else:
            level = 'LOW'
        
        return {'score': risk_score, 'level': level, 'alerts': alerts}
    
    def _analyze_dark_horse(self, trend_analysis: Dict, bus_factor: Dict) -> Dict:
        """黑马分析"""
        if self.structural_tier in ['GIANT', 'MATURE']:
            return {'is_dark_horse': False, 'score': 0, 'reasons': ['已超出黑马范畴']}
        
        score = 0
        reasons = []
        
        # 强劲动量
        if trend_analysis['momentum']['score'] >= 70:
            score += 30
            reasons.append('强劲增长动量')
        
        # 高潜力
        if trend_analysis['potential']['score'] >= 60:
            score += 25
            reasons.append('高增长潜力')
        
        # 低阻力
        if trend_analysis['resistance']['score'] <= 30:
            score += 20
            reasons.append('发展阻力小')
        
        # 健康的贡献者生态
        if bus_factor.get('risk_level') in ['LOW', 'MEDIUM']:
            score += 15
            reasons.append('贡献者生态健康')
        
        return {
            'is_dark_horse': score >= 55,
            'score': min(100, max(0, score)),
            'reasons': reasons
        }
    
    def _detect_change_points(self, series: pd.Series) -> List[Dict]:
        """变点检测"""
        if len(series) < 12:
            return []
        
        results = []
        window = 6
        
        for i in range(window, len(series) - window):
            before = series.iloc[i-window:i].mean()
            after = series.iloc[i:i+window].mean()
            change_rate = (after - before) / (before + 0.1)
            
            if abs(change_rate) > 0.3:
                if change_rate > 0.3:
                    cp_type, desc = 'ACCELERATION', '进入快速增长期'
                else:
                    cp_type, desc = 'DECELERATION', '活跃度显著下降'
                
                results.append({
                    'index': i,
                    'date': str(series.index[i])[:7],
                    'type': cp_type,
                    'change_rate': round(change_rate, 3),
                    'description': desc
                })
        
        return results[:3]
    
    def _generate_recommendations(self, structural_tier: str, temporal_state: Dict,
                                  activity_state: Dict, trend_analysis: Dict,
                                  risk_analysis: Dict, bus_factor: Dict,
                                  etd_analysis: Dict) -> List[str]:
        """生成建议"""
        recs = []
        
        # 风险相关建议
        if risk_analysis['level'] in ['CRITICAL', 'HIGH']:
            recs.append(f"⚠️ 项目风险较高({risk_analysis['level']})，建议优先处理：{', '.join(risk_analysis['alerts'][:2])}")
        
        # 时间趋势建议
        if temporal_state['state'] == 'DECLINING':
            recs.append("⏬ 时间趋势显示衰退，建议分析原因并采取激活措施")
        
        # 活跃状态建议
        if activity_state['state'] == 'ZOMBIE':
            recs.append("💀 项目处于僵尸状态，建议重新评估项目价值或考虑归档")
        elif activity_state['state'] == 'DORMANT':
            recs.append("😴 项目活跃度较低，建议增加社区运营和技术博客曝光")
        
        # 趋势建议
        if trend_analysis['trend_class'] in ['STRONG_DOWN', 'MODERATE_DOWN']:
            recs.append("📉 当前趋势下行，建议关注并采取应对措施")
        
        # 阻力建议
        if trend_analysis['resistance']['score'] >= 60:
            recs.append("🛑 发展阻力较大，建议组织专项清理活动")
        
        # Bus Factor建议
        if bus_factor.get('risk_level') in ['CRITICAL', 'HIGH']:
            recs.append("👥 贡献者过于集中，建议培养更多核心贡献者")
        
        # ETD建议
        if etd_analysis['etd_status'] in ['CRITICAL', 'WARNING']:
            recs.append(f"⏳ {etd_analysis['description']}")
        
        # 层级特定建议
        if structural_tier == 'EMERGING':
            recs.append("🌱 新兴项目，建议加强文档建设和社区引导")
        elif structural_tier == 'GROWING':
            recs.append("📈 成长型项目，建议保持当前发展节奏，关注规模化挑战")
        elif structural_tier == 'MATURE':
            recs.append("🏢 成熟项目，建议关注技术债务和安全更新")
        elif structural_tier == 'GIANT':
            recs.append("🏛️ 巨型项目，建议关注生态治理和社区健康")
        
        # 默认建议
        if not recs:
            recs.append("✅ 项目状态健康，保持当前运营节奏即可")
        
        return recs[:5]  # 最多5条建议
    
    def generate_report(self, result: AnalysisResult) -> str:
        """生成详细报告"""
        report = f"""
{'═'*70}
                    {result.project_name} 深度诊断报告 (v6.0)
                    GitHub 项目分析器 - 数据流重构版
{'═'*70}

【三层状态分离】
  结构层级: {result.structural_tier} ({TIER_NAMES[result.structural_tier]})
  时间趋势: {result.temporal_state}
  活跃状态: {result.activity_state}

{'─'*70}

【GMM结构层级分析】
  最佳层级: {result.structural_tier} (置信度: {result.tier_confidence:.0%})
  层级概率分布:
"""
        for tier, prob in result.tier_probabilities.items():
            bar_length = int(prob * 30)
            bar = '█' * bar_length + '░' * (30 - bar_length)
            report += f"    {tier:10s} {bar} {prob:.1%}\n"
        
        report += f"""
{'─'*70}

【健康评估】
  综合评分: {result.health_score}/100 ({result.health_grade})
  各维度贡献:
"""
        for dim, contrib in result.dimension_scores.items():
            report += f"    • {dim}: {contrib:.1%}\n"
        
        report += f"""
{'─'*70}

【趋势分析】
  综合趋势: {result.trend_analysis['trend_score']}分 ({result.trend_analysis['trend_description']})
  
  ┌ 动量 (Momentum): {result.trend_analysis['momentum']['score']}/100
  │   {result.trend_analysis['momentum']['description']}
  │
  ├ 阻力 (Resistance): {result.trend_analysis['resistance']['score']}/100
  │   {result.trend_analysis['resistance']['description']}
  │
  └ 潜力 (Potential): {result.trend_analysis['potential']['score']}/100
      {result.trend_analysis['potential']['description']}

{'─'*70}

【风险分析】
  风险等级: {result.risk_analysis['level']} ({result.risk_analysis['score']}分)
"""
        if result.risk_analysis['alerts']:
            for alert in result.risk_analysis['alerts']:
                report += f"    ⚠️  {alert}\n"
        
        report += f"""
{'─'*70}

【趋势预测】
"""
        if result.trend_predictions.get('openrank'):
            pred = result.trend_predictions['openrank']
            report += f"""
  OpenRank 趋势预测:
    当前值: {pred['current_value']:.2f}
    预测方向: {pred['direction']} (置信度: {pred['direction_confidence']:.0%})
    未来6个月平均: {pred['future_avg']:.2f}
"""
        
        report += f"""
{'─'*70}

【专项分析】
  Bus Factor 2.0: {result.bus_factor_2.get('effective_bus_factor', 'N/A')}
    {result.bus_factor_2.get('description', '')}
  
  ETD寿命分析: {result.etd_analysis['etd_status']}
    {result.etd_analysis.get('description', '')}
  
  黑马潜力: {'是' if result.dark_horse_analysis.get('is_dark_horse') else '否'} 
    (得分: {result.dark_horse_analysis.get('score', 0)}/100)

{'─'*70}

【GitHub数据验证】
"""
        if 'error' in result.conclusion_validation:
            report += f"    {result.conclusion_validation['error']}\n"
        else:
            report += f"    整体验证: {'通过' if result.conclusion_validation.get('overall_valid') else '需复核'}\n"
            report += f"    验证置信度: {result.conclusion_validation.get('confidence', 0)}%\n"
        
        report += f"""
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
    
    def _save_reports(self, result: AnalysisResult):
        """保存报告"""
        try:
            # 保存文本报告
            txt_file = f"{self.org}_{self.repo}_v6_report.txt"
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(result.detailed_report)
            print(f"📝 文本报告已保存: {txt_file}")
            
            # 保存JSON数据
            json_data = {
                'project': result.project_name,
                'generated_at': datetime.now().isoformat(),
                'structural_tier': result.structural_tier,
                'temporal_state': result.temporal_state,
                'activity_state': result.activity_state,
                'health': {
                    'score': result.health_score,
                    'grade': result.health_grade
                },
                'trend_analysis': result.trend_analysis,
                'risk_analysis': result.risk_analysis,
                'recommendations': result.recommendations
            }
            
            json_file = f"{self.org}_{self.repo}_v6_data.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            print(f"📊 JSON数据已保存: {json_file}")
            
        except Exception as e:
            print(f"保存报告时出错: {e}")
    
    def plot_dashboard(self, result: AnalysisResult):
        """绘制仪表板"""
        try:
            fig = plt.figure(figsize=(16, 12))
            fig.suptitle(f'{self.org}/{self.repo} 项目分析仪表板', fontsize=16, fontweight='bold')
            
            # 创建子图布局
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
            
            # 1. 三层状态图
            ax1 = fig.add_subplot(gs[0, 0])
            self._plot_three_layer_status(ax1, result)
            
            # 2. 健康评分仪表盘
            ax2 = fig.add_subplot(gs[0, 1])
            self._plot_health_gauge(ax2, result)
            
            # 3. GMM概率分布
            ax3 = fig.add_subplot(gs[0, 2])
            self._plot_gmm_probabilities(ax3, result)
            
            # 4. 趋势分析雷达图
            ax4 = fig.add_subplot(gs[1, 0], polar=True)
            self._plot_trend_radar(ax4, result)
            
            # 5. 风险等级
            ax5 = fig.add_subplot(gs[1, 1])
            self._plot_risk_gauge(ax5, result)
            
            # 6. 时间序列趋势
            ax6 = fig.add_subplot(gs[1, 2])
            self._plot_time_series(ax6, result)
            
            # 7. Bus Factor分析
            ax7 = fig.add_subplot(gs[2, 0])
            self._plot_bus_factor(ax7, result)
            
            # 8. 潜力分析
            ax8 = fig.add_subplot(gs[2, 1])
            self._plot_potential(ax8, result)
            
            # 9. 建议关键词
            ax9 = fig.add_subplot(gs[2, 2])
            self._plot_recommendations(ax9, result)
            
            plt.tight_layout()
            plt.savefig(f"{self.org}_{self.repo}_v6_dashboard.png", dpi=150, bbox_inches='tight')
            print(f"📈 仪表板图表已保存: {self.org}_{self.repo}_v6_dashboard.png")
            plt.show()
            
        except Exception as e:
            print(f"绘图失败: {e}")
    
    def _plot_three_layer_status(self, ax, result: AnalysisResult):
        """绘制三层状态图"""
        layers = ['结构层级', '时间趋势', '活跃状态']
        statuses = [result.structural_tier, result.temporal_state, result.activity_state]
        colors = [COLORS['primary'], COLORS['warning'], COLORS['success']]
        
        y_pos = np.arange(len(layers))
        ax.barh(y_pos, [1, 1, 1], color='lightgray', alpha=0.3)
        bars = ax.barh(y_pos, [0.8, 0.8, 0.8], color=colors, alpha=0.8)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(layers, fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        
        for i, (bar, status) in enumerate(zip(bars, statuses)):
            ax.text(0.4, i, status, va='center', ha='center', fontsize=9, fontweight='bold', color='white')
        
        ax.set_title('三层状态分离', fontsize=11, fontweight='bold', pad=10)
        ax.grid(False)
    
    def _plot_health_gauge(self, ax, result: AnalysisResult):
        """绘制健康评分仪表盘"""
        score = result.health_score
        
        # 清空坐标轴
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # 绘制仪表盘
        center_x, center_y = 0.5, 0.5
        radius = 0.4
        
        # 危险区域
        danger_angle = np.linspace(np.pi, np.pi + np.pi * 0.35, 50)
        x_danger = center_x + radius * np.cos(danger_angle)
        y_danger = center_y + radius * np.sin(danger_angle)
        ax.fill_between(x_danger, center_y, y_danger, color=COLORS['danger'], alpha=0.3)
        
        # 警告区域
        warning_angle = np.linspace(np.pi + np.pi * 0.35, np.pi + np.pi * 0.65, 50)
        x_warning = center_x + radius * np.cos(warning_angle)
        y_warning = center_y + radius * np.sin(warning_angle)
        ax.fill_between(x_warning, center_y, y_warning, color=COLORS['warning'], alpha=0.3)
        
        # 安全区域
        safe_angle = np.linspace(np.pi + np.pi * 0.65, 2*np.pi, 50)
        x_safe = center_x + radius * np.cos(safe_angle)
        y_safe = center_y + radius * np.sin(safe_angle)
        ax.fill_between(x_safe, center_y, y_safe, color=COLORS['success'], alpha=0.3)
        
        # 指针
        angle = np.pi + np.pi * (score / 100)
        x_tip = center_x + radius * 0.8 * np.cos(angle)
        y_tip = center_y + radius * 0.8 * np.sin(angle)
        ax.plot([center_x, x_tip], [center_y, y_tip], 'k-', lw=2)
        
        # 中心点
        ax.add_patch(plt.Circle((center_x, center_y), 0.02, color='black'))
        
        # 分数和等级
        ax.text(center_x, center_y - 0.1, f'{score:.0f}/100', 
                ha='center', va='center', fontsize=14, fontweight='bold')
        ax.text(center_x, center_y - 0.2, result.health_grade, 
                ha='center', va='center', fontsize=12, fontweight='bold')
        
        ax.axis('off')
        ax.set_title('健康评分', fontsize=11, fontweight='bold', pad=10)
    
    def _plot_gmm_probabilities(self, ax, result: AnalysisResult):
        """绘制GMM概率分布"""
        tiers = list(result.tier_probabilities.keys())
        probs = list(result.tier_probabilities.values())
        colors = [TIER_BENCHMARKS[t].get('color', COLORS['primary']) for t in tiers]
        
        bars = ax.bar(tiers, probs, color=colors, alpha=0.8)
        ax.set_ylim(0, 1)
        
        for bar, prob in zip(bars, probs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{prob:.1%}', ha='center', va='bottom', fontsize=8)
        
        ax.set_ylabel('概率', fontsize=9)
        ax.set_title(f'GMM层级概率 (最佳: {result.structural_tier})', fontsize=11, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
    
    def _plot_trend_radar(self, ax, result: AnalysisResult):
        """绘制趋势雷达图"""
        categories = ['动量', '阻力', '潜力']
        values = [
            result.trend_analysis['momentum']['score'],
            100 - result.trend_analysis['resistance']['score'],
            result.trend_analysis['potential']['score']
        ]
        values = values + [values[0]]
        
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += [angles[0]]
        
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # 绘制网格
        for i in [20, 40, 60, 80, 100]:
            ax.plot(angles, [i] * len(angles), color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # 绘制数据
        ax.plot(angles, values, 'o-', color=COLORS['primary'], linewidth=2)
        ax.fill(angles, values, color=COLORS['primary'], alpha=0.2)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_ylim(0, 100)
        ax.grid(True)
        
        ax.set_title('趋势三维分析', fontsize=11, fontweight='bold', pad=20)
    
    def _plot_risk_gauge(self, ax, result: AnalysisResult):
        """绘制风险等级"""
        risk_levels = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
        risk_scores = [80, 60, 40, 20]
        colors = [COLORS['danger'], COLORS['warning'], COLORS['info'], COLORS['success']]
        
        current_risk = result.risk_analysis['level']
        current_idx = risk_levels.index(current_risk) if current_risk in risk_levels else 0
        
        # 绘制所有风险等级
        for i, (level, score, color) in enumerate(zip(risk_levels, risk_scores, colors)):
            alpha = 0.8 if i == current_idx else 0.3
            ax.barh(level, score, color=color, alpha=alpha)
            ax.text(score + 2, i, f'{score}分', va='center', fontsize=9)
        
        ax.set_xlim(0, 100)
        ax.set_title(f'风险等级: {current_risk}', fontsize=11, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)
    
    def _plot_time_series(self, ax, result: AnalysisResult):
        """绘制时间序列"""
        if 'openrank' in self.od_df.columns:
            data = self.od_df['openrank']
            ax.plot(data.index, data.values, color=COLORS['primary'], lw=2)
            ax.fill_between(data.index, 0, data.values, color=COLORS['primary'], alpha=0.2)
            
            # 标记变点
            for cp in result.change_points[:2]:
                idx = cp['index']
                if idx < len(data):
                    ax.axvline(x=data.index[idx], color=COLORS['warning'], linestyle='--', alpha=0.7)
                    ax.text(data.index[idx], data.max() * 0.9, cp['type'][:3], 
                           rotation=90, fontsize=8, ha='right')
            
            ax.set_xlabel('时间', fontsize=9)
            ax.set_ylabel('OpenRank', fontsize=9)
            ax.tick_params(axis='x', rotation=30)
            ax.grid(True, alpha=0.3)
            ax.set_title('OpenRank时间序列', fontsize=11, fontweight='bold')
    
    def _plot_bus_factor(self, ax, result: AnalysisResult):
        """绘制Bus Factor分析"""
        bf = result.bus_factor_2.get('effective_bus_factor', 1)
        risk = result.bus_factor_2.get('risk_level', 'UNKNOWN')
        
        colors = {'CRITICAL': COLORS['danger'], 'HIGH': COLORS['warning'], 
                 'MEDIUM': COLORS['info'], 'LOW': COLORS['success'], 'UNKNOWN': 'gray'}
        
        ax.barh(['Bus Factor'], [min(10, bf)], color=colors.get(risk, 'gray'), alpha=0.8)
        ax.set_xlim(0, 10)
        ax.text(min(10, bf) + 0.2, 0, f'{bf:.1f}', va='center', fontsize=10, fontweight='bold')
        
        ax.set_title(f'Bus Factor: {risk}', fontsize=11, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)
    
    def _plot_potential(self, ax, result: AnalysisResult):
        """绘制潜力分析"""
        current = result.trend_analysis['potential']['current_openrank']
        tier_benchmark = result.trend_analysis['potential']['tier_benchmark']
        next_tier = 'N/A'
        
        # 计算与下一层级的差距
        tiers = ['EMERGING', 'GROWING', 'MATURE', 'GIANT']
        if result.structural_tier in tiers:
            idx = tiers.index(result.structural_tier)
            if idx < len(tiers) - 1:
                next_tier = tiers[idx + 1]
        
        categories = ['当前', '当前层级基准', '下一层级目标']
        values = [current, tier_benchmark, TIER_BENCHMARKS.get(next_tier, {}).get('openrank', current * 1.5)]
        
        colors = [COLORS['primary'], COLORS['warning'], COLORS['success']]
        bars = ax.bar(categories, values, color=colors, alpha=0.8)
        
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.02,
                   f'{value:.1f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel('OpenRank', fontsize=9)
        ax.set_title('增长潜力分析', fontsize=11, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
    
    def _plot_recommendations(self, ax, result: AnalysisResult):
        """绘制建议关键词"""
        if not result.recommendations:
            ax.text(0.5, 0.5, '无建议', ha='center', va='center')
            return
        
        # 提取关键词
        keywords = []
        for rec in result.recommendations:
            # 简单的关键词提取
            if '风险' in rec:
                keywords.append('风险管理')
            if '增长' in rec or '发展' in rec:
                keywords.append('增长策略')
            if '社区' in rec:
                keywords.append('社区建设')
            if '贡献者' in rec:
                keywords.append('贡献者培养')
            if '技术' in rec:
                keywords.append('技术优化')
            if '安全' in rec:
                keywords.append('安全维护')
        
        # 去重
        keywords = list(set(keywords))[:5]
        
        if not keywords:
            keywords = ['保持稳定', '持续观察']
        
        # 绘制词云样式
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        n = len(keywords)
        for i, keyword in enumerate(keywords):
            size = 12 - i * 2
            x = 0.1 + (i % 3) * 0.3
            y = 0.8 - (i // 3) * 0.4
            ax.text(x, y, keyword, fontsize=size, fontweight='bold', 
                   ha='center', va='center', alpha=0.8)
        
        ax.axis('off')
        ax.set_title('建议关键词', fontsize=11, fontweight='bold', pad=10)


# ============== 主入口 ==============
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  GitHub 项目深度分析器 v6.0 - 数据流重构版")
    print("  OpenDigger(趋势) + GitHub(锚定) + 三层状态分离")
    print("="*60 + "\n")
    
    # 从命令行参数获取URL
    import sys
    if len(sys.argv) > 1:
        url = sys.argv[1]
        print(f"使用命令行参数: {url}")
    else:
        url = input("请输入 GitHub 项目地址 (例如: facebook/react): ").strip()
    
    if not url:
        print("使用默认项目: facebook/react")
        url = "facebook/react"
    
    # 获取GitHub Token
    token = os.getenv('GITHUB_TOKEN')
    if not token and len(sys.argv) > 2:
        token = sys.argv[2]
    
    if not token:
        use_token = input("是否使用 GitHub Token? (y/n, 推荐使用): ").strip().lower()
        if use_token == 'y':
            token = input("请输入 GitHub API Token: ").strip()
    
    analyzer = ProjectAnalyzerV6(url, github_token=token)
    result = analyzer.analyze()
    
    if result:
        print("\n" + "="*60)
        print("✅ 分析完成！总结:")
        print(f"   结构层级: {result.structural_tier} ({TIER_NAMES[result.structural_tier]})")
        print(f"   时间趋势: {result.temporal_state}")
        print(f"   活跃状态: {result.activity_state}")
        print(f"   健康评分: {result.health_score}/100 ({result.health_grade})")
        print(f"   风险等级: {result.risk_analysis['level']}")
        print("="*60)
    else:
        print("\n❌ 分析失败，请检查输入和网络连接")