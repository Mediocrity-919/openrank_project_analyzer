"""
GitHub 项目深度分析器 v3.0 - 专业版
=====================================
核心改进：
1. 智能图表选择 - 根据可用数据自动选择最佳图表
2. 分阶段专属算法 - 每个层级使用最适配的算法
3. 专业可视化设计 - 6宫格布局，避免数据缺失问题
4. 高级算法引入 - Gompertz曲线、指数平滑、CUSUM检测等
"""

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Wedge
import seaborn as sns
import re
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from scipy.stats import pearsonr, linregress
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')

# ============== 显示设置 ==============
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.labelcolor'] = '#333333'
plt.rcParams['xtick.color'] = '#666666'
plt.rcParams['ytick.color'] = '#666666'

# ============== 颜色主题 ==============
COLORS = {
    'primary': '#2E86AB',
    'success': '#28A745',
    'warning': '#FFC107',
    'danger': '#DC3545',
    'info': '#17A2B8',
    'secondary': '#6C757D',
    'light': '#F8F9FA',
    'dark': '#343A40',
    'gradient_start': '#667eea',
    'gradient_end': '#764ba2'
}

# ============== 层级配置 ==============
@dataclass
class TierConfig:
    name: str
    description: str
    algorithms: List[str]
    health_threshold: float
    growth_threshold: float
    color: str

TIER_CONFIGS = {
    'GIANT': TierConfig(
        name='巨型项目',
        description='生态级开源项目，拥有庞大社区和完善治理',
        algorithms=['STL趋势分解', 'CUSUM异常检测', '贡献者生态分析'],
        health_threshold=1.2,
        growth_threshold=0.05,
        color='#9B59B6'
    ),
    'MATURE': TierConfig(
        name='成熟项目',
        description='稳定运营的中型项目，社区活跃度良好',
        algorithms=['移动平均趋势', '债务效率分析', '响应时效分析'],
        health_threshold=1.0,
        growth_threshold=0.15,
        color='#3498DB'
    ),
    'GROWING': TierConfig(
        name='成长项目',
        description='快速发展期项目，增长势头明显',
        algorithms=['Gompertz曲线拟合', '增长动力分析', '转化漏斗分析'],
        health_threshold=0.9,
        growth_threshold=0.30,
        color='#2ECC71'
    ),
    'EMERGING': TierConfig(
        name='新兴项目',
        description='起步阶段项目，潜力待发掘',
        algorithms=['初始动力评估', '关注热度分析', '早期转化分析'],
        health_threshold=0.8,
        growth_threshold=0.50,
        color='#E67E22'
    )
}

# ============== 诊断结果类 ==============
@dataclass
class AnalysisResult:
    # 基础信息
    project_name: str
    tier: str
    tier_config: TierConfig
    lifecycle: str
    vitality: str
    
    # 核心评分
    health_score: float
    health_grade: str
    dimension_scores: Dict[str, float]
    
    # 高级分析
    trend_analysis: Dict
    growth_analysis: Dict
    risk_analysis: Dict
    dark_horse_analysis: Dict
    
    # 建议
    recommendations: List[str]
    summary: str

# ============== 核心分析器 ==============
class ProjectAnalyzerV3:
    """专业版项目分析器 - 分阶段算法"""
    
    CORE_METRICS = [
        "openrank", "activity", "stars", "attention",
        "participants", "new_contributors", "inactive_contributors",
        "bus_factor", "issues_new", "issues_closed",
        "pr_new", "pr_merged"
    ]
    
    def __init__(self, url: str):
        self.org, self.repo = self._parse_url(url)
        self.df = pd.DataFrame()
        self.tier = None
        self.config = None
    
    def _parse_url(self, url: str) -> Tuple[str, str]:
        match = re.search(r"github\.com/([^/]+)/([^/]+)", url)
        if match:
            return match.group(1), match.group(2).replace(".git", "")
        if "/" in url and "http" not in url:
            parts = url.split('/')
            return parts[0], parts[1]
        raise ValueError("无效的 GitHub URL")
    
    # ==================== 数据获取 ====================
    def fetch_data(self) -> bool:
        print(f"\n{'='*60}")
        print(f"  正在分析: {self.org}/{self.repo}")
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
            print("❌ 无法获取数据")
            return False
        
        self.df = pd.DataFrame(raw_data).fillna(0)
        self.df.index = pd.to_datetime(self.df.index)
        self.df = self.df.sort_index()
        
        # 保存原始数据
        self.df.to_csv(f"{self.org}_{self.repo}_data.csv", encoding='utf-8-sig')
        
        # 确定层级
        self.tier = self._classify_tier()
        self.config = TIER_CONFIGS[self.tier]
        
        print(f"✓ 获取 {len(self.df)} 个月数据")
        print(f"✓ 项目层级: {self.tier} ({self.config.name})")
        print(f"✓ 使用算法: {', '.join(self.config.algorithms)}")
        
        return True
    
    def _classify_tier(self) -> str:
        avg_or = self.df['openrank'].mean() if 'openrank' in self.df else 0
        total_stars = self.df['stars'].sum() if 'stars' in self.df else 0
        
        if avg_or >= 50 or total_stars >= 10000:
            return 'GIANT'
        elif avg_or >= 15 or total_stars >= 2000:
            return 'MATURE'
        elif avg_or >= 3 or total_stars >= 300:
            return 'GROWING'
        return 'EMERGING'
    
    # ==================== 生命周期识别 ====================
    def analyze_lifecycle(self) -> str:
        if len(self.df) < 12:
            return 'INCUBATION'
        
        openrank = self.df['openrank']
        n = len(openrank)
        
        # 三段分析
        q1 = openrank.iloc[:n//3].mean()
        q2 = openrank.iloc[n//3:2*n//3].mean()
        q3 = openrank.iloc[2*n//3:].mean()
        
        # 最近趋势
        recent = openrank.tail(6)
        slope = linregress(range(len(recent)), recent.values).slope
        
        if q1 < q2 < q3 and slope > 0:
            return 'GROWTH'
        elif q3 >= q2 * 0.85 and q2 >= q1:
            return 'MATURITY'
        elif q3 < q2 * 0.7:
            return 'REVIVAL' if slope > 0.3 else 'DECLINE'
        return 'MATURITY'
    
    # ==================== 生命状态诊断 ====================
    def analyze_vitality(self) -> str:
        activity = self.df['activity']
        recent = activity.tail(6)
        slope = linregress(range(len(recent)), recent.values).slope
        
        peak = activity.max()
        current = recent.mean()
        
        # 成熟度指数
        maturity = self.df['openrank'].sum() / 100 * self.df['participants'].max() / 50
        
        if slope > 0:
            return 'THRIVING'
        
        if self.tier in ['GIANT', 'MATURE'] and maturity > 3 and current > peak * 0.2:
            return 'STABLE'
        
        if current < peak * 0.1:
            return 'ZOMBIE'
        
        return 'DORMANT'
    
    # ==================== 分阶段算法 ====================
    def run_tier_algorithms(self) -> Dict:
        """根据项目层级运行适配算法"""
        if self.tier == 'GIANT':
            return self._algorithms_giant()
        elif self.tier == 'MATURE':
            return self._algorithms_mature()
        elif self.tier == 'GROWING':
            return self._algorithms_growing()
        else:
            return self._algorithms_emerging()
    
    def _algorithms_giant(self) -> Dict:
        """巨型项目算法组"""
        result = {'tier': 'GIANT', 'algorithms_used': []}
        
        # 算法1: STL 趋势分解
        if 'openrank' in self.df.columns and len(self.df) >= 24:
            try:
                from statsmodels.tsa.seasonal import STL
                stl = STL(self.df['openrank'], seasonal=13, robust=True).fit()
                trend = stl.trend
                
                # 趋势稳定性 (越小越稳定)
                stability = trend.std() / (trend.mean() + 0.1)
                # 长期方向
                direction = linregress(range(12), trend.tail(12).values).slope
                
                result['stl_stability'] = round(stability, 3)
                result['stl_direction'] = round(direction, 3)
                result['trend_status'] = '稳定上升' if direction > 0.1 else ('稳定' if direction > -0.1 else '下滑')
                result['algorithms_used'].append('STL趋势分解')
            except:
                pass
        
        # 算法2: CUSUM 异常检测
        if 'activity' in self.df.columns:
            activity = self.df['activity'].values
            mean_val = np.mean(activity)
            cusum = np.cumsum(activity - mean_val)
            
            # 检测显著偏离
            threshold = 3 * np.std(activity)
            anomalies = np.sum(np.abs(cusum) > threshold)
            
            result['cusum_anomalies'] = int(anomalies)
            result['cusum_status'] = '异常较多' if anomalies > 5 else '正常'
            result['algorithms_used'].append('CUSUM异常检测')
        
        # 算法3: 贡献者生态分析
        if 'participants' in self.df.columns and 'new_contributors' in self.df.columns:
            participants = self.df['participants'].tail(12)
            new_contrib = self.df['new_contributors'].tail(12)
            
            # 生态健康度 = 新增贡献者占比的稳定性
            ratio = new_contrib / (participants + 1)
            eco_health = 1 - ratio.std() / (ratio.mean() + 0.1)
            
            result['ecosystem_health'] = round(max(0, min(1, eco_health)), 2)
            result['algorithms_used'].append('贡献者生态分析')
        
        return result
    
    def _algorithms_mature(self) -> Dict:
        """成熟项目算法组"""
        result = {'tier': 'MATURE', 'algorithms_used': []}
        
        # 算法1: 指数移动平均趋势
        if 'openrank' in self.df.columns:
            openrank = self.df['openrank']
            ema12 = openrank.ewm(span=12).mean()
            ema6 = openrank.ewm(span=6).mean()
            
            # 金叉/死叉信号
            current_signal = 'GOLDEN' if ema6.iloc[-1] > ema12.iloc[-1] else 'DEATH'
            trend_strength = abs(ema6.iloc[-1] - ema12.iloc[-1]) / (ema12.iloc[-1] + 0.1)
            
            result['ema_signal'] = current_signal
            result['ema_strength'] = round(trend_strength, 3)
            result['algorithms_used'].append('EMA趋势分析')
        
        # 算法2: Issue 债务效率
        if 'issues_closed' in self.df.columns and 'issues_new' in self.df.columns:
            closed = self.df['issues_closed'].tail(6).mean()
            new = self.df['issues_new'].tail(6).mean()
            
            debt_ratio = closed / (new + 0.1)
            
            # 债务趋势
            debt_history = self.df['issues_closed'] / (self.df['issues_new'] + 0.1)
            debt_trend = linregress(range(min(12, len(debt_history))), 
                                   debt_history.tail(12).values).slope
            
            result['debt_ratio'] = round(debt_ratio, 2)
            result['debt_trend'] = round(debt_trend, 3)
            result['debt_status'] = '健康' if debt_ratio >= 1 else ('警告' if debt_ratio >= 0.7 else '危险')
            result['algorithms_used'].append('债务效率分析')
        
        # 算法3: PR 响应效率
        if 'pr_merged' in self.df.columns and 'pr_new' in self.df.columns:
            merged = self.df['pr_merged'].tail(6).mean()
            new = self.df['pr_new'].tail(6).mean()
            
            pr_efficiency = merged / (new + 0.1)
            result['pr_efficiency'] = round(pr_efficiency, 2)
            result['algorithms_used'].append('PR响应分析')
        
        return result
    
    def _algorithms_growing(self) -> Dict:
        """成长项目算法组"""
        result = {'tier': 'GROWING', 'algorithms_used': []}
        
        # 算法1: Gompertz 曲线拟合 (S型增长)
        if 'openrank' in self.df.columns and len(self.df) >= 12:
            def gompertz(t, a, b, c):
                return a * np.exp(-b * np.exp(-c * t))
            
            try:
                openrank = self.df['openrank'].values
                t = np.arange(len(openrank))
                
                # 拟合
                popt, _ = curve_fit(gompertz, t, openrank, 
                                   p0=[max(openrank)*2, 1, 0.1],
                                   maxfev=5000)
                
                # 预测未来6个月
                future_t = np.arange(len(openrank), len(openrank) + 6)
                future_vals = gompertz(future_t, *popt)
                
                # 增长潜力 = 预测增长率
                growth_potential = (future_vals[-1] - openrank[-1]) / (openrank[-1] + 0.1)
                
                result['gompertz_params'] = [round(p, 3) for p in popt]
                result['growth_potential'] = round(growth_potential, 2)
                result['algorithms_used'].append('Gompertz曲线拟合')
            except:
                # 降级到线性预测
                slope = linregress(range(len(openrank)), openrank).slope
                result['linear_growth'] = round(slope, 3)
                result['algorithms_used'].append('线性增长分析')
        
        # 算法2: 增长动力分解
        if 'openrank' in self.df.columns:
            openrank = self.df['openrank']
            
            # 一阶导数 (速度)
            velocity = openrank.diff().tail(6).mean()
            # 二阶导数 (加速度)
            acceleration = openrank.diff().diff().tail(6).mean()
            
            result['growth_velocity'] = round(velocity, 3)
            result['growth_acceleration'] = round(acceleration, 3)
            result['growth_phase'] = '爆发期' if acceleration > 0.1 else ('平稳期' if acceleration > -0.1 else '减速期')
            result['algorithms_used'].append('增长动力分析')
        
        # 算法3: Star转化漏斗
        if 'stars' in self.df.columns and 'participants' in self.df.columns:
            stars = self.df['stars'].tail(12)
            participants = self.df['participants'].tail(12)
            
            # 转化率
            if stars.sum() > 0:
                conversion = participants.diff().sum() / (stars.sum() + 0.1)
                result['star_conversion'] = round(conversion, 4)
                result['algorithms_used'].append('转化漏斗分析')
        
        return result
    
    def _algorithms_emerging(self) -> Dict:
        """新兴项目算法组"""
        result = {'tier': 'EMERGING', 'algorithms_used': []}
        
        # 算法1: 初始动力评估
        if 'openrank' in self.df.columns:
            openrank = self.df['openrank']
            
            # 月均增长率
            if len(openrank) >= 3:
                start = openrank.iloc[0] + 0.1
                end = openrank.iloc[-1]
                months = len(openrank)
                
                monthly_growth = (end / start) ** (1/months) - 1
                result['monthly_growth_rate'] = round(monthly_growth, 3)
                
                # 增长稳定性
                growth_std = openrank.pct_change().std()
                result['growth_stability'] = round(1 / (growth_std + 0.1), 2)
                result['algorithms_used'].append('初始动力评估')
        
        # 算法2: 关注热度分析
        if 'stars' in self.df.columns:
            stars = self.df['stars']
            
            # 热度趋势
            heat_trend = linregress(range(min(6, len(stars))), stars.tail(6).values).slope
            result['heat_trend'] = round(heat_trend, 2)
            result['heat_status'] = '升温' if heat_trend > 0.5 else ('稳定' if heat_trend > -0.5 else '降温')
            result['algorithms_used'].append('关注热度分析')
        
        # 算法3: 早期转化分析
        if 'participants' in self.df.columns:
            participants = self.df['participants']
            
            # 贡献者增长
            if len(participants) >= 3:
                contrib_growth = participants.iloc[-1] / (participants.iloc[0] + 0.1) - 1
                result['contributor_growth'] = round(contrib_growth, 2)
                result['algorithms_used'].append('早期转化分析')
        
        return result
    
    # ==================== 综合分析 ====================
    def analyze_risk(self, vitality: str, algo_result: Dict) -> Dict:
        """风险评估"""
        risk_score = 0
        alerts = []
        
        # 活跃度风险
        if 'activity' in self.df.columns:
            activity = self.df['activity']
            slope = linregress(range(min(12, len(activity))), activity.tail(12).values).slope
            
            if slope < -0.5:
                risk_score += 30
                alerts.append('活跃度显著下降')
            elif slope < 0:
                risk_score += 15
                alerts.append('活跃度轻微下滑')
        
        # Bus Factor 风险
        if 'bus_factor' in self.df.columns:
            bf = self.df['bus_factor'].tail(3).mean()
            if bf <= 1:
                risk_score += 30
                alerts.append(f'Bus Factor 极低 ({bf:.0f})')
            elif bf <= 2:
                risk_score += 15
                alerts.append(f'Bus Factor 偏低 ({bf:.0f})')
        
        # 债务风险
        debt = algo_result.get('debt_ratio', 1.0)
        if debt < 0.5:
            risk_score += 25
            alerts.append('技术债务严重')
        elif debt < 0.8:
            risk_score += 10
            alerts.append('技术债务偏高')
        
        # 状态风险
        if vitality == 'ZOMBIE':
            risk_score += 40
            alerts.append('项目处于僵尸状态')
        elif vitality == 'DORMANT':
            risk_score += 15
            alerts.append('项目处于休眠状态')
        
        level = 'CRITICAL' if risk_score >= 50 else 'HIGH' if risk_score >= 30 else 'MEDIUM' if risk_score >= 15 else 'LOW'
        
        return {
            'score': risk_score,
            'level': level,
            'alerts': alerts
        }
    
    def analyze_dark_horse(self, algo_result: Dict) -> Dict:
        """黑马识别"""
        if self.tier in ['GIANT', 'MATURE']:
            return {
                'is_dark_horse': False,
                'score': 0,
                'reason': f'{self.config.name}已超出黑马范畴'
            }
        
        score = 0
        reasons = []
        
        # 增长动力
        acc = algo_result.get('growth_acceleration', 0)
        if acc > 0.1:
            score += 30
            reasons.append('增长加速明显')
        elif acc > 0:
            score += 15
            reasons.append('增长势头向上')
        
        # 增长率
        growth = algo_result.get('monthly_growth_rate', 0)
        if growth > 0.2:
            score += 30
            reasons.append(f'月增长率{growth*100:.0f}%')
        elif growth > 0.1:
            score += 15
            reasons.append(f'月增长率{growth*100:.0f}%')
        
        # 热度
        heat = algo_result.get('heat_trend', 0)
        if heat > 1:
            score += 25
            reasons.append('关注热度飙升')
        elif heat > 0:
            score += 10
            reasons.append('关注度上升')
        
        # 转化
        conv = algo_result.get('star_conversion', 0)
        if conv > 0.05:
            score += 15
            reasons.append('Star转化率高')
        
        return {
            'is_dark_horse': score >= 50,
            'score': score,
            'reasons': reasons,
            'verdict': '潜力黑马' if score >= 50 else '暂未达标'
        }
    
    def calculate_health_score(self, vitality: str, algo_result: Dict, risk: Dict) -> Tuple[float, str, Dict]:
        """计算健康评分"""
        scores = {}
        
        # 活力得分
        vitality_map = {'THRIVING': 100, 'STABLE': 80, 'DORMANT': 50, 'ZOMBIE': 20}
        scores['活力'] = vitality_map.get(vitality, 60)
        
        # 风险反向得分
        scores['安全'] = max(0, 100 - risk['score'] * 2)
        
        # 债务得分
        debt = algo_result.get('debt_ratio', 1.0)
        scores['维护'] = min(100, debt / 1.5 * 100)
        
        # 增长得分
        if self.tier in ['GROWING', 'EMERGING']:
            growth = algo_result.get('growth_acceleration', algo_result.get('monthly_growth_rate', 0))
            scores['增长'] = min(100, 50 + growth * 100)
        else:
            stability = algo_result.get('stl_stability', algo_result.get('ema_strength', 0.5))
            scores['稳定'] = min(100, (1 - stability) * 100)
        
        # 加权总分
        total = np.mean(list(scores.values()))
        
        # 状态调整
        if vitality == 'STABLE':
            total = max(total, 70)
        elif vitality == 'ZOMBIE':
            total = min(total, 35)
        
        grade = self._score_to_grade(total)
        
        return round(total, 1), grade, scores
    
    def _score_to_grade(self, score: float) -> str:
        if score >= 85: return 'A+'
        elif score >= 75: return 'A'
        elif score >= 65: return 'B+'
        elif score >= 55: return 'B'
        elif score >= 45: return 'C'
        elif score >= 35: return 'D'
        return 'F'
    
    def generate_recommendations(self, vitality: str, algo_result: Dict, risk: Dict) -> List[str]:
        """生成建议"""
        recs = []
        
        if risk['level'] in ['CRITICAL', 'HIGH']:
            recs.append('⚠️ 建议加强社区运营，发布技术博客保持项目活跃度')
        
        if algo_result.get('debt_ratio', 1) < 0.8:
            recs.append('📋 建议组织 Bug Bash 集中处理积压 Issue')
        
        if 'Bus Factor' in str(risk['alerts']):
            recs.append('👥 建议培养更多核心贡献者，降低单点依赖')
        
        if vitality == 'DORMANT':
            recs.append('💤 建议发布 Roadmap 或新版本预告激活社区')
        
        if vitality == 'STABLE':
            recs.append('✅ 项目已成熟，保持定期安全更新即可')
        
        if algo_result.get('growth_acceleration', 0) > 0.1:
            recs.append('🚀 增长势头良好，建议加大推广力度')
        
        if not recs:
            recs.append('✨ 项目状态健康，继续保持当前节奏')
        
        return recs
    
    # ==================== 专业可视化 ====================
    def plot_professional_charts(self, result: AnalysisResult):
        """绘制专业6宫格图表"""
        fig = plt.figure(figsize=(18, 12), facecolor='white')
        
        # 2行3列布局
        gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.25,
                             left=0.06, right=0.94, top=0.88, bottom=0.08)
        
        # 图1: OpenRank 趋势 + 预测
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_trend_chart(ax1, result)
        
        # 图2: 健康仪表盘
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_gauge_chart(ax2, result)
        
        # 图3: 活跃度热力图
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_activity_heatmap(ax3)
        
        # 图4: 贡献者流动图
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_contributor_flow(ax4)
        
        # 图5: 债务率趋势
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_debt_trend(ax5, result)
        
        # 图6: 综合诊断卡片
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_diagnosis_card(ax6, result)
        
        # 总标题
        fig.suptitle(
            f'{self.org}/{self.repo}  深度诊断报告',
            fontsize=20, fontweight='bold', color=COLORS['dark'], y=0.96
        )
        
        # 副标题
        fig.text(0.5, 0.92, 
                f'层级: {result.tier_config.name}  |  状态: {result.vitality}  |  评级: {result.health_grade} ({result.health_score}分)',
                ha='center', fontsize=12, color=COLORS['secondary'])
        
        # 保存
        filename = f"{self.org}_{self.repo}_report.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"\n📊 图表已保存: {filename}")
        plt.show()
    
    def _plot_trend_chart(self, ax, result: AnalysisResult):
        """OpenRank 趋势图"""
        if 'openrank' not in self.df.columns:
            ax.text(0.5, 0.5, 'OpenRank 数据不可用', ha='center', va='center', fontsize=12)
            ax.set_title('OpenRank 趋势', fontsize=12, fontweight='bold')
            return
        
        openrank = self.df['openrank']
        dates = self.df.index
        
        # 主线
        ax.plot(dates, openrank, color=COLORS['primary'], lw=2.5, label='OpenRank')
        ax.fill_between(dates, openrank, alpha=0.2, color=COLORS['primary'])
        
        # EMA 趋势线
        ema = openrank.ewm(span=6).mean()
        ax.plot(dates, ema, '--', color=COLORS['danger'], lw=1.5, label='趋势线(EMA6)')
        
        # 标注最高点和最新点
        max_idx = openrank.idxmax()
        ax.scatter([max_idx], [openrank[max_idx]], color=COLORS['success'], s=100, zorder=5)
        ax.annotate(f'峰值:{openrank[max_idx]:.1f}', xy=(max_idx, openrank[max_idx]),
                   xytext=(5, 10), textcoords='offset points', fontsize=9)
        
        ax.scatter([dates[-1]], [openrank.iloc[-1]], color=COLORS['info'], s=100, zorder=5)
        ax.annotate(f'当前:{openrank.iloc[-1]:.1f}', xy=(dates[-1], openrank.iloc[-1]),
                   xytext=(5, -15), textcoords='offset points', fontsize=9)
        
        ax.set_title('OpenRank 趋势分析', fontsize=12, fontweight='bold', pad=10)
        ax.legend(loc='upper left', fontsize=9)
        ax.tick_params(axis='x', rotation=30, labelsize=8)
        ax.set_ylabel('OpenRank', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    def _plot_gauge_chart(self, ax, result: AnalysisResult):
        """健康仪表盘"""
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # 绘制弧形仪表
        colors_arc = [COLORS['danger'], COLORS['warning'], COLORS['success']]
        angles = [0, 45, 90, 135, 180]
        
        for i in range(3):
            wedge = Wedge((0, 0), 1.2, angles[i], angles[i+1], width=0.3,
                         facecolor=colors_arc[i], alpha=0.3)
            ax.add_patch(wedge)
        
        # 指针
        score = result.health_score
        angle = 180 - (score / 100 * 180)
        angle_rad = np.radians(angle)
        
        x_end = 0.9 * np.cos(angle_rad)
        y_end = 0.9 * np.sin(angle_rad)
        
        ax.annotate('', xy=(x_end, y_end), xytext=(0, 0),
                   arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=3))
        
        # 中心圆
        circle = Circle((0, 0), 0.15, facecolor='white', edgecolor=COLORS['dark'], lw=2)
        ax.add_patch(circle)
        
        # 分数
        ax.text(0, -0.35, f'{score:.0f}', ha='center', va='center', 
               fontsize=36, fontweight='bold', color=COLORS['dark'])
        ax.text(0, -0.6, result.health_grade, ha='center', va='center',
               fontsize=14, color=COLORS['secondary'])
        
        # 标签
        ax.text(-1.3, 0, '风险', ha='center', fontsize=9, color=COLORS['danger'])
        ax.text(0, 1.1, '警告', ha='center', fontsize=9, color=COLORS['warning'])
        ax.text(1.3, 0, '健康', ha='center', fontsize=9, color=COLORS['success'])
        
        ax.set_title('健康评分', fontsize=12, fontweight='bold', pad=20)
    
    def _plot_activity_heatmap(self, ax):
        """活跃度热力图 (最近12个月)"""
        if 'activity' not in self.df.columns:
            ax.text(0.5, 0.5, '活跃度数据不可用', ha='center', va='center', fontsize=12)
            ax.set_title('活跃度分布', fontsize=12, fontweight='bold')
            ax.axis('off')
            return
        
        # 取最近12个月数据
        activity = self.df['activity'].tail(12)
        
        # 归一化
        norm_activity = (activity - activity.min()) / (activity.max() - activity.min() + 0.1)
        
        # 绘制水平条形图
        colors = [plt.cm.RdYlGn(v) for v in norm_activity]
        months = [d.strftime('%Y-%m') for d in activity.index]
        
        bars = ax.barh(range(len(activity)), activity.values, color=colors, edgecolor='white', height=0.8)
        
        ax.set_yticks(range(len(activity)))
        ax.set_yticklabels(months, fontsize=8)
        ax.set_xlabel('Activity', fontsize=10)
        ax.set_title('月度活跃度分布', fontsize=12, fontweight='bold', pad=10)
        
        # 标注数值
        for i, (bar, val) in enumerate(zip(bars, activity.values)):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                   f'{val:.0f}', va='center', fontsize=8)
        
        ax.invert_yaxis()
    
    def _plot_contributor_flow(self, ax):
        """贡献者流动图"""
        has_new = 'new_contributors' in self.df.columns
        has_inactive = 'inactive_contributors' in self.df.columns
        has_participants = 'participants' in self.df.columns
        
        if not has_participants:
            ax.text(0.5, 0.5, '贡献者数据不可用', ha='center', va='center', fontsize=12)
            ax.set_title('贡献者动态', fontsize=12, fontweight='bold')
            ax.axis('off')
            return
        
        data = self.df.tail(12)
        x = range(len(data))
        
        # 总贡献者线
        ax.plot(x, data['participants'], 'o-', color=COLORS['primary'], 
               lw=2, markersize=6, label='总贡献者')
        
        # 新增/流失柱状图
        if has_new and has_inactive:
            width = 0.35
            ax2 = ax.twinx()
            ax2.bar([i - width/2 for i in x], data['new_contributors'], 
                   width=width, color=COLORS['success'], alpha=0.7, label='新增')
            ax2.bar([i + width/2 for i in x], -data['inactive_contributors'],
                   width=width, color=COLORS['danger'], alpha=0.7, label='流失')
            ax2.axhline(0, color='gray', lw=0.5)
            ax2.set_ylabel('新增/流失', fontsize=9)
            ax2.legend(loc='upper right', fontsize=8)
        
        ax.set_xticks(x)
        ax.set_xticklabels([d.strftime('%m') for d in data.index], fontsize=8)
        ax.set_xlabel('月份', fontsize=9)
        ax.set_ylabel('总贡献者', fontsize=9)
        ax.set_title('贡献者流动分析', fontsize=12, fontweight='bold', pad=10)
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _plot_debt_trend(self, ax, result: AnalysisResult):
        """债务率趋势"""
        if 'issues_closed' not in self.df.columns or 'issues_new' not in self.df.columns:
            ax.text(0.5, 0.5, 'Issue 数据不可用', ha='center', va='center', fontsize=12)
            ax.set_title('技术债务', fontsize=12, fontweight='bold')
            ax.axis('off')
            return
        
        debt = self.df['issues_closed'] / (self.df['issues_new'] + 0.1)
        debt = debt.tail(12)
        
        # 颜色映射
        colors = [COLORS['success'] if v >= 1 else COLORS['warning'] if v >= 0.7 else COLORS['danger'] 
                 for v in debt.values]
        
        bars = ax.bar(range(len(debt)), debt.values, color=colors, edgecolor='white', alpha=0.8)
        
        # 参考线
        ax.axhline(1.0, color=COLORS['success'], linestyle='--', lw=1.5, label='健康线')
        ax.axhline(0.7, color=COLORS['warning'], linestyle='--', lw=1.5, label='警戒线')
        
        ax.set_xticks(range(len(debt)))
        ax.set_xticklabels([d.strftime('%m') for d in debt.index], fontsize=8)
        ax.set_xlabel('月份', fontsize=9)
        ax.set_ylabel('债务率 (关闭/新增)', fontsize=9)
        ax.set_title('Issue 处理效率', fontsize=12, fontweight='bold', pad=10)
        ax.legend(fontsize=8, loc='upper right')
        
        # 当前值标注
        current = debt.iloc[-1]
        ax.text(len(debt)-1, current + 0.1, f'{current:.2f}', ha='center', fontsize=9, fontweight='bold')
    
    def _plot_diagnosis_card(self, ax, result: AnalysisResult):
        """诊断卡片"""
        ax.axis('off')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        
        # 背景卡片
        card = FancyBboxPatch((0.2, 0.2), 9.6, 9.6, 
                              boxstyle="round,pad=0.1,rounding_size=0.3",
                              facecolor='#f8f9fa', edgecolor='#dee2e6', lw=2)
        ax.add_patch(card)
        
        # 标题
        ax.text(5, 9.2, '诊断摘要', ha='center', fontsize=14, fontweight='bold', color=COLORS['dark'])
        ax.axhline(y=8.8, xmin=0.1, xmax=0.9, color='#dee2e6', lw=1)
        
        # 内容
        y_pos = 8.2
        items = [
            ('层级', f"{result.tier} ({result.tier_config.name})"),
            ('周期', f"{result.lifecycle}"),
            ('状态', f"{result.vitality}"),
            ('', ''),  # 空行
            ('风险', f"{result.risk_analysis['level']} (分数:{result.risk_analysis['score']})"),
            ('黑马', f"{result.dark_horse_analysis.get('verdict', 'N/A')}"),
        ]
        
        for label, value in items:
            if label:
                ax.text(0.8, y_pos, f'{label}:', fontsize=10, color=COLORS['secondary'])
                ax.text(3, y_pos, value, fontsize=10, color=COLORS['dark'], fontweight='bold')
            y_pos -= 0.9
        
        # 分隔线
        ax.axhline(y=4.5, xmin=0.1, xmax=0.9, color='#dee2e6', lw=1)
        
        # 关键建议
        ax.text(5, 4, '关键建议', ha='center', fontsize=11, fontweight='bold', color=COLORS['dark'])
        
        y_pos = 3.3
        for rec in result.recommendations[:3]:
            # 截断过长文本
            if len(rec) > 25:
                rec = rec[:25] + '...'
            ax.text(0.8, y_pos, rec, fontsize=9, color=COLORS['secondary'])
            y_pos -= 0.8
        
        # 使用的算法
        ax.text(0.8, 0.8, f"算法: {', '.join(result.trend_analysis.get('algorithms_used', [])[:2])}", 
               fontsize=8, color='#adb5bd')
    
    # ==================== 文字报告 ====================
    def generate_report(self, result: AnalysisResult) -> str:
        """生成详细文字报告"""
        report = f"""
{'='*70}
                    项目深度诊断报告
{'='*70}

项目：{result.project_name}
生成时间：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

{'─'*70}
一、项目概况
{'─'*70}

  层级分类：{result.tier} - {result.tier_config.name}
  层级说明：{result.tier_config.description}
  使用算法：{', '.join(result.tier_config.algorithms)}
  
  生命周期：{result.lifecycle}
  当前状态：{result.vitality}

{'─'*70}
二、健康评估
{'─'*70}

  综合评分：{result.health_score}/100 分
  健康等级：{result.health_grade}
  
  分项得分：
"""
        for dim, score in result.dimension_scores.items():
            report += f"    • {dim}：{score:.0f}/100\n"
        
        report += f"""
{'─'*70}
三、算法分析结果
{'─'*70}

  使用算法：{', '.join(result.trend_analysis.get('algorithms_used', ['N/A']))}
"""
        for key, value in result.trend_analysis.items():
            if key != 'algorithms_used' and key != 'tier':
                report += f"    • {key}：{value}\n"
        
        report += f"""
{'─'*70}
四、风险评估
{'─'*70}

  风险等级：{result.risk_analysis['level']}
  风险分数：{result.risk_analysis['score']}/100
  
  风险提示：
"""
        for alert in result.risk_analysis['alerts']:
            report += f"    ⚠️ {alert}\n"
        if not result.risk_analysis['alerts']:
            report += "    ✅ 未发现明显风险\n"
        
        report += f"""
{'─'*70}
五、黑马分析
{'─'*70}

  是否黑马：{'是 ✨' if result.dark_horse_analysis.get('is_dark_horse') else '否'}
  潜力分数：{result.dark_horse_analysis.get('score', 0)}/100
  判定理由：
"""
        for reason in result.dark_horse_analysis.get('reasons', ['暂无']):
            report += f"    • {reason}\n"
        
        report += f"""
{'─'*70}
六、改进建议
{'─'*70}

"""
        for i, rec in enumerate(result.recommendations, 1):
            report += f"  {i}. {rec}\n"
        
        report += f"""
{'='*70}
                         报告结束
{'='*70}
"""
        return report
    
    # ==================== 主流程 ====================
    def run(self) -> Optional[AnalysisResult]:
        """执行分析"""
        if not self.fetch_data():
            return None
        
        # 分析
        lifecycle = self.analyze_lifecycle()
        vitality = self.analyze_vitality()
        algo_result = self.run_tier_algorithms()
        risk = self.analyze_risk(vitality, algo_result)
        dark_horse = self.analyze_dark_horse(algo_result)
        score, grade, dimensions = self.calculate_health_score(vitality, algo_result, risk)
        recommendations = self.generate_recommendations(vitality, algo_result, risk)
        
        # 构建结果
        result = AnalysisResult(
            project_name=f"{self.org}/{self.repo}",
            tier=self.tier,
            tier_config=self.config,
            lifecycle=lifecycle,
            vitality=vitality,
            health_score=score,
            health_grade=grade,
            dimension_scores=dimensions,
            trend_analysis=algo_result,
            growth_analysis={},
            risk_analysis=risk,
            dark_horse_analysis=dark_horse,
            recommendations=recommendations,
            summary=""
        )
        
        # 生成报告
        result.summary = self.generate_report(result)
        
        # 输出
        print(result.summary)
        
        # 保存报告
        with open(f"{self.org}_{self.repo}_report.txt", 'w', encoding='utf-8') as f:
            f.write(result.summary)
        
        # 绘图
        self.plot_professional_charts(result)
        
        return result


# ==================== 入口 ====================
if __name__ == "__main__":
    url = input("请输入 GitHub 项目地址: ").strip()
    analyzer = ProjectAnalyzerV3(url)
    analyzer.run()
