"""
GitHub 项目深度分析器 v2.1 - 优化版
=====================================
改进内容：
1. 精简数据获取，只获取关键指标
2. 保存原始数据到 CSV
3. 增加详细文字描述与术语解释
4. 优化图表布局
5. 改进 ETD 算法，区分成熟稳定与真正衰退
6. 丰富最终报告文字
"""

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from statsmodels.tsa.seasonal import STL
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')

# ============== 显示设置 ==============
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'
sns.set_theme(style="whitegrid", font='SimHei', context="paper")

# ============== 术语解释字典 ==============
TERMINOLOGY = {
    'GIANT': '巨型项目 - OpenRank≥50 或 累计Star≥10000，如 Linux、React 等生态级项目',
    'MATURE': '成熟项目 - OpenRank 15-50，已建立稳定社区和维护流程的项目',
    'GROWING': '成长项目 - OpenRank 3-15，处于快速发展期的项目',
    'EMERGING': '新兴项目 - OpenRank<3，刚起步或小型项目',
    'THRIVING': '繁荣状态 - 活跃度持续上升，社区活力充沛',
    'STABLE_MATURE': '成熟稳定 - 虽然活跃度下降，但项目功能完善、无需频繁更新，属于正常生命周期',
    'DORMANT': '休眠状态 - 暂时沉寂但核心团队仍在，可能在积蓄力量或等待下一版本',
    'ZOMBIE': '僵尸状态 - 核心贡献者流失、无人维护、Issue 无响应，面临废弃风险',
    'INCUBATION': '孵化期 - 项目初创阶段，数据积累不足12个月',
    'GROWTH': '成长期 - OpenRank 持续攀升，社区快速扩张',
    'MATURITY': '成熟期 - OpenRank 高位稳定，功能完善',
    'DECLINE': '衰退期 - OpenRank 持续下降，需警惕',
    'REVIVAL': '复苏期 - 曾经衰退但近期反弹',
    'ETD': '预计枯竭时间 (Estimated Time to Depletion) - 基于活跃度趋势预测的剩余活跃月数',
    'Bus Factor': '巴士系数 - 项目依赖的核心贡献者数量，数值越低表示单点风险越高',
    'Debt Ratio': '技术债务率 - Issue关闭数/新增数，>1表示处理效率高于新增速度',
    'C_conv': '转化系数 - Star 关注转化为实际贡献者的效率',
}

# ============== 数据类定义 ==============
@dataclass
class TierThresholds:
    """各层级阈值配置"""
    debt_healthy: float
    activity_tolerance: int
    growth_rate: float
    load_factor: float
    description: str

TIER_CONFIG = {
    'GIANT': TierThresholds(1.2, 24, 0.05, 3.0, '巨型生态级项目'),
    'MATURE': TierThresholds(1.0, 12, 0.15, 2.5, '成熟稳定型项目'),
    'GROWING': TierThresholds(0.9, 6, 0.30, 2.0, '快速成长型项目'),
    'EMERGING': TierThresholds(0.8, 3, 0.50, 1.5, '新兴孵化型项目'),
}

@dataclass
class DiagnosisResult:
    """诊断结果结构"""
    tier: str
    tier_desc: str
    lifecycle: str
    lifecycle_desc: str
    vitality: str
    vitality_desc: str
    health_score: float
    health_grade: str
    health_breakdown: Dict[str, float]
    dark_horse: Dict
    risk_assessment: Dict
    etd_analysis: Dict  # 新增：详细的寿命分析
    pathology_labels: List[str]
    recommendations: List[str]
    detailed_summary: str  # 新增：详细文字总结

# ============== 核心分析器类 ==============
class ProjectAnalyzerV2:
    """分层适配的项目分析器 v2.1"""
    
    # 关键指标（精简版）
    KEY_METRICS = [
        "openrank", "activity", "attention", "stars",
        "participants", "new_contributors", "inactive_contributors",
        "bus_factor", "issues_new", "issues_closed",
        "pr_new", "pr_merged"
    ]
    
    def __init__(self, url: str):
        self.org, self.repo = self._parse_github_url(url)
        self.df = pd.DataFrame()
        self.tier = None
        self.config = None
        
    def _parse_github_url(self, url: str) -> Tuple[str, str]:
        """解析 GitHub URL"""
        match = re.search(r"github\.com/([^/]+)/([^/]+)", url)
        if match:
            return match.group(1), match.group(2).replace(".git", "")
        if "/" in url and "http" not in url:
            parts = url.split('/')
            return parts[0], parts[1]
        raise ValueError("无效的 GitHub 网址格式")
    
    # ==================== 数据获取 ====================
    def fetch_data(self) -> bool:
        """从 OpenDigger 获取关键指标（精简版）"""
        print(f"🚀 正在检索项目 [{self.org}/{self.repo}] 的关键指标...")
        raw_data = {}
        
        for metric in self.KEY_METRICS:
            api_url = f"https://oss.open-digger.cn/github/{self.org}/{self.repo}/{metric}.json"
            try:
                res = requests.get(api_url, timeout=15)
                if res.status_code == 200:
                    data = res.json()
                    monthly = {k: v for k, v in data.items() if re.match(r'^\d{4}-\d{2}$', str(k))}
                    if monthly:
                        raw_data[metric] = pd.Series(monthly)
            except Exception:
                continue
        
        if not raw_data:
            print("❌ 无法获取项目数据")
            return False
            
        self.df = pd.DataFrame(raw_data).fillna(0)
        self.df.index = pd.to_datetime(self.df.index)
        self.df = self.df.sort_index()
        
        # 保存原始数据到 CSV
        self._save_raw_data()
        
        # 确定项目层级
        self.tier = self._classify_tier()
        self.config = TIER_CONFIG[self.tier]
        
        print(f"✅ 已获取 {len(self.df)} 个月数据，共 {len(self.df.columns)} 个指标")
        print(f"📊 项目层级: {self.tier} ({self.config.description})")
        return True
    
    def _save_raw_data(self):
        """保存原始数据到 CSV"""
        raw_file = f"{self.org}_{self.repo}_raw_data.csv"
        self.df.to_csv(raw_file, encoding='utf-8-sig')
        print(f"💾 原始数据已保存至: {raw_file}")
    
    # ==================== 层级分类 ====================
    def _classify_tier(self) -> str:
        """基于多维指标的项目层级分类"""
        avg_openrank = self.df['openrank'].mean() if 'openrank' in self.df else 0
        total_stars = self.df['stars'].sum() if 'stars' in self.df else 0
        max_participants = self.df['participants'].max() if 'participants' in self.df else 0
        
        if avg_openrank >= 50 or total_stars >= 10000:
            return 'GIANT'
        elif avg_openrank >= 15 or total_stars >= 2000:
            return 'MATURE'
        elif avg_openrank >= 3 or total_stars >= 300:
            return 'GROWING'
        else:
            return 'EMERGING'
    
    # ==================== 生命周期识别 ====================
    def identify_lifecycle(self) -> Tuple[str, str]:
        """识别生命周期阶段，返回 (阶段, 描述)"""
        if len(self.df) < 12:
            return 'INCUBATION', TERMINOLOGY['INCUBATION']
        
        openrank = self.df['openrank']
        n = len(openrank)
        
        q1_avg = openrank.iloc[:n//3].mean()
        q2_avg = openrank.iloc[n//3:2*n//3].mean()
        q3_avg = openrank.iloc[2*n//3:].mean()
        recent_slope = np.polyfit(range(6), openrank.tail(6).values, 1)[0]
        
        if q1_avg < q2_avg < q3_avg and recent_slope > 0:
            stage = 'GROWTH'
        elif q2_avg > q1_avg and q3_avg >= q2_avg * 0.85:
            stage = 'MATURITY'
        elif q3_avg < q2_avg * 0.7:
            stage = 'REVIVAL' if recent_slope > 0.5 else 'DECLINE'
        else:
            stage = 'MATURITY' if q3_avg >= q2_avg * 0.9 else 'GROWTH'
        
        return stage, TERMINOLOGY[stage]
    
    # ==================== 生命状态诊断 ====================
    def diagnose_vitality(self) -> Tuple[str, str]:
        """项目生命状态诊断，返回 (状态, 描述)"""
        activity = self.df['activity']
        participants = self.df['participants']
        openrank = self.df['openrank']
        
        recent_activity = activity.tail(6).mean()
        peak_activity = activity.max()
        recent_slope = np.polyfit(range(min(6, len(activity))), activity.tail(6).values, 1)[0]
        
        # 成熟度指数
        maturity_index = (openrank.sum() / 100) * (participants.max() / 50)
        contributor_trend = participants.tail(6).diff().mean()
        
        # 不活跃比例
        if 'inactive_contributors' in self.df.columns:
            inactive_ratio = self.df['inactive_contributors'].tail(3).mean() / (participants.tail(3).mean() + 1)
        else:
            inactive_ratio = 0
        
        if recent_slope > 0:
            return 'THRIVING', TERMINOLOGY['THRIVING']
        
        # 区分成熟稳定与僵尸
        if self.tier in ['GIANT', 'MATURE']:
            if maturity_index > 3 and recent_activity > peak_activity * 0.2:
                return 'STABLE_MATURE', TERMINOLOGY['STABLE_MATURE']
        else:
            if maturity_index > 5 and recent_activity > peak_activity * 0.3:
                return 'STABLE_MATURE', TERMINOLOGY['STABLE_MATURE']
        
        if contributor_trend < -1 and recent_activity < peak_activity * 0.1 and inactive_ratio > 0.5:
            return 'ZOMBIE', TERMINOLOGY['ZOMBIE']
        
        return 'DORMANT', TERMINOLOGY['DORMANT']
    
    # ==================== 健康度计算 ====================
    def calculate_health_metrics(self) -> Dict:
        """计算健康度指标"""
        result = {}
        
        # Issue 债务率
        if 'issues_closed' in self.df.columns and 'issues_new' in self.df.columns:
            closed_ma = self.df['issues_closed'].tail(3).mean()
            new_ma = self.df['issues_new'].tail(3).mean()
            result['debt_ratio'] = closed_ma / (new_ma + 0.1)
        else:
            result['debt_ratio'] = 1.0
        
        # 人均负荷
        if 'activity' in self.df.columns and 'participants' in self.df.columns:
            current_load = self.df['activity'].tail(3).mean() / (self.df['participants'].tail(3).mean() + 1)
            historical_load = self.df['activity'].mean() / (self.df['participants'].mean() + 1)
            result['load_ratio'] = current_load / (historical_load + 0.1)
        else:
            result['load_ratio'] = 1.0
        
        # Bus Factor
        if 'bus_factor' in self.df.columns:
            result['bus_factor'] = self.df['bus_factor'].tail(3).mean()
        else:
            result['bus_factor'] = None
        
        # PR 效率
        if 'pr_merged' in self.df.columns and 'pr_new' in self.df.columns:
            result['pr_efficiency'] = self.df['pr_merged'].tail(6).mean() / (self.df['pr_new'].tail(6).mean() + 0.1)
        else:
            result['pr_efficiency'] = 1.0
        
        # 增长加速度
        if 'openrank' in self.df.columns:
            openrank_diff = self.df['openrank'].diff()
            result['growth_acceleration'] = openrank_diff.diff().tail(6).mean()
        else:
            result['growth_acceleration'] = 0
        
        # 贡献者更新率
        if 'new_contributors' in self.df.columns and 'inactive_contributors' in self.df.columns:
            new_avg = self.df['new_contributors'].tail(6).mean()
            inactive_avg = self.df['inactive_contributors'].tail(6).mean()
            result['contributor_renewal'] = new_avg / (inactive_avg + 0.1)
        else:
            result['contributor_renewal'] = 1.0
        
        return result
    
    # ==================== 改进的 ETD 算法 ====================
    def analyze_etd(self, vitality: str) -> Dict:
        """
        改进的预计寿命分析
        区分：真正衰退 vs 成熟稳定 vs 活力充沛
        """
        result = {
            'etd_months': float('inf'),
            'etd_status': 'HEALTHY',
            'etd_description': '',
            'is_mature_stable': False,
            'confidence': 'HIGH'
        }
        
        if 'activity' not in self.df.columns or len(self.df) < 6:
            result['etd_description'] = '数据不足，无法进行寿命预测'
            result['confidence'] = 'LOW'
            return result
        
        activity = self.df['activity'].tail(12)
        X = np.arange(len(activity)).reshape(-1, 1)
        model = LinearRegression().fit(X, activity.values)
        slope = model.coef_[0]
        current_activity = activity.iloc[-1]
        
        # 情况1：活跃度上升
        if slope >= 0:
            result['etd_months'] = float('inf')
            result['etd_status'] = 'THRIVING'
            result['etd_description'] = '活跃度呈上升趋势，项目生命力强劲，无枯竭风险。'
            return result
        
        # 情况2：成熟稳定项目（虽然下降但属于正常）
        if vitality == 'STABLE_MATURE':
            result['etd_months'] = float('inf')
            result['etd_status'] = 'STABLE'
            result['is_mature_stable'] = True
            result['etd_description'] = (
                f'项目已进入成熟稳定期。虽然活跃度月均下降 {abs(slope):.1f} 点，'
                f'但这是成熟项目的正常特征——功能完善后无需频繁更新。'
                f'当前活跃度 {current_activity:.0f} 仍维持在健康水平。'
            )
            return result
        
        # 情况3：真正的衰退
        if slope < 0 and current_activity > 0:
            etd = -current_activity / slope
            result['etd_months'] = max(0, etd)
            
            if etd < 6:
                result['etd_status'] = 'CRITICAL'
                result['etd_description'] = (
                    f'⚠️ 高危预警：按当前衰减速度（月均 -{abs(slope):.1f}），'
                    f'预计 {etd:.1f} 个月后活跃度将归零。'
                    f'建议立即采取措施激活社区。'
                )
            elif etd < 12:
                result['etd_status'] = 'WARNING'
                result['etd_description'] = (
                    f'⚡ 衰退预警：活跃度呈下降趋势，预计 {etd:.1f} 个月后可能枯竭。'
                    f'建议加强社区运营，发布新功能或 Roadmap 以提振信心。'
                )
            else:
                result['etd_status'] = 'CAUTION'
                result['etd_description'] = (
                    f'📉 温和下降：活跃度有所下滑，预计 {etd:.1f} 个月后可能低迷。'
                    f'目前尚有缓冲时间，建议观察并适时调整运营策略。'
                )
        
        return result
    
    # ==================== 黑马识别 ====================
    def identify_dark_horse(self) -> Dict:
        """黑马项目识别"""
        if self.tier in ['GIANT', 'MATURE']:
            return {
                'is_dark_horse': False,
                'confidence': 0,
                'reason': f'{self.tier} 层级项目已超出黑马范畴（{TERMINOLOGY[self.tier]}）',
                'metrics': {},
                'description': '黑马项目特指处于成长初期但展现出高潜力的项目。成熟或巨型项目已脱离黑马阶段。'
            }
        
        score = 0
        reasons = []
        metrics = {}
        
        # 条件1: 相关性耦合
        if 'stars' in self.df.columns and 'participants' in self.df.columns:
            stars_series = self.df['stars'].tail(12)
            participants_series = self.df['participants'].tail(12)
            if len(stars_series) >= 6:
                corr, p_value = pearsonr(stars_series.values, participants_series.values)
                metrics['correlation'] = corr
                if corr > 0.6 and p_value < 0.05:
                    score += 25
                    reasons.append(f"Star-贡献者强相关 (r={corr:.2f})：关注度有效转化为实际贡献")
                elif corr < 0.3:
                    score -= 10
                    reasons.append(f"⚠️ Star-贡献者弱相关 (r={corr:.2f})：可能存在营销泡沫")
        
        # 条件2: 增长加速度
        if 'openrank' in self.df.columns:
            openrank_diff = self.df['openrank'].diff()
            acceleration = openrank_diff.diff().tail(6).mean()
            metrics['acceleration'] = acceleration
            if acceleration > 0.5:
                score += 25
                reasons.append(f"增长加速明显 (a={acceleration:.2f})：势头正劲")
            elif acceleration > 0:
                score += 10
                reasons.append(f"增长加速中 (a={acceleration:.2f})")
        
        # 条件3: 相对增长率
        if 'openrank' in self.df.columns and len(self.df) >= 12:
            early_avg = self.df['openrank'].head(6).mean() + 0.1
            recent_avg = self.df['openrank'].tail(6).mean()
            relative_growth = (recent_avg - early_avg) / early_avg
            metrics['relative_growth'] = relative_growth
            threshold = self.config.growth_rate
            if relative_growth > threshold * 2:
                score += 25
                reasons.append(f"超高增长率 {relative_growth*100:.0f}%")
            elif relative_growth > threshold:
                score += 15
                reasons.append(f"高增长率 {relative_growth*100:.0f}%")
        
        # 条件4: 新贡献者增速
        if 'new_contributors' in self.df.columns:
            new_contrib = self.df['new_contributors'].tail(6)
            if new_contrib.mean() > new_contrib.head(3).mean():
                score += 15
                reasons.append("新贡献者持续涌入")
        
        is_dark_horse = score >= 55
        description = (
            f"{'🏇 该项目具备黑马潜质！' if is_dark_horse else '该项目暂不符合黑马标准。'}"
            f"综合评分 {score}/100，判定阈值为 55 分。"
        )
        
        return {
            'is_dark_horse': is_dark_horse,
            'confidence': min(score, 100),
            'reasons': reasons,
            'metrics': metrics,
            'description': description
        }
    
    # ==================== 风险评估 ====================
    def assess_risk(self, etd_analysis: Dict) -> Dict:
        """风险评估"""
        result = {
            'risk_level': 'LOW',
            'risk_score': 0,
            'alerts': [],
            'description': ''
        }
        
        alerts = []
        risk_score = 0
        
        # ETD 风险
        if etd_analysis['etd_status'] == 'CRITICAL':
            risk_score += 40
            alerts.append(f"🚨 活跃度濒临枯竭 (ETD: {etd_analysis['etd_months']:.1f}月)")
        elif etd_analysis['etd_status'] == 'WARNING':
            risk_score += 25
            alerts.append(f"⚠️ 活跃度持续下滑 (ETD: {etd_analysis['etd_months']:.1f}月)")
        
        # Bus Factor 风险
        health = self.calculate_health_metrics()
        if health.get('bus_factor') and health['bus_factor'] <= 2:
            risk_score += 25
            alerts.append(f"🚌 Bus Factor 过低 ({health['bus_factor']:.0f})：存在单点失效风险")
        
        # 债务风险
        if health['debt_ratio'] < self.config.debt_healthy * 0.7:
            risk_score += 20
            alerts.append(f"📋 技术债务堆积 (Debt Ratio: {health['debt_ratio']:.2f})")
        
        # 过载风险
        if health['load_ratio'] > self.config.load_factor:
            risk_score += 15
            alerts.append(f"🔥 核心贡献者过载 (负荷比: {health['load_ratio']:.1f}x)")
        
        # 确定风险等级
        if risk_score >= 50:
            result['risk_level'] = 'CRITICAL'
        elif risk_score >= 30:
            result['risk_level'] = 'HIGH'
        elif risk_score >= 15:
            result['risk_level'] = 'MEDIUM'
        else:
            result['risk_level'] = 'LOW'
        
        result['risk_score'] = risk_score
        result['alerts'] = alerts
        result['description'] = self._generate_risk_description(result['risk_level'], alerts)
        
        return result
    
    def _generate_risk_description(self, level: str, alerts: List[str]) -> str:
        """生成风险描述"""
        if level == 'LOW':
            return '风险等级：低。项目运行状态良好，未发现明显风险指标。建议保持当前运营节奏。'
        elif level == 'MEDIUM':
            return f'风险等级：中等。存在 {len(alerts)} 项需要关注的指标，建议定期监控并适时调整。'
        elif level == 'HIGH':
            return f'风险等级：高。发现 {len(alerts)} 项风险指标，建议尽快采取干预措施。'
        else:
            return f'风险等级：危急！存在 {len(alerts)} 项严重风险，项目可能面临重大挑战，需立即行动。'
    
    # ==================== 综合评分 ====================
    def calculate_composite_score(self, health: Dict, vitality: str, risk: Dict) -> Dict:
        """综合健康评分"""
        weights = {
            'GIANT': {'债务管理': 0.30, '稳定性': 0.25, '持续性': 0.25, '风险': 0.20},
            'MATURE': {'债务管理': 0.25, '效率': 0.25, '稳定性': 0.25, '风险': 0.25},
            'GROWING': {'增长力': 0.30, '转化率': 0.25, '债务管理': 0.25, '风险': 0.20},
            'EMERGING': {'动力': 0.35, '热度': 0.25, '转化率': 0.20, '风险': 0.20}
        }
        w = weights[self.tier]
        
        scores = {}
        
        # 通用
        scores['债务管理'] = min(health['debt_ratio'] / 1.5 * 100, 100)
        scores['风险'] = max(0, 100 - risk['risk_score'] * 2)
        
        # 层级特定
        if self.tier == 'GIANT':
            scores['稳定性'] = 80 if vitality in ['THRIVING', 'STABLE_MATURE'] else 50
            scores['持续性'] = min(health.get('contributor_renewal', 0.5) * 100, 100)
        elif self.tier == 'MATURE':
            scores['效率'] = min(health.get('pr_efficiency', 0.5) * 100, 100)
            scores['稳定性'] = 80 if vitality in ['THRIVING', 'STABLE_MATURE'] else 50
        elif self.tier == 'GROWING':
            scores['增长力'] = min(50 + health.get('growth_acceleration', 0) * 50, 100)
            scores['转化率'] = 70  # 默认中等
        else:
            scores['动力'] = 70
            scores['热度'] = 60
            scores['转化率'] = 60
        
        # 计算加权总分
        composite = sum(w.get(k, 0) * scores.get(k, 50) for k in w.keys())
        
        # 状态调整
        if vitality == 'STABLE_MATURE':
            composite = max(composite, 65)
        elif vitality == 'ZOMBIE':
            composite = min(composite, 35)
        elif vitality == 'THRIVING':
            composite = min(composite + 10, 100)
        
        return {
            'score': round(composite, 1),
            'grade': self._score_to_grade(composite),
            'breakdown': {k: round(scores.get(k, 50), 1) for k in w.keys()}
        }
    
    def _score_to_grade(self, score: float) -> str:
        if score >= 85: return 'A+ (卓越)'
        elif score >= 75: return 'A (优秀)'
        elif score >= 65: return 'B+ (良好)'
        elif score >= 55: return 'B (中等)'
        elif score >= 45: return 'C (需关注)'
        elif score >= 35: return 'D (风险)'
        else: return 'F (危机)'
    
    # ==================== 建议生成 ====================
    def generate_recommendations(self, health: Dict, vitality: str, dark_horse: Dict, risk: Dict) -> List[str]:
        """生成针对性建议"""
        recs = []
        
        if health['load_ratio'] > 2:
            recs.append("🎯 人均负荷过高：建议标记 'Good First Issue' 以吸引新贡献者分担工作")
        
        if health['debt_ratio'] < 0.8:
            recs.append("📋 Issue 积压严重：建议组织 Bug Bash 活动集中处理，或引入 Issue 分类机器人")
        
        if risk['risk_level'] in ['HIGH', 'CRITICAL']:
            recs.append("⚡ 风险等级较高：建议通过定期直播、技术博客保持项目曝光，激活社区")
        
        if vitality == 'DORMANT':
            recs.append("💤 项目处于休眠：建议发布 Roadmap 或新版本预告，重新激活社区期待")
        
        if vitality == 'STABLE_MATURE':
            recs.append("🏆 项目已成熟：保持定期安全更新和 Bug 修复即可，无需追求高活跃度")
        
        if dark_horse.get('is_dark_horse'):
            recs.append("🚀 黑马潜力显现：建议加大推广投入，抓住增长窗口扩大影响力")
        
        if health.get('bus_factor') and health['bus_factor'] <= 2:
            recs.append("🚌 Bus Factor 过低：建议培养更多核心贡献者，降低单点依赖风险")
        
        if not recs:
            recs.append("✅ 项目状态健康，保持当前运营节奏即可")
        
        return recs
    
    # ==================== 详细总结生成 ====================
    def generate_detailed_summary(self, diagnosis) -> str:
        """生成详细的文字总结"""
        summary = f"""
═══════════════════════════════════════════════════════════════════════════════
                    {self.org}/{self.repo} 深度诊断报告
═══════════════════════════════════════════════════════════════════════════════

【项目基本画像】
• 项目层级：{diagnosis.tier} — {diagnosis.tier_desc}
• 生命周期：{diagnosis.lifecycle} — {diagnosis.lifecycle_desc}
• 当前状态：{diagnosis.vitality} — {diagnosis.vitality_desc}

【健康度评估】
• 综合评分：{diagnosis.health_score}/100 分，评级为 {diagnosis.health_grade}
• 各维度得分：
"""
        for k, v in diagnosis.health_breakdown.items():
            summary += f"  - {k}：{v}/100\n"
        
        summary += f"""
【预期寿命分析】
{diagnosis.etd_analysis['etd_description']}

【黑马潜力分析】
{diagnosis.dark_horse['description']}
"""
        if diagnosis.dark_horse.get('reasons'):
            summary += "• 判定依据：\n"
            for r in diagnosis.dark_horse['reasons']:
                summary += f"  - {r}\n"
        
        summary += f"""
【风险评估】
{diagnosis.risk_assessment['description']}
"""
        if diagnosis.risk_assessment['alerts']:
            summary += "• 风险警报：\n"
            for a in diagnosis.risk_assessment['alerts']:
                summary += f"  - {a}\n"
        
        if diagnosis.pathology_labels:
            summary += "\n【病理标签】\n"
            for label in diagnosis.pathology_labels:
                summary += f"• {label}\n"
        
        summary += "\n【改进建议】\n"
        for i, rec in enumerate(diagnosis.recommendations, 1):
            summary += f"{i}. {rec}\n"
        
        summary += """
═══════════════════════════════════════════════════════════════════════════════
                              术语解释
═══════════════════════════════════════════════════════════════════════════════
"""
        for term in ['ETD', 'Bus Factor', 'Debt Ratio']:
            summary += f"• {term}：{TERMINOLOGY[term]}\n"
        
        summary += "═══════════════════════════════════════════════════════════════════════════════\n"
        
        return summary
    
    # ==================== 可视化 ====================
    def plot_comprehensive_charts(self, diagnosis: DiagnosisResult):
        """绘制优化后的综合图表"""
        fig = plt.figure(figsize=(20, 16))
        
        # 使用更紧凑的布局
        gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.35,
                             left=0.05, right=0.95, top=0.92, bottom=0.05)
        
        # 第一行
        ax1 = fig.add_subplot(gs[0, 0], projection='polar')
        self._plot_radar(ax1, diagnosis.health_breakdown, diagnosis.health_grade)
        
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_openrank_trend(ax2)
        
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_correlation(ax3, diagnosis.dark_horse)
        
        # 第二行
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_activity_prediction(ax4, diagnosis.etd_analysis)
        
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_contributor_flow(ax5)
        
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_debt_ratio(ax6)
        
        # 第三行
        ax7 = fig.add_subplot(gs[2, 0])
        self._plot_lifecycle_stage(ax7, diagnosis.lifecycle)
        
        ax8 = fig.add_subplot(gs[2, 1])
        self._plot_health_gauge(ax8, diagnosis.health_score, diagnosis.health_grade)
        
        ax9 = fig.add_subplot(gs[2, 2])
        self._plot_summary_text(ax9, diagnosis)
        
        # 总标题
        fig.suptitle(
            f"📊 {self.org}/{self.repo} 深度诊断报告\n"
            f"层级: {diagnosis.tier} | 状态: {diagnosis.vitality} | 评级: {diagnosis.health_grade}",
            fontsize=16, fontweight='bold', y=0.98
        )
        
        # 保存
        img_path = f"{self.org}_{self.repo}_diagnosis_v2.png"
        plt.savefig(img_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"🖼️ 诊断图表已保存至: {img_path}")
        plt.show()
    
    def _plot_radar(self, ax, breakdown: Dict, grade: str):
        """雷达图"""
        categories = list(breakdown.keys())
        values = list(breakdown.values())
        
        if not categories:
            ax.text(0.5, 0.5, "数据不足", ha='center', va='center')
            return
        
        values += values[:1]
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, color='#2E86AB')
        ax.fill(angles, values, alpha=0.25, color='#2E86AB')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=9)
        ax.set_ylim(0, 100)
        ax.set_title(f"多维健康评估\n综合: {grade}", fontsize=11, pad=15)
    
    def _plot_openrank_trend(self, ax):
        """OpenRank 趋势"""
        if 'openrank' not in self.df.columns:
            ax.text(0.5, 0.5, "OpenRank 数据不可用", ha='center', va='center', transform=ax.transAxes)
            return
        
        openrank = self.df['openrank']
        ax.plot(self.df.index, openrank, 'b-', lw=2, label='OpenRank')
        
        # 趋势线
        z = np.polyfit(range(len(openrank)), openrank.values, 1)
        p = np.poly1d(z)
        ax.plot(self.df.index, p(range(len(openrank))), 'r--', lw=1.5, label='趋势线')
        
        ax.set_title("OpenRank 趋势", fontsize=11)
        ax.legend(loc='best', fontsize=8)
        ax.tick_params(axis='x', rotation=30, labelsize=8)
        ax.set_ylabel("OpenRank")
    
    def _plot_correlation(self, ax, dark_horse: Dict):
        """Star-贡献者相关性"""
        if 'stars' not in self.df.columns or 'participants' not in self.df.columns:
            ax.text(0.5, 0.5, "数据不可用", ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Star-贡献者耦合", fontsize=11)
            return
        
        data = self.df[['stars', 'participants']].tail(12).dropna()
        if len(data) < 3:
            ax.text(0.5, 0.5, "数据点不足", ha='center', va='center', transform=ax.transAxes)
            return
        
        sns.regplot(x='stars', y='participants', data=data, ax=ax,
                   scatter_kws={'alpha': 0.6, 's': 50}, line_kws={'color': 'blue'})
        
        corr = dark_horse.get('metrics', {}).get('correlation', None)
        corr_text = f"r={corr:.2f}" if corr else "r=N/A"
        
        ax.set_title(f"Star-贡献者耦合验证\n({corr_text})", fontsize=11)
        ax.set_xlabel("新增 Star", fontsize=9)
        ax.set_ylabel("贡献者数", fontsize=9)
    
    def _plot_activity_prediction(self, ax, etd: Dict):
        """活跃度预测"""
        if 'activity' not in self.df.columns:
            ax.text(0.5, 0.5, "Activity 数据不可用", ha='center', va='center', transform=ax.transAxes)
            return
        
        activity = self.df['activity'].tail(12)
        x = np.arange(len(activity))
        
        ax.scatter(x, activity.values, color='#2E86AB', s=50, label='历史活跃度', zorder=5)
        
        # 回归线
        z = np.polyfit(x, activity.values, 1)
        p = np.poly1d(z)
        
        # 根据 ETD 状态选择颜色
        color = {'CRITICAL': 'red', 'WARNING': 'orange', 'CAUTION': 'gold'}.get(etd['etd_status'], 'green')
        
        if etd['etd_months'] < float('inf') and etd['etd_months'] < 24:
            ext_x = np.array([0, len(x) + int(etd['etd_months'])])
            ax.plot(ext_x, p(ext_x), '--', color=color, lw=2, label=f"预测 (ETD: {etd['etd_months']:.1f}月)")
        else:
            ax.plot(x, p(x), '--', color=color, lw=2, label='趋势线')
        
        ax.axhline(0, color='gray', lw=1, linestyle=':')
        ax.set_ylim(bottom=0)
        ax.set_xlabel("月份 (最近12个月)", fontsize=9)
        ax.set_ylabel("Activity", fontsize=9)
        ax.set_title(f"活跃度预测\n状态: {etd['etd_status']}", fontsize=11)
        ax.legend(fontsize=8)
    
    def _plot_contributor_flow(self, ax):
        """贡献者流动"""
        new_col = 'new_contributors' if 'new_contributors' in self.df.columns else None
        inactive_col = 'inactive_contributors' if 'inactive_contributors' in self.df.columns else None
        
        if not new_col and not inactive_col:
            ax.text(0.5, 0.5, "贡献者数据不可用", ha='center', va='center', transform=ax.transAxes)
            ax.set_title("贡献者流动", fontsize=11)
            return
        
        data = self.df.tail(12)
        x = range(len(data))
        
        if new_col:
            ax.bar([i-0.2 for i in x], data[new_col].values, width=0.4, 
                   color='#2ecc71', alpha=0.7, label='新增贡献者')
        if inactive_col:
            ax.bar([i+0.2 for i in x], data[inactive_col].values, width=0.4,
                   color='#e74c3c', alpha=0.7, label='流失贡献者')
        
        ax.set_title("贡献者流动对比", fontsize=11)
        ax.legend(fontsize=8)
        ax.set_xlabel("月份", fontsize=9)
        ax.set_ylabel("人数", fontsize=9)
    
    def _plot_debt_ratio(self, ax):
        """债务率"""
        if 'issues_closed' not in self.df.columns or 'issues_new' not in self.df.columns:
            ax.text(0.5, 0.5, "Issue 数据不可用", ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Issue 债务率", fontsize=11)
            return
        
        debt = self.df['issues_closed'] / (self.df['issues_new'] + 0.1)
        debt = debt.tail(12)
        
        colors = ['#2ecc71' if v >= 1 else '#f39c12' if v >= 0.8 else '#e74c3c' for v in debt.values]
        ax.bar(range(len(debt)), debt.values, color=colors, alpha=0.7)
        ax.axhline(1.0, color='green', linestyle='--', lw=1.5, label='健康线 (1.0)')
        ax.axhline(0.8, color='orange', linestyle='--', lw=1.5, label='警戒线 (0.8)')
        
        ax.set_title("Issue 债务率趋势", fontsize=11)
        ax.set_xlabel("月份", fontsize=9)
        ax.set_ylabel("Debt Ratio", fontsize=9)
        ax.legend(fontsize=8, loc='upper right')
    
    def _plot_lifecycle_stage(self, ax, lifecycle: str):
        """生命周期阶段"""
        stages = ['INCUBATION', 'GROWTH', 'MATURITY', 'DECLINE', 'REVIVAL']
        stage_names = ['孵化期', '成长期', '成熟期', '衰退期', '复苏期']
        colors = ['#3498db', '#2ecc71', '#f1c40f', '#e74c3c', '#9b59b6']
        
        current_idx = stages.index(lifecycle) if lifecycle in stages else 2
        
        # 横向进度条形式
        ax.barh(stage_names, [1]*5, color='lightgray', alpha=0.3)
        ax.barh(stage_names[current_idx], 1, color=colors[current_idx], alpha=0.8)
        
        ax.set_xlim(0, 1.2)
        ax.set_title(f"生命周期阶段: {stage_names[current_idx]}", fontsize=11)
        ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        
        # 添加箭头指示
        ax.annotate('◀ 当前', xy=(1.05, current_idx), fontsize=10, color=colors[current_idx])
    
    def _plot_health_gauge(self, ax, score: float, grade: str):
        """健康度仪表盘"""
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        
        # 绘制弧形仪表
        theta = np.linspace(0, np.pi, 100)
        r = 40
        x = 50 + r * np.cos(theta)
        y = 20 + r * np.sin(theta)
        
        # 背景弧
        ax.plot(x, y, 'lightgray', lw=15, solid_capstyle='round')
        
        # 得分弧 (根据分数计算长度)
        score_ratio = score / 100
        theta_score = np.linspace(0, np.pi * score_ratio, int(100 * score_ratio))
        x_score = 50 + r * np.cos(theta_score)
        y_score = 20 + r * np.sin(theta_score)
        
        # 颜色根据分数
        if score >= 75:
            color = '#2ecc71'
        elif score >= 55:
            color = '#f1c40f'
        elif score >= 35:
            color = '#f39c12'
        else:
            color = '#e74c3c'
        
        ax.plot(x_score, y_score, color, lw=15, solid_capstyle='round')
        
        # 分数文字
        ax.text(50, 35, f"{score:.0f}", ha='center', va='center', fontsize=28, fontweight='bold')
        ax.text(50, 18, grade, ha='center', va='center', fontsize=12)
        ax.text(50, 75, "健康评分", ha='center', va='center', fontsize=12)
        
        ax.axis('off')
        ax.set_aspect('equal')
    
    def _plot_summary_text(self, ax, diagnosis: DiagnosisResult):
        """诊断摘要文本"""
        ax.axis('off')
        
        # 构建摘要文本
        summary_lines = [
            f"📋 诊断摘要",
            f"─" * 30,
            f"",
            f"🏷️ 层级: {diagnosis.tier}",
            f"   {diagnosis.tier_desc[:20]}...",
            f"",
            f"🔄 周期: {diagnosis.lifecycle}",
            f"   {diagnosis.lifecycle_desc[:20]}...",
            f"",
            f"💓 状态: {diagnosis.vitality}",
            f"   {diagnosis.vitality_desc[:20]}...",
            f"",
            f"─" * 30,
            f"",
        ]
        
        # 添加关键标签
        if diagnosis.pathology_labels:
            summary_lines.append("⚠️ 风险标签:")
            for label in diagnosis.pathology_labels[:2]:
                short_label = label[:25] + "..." if len(label) > 25 else label
                summary_lines.append(f"  • {short_label}")
        else:
            summary_lines.append("✅ 无异常标签")
        
        summary_lines.append("")
        summary_lines.append(f"─" * 30)
        summary_lines.append(f"📊 风险等级: {diagnosis.risk_assessment['risk_level']}")
        
        text = "\n".join(summary_lines)
        
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', 
                        edgecolor='#dee2e6', alpha=0.9))
    
    # ==================== 病理标签 ====================
    def generate_pathology_labels(self, health: Dict, vitality: str, risk: Dict) -> List[str]:
        """生成病理标签"""
        labels = []
        
        if health.get('bus_factor') and health['bus_factor'] <= 2:
            labels.append(f"Bus Factor = {health['bus_factor']:.0f} (单点失效风险)")
        
        if health['debt_ratio'] < self.config.debt_healthy * 0.7:
            labels.append(f"Debt Ratio = {health['debt_ratio']:.2f} (维护效率不足)")
        
        if health['load_ratio'] > self.config.load_factor:
            labels.append(f"Load = {health['load_ratio']:.1f}x (核心过载)")
        
        if vitality == 'ZOMBIE':
            labels.append("状态: ZOMBIE (僵尸项目)")
        
        return labels
    
    # ==================== 保存报告 ====================
    def _save_report(self, diagnosis: DiagnosisResult):
        """保存报告到 CSV"""
        summary = {
            "项目": [f"{self.org}/{self.repo}"],
            "层级": [diagnosis.tier],
            "层级说明": [diagnosis.tier_desc],
            "生命周期": [diagnosis.lifecycle],
            "周期说明": [diagnosis.lifecycle_desc],
            "生命状态": [diagnosis.vitality],
            "状态说明": [diagnosis.vitality_desc],
            "健康评分": [diagnosis.health_score],
            "评级": [diagnosis.health_grade],
            "风险等级": [diagnosis.risk_assessment['risk_level']],
            "ETD状态": [diagnosis.etd_analysis['etd_status']],
            "ETD月数": [diagnosis.etd_analysis['etd_months'] if diagnosis.etd_analysis['etd_months'] < float('inf') else 'N/A'],
            "是否黑马": [diagnosis.dark_horse['is_dark_horse']],
            "黑马置信度": [diagnosis.dark_horse['confidence']],
            "病理标签": ['; '.join(diagnosis.pathology_labels)],
            "建议": ['; '.join(diagnosis.recommendations)]
        }
        
        output_file = f"{self.org}_{self.repo}_diagnosis_v2.csv"
        pd.DataFrame(summary).to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"💾 诊断报告已导出至: {output_file}")
        
        # 保存详细文字报告
        txt_file = f"{self.org}_{self.repo}_detailed_report.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(diagnosis.detailed_summary)
        print(f"📝 详细报告已导出至: {txt_file}")
    
    # ==================== 主流程 ====================
    def run(self) -> Optional[DiagnosisResult]:
        """执行完整诊断流程"""
        if not self.fetch_data():
            return None
        
        # 各模块计算
        lifecycle, lifecycle_desc = self.identify_lifecycle()
        vitality, vitality_desc = self.diagnose_vitality()
        health = self.calculate_health_metrics()
        etd_analysis = self.analyze_etd(vitality)
        dark_horse = self.identify_dark_horse()
        risk = self.assess_risk(etd_analysis)
        composite = self.calculate_composite_score(health, vitality, risk)
        pathology = self.generate_pathology_labels(health, vitality, risk)
        recommendations = self.generate_recommendations(health, vitality, dark_horse, risk)
        
        # 构建诊断结果
        diagnosis = DiagnosisResult(
            tier=self.tier,
            tier_desc=TERMINOLOGY[self.tier],
            lifecycle=lifecycle,
            lifecycle_desc=lifecycle_desc,
            vitality=vitality,
            vitality_desc=vitality_desc,
            health_score=composite['score'],
            health_grade=composite['grade'],
            health_breakdown=composite['breakdown'],
            dark_horse=dark_horse,
            risk_assessment=risk,
            etd_analysis=etd_analysis,
            pathology_labels=pathology,
            recommendations=recommendations,
            detailed_summary=""
        )
        
        # 生成详细总结
        diagnosis.detailed_summary = self.generate_detailed_summary(diagnosis)
        
        # 输出
        print(diagnosis.detailed_summary)
        self.plot_comprehensive_charts(diagnosis)
        self._save_report(diagnosis)
        
        return diagnosis


# ==================== 入口 ====================
if __name__ == "__main__":
    url = input("请输入待评估的 GitHub 网址: ").strip()
    analyzer = ProjectAnalyzerV2(url)
    analyzer.run()
