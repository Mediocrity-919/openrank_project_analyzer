# compare.py
"""
GitHub项目对比分析器 - 基于v4.py组件
对比两个项目的健康程度、发展趋势和潜力
"""

import sys
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import re
import os
from datetime import datetime
import json

# 导入v4.py中的组件
try:
    from v4 import (
        GMMTierClassifier, MomentumCalculator, ResistanceCalculator, 
        PotentialCalculator, ProphetPredictor, AHPHealthEvaluator,
        BacktestValidator, BusFactorCalculator, ChangePointDetector,
        ConversionFunnelAnalyzer, ETDAnalyzer, GitHubAPIAnalyzer,
        TIER_BENCHMARKS, TIER_NAMES, COLORS
    )
except ImportError as e:
    print(f"导入v4.py组件失败: {e}")
    print("请确保v4.py文件存在于同一目录下")
    sys.exit(1)
except Exception as e:
    print(f"导入组件时发生未知错误: {e}")
    sys.exit(1)

# 设置matplotlib配置，避免重复key警告
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8

class ProjectDataFetcher:
    """从OpenDigger获取项目数据"""
    
    CORE_METRICS = [
        "openrank", "activity", "stars", "attention",
        "participants", "new_contributors", "inactive_contributors",
        "bus_factor", "issues_new", "issues_closed", "pr_new", "pr_merged"
    ]
    
    def __init__(self, url: str):
        self.org, self.repo = self._parse_url(url)
        self.df = pd.DataFrame()
        
    def _parse_url(self, url: str):
        match = re.search(r"github\.com/([^/]+)/([^/]+)", url)
        if match:
            return match.group(1), match.group(2).replace(".git", "")
        if "/" in url and "http" not in url:
            parts = url.split('/')
            return parts[0], parts[1]
        raise ValueError("无效的 GitHub URL")
    
    def fetch_data(self) -> bool:
        """从OpenDigger获取数据"""
        # print(f"获取项目数据: {self.org}/{self.repo}")
        
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
            except Exception as e:
                # print(f"获取指标 {metric} 失败: {e}")
                continue
        
        if not raw_data:
            # print("无法获取数据")
            return False
        
        self.df = pd.DataFrame(raw_data).fillna(0)
        self.df.index = pd.to_datetime(self.df.index)
        self.df = self.df.sort_index()
        
        # print(f"成功获取 {len(self.df)} 个月数据")
        return True
    
    def get_tier_classification(self):
        """使用GMM进行项目分层"""
        if self.df.empty:
            return None, {}, 0
        
        classifier = GMMTierClassifier()
        classifier.fit()
        
        metrics = {
            'avg_openrank': self.df['openrank'].mean() if 'openrank' in self.df else 0,
            'total_stars': self.df['stars'].sum() if 'stars' in self.df else 0,
            'max_participants': self.df['participants'].max() if 'participants' in self.df else 0
        }
        
        tier, probabilities, confidence = classifier.predict_proba(metrics)
        return tier, probabilities, confidence
    
    def analyze_vitality(self):
        """分析项目活力状态"""
        if 'activity' not in self.df or self.df.empty:
            return 'UNKNOWN'
        
        activity = self.df['activity']
        if len(activity) < 6:
            return 'UNKNOWN'
        
        from scipy.stats import linregress
        recent = activity.tail(6)
        slope = linregress(range(len(recent)), recent.values).slope
        peak, current = activity.max(), recent.mean()
        
        if slope > 0.5:
            return 'THRIVING'
        elif slope > 0:
            return 'STABLE'
        elif current > peak * 0.3:
            return 'DORMANT'
        else:
            return 'ZOMBIE'


class ProjectComparativeAnalyzer:
    """项目对比分析器"""
    
    def __init__(self, project1_url: str, project2_url: str):
        self.project1 = ProjectDataFetcher(project1_url)
        self.project2 = ProjectDataFetcher(project2_url)
        self.project1_data = None
        self.project2_data = None
        self.project1_analysis = {}
        self.project2_analysis = {}
        
    def run_analysis(self):
        """执行完整分析"""
        # print("="*60)
        # print("GitHub项目对比分析器")
        # print("="*60)
        
        # 获取数据
        if not self.project1.fetch_data():
            # print("项目1数据获取失败")
            return False
        if not self.project2.fetch_data():
            # print("项目2数据获取失败")
            return False
        
        self.project1_data = self.project1.df
        self.project2_data = self.project2.df
        
        # 分析项目1
        # print(f"\n分析项目1: {self.project1.org}/{self.project1.repo}")
        self.project1_analysis = self._analyze_project(self.project1)
        
        # 分析项目2
        # print(f"\n分析项目2: {self.project2.org}/{self.project2.repo}")
        self.project2_analysis = self._analyze_project(self.project2)
        
        # 计算潜力评分
        self.project1_analysis['potential_score'] = self._calculate_potential_score(self.project1_analysis)
        self.project2_analysis['potential_score'] = self._calculate_potential_score(self.project2_analysis)
        
        # 生成报告
        self.generate_comparative_report()
        
        # 绘制图表
        self.plot_comparative_dashboard()
        
        return True
    
    def _analyze_project(self, project_fetcher):
        """分析单个项目"""
        analysis = {}
        
        # 基础信息
        analysis['project_name'] = f"{project_fetcher.org}/{project_fetcher.repo}"
        
        # GMM分层
        tier, probabilities, confidence = project_fetcher.get_tier_classification()
        analysis['tier'] = tier
        analysis['tier_probabilities'] = probabilities
        analysis['tier_confidence'] = confidence
        
        # 活力状态
        analysis['vitality'] = project_fetcher.analyze_vitality()
        
        # 动量分析
        momentum_calc = MomentumCalculator()
        analysis['momentum'] = momentum_calc.calculate(project_fetcher.df)
        
        # 阻力分析
        resistance_calc = ResistanceCalculator()
        analysis['resistance'] = resistance_calc.calculate(project_fetcher.df)
        
        # 潜力分析
        potential_calc = PotentialCalculator()
        analysis['potential'] = potential_calc.calculate(project_fetcher.df, tier)
        
        # 巴士系数
        bus_factor_calc = BusFactorCalculator()
        analysis['bus_factor'] = bus_factor_calc.calculate(project_fetcher.df)
        
        # 转化漏斗
        funnel_analyzer = ConversionFunnelAnalyzer()
        analysis['funnel'] = funnel_analyzer.analyze(project_fetcher.df)
        
        # ETD寿命分析
        etd_analyzer = ETDAnalyzer()
        analysis['etd'] = etd_analyzer.analyze(project_fetcher.df, analysis['vitality'], tier)
        
        # 健康评分 (AHP)
        ahp_evaluator = AHPHealthEvaluator()
        trend_3d = {
            'momentum': analysis['momentum'],
            'resistance': analysis['resistance'],
            'potential': analysis['potential']
        }
        
        # 计算风险评分（简化版）
        risk_score = self._calculate_risk_score(analysis)
        analysis['risk'] = {'score': risk_score}
        
        # 生成预测
        predictor = ProphetPredictor()
        predictions = {}
        for metric in ['openrank', 'attention', 'stars']:
            if metric in project_fetcher.df:
                pred_result = predictor.predict(project_fetcher.df, metric, periods=6)
                predictions[metric] = pred_result
        analysis['predictions'] = predictions
        
        # 健康评分
        health_score, dimension_scores = ahp_evaluator.calculate_health_score(
            analysis['vitality'], trend_3d, analysis['risk'], tier, predictions
        )
        analysis['health_score'] = health_score
        analysis['dimension_scores'] = dimension_scores
        
        # 确定健康等级
        grades = [(85, 'A+'), (75, 'A'), (65, 'B+'), (55, 'B'), (45, 'C'), (35, 'D'), (0, 'F')]
        analysis['health_grade'] = next(g for t, g in grades if health_score >= t)
        
        return analysis
    
    def _calculate_risk_score(self, analysis):
        """计算风险评分"""
        risk = 0
        
        if analysis['vitality'] == 'ZOMBIE':
            risk += 40
        elif analysis['vitality'] == 'DORMANT':
            risk += 20
        
        if analysis['resistance']['total'] > 70:
            risk += 30
        elif analysis['resistance']['total'] > 50:
            risk += 15
        
        if analysis['bus_factor']['risk_level'] in ['CRITICAL', 'HIGH']:
            risk += 20
        
        return min(100, risk)
    
    def _calculate_potential_score(self, analysis):
        """
        计算综合潜力评分
        权重：动量40% + 潜力30% + 稳定性20% + 巴士系数10%
        """
        score = 0
        
        # 动量贡献 (40%)
        momentum = analysis['momentum']['total']
        score += momentum * 0.4
        
        # 潜力贡献 (30%)
        potential = analysis['potential']['remaining_space']
        score += potential * 0.3
        
        # 稳定性贡献 (20%) - 抵抗力的反面
        stability = 100 - analysis['resistance']['total']
        score += stability * 0.2
        
        # 巴士系数贡献 (10%)
        bus_factor_score = 100
        risk_map = {'CRITICAL': 20, 'HIGH': 40, 'MEDIUM': 70, 'LOW': 100}
        risk_level = analysis['bus_factor']['risk_level']
        if risk_level in risk_map:
            bus_factor_score = risk_map[risk_level]
        score += bus_factor_score * 0.1
        
        return round(score, 1)
    
    def generate_comparative_report(self):
        """生成对比分析报告"""
        # 使用更简单的表格格式
        project1_name = self.project1_analysis['project_name']
        project2_name = self.project2_analysis['project_name']
        
        report = f"""
{'='*80}
                    GitHub项目对比分析报告
                      生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}

【项目基本信息】
-------------------------------------------------------------------------------
项目: {project1_name}
层级: {self.project1_analysis['tier']} ({TIER_NAMES.get(self.project1_analysis['tier'], '未知')})
健康评分: {self.project1_analysis['health_score']:.1f}/100 ({self.project1_analysis['health_grade']})
潜力评分: {self.project1_analysis['potential_score']:.1f}/100
活力状态: {self.project1_analysis['vitality']}

-------------------------------------------------------------------------------
项目: {project2_name}
层级: {self.project2_analysis['tier']} ({TIER_NAMES.get(self.project2_analysis['tier'], '未知')})
健康评分: {self.project2_analysis['health_score']:.1f}/100 ({self.project2_analysis['health_grade']})
潜力评分: {self.project2_analysis['potential_score']:.1f}/100
活力状态: {self.project2_analysis['vitality']}

{'─'*80}

【详细对比分析】

1. 动量对比
   • 项目1: {self.project1_analysis['momentum']['total']:.1f} - {self.project1_analysis['momentum']['interpretation']}
   • 项目2: {self.project2_analysis['momentum']['total']:.1f} - {self.project2_analysis['momentum']['interpretation']}
   • 差异: {self.project1_analysis['momentum']['total'] - self.project2_analysis['momentum']['total']:+.1f}

2. 阻力对比
   • 项目1: {self.project1_analysis['resistance']['total']:.1f} - {self.project1_analysis['resistance']['interpretation']}
   • 项目2: {self.project2_analysis['resistance']['total']:.1f} - {self.project2_analysis['resistance']['interpretation']}
   • 差异: {self.project1_analysis['resistance']['total'] - self.project2_analysis['resistance']['total']:+.1f}

3. 增长潜力对比
   • 项目1: {self.project1_analysis['potential']['remaining_space']:.1f}% - {self.project1_analysis['potential']['interpretation']}
   • 项目2: {self.project2_analysis['potential']['remaining_space']:.1f}% - {self.project2_analysis['potential']['interpretation']}
   • 差异: {self.project1_analysis['potential']['remaining_space'] - self.project2_analysis['potential']['remaining_space']:+.1f}%

4. 巴士系数对比
   • 项目1: {self.project1_analysis['bus_factor']['effective_bus_factor']:.1f} - {self.project1_analysis['bus_factor']['description']}
   • 项目2: {self.project2_analysis['bus_factor']['effective_bus_factor']:.1f} - {self.project2_analysis['bus_factor']['description']}

5. 寿命预测对比
   • 项目1: {self._format_etd(self.project1_analysis['etd'])}
   • 项目2: {self._format_etd(self.project2_analysis['etd'])}

{'─'*80}

【综合评估与建议】

"""
        
        # 健康评分对比
        health_diff = self.project1_analysis['health_score'] - self.project2_analysis['health_score']
        if abs(health_diff) < 5:
            report += "健康程度相近，各有优势。\n"
        elif health_diff > 0:
            report += f"项目1健康程度更高，领先{health_diff:.1f}分。\n"
        else:
            report += f"项目2健康程度更高，领先{-health_diff:.1f}分。\n"
        
        # 潜力评分对比
        potential_diff = self.project1_analysis['potential_score'] - self.project2_analysis['potential_score']
        if abs(potential_diff) < 5:
            report += "增长潜力相近。\n"
        elif potential_diff > 0:
            report += f"项目1增长潜力更大，领先{potential_diff:.1f}分。\n"
        else:
            report += f"项目2增长潜力更大，领先{-potential_diff:.1f}分。\n"
        
        # 风险对比
        risk_diff = self.project1_analysis['risk']['score'] - self.project2_analysis['risk']['score']
        if abs(risk_diff) < 10:
            report += "风险水平相近。\n"
        elif risk_diff > 0:
            report += f"项目1风险较高，建议关注。\n"
        else:
            report += f"项目2风险较高，建议关注。\n"
        
        # 添加GMM分层概率信息
        report += f"""
{'─'*80}

【GMM分层概率分布】
项目1 ({project1_name}):
"""
        for tier, prob in self.project1_analysis['tier_probabilities'].items():
            bar_length = int(prob * 20)
            bar = '█' * bar_length + '░' * (20 - bar_length)
            report += f"  {tier:10s} {bar} {prob:.1%}\n"
        
        report += f"""
项目2 ({project2_name}):
"""
        for tier, prob in self.project2_analysis['tier_probabilities'].items():
            bar_length = int(prob * 20)
            bar = '█' * bar_length + '░' * (20 - bar_length)
            report += f"  {tier:10s} {bar} {prob:.1%}\n"
        
        report += f"""
{'='*80}
"""
        
        # 保存报告
        filename = f"compare_{self.project1.org}_{self.project1.repo}_vs_{self.project2.org}_{self.project2.repo}_report.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # print(f"对比报告已保存: {filename}")
        # 注释掉直接打印报告，避免Windows终端编码问题
        # print(report)
        
        return report
    
    def _format_etd(self, etd_analysis):
        """格式化ETD显示"""
        if etd_analysis['etd_months'] == float('inf'):
            return "无限 (健康)"
        else:
            return f"{etd_analysis['etd_months']:.1f}个月 ({etd_analysis['etd_status']})"
    
    def plot_comparative_dashboard(self):
        """绘制对比仪表盘 (九宫格)"""
        fig = plt.figure(figsize=(18, 12))
        fig.patch.set_facecolor('white')
        
        # 使用 GridSpec 创建3x3布局
        gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3,
                     left=0.05, right=0.95, top=0.92, bottom=0.05)
        
        # 创建子图 - 修复极坐标子图的创建
        ax1 = fig.add_subplot(gs[0, 0])  # OpenRank对比
        ax2 = fig.add_subplot(gs[0, 1])  # Attention对比
        ax3 = fig.add_subplot(gs[0, 2])  # Stars对比
        ax4 = fig.add_subplot(gs[1, 0])  # 项目1健康仪表盘
        ax5 = fig.add_subplot(gs[1, 1])  # 项目1增长潜力
        ax6 = fig.add_subplot(gs[1, 2], projection='polar')  # 项目1三维趋势（极坐标）
        ax7 = fig.add_subplot(gs[2, 0])  # 项目2健康仪表盘
        ax8 = fig.add_subplot(gs[2, 1])  # 项目2增长潜力
        ax9 = fig.add_subplot(gs[2, 2], projection='polar')  # 项目2三维趋势（极坐标）
        
        # 绘制各子图
        self._plot_openrank_comparison(ax1)
        self._plot_attention_comparison(ax2)
        self._plot_stars_comparison(ax3)
        self._plot_health_gauge(ax4, self.project1_analysis, 1)
        self._plot_potential_bar(ax5, self.project1_analysis, 1)
        self._plot_trend_3d_radar(ax6, self.project1_analysis, 1)
        self._plot_health_gauge(ax7, self.project2_analysis, 2)
        self._plot_potential_bar(ax8, self.project2_analysis, 2)
        self._plot_trend_3d_radar(ax9, self.project2_analysis, 2)
        
        # 添加总标题
        fig.suptitle(
            f'GitHub项目对比分析: {self.project1.org}/{self.project1.repo} vs {self.project2.org}/{self.project2.repo}\n'
            f'健康评分: {self.project1_analysis["health_score"]:.1f}({self.project1_analysis["health_grade"]}) vs '
            f'{self.project2_analysis["health_score"]:.1f}({self.project2_analysis["health_grade"]}) | '
            f'潜力评分: {self.project1_analysis["potential_score"]:.1f} vs {self.project2_analysis["potential_score"]:.1f}',
            fontsize=14, fontweight='bold', y=0.98
        )
        
        # 保存图表
        filename = f"compare_{self.project1.org}_{self.project1.repo}_vs_{self.project2.org}_{self.project2.repo}_dashboard.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
        # print(f"对比图表已保存: {filename}")
        # plt.show()
    
    def _plot_openrank_comparison(self, ax):
        """绘制OpenRank对比图"""
        if 'openrank' not in self.project1_data or 'openrank' not in self.project2_data:
            ax.text(0.5, 0.5, '无OpenRank数据', ha='center', va='center')
            return
        
        data1 = self.project1_data['openrank']
        data2 = self.project2_data['openrank']
        
        # 确保时间索引对齐
        common_dates = data1.index.intersection(data2.index)
        if len(common_dates) == 0:
            ax.text(0.5, 0.5, '时间索引不匹配', ha='center', va='center')
            return
        
        data1_aligned = data1.loc[common_dates]
        data2_aligned = data2.loc[common_dates]
        
        ax.plot(common_dates, data1_aligned.values, color=COLORS['primary'], 
                lw=2, label=f'{self.project1.org}/{self.project1.repo}')
        ax.plot(common_dates, data2_aligned.values, color=COLORS['success'], 
                lw=2, label=f'{self.project2.org}/{self.project2.repo}')
        
        # 添加预测（如果有）
        if 'openrank' in self.project1_analysis['predictions']:
            pred1 = self.project1_analysis['predictions']['openrank']
            if 'forecast' in pred1 and pred1['forecast']:
                last_date = common_dates[-1]
                future_dates = pd.date_range(
                    start=last_date + pd.DateOffset(months=1),
                    periods=len(pred1['forecast']),
                    freq='ME'
                )
                ax.plot(future_dates, pred1['forecast'], color=COLORS['primary'], 
                       lw=2, linestyle='--', alpha=0.7)
        
        if 'openrank' in self.project2_analysis['predictions']:
            pred2 = self.project2_analysis['predictions']['openrank']
            if 'forecast' in pred2 and pred2['forecast']:
                last_date = common_dates[-1]
                future_dates = pd.date_range(
                    start=last_date + pd.DateOffset(months=1),
                    periods=len(pred2['forecast']),
                    freq='ME'
                )
                ax.plot(future_dates, pred2['forecast'], color=COLORS['success'], 
                       lw=2, linestyle='--', alpha=0.7)
        
        ax.set_title('OpenRank趋势对比', fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=30)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
        ax.set_ylabel('OpenRank', fontsize=10)
    
    def _plot_attention_comparison(self, ax):
        """绘制Attention对比图"""
        if 'attention' not in self.project1_data or 'attention' not in self.project2_data:
            ax.text(0.5, 0.5, '无Attention数据', ha='center', va='center')
            return
        
        data1 = self.project1_data['attention']
        data2 = self.project2_data['attention']
        
        common_dates = data1.index.intersection(data2.index)
        if len(common_dates) == 0:
            ax.text(0.5, 0.5, '时间索引不匹配', ha='center', va='center')
            return
        
        data1_aligned = data1.loc[common_dates]
        data2_aligned = data2.loc[common_dates]
        
        ax.plot(common_dates, data1_aligned.values, color=COLORS['primary'], lw=2)
        ax.plot(common_dates, data2_aligned.values, color=COLORS['success'], lw=2)
        
        ax.set_title('Attention趋势对比', fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=30)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel('Attention', fontsize=10)
    
    def _plot_stars_comparison(self, ax):
        """绘制Stars对比图"""
        if 'stars' not in self.project1_data or 'stars' not in self.project2_data:
            ax.text(0.5, 0.5, '无Stars数据', ha='center', va='center')
            return
        
        data1 = self.project1_data['stars']
        data2 = self.project2_data['stars']
        
        common_dates = data1.index.intersection(data2.index)
        if len(common_dates) == 0:
            ax.text(0.5, 0.5, '时间索引不匹配', ha='center', va='center')
            return
        
        data1_aligned = data1.loc[common_dates]
        data2_aligned = data2.loc[common_dates]
        
        ax.plot(common_dates, data1_aligned.values, color=COLORS['primary'], lw=2)
        ax.plot(common_dates, data2_aligned.values, color=COLORS['success'], lw=2)
        
        ax.set_title('Stars趋势对比', fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=30)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel('Stars', fontsize=10)
    
    def _plot_health_gauge(self, ax, analysis, project_num):
        """绘制健康评分仪表盘"""
        score = analysis['health_score']
        
        # 清空坐标轴
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.7)
        
        # 绘制仪表盘外框
        center_x, center_y = 0.5, 0.35
        radius = 0.35
        
        # 绘制彩色扇形区域
        total_angle = np.pi  # 半圆，180度
        sector_angle = np.pi / 3  # 60度
        
        # 危险区域 (0-33.3分)
        danger_angles = np.linspace(np.pi, np.pi - sector_angle, 50)
        danger_x = [center_x] + list(center_x + radius * np.cos(danger_angles)) + [center_x]
        danger_y = [center_y] + list(center_y + radius * np.sin(danger_angles)) + [center_y]
        ax.fill(danger_x, danger_y, color=COLORS['danger'], alpha=0.3)
        
        # 警告区域 (33.3-66.6分)
        warning_angles = np.linspace(np.pi - sector_angle, np.pi - 2*sector_angle, 50)
        warning_x = [center_x] + list(center_x + radius * np.cos(warning_angles)) + [center_x]
        warning_y = [center_y] + list(center_y + radius * np.sin(warning_angles)) + [center_y]
        ax.fill(warning_x, warning_y, color=COLORS['warning'], alpha=0.3)
        
        # 安全区域 (66.6-100分)
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
            
            if i % 20 == 0:
                x_text = center_x + (radius + 0.05) * np.cos(angle)
                y_text = center_y + (radius + 0.05) * np.sin(angle)
                ax.text(x_text, y_text, str(i), ha='center', va='center', fontsize=8)
        
        # 绘制指针
        angle = np.pi - (score / 100) * total_angle
        needle_length = radius - 0.05
        
        x_tip = center_x + needle_length * np.cos(angle)
        y_tip = center_y + needle_length * np.sin(angle)
        
        angle_left = angle + 0.1
        angle_right = angle - 0.1
        x_left = center_x + 0.05 * np.cos(angle_left)
        y_left = center_y + 0.05 * np.sin(angle_left)
        x_right = center_x + 0.05 * np.cos(angle_right)
        y_right = center_y + 0.05 * np.sin(angle_right)
        
        ax.fill([x_tip, x_left, x_right], [y_tip, y_left, y_right], color='black')
        
        # 中心圆点
        ax.add_patch(plt.Circle((center_x, center_y), 0.02, color='black'))
        
        # 添加分数和等级
        ax.text(center_x, center_y - 0.1, f'{score:.0f}/100', 
                ha='center', va='center', fontsize=14, fontweight='bold')
        ax.text(center_x, center_y - 0.18, analysis['health_grade'], 
                ha='center', va='center', fontsize=12, fontweight='bold')
        
        # 项目标签
        color = COLORS['primary'] if project_num == 1 else COLORS['success']
        ax.text(0.5, 1.1, f'项目{project_num}健康评分', 
                transform=ax.transAxes, ha='center', fontsize=11, fontweight='bold', color=color)
        
        ax.set_aspect('equal')
        ax.axis('off')
    
    def _plot_potential_bar(self, ax, analysis, project_num):
        """绘制增长潜力图"""
        current = analysis['potential']['current_position']
        ceiling = analysis['potential']['growth_ceiling']
        remaining = analysis['potential']['remaining_space']
        
        # 获取历史峰值
        if project_num == 1:
            df = self.project1_data
        else:
            df = self.project2_data
            
        historical_max = df['openrank'].max() if 'openrank' in df else current
        
        # 创建分组柱状图
        categories = ['当前位置', '历史峰值', '增长上限']
        values = [
            current,
            historical_max,
            min(ceiling, historical_max * 2)
        ]
        
        colors = [COLORS['primary'], COLORS['warning'], COLORS['success']]
        
        bars = ax.bar(categories, values, color=colors, alpha=0.8, width=0.6)
        
        # 在每个柱子上添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.02,
                    f'{value:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 在顶部添加剩余空间百分比
        color = COLORS['primary'] if project_num == 1 else COLORS['success']
        ax.text(0.5, 1.05, f'剩余增长空间: {remaining:.0f}%', 
                transform=ax.transAxes, ha='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.2))
        
        # 项目标签
        ax.text(0.5, 1.15, f'项目{project_num}增长潜力', 
                transform=ax.transAxes, ha='center', fontsize=11, fontweight='bold', color=color)
        
        ax.set_ylim(0, max(values) * 1.2)
        ax.set_ylabel('OpenRank', fontsize=10)
        ax.grid(True, axis='y', alpha=0.3)
    
    def _plot_trend_3d_radar(self, ax, analysis, project_num):
        """绘制三维趋势雷达图 - 修复极坐标属性错误"""
        categories = ['动量', '稳定性', '潜力']
        values = [
            analysis['momentum']['total'],
            100 - analysis['resistance']['total'],
            analysis['potential']['remaining_space']
        ]
        values = values + [values[0]]  # 闭合雷达图
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += [angles[0]]
        
        # 设置极坐标（如果ax是极坐标对象）
        if hasattr(ax, 'set_theta_offset'):
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
        
        # 使用默认雷达网格
        ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.7, color='gray')
        
        # 设置网格线
        ax.set_rticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=7)
        
        # 绘制数据线
        color = COLORS['primary'] if project_num == 1 else COLORS['success']
        ax.plot(angles, values, 'o-', color=color, linewidth=3, markersize=6, markerfacecolor='white')
        
        # 设置刻度标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10, fontweight='bold')
        
        # 在每个数据点添加数值标签
        for i, (angle, value) in enumerate(zip(angles[:-1], values[:-1])):
            label_angle = angle
            label_radius = value + 5
            ax.text(label_angle, label_radius, f'{value:.0f}', 
                    ha='center', va='center', fontsize=9, fontweight='bold', 
                    color=color, bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        
        ax.set_ylim(0, 100)
        
        # 项目标签
        title_color = COLORS['primary'] if project_num == 1 else COLORS['success']
        ax.set_title(f'项目{project_num}三维趋势分析', 
                    fontsize=11, fontweight='bold', pad=25, color=title_color)


def main():
    """主函数"""
    # print("\n" + "="*60)
    # print("  GitHub项目对比分析器")
    # print("  对比两个项目的健康程度、发展趋势和潜力")
    # print("="*60 + "\n")
    
    # 解析命令行参数
    url1 = url2 = None
    
    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help']:
            # print(f"  {sys.argv[0]} <项目1地址> <项目2地址>")
            # print(f"  {sys.argv[0]} facebook/react vuejs/vue")
            # print("\n")
            # print("第一个GitHub项目")
            # print("第二个GitHub项目")
            # print("  -h, --help   显示帮助信息")
            # print("\n示例:")
            # print(f"  {sys.argv[0]} facebook/react vuejs/vue")
            # print(f"  {sys.argv[0]} https://github.com/facebook/react https://github.com/vuejs/vue")
            sys.exit(0)
        
        if len(sys.argv) >= 3:
            url1 = sys.argv[1].strip()
            url2 = sys.argv[2].strip()
    
    # 如果命令行没有提供参数，使用交互式输入
    if not url1:
        # url1 = input("请输入第一个GitHub项目地址: ").strip()
        # if not url1:
        url1 = "facebook/react"
    
    if not url2:
        # url2 = input("请输入第二个GitHub项目地址: ").strip()
        # if not url2:
        url2 = "vuejs/vue"
    
    # print(f"\n开始对比分析:")
    # print(f"项目1: {url1}")
    # print(f"项目2: {url2}")
    # print("-"*40)
    
    # 执行分析
    analyzer = ProjectComparativeAnalyzer(url1, url2)
    success = analyzer.run_analysis()
    
    if success:
        # print("\n对比分析完成")
        # print(f"项目1潜力评分: {analyzer.project1_analysis['potential_score']:.1f}")
        # print(f"项目2潜力评分: {analyzer.project2_analysis['potential_score']:.1f}")
        
        # if analyzer.project1_analysis['potential_score'] > analyzer.project2_analysis['potential_score']:
        #     print(f"项目1更具发展潜力")
        # elif analyzer.project1_analysis['potential_score'] < analyzer.project2_analysis['potential_score']:
        #     print(f"项目2更具发展潜力")
        # else:
        #     print(f"两个项目潜力相当")
        sys.exit(0)
    else:
        # print("\n分析失败，请检查项目地址是否正确")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # print("\n\n程序被用户中断")
        sys.exit(1)
    except Exception as e:
        # print(f"\n\n程序运行出错: {e}")
        # import traceback
        # traceback.print_exc()
        sys.exit(1)