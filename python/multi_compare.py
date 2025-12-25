"""
支持任意数量项目的综合对比分析
"""

import sys
import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
from datetime import datetime
import json
import warnings

sys.dont_write_bytecode = True

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")

OUTPUT_DIR = "multi_compare_output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    # print(f"创建输出目录: {OUTPUT_DIR}")

plt.rcParams.update({
    'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans'],
    'font.family': 'sans-serif',
    'axes.unicode_minus': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.labelsize': 10,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 150,
})

COLORS = {
    'primary': '#2E86AB',
    'success': '#28A745',
    'warning': '#FFC107',
    'danger': '#DC3545',
    'info': '#17A2B8',
    'secondary': '#6C757D',
    'purple': '#9B59B6',
    'orange': '#E67E22',
    'green': '#1ABC9C',
    'blue': '#3498DB',
    'red': '#E74C3C',
    'yellow': '#F1C40F',
    'brown': '#7F8C8D',
}

try:
    from v4 import (
        GMMTierClassifier, MomentumCalculator, ResistanceCalculator, 
        PotentialCalculator, ProphetPredictor, AHPHealthEvaluator,
        BusFactorCalculator, ChangePointDetector, 
        ConversionFunnelAnalyzer, ETDAnalyzer, GitHubAPIAnalyzer,
        TIER_BENCHMARKS, TIER_NAMES
    )
    # print("成功导入v4.py组件")
except ImportError as e:
    # print(f"导入v4.py组件失败: {e}")
    # print("请确保v4.py文件存在于同一目录下")
    sys.exit(1)
except Exception as e:
    # print(f"导入组件时发生未知错误: {e}")
    sys.exit(1)

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
        self.project_name = f"{self.org}/{self.repo}"
        
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

class MultiProjectAnalyzer:
    """多项目对比分析器"""
    
    def __init__(self, project_urls: list):
        self.project_urls = project_urls
        self.projects = [] 
        self.analyses = []  
        self.project_names = []  
        self.output_dir = OUTPUT_DIR
    
    def fetch_all_data(self) -> bool:
        """获取所有项目的数据"""
        for url in self.project_urls:
            try:
                fetcher = ProjectDataFetcher(url)
                if fetcher.fetch_data():
                    self.projects.append(fetcher)
                    self.project_names.append(fetcher.project_name)
                    # print(f"成功获取: {fetcher.project_name}")
                else:
                    # print(f"跳过项目: {url}")
                    continue
            except Exception as e:
                # print(f"处理项目 {url} 时出错: {e}")
                continue
        
        if not self.projects:
            # print("所有项目数据获取失败")
            return False
        
        # print(f"\n成功获取 {len(self.projects)} 个项目的数据")
        return True
    
    def analyze_all_projects(self):
        """分析所有项目"""
        # print("\n开始分析所有项目...")
        for i, project in enumerate(self.projects):
            # print(f"分析项目 {i+1}/{len(self.projects)}: {project.project_name}")
            analysis = self._analyze_single_project(project)
            self.analyses.append(analysis)
        # print("所有项目分析完成")
    
    def _analyze_single_project(self, project):
        """分析单个项目"""
        analysis = {}
        analysis['project_name'] = project.project_name
        tier, probabilities, confidence = project.get_tier_classification()
        analysis['tier'] = tier
        analysis['tier_probabilities'] = probabilities
        analysis['tier_confidence'] = confidence
        analysis['vitality'] = project.analyze_vitality()
        momentum_calc = MomentumCalculator()
        analysis['momentum'] = momentum_calc.calculate(project.df)
        resistance_calc = ResistanceCalculator()
        analysis['resistance'] = resistance_calc.calculate(project.df)
        potential_calc = PotentialCalculator()
        analysis['potential'] = potential_calc.calculate(project.df, tier)
        bus_factor_calc = BusFactorCalculator()
        analysis['bus_factor'] = bus_factor_calc.calculate(project.df)

        ahp_evaluator = AHPHealthEvaluator()
        trend_3d = {
            'momentum': analysis['momentum'],
            'resistance': analysis['resistance'],
            'potential': analysis['potential']
        }
        
        risk_score = self._calculate_risk_score(analysis, project)
        analysis['risk'] = risk_score
        
        predictor = ProphetPredictor()
        predictions = {}
        for metric in ['openrank', 'attention', 'stars']:
            if metric in project.df:
                try:
                    pred_result = predictor.predict(project.df, metric, periods=6)
                    predictions[metric] = pred_result
                except Exception as e:
                    # print(f"  预测 {metric} 失败: {e}")
                    predictions[metric] = {}
        analysis['predictions'] = predictions
        
        try:
            health_score, dimension_scores = ahp_evaluator.calculate_health_score(
                analysis['vitality'], trend_3d, analysis['risk'], tier, predictions
            )
            analysis['health_score'] = health_score
            analysis['dimension_scores'] = dimension_scores
        except Exception as e:
            # print(f"  计算健康评分失败: {e}")
            analysis['health_score'] = 50
            analysis['dimension_scores'] = {}
        
        # 确定健康等级
        grades = [(85, 'A+'), (75, 'A'), (65, 'B+'), (55, 'B'), (45, 'C'), (35, 'D'), (0, 'F')]
        analysis['health_grade'] = next(g for t, g in grades if analysis['health_score'] >= t)
        
        # print(f"{project.project_name}: 健康评分 {analysis['health_score']:.1f} ({analysis['health_grade']})")
        return analysis
    
    def _calculate_risk_score(self, analysis, project=None):
        """计算风险评分 - 与v4.py保持一致"""
        risk_score = 0
        
        # 如果提供了project，考虑活跃度趋势
        if project and hasattr(project, 'df') and 'activity' in project.df:
            from scipy.stats import linregress
            df = project.df
            x_vals = range(min(12, len(df)))
            y_vals = df['activity'].tail(12).values
            if len(y_vals) >= 2:  # 确保有足够的数据点进行回归
                slope, intercept, r_value, p_value, std_err = linregress(x_vals, y_vals)
                if slope < -0.5:
                    risk_score += 30
                elif slope < 0:
                    risk_score += 15
        
        # 阻力状态评估
        if 'resistance' in analysis and 'status' in analysis['resistance']:
            if analysis['resistance']['status'] == 'HEAVY':
                risk_score += 25
        
        # 活力状态评估（与v4.py保持一致）
        vitality = analysis['vitality']
        if vitality == 'ZOMBIE':
            risk_score += 30  # 与v4.py保持一致
        elif vitality == 'DORMANT':
            risk_score += 15  # 与v4.py保持一致
        
        # 巴士系数风险
        if 'bus_factor' in analysis and 'risk_level' in analysis['bus_factor']:
            if analysis['bus_factor']['risk_level'] in ['CRITICAL', 'HIGH']:
                risk_score += 20  # 保留此评估
        
        # 返回完整的风险字典结构，与v4.py保持一致
        if risk_score >= 50:
            level = 'CRITICAL'
        elif risk_score >= 30:
            level = 'HIGH'
        elif risk_score >= 15:
            level = 'MEDIUM'
        else:
            level = 'LOW'
        
        return {'score': risk_score, 'level': level, 'alerts': []}
    
    def generate_visualizations(self):
        """生成所有可视化图表"""
        # print(f"\n生成可视化图表到目录: {self.output_dir}")
        
        if not self.projects or not self.analyses:
            # print("没有数据可用于生成图表")
            return
        self._plot_trend_comparison('openrank', 'OpenRank趋势预测对比')

        self._plot_trend_comparison('attention', 'Attention趋势预测对比')

        self._plot_trend_comparison('stars', 'Stars趋势预测对比')

        self._plot_health_comparison()

        self._plot_trend_comparison_bar()

        self._plot_triple_metric_comparison()
        
        # print("可视化图表生成完成")
    
    def _plot_trend_comparison(self, metric, title):
        """绘制趋势预测对比图"""
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            colors = list(COLORS.values())
            
            for i, (project, analysis) in enumerate(zip(self.projects, self.analyses)):
                if metric not in project.df:
                    continue
                
                color = colors[i % len(colors)]
                data = project.df[metric]
                ax.plot(data.index, data.values, color=color, lw=2, label=project.project_name)
                if metric in analysis.get('predictions', {}):
                    pred_result = analysis['predictions'][metric]
                    if 'forecast' in pred_result and pred_result['forecast']:
                        last_date = data.index[-1]
                        future_dates = pd.date_range(
                            start=last_date + pd.DateOffset(months=1),
                            periods=len(pred_result['forecast']),
                            freq='ME'
                        )
                        ax.plot(future_dates, pred_result['forecast'], color=color, 
                               lw=2, linestyle='--', alpha=0.7)
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.tick_params(axis='x', rotation=30)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=9)
            ax.set_ylabel(metric.capitalize(), fontsize=10)
            filename = f"multi_{metric}_trend_comparison.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.tight_layout()
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            # print(f"保存图表: {filename}")
            plt.close()
        except Exception as e:
            # print(f"绘制 {title} 失败: {e}")
            pass
    
    def _plot_health_comparison(self):
        """绘制健康程度对比柱形图"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            project_names = [analysis.get('project_name', f'项目{i+1}') for i, analysis in enumerate(self.analyses)]
            health_scores = [analysis.get('health_score', 0) for analysis in self.analyses]
            colors = list(COLORS.values())
            bar_colors = colors[:len(project_names)]
            
            bars = ax.bar(project_names, health_scores, color=bar_colors, alpha=0.8)
            for bar, score in zip(bars, health_scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1, 
                        f'{score:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            ax.set_title('健康程度对比', fontsize=14, fontweight='bold')
            ax.set_ylabel('健康评分', fontsize=10)
            ax.tick_params(axis='x', rotation=30)
            ax.grid(True, axis='y', alpha=0.3)
            ax.set_ylim(0, 100)
            filename = "multi_health_comparison.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.tight_layout()
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            # print(f"保存图表: {filename}")
            plt.close()
        except Exception as e:
            # print(f"绘制健康程度对比图失败: {e}")
            pass
    
    def _plot_trend_comparison_bar(self):
        """绘制发展趋势对比柱形图"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            project_names = [analysis.get('project_name', f'项目{i+1}') for i, analysis in enumerate(self.analyses)]
            momentum_scores = [analysis.get('momentum', {}).get('total', 0) for analysis in self.analyses]
            
            # 为每个项目分配不同颜色
            colors = list(COLORS.values())
            bar_colors = colors[:len(project_names)]
            
            bars = ax.bar(project_names, momentum_scores, color=bar_colors, alpha=0.8)
            
            # 在每个柱子上添加数值标签
            for bar, score in zip(bars, momentum_scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1, 
                        f'{score:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            ax.set_title('发展趋势对比 (动量)', fontsize=14, fontweight='bold')
            ax.set_ylabel('动量评分', fontsize=10)
            ax.tick_params(axis='x', rotation=30)
            ax.grid(True, axis='y', alpha=0.3)
            ax.set_ylim(0, 100)
            
            # 保存图表
            filename = "multi_trend_comparison.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.tight_layout()
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            # print(f"保存图表: {filename}")
            plt.close()
        except Exception as e:
            # print(f"绘制发展趋势对比图失败: {e}")
            pass
    
    def _plot_triple_metric_comparison(self):
        """绘制阻力、潜力、稳定性柱形图对比"""
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            project_names = [analysis.get('project_name', f'项目{i+1}') for i, analysis in enumerate(self.analyses)]
            n_projects = len(project_names)
            
            if n_projects == 0:
                return
            
            # 准备数据
            resistance = []
            potential = []
            stability = []
            
            for analysis in self.analyses:
                resistance.append(analysis.get('resistance', {}).get('total', 0))
                potential.append(analysis.get('potential', {}).get('remaining_space', 0))
                # 稳定性 = 100 - 阻力
                resistance_val = analysis.get('resistance', {}).get('total', 0)
                stability.append(100 - resistance_val)
            
            # 设置柱子宽度和位置
            width = 0.25
            x = np.arange(n_projects)
            
            # 绘制三组柱子
            rects1 = ax.bar(x - width, resistance, width, label='阻力', color=COLORS['danger'])
            rects2 = ax.bar(x, potential, width, label='潜力', color=COLORS['success'])
            rects3 = ax.bar(x + width, stability, width, label='稳定性', color=COLORS['primary'])
            
            # 添加数值标签
            for rects in [rects1, rects2, rects3]:
                for rect in rects:
                    height = rect.get_height()
                    ax.text(rect.get_x() + rect.get_width()/2., height + 1, 
                            f'{height:.1f}', ha='center', va='bottom', fontsize=8)
            
            ax.set_title('阻力、潜力、稳定性对比', fontsize=14, fontweight='bold')
            ax.set_ylabel('评分', fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels(project_names, rotation=30)
            ax.legend()
            ax.grid(True, axis='y', alpha=0.3)
            ax.set_ylim(0, 100)
            
            # 保存图表
            filename = "multi_triple_metric_comparison.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.tight_layout()
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            # print(f"保存图表: {filename}")
            plt.close()
        except Exception as e:
            # print(f"绘制三指标对比图失败: {e}")
            pass
    
    def generate_report(self):
        """生成综合对比报告"""
        # print(f"\n生成综合对比报告到目录: {self.output_dir}")
        
        if not self.analyses:
            # print("没有分析数据可用于生成报告")
            return ""
        
        report = f"""
{'='*80}
                    GitHub多项目综合对比分析报告
                      生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}

【项目基本信息】
{'-'*80}
"""
        
        # 项目基本信息
        for i, analysis in enumerate(self.analyses):
            report += f"""
项目 {i+1}: {analysis.get('project_name', '未知项目')}
层级: {analysis.get('tier', '未知')} ({TIER_NAMES.get(analysis.get('tier', ''), '未知')})
健康评分: {analysis.get('health_score', 0):.1f}/100 ({analysis.get('health_grade', '未知')})
活力状态: {analysis.get('vitality', '未知')}
"""
        
        report += f"""
{'-'*80}

【详细指标对比】
{'-'*80}
"""
        
        # 指标对比表格
        report += "\n指标对比表格:\n"
        report += "{:<20} {:<15} {:<15} {:<15} {:<15}\n".format(
            "项目名称", "健康评分", "动量", "阻力", "潜力"
        )
        report += "-" * 80 + "\n"
        
        for analysis in self.analyses:
            momentum_total = analysis.get('momentum', {}).get('total', 0)
            resistance_total = analysis.get('resistance', {}).get('total', 0)
            potential_space = analysis.get('potential', {}).get('remaining_space', 0)
            
            report += "{:<20} {:<15.1f} {:<15.1f} {:<15.1f} {:<15.1f}\n".format(
                analysis.get('project_name', '未知项目'),
                analysis.get('health_score', 0),
                momentum_total,
                resistance_total,
                potential_space
            )
        
        report += f"""

【项目排名】
{'-'*80}
"""
        
        # 健康评分排名
        try:
            health_ranked = sorted(self.analyses, key=lambda x: x.get('health_score', 0), reverse=True)
            report += "\n1. 健康评分排名:\n"
            for i, analysis in enumerate(health_ranked, 1):
                report += f"   {i}. {analysis.get('project_name', '未知项目')}: {analysis.get('health_score', 0):.1f}/100 ({analysis.get('health_grade', '未知')})\n"
        except:
            report += "\n1. 健康评分排名: 数据不足\n"
        
        # 动量排名
        try:
            momentum_ranked = sorted(self.analyses, key=lambda x: x.get('momentum', {}).get('total', 0), reverse=True)
            report += "\n2. 动量排名:\n"
            for i, analysis in enumerate(momentum_ranked, 1):
                report += f"   {i}. {analysis.get('project_name', '未知项目')}: {analysis.get('momentum', {}).get('total', 0):.1f}\n"
        except:
            report += "\n2. 动量排名: 数据不足\n"
        
        # 潜力排名
        try:
            potential_ranked = sorted(self.analyses, key=lambda x: x.get('potential', {}).get('remaining_space', 0), reverse=True)
            report += "\n3. 潜力排名:\n"
            for i, analysis in enumerate(potential_ranked, 1):
                report += f"   {i}. {analysis.get('project_name', '未知项目')}: {analysis.get('potential', {}).get('remaining_space', 0):.1f}%\n"
        except:
            report += "\n3. 潜力排名: 数据不足\n"
        
        # 稳定性排名
        try:
            stability_ranked = sorted(self.analyses, key=lambda x: 100 - x.get('resistance', {}).get('total', 0), reverse=True)
            report += "\n4. 稳定性排名:\n"
            for i, analysis in enumerate(stability_ranked, 1):
                stability_score = 100 - analysis.get('resistance', {}).get('total', 0)
                report += f"   {i}. {analysis.get('project_name', '未知项目')}: {stability_score:.1f}\n"
        except:
            report += "\n4. 稳定性排名: 数据不足\n"
        
        report += f"""

【最大值分析】
{'-'*80}
"""
        
        try:
            # 健康评分最大值
            if self.analyses:
                max_health = max(self.analyses, key=lambda x: x.get('health_score', 0))
                report += f"\n1. 健康评分最高项目: {max_health.get('project_name', '未知项目')} ({max_health.get('health_score', 0):.1f})\n"
                
                # 动量最大值
                max_momentum = max(self.analyses, key=lambda x: x.get('momentum', {}).get('total', 0))
                report += f"2. 动量最高项目: {max_momentum.get('project_name', '未知项目')} ({max_momentum.get('momentum', {}).get('total', 0):.1f})\n"
                
                # 潜力最大值
                max_potential = max(self.analyses, key=lambda x: x.get('potential', {}).get('remaining_space', 0))
                report += f"3. 潜力最高项目: {max_potential.get('project_name', '未知项目')} ({max_potential.get('potential', {}).get('remaining_space', 0):.1f}%)\n"
                
                # 稳定性最大值
                max_stability = max(self.analyses, key=lambda x: 100 - x.get('resistance', {}).get('total', 0))
                stability_score = 100 - max_stability.get('resistance', {}).get('total', 0)
                report += f"4. 稳定性最高项目: {max_stability.get('project_name', '未知项目')} ({stability_score:.1f})\n"
        except:
            report += "\n最大值分析: 数据不足\n"
        
        # 生成图像路径信息
        report += f"\n5. 生成的可视化图表:\n"
        chart_files = [
            "multi_openrank_trend_comparison.png",
            "multi_attention_trend_comparison.png", 
            "multi_stars_trend_comparison.png",
            "multi_health_comparison.png",
            "multi_trend_comparison.png",
            "multi_triple_metric_comparison.png"
        ]
        for chart in chart_files:
            report += f"   - {chart}\n"
        
        report += f"""

【综合分析】
{'-'*80}
"""
        
        # 综合评估
        report += "\n综合评估:\n"
        report += "-" * 80 + "\n"
        
        for analysis in self.analyses:
            report += f"\n{analysis.get('project_name', '未知项目')}:\n"
            report += f"   优势: {self._get_strengths(analysis)}\n"
            report += f"   劣势: {self._get_weaknesses(analysis)}\n"
            report += f"   建议: {self._get_recommendations(analysis)}\n"
        
        report += f"""

【结论与建议】
{'-'*80}
"""
        
        # 给出综合结论
        if self.analyses:
            report += "\n综合结论:\n"
            try:
                best_project = max(self.analyses, key=lambda x: x.get('health_score', 0))
                worst_project = min(self.analyses, key=lambda x: x.get('health_score', 0))
                
                report += f"1. 综合表现最佳项目: {best_project.get('project_name', '未知项目')} (健康评分: {best_project.get('health_score', 0):.1f})\n"
                report += f"2. 综合表现最差项目: {worst_project.get('project_name', '未知项目')} (健康评分: {worst_project.get('health_score', 0):.1f})\n"
                
                avg_health = np.mean([a.get('health_score', 0) for a in self.analyses])
                report += f"3. 平均健康评分: {avg_health:.1f}\n"
                
                report += "\n总体建议:\n"
                if avg_health >= 70:
                    report += "   所有项目整体健康状况良好，建议继续保持并加强交流协作。\n"
                elif avg_health >= 50:
                    report += "   项目整体健康状况一般，建议关注评分较低的项目，提升社区活跃度。\n"
                else:
                    report += "   部分项目存在健康问题，建议重点关注健康评分较低的项目，采取措施提升其活力。\n"
            except:
                report += "   数据不足，无法给出具体结论。\n"
        
        report += f"""

{'='*80}
所有分析结果已保存至目录: {self.output_dir}
包括6张可视化图表和本报告文件。
{'='*80}
"""
        
        # 保存报告
        filename = "multi_project_comparison_report.txt"
        filepath = os.path.join(self.output_dir, filename)
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report)
            
            # print(f"综合对比报告已保存: {filepath}")
            
            # 在控制台显示报告摘要
            # print("\n" + "="*60)
            # print("报告摘要:")
            # print("="*60)
            # for analysis in self.analyses:
            #     print(f"{analysis.get('project_name', '未知项目')}: "
            #           f"健康评分 {analysis.get('health_score', 0):.1f} ({analysis.get('health_grade', '未知')}")
            # print("="*60)
            
        except Exception as e:
            # print(f"✗ 保存报告失败: {e}")
            pass
        
        return report
    
    def _get_strengths(self, analysis):
        """获取项目优势"""
        strengths = []
        
        health_score = analysis.get('health_score', 0)
        momentum_total = analysis.get('momentum', {}).get('total', 0)
        resistance_total = analysis.get('resistance', {}).get('total', 0)
        potential_space = analysis.get('potential', {}).get('remaining_space', 0)
        
        if health_score >= 80:
            strengths.append("健康状况良好")
        if momentum_total >= 70:
            strengths.append("发展势头强劲")
        if resistance_total <= 30:
            strengths.append("技术阻力较小")
        if potential_space >= 50:
            strengths.append("增长潜力较大")
        
        return "、".join(strengths) if strengths else "暂无明显优势"
    
    def _get_weaknesses(self, analysis):
        """获取项目劣势"""
        weaknesses = []
        
        health_score = analysis.get('health_score', 0)
        momentum_total = analysis.get('momentum', {}).get('total', 0)
        resistance_total = analysis.get('resistance', {}).get('total', 0)
        potential_space = analysis.get('potential', {}).get('remaining_space', 0)
        
        if health_score < 60:
            weaknesses.append("健康状况不佳")
        if momentum_total < 40:
            weaknesses.append("发展势头疲软")
        if resistance_total > 70:
            weaknesses.append("技术阻力较大")
        if potential_space < 20:
            weaknesses.append("增长潜力有限")
        
        return "、".join(weaknesses) if weaknesses else "暂无明显劣势"
    
    def _get_recommendations(self, analysis):
        """获取改进建议"""
        recommendations = []
        
        momentum_total = analysis.get('momentum', {}).get('total', 0)
        resistance_total = analysis.get('resistance', {}).get('total', 0)
        potential_space = analysis.get('potential', {}).get('remaining_space', 0)
        
        if momentum_total < 50:
            recommendations.append("加强社区活跃度，吸引更多贡献者")
        if resistance_total > 60:
            recommendations.append("关注技术债务，及时修复问题")
        if potential_space < 30:
            recommendations.append("寻找新的增长点，拓展项目影响力")
        
        return "、".join(recommendations) if recommendations else "保持现有良好状态"

def get_project_urls_from_input():
    """从用户输入获取项目URL"""
    print("\n请输入要分析的GitHub项目（每行一个，输入空行结束）:")
    print("格式: owner/repo 或 https://github.com/owner/repo")
    print("示例: facebook/react")
    print("      https://github.com/vuejs/vue")
    print()
    
    urls = []
    i = 1
    while True:
        url = input(f"项目 {i} (输入空行结束): ").strip()
        if not url:
            break
        urls.append(url)
        i += 1
    
    return urls

def get_project_urls_from_args():
    """从命令行参数获取项目URL"""
    if len(sys.argv) < 2:
        return []
    
    # 检查是否是帮助请求
    if sys.argv[1] in ['-h', '--help']:
        return []
    
    return sys.argv[1:]

def main():
    """主函数"""
    # print("\n" + "="*60)
    # print("  GitHub多项目对比分析器")
    # print("  支持任意数量项目的综合对比分析")
    # print("="*60 + "\n")
    
    # 获取项目URL列表
    project_urls = get_project_urls_from_args()
    
    # 如果没有命令行参数，提示用户输入
    if not project_urls:
        # print("检测到没有命令行参数，请手动输入项目地址。")
        # project_urls = get_project_urls_from_input()
        # 
        # if not project_urls:
        #     print("\n没有输入任何项目地址。")
        #     print("\n使用方法:")
        #     print(f"  {sys.argv[0]} <项目1地址> <项目2地址> ... <项目n地址>")
        #     print(f"  或直接运行 {sys.argv[0]} 然后按提示输入项目地址")
        #     print("\n示例:")
        #     print(f"  {sys.argv[0]} facebook/react vuejs/vue angular/angular")
        #     print(f"  {sys.argv[0]} https://github.com/facebook/react https://github.com/vuejs/vue")
        #     print(f"\n输出目录: {OUTPUT_DIR}")
        project_urls = ["facebook/react", "vuejs/vue"]
        # sys.exit(0)
    
    # print(f"\n开始分析 {len(project_urls)} 个项目:")
    # for i, url in enumerate(project_urls, 1):
    #     print(f"  {i}. {url}")
    # print(f"\n输出目录: {OUTPUT_DIR}")
    # print("-"*60)
    
    # 创建分析器并执行分析
    analyzer = MultiProjectAnalyzer(project_urls)
    
    if not analyzer.fetch_all_data():
        # print("\n数据获取失败，分析终止")
        sys.exit(1)
    
    analyzer.analyze_all_projects()
    analyzer.generate_visualizations()
    analyzer.generate_report()
    
    # print(f"\n" + "="*60)
    # print(f"多项目综合对比分析完成!")
    # print(f"所有文件已保存到目录: {OUTPUT_DIR}")
    # print("生成的文件:")
    # print("  - 6张可视化图表 (.png)")
    # print("  - 1份综合对比报告 (.txt)")
    # print("="*60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n程序运行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)