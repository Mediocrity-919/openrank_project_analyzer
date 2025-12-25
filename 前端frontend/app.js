// 配置
const API_BASE = 'http://localhost:5501';
let currentAnalysisId = null;

// DOM元素
const tabBtns = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');

// 在DOM内容加载完成后获取表单元素并绑定事件
document.addEventListener('DOMContentLoaded', () => {
    const analyzeForm = document.getElementById('analyze-form');
    const compareForm = document.getElementById('compare-form');
    const multiCompareForm = document.getElementById('multi-compare-form');
    const refreshResultsBtn = document.getElementById('refresh-results');
    const clearResultsBtn = document.getElementById('clear-results');
    const loadingOverlay = document.getElementById('loading');
    const showHelpBtn = document.getElementById('show-help');
    const helpDialog = document.getElementById('help-dialog');
    const closeModalBtn = document.querySelector('.close-modal');

    // 分析单个项目
    if (analyzeForm) {
        analyzeForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const projectUrl = document.getElementById('project-url').value;
            
            if (!projectUrl) {
                showMessage('请输入GitHub项目URL', 'error');
                return;
            }
            
            try {
                showLoading();
                showMessage('开始分析项目...', 'info');
                
                const response = await fetch(`${API_BASE}/api/analyze`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ projectUrl }),
                });
                
                const data = await response.json();
                
                if (data.success) {
                    displayAnalysisResults(data);
                    showMessage('分析完成！', 'success');
                } else {
                    showMessage(data.error || '分析失败', 'error');
                }
            } catch (error) {
                console.error('分析错误:', error);
                showMessage('网络请求失败，请检查后端服务', 'error');
            } finally {
                hideLoading();
            }
        });
    }

    // 对比两个项目
    if (compareForm) {
        compareForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const project1Url = document.getElementById('project1-url').value;
            const project2Url = document.getElementById('project2-url').value;
            
            if (!project1Url || !project2Url) {
                showMessage('请输入两个项目的URL', 'error');
                return;
            }
            
            try {
                showLoading();
                showMessage('开始对比项目...', 'info');
                
                const response = await fetch(`${API_BASE}/api/compare`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ project1Url, project2Url }),
                });
                
                const data = await response.json();
                
                if (data.success) {
                    displayComparisonResults(data);
                    showMessage('对比完成！', 'success');
                } else {
                    showMessage(data.error || '对比失败', 'error');
                }
            } catch (error) {
                console.error('对比错误:', error);
                showMessage('网络请求失败，请检查后端服务', 'error');
            } finally {
                hideLoading();
            }
        });
    }

    // 多项目对比
    if (multiCompareForm) {
        multiCompareForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const projectUrlsText = document.getElementById('multi-project-urls').value;
            
            // 解析URL列表
            const projectUrls = projectUrlsText
                .split('\n')
                .map(url => url.trim())
                .filter(url => url);
            
            if (projectUrls.length < 2) {
                showMessage('请输入至少两个GitHub项目URL', 'error');
                return;
            }
            
            try {
                showLoading();
                showMessage(`开始对比 ${projectUrls.length} 个项目...`, 'info');
                
                const response = await fetch(`${API_BASE}/api/multi-compare`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ projectUrls }),
                });
                
                const data = await response.json();
                
                if (data.success) {
                    displayMultiComparisonResults(data);
                    showMessage('多项目对比完成！', 'success');
                } else {
                    showMessage(data.error || '多项目对比失败', 'error');
                }
            } catch (error) {
                console.error('多项目对比错误:', error);
                showMessage('网络请求失败，请检查后端服务', 'error');
            } finally {
                hideLoading();
            }
        });
    }

    // 清除所有结果
    if (clearResultsBtn) {
        clearResultsBtn.addEventListener('click', async () => {
            if (!confirm('确定要清除所有历史结果吗？')) return;
            
            try {
                const response = await fetch(`${API_BASE}/api/results`, {
                    method: 'DELETE',
                });
                
                const data = await response.json();
                
                if (data.success) {
                    showMessage(data.message, 'success');
                    loadHistoryResults();
                }
            } catch (error) {
                console.error('清除结果失败:', error);
                showMessage('清除失败', 'error');
            }
        });
    }

    // 刷新结果列表
    if (refreshResultsBtn) {
        refreshResultsBtn.addEventListener('click', loadHistoryResults);
    }

    // 帮助对话框
    if (showHelpBtn) {
        showHelpBtn.addEventListener('click', () => {
            helpDialog.style.display = 'flex';
        });
    }

    if (closeModalBtn) {
        closeModalBtn.addEventListener('click', () => {
            helpDialog.style.display = 'none';
        });
    }

    if (helpDialog) {
        helpDialog.addEventListener('click', (e) => {
            if (e.target === helpDialog) {
                helpDialog.style.display = 'none';
            }
        });
    }

    // 加载历史结果
    loadHistoryResults();
    
    // 检查后端连接
    checkBackendConnection();
});

// 切换标签页
tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        const tabId = btn.dataset.tab;
        
        // 更新按钮状态
        tabBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        
        // 显示对应内容，隐藏其他内容
        tabContents.forEach(content => {
            content.classList.remove('active');
            if (content.id === `${tabId}-tab`) {
                content.classList.add('active');
            }
        });
    });
});

// 示例项目点击
document.querySelectorAll('.example-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const url1 = btn.dataset.url1;
        const url2 = btn.dataset.url2;
        const urls = btn.dataset.urls;
        
        if (url1 && url2) {
            // 对比模式
            document.getElementById('project1-url').value = url1;
            document.getElementById('project2-url').value = url2;
            
            // 切换到对比标签
            document.querySelector('[data-tab="compare"]').click();
        } else if (urls) {
            // 多项目模式
            document.getElementById('multi-project-urls').value = urls;
            
            // 切换到多项目对比标签
            document.querySelector('[data-tab="multi-compare"]').click();
        } else if (btn.dataset.url) {
            // 单项目模式
            document.getElementById('project-url').value = btn.dataset.url;
            
            // 切换到分析标签
            document.querySelector('[data-tab="analyze"]').click();
        }
    });
});

// 显示加载动画
function showLoading() {
    const loadingOverlay = document.getElementById('loading');
    if (loadingOverlay) loadingOverlay.style.display = 'flex';
}

// 隐藏加载动画
function hideLoading() {
    const loadingOverlay = document.getElementById('loading');
    if (loadingOverlay) loadingOverlay.style.display = 'none';
}

// 显示消息
function showMessage(message, type = 'info') {
    // 移除旧的消息
    const oldMessage = document.querySelector('.alert-message');
    if (oldMessage) oldMessage.remove();
    
    // 创建新消息
    const alert = document.createElement('div');
    alert.className = `alert-message alert-${type}`;
    alert.textContent = message;
    
    // 样式
    alert.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 15px 20px;
        border-radius: 10px;
        color: white;
        font-weight: 600;
        z-index: 1000;
        animation: slideIn 0.3s ease;
    `;
    
    if (type === 'success') {
        alert.style.background = 'linear-gradient(135deg, #28a745, #20c997)';
    } else if (type === 'error') {
        alert.style.background = 'linear-gradient(135deg, #dc3545, #c82333)';
    } else {
        alert.style.background = 'linear-gradient(135deg, #17a2b8, #138496)';
    }
    
    document.body.appendChild(alert);
    
    // 3秒后自动移除
    setTimeout(() => {
        alert.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => alert.remove(), 300);
    }, 3000);
    
    // 添加CSS动画
    if (!document.querySelector('#alert-animations')) {
        const style = document.createElement('style');
        style.id = 'alert-animations';
        style.textContent = `
            @keyframes slideIn {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
            @keyframes slideOut {
                from { transform: translateX(0); opacity: 1; }
                to { transform: translateX(100%); opacity: 0; }
            }
        `;
        document.head.appendChild(style);
    }
}

// 显示分析结果
function displayAnalysisResults(data) {
    const container = document.getElementById('analyze-results');
    if (!container) return;
    
    // 提取项目名
    const projectName = data.project;
    const summary = data.summary || {};
    
    // 构建结果HTML
    let html = `
        <div class="result-header">
            <h3><i class="fas fa-chart-bar"></i> 分析结果: ${projectName}</h3>
            <div class="result-actions">`;
    
    // 报告下载
    if (data.results.report) {
        html += `<button class="btn btn-secondary" onclick="downloadReport('${projectName}', '${data.results.report}')">
                    <i class="fas fa-download"></i> 下载报告
                </button>`;
    }
    
    // 图片下载
    if (data.results.imageUrl) {
        html += `<a href="${data.results.imageUrl}" download class="btn btn-secondary">
                    <i class="fas fa-image"></i> 下载图片
                </a>`;
    }
    
    // CSV下载
    if (data.results.csvUrl) {
        html += `<a href="${data.results.csvUrl}" download class="btn btn-secondary">
                    <i class="fas fa-file-csv"></i> 下载CSV
                </a>`;
    }
    
    // 复制报告
    if (data.results.report) {
        html += `<button class="btn btn-secondary" onclick="copyToClipboard('${data.results.report}')">
                    <i class="fas fa-copy"></i> 复制报告
                </button>`;
    }
    
    html += `</div></div>
        <div class="result-summary">
            <h4><i class="fas fa-info-circle"></i> 项目摘要</h4>
            <div class="summary-grid">
                <div class="summary-item">
                    <div class="label">健康评分</div>
                    <div class="value ${getHealthClass(summary.healthScore)}">${summary.healthScore || 'N/A'}</div>
                    <div class="grade">${summary.healthGrade || ''}</div>
                </div>
                <div class="summary-item">
                    <div class="label">项目层级</div>
                    <div class="value">${summary.tier || 'N/A'}</div>
                </div>
                <div class="summary-item">
                    <div class="label">活力状态</div>
                    <div class="value ${getVitalityClass(summary.vitality)}">${summary.vitality || 'N/A'}</div>
                </div>
                <div class="summary-item">
                    <div class="label">增长潜力</div>
                    <div class="value ${getPotentialClass(summary.potential)}">${summary.potential || 0}%</div>
                </div>
            </div>
        </div>`;
    
    // 如果有图像
    if (data.results.image) {
        html += `
            <div class="result-image">
                <h4><i class="fas fa-chart-pie"></i> 分析图表</h4>
                <img src="${data.results.image}" alt="${projectName} 分析图表">
            </div>
        `;
    }
    
    // 如果有报告
    if (data.results.report) {
        html += `
            <div class="result-report">
                <h4><i class="fas fa-file-alt"></i> 详细报告</h4>
                <pre>${data.results.report}</pre>
            </div>
        `;
    }
    
    container.innerHTML = html;
    container.style.display = 'block';
}

// 显示对比结果
function displayComparisonResults(data) {
    const container = document.getElementById('compare-results');
    if (!container) return;
    const summary = data.summary || {};
    
    // 判断胜者
    let winnerText = '';
    if (summary.winner === 'project1') {
        winnerText = `${data.project1} 更健康`;
    } else if (summary.winner === 'project2') {
        winnerText = `${data.project2} 更健康`;
    } else {
        winnerText = '两个项目相当';
    }
    
    // 构建结果HTML
    let html = `
        <div class="result-header">
            <h3><i class="fas fa-balance-scale"></i> 对比结果: ${data.project1} vs ${data.project2}</h3>
            <div class="result-actions">`;
    
    // 报告下载
    if (data.results.report) {
        html += `<button class="btn btn-secondary" onclick="downloadReport('对比报告', '${data.results.report}')">
                    <i class="fas fa-download"></i> 下载报告
                </button>`;
    }
    
    // 图片下载
    if (data.results.imageUrl) {
        html += `<a href="${data.results.imageUrl}" download class="btn btn-secondary">
                    <i class="fas fa-image"></i> 下载图片
                </a>`;
    }
    
    // CSV下载
    if (data.results.csvUrl) {
        html += `<a href="${data.results.csvUrl}" download class="btn btn-secondary">
                    <i class="fas fa-file-csv"></i> 下载CSV
                </a>`;
    }
    
    // 复制报告
    if (data.results.report) {
        html += `<button class="btn btn-secondary" onclick="copyToClipboard('${data.results.report}')">
                    <i class="fas fa-copy"></i> 复制报告
                </button>`;
    }
    
    html += `</div></div>
        <div class="result-summary">
            <h4><i class="fas fa-trophy"></i> 对比摘要</h4>
            <div style="text-align: center; margin: 20px 0; font-size: 1.2rem; color: #764ba2;">
                ${winnerText}
            </div>
            
            <div class="summary-grid">
                <div class="summary-item">
                    <div class="label">${data.project1} 健康评分</div>
                    <div class="value ${getHealthClass(summary.healthScore1)}">${summary.healthScore1 || 'N/A'}</div>
                </div>
                <div class="summary-item">
                    <div class="label">${data.project2} 健康评分</div>
                    <div class="value ${getHealthClass(summary.healthScore2)}">${summary.healthScore2 || 'N/A'}</div>
                </div>
                <div class="summary-item">
                    <div class="label">${data.project1} 层级</div>
                    <div class="value">${summary.tier1 || 'N/A'}</div>
                </div>
                <div class="summary-item">
                    <div class="label">${data.project2} 层级</div>
                    <div class="value">${summary.tier2 || 'N/A'}</div>
                </div>
            </div>
        </div>`;
    
    // 如果有图像
    if (data.results.image) {
        html += `
            <div class="result-image">
                <h4><i class="fas fa-chart-bar"></i> 对比图表</h4>
                <img src="${data.results.image}" alt="对比分析图表">
            </div>
        `;
    }
    
    // 如果有报告
    if (data.results.report) {
        html += `
            <div class="result-report">
                <h4><i class="fas fa-file-alt"></i> 详细报告</h4>
                <pre>${data.results.report}</pre>
            </div>
        `;
    }
    
    // 如果有对比结论
    if (summary.comparison && summary.comparison.length > 0) {
        html += `
            <div class="result-summary">
                <h4><i class="fas fa-lightbulb"></i> 对比结论</h4>
                <ul style="margin-left: 20px; color: #555;">
                    ${summary.comparison.map(item => `<li>${item}</li>`).join('')}
                </ul>
            </div>
        `;
    }
    
    container.innerHTML = html;
    container.style.display = 'block';
}

// 历史记录筛选
let currentFilter = 'all';
let allResults = [];

// 加载历史结果
async function loadHistoryResults() {
    try {
        const response = await fetch(`${API_BASE}/api/results`);
        const data = await response.json();
        
        if (data.success) {
            allResults = data.results;
            displayHistoryResults(allResults, currentFilter);
        }
    } catch (error) {
        console.error('加载历史结果失败:', error);
        const container = document.getElementById('results-list');
        if (container) {
            container.innerHTML = '<p style="text-align: center; color: #666;">加载历史结果失败，请稍后重试</p>';
        }
    }
}

// 显示历史结果
function displayHistoryResults(results, filter) {
    const container = document.getElementById('results-list');
    if (!container) return;
    
    if (!results || results.length === 0) {
        container.innerHTML = '<p style="text-align: center; color: #666;">暂无历史结果</p>';
        return;
    }
    
    // 根据筛选条件过滤结果
    let filteredResults = results;
    if (filter === 'single') {
        filteredResults = results.filter(result => result.type === 'single');
    } else if (filter === 'compare') {
        filteredResults = results.filter(result => result.type === 'compare');
    } else if (filter === 'multi') {
        filteredResults = results.filter(result => result.type === 'multi');
    }
    
    if (filteredResults.length === 0) {
        container.innerHTML = '<p style="text-align: center; color: #666;">暂无符合条件的历史结果</p>';
        return;
    }
    
    const html = filteredResults.map(result => {
        const date = new Date(result.timestamp).toLocaleString('zh-CN');
        
        // 确定结果类型文本
        let typeText = '';
        switch (result.type) {
            case 'single':
                typeText = '单项目分析';
                break;
            case 'compare':
                typeText = '项目对比';
                break;
            case 'multi':
                typeText = '多项目对比';
                break;
            default:
                typeText = '分析结果';
        }
        
        // 为多项目对比添加多个图片链接
        let imageLinks = '';
        if (result.type === 'multi' && result.files.images && result.files.images.length > 1) {
            // 如果是多项目对比且有多张图片，显示查看所有图片的链接
            imageLinks = `
                <div class="dropdown">
                    <button class="btn btn-secondary dropdown-toggle">
                        <i class="fas fa-image"></i> 查看图表
                    </button>
                    <div class="dropdown-menu">
                        ${result.files.images.map((img, index) => `
                            <a href="${img.url || result.files.image}" target="_blank" class="dropdown-item" onclick="event.stopPropagation();">
                                ${getReadableImageTitle(img.name || `图片${index + 1}`)}
                            </a>
                        `).join('')}
                    </div>
                </div>
            `;
        } else if (result.files.image) {
            // 否则显示单个图片链接
            imageLinks = `<a href="${result.files.image}" target="_blank" class="btn btn-secondary">
                        <i class="fas fa-image"></i> 图表
                    </a>`;
        }
        
        return `
            <div class="result-item ${result.type}">
                <div class="result-info">
                    <span class="result-type ${result.type}">${typeText}</span>
                    <span class="result-name">${result.project}</span>
                    <span class="result-date">${date}</span>
                </div>
                <div class="result-actions">
                    ${result.files.report ? `<a href="${result.files.report}" target="_blank" class="btn btn-secondary">
                        <i class="fas fa-file-alt"></i> 报告
                    </a>` : ''}
                    ${imageLinks}
                    ${result.files.csv ? `<a href="${result.files.csv}" download class="btn btn-secondary">
                        <i class="fas fa-file-csv"></i> 下载CSV
                    </a>` : ''}
                    <button class="btn btn-danger" onclick="deleteHistoryResult('${result.id}')">
                        <i class="fas fa-trash"></i> 删除
                    </button>
                </div>
            </div>
        `;
    }).join('');
    
    container.innerHTML = html;
}

// 删除历史结果
async function deleteHistoryResult(id) {
    if (confirm('确定要删除这个历史结果吗？')) {
        try {
            const response = await fetch(`${API_BASE}/api/results/${id}`, {
                method: 'DELETE'
            });
            
            const data = await response.json();
            if (data.success) {
                showMessage('删除成功', 'success');
                loadHistoryResults();
            } else {
                showMessage('删除失败', 'error');
            }
        } catch (error) {
            console.error('删除历史结果失败:', error);
            showMessage('删除失败', 'error');
        }
    }
}

// 添加筛选按钮事件监听器
document.addEventListener('DOMContentLoaded', () => {
    // 筛选按钮事件
    const filterBtns = document.querySelectorAll('.filter-btn');
    filterBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            // 更新按钮状态
            filterBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // 应用筛选
            currentFilter = btn.dataset.filter;
            displayHistoryResults(allResults, currentFilter);
        });
    });
});

// 辅助函数：获取健康评分CSS类
function getHealthClass(score) {
    if (!score) return '';
    if (score >= 80) return 'good';
    if (score >= 60) return 'warning';
    return 'danger';
}

// 辅助函数：获取活力状态CSS类
function getVitalityClass(vitality) {
    if (!vitality) return '';
    if (vitality === 'THRIVING') return 'good';
    if (vitality === 'STABLE') return 'warning';
    return 'danger';
}

// 辅助函数：获取潜力CSS类
function getPotentialClass(potential) {
    if (!potential) return '';
    if (potential >= 50) return 'good';
    if (potential >= 20) return 'warning';
    return 'danger';
}

// 下载报告
function downloadReport(filename, content) {
    if (!content) {
        showMessage('没有可下载的内容', 'error');
        return;
    }
    
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${filename}_分析报告.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// 复制到剪贴板
function copyToClipboard(text) {
    if (!text) {
        showMessage('没有可复制的内容', 'error');
        return;
    }
    
    navigator.clipboard.writeText(text).then(() => {
        showMessage('已复制到剪贴板', 'success');
    }).catch(err => {
        console.error('复制失败:', err);
        showMessage('复制失败', 'error');
    });
}

// 显示多项目对比结果
function displayMultiComparisonResults(data) {
    const container = document.getElementById('multi-compare-results');
    if (!container) return;
    const summary = data.summary || {};
    const projects = data.projects || [];
    
    // 构建结果HTML
    let html = `
        <div class="result-header">
            <h3><i class="fas fa-code-branch"></i> 多项目对比结果</h3>
            <div class="result-actions">`;
    
    // 报告下载
    if (data.results.report) {
        html += `<button class="btn btn-secondary" onclick="downloadReport('多项目对比报告', '${data.results.report}')">
                    <i class="fas fa-download"></i> 下载报告
                </button>`;
    }
    
    // 图片下载
    if (data.results.imageUrl) {
        html += `<a href="${data.results.imageUrl}" download class="btn btn-secondary">
                    <i class="fas fa-image"></i> 下载主图
                </a>`;
    }
    
    // CSV下载
    if (data.results.csvUrl) {
        html += `<a href="${data.results.csvUrl}" download class="btn btn-secondary">
                    <i class="fas fa-file-csv"></i> 下载CSV
                </a>`;
    }
    
    // 复制报告
    if (data.results.report) {
        html += `<button class="btn btn-secondary" onclick="copyToClipboard('${data.results.report}')">
                    <i class="fas fa-copy"></i> 复制报告
                </button>`;
    }
    
    html += `</div></div>
        
        <div class="result-summary">
            <h4><i class="fas fa-info-circle"></i> 对比摘要</h4>
            <div class="summary-grid">
                <div class="summary-item">
                    <div class="label">参与对比项目数</div>
                    <div class="value">${projects.length}</div>
                </div>`;
    
    // 最佳项目
    if (summary.bestProject) {
        html += `<div class="summary-item">
                    <div class="label">最佳项目</div>
                    <div class="value good">${summary.bestProject}</div>
                </div>`;
    }
    
    // 平均健康评分
    if (summary.averageHealthScore) {
        html += `<div class="summary-item">
                    <div class="label">平均健康评分</div>
                    <div class="value ${getHealthClass(summary.averageHealthScore)}">${summary.averageHealthScore.toFixed(2)}</div>
                </div>`;
    }
    
    html += `</div></div>`;
    
    // 如果有多张图像，显示所有图像
    if (data.results.images && data.results.images.length > 0) {
        html += `
            <div class="result-image">
                <h4><i class="fas fa-chart-bar"></i> 多项目对比图表</h4>
                <div class="image-gallery">`;
        
        data.results.images.forEach((image, index) => {
            html += `
                <div class="image-item">
                    <h5>${getReadableImageTitle(image.name)}</h5>
                    <img src="${image.data}" alt="${image.name}">
                    <a href="${image.url}" download class="btn btn-secondary btn-small">
                        <i class="fas fa-download"></i> 下载
                    </a>
                </div>`;
        });
        
        html += `
                </div>
            </div>
        `;
    }
    // 如果只有单张图像（向后兼容）
    else if (data.results.image) {
        html += `
            <div class="result-image">
                <h4><i class="fas fa-chart-bar"></i> 多项目对比图表</h4>
                <img src="${data.results.image}" alt="多项目对比分析图表">
            </div>
        `;
    }
    
    // 如果有报告
    if (data.results.report) {
        html += `
            <div class="result-report">
                <h4><i class="fas fa-file-alt"></i> 详细报告</h4>
                <pre>${data.results.report}</pre>
            </div>
        `;
    }
    
    // 如果有项目列表
    if (projects.length > 0) {
        html += `
            <div class="result-summary">
                <h4><i class="fas fa-list"></i> 参与对比项目</h4>
                <ul style="margin-left: 20px; color: #555;">
                    ${projects.map(project => `<li>${project}</li>`).join('')}
                </ul>
            </div>
        `;
    }
    
    container.innerHTML = html;
    container.style.display = 'block';
}

// 获取图像标题的可读版本
function getReadableImageTitle(imageName) {
    const titleMap = {
        'multi_trend_comparison.png': '发展趋势对比图',
        'multi_health_comparison.png': '健康程度对比图',
        'multi_stars_trend_comparison.png': 'Stars趋势对比图',
        'multi_openrank_trend_comparison.png': 'OpenRank趋势对比图',
        'multi_attention_trend_comparison.png': 'Attention趋势对比图',
        'multi_triple_metric_comparison.png': '三指标对比图'
    };
    
    return titleMap[imageName] || imageName.replace('.png', '').replace(/_/g, ' ');
}

// 检查后端连接
async function checkBackendConnection() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        if (response.ok) {
            console.log('后端连接正常');
        }
    } catch (error) {
        console.warn('后端连接失败，请确保后端服务正在运行');
    }
}