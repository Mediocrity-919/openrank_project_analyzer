const express = require('express');
const cors = require('cors');
const { exec } = require('child_process');
const path = require('path');
const fs = require('fs');
const util = require('util');

// 加载环境变量
require('dotenv').config();

const app = express();
const PORT = parseInt(process.env.PORT) || 5501;

const CONFIG = {
    pythonPath: process.env.PYTHON_PATH || 'python',
    scriptTimeout: 600000, 
    maxFileSize: 10 * 1024 * 1024, 
    uploadDir: path.join(__dirname, 'uploads'),
    resultsDir: path.join(__dirname, 'results'),
    historyFile: path.join(__dirname, 'results', 'history.json')
};

// 确保目录存在
[CONFIG.uploadDir, CONFIG.resultsDir].forEach(dir => {
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
});

// 确保历史记录文件存在
if (!fs.existsSync(CONFIG.historyFile)) {
    fs.writeFileSync(CONFIG.historyFile, JSON.stringify([], null, 2), 'utf-8');
}

// 历史记录相关函数

// 读取历史记录
function readHistory() {
    try {
        // 检查文件是否存在
        if (!fs.existsSync(CONFIG.historyFile)) {
            // 如果文件不存在，返回空数组
            return [];
        }
        
        // 读取文件内容
        let data = fs.readFileSync(CONFIG.historyFile, 'utf-8');
        
        // 移除BOM（字节顺序标记）
        if (data.charCodeAt(0) === 0xFEFF) {
            data = data.slice(1);
        }
        
        // 尝试解析JSON
        const parsed = JSON.parse(data);
        
        // 确保返回的是数组
        return Array.isArray(parsed) ? parsed : [];
    } catch (error) {
        console.error('读取历史记录失败:', error);
        // 如果解析失败，尝试重新创建文件
        try {
            fs.writeFileSync(CONFIG.historyFile, JSON.stringify([], null, 2), 'utf-8');
            console.log('已重新创建历史记录文件');
        } catch (writeError) {
            console.error('重新创建历史记录文件失败:', writeError);
        }
        return [];
    }
}

// 保存历史记录
function saveHistory(history) {
    try {
        fs.writeFileSync(CONFIG.historyFile, JSON.stringify(history, null, 2), 'utf-8');
        return true;
    } catch (error) {
        console.error('保存历史记录失败:', error);
        return false;
    }
}

// 添加历史记录
function addHistoryResult(result) {
    const history = readHistory();
    history.unshift(result); // 最新的结果放在前面
    return saveHistory(history);
}

// 删除历史记录
function deleteHistoryResult(id) {
    const history = readHistory();
    const newHistory = history.filter(item => item.id !== id);
    return saveHistory(newHistory);
}

// 获取单个历史记录
function getHistoryResult(id) {
    const history = readHistory();
    return history.find(item => item.id === id);
}

// 中间件
app.use(cors());
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// 静态文件服务
app.use('/results', express.static(CONFIG.resultsDir));
// 添加前端静态文件服务
app.use(express.static(path.join(__dirname, '../frontend')));

// 健康检查
app.get('/health', (req, res) => {
    res.json({ status: 'OK', timestamp: new Date().toISOString() });
});

// 根路径重定向到前端首页
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, '../frontend/index.html'));
});

// 运行Python脚本的通用函数
const runPythonScript = async (scriptPath, args, timeout = CONFIG.scriptTimeout) => {
    return new Promise((resolve, reject) => {
        // 清理参数，移除所有反引号和多余空格
        const cleanedArgs = args.map(arg => {
            // 移除所有反引号
            let cleaned = arg.replace(/`/g, '');
            // 移除多余空格
            cleaned = cleaned.replace(/\s+/g, ' ').trim();
            return cleaned;
        });
        
        const command = `${CONFIG.pythonPath} "${scriptPath}" ${cleanedArgs.map(arg => `"${arg}"`).join(' ')}`;
        console.log(`执行命令: ${command}`);
        
        const child = exec(command, { 
            timeout,
            maxBuffer: CONFIG.maxFileSize,
            cwd: path.join(__dirname, '../python') // 设置工作目录为python目录
        }, (error, stdout, stderr) => {
            if (error) {
                console.error(`执行错误: ${error.message}`);
                // 过滤掉已知的警告信息，只保留真正的错误
                const filteredStderr = stderr
                    .replace(/Duplicate key in file WindowsPath[^\n]+\n/g, '')
                    .replace(/Importing plotly failed[^\n]+\n/g, '')
                    .replace(/\d+:\d+:\d+ - cmdstanpy - INFO - Chain \[\d+\] [^\n]+\n/g, '')
                    .trim();
                
                if (filteredStderr) {
                    console.error(`过滤后错误: ${filteredStderr}`);
                    reject({ error: error.message, stderr: filteredStderr });
                } else {
                    // 如果只有警告，没有真正的错误，就不拒绝
                    resolve({ stdout, stderr: '' });
                }
                return;
            }
            
            resolve({ stdout, stderr });
        });
    });
};

// 1. 分析单个项目
app.post('/api/analyze', async (req, res) => {
    try {
        const { projectUrl, githubToken } = req.body;
        
        if (!projectUrl) {
            return res.status(400).json({ error: '请输入项目URL' });
        }

        // 提取组织和仓库名
        const match = projectUrl.match(/github\.com\/([^\/]+)\/([^\/]+)/);
        if (!match) {
            return res.status(400).json({ error: '无效的GitHub URL' });
        }

        const org = match[1];
        const repo = match[2].replace('.git', '');
        
        console.log(`开始分析: ${org}/${repo}`);
        
        // 准备参数
        const args = [projectUrl];
        if (githubToken) {
            args.push(githubToken);
        }

        // 清理可能存在的旧文件
        const baseName = `${org}_${repo}`;
        const oldFiles = [
            `${baseName}_v45_report.png`,
            `${baseName}_v45_report.txt`,
            `${baseName}_v45_data.json`
        ];
        
        oldFiles.forEach(file => {
            const filePath = path.join(CONFIG.resultsDir, file);
            if (fs.existsSync(filePath)) {
                fs.unlinkSync(filePath);
            }
        });

        // 运行分析脚本
        const scriptPath = path.join(__dirname, '../python/v4.py');
        console.log('运行Python脚本...');
        
        const result = await runPythonScript(scriptPath, args);
        
        console.log('Python脚本执行完成');
        
        // 查找生成的文件
        const results = {
            image: null,
            report: null,
            data: null,
            imageUrl: null,
            csvUrl: null
        };

        // 查找图像文件
        const possibleImageNames = [
            `${baseName}_v45_report.png`,
            `${baseName}_report.png`,
            `v45_report.png`
        ];

        for (const imageName of possibleImageNames) {
            const imagePath = path.join(CONFIG.resultsDir, imageName);
            if (fs.existsSync(imagePath)) {
                const imageBuffer = fs.readFileSync(imagePath);
                results.image = `data:image/png;base64,${imageBuffer.toString('base64')}`;
                results.imageUrl = `/results/${imageName}`;
                break;
            }
        }

        // 查找报告文件
        const reportPath = path.join(CONFIG.resultsDir, `${baseName}_v45_report.txt`);
        if (fs.existsSync(reportPath)) {
            results.report = fs.readFileSync(reportPath, 'utf-8');
            results.reportUrl = `/results/${baseName}_v45_report.txt`;
        }

        // 查找数据文件
        const dataPath = path.join(CONFIG.resultsDir, `${baseName}_v45_data.json`);
        if (fs.existsSync(dataPath)) {
            try {
                results.data = JSON.parse(fs.readFileSync(dataPath, 'utf-8'));
                results.dataUrl = `/results/${baseName}_v45_data.json`;
            } catch (e) {
                console.error('解析JSON数据失败:', e);
            }
        }

        // 查找CSV文件
        const possibleCsvNames = [
            `${baseName}_data.csv`,
            `${baseName}_v45_data.csv`,
            `${baseName}.csv`
        ];

        for (const csvName of possibleCsvNames) {
            const csvPath = path.join(CONFIG.resultsDir, csvName);
            if (fs.existsSync(csvPath)) {
                results.csvUrl = `/results/${csvName}`;
                break;
            }
        }

        // 如果没有找到任何结果文件，尝试在脚本目录查找
        if (!results.image || !results.report || !results.csvUrl) {
            const scriptDir = path.join(__dirname, '../python');
            
            // 查找图像
            if (!results.image) {
                for (const imageName of possibleImageNames) {
                    const imagePath = path.join(scriptDir, imageName);
                    if (fs.existsSync(imagePath)) {
                        // 复制到结果目录
                        const destPath = path.join(CONFIG.resultsDir, imageName);
                        fs.copyFileSync(imagePath, destPath);
                        
                        const imageBuffer = fs.readFileSync(destPath);
                        results.image = `data:image/png;base64,${imageBuffer.toString('base64')}`;
                        results.imageUrl = `/results/${imageName}`;
                        break;
                    }
                }
            }
            
            // 查找报告
            if (!results.report) {
                const scriptReportPath = path.join(scriptDir, `${baseName}_v45_report.txt`);
                if (fs.existsSync(scriptReportPath)) {
                    // 复制到结果目录
                    const destPath = path.join(CONFIG.resultsDir, `${baseName}_v45_report.txt`);
                    fs.copyFileSync(scriptReportPath, destPath);
                    
                    results.report = fs.readFileSync(destPath, 'utf-8');
                    results.reportUrl = `/results/${baseName}_v45_report.txt`;
                }
            }
            
            // 查找CSV文件
            if (!results.csvUrl) {
                for (const csvName of possibleCsvNames) {
                    const csvPath = path.join(scriptDir, csvName);
                    if (fs.existsSync(csvPath)) {
                        // 复制到结果目录
                        const destPath = path.join(CONFIG.resultsDir, csvName);
                        fs.copyFileSync(csvPath, destPath);
                        
                        results.csvUrl = `/results/${csvName}`;
                        break;
                    }
                }
            }
        }

        // 从报告中提取关键信息
        const analysisSummary = extractAnalysisSummary(results.report, org, repo);
        
        // 保存到历史记录
        const historyResult = {
            id: Date.now().toString() + Math.random().toString(36).substr(2, 5),
            type: 'single',
            project: `${org}/${repo}`,
            projects: [`${org}/${repo}`],
            timestamp: new Date().toISOString(),
            summary: analysisSummary,
            files: {
                image: results.imageUrl,
                report: results.reportUrl,
                csv: results.csvUrl
            }
        };
        
        addHistoryResult(historyResult);
        
        res.json({
            success: true,
            project: `${org}/${repo}`,
            results,
            summary: analysisSummary,
            timestamp: new Date().toISOString()
        });

    } catch (error) {
        console.error('分析错误:', error);
        res.status(500).json({ 
            error: '分析失败', 
            message: error.message || '未知错误'
        });
    }
});

// 2. 对比两个项目
app.post('/api/compare', async (req, res) => {
    try {
        const { project1Url, project2Url, githubToken } = req.body;
        
        if (!project1Url || !project2Url) {
            return res.status(400).json({ error: '请输入两个项目URL' });
        }

        // 提取组织和仓库名
        const match1 = project1Url.match(/github\.com\/([^\/]+)\/([^\/]+)/);
        const match2 = project2Url.match(/github\.com\/([^\/]+)\/([^\/]+)/);
        
        if (!match1 || !match2) {
            return res.status(400).json({ error: '无效的GitHub URL格式' });
        }

        const org1 = match1[1];
        const repo1 = match1[2].replace('.git', '');
        const org2 = match2[1];
        const repo2 = match2[2].replace('.git', '');
        
        console.log(`开始对比: ${org1}/${repo1} vs ${org2}/${repo2}`);
        
        // 准备参数
        const args = [project1Url, project2Url];
        if (githubToken) {
            args.push(githubToken);
        }

        // 清理可能存在的旧文件
        const baseName = `compare_${org1}_${repo1}_vs_${org2}_${repo2}`;
        const oldFiles = [
            `${baseName}_dashboard.png`,
            `${baseName}_report.txt`
        ];
        
        oldFiles.forEach(file => {
            const filePath = path.join(CONFIG.resultsDir, file);
            if (fs.existsSync(filePath)) {
                fs.unlinkSync(filePath);
            }
        });

        // 运行对比脚本
        const scriptPath = path.join(__dirname, '../python/compare.py');
        console.log('运行Python对比脚本...');
        
        const result = await runPythonScript(scriptPath, args);
        
        console.log('Python对比脚本执行完成');
        
        // 查找生成的文件
        const results = {
            image: null,
            report: null,
            imageUrl: null,
            csvUrl: null
        };

        // 查找图像文件
        const possibleImageNames = [
            `${baseName}_dashboard.png`,
            `compare_${org1}_${repo1}_vs_${org2}_${repo2}.png`,
            'compare_dashboard.png'
        ];

        for (const imageName of possibleImageNames) {
            const imagePath = path.join(CONFIG.resultsDir, imageName);
            if (fs.existsSync(imagePath)) {
                const imageBuffer = fs.readFileSync(imagePath);
                results.image = `data:image/png;base64,${imageBuffer.toString('base64')}`;
                results.imageUrl = `/results/${imageName}`;
                break;
            }
        }

        // 查找报告文件
        const reportPath = path.join(CONFIG.resultsDir, `${baseName}_report.txt`);
        if (fs.existsSync(reportPath)) {
            results.report = fs.readFileSync(reportPath, 'utf-8');
            results.reportUrl = `/results/${baseName}_report.txt`;
        }

        // 查找CSV文件
        const possibleCsvNames = [
            `${baseName}_data.csv`,
            `compare_${org1}_${repo1}_vs_${org2}_${repo2}.csv`,
            `compare_data.csv`
        ];

        for (const csvName of possibleCsvNames) {
            const csvPath = path.join(CONFIG.resultsDir, csvName);
            if (fs.existsSync(csvPath)) {
                results.csvUrl = `/results/${csvName}`;
                break;
            }
        }

        // 如果没有找到任何结果文件，尝试在脚本目录查找
        if (!results.image || !results.report || !results.csvUrl) {
            const scriptDir = path.join(__dirname, '../python');
            
            // 查找图像
            if (!results.image) {
                for (const imageName of possibleImageNames) {
                    const imagePath = path.join(scriptDir, imageName);
                    if (fs.existsSync(imagePath)) {
                        // 复制到结果目录
                        const destPath = path.join(CONFIG.resultsDir, imageName);
                        fs.copyFileSync(imagePath, destPath);
                        
                        const imageBuffer = fs.readFileSync(destPath);
                        results.image = `data:image/png;base64,${imageBuffer.toString('base64')}`;
                        results.imageUrl = `/results/${imageName}`;
                        break;
                    }
                }
            }
            
            // 查找报告
            if (!results.report) {
                const scriptReportPath = path.join(scriptDir, `${baseName}_report.txt`);
                if (fs.existsSync(scriptReportPath)) {
                    // 复制到结果目录
                    const destPath = path.join(CONFIG.resultsDir, `${baseName}_report.txt`);
                    fs.copyFileSync(scriptReportPath, destPath);
                    
                    results.report = fs.readFileSync(destPath, 'utf-8');
                    results.reportUrl = `/results/${baseName}_report.txt`;
                }
            }
            
            // 查找CSV文件
            if (!results.csvUrl) {
                for (const csvName of possibleCsvNames) {
                    const csvPath = path.join(scriptDir, csvName);
                    if (fs.existsSync(csvPath)) {
                        // 复制到结果目录
                        const destPath = path.join(CONFIG.resultsDir, csvName);
                        fs.copyFileSync(csvPath, destPath);
                        
                        results.csvUrl = `/results/${csvName}`;
                        break;
                    }
                }
            }
        }

        // 从报告中提取关键对比信息
        const comparisonSummary = extractComparisonSummary(results.report, org1, repo1, org2, repo2);
        
        // 保存到历史记录
        const historyResult = {
            id: Date.now().toString() + Math.random().toString(36).substr(2, 5),
            type: 'compare',
            project: `${org1}/${repo1} vs ${org2}/${repo2}`,
            projects: [`${org1}/${repo1}`, `${org2}/${repo2}`],
            timestamp: new Date().toISOString(),
            summary: comparisonSummary,
            files: {
                image: results.imageUrl,
                report: results.reportUrl,
                csv: results.csvUrl
            }
        };
        
        addHistoryResult(historyResult);
        
        res.json({
            success: true,
            project1: `${org1}/${repo1}`,
            project2: `${org2}/${repo2}`,
            results,
            summary: comparisonSummary,
            timestamp: new Date().toISOString()
        });

    } catch (error) {
        console.error('对比错误:', error);
        res.status(500).json({ 
            error: '对比失败', 
            message: error.message || '未知错误'
        });
    }
});

// 3. 获取历史结果
app.get('/api/results', (req, res) => {
    try {
        const history = readHistory();
        res.json({ success: true, results: history });
    } catch (error) {
        console.error('获取历史结果失败:', error);
        res.status(500).json({ error: '获取结果失败' });
    }
});

// 获取单个历史结果
app.get('/api/results/:id', (req, res) => {
    try {
        const result = getHistoryResult(req.params.id);
        if (result) {
            res.json({ success: true, result });
        } else {
            res.status(404).json({ error: '结果不存在' });
        }
    } catch (error) {
        console.error('获取单个结果失败:', error);
        res.status(500).json({ error: '获取结果失败' });
    }
});

// 删除单个历史结果
app.delete('/api/results/:id', (req, res) => {
    try {
        const success = deleteHistoryResult(req.params.id);
        if (success) {
            res.json({ success: true, message: '删除成功' });
        } else {
            res.status(404).json({ error: '结果不存在' });
        }
    } catch (error) {
        console.error('删除结果失败:', error);
        res.status(500).json({ error: '删除失败' });
    }
});

// 3. 多项目对比
app.post('/api/multi-compare', async (req, res) => {
    try {
        const { projectUrls, githubToken } = req.body;
        
        if (!projectUrls || projectUrls.length < 2) {
            return res.status(400).json({ error: '请输入至少两个项目URL' });
        }

        // 提取组织和仓库名，用于生成文件名
        const projectNames = projectUrls.map(url => {
            const match = url.match(/github\.com\/([^\/]+)\/([^\/]+)/);
            if (match) {
                return `${match[1]}_${match[2].replace('.git', '')}`;
            }
            return null;
        }).filter(Boolean);

        if (projectNames.length === 0) {
            return res.status(400).json({ error: '无效的GitHub URL格式' });
        }
        
        console.log(`开始多项目对比: ${projectNames.join(', ')}`);
        
        // 准备参数
        const args = [...projectUrls];
        if (githubToken) {
            args.push(githubToken);
        }

        // 清理可能存在的旧文件
        const baseName = `multi_${projectNames.join('_')}`;
        const oldFiles = [
            `${baseName}_comparison_report.txt`,
            `${baseName}_comparison.png`,
            `${baseName}_dashboard.png`,
            `${baseName}_report.txt`,
            `${baseName}_data.csv`
        ];
        
        oldFiles.forEach(file => {
            const filePath = path.join(CONFIG.resultsDir, file);
            if (fs.existsSync(filePath)) {
                fs.unlinkSync(filePath);
            }
        });

        // 运行多项目对比脚本
        const scriptPath = path.join(__dirname, '../python/multi_compare.py');
        console.log('运行Python多项目对比脚本...');
        
        const result = await runPythonScript(scriptPath, args);
        
        console.log('Python多项目对比脚本执行完成');
        
        // 查找生成的文件
        const results = {
            image: null,
            report: null,
            csv: null,
            imageUrl: null,
            csvUrl: null
        };

        // 查找报告文件（优先查找带时间戳的）
        const possibleReportNames = [
            // 带时间戳的报告文件（新的格式，优先）
            `multi_project_comparison_report.txt`,
            `${baseName}_report.txt`,
            `${baseName}_comparison_report.txt`
        ];

        // 首先尝试查找带时间戳的最新文件
        const scriptDir = path.join(__dirname, '../python');
        const multiOutputDir = path.join(scriptDir, 'multi_compare_output');

        // 首先查找带时间戳的报告文件
        let foundReport = false;
        if (fs.existsSync(multiOutputDir)) {
            const files = fs.readdirSync(multiOutputDir);
            // 查找最新的带时间戳的报告文件
            const timestampedReports = files
                .filter(file => file.startsWith('20') && file.includes('_multi_project_comparison_report.txt'))
                .sort()
                .reverse(); // 最新的在前面
            
            if (timestampedReports.length > 0) {
                const latestReport = timestampedReports[0];
                const reportPath = path.join(multiOutputDir, latestReport);
                
                // 复制到结果目录
                const destPath = path.join(CONFIG.resultsDir, latestReport);
                fs.copyFileSync(reportPath, destPath);
                
                results.report = fs.readFileSync(destPath, 'utf-8');
                results.reportUrl = `/results/${latestReport}`;
                foundReport = true;
            }
        }
        
        // 如果没有找到带时间戳的报告，尝试找固定名称的报告（向后兼容）
        if (!foundReport) {
            for (const reportName of possibleReportNames) {
                const reportPath = path.join(CONFIG.resultsDir, reportName);
                if (fs.existsSync(reportPath)) {
                    results.report = fs.readFileSync(reportPath, 'utf-8');
                    results.reportUrl = `/results/${reportName}`;
                    break;
                }
            }
        }
        
        // 查找CSV文件（如果之前没有找到，尝试从脚本目录或multi_compare_output目录）
        if (!results.csvUrl) {
            const possibleCsvNames = [
                `${baseName}_data.csv`,
                `multi_project_data.csv`
            ];
            
            for (const csvName of possibleCsvNames) {
                // 首先尝试在结果目录查找
                let csvPath = path.join(CONFIG.resultsDir, csvName);
                let sourceDir = 'results';
                
                // 如果结果目录中不存在，尝试在scriptDir查找
                if (!fs.existsSync(csvPath)) {
                    csvPath = path.join(scriptDir, csvName);
                    sourceDir = 'script';
                }
                
                // 如果scriptDir中也不存在，尝试在multiOutputDir查找
                if (!fs.existsSync(csvPath) && fs.existsSync(multiOutputDir)) {
                    csvPath = path.join(multiOutputDir, csvName);
                    sourceDir = 'multiOutput';
                }
                
                if (fs.existsSync(csvPath)) {
                    // 如果文件在其他目录，需要复制到结果目录
                    if (sourceDir !== 'results') {
                        const destPath = path.join(CONFIG.resultsDir, csvName);
                        fs.copyFileSync(csvPath, destPath);
                    }
                    
                    results.csvUrl = `/results/${csvName}`;
                    break;
                }
            }
        }
        
        // 查找所有可能的图像文件（包含时间戳和固定名称）
        const possibleImagePatterns = [
            // 带时间戳的文件（新的格式）
            { pattern: 'multi_trend_comparison.png', type: 'trend' },
            { pattern: 'multi_health_comparison.png', type: 'health' },
            { pattern: 'multi_stars_trend_comparison.png', type: 'stars' },
            { pattern: 'multi_openrank_trend_comparison.png', type: 'openrank' },
            { pattern: 'multi_attention_trend_comparison.png', type: 'attention' },
            { pattern: 'multi_triple_metric_comparison.png', type: 'triple' },
            // 兼容旧格式
            { pattern: `${baseName}_comparison.png`, type: 'comparison' },
            { pattern: `${baseName}_dashboard.png`, type: 'dashboard' }
        ];

        // 查找所有可能的图像文件
        results.images = []; // 存储所有图像
        
        // 首先查找带时间戳的文件（优先）- 简化处理逻辑
        const timestampedImages = [];
        if (fs.existsSync(multiOutputDir)) {
            const files = fs.readdirSync(multiOutputDir);
            console.log(`[DEBUG] 在multi_compare_output目录中找到 ${files.length} 个文件`);
            files.forEach(file => {
                // 检查是否是multi_compare_output目录中的图片文件
                if (file.startsWith('20') && file.includes('_multi_') && file.endsWith('.png')) {
                    timestampedImages.push(file);
                }
            });
            console.log(`[DEBUG] 匹配的图片文件: ${timestampedImages.length} 个`);
            timestampedImages.forEach(f => console.log(`  - ${f}`));
        } else {
            console.log(`[DEBUG] multi_compare_output目录不存在`);
        }
        
        // 按时间戳分组处理图片，确保处理最新批次的所有图片
        if (timestampedImages.length === 0) {
            console.log('[DEBUG] 没有找到任何图片文件，跳过图片处理');
        } else {
            // 修复时间戳提取逻辑 - 提取完整的日期时间戳部分
            const timestampFromFilename = (filename) => {
                const match = filename.match(/^([0-9]{8}_[0-9]{6})/);
                return match ? match[1] : null;
            };
            
            const timestamps = timestampedImages
                .map(f => timestampFromFilename(f))
                .filter(ts => ts !== null)
                .sort()
                .reverse();
            
            const latestTimestamp = timestamps[0]; // 获取最新时间戳
            
            console.log(`[DEBUG] 找到 ${timestampedImages.length} 个图片文件，最新时间戳: ${latestTimestamp}`);
            
            // 处理最新时间戳的所有图片
            const latestBatchImages = timestampedImages.filter(f => f.startsWith(latestTimestamp));
            console.log(`[DEBUG] 处理最新批次图片数量: ${latestBatchImages.length}`);
            latestBatchImages.forEach(f => console.log(`  - ${f}`));
            
            latestBatchImages.forEach(file => {
                const imagePath = path.join(multiOutputDir, file);
                console.log(`[DEBUG] 处理文件: ${file}, 路径: ${imagePath}`);
                if (fs.existsSync(imagePath)) {
                    // 提取简洁的文件名，但保留时间戳后缀避免冲突
                    let cleanFileName = file;
                    // timestamp变量已定义在上面
                    
                        if (file.startsWith(latestTimestamp) && file.includes('_multi_')) {
                            // 提取时间戳后的部分作为文件名
                            const suffix = file.substring(latestTimestamp.length + 1); // +1 for underscore
                            cleanFileName = suffix;
                        }
                    
                        // 复制到结果目录（使用简洁文件名，但添加时间戳区分）
                        const destFileName = `${latestTimestamp}_${cleanFileName}`;
                        const destPath = path.join(CONFIG.resultsDir, destFileName);
                        console.log(`[DEBUG] 复制到: ${destPath}`);
                        fs.copyFileSync(imagePath, destPath);
                    
                        const imageBuffer = fs.readFileSync(destPath);
                        const imageData = `data:image/png;base64,${imageBuffer.toString('base64')}`;
                        const imageUrl = `/results/${destFileName}`;
                    
                        // 确定图片类型
                        let imageType = 'unknown';
                        for (const pattern of possibleImagePatterns) {
                            if (cleanFileName.includes(pattern.pattern.replace('.png', '').replace('multi_', ''))) {
                                imageType = pattern.type;
                                break;
                            }
                        }
                    
                        // 添加到图像数组（使用简洁名称作为显示名）
                        results.images.push({
                            data: imageData,
                            url: imageUrl,
                            name: cleanFileName,  // 显示用简洁名
                            fileName: destFileName,  // 实际文件名（带时间戳）
                            type: imageType,
                            originalFile: file    // 保留原始文件名用于调试
                        });
                    
                        console.log(`[DEBUG] 添加图片到结果数组: ${cleanFileName} (类型: ${imageType})`);
                    
                        // 保留第一张图作为主图
                        if (!results.image) {
                            results.image = imageData;
                            results.imageUrl = imageUrl;
                            console.log(`[DEBUG] 设置为主图: ${cleanFileName}`);
                        }
                } else {
                    console.log(`[DEBUG] 文件不存在: ${imagePath}`);
                }
            });
            
            console.log(`[DEBUG] 最终添加到结果数组的图片数量: ${results.images.length}`);
            results.images.forEach((img, index) => {
                console.log(`  ${index + 1}. ${img.name} (${img.type})`);
            });
        }
        
        // 如果没有找到带时间戳的图片，尝试找固定名称的图片（向后兼容）
        if (results.images.length === 0) {
            console.log('[DEBUG] 没有找到带时间戳的图片，尝试向后兼容模式');
            for (const imageName of possibleImagePatterns.map(p => p.pattern)) {
                const imagePath = path.join(CONFIG.resultsDir, imageName);
                if (fs.existsSync(imagePath)) {
                    const imageBuffer = fs.readFileSync(imagePath);
                    const imageData = `data:image/png;base64,${imageBuffer.toString('base64')}`;
                    const imageUrl = `/results/${imageName}`;
                    
                    // 添加到图像数组
                    results.images.push({
                        data: imageData,
                        url: imageUrl,
                        name: imageName
                    });
                    
                    // 保留第一张图作为主图
                    if (!results.image) {
                        results.image = imageData;
                        results.imageUrl = imageUrl;
                    }
                }
            }
        }

        // 从报告中提取关键信息
        const comparisonSummary = extractMultiComparisonSummary(results.report, projectNames);
        
        // 保存到历史记录
        const historyResult = {
            id: Date.now().toString() + Math.random().toString(36).substr(2, 5),
            type: 'multi',
            project: `${projectNames.map(name => name.replace('_', '/')).join(', ')}`,
            projects: projectNames.map(name => name.replace('_', '/')),
            timestamp: new Date().toISOString(),
            summary: comparisonSummary,
            files: {
                image: results.imageUrl,
                images: results.images,  // 添加完整的图片数组
                report: results.reportUrl,
                csv: results.csvUrl
            }
        };
        
        addHistoryResult(historyResult);
        
        res.json({
            success: true,
            projects: projectNames.map(name => name.replace('_', '/')),
            results,
            summary: comparisonSummary,
            timestamp: new Date().toISOString()
        });

    } catch (error) {
        console.error('多项目对比错误:', error);
        res.status(500).json({ 
            error: '多项目对比失败', 
            message: error.message || '未知错误'
        });
    }
});

// 4. 清除结果文件
app.delete('/api/results', (req, res) => {
    try {
        const files = fs.readdirSync(CONFIG.resultsDir);
        let deletedCount = 0;
        
        files.forEach(file => {
            const filePath = path.join(CONFIG.resultsDir, file);
            fs.unlinkSync(filePath);
            deletedCount++;
        });
        
        res.json({ 
            success: true, 
            message: `已清除 ${deletedCount} 个文件` 
        });
    } catch (error) {
        res.status(500).json({ error: '清除失败' });
    }
});

// 辅助函数：从报告中提取关键信息
function extractAnalysisSummary(report, org, repo) {
    if (!report) return null;
    
    const summary = {
        healthScore: null,
        healthGrade: null,
        tier: null,
        vitality: null,
        momentum: null,
        potential: null,
        recommendations: []
    };
    
    try {
        // 提取健康评分和等级
        const healthMatch = report.match(/综合评分:\s+([\d.]+)\/100\s+\(([^)]+)\)/);
        if (healthMatch) {
            summary.healthScore = parseFloat(healthMatch[1]);
            summary.healthGrade = healthMatch[2];
        }
        
        // 提取层级
        const tierMatch = report.match(/层级:\s+(\w+)\s+\(([^)]+)\)/);
        if (tierMatch) {
            summary.tier = tierMatch[1];
        }
        
        // 提取活力状态
        const vitalityMatch = report.match(/当前状态:\s+(\w+)/);
        if (vitalityMatch) {
            summary.vitality = vitalityMatch[1];
        }
        
        // 提取动量
        const momentumMatch = report.match(/动量\s*\(Momentum\):\s+([\d.]+)/);
        if (momentumMatch) {
            summary.momentum = parseFloat(momentumMatch[1]);
        }
        
        // 提取潜力
        const potentialMatch = report.match(/潜力\s*\(Potential\):\s+([\d.]+)%/);
        if (potentialMatch) {
            summary.potential = parseFloat(potentialMatch[1]);
        }
        
        // 提取建议（简单版本）
        const lines = report.split('\n');
        for (let i = 0; i < lines.length; i++) {
            if (lines[i].includes('改进建议') || lines[i].includes('建议')) {
                for (let j = i + 1; j < Math.min(i + 6, lines.length); j++) {
                    if (lines[j].trim() && lines[j].match(/^\s*\d+\./)) {
                        summary.recommendations.push(lines[j].trim());
                    }
                }
                break;
            }
        }
        
    } catch (error) {
        console.error('提取分析摘要失败:', error);
    }
    
    return summary;
}

// 辅助函数：从对比报告中提取关键信息
function extractComparisonSummary(report, org1, repo1, org2, repo2) {
    if (!report) return null;
    
    const summary = {
        healthScore1: null,
        healthScore2: null,
        tier1: null,
        tier2: null,
        vitality1: null,
        vitality2: null,
        winner: null,
        comparison: []
    };
    
    try {
        const lines = report.split('\n');
        
        // 放宽匹配条件，不要求项目名称和健康评分在同一行
        // 先提取所有健康评分和层级信息
        const healthScores = [];
        const tiers = [];
        
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            
            // 提取所有健康评分，不管在哪一行
            const healthMatch = line.match(/([\d.]+)\/100/);
            if (healthMatch) {
                healthScores.push(parseFloat(healthMatch[1]));
            }
            
            // 提取所有层级信息，不管在哪一行
            const tierMatch = line.match(/层级[:：]\s*(\w+)/);
            if (tierMatch) {
                tiers.push(tierMatch[1]);
            }
            
            // 提取活力状态
            const vitalityMatch = line.match(/当前状态[:：]\s*(\w+)/);
            if (vitalityMatch) {
                tiers.push(vitalityMatch[1]);
            }
            
            // 对比结果
            if (line.includes('领先') || line.includes('优势') || line.includes('更健康')) {
                summary.comparison.push(line.trim());
                
                if (line.includes(`${org1}/${repo1}`)) {
                    summary.winner = 'project1';
                } else if (line.includes(`${org2}/${repo2}`)) {
                    summary.winner = 'project2';
                }
            }
        }
        
        // 分配健康评分和层级信息
        if (healthScores.length >= 2) {
            summary.healthScore1 = healthScores[0];
            summary.healthScore2 = healthScores[2];
        } else if (healthScores.length === 1) {
            // 如果只有一个健康评分，可能是报告格式不同，尝试其他方式
            const project1Match = report.match(new RegExp(`${org1}\/${repo1}[^\n]*([\d.]+)\/100`));
            const project2Match = report.match(new RegExp(`${org2}\/${repo2}[^\n]*([\d.]+)\/100`));
            
            if (project1Match) {
                summary.healthScore1 = parseFloat(project1Match[1]);
            }
            if (project2Match) {
                summary.healthScore2 = parseFloat(project2Match[1]);
            }
        }
        
        // 分配层级信息
        if (tiers.length >= 2) {
            summary.tier1 = tiers[0];
            summary.tier2 = tiers[1];
        } else {
            // 尝试其他方式提取层级
            const tier1Match = report.match(new RegExp(`${org1}\/${repo1}[^\n]*层级[:：]\s*(\w+)`));
            const tier2Match = report.match(new RegExp(`${org2}\/${repo2}[^\n]*层级[:：]\s*(\w+)`));
            
            if (tier1Match) {
                summary.tier1 = tier1Match[1];
            }
            if (tier2Match) {
                summary.tier2 = tier2Match[1];
            }
        }
        
        // 尝试提取活力状态
        const vitality1Match = report.match(new RegExp(`${org1}\/${repo1}[^\n]*当前状态[:：]\s*(\w+)`));
        const vitality2Match = report.match(new RegExp(`${org2}\/${repo2}[^\n]*当前状态[:：]\s*(\w+)`));
        
        if (vitality1Match) {
            summary.vitality1 = vitality1Match[1];
        }
        if (vitality2Match) {
            summary.vitality2 = vitality2Match[1];
        }
        
        // 如果还是没有找到，尝试更简单的匹配方式
        if (!summary.healthScore1 || !summary.healthScore2) {
            // 尝试匹配任何格式的健康评分
            const allScores = report.match(/[\d.]+\/100/g);
            if (allScores && allScores.length >= 2) {
                summary.healthScore1 = parseFloat(allScores[0]);
                summary.healthScore2 = parseFloat(allScores[1]);
            }
        }
        
        // 调试信息
        console.log('提取的健康评分:', healthScores);
        console.log('提取的层级:', tiers);
        console.log('最终对比摘要:', summary);
        
    } catch (error) {
        console.error('提取对比摘要失败:', error);
        console.error('报告内容:', report.substring(0, 500) + '...'); // 只显示前500字符
    }
    
    return summary;
}

// 辅助函数：从多项目对比报告中提取关键信息
function extractMultiComparisonSummary(report, projectNames) {
    if (!report) return null;
    
    const summary = {
        bestProject: null,
        averageHealthScore: null,
        projectCount: projectNames.length,
        comparison: []
    };
    
    try {
        const lines = report.split('\n');
        let healthScores = [];
        
        // 提取各项目的健康评分
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            
            // 提取健康评分
            const healthMatch = line.match(/([\d.]+)\/100/);
            if (healthMatch) {
                healthScores.push(parseFloat(healthMatch[1]));
            }
            
            // 提取最佳项目
            if (line.includes('最佳项目') || line.includes('最健康')) {
                for (const projectName of projectNames) {
                    const formattedName = projectName.replace('_', '/');
                    if (line.includes(formattedName)) {
                        summary.bestProject = formattedName;
                        break;
                    }
                }
            }
            
            // 提取对比结论
            if (line.includes('领先') || line.includes('优势') || line.includes('更健康')) {
                summary.comparison.push(line.trim());
            }
        }
        
        // 计算平均健康评分
        if (healthScores.length > 0) {
            const sum = healthScores.reduce((acc, score) => acc + score, 0);
            summary.averageHealthScore = sum / healthScores.length;
        }
        
    } catch (error) {
        console.error('提取多项目对比摘要失败:', error);
    }
    
    return summary;
}

// 错误处理中间件
app.use((err, req, res, next) => {
    console.error('服务器错误:', err);
    res.status(500).json({ error: '服务器内部错误' });
});

// 启动服务器
app.listen(PORT, () => {
    console.log(`服务器运行在 http://localhost:${PORT}`);
    console.log(`结果目录: ${CONFIG.resultsDir}`);
});