/**
 * 无人机检测平台 - 主 JS
 */

// ── 侧边栏折叠 ──────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    const sidebar = document.getElementById('sidebar');
    const toggle = document.getElementById('sidebarToggle');
    if (toggle) {
        toggle.addEventListener('click', () => {
            sidebar.classList.toggle('collapsed');
            const icon = toggle.querySelector('i');
            icon.className = sidebar.classList.contains('collapsed')
                ? 'ri-menu-unfold-line' : 'ri-menu-fold-line';
        });
    }
});

// ── 工具函数 ──────────────────────────────────────────────
function showToast(message, type = 'info', duration = 3000) {
    let toast = document.querySelector('.toast');
    if (!toast) {
        toast = document.createElement('div');
        toast.className = 'toast';
        document.body.appendChild(toast);
    }
    const icons = { success: 'ri-check-line', error: 'ri-error-warning-line', info: 'ri-information-line' };
    toast.className = `toast ${type}`;
    toast.innerHTML = `<i class="${icons[type] || icons.info}"></i>${message}`;
    setTimeout(() => toast.classList.add('show'), 10);
    setTimeout(() => toast.classList.remove('show'), duration);
}

function showLoading(text = '检测中...') {
    let overlay = document.getElementById('loadingOverlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.id = 'loadingOverlay';
        overlay.className = 'loading-overlay';
        overlay.innerHTML = `<div class="spinner"></div><p id="loadingText">${text}</p>`;
        document.body.appendChild(overlay);
    } else {
        document.getElementById('loadingText').textContent = text;
    }
    overlay.classList.add('show');
}

function hideLoading() {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) overlay.classList.remove('show');
}

// ── 拖拽上传通用处理 ──────────────────────────────────────
function initDragDrop(zone, onFile) {
    if (!zone) return;
    ['dragenter', 'dragover'].forEach(e => {
        zone.addEventListener(e, ev => {
            ev.preventDefault();
            zone.classList.add('dragover');
        });
    });
    ['dragleave', 'drop'].forEach(e => {
        zone.addEventListener(e, ev => {
            ev.preventDefault();
            zone.classList.remove('dragover');
        });
    });
    zone.addEventListener('drop', ev => {
        const files = ev.dataTransfer.files;
        if (files.length) onFile(files);
    });
}

// ── 文件类型判断（兼容拖拽时 file.type 为空的情况）─────────
const IMAGE_EXTS = ['png','jpg','jpeg','bmp','webp','tiff','gif'];
const VIDEO_EXTS = ['mp4','avi','mov','mkv','flv','wmv'];

function getFileExt(file) {
    return (file.name || '').split('.').pop().toLowerCase();
}

function isImageFile(file) {
    if (file.type && file.type.startsWith('image/')) return true;
    return IMAGE_EXTS.includes(getFileExt(file));
}

function isVideoFile(file) {
    if (file.type && file.type.startsWith('video/')) return true;
    return VIDEO_EXTS.includes(getFileExt(file));
}

// ── API 请求封装 ──────────────────────────────────────────
async function apiDetectImage(file, confidence = 0.25) {
    const form = new FormData();
    form.append('file', file);
    form.append('confidence', confidence);
    const res = await fetch('/api/detect/image', { method: 'POST', body: form });
    return res.json();
}

async function apiDetectBatch(files, confidence = 0.25) {
    const form = new FormData();
    for (const f of files) form.append('files', f);
    form.append('confidence', confidence);
    const res = await fetch('/api/detect/batch', { method: 'POST', body: form });
    return res.json();
}

async function apiDetectVideo(file, confidence = 0.25) {
    const form = new FormData();
    form.append('file', file);
    form.append('confidence', confidence);
    const res = await fetch('/api/detect/video', { method: 'POST', body: form });
    return res.json();
}

async function apiGetStats() {
    const res = await fetch('/api/stats');
    return res.json();
}

async function apiGetHistory(limit = 50) {
    const res = await fetch(`/api/history?limit=${limit}`);
    return res.json();
}

async function apiClearHistory() {
    const res = await fetch('/api/history/clear', { method: 'POST' });
    return res.json();
}

async function apiCleanup() {
    const res = await fetch('/api/cleanup', { method: 'POST' });
    return res.json();
}

async function apiAiAnalyze(detections, confidence, imageBase64 = null) {
    const res = await fetch('/api/ai/analyze', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            detections: detections,
            confidence: confidence,
            image_base64: imageBase64,
        })
    });
    return res.json();
}

// 将图片转为 base64
function imageToBase64(imgElement) {
    return new Promise((resolve) => {
        const canvas = document.createElement('canvas');
        canvas.width = imgElement.naturalWidth;
        canvas.height = imgElement.naturalHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(imgElement, 0, 0);
        // 压缩到合理大小
        const dataUrl = canvas.toDataURL('image/jpeg', 0.7);
        resolve(dataUrl.split(',')[1]);
    });
}

// 渲染 AI 分析结果（Markdown 简易渲染）
function renderMarkdown(text) {
    if (!text) return '';
    return text
        .replace(/^### (.+)$/gm, '<h4 style="margin:12px 0 6px; color:var(--accent);">$1</h4>')
        .replace(/^## (.+)$/gm, '<h3 style="margin:14px 0 8px; color:var(--text-primary);">$1</h3>')
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.+?)\*/g, '<em>$1</em>')
        .replace(/`(.+?)`/g, '<code style="background:var(--bg-secondary);padding:2px 6px;border-radius:4px;font-size:12px;">$1</code>')
        .replace(/^- (.+)$/gm, '<li style="margin:4px 0; margin-left:16px;">$1</li>')
        .replace(/^\d+\. (.+)$/gm, '<li style="margin:4px 0; margin-left:16px;">$1</li>')
        .replace(/\n/g, '<br>');
}

// ── 渲染检测结果 ──────────────────────────────────────────
function renderDetections(container, detections) {
    if (!container) return;
    if (!detections || detections.length === 0) {
        container.innerHTML = `
            <div class="empty-state" style="padding:30px">
                <i class="ri-checkbox-circle-line"></i>
                <h3>未检测到目标</h3>
                <p>尝试降低置信度阈值</p>
            </div>`;
        return;
    }
    container.innerHTML = detections.map(d => `
        <div class="detection-item ${d.class}">
            <div class="dot"></div>
            <div class="info">
                <div class="name">${d.class === 'drone' ? '🚁 无人机' : '🐦 鸟类'}</div>
                <div class="meta">位置: [${d.bbox.join(', ')}]${d.area ? ` | 面积: ${d.area}px²` : ''}</div>
            </div>
            <div class="conf">${d.confidence}%</div>
        </div>
    `).join('');
}

function renderDetectionSummary(droneCount, birdCount, inferenceTime) {
    return `
        <div class="stats-grid" style="margin-top:16px">
            <div class="stat-card">
                <div class="stat-icon green"><i class="ri-drone-line"></i></div>
                <div class="stat-info"><h3>${droneCount}</h3><p>无人机</p></div>
            </div>
            <div class="stat-card">
                <div class="stat-icon orange"><i class="ri-bird-line"></i></div>
                <div class="stat-info"><h3>${birdCount}</h3><p>鸟类</p></div>
            </div>
            <div class="stat-card">
                <div class="stat-icon blue"><i class="ri-timer-line"></i></div>
                <div class="stat-info"><h3>${inferenceTime}ms</h3><p>推理耗时</p></div>
            </div>
        </div>`;
}
