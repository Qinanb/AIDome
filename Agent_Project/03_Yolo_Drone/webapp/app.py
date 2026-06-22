"""
Flask 无人机检测可视化平台
支持: 图片检测 | 视频检测 | 摄像头实时检测 | 批量检测 | 检测历史 | 统计面板
"""
import os
import sys
import cv2
import json
import time
import uuid
import shutil
import threading
from flask import (
    Flask, render_template, request, jsonify, Response,
    redirect, url_for, send_from_directory
)
from werkzeug.utils import secure_filename

# 将项目根目录加入路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detector import DroneDetector
import ai_analyzer

# ── 配置 ──────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(PROJECT_DIR, 'runs', 'detect', 'Drone_Project', 'yolov8n_run1', 'weights', 'best.pt')

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'static', 'uploads')
app.config['RESULT_FOLDER'] = os.path.join(BASE_DIR, 'static', 'results')

ALLOWED_IMAGE_EXT = {'png', 'jpg', 'jpeg', 'bmp', 'webp', 'tiff'}
ALLOWED_VIDEO_EXT = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# ── 加载模型 ──────────────────────────────────────────────────
print(f"[*] 正在加载模型: {MODEL_PATH}")
detector = DroneDetector(MODEL_PATH)
print(f"[*] 模型加载完成! 类别: {detector.class_names}")

# 全局摄像头状态
camera_state = {'running': False, 'thread': None, 'frame': None, 'detections': [], 'lock': threading.Lock()}

# 全局视频检测状态
video_state = {'running': False, 'thread': None, 'frame': None, 'detections': [],
               'progress': 0, 'total_frames': 0, 'current_frame': 0, 'total_detections': 0,
               'conf': 0.25, 'lock': threading.Lock()}


def allowed_file(filename, allowed_ext):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_ext


def gen_unique_name(filename):
    """生成唯一文件名"""
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else 'jpg'
    return f"{uuid.uuid4().hex[:8]}_{secure_filename(filename)}"


# ── 页面路由 ──────────────────────────────────────────────────
@app.route('/')
def index():
    """主面板"""
    return render_template('index.html')


@app.route('/image')
def image_page():
    """图片检测页"""
    return render_template('image.html')


@app.route('/video')
def video_page():
    """视频检测页"""
    return render_template('video.html')


@app.route('/camera')
def camera_page():
    """摄像头检测页"""
    return render_template('camera.html')


@app.route('/batch')
def batch_page():
    """批量检测页"""
    return render_template('batch.html')


@app.route('/history')
def history_page():
    """检测历史页"""
    return render_template('history.html')


@app.route('/settings')
def settings_page():
    """设置页"""
    return render_template('settings.html')


# ── API 路由 ──────────────────────────────────────────────────
@app.route('/api/detect/image', methods=['POST'])
def api_detect_image():
    """图片检测 API"""
    if 'file' not in request.files:
        return jsonify({'error': '未上传文件'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '文件名为空'}), 400

    if not allowed_file(file.filename, ALLOWED_IMAGE_EXT):
        return jsonify({'error': f'不支持的格式，仅支持: {", ".join(ALLOWED_IMAGE_EXT)}'}), 400

    conf = float(request.form.get('confidence', 0.25))

    # 保存上传文件
    filename = gen_unique_name(file.filename)
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(upload_path)

    try:
        result_path, detections, inference_time = detector.detect_image(
            upload_path, conf=conf, save=True, save_dir=app.config['RESULT_FOLDER']
        )
        # 转为相对路径
        rel_result = os.path.relpath(result_path, BASE_DIR).replace('\\', '/')
        return jsonify({
            'success': True,
            'result_image': '/' + rel_result,
            'original_image': '/static/uploads/' + filename,
            'detections': detections,
            'count': len(detections),
            'inference_time': inference_time,
            'drone_count': sum(1 for d in detections if d['class'] == 'drone'),
            'bird_count': sum(1 for d in detections if d['class'] == 'bird'),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/detect/batch', methods=['POST'])
def api_detect_batch():
    """批量图片检测 API"""
    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': '未上传文件'}), 400

    conf = float(request.form.get('confidence', 0.25))
    results = []

    for file in files:
        if file.filename == '' or not allowed_file(file.filename, ALLOWED_IMAGE_EXT):
            continue

        filename = gen_unique_name(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)

        try:
            result_path, detections, inference_time = detector.detect_image(
                upload_path, conf=conf, save=True, save_dir=app.config['RESULT_FOLDER']
            )
            rel_result = os.path.relpath(result_path, BASE_DIR).replace('\\', '/')
            results.append({
                'filename': file.filename,
                'result_image': '/' + rel_result,
                'original_image': '/static/uploads/' + filename,
                'detections': detections,
                'count': len(detections),
                'inference_time': inference_time,
                'drone_count': sum(1 for d in detections if d['class'] == 'drone'),
                'bird_count': sum(1 for d in detections if d['class'] == 'bird'),
            })
        except Exception as e:
            results.append({
                'filename': file.filename,
                'error': str(e),
            })

    return jsonify({
        'success': True,
        'results': results,
        'total': len(results),
    })


@app.route('/api/detect/video', methods=['POST'])
def api_detect_video():
    """视频检测 API — 上传视频并启动后台逐帧处理"""
    if video_state['running']:
        return jsonify({'error': '有视频正在处理中，请等待完成或先停止'}), 400

    if 'file' not in request.files:
        return jsonify({'error': '未上传文件'}), 400

    file = request.files['file']
    if not allowed_file(file.filename, ALLOWED_VIDEO_EXT):
        return jsonify({'error': f'不支持的格式，仅支持: {", ".join(ALLOWED_VIDEO_EXT)}'}), 400

    conf = float(request.form.get('confidence', 0.25))
    filename = gen_unique_name(file.filename)
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(upload_path)

    # 获取视频信息
    cap = cv2.VideoCapture(upload_path)
    if not cap.isOpened():
        return jsonify({'error': '无法打开视频文件'}), 400
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    cap.release()

    # 重置状态
    video_state.update({
        'running': True, 'frame': None, 'detections': [], 'progress': 0,
        'total_frames': total_frames, 'current_frame': 0, 'total_detections': 0,
        'conf': conf,
    })

    def process_video():
        cap = cv2.VideoCapture(upload_path)
        while video_state['running']:
            ret, frame = cap.read()
            if not ret:
                break
            annotated, detections, _ = detector.detect_video_frame(frame, conf=video_state['conf'])
            with video_state['lock']:
                video_state['frame'] = detector.get_frame_jpeg(annotated)
                video_state['detections'] = detections
                video_state['current_frame'] += 1
                video_state['total_detections'] += len(detections)
                video_state['progress'] = video_state['current_frame'] / max(total_frames, 1) * 100
            # 控制播放速度，匹配原始帧率
            time.sleep(1.0 / fps)
        cap.release()
        video_state['running'] = False

    t = threading.Thread(target=process_video, daemon=True)
    t.start()
    video_state['thread'] = t

    return jsonify({
        'success': True,
        'message': '视频处理已启动',
        'total_frames': total_frames,
        'fps': round(fps, 1),
    })


@app.route('/api/video/stream')
def api_video_stream():
    """视频检测结果流 (MJPEG)"""
    def generate():
        while video_state['running'] or video_state['frame']:
            with video_state['lock']:
                frame = video_state['frame']
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                time.sleep(0.05)
            # 检测完成后继续发送最后一帧一小段时间再停止
            if not video_state['running'] and video_state['frame']:
                time.sleep(0.5)
                break
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/video/progress')
def api_video_progress():
    """获取视频处理进度"""
    with video_state['lock']:
        return jsonify({
            'running': video_state['running'],
            'progress': round(video_state['progress'], 1),
            'current_frame': video_state['current_frame'],
            'total_frames': video_state['total_frames'],
            'total_detections': video_state['total_detections'],
            'detections': video_state['detections'],
        })


@app.route('/api/video/stop', methods=['POST'])
def api_video_stop():
    """停止视频处理"""
    video_state['running'] = False
    return jsonify({'message': '视频处理已停止'})


@app.route('/api/camera/start', methods=['POST'])
def api_camera_start():
    """启动摄像头检测"""
    if camera_state['running']:
        return jsonify({'message': '摄像头已在运行'})

    camera_state['running'] = True
    conf = float(request.json.get('confidence', 0.25)) if request.json else 0.25

    def camera_loop():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            camera_state['running'] = False
            return

        while camera_state['running']:
            ret, frame = cap.read()
            if not ret:
                break
            annotated, detections, _ = detector.detect_video_frame(frame, conf=conf)
            with camera_state['lock']:
                camera_state['frame'] = detector.get_frame_jpeg(annotated)
                camera_state['detections'] = detections

        cap.release()
        with camera_state['lock']:
            camera_state['frame'] = None

    t = threading.Thread(target=camera_loop, daemon=True)
    t.start()
    camera_state['thread'] = t

    return jsonify({'message': '摄像头已启动'})


@app.route('/api/camera/stop', methods=['POST'])
def api_camera_stop():
    """停止摄像头检测"""
    camera_state['running'] = False
    return jsonify({'message': '摄像头已停止'})


@app.route('/api/camera/stream')
def api_camera_stream():
    """摄像头视频流 (MJPEG)"""
    def generate():
        while camera_state['running']:
            with camera_state['lock']:
                frame = camera_state['frame']
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                time.sleep(0.05)

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/camera/detections')
def api_camera_detections():
    """获取当前摄像头检测结果"""
    with camera_state['lock']:
        return jsonify({
            'running': camera_state['running'],
            'detections': camera_state['detections'],
        })


@app.route('/api/stats')
def api_stats():
    """获取检测统计"""
    return jsonify(detector.get_stats())


@app.route('/api/history')
def api_history():
    """获取检测历史"""
    limit = int(request.args.get('limit', 50))
    return jsonify(detector.get_history(limit))


@app.route('/api/history/clear', methods=['POST'])
def api_clear_history():
    """清空检测历史"""
    detector.clear_history()
    return jsonify({'message': '历史已清空'})


@app.route('/api/cleanup', methods=['POST'])
def api_cleanup():
    """清理临时文件"""
    for folder in [app.config['UPLOAD_FOLDER'], app.config['RESULT_FOLDER']]:
        for f in os.listdir(folder):
            fp = os.path.join(folder, f)
            try:
                if os.path.isfile(fp):
                    os.remove(fp)
            except:
                pass
    return jsonify({'message': '临时文件已清理'})


# ── AI 分析路由 ────────────────────────────────────────────
@app.route('/api/ai/config', methods=['GET'])
def api_ai_config():
    """获取 AI 配置（脱敏）"""
    cfg = ai_analyzer.load_config()
    key = cfg.get('api_key', '')
    masked_key = key[:8] + '***' + key[-4:] if len(key) > 12 else ('***' if key else '')
    return jsonify({
        'api_base': cfg.get('api_base', ''),
        'api_key': masked_key,
        'api_key_set': bool(key),
        'model': cfg.get('model', ''),
        'configured': ai_analyzer.is_configured(),
    })


@app.route('/api/ai/config', methods=['POST'])
def api_ai_config_save():
    """保存 AI 配置"""
    data = request.json or {}
    cfg = ai_analyzer.load_config()
    if 'api_base' in data:
        cfg['api_base'] = data['api_base'].strip().rstrip('/')
    if 'api_key' in data and data['api_key'] and '***' not in data['api_key']:
        cfg['api_key'] = data['api_key'].strip()
    if 'model' in data:
        cfg['model'] = data['model'].strip()
    ai_analyzer.save_config(cfg)
    return jsonify({'message': '配置已保存', 'configured': ai_analyzer.is_configured()})


@app.route('/api/ai/analyze', methods=['POST'])
def api_ai_analyze():
    """AI 分析检测结果"""
    if not ai_analyzer.is_configured():
        return jsonify({'error': 'AI 未配置，请在设置页面填写 API 信息'}), 400

    data = request.json or {}
    detections = data.get('detections', [])
    conf_threshold = data.get('confidence', 0.25)
    image_base64 = data.get('image_base64', None)

    result, error = ai_analyzer.analyze_detection(
        detections=detections,
        image_base64=image_base64,
        conf_threshold=conf_threshold
    )

    if error:
        return jsonify({'error': error}), 400

    return jsonify({'success': True, 'analysis': result})


# ── 启动 ──────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "=" * 55)
    print("  Drone Detection Platform")
    print("  http://127.0.0.1:5000")
    print("=" * 55 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
