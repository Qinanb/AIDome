"""
一键启动脚本
"""
import os
import sys

# 确保工作目录正确
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 检查模型文件
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(project_dir, 'runs', 'detect', 'Drone_Project', 'yolov8n_run1', 'weights', 'best.pt')

if not os.path.exists(model_path):
    print(f"[!] 模型文件不存在: {model_path}")
    print("[!] 请确认已训练完成并生成了 best.pt")
    sys.exit(1)

print("[*] 模型文件已找到 OK")

# 启动 Flask
from app import app
print("\n" + "=" * 55)
print("  Drone Detection Platform")
print("  http://127.0.0.1:5000")
print("=" * 55 + "\n")
app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
