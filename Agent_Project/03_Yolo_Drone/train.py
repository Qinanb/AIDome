from ultralytics import YOLO
import torch

if __name__ == '__main__':
    # 强制清理显存碎片，腾出空间
    torch.cuda.empty_cache() 

    # 加载最轻量、最快的 Nano 模型，极其适合小目标和实时的视频检测
    model = YOLO('yolov8n.pt') 

    # 启动训练
    results = model.train(
        data='data.yaml',         # 刚刚写好的 yaml 文件路径
        epochs=100,               # 训练 100 轮是一个很好的起点
        imgsz=640,                # 图像输入尺寸
        cache='disk',             # ✅ 硬盘缓存！不上内存，利用固态硬盘极大加速读取
        batch=16,                 # 💡 8G显存的安全线
        workers=4,                # 💡 16G 内存的安全线，控制 CPU 搬运图片的速度
        device=0,                 # 使用独立显卡
        amp=True,                 # 开启半精度加速，省显存且提速
        project='Drone_Project',  # 训练结果保存的主文件夹
        name='yolov8n_run1'       # 本次实验的名字
    )