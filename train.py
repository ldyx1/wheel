from ultralytics import YOLO
# 加载模型
# model = YOLO('yolov8-seg.yaml').load('yolov8s-seg.pt')  # 从YAML构建并转移权重
model = YOLO('yolov8s-seg.pt')
if __name__ == '__main__':
    # 训练模型
    results = model.train(data='wheel_seg.yaml', epochs=300, imgsz=640,bytes=16)
    # 验证模型
    #metrics = model.val()