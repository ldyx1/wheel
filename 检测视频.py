from ultralytics import YOLO

# 加载模型
model = YOLO(r'D:\yolov8\ultralytics-main\runs\segment\train4\weights\best.pt')

# 处理视频并获取结果
results = model(
    #r"C:\Users\zqk\Desktop\OpenCV\索道OpenCV\索道相关文件\汇报代码\第二次\较合理视频\压索-近平-60S-S.MP4",
    r"E:\高速下载__06-21索道视频\压索-近仰-60S-S.MP4",
    show=True,  # 显示结果
    conf=0.5,   # 置信度阈值
    save=False  # 是否保存结果视频，设为True可保存
)

# 可以遍历结果进行后续处理
for result in results:
    # 处理每帧的结果
    boxes = result.boxes  # 边界框信息
    masks = result.masks  # 分割掩码信息
    # 可以在这里添加自定义处理逻辑