from ultralytics import YOLO
import cv2  # 用于手动控制图片显示窗口（避免一闪而过）

# 1. 加载训练好的分割模型（路径不变）
model = YOLO(r'D:\yolov8\ultralytics-main\runs\segment\train4\weights\best.pt')

# 2. 单张图片路径（替换为你的图片路径，支持jpg/png等格式）
image_path = r"D:\yolov8\ultralytics-main\Dataset_Wheel(seg)_Hole(seg)(58)\images\1 (54).jpg"  # 示例：桌面的测试图片
#image_path = r"D:\yolov8\ultralytics-main\Dataset_Wheel(seg)_Hole(seg)(84)\images\1(84).jpg"
# 3. 推理图片（关键：输入改为图片路径，其他参数保留）
# stream=True：启用流式推理，方便后续获取单帧结果（图片推理时返回1帧结果）
results = model(
    image_path,
    show=False,  # 先关闭自动显示（后续用cv2手动显示，便于控制窗口）
    conf=0.5,    # 置信度阈值：只显示置信度≥0.5的目标
    save=False   # 是否保存结果图片（需保存时设为True，结果存于runs/segment/predict）
)

# 4. 处理推理结果（获取带分割/检测框的图片）
# 结果列表中只有1个元素（单张图片），用results[0]获取
result = results[0]
# 将推理结果转换为OpenCV格式的图片（BGR通道，便于cv2显示）
result_image = result.plot()  # plot()会自动绘制边界框、分割掩码、类别标签

# 5. 手动显示图片（避免一闪而过）
cv2.namedWindow("YOLOv8 Segmentation Result", cv2.WINDOW_NORMAL)  # 创建可缩放窗口
cv2.imshow("YOLOv8 Segmentation Result", result_image)  # 显示结果图片
print("按下 'q' 键关闭窗口...")
# 等待按键：按下q键关闭窗口（避免程序直接退出）
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 6. 释放资源（必加，避免内存泄漏）
cv2.destroyAllWindows()