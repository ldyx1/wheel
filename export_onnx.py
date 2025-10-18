from ultralytics import YOLO
# 1. 加载你的 YOLOv8 模型
# 请确保路径 'best.pt' 正确，如果文件就在此脚本旁边，则无需修改
model = YOLO(r'C:\Users\zqk\Desktop\zdbz\best.pt')

# 2. 导出模型
# format='onnx' 指定导出为 ONNX 格式
# 导出的文件 'best.onnx' 会自动保存在与 'best.pt' 相同的目录下
model.export(format='onnx')
print("模型已成功导出为 best.onnx！")