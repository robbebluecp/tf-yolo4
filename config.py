import prepare

font_path = 'font_data/FiraMono-Medium.otf'
classes_path = r"D:\Datasets\asl-voc\classes.txt"
anchors_path = 'model_data/yolo4_anchors.txt'
classes_names = prepare.PrepareConfig().get_classes(classes_path)
anchors = prepare.PrepareConfig().get_anchors(anchors_path)
num_classes = len(classes_names)
num_anchors = len(anchors) // 3


image_input_shape = (608, 608)
# anchor使用顺序
anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
# 放缩比例
scale_size = [32, 16, 8]
# 数据增强参数
max_boxes = 20
jitter = 0.3
hue = 0.1
sat = 1.5
val = 1.5
# iou阈值
ignore_thresh = 0.5


validation_split = 0.1
batch_size = 8
epochs = 1000

score = 0.5
iou = 0.5

label_path = r"D:\Datasets\asl-voc\train.txt"
