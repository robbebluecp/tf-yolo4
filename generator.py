"""
generator
"""

import numpy as np
from tools import utils_image, utils
import config


def preprocess_true_boxes(true_boxes: np.ndarray,
                          input_shape,
                          anchors,
                          num_classes):
    """

    :param true_boxes:      (N, max_box, 5)
    :param input_shape:     (416, 416)
    :param anchors:         (9, 2)
    :param num_classes:     ...
    :return:
    """
    assert (true_boxes[..., 4] < num_classes).all(), 'class id must be less than num_classes'

    num_layers = len(anchors) // 3  # default setting

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')

    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2

    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    # 剔除0项 Eliminate 0 items
    valid_mask = boxes_wh[..., 0] > 0
    # 放缩 Zoom

    true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

    batch_size = true_boxes.shape[0]

    grid_shapes = [input_shape // config.scale_size[l] for l in range(num_layers)]

    y_true = [np.zeros((batch_size, grid_shapes[l][0], grid_shapes[l][1], len(config.anchor_mask[l]), 5 + num_classes),
                       dtype='float32') for l in range(num_layers)]

    # 开始挑选最优anchor
    for N in range(batch_size):
        # Discard zero rows， 因为默认20个zeors_like, 一般都不够20的，有很多0填充项
        # (x, 2)
        wh = boxes_wh[N, valid_mask[N]]
        if len(boxes_wh) == 0:
            continue

        # 对于每个box，从9个anchor中，选出最佳anchor
        # (x, )
        best_anchor_indexes = utils.iou_area_index(wh, anchors)

        for t, n in enumerate(best_anchor_indexes):
            for l in range(num_layers):
                if n in config.anchor_mask[l]:
                    i = np.floor(true_boxes[N, t, 0] * grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[N, t, 1] * grid_shapes[l][0]).astype('int32')
                    k = config.anchor_mask[l].index(n)
                    c = true_boxes[N, t, 4].astype('int32')
                    y_true[l][N, j, i, k, 0:4] = true_boxes[N, t, 0:4]
                    y_true[l][N, j, i, k, 4] = 1
                    y_true[l][N, j, i, k, 5 + c] = 1
    return y_true


def data_generator(label_lines, batch_size, input_shape, anchors, num_classes):
    """

    :param label_lines:         record with file path and annotations,
                                for example:
                                    /Users/robbe/others/tf_data/voc2007/images/000017.jpg 185,62,279,199,14 90,78,403,336,12
                                    /Users/robbe/others/tf_data/voc2007/images/000012.jpg 156,97,351,270,6
                                /Users/robbe/others/tf_data/voc2007/images/000017.jpg   point to the absolute path name of an image
                                156,97,351,270,6    means that object's box(es) are xmin=156, ymin=97, xmax=351, ymax=270 which labeled with class-id 6
    :param batch_size:          batch size
    :param input_shape:         images' input shape, generally we use 608 or 416
    :param anchors:             all the anchors
    :param num_classes:         total count of classes, value of voc is 20
    :return:
    """

    n = len(label_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(label_lines)

            label_line = label_lines[i]
            info = label_line.split()
            image_file_path, cors = info[0], info[1:]
            cors = np.array([np.array(list(map(int, box.split(',')))) for box in cors], dtype=int)
            # Augment
            new_image, new_box = utils_image.Augment(img_path=image_file_path, boxes=cors)()
            new_box = np.concatenate([new_box, np.zeros(shape=(20 - len(new_box), 5))])

            image_data.append(new_image)
            box_data.append(new_box)

            i = (i + 1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)

if __name__ == '__main__':
   a = data_generator(['/Users/robbe/others/tf_data/voc2007/images/009819.jpg 369,34,465,188,8 232,51,383,267,7',
                       '/Users/robbe/others/tf_data/voc2007/images/009822.jpg 147,170,184,195,6 113,170,150,203,6 342,184,358,221,14 108,180,132,200,14 142,177,164,228,14 196,183,217,228,14 22,226,84,300,13 98,234,155,309,13 166,249,225,341,13 244,271,320,372,13 216,208,262,266,13 79,213,117,273,13 256,230,314,344,14 177,221,220,314,14 104,204,153,284,14 36,196,83,280,14 83,191,115,256,14 372,196,500,324,6 6,176,25,225,14 26,183,48,209,14 65,175,83,197,14 222,190,254,250,14',
                       '/Users/robbe/others/tf_data/voc2007/images/009823.jpg 3,4,498,374,11'], 1, input_shape=(608, 608), anchors=config.anchors, num_classes=config.num_classes)
   for i in a:
       print(i[0], '\n', i[1])
       exit()