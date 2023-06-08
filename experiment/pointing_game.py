class objectbox:
    def __init__(self):
        self.boxes = {}

    def insert(self, name, x):
        if name not in self.boxes:
            self.boxes[name] = []
        self.boxes[name].append([x[0], int(x[1]), int(x[2]), int(x[3]), int(x[4])])


def get_box_for_val(path=r'D:\ImageNet_TEST\LOC_val_solution.csv'):
    import csv
    obj = objectbox()
    with open(path, 'r') as f:
        reader = csv.reader(f)
        result = list(reader)
    for name, box in result[1:]:
        boxes = box.split(' ')[:-1]
        object_num = (len(boxes) + 1) // 5
        for i in range(object_num):
            key = i * 5
            obj.insert(name, boxes[key:key + 5])
    return obj


def get_image_for_val(root=r'D:\ImageNet_TEST\val', name=None, class_id=None):
    from PIL import Image
    assert name is not None
    if class_id is not None:
        path = os.path.join(root, class_id)
        img = Image.open(os.path.join(path, name + '.JPEG')).convert('RGB')
        img = np.array(img)
    else:
        for class_id in os.listdir(root):
            path = os.path.join(root, class_id)
            if os.path.exists(os.path.join(path, name + '.JPEG')):
                img = Image.open(os.path.join(path, name + '.JPEG'))
                img = np.array(img)
                return img
    return img


def check_point(point, boxes, ori_H, ori_W, cls):
    ret = 0
    for box in boxes:
        if box[0] == cls:
            box = box[1:]
            t = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
            t = [t[0] * 224 // ori_W, t[1] * 224 // ori_H,
                 t[2] * 224 // ori_W, t[3] * 224 // ori_H]
            if point[0] >= t[1] and point[0] <= t[3] and point[1] >= t[0] and point[1] <= t[2]:
                ret = 1
                break
    return ret


if __name__ == '__main__':
    from tqdm import tqdm
    import os
    from glob import glob
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    data_path = r'/data/shiwei/ICLR2023/LOC_val_solution.csv'
    heatmap_root = r"/data/shiwei/ICLR2023/AUG/AUG_VGG19BN_TWOSTAGE/"

    objs = get_box_for_val(data_path)
    sub_class = os.listdir(heatmap_root)
    verbose = tqdm(sub_class)

    hit = 0
    total = 0

    for cls in verbose:
        path = os.path.join(heatmap_root, cls)
        for file in glob(os.path.join(path, '*.npy')):
            heatmap = np.load(file)
            # heatmap = np.random.rand(224, 224)

            H, W = np.shape(heatmap)
            htmap = np.reshape(heatmap, -1)
            point = np.argmax(htmap)
            point = (point // W, point % W)
            # point = (128, 128)
            name = os.path.basename(file).split('.')[0]
            obj = objs.boxes[name]
            img = get_image_for_val(root='/data/shiwei/ilsvrc2012/val/', name=name,
                                    class_id=objs.boxes[name][0][0])
            ori_H, ori_W = np.shape(img)[0], np.shape(img)[1]

            hit += check_point(point, obj, ori_H, ori_W, cls)
            total += 1

            verbose.set_description('{:.7f}'.format(hit / total))

    print(hit / total)

