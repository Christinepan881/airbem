import json
import os
import cv2
from detectron2.data import DatasetCatalog, MetadataCatalog

def create_airbem_semantic_dicts(img_dir, mask_dir):
    images = os.listdir(img_dir)
    dataset_dicts = []
    for img in images:
        record = {}
        filename = os.path.join(img_dir, img)
        height, width = cv2.imread(filename).shape[:2]
        record["file_name"] = filename
        record["image_id"] = img.split('.')[0]
        record["height"] = height
        record["width"] = width
        record["sem_seg_file_name"] = os.path.join(mask_dir, record["image_id"]+'.png')
        record["pan_seg_file_name"] = os.path.join(mask_dir, record["image_id"]+'.png')
        dataset_dicts.append(record)
    return dataset_dicts

def generate_airbem_semantic_dicts():
    img_dir = "/home/cpan14/datasets/AirBEM/ir_dir"
    mask_dir = "/home/cpan14/datasets/AirBEM/mask_dir_HW_012"
    for mode in ['train', 'val']:
        cur_img_dir = os.path.join(img_dir, mode)
        cur_mask_dir = os.path.join(mask_dir, mode)
        data_dicts = create_airbem_semantic_dicts(cur_img_dir, cur_mask_dir)
        json_path = "/home/cpan14/datasets/AirBEM/airbem_semantic_{}_dicts_012.json".format(mode)
        with open(json_path, 'w') as f:
            json.dump(data_dicts, f)

def get_airbem_semantic_dicts(json_path):
    with open(json_path, 'r') as f:
        data_dicts = json.load(f)
    return data_dicts

def register_airbem_semantic(data_base_dir="/home/cpan14/datasets/AirBEM", mode='train', metadata=None):
    # json_path = os.path.join(data_base_dir, "airbem_semantic_{}_dicts.json".format(mode))
    json_path = os.path.join(data_base_dir, "airbem_semantic_{}_dicts_012.json".format(mode))
    DatasetCatalog.register("airbem_" + mode, lambda: get_airbem_semantic_dicts(json_path))

    image_root = os.path.join(data_base_dir, "ir_dir", mode)
    # mask_root = os.path.join(data_base_dir, "mask_dir_HW", mode)
    mask_root = os.path.join(data_base_dir, "mask_dir_HW_012", mode)
    MetadataCatalog.get("airbem_" + mode).set(# thing_classes=["in/ex-filtration", "thermal bridge"], \
                                              image_root=image_root,
                                              mask_root=mask_root,
                                              json_file=json_path,
                                              evaluator_type="sem_seg",
                                              ignore_label=255, **metadata)
    
    # MetadataCatalog.get("airbem_" + mode).set(stuff_colors=AIRBEM_2_COLORS[:], **metadata)

def get_airbem_metadata():
    meta = {}
    # The following metadata maps contiguous id from [0, #thing categories +
    # #stuff categories) to their names and colors. We have to replica of the
    # same name and color under "thing_*" and "stuff_*" because the current
    # visualization function in D2 handles thing and class classes differently
    # due to some heuristic used in Panoptic FPN. We keep the same naming to
    # enable reusing existing visualization functions.
    thing_classes = [k["name"] for k in AIRBEM_2_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in AIRBEM_2_CATEGORIES if k["isthing"] == 1]
    stuff_classes = [k["name"] for k in AIRBEM_2_CATEGORIES]
    stuff_colors = [k["color"] for k in AIRBEM_2_CATEGORIES]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    # Convert category id for training:
    #   category id: like semantic segmentation, it is the class id for each
    #   pixel. Since there are some classes not used in evaluation, the category
    #   id is not always contiguous and thus we have two set of category ids:
    #       - original category id: category id in the original dataset, mainly
    #           used for evaluation.
    #       - contiguous category id: [0, #classes), in order to train the linear
    #           softmax classifier.
    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    for i, cat in enumerate(AIRBEM_2_CATEGORIES):
        if cat["isthing"]:
            thing_dataset_id_to_contiguous_id[cat["id"]] = i
        # else:
        #     stuff_dataset_id_to_contiguous_id[cat["id"]] = i

        # in order to use sem_seg evaluator
        stuff_dataset_id_to_contiguous_id[cat["id"]] = i

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    return meta

# AIRBEM_2_CATEGORIES = [
#     {"color": [128, 0, 0], "id": 0, "isthing": 1, "name": "thermal-bridge"},
#     {"color": [0, 128, 0], "id": 1, "isthing": 1, "name": "in/ex-filtration"},]

# AIRBEM_2_CATEGORIES = [
#     {"color": [128, 0, 0], "id": 1, "isthing": 1, "name": "thermal-bridge"},
#     {"color": [0, 128, 0], "id": 2, "isthing": 1, "name": "in/ex-filtration"},]

AIRBEM_2_CATEGORIES = [ # BGR
    {"color": [255, 255, 255], "id": 0, "isthing": 0, "name": "background"},
    {"color": [255, 0, 0], "id": 1, "isthing": 1, "name": "thermal-bridge"},
    {"color": [0, 255, 0], "id": 2, "isthing": 1, "name": "in/ex-filtration"},]


AIRBEM_2_COLORS = [k["color"] for k in AIRBEM_2_CATEGORIES]

AIRBEM_MODES = ['train', 'val']
AIRBEM_DIR = "/home/cpan14/datasets/AirBEM"
for m in AIRBEM_MODES:
    register_airbem_semantic(data_base_dir=AIRBEM_DIR, mode=m, metadata=get_airbem_metadata())
