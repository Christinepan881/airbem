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

        dataset_dicts.append(record)
    return dataset_dicts

def get_airbem_semantic_dicts(json_path):
    with open(json_path, 'r') as f:
        data_dicts = json.load(f)
    return data_dicts

def register_airbem_semantic(data_base_dir="/home/cpan14/datasets/AirBEM", mode='train', metadata=None):
    json_path = os.path.join(data_base_dir, "airbem_semantic_{}_dicts.json".format(mode))
    DatasetCatalog.register("airbem_" + mode, lambda: get_airbem_semantic_dicts(json_path))

    image_root = os.path.join(data_base_dir, "ir_dir", mode)
    mask_root = os.path.join(data_base_dir, "mask_dir_HW", mode)
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


AIRBEM_2_CATEGORIES = [
    {"color": [128, 0, 0], "id": 0, "isthing": 1, "name": "thermal-bridge"},
    {"color": [0, 128, 0], "id": 1, "isthing": 1, "name": "in/ex-filtration"},]

AIRBEM_2_COLORS = [k["color"] for k in AIRBEM_2_CATEGORIES]

AIRBEM_MODES = ['train', 'val']
AIRBEM_DIR = "/home/cpan14/datasets/AirBEM"
for m in AIRBEM_MODES:
    register_airbem_semantic(data_base_dir=AIRBEM_DIR, mode=m, metadata=get_airbem_metadata())

# train_dicts = get_airbem_semantic_dicts("/home/cpan14/datasets/AirBEM/ir_dir/train", "/home/cpan14/datasets/AirBEM/mask_dir_HW/train")
# with open("/home/cpan14/datasets/AirBEM/airbem_semantic_train_dicts.json", 'w') as f:
#     json.dump(train_dicts, f)

# val_dicts = get_airbem_semantic_dicts("/home/cpan14/datasets/AirBEM/ir_dir/val", "/home/cpan14/datasets/AirBEM/mask_dir_HW/val")
# with open("/home/cpan14/datasets/AirBEM/airbem_semantic_val_dicts.json", 'w') as f:
#     json.dump(val_dicts, f)

# for d in ["train", "val"]:
#     DatasetCatalog.register("balloon_" + d, lambda d=d: get_balloon_dicts("balloon/" + d))
#     MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"])
# balloon_metadata = MetadataCatalog.get("balloon_train")

# with open("/home/cpan14/datasets/AirBEM/airbem_semantic_train_dicts.json", 'r') as f:
#     train_dicts = json.load(f)

# condor_ssh_to_job 5335656.0
# singularity run --nv videomae-sand 
# source .venv-oneformer/bin/activate
# cd OneFormer
# python train_net.py --dist-url 'tcp://127.0.0.1:50163' \
#     --num-gpus 1 \
#     --config-file configs/airbem/oneformer_swin_tiny_bs16_50ep.yaml \
#     OUTPUT_DIR outputs/test_oneformer_airbem WANDB.NAME test_oneformer_airbem

# [04/13 23:50:15 d2.utils.events]:  eta: 3 days, 0:25:58  iter: 279  total_loss: 59.2  loss_ce: 0.5937  loss_mask: 1.41  loss_dice: 3.508  loss_contrastive: 0.7078  loss_ce_0: 0.4006  loss_mask_0: 1.35  loss_dice_0: 3.948  loss_ce_1: 0.5565  loss_mask_1: 1.552  loss_dice_1: 3.693  loss_ce_2: 0.5595  loss_mask_2: 1.571  loss_dice_2: 4.059  loss_ce_3: 0.5599  loss_mask_3: 1.407  loss_dice_3: 3.815  loss_ce_4: 0.5468  loss_mask_4: 1.5  loss_dice_4: 3.948  loss_ce_5: 0.5295  loss_mask_5: 1.38  loss_dice_5: 3.699  loss_ce_6: 0.5377  loss_mask_6: 1.498  loss_dice_6: 3.732  loss_ce_7: 0.5352  loss_mask_7: 1.351  loss_dice_7: 3.964  loss_ce_8: 0.5521  loss_mask_8: 1.501  loss_dice_8: 3.707    time: 0.7098  last_time: 0.7054  data_time: 0.0058  last_data_time: 0.0064   lr: 0.0001  max_mem: 13632M
# [04/13 23:51:12 d2.utils.events]:  eta: 3 days, 0:24:23  iter: 359  total_loss: 59.9  loss_ce: 0.5766  loss_mask: 1.388  loss_dice: 3.364  loss_contrastive: 0.705  loss_ce_0: 0.3563  loss_mask_0: 1.046  loss_dice_0: 4.092  loss_ce_1: 0.491  loss_mask_1: 1.221  loss_dice_1: 4.004  loss_ce_2: 0.5595  loss_mask_2: 1.442  loss_dice_2: 3.399  loss_ce_3: 0.5705  loss_mask_3: 1.526  loss_dice_3: 3.291  loss_ce_4: 0.5577  loss_mask_4: 1.898  loss_dice_4: 3.311  loss_ce_5: 0.4988  loss_mask_5: 1.361  loss_dice_5: 3.615  loss_ce_6: 0.5216  loss_mask_6: 1.095  loss_dice_6: 3.352  loss_ce_7: 0.5104  loss_mask_7: 1.588  loss_dice_7: 3.604  loss_ce_8: 0.5364  loss_mask_8: 1.471  loss_dice_8: 3.376    time: 0.7097  last_time: 0.7070  data_time: 0.0056  last_data_time: 0.0051   lr: 0.0001  max_mem: 13632M