import os
from MR2_evaluate.detect_all_utils.coco import COCO
from MR2_evaluate.detect_all_utils.eval_MR_multisetup import COCOeval

#initialize COCO ground truth api
# annFile = '../CityPerson_val.json'
# initialize COCO detections api
# resFile = '../CityPerson_det.json'

def validate(annFile, dt_path):
    mean_MR = []
    my_id_setup = []
    for id_setup in range(0, 4):
        cocoGt = COCO(annFile)
        cocoDt = cocoGt.loadRes(dt_path)
        imgIds = sorted(cocoGt.getImgIds())
        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.params.imgIds = imgIds
        cocoEval.evaluate(id_setup)
        cocoEval.accumulate()
        mean_MR.append(cocoEval.summarize_nofile(id_setup))
        my_id_setup.append(id_setup)
    return mean_MR

# mean_MR = validate(annFile,resFile)
# print(mean_MR)