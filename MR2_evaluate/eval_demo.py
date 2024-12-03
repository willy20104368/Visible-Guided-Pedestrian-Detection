from coco import COCO
from eval_MR_multisetup import COCOeval
import os

## MR2 evaluate function
def MR2_evaluate(annFile,resFile,output):
    
    annType = 'bbox'      #specify type here
    print('Running demo for *%s* results.'%(annType))
    os.makedirs('result',exist_ok=True)
    res_file = open(f"result/{output}.txt", "w")
    for id_setup in range(0,4):
        cocoGt = COCO(annFile)
        cocoDt = cocoGt.loadRes(resFile)
        imgIds = sorted(cocoGt.getImgIds())
        cocoEval = COCOeval(cocoGt,cocoDt,annType)
        cocoEval.params.imgIds  = imgIds
        cocoEval.evaluate(id_setup)
        cocoEval.accumulate()
        cocoEval.summarize(id_setup,res_file)
    res_file.close()



def MR2_evaluate_nofile(annFile,resFile):
    
    mean_MR = []
    annType = 'bbox'      #specify type here
    print('Running demo for *%s* results.'%(annType))

    for id_setup in range(0,4):
        cocoGt = COCO(annFile)
        cocoDt = cocoGt.loadRes(resFile)
        imgIds = sorted(cocoGt.getImgIds())
        cocoEval = COCOeval(cocoGt,cocoDt,annType)
        cocoEval.params.imgIds  = imgIds
        cocoEval.evaluate(id_setup)
        cocoEval.accumulate()
        mean_MR.append(cocoEval.summarize_nofile(id_setup))

    return mean_MR

if __name__ == '__main__':

    #initialize COCO ground truth api
    annFile = 'CityPerson_val.json'
    # initialize COCO detections api
    resFile = 'VLPD-Result-262.json'
    # output txt
    output = 'VLPD'
    ## running evaluation
    MR2_evaluate(annFile,resFile,output)