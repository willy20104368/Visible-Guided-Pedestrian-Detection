# Visible Guided Pedestrian Detection
![image](https://github.com/willy20104368/Visible-Guided-Pedestrian-Detection/blob/main/overall.png)
The main ideas are as follows:
-RepC4ELAN-CA: Enhancing representation capability without introducing any time increment.
-Visible Guided Auxiliary head: Learns information from the visible region and is removed during inference.
-Visible-guided label assignment: Constrains the number and positions of extremely hard positive samples.
## Training anchor-based
```
python train_aux_MR.py --workers 8 --device 2 --batch-size 2 --data data/CityPerson_fg.yaml --img 2048 2048 --cfg cfg/training/VGPD_ab.yaml --weights '' --name VGPD_ab --epochs 300 --hyp data/hyp.scratch.rep.yaml --adam

```
## Training anchor-free
```
python train_anchor_free.py --workers 8 --device 2 --batch-size 2 --data data/CityPerson_fg.yaml --img 2048 2048 --cfg anchor_free/VGPD_aux_af.yaml --weights '' --name VGPD_aux_af --epochs 500 --hyp data/hyp.scratch.rep.yaml --adam --project runs/anchor_free
```
## Test anchor-based
```
python evaluate_MR.py --weights path_to_weight --data data/CityPerson.yaml --img-size 2048 --task val --project  --name VGPD_Anchor_based_CP --device 1 --batch-size 8 --save-mr --conf 0.001 --iou 0.65 --deploy
```
## Test anchor-free
```
python evaluate_MR.py --weights path_to_weight --data data/CityPerson.yaml --img-size 2048 --task val --project Thesis_result --name VGPD_AF_CP --device 1 --batch-size 8 --save-mr --conf 0.001 --iou 0.7 --deploy --anchor-free --no-trace
```

## Reference
This repository is based on [YOLOv7](https://github.com/WongKinYiu/yolov7?tab=readme-ov-file#training).
