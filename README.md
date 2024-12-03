# Visible Guided Pedestrian Detection

## Training anchor-based
```
python train_aux_MR.py --workers 8 --device 2 --batch-size 2 --data data/CityPerson_fg.yaml --img 2048 2048 --cfg cfg/training/VGPD_ab.yaml --weights '' --name VGPD_ab --epochs 300 --hyp data/hyp.scratch.rep.yaml --adam
```
## Training anchor-based

'''
python train_anchor_free.py --workers 8 --device 2 --batch-size 2 --data data/CityPerson_fg.yaml --img 2048 2048 --cfg anchor_free/VGPD_aux_af.yaml --weights '' --name VGPD_aux_af --epochs 500 --hyp data/hyp.scratch.rep.yaml --adam --project runs/anchor_free
'''
