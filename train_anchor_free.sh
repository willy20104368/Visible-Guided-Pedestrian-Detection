python train_anchor_free.py --workers 8 --device 2 --batch-size 2 --data data/CityPerson.yaml --img 2048 2048 --cfg cfg/training/VGPD_aux_af.yaml --weights '' --name VGPD_aux_af --epochs 500 --hyp data/hyp.scratch.rep.yaml --adam