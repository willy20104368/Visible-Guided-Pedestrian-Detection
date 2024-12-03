#############################################################
# CityPersons

# CP anchor based
python evaluate_MR.py --weights cfg/Thesis_deploy/ckpt/SC_k3_L_ASFF_PAN_aux_nopg.pt --data data/CityPerson.yaml --img-size 2048 --task val --project  --name VGPD_Anchor_based_CP --device 1 --batch-size 8 --save-mr --conf 0.001 --iou 0.65 --deploy --exist-ok

# CP anchor free
python evaluate_MR.py --weights cfg/Thesis_deploy/ckpt/V2f_aux_rep_af_noM_CP.pt --data data/CityPerson.yaml --img-size 2048 --task val --project Thesis_result --name VGPD_AF_CP_noVGLA --device 1 --batch-size 8 --save-mr --conf 0.001 --iou 0.7 --deploy --anchor-free --no-trace --exist-ok
