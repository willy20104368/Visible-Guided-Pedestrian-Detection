import utils.autoanchor as autoAnchor
 
# cp
# new_anchors = autoAnchor.kmean_anchors(path=r"/home/willy20104368/yolov7/data/CityPerson.yaml", n=9, img_size=2048, thr=4, gen=1000, verbose=False)
# print(new_anchors)

# tju 
# new_anchors = autoAnchor.kmean_anchors(path=r"/home/willy20104368/yolov7/data/TJU_Ped_Traffic.yaml", n=9, img_size=1632, thr=4, gen=3000, verbose=False)
# print(new_anchors)

# caltech
# new_anchors = autoAnchor.kmean_anchors(path=r"/home/willy20104368/yolov7/data/Caltech0501.yaml", n=9, img_size=640, thr=4, gen=3000, verbose=False)

# coco
new_anchors = autoAnchor.kmean_anchors(path=r"/home/willy20104368/yolov7/data/Caltech_fg.yaml", n=6, img_size=640, thr=4, gen=1000, verbose=False)
print(new_anchors)