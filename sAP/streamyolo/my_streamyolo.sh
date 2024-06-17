dataDir="/home/shuxue3/gjt/new/streamyolo-sc-obj/data/"  # path/to/your/data
config="/home/shuxue3/gjt/new/streamyolo-sc-obj/cfgs/l_s50_onex_dfp_tal_flip.py" # path/to/your/cfg
weights="/home/shuxue3/gjt/new/streamyolo-sc-obj/3776.pth"  # path/to/your/checkpoint_path

scale=0.5

python streamyolo_det.py \                                     
	--data-root "$dataDir/Argoverse-1.1/tracking" \
	--annot-path "$dataDir/Argoverse-HD/annotations/val.json" \
	--fps 5 \
	--weights $weights \
	--in_scale 0.5 \
	--no-mask \
	--out-dir "./data/online_resuklt/l_s50" \
	--overwrite \
	--config $config \
#    &&

# gt_data="/home/shuxue3/gjt/new/streamyolo-sc-obj/sAP/streamyolo/data/online_resuklt/l_s50/vis_gt"
# # --data-root "$dataDir/Argoverse-1.1/tracking" \

# python  streaming_eval.py \
# 	--data-root "$gt_data" \
# 	--annot-path "$dataDir/Argoverse-HD/annotations/val.json" \
# 	--fps 5 \
# 	--eta 0 \
# 	--result-dir "./data/online_resuklt/l_s50" \
# 	--out-dir "./data/online_resuklt/l_s50" \
#     --vis-dir "./data/online_resuklt/l_s50/vis" \
# 	--overwrite \
# 	# --vis-scale 0.5 \

