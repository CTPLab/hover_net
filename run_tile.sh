python run_infer.py \
--gpu='0,1,2,3,4,5,6,7' \
--nr_types=6 \
--type_info_path=type_info.json \
--batch_size=32 \
--model_mode=fast \
--model_path=/raid/jiqing/Github/hover_net/logs/1/net_epoch=4.tar \
--nr_inference_workers=32 \
--nr_post_proc_workers=32 \
tile \
--input_dir=/raid/jiqing/Data/SCRC/TMA_spots/ \
--output_dir=/raid/jiqing/Data/SCRC_pred/ \

