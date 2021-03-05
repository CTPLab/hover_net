python run_infer.py \
--gpu='0' \
--nr_types=6 \
--type_info_path=type_info.json \
--batch_size=16 \
--model_mode=fast \
--model_path=/home/histopath/Github/hover_net/logs/0/net_epoch=1.tar \
--nr_inference_workers=16 \
--nr_post_proc_workers=16 \
tile \
--input_dir=/home/histopath/Data/SCRC_nuclei/TMA_spots/ \
--output_dir=/home/histopath/Data/SCRC_nuclei/TMA_pred/ \

