python run_infer.py \
--gpu='0' \
--nr_types=6 \
--type_info_path=type_info.json \
--batch_size=8 \
--model_mode=fast \
--model_path=/home/histopath/Model/hovernet/logs/1/net_epoch=50.tar \
--nr_inference_workers=8 \
--nr_post_proc_workers=16 \
tile \
--input_dir=/home/histopath/Data/SCRC_cell_test/ \
--output_dir=/home/histopath/Data/SCRC_cell_test/ \

