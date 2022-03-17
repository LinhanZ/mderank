#This is script for MDERank testing
# test MDERank
dataset_name=Inspec
CUDA_VISIBLE_DEVICES=3 python MDERank/mderank_kpebert.py --dataset_dir data/$dataset_name --batch_size 1 --distance cos --doc_embed_mode max \
--model_name_or_path model_name_or_path --log_dir KPEBERT/result/ --dataset_name $dataset_name --layer_num -1

