#This is script for MDERank testing
# test MDERank
# Dataset name: Inspec, SemEval2010, SemEval2017, DUC2001, nus, krapivin
# Please download data first and save in 'data' folder.
dataset_name=Inspec
CUDA_VISIBLE_DEVICES=0 python MDERank/mderank_kpebert.py --dataset_dir data/$dataset_name --batch_size 1 --distance cos --doc_embed_mode max \
--model_name_or_path model_name_or_path --log_dir log_path --dataset_name $dataset_name --layer_num -1

