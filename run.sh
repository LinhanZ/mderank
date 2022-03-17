#This is script for MDERank testing
stage=$1

# test MDERank
if  [ $stage -eq 1 ];then
  dataset_name=krapivin
  CUDA_VISIBLE_DEVICES=3 python MDERank/PDRank_main.py --dataset_dir data/$dataset_name --batch_size 1  --doc_embed_mode max \
  --checkpoints /home/zhanglinhan.zlh/word_embedding_bias/results/test-mlm/checkpoint-4000/pytorch_model.bin --log_dir results --dataset_name $dataset_name
fi

#test KPEBERT
if  [ $stage -eq 2 ];then
  dataset_name=Inspec
  CUDA_VISIBLE_DEVICES=3 python MDERank/mderank_kpebert.py --dataset_dir data/$dataset_name --batch_size 1 --distance cos --doc_embed_mode max \
  --model_name_or_path KPEBERT/result/distance_0.2_bs32_YAKE_cos/checkpoint-70000 --log_dir KPEBERT/result/ --dataset_name $dataset_name --layer_num -1
fi
