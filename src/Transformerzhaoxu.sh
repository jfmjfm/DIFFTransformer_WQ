#export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/zhaoxu/ \
  --data_path merged_all.csv \
  --model_id zhaoxu2 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 288 \
  --pred_len 48 \
  --e_layers 8 \
  --enc_in 14 \
  --dec_in 14 \
  --c_out 14 \
  --des 'zhaoxu' \
  --d_model 512\
  --d_ff 512\
  --do_predict\
  --target 'TOC'\
  --inverse \
  --freq 'h' \
  --itr 1
