#!/bin/bash

python /root/autodl-tmp/gnn_based_rec/run.py \
--model_name 'TDSGCN' \
--epochs 10 \
--n_layers 2 \
--hidden_dim 512 \
--dims '[512, 512]' \
--top_k 10 \
--batch_size 128 \
--dgi_graph_act 'sigmoid' \
--time_step 30 \
--loss_weight_method 'MS' \
--learning_rate 0.01 \
--scaling_factor 0.3
# --weight \
# --use_multi_label \
