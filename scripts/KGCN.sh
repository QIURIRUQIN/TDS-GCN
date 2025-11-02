#!/bin/bash

python /root/autodl-tmp/gnn_based_rec/run.py \
--model_name 'KGCN' \
--epochs 10 \
--n_layers 2 \
--hidden_dim 512 \
--dims '[512, 512]' \
--top_k 10 \
--batch_size 128 \
--dgi_graph_act 'sigmoid' \
--time_step 30 \
--learning_rate 0.0001 \
--use_multi_label \
--rating_class 5 \
--loss_weight_method 'HM' \
# --handle_over_corr \
# --weight \
