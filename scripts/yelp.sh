#!/bin/bash

python /root/autodl-tmp/gnn_based_rec/run.py \
--model_name 'my_model' \
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
--use_multi_label \
--rating_class 3
# --weight \

