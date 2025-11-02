#!/bin/bash

python /root/autodl-tmp/gnn_based_rec/run.py \
--model_name 'Afd_LightGCN' \
--epochs 10 \
--n_layers 2 \
--hidden_dim 512 \
--weight \
--top_k 10 \
--learning_rate 0.0001
