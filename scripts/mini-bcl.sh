#!/bin/bash

#MY_PYTHON="CUDA_VISIBLE_DEVICES=3 python"

CIFAR_100i="--data_path data/ --save_path results/ --batch_size 10 --data_file mini-cl.pt  --cuda yes --seed 0 --n_epochs 1 --use 1 --inner_steps 3 --memory_strength 100 --temperature 5 --adapt no"
for i in {1..5}
do
    CUDA_VISIBLE_DEVICES=3 python2 main.py $CIFAR_100i --model bcl-adapt --lr 0.05 --n_memories 65 --replay_batch_size 128
done


