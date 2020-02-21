#!/bin/bash

#MY_PYTHON="CUDA_VISIBLE_DEVICES=4 python"



CIFAR_100i="--data_path data/ --save_path results/ --batch_size 10 --data_file cifar-cl.pt --cuda yes --n_epochs 1 --use 1.0 --inner_steps 3 --beta 0.3 --adapt no" 
for i in {1..5}
do
    CUDA_VISIBLE_DEVICES=0 python2 main.py $CIFAR_100i --model bcl-adapt --lr 0.1 --n_memories 65 --temperature 8 --memory_strength 100 --replay_batch_size 64
done



