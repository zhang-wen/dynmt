#python $1 --dynet-gpu --dynet_mem 5000 FOR,BACK,PARAM --dynet-gpu-ids 2
#python $1 --dynet-gpu --dynet_mem 5000 --dynet-gpu-ids 2
python $1 --dynet-gpu --dynet_mem 3000,6000,1000 --dynet-gpu-ids 1 -DHAVE_CUDA
#python $1 --dynet-gpu
