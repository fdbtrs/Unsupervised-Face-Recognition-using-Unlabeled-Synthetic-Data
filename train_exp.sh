export OMP_NUM_THREADS=4
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 \
--node_rank=0 --master_addr="127.0.0.1" --master_port=1234 train.py --num_ops $1 --mag $2 --aug_operation $3 --margin $4 --q_size $5 --q_type $6 --db_size $7
ps -ef | grep "train" | grep -v grep | awk '{print "kill -9 "$2}' | sh
