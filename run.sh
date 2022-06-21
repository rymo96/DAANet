CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 nohup python -u -m torch.distributed.launch --nproc_per_node 4 train_dist_ema.py --data-dir /benchmarks/ILSVRC2012_LMDB &
