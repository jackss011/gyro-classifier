CUDA_VISIBLE_DEVICES=0 python main_ternary.py  --epochs 500 --delta 0.1 --multiplier 0.1 --maxdelta 0.97 --delta_regime lin;
CUDA_VISIBLE_DEVICES=0 python main_ternary.py  --epochs 500 --delta 0.1 --multiplier 1.7 --maxdelta 0.97 --delta_regime log;
CUDA_VISIBLE_DEVICES=0 python main_ternary.py  --epochs 500 --delta 0.1 --multiplier 1.9 --maxdelta 0.97 --delta_regime log;
CUDA_VISIBLE_DEVICES=0 python main_ternary.py  --epochs 500 --delta 0.1 --multiplier 0.17 --maxdelta 0.97 --delta_regime lin;

