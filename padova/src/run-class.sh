# python train.py --bs 256 --lr 0.005 
# python train_binary.py --bs 128 --lr 0.02
# python train_ternary.py --bs 128 --lr 0.01 --dreg log --dmin 0 --dmax 0.3 --dmaxep 250

python train_ternary.py --bs 256 --lr 0.02 --dreg log --dmin 0 --dmax 0.3 --dmaxep 250
python train_ternary.py --bs 256 --lr 0.05 --dreg log --dmin 0 --dmax 0.3 --dmaxep 250
