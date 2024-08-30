# batch one
# python train_triplet.py --model full --bs 256 --lr 0.01     --margin 1.5
# python train_triplet.py --model full --bs 256 --lr 0.001    --margin 1.5
# python train_triplet.py --model full --bs 128 --lr 0.01     --margin 1.5
# python train_triplet.py --model full --bs 128 --lr 0.001    --margin 1.5

# python train_triplet.py --model full --bs 256 --lr 0.01     --margin 5.0
# python train_triplet.py --model full --bs 256 --lr 0.001    --margin 5.0
# python train_triplet.py --model full --bs 128 --lr 0.01     --margin 5.0
# python train_triplet.py --model full --bs 128 --lr 0.001    --margin 5.0

# python train_triplet.py --model full --bs 256 --lr 0.01     --margin 10.0
# python train_triplet.py --model full --bs 256 --lr 0.001    --margin 10.0
# python train_triplet.py --model full --bs 128 --lr 0.01     --margin 10.0
# python train_triplet.py --model full --bs 128 --lr 0.001    --margin 10.0


python train_triplet.py --model ter --bs 256 --lr 0.01      --margin 1.5   --dreg const    --dmax 0.2   --dmaxep 20
python train_triplet.py --model ter --bs 256 --lr 0.01      --margin 1.5   --dreg const    --dmax 0.4   --dmaxep 20
python train_triplet.py --model ter --bs 256 --lr 0.01      --margin 1.5   --dreg const    --dmax 0.6   --dmaxep 20
python train_triplet.py --model ter --bs 256 --lr 0.01      --margin 1.5   --dreg square    --dmax 0.2  --dmaxep 20
python train_triplet.py --model ter --bs 256 --lr 0.01      --margin 1.5   --dreg square    --dmax 0.4  --dmaxep 20
python train_triplet.py --model ter --bs 256 --lr 0.01      --margin 1.5   --dreg square    --dmax 0.6  --dmaxep 20
python train_triplet.py --model ter --bs 256 --lr 0.01      --margin 1.5   --dreg log       --dmax 0.2  --dmaxep 20
python train_triplet.py --model ter --bs 256 --lr 0.01      --margin 1.5   --dreg log       --dmax 0.4  --dmaxep 20
python train_triplet.py --model ter --bs 256 --lr 0.01      --margin 1.5   --dreg log       --dmax 0.6  --dmaxep 20

python train_triplet.py --model ter --bs 256 --lr 0.01      --margin 3.0   --dreg const    --dmax 0.2   --dmaxep 20
python train_triplet.py --model ter --bs 256 --lr 0.01      --margin 3.0   --dreg const    --dmax 0.4   --dmaxep 20
python train_triplet.py --model ter --bs 256 --lr 0.01      --margin 3.0   --dreg const    --dmax 0.6   --dmaxep 20
python train_triplet.py --model ter --bs 256 --lr 0.01      --margin 3.0   --dreg square    --dmax 0.2  --dmaxep 20
python train_triplet.py --model ter --bs 256 --lr 0.01      --margin 3.0   --dreg square    --dmax 0.4  --dmaxep 20
python train_triplet.py --model ter --bs 256 --lr 0.01      --margin 3.0   --dreg square    --dmax 0.6  --dmaxep 20
python train_triplet.py --model ter --bs 256 --lr 0.01      --margin 3.0   --dreg log    --dmax 0.2     --dmaxep 20
python train_triplet.py --model ter --bs 256 --lr 0.01      --margin 3.0   --dreg log    --dmax 0.4     --dmaxep 20
python train_triplet.py --model ter --bs 256 --lr 0.01      --margin 3.0   --dreg log    --dmax 0.6     --dmaxep 20

python train_triplet.py --model ter --bs 256 --lr 0.01      --margin 5.0   --dreg const    --dmax 0.2   --dmaxep 20
python train_triplet.py --model ter --bs 256 --lr 0.01      --margin 5.0   --dreg const    --dmax 0.4   --dmaxep 20
python train_triplet.py --model ter --bs 256 --lr 0.01      --margin 5.0   --dreg const    --dmax 0.6   --dmaxep 20
python train_triplet.py --model ter --bs 256 --lr 0.01      --margin 5.0   --dreg square    --dmax 0.2  --dmaxep 20
python train_triplet.py --model ter --bs 256 --lr 0.01      --margin 5.0   --dreg square    --dmax 0.4  --dmaxep 20
python train_triplet.py --model ter --bs 256 --lr 0.01      --margin 5.0   --dreg square    --dmax 0.6  --dmaxep 20
python train_triplet.py --model ter --bs 256 --lr 0.01      --margin 5.0   --dreg log    --dmax 0.2     --dmaxep 20
python train_triplet.py --model ter --bs 256 --lr 0.01      --margin 5.0   --dreg log    --dmax 0.4     --dmaxep 20
python train_triplet.py --model ter --bs 256 --lr 0.01      --margin 5.0   --dreg log    --dmax 0.6     --dmaxep 20


python train_triplet.py --model bin --bs 256 --lr 0.01      --margin 1.5
python train_triplet.py --model bin --bs 256 --lr 0.001     --margin 1.5
python train_triplet.py --model bin --bs 128 --lr 0.01      --margin 1.5
python train_triplet.py --model bin --bs 128 --lr 0.001     --margin 1.5

python train_triplet.py --model bin --bs 256 --lr 0.01      --margin 5.0
python train_triplet.py --model bin --bs 256 --lr 0.001     --margin 5.0
python train_triplet.py --model bin --bs 128 --lr 0.01      --margin 5.0
python train_triplet.py --model bin --bs 128 --lr 0.001     --margin 5.0

python train_triplet.py --model bin --bs 256 --lr 0.01      --margin 10.0
python train_triplet.py --model bin --bs 256 --lr 0.001     --margin 10.0
python train_triplet.py --model bin --bs 128 --lr 0.01      --margin 10.0
python train_triplet.py --model bin --bs 128 --lr 0.001     --margin 10.0
