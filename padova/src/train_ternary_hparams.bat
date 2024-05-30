@REM python train_ternary.py --dreg "linear" --dmax 0.05
@REM python train_ternary.py --dreg "linear" --dmax 0.10
@REM python train_ternary.py --dreg "linear" --dmax 0.20

@REM python train_ternary.py --dreg "square" --dmax 0.05
@REM python train_ternary.py --dreg "square" --dmax 0.10
@REM python train_ternary.py --dreg "square" --dmax 0.20

@REM python train_ternary.py --dreg "sqrt" --dmax 0.05
@REM python train_ternary.py --dreg "sqrt" --dmax 0.10
@REM python train_ternary.py --dreg "sqrt" --dmax 0.20

@REM python train_ternary.py --dreg "const" --dmax 0.05
@REM python train_ternary.py --dreg "const" --dmax 0.10
@REM python train_ternary.py --dreg "const" --dmax 0.20


@REM ============= BATCH 2 ========================== (logs1)
@REM ================================================

@REM python train_ternary.py --dreg "const"  --dmax 0.3
@REM python train_ternary.py --dreg "linear" --dmax 0.3
@REM python train_ternary.py --dreg "square" --dmax 0.3
@REM python train_ternary.py --dreg "sqrt"   --dmax 0.3

@REM python train_ternary.py --dreg "const"  --dmax 0.5
@REM python train_ternary.py --dreg "linear" --dmax 0.5
@REM python train_ternary.py --dreg "square" --dmax 0.5
@REM python train_ternary.py --dreg "sqrt"   --dmax 0.5

@REM python train_ternary.py --dreg "const"  --dmax 0.7
@REM python train_ternary.py --dreg "linear" --dmax 0.7
@REM python train_ternary.py --dreg "square" --dmax 0.7
@REM python train_ternary.py --dreg "sqrt"   --dmax 0.7


@REM ================= LOG ========================== (logs2)
@REM ================================================

@REM python train_ternary.py --dreg "log" --dmax 0.05
@REM python train_ternary.py --dreg "log" --dmax 0.10
@REM python train_ternary.py --dreg "log" --dmax 0.20
@REM python train_ternary.py --dreg "log" --dmax 0.30
@REM python train_ternary.py --dreg "log" --dmax 0.50
@REM python train_ternary.py --dreg "log" --dmax 0.70

@REM python train_ternary.py --dreg "log" --dmax 0.05 --dmaxep 250
@REM python train_ternary.py --dreg "log" --dmax 0.10 --dmaxep 250
@REM python train_ternary.py --dreg "log" --dmax 0.20 --dmaxep 250
@REM python train_ternary.py --dreg "log" --dmax 0.30 --dmaxep 250
@REM python train_ternary.py --dreg "log" --dmax 0.50 --dmaxep 250
@REM python train_ternary.py --dreg "log" --dmax 0.70 --dmaxep 250

@REM python train_ternary.py --dreg "square" --dmax 0.05 --dmaxep 250
@REM python train_ternary.py --dreg "square" --dmax 0.10 --dmaxep 250
@REM python train_ternary.py --dreg "square" --dmax 0.20 --dmaxep 250
@REM python train_ternary.py --dreg "square" --dmax 0.30 --dmaxep 250
@REM python train_ternary.py --dreg "square" --dmax 0.50 --dmaxep 250
@REM python train_ternary.py --dreg "square" --dmax 0.70 --dmaxep 250


@REM ================= DMIN ========================= (logs3)
@REM ================================================

@REM python train_ternary.py --dreg "linear" --dmax 0.20  --dmin 0.1
@REM python train_ternary.py --dreg "linear" --dmax 0.20  --dmin 0.01
@REM python train_ternary.py --dreg "linear" --dmax 0.20  --dmin 0.001
@REM python train_ternary.py --dreg "linear" --dmax 0.20  --dmin 0.0001
@REM python train_ternary.py --dreg "linear" --dmax 0.20  --dmin 0.0000


@REM ================= BINARY ACTIVATIONS ================= (logs4 ~not yet impl)
@REM ======================================================

python train_ternary.py --dreg "linear" --dmax 0.05  --af32
python train_ternary.py --dreg "linear" --dmax 0.10  --af32
python train_ternary.py --dreg "linear" --dmax 0.20  --af32

python train_ternary.py --dreg "square" --dmax 0.05  --af32
python train_ternary.py --dreg "square" --dmax 0.10  --af32
python train_ternary.py --dreg "square" --dmax 0.20  --af32

python train_ternary.py --dreg "sqrt" --dmax 0.05  --af32
python train_ternary.py --dreg "sqrt" --dmax 0.10  --af32
python train_ternary.py --dreg "sqrt" --dmax 0.20  --af32

python train_ternary.py --dreg "const" --dmax 0.05  --af32
python train_ternary.py --dreg "const" --dmax 0.10  --af32
python train_ternary.py --dreg "const" --dmax 0.20  --af32
