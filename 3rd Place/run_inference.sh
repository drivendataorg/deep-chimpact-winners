#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

# 1x V100
BATCH=16

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

echo "Model: run-20211101-1526-down-a1p50-f7s1"
cd $HOME/solution/models/run-20211101-1526-down-a1p50-f7s1
python3 run.py --job=test --batch_size=$BATCH

echo "Model: run-20211101-2141-down-a1p50-f9s1"
cd $HOME/solution/models/run-20211101-2141-down-a1p50-f9s1
python3 run.py --job=test --batch_size=$BATCH

echo "Model: run-20211103-1552-down-a2p50-f7s1"
cd $HOME/solution/models/run-20211103-1552-down-a2p50-f7s1
python3 run.py --job=test --batch_size=$BATCH

echo "Model: run-20211103-1744-down-a2p80-f7s1"
cd $HOME/solution/models/run-20211103-1744-down-a2p80-f7s1
python3 run.py --job=test --batch_size=$BATCH

echo "Model: run-20211103-1749-down-a1p80-f7s1"
cd $HOME/solution/models/run-20211103-1749-down-a1p80-f7s1
python3 run.py --job=test --batch_size=$BATCH

echo "Model: run-20211104-1815-down-a4p50-f7s1"
cd $HOME/solution/models/run-20211104-1815-down-a4p50-f7s1
python3 run.py --job=test --batch_size=$BATCH

echo "Model: run-20211107-0019-down-a4p50-f7s2"
cd $HOME/solution/models/run-20211107-0019-down-a4p50-f7s2
python3 run.py --job=test --batch_size=$BATCH

echo "Model: run-20211113-1639-full-a1p50-f7s1"
cd $HOME/solution/models/run-20211113-1639-full-a1p50-f7s1
python3 run.py --job=test --batch_size=$BATCH

echo "Model: run-20211113-1642-full-a1p50-f9s1"
cd $HOME/solution/models/run-20211113-1642-full-a1p50-f9s1
python3 run.py --job=test --batch_size=$BATCH

echo "Model: run-20211113-1645-full-a2p50-f7s1"
cd $HOME/solution/models/run-20211113-1645-full-a2p50-f7s1
python3 run.py --job=test --batch_size=$BATCH

echo "Model: run-20211113-1648-full-a4p50-f7s1"
cd $HOME/solution/models/run-20211113-1648-full-a4p50-f7s1
python3 run.py --job=test --batch_size=$BATCH

echo "Model: run-20211113-1847-full-a4p50-f7s2"
cd $HOME/solution/models/run-20211113-1847-full-a4p50-f7s2
python3 run.py --job=test --batch_size=$BATCH

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

cd $HOME/solution
echo "Ensembling..."
python3 ensemble.py

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

