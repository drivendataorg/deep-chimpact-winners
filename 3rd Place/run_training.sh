#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

# TPUv3-8 settings

# TPU=node_name
# BATCH=96
# MIX=None
# DROP=False


# 8x V100 settings

TPU=None
BATCH=32
MIX=mixed_float16
DROP=True

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

echo "Model: run-20211101-1526-down-a1p50-f7s1"
cd $HOME/solution/models/run-20211101-1526-down-a1p50-f7s1
python3 run.py --tpu_ip_or_name=$TPU --batch_size=$BATCH --mixed_precision=$MIX --drop_remainder=$DROP

echo "Model: run-20211101-2141-down-a1p50-f9s1"
cd $HOME/solution/models/run-20211101-2141-down-a1p50-f9s1
python3 run.py --tpu_ip_or_name=$TPU --batch_size=$BATCH --mixed_precision=$MIX --drop_remainder=$DROP

echo "Model: run-20211103-1552-down-a2p50-f7s1"
cd $HOME/solution/models/run-20211103-1552-down-a2p50-f7s1
python3 run.py --tpu_ip_or_name=$TPU --batch_size=$BATCH --mixed_precision=$MIX --drop_remainder=$DROP

echo "Model: run-20211103-1744-down-a2p80-f7s1"
cd $HOME/solution/models/run-20211103-1744-down-a2p80-f7s1
python3 run.py --tpu_ip_or_name=$TPU --batch_size=$BATCH --mixed_precision=$MIX --drop_remainder=$DROP

echo "Model: run-20211103-1749-down-a1p80-f7s1"
cd $HOME/solution/models/run-20211103-1749-down-a1p80-f7s1
python3 run.py --tpu_ip_or_name=$TPU --batch_size=$BATCH --mixed_precision=$MIX --drop_remainder=$DROP

echo "Model: run-20211104-1815-down-a4p50-f7s1"
cd $HOME/solution/models/run-20211104-1815-down-a4p50-f7s1
python3 run.py --tpu_ip_or_name=$TPU --batch_size=$BATCH --mixed_precision=$MIX --drop_remainder=$DROP

echo "Model: run-20211107-0019-down-a4p50-f7s2"
cd $HOME/solution/models/run-20211107-0019-down-a4p50-f7s2
python3 run.py --tpu_ip_or_name=$TPU --batch_size=$BATCH --mixed_precision=$MIX --drop_remainder=$DROP

echo "Model: run-20211113-1639-full-a1p50-f7s1"
cd $HOME/solution/models/run-20211113-1639-full-a1p50-f7s1
python3 run.py --tpu_ip_or_name=$TPU --batch_size=$BATCH --mixed_precision=$MIX --drop_remainder=$DROP

echo "Model: run-20211113-1642-full-a1p50-f9s1"
cd $HOME/solution/models/run-20211113-1642-full-a1p50-f9s1
python3 run.py --tpu_ip_or_name=$TPU --batch_size=$BATCH --mixed_precision=$MIX --drop_remainder=$DROP

echo "Model: run-20211113-1645-full-a2p50-f7s1"
cd $HOME/solution/models/run-20211113-1645-full-a2p50-f7s1
python3 run.py --tpu_ip_or_name=$TPU --batch_size=$BATCH --mixed_precision=$MIX --drop_remainder=$DROP

echo "Model: run-20211113-1648-full-a4p50-f7s1"
cd $HOME/solution/models/run-20211113-1648-full-a4p50-f7s1
python3 run.py --tpu_ip_or_name=$TPU --batch_size=$BATCH --mixed_precision=$MIX --drop_remainder=$DROP

echo "Model: run-20211113-1847-full-a4p50-f7s2"
cd $HOME/solution/models/run-20211113-1847-full-a4p50-f7s2
python3 run.py --tpu_ip_or_name=$TPU --batch_size=$BATCH --mixed_precision=$MIX --drop_remainder=$DROP

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

