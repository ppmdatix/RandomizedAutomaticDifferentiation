$supersub="True"
$heuristic="greedy" 
$split="per_raw"
$keep_frac=0.1

python mnist_launch.py --exp_root=mnistexperiments --exp_name=0000-project --simple=True --lr=0.000527 --weight_decay=1.009799e-03 --keep_frac=$keep_frac --bootstrap_train=True --supersub=$supersub --heuristic=$heuristic --split=$split
python mnist_launch.py --exp_root=mnistexperiments --exp_name=0001-project --simple=True --lr=0.000527 --weight_decay=1.009799e-03 --keep_frac=$keep_frac --bootstrap_train=True --supersub=$supersub --heuristic=$heuristic --split=$split
python mnist_launch.py --exp_root=mnistexperiments --exp_name=0002-project --simple=True --lr=0.000527 --weight_decay=1.009799e-03 --keep_frac=$keep_frac --bootstrap_train=True --supersub=$supersub --heuristic=$heuristic --split=$split
python mnist_launch.py --exp_root=mnistexperiments --exp_name=0003-project --simple=True --lr=0.000527 --weight_decay=1.009799e-03 --keep_frac=$keep_frac --bootstrap_train=True --supersub=$supersub --heuristic=$heuristic --split=$split
python mnist_launch.py --exp_root=mnistexperiments --exp_name=0004-project --simple=True --lr=0.000527 --weight_decay=1.009799e-03 --keep_frac=$keep_frac --bootstrap_train=True --supersub=$supersub --heuristic=$heuristic --split=$split

$supersub="False"


python mnist_launch.py --exp_root=mnistexperiments --exp_name=0005-smallbatch --simple=True --lr=0.000645 --weight_decay=5.687505e-05 --batch_size=20 --bootstrap_train=True --supersub=$supersub 
python mnist_launch.py --exp_root=mnistexperiments --exp_name=0006-smallbatch --simple=True --lr=0.000645 --weight_decay=5.687505e-05 --batch_size=20 --bootstrap_train=True --supersub=$supersub 
python mnist_launch.py --exp_root=mnistexperiments --exp_name=0007-smallbatch --simple=True --lr=0.000645 --weight_decay=5.687505e-05 --batch_size=20 --bootstrap_train=True --supersub=$supersub 
python mnist_launch.py --exp_root=mnistexperiments --exp_name=0008-smallbatch --simple=True --lr=0.000645 --weight_decay=5.687505e-05 --batch_size=20 --bootstrap_train=True --supersub=$supersub 
python mnist_launch.py --exp_root=mnistexperiments --exp_name=0009-smallbatch --simple=True --lr=0.000645 --weight_decay=5.687505e-05 --batch_size=20 --bootstrap_train=True --supersub=$supersub 

python mnist_launch.py --exp_root=mnistexperiments --exp_name=0010-baseline --simple=True --lr=0.001350 --weight_decay=4.066478e-07 --bootstrap_train=True --supersub=$supersub 
python mnist_launch.py --exp_root=mnistexperiments --exp_name=0011-baseline --simple=True --lr=0.001350 --weight_decay=4.066478e-07 --bootstrap_train=True --supersub=$supersub  
python mnist_launch.py --exp_root=mnistexperiments --exp_name=0012-baseline --simple=True --lr=0.001350 --weight_decay=4.066478e-07 --bootstrap_train=True --supersub=$supersub 
python mnist_launch.py --exp_root=mnistexperiments --exp_name=0013-baseline --simple=True --lr=0.001350 --weight_decay=4.066478e-07 --bootstrap_train=True --supersub=$supersub 
python mnist_launch.py --exp_root=mnistexperiments --exp_name=0014-baseline --simple=True --lr=0.001350 --weight_decay=4.066478e-07 --bootstrap_train=True --supersub=$supersub 

python mnist_launch.py --exp_root=mnistexperiments --exp_name=0015-samesample --simple=True --lr=0.000452 --weight_decay=4.058855e-06 --keep_frac=0.1 --sparse=True --bootstrap_train=True --supersub=$supersub
python mnist_launch.py --exp_root=mnistexperiments --exp_name=0016-samesample --simple=True --lr=0.000452 --weight_decay=4.058855e-06 --keep_frac=0.1 --sparse=True --bootstrap_train=True --supersub=$supersub
python mnist_launch.py --exp_root=mnistexperiments --exp_name=0017-samesample --simple=True --lr=0.000452 --weight_decay=4.058855e-06 --keep_frac=0.1 --sparse=True --bootstrap_train=True --supersub=$supersub
python mnist_launch.py --exp_root=mnistexperiments --exp_name=0018-samesample --simple=True --lr=0.000452 --weight_decay=4.058855e-06 --keep_frac=0.1 --sparse=True --bootstrap_train=True --supersub=$supersub
python mnist_launch.py --exp_root=mnistexperiments --exp_name=0019-samesample --simple=True --lr=0.000452 --weight_decay=4.058855e-06 --keep_frac=0.1 --sparse=True --bootstrap_train=True --supersub=$supersub

python mnist_launch.py --exp_root=mnistexperiments --exp_name=0020-diffsample --simple=True --lr=0.000934 --weight_decay=1.254031e-06 --keep_frac=0.1 --sparse=True --bootstrap_train=True --full_random=True --supersub=$supersub
python mnist_launch.py --exp_root=mnistexperiments --exp_name=0021-diffsample --simple=True --lr=0.000934 --weight_decay=1.254031e-06 --keep_frac=0.1 --sparse=True --bootstrap_train=True --full_random=True --supersub=$supersub
python mnist_launch.py --exp_root=mnistexperiments --exp_name=0022-diffsample --simple=True --lr=0.000934 --weight_decay=1.254031e-06 --keep_frac=0.1 --sparse=True --bootstrap_train=True --full_random=True --supersub=$supersub
python mnist_launch.py --exp_root=mnistexperiments --exp_name=0023-diffsample --simple=True --lr=0.000934 --weight_decay=1.254031e-06 --keep_frac=0.1 --sparse=True --bootstrap_train=True --full_random=True --supersub=$supersub
python mnist_launch.py --exp_root=mnistexperiments --exp_name=0024-diffsample --simple=True --lr=0.000934 --weight_decay=1.254031e-06 --keep_frac=0.1 --sparse=True --bootstrap_train=True --full_random=True --supersub=$supersub
