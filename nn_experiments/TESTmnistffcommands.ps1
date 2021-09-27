$supersub="True"
$heuristic="greedy" 
$split="per_raw"
$keep_frac=0.1


python mnist_launch.py --exp_root=TESTmnistexperiments --exp_name=0000-greedy --simple=True --lr=0.000527 --weight_decay=1.009799e-03 --keep_frac=$keep_frac --bootstrap_train=True --supersub=$supersub --heuristic=$heuristic --split=$split --epochs=1

$supersub="True"
$heuristic="UCB" 
$split="per_raw"
$keep_frac=0.1



python mnist_launch.py --exp_root=TESTmnistexperiments --exp_name=0000-UCB --simple=True --lr=0.000527 --weight_decay=1.009799e-03 --keep_frac=$keep_frac --bootstrap_train=True --supersub=$supersub --heuristic=$heuristic --split=$split --epochs=1


$supersub="True"
$heuristic="Thomspon" 
$split="per_raw"
$keep_frac=0.1


python mnist_launch.py --exp_root=TESTmnistexperiments --exp_name=0000-Thompson --simple=True --lr=0.000527 --weight_decay=1.009799e-03 --keep_frac=$keep_frac --bootstrap_train=True --supersub=$supersub --heuristic=$heuristic --split=$split --epochs=1


$supersub="True"
$heuristic="other" 
$split="per_raw"
$keep_frac=0.1



python mnist_launch.py --exp_root=TESTmnistexperiments --exp_name=0000-other --simple=True --lr=0.000527 --weight_decay=1.009799e-03 --keep_frac=$keep_frac --bootstrap_train=True --supersub=$supersub --heuristic=$heuristic --split=$split --epochs=1
