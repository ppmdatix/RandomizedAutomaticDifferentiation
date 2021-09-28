
$heuristic="Thompson" 
$split="per_column"
$keep_frac=0.5
$folder = "COMPheuristic" + "-" + $split + "-" + $heuristic + $keep_frac
$lr=0.000527
$files = $folder 

$seeds = 0..5
For($i=0;$i -lt 10;$i++) 
{ 
   $seed = $seeds[$i]
   $file = $files + $seed 
   $supersub="True"
   python mnist_launch.py --exp_root=$folder --exp_name=$file --simple=True --lr=$lr --weight_decay=1.009799e-03 --keep_frac=$keep_frac --bootstrap_train=True --supersub=$supersub --heuristic=$heuristic --split=$split --seed=$seed 
   # $supersub="False"
   # $file = "base" + $seed
   # python mnist_launch.py --exp_root=$folder --exp_name=$file --simple=True --lr=$lr --weight_decay=1.009799e-03 --keep_frac=1 --bootstrap_train=True --seed=$seed --supersub=$supersub --plot="True"

}