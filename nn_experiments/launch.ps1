
$split="per_column"
$keep_frac=0.5
$heuristic="Thompson" 
$folder = "stuck-heur-rand--2-3-layers" + "-" + $split + "-" + $heuristic + $keep_frac
$lr=0.000527
$batch_size = 128
$files = "stuck-heur-rand--2-3-layers" + "-" + $split + "-" + $heuristic + $keep_frac
$supersub="True"

$seeds = 0..3
For($i=0;$i -lt 3;$i++) 
{ 
   $seed = $seeds[$i]


   $heuristic="stuck"
   $files = "stuck-heur-rand--2-3-layers" + "-" + $split + "-" + $heuristic + $keep_frac
   $file = $files + $seed 

   python mnist_launch.py `
   		--exp_root=$folder --exp_name=$file --simple=True --lr=$lr --weight_decay=1.009799e-03 `
   		--keep_frac=$keep_frac --bootstrap_train=True --supersub=$supersub --heuristic=$heuristic `
   		--split=$split --seed=$seed --batch_size=$batch_size 

   $heuristic="Thompson" 
   $files = "stuck-heur-rand--2-3-layers" + "-" + $split + "-" + $heuristic + $keep_frac
   $file = $files + $seed 
   python mnist_launch.py `
   		--exp_root=$folder --exp_name=$file --simple=True --lr=$lr --weight_decay=1.009799e-03 `
   		--keep_frac=$keep_frac --bootstrap_train=True --supersub=$supersub --heuristic=$heuristic `
   		--split=$split --seed=$seed --batch_size=$batch_size 

   $heuristic="random"
   $files = "stuck-heur-rand--2-3-layers" + "-" + $split + "-" + $heuristic + $keep_frac
   $file = $files + $seed 
   python mnist_launch.py `
   		--exp_root=$folder --exp_name=$file --simple=True --lr=$lr --weight_decay=1.009799e-03 `
   		--keep_frac=$keep_frac --bootstrap_train=True --supersub=$supersub --heuristic=$heuristic `
   		--split=$split --seed=$seed --batch_size=$batch_size 
   	
   # $supersub="False"
   # $file = "base" + $seed
   # python mnist_launch.py --exp_root=$folder --exp_name=$file --simple=True --lr=$lr --weight_decay=1.009799e-03 --keep_frac=1 --bootstrap_train=True --seed=$seed --supersub=$supersub --plot="True"

}