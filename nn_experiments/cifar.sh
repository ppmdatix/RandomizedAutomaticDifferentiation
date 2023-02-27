
python cifar_launch.py  --exp_root=cifarexperiments --exp_name=00-ssb-Kmax1-M1-0o1kf --simple=True --lr=0.000435 --weight_decay=5.485464e-05 --keep_frac=0.1   --repeat_ssb=1   --draw_ssb=1 --supersub_from_rad=True
python cifar_launch.py  --exp_root=cifarexperiments --exp_name=01-ssb-Kmax1-M1-0o1kf --simple=True --lr=0.000435 --weight_decay=5.485464e-05 --keep_frac=0.1   --repeat_ssb=1   --draw_ssb=1 --supersub_from_rad=True


python cifar_launch.py  --exp_root=cifarexperiments --exp_name=00-ssb-Kmax1-M5-0o1kf --simple=True --lr=0.000435 --weight_decay=5.485464e-05 --keep_frac=0.1   --repeat_ssb=1   --draw_ssb=5 --supersub_from_rad=True
python cifar_launch.py  --exp_root=cifarexperiments --exp_name=01-ssb-Kmax1-M5-0o1kf --simple=True --lr=0.000435 --weight_decay=5.485464e-05 --keep_frac=0.1   --repeat_ssb=1   --draw_ssb=5 --supersub_from_rad=True

python cifar_launch.py  --exp_root=cifarexperiments --exp_name=00-ssb-Kmax5-M1-0o1kf --simple=True --lr=0.000435 --weight_decay=5.485464e-05 --keep_frac=0.1   --repeat_ssb=5   --draw_ssb=1 --supersub_from_rad=True
python cifar_launch.py  --exp_root=cifarexperiments --exp_name=01-ssb-Kmax5-M1-0o1kf --simple=True --lr=0.000435 --weight_decay=5.485464e-05 --keep_frac=0.1   --repeat_ssb=5   --draw_ssb=1 --supersub_from_rad=True


python cifar_launch.py  --exp_root=cifarexperiments --exp_name=00-ssb-Kmax5-M5-0o1kf --simple=True --lr=0.000435 --weight_decay=5.485464e-05 --keep_frac=0.1   --repeat_ssb=5   --draw_ssb=5 --supersub_from_rad=True
python cifar_launch.py  --exp_root=cifarexperiments --exp_name=01-ssb-Kmax5-M5-0o1kf --simple=True --lr=0.000435 --weight_decay=5.485464e-05 --keep_frac=0.1   --repeat_ssb=5   --draw_ssb=5 --supersub_from_rad=True


python cifar_launch.py  --exp_root=cifarexperiments --exp_name=00-baseline --simple=True --lr=0.000435 --weight_decay=5.485464e-05 --keep_frac=1.0  --supersub_from_rad=False
python cifar_launch.py  --exp_root=cifarexperiments --exp_name=01-baseline --simple=True --lr=0.000435 --weight_decay=5.485464e-05 --keep_frac=1.0  --supersub_from_rad=False
python cifar_launch.py  --exp_root=cifarexperiments --exp_name=02-baseline --simple=True --lr=0.000435 --weight_decay=5.485464e-05 --keep_frac=1.0  --supersub_from_rad=False
