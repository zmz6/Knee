python train_iter_hip_edge.py --model unet --downsample 1 &
python train_iter_hip_edge.py --model scn --downsample 1
python train_iter_hip_edge.py --model four_unet --downsample 1 &
python train_iter_hip_edge.py --model hg_4 --downsample 4
python train_iter_hip_edge.py --model simple_34 --downsample 4 &
python train_iter_hip_edge.py --model hrnet_18 --downsample 4
python train_iter_hip_edge.py --model headscn --downsample 1 --beta 1.5e-5&
python train_iter_hip_edge.py --model headscn --downsample 1 --beta 1e-6
python train_iter_hip_edge.py --model headscn --downsample 1 --beta 1e-4 &
python train_iter_hip_edge.py --model headscn --downsample 1 --beta 1e-3
