#python train_iter_knee.py --model unet --downsample 1 &
#python train_iter_knee.py --model scn --downsample 1
#python train_iter_knee.py --model four_unet --downsample 1 &
#python train_iter_knee.py --model hg_4 --downsample 4
#python train_iter_knee.py --model simple_34 --downsample 4 &
#python train_iter_knee.py --model hrnet_18 --downsample 4
python train_iter_knee.py --model nl_unet --downsample 1 --cv 1 --cuda 0 &
python train_iter_knee.py --model nl_unet --downsample 1 --cv 2 --cuda 0 &
python train_iter_knee.py --model nl_unet --downsample 1 --cv 3 --cuda 1 &
python train_iter_knee.py --model nl_unet --downsample 1 --cv 4 --cuda 1 &
python train_iter_knee.py --model nl_unet --downsample 1 --cv 5 --cuda 2 &
python train_iter_knee.py --model nl_unet --downsample 1 --cv 6 --cuda 2 &