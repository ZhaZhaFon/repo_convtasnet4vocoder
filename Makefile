prepare-BZNSYP:
	python preprocess.py --dataset biaobei --dataset_dir /home/jiangji/basic/dataset/BZNSYP --save_dir /home/jiangji/basic/dataset/BZNSYP-convtasnet 

train:
	rm -rf /home/jiangji/basic/exp-basis/convtasnet_0419
	CUDA_VISIBLE_DEVICES=3 python train.py --save_dir /home/jiangji/basic/exp-basis/convtasnet_0419