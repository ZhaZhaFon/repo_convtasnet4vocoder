prepare-BZNSYP:
	python preprocess.py --dataset biaobei --dataset_dir /home/jiangji/basic/dataset/BZNSYP --save_dir /home/jiangji/basic/dataset-basismelgan/BZNSYP-convtasnet 

train:
	CUDA_VISIBLE_DEVICES=3 python train.py --save_dir /home/jiangji/basic/exp-basismelgan/convtasnet_0419

prepare-basismelgan:
	CUDA_VISIBLE_DEVICES=3 python generator.py --convtasnet_path /home/jiangji/basic/exp-basismelgan/convtasnet_0419/checkpoint_60000.pth.tar --save_dir /home/jiangji/basic/dataset-basismelgan/BZNSYP-DecoderBasis
