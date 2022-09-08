# LSREN

Open source code for the paper "A new metric for out-of-distribution detection based on multi-classifiers".

In the catalog LSR-Visual is the code for verifying label smoothing discrepancies using the mnist dataset. Usage examples.

python main.py --seed 0 --net lenet --smoothness 0.1 --epochs 20

python visual_fea.py --dataset_name mnist --dataset_id 0 --net lenet --seed_begin 0 --seed_end 0 --epochs 20

In the catalog SOOD-Evaluate are the evaluation and ablation experiments of the proposed method LSREN on the SOOD dataset. Examples of use.

python main.py --net resnet_euc --in_dataset cifar80 --epochs 100 --samples_pre_class_num 5000 --ood_batch_size 1000
--excute_list train cal_magnitude cal_score --seed_begin 0 --seed_end 0 --ensemble_num 1 --dataset_id 0 --inferance eval

In the catalogs OOD-Benchmark and OSR-Benchmark are the evaluation experiments of the proposed method LSREN on OOD and OSR benchmark datasets. Examples of use.

python main.py --in_dataset tiny_imagenet --ensemble_num 1 --seed 6 --epochs 100 --excute_list train cal_magnitude cal_score

python main.py --in_dataset tiny_imagenet --ensemble_num 2 --seed 6 --epochs 100 --excute_list train cal_magnitude cal_score

python main.py --in_dataset tiny_imagenet --ensemble_num 25 --seed 6 --epochs 100 --excute_list train cal_magnitude cal_score

python ensemble.py --in_dataset tiny_imagenet --ensemble_num 25 --seed 6 --epochs 100 --excute_list cal_score cal_metric write_to_excel cal_acc
