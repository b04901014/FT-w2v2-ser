#datadir, labelpath, savingpath, GPUid, outputname, numexps, wav2vecpath
if [ -d "$3" ]; then
    echo "$3 directory already exists!";
    exit 1;
fi
CUDA_VISIBLE_DEVICES=$4 python run_pretrain.py --precision 16 --datadir $1 --labelpath $2 --labeling_method hard --saving_path $3 --training_step 10000 --save_top_k 1 --wav2vecpath $7 &&
w2v_path=($3/w2v-*);
CUDA_VISIBLE_DEVICES=$4 python cluster.py --datadir $1 --outputdir $3/clusters --model_path $w2v_path --labelpath $2 --model_type wav2vec --sample_ratio 1.0 --num_clusters "64,512,4096" --wav2vecpath $7 &&
CUDA_VISIBLE_DEVICES=$4 python run_second.py --saving_path $3 --precision 16 --datadir $1 --labelpath $3/clusters/all-clus.json --training_step 20000 --warmup_step 100 --save_top_k 1 --lr 1e-4 --batch_size 64 --num_clusters "64,512,4096" --use_bucket_sampler --dynamic_batch &&
mv $3/last.ckpt $3/our_method.ckpt
w2v2_path=$3/our_method.ckpt;
mkdir $3/labels
cp $2 $3/labels/
CUDA_VISIBLE_DEVICES=$4 python run_downstream_custom_multiple_fold.py --precision 16 --num_exps $6 --datadir $1 --labeldir $3/labels --pretrained_path $w2v2_path --outputfile $3/$5.log
