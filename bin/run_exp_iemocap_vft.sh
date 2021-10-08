#datadir, labelpath, savingpath, GPUid, outputname, numexps
if [ -d "$3" ]; then
    echo "$3 directory already exists!";
    exit 1;
fi
mkdir -p $3/labels
cp $2 $3/labels/
CUDA_VISIBLE_DEVICES=$4 python run_downstream_custom_multiple_fold.py --precision 16 --num_exps $6 --datadir $1 --labeldir $3/labels --outputfile $3/$5.log
