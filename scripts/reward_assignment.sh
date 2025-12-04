export DECORD_EOF_RETRY_MAX=40960

: "${MODEL_NAME:?Env var MODEL_NAME missing}" "${1:?Dataset name missing}"

VDIR="data/$1"
[[ "$1" =~ ^(longvideobench|videomme)$ ]] && VDIR+="/videos"

mkdir -p rewards

python pipeline/reward_assignment.py \
    --data_file "data/${1}_meta.json" \
    --video_dir "$VDIR" \
    --dataset "$1" \
    --output_file "rewards/${MODEL_NAME##*/}_${1}.json" \
    --num_workers $(nproc) \
    --batch_size 100 \
    --concurrency 800