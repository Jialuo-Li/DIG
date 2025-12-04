vdir="data/$1"

[ "$1" == "longvideobench" ] && vdir="data/$1/videos"
[ "$1" == "videomme" ]       && vdir="data/$1/data"

CONCURRENCY=300

python pipeline/query_identification.py \
    --data_file "data/$1.json" \
    --video_dir "$vdir" \
    --dataset "$1" \
    --output_file "data/${1}_meta.json" \
    --batch_size 500 \
    --concurrency $CONCURRENCY