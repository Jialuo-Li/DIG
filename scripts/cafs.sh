vdir="data/$1"

[ "$1" == "longvideobench" ] && vdir="data/$1/videos"
[ "$1" == "videomme" ]       && vdir="data/$1/data"

accelerate launch --num_processes 96 pipeline/cafs.py \
    --dataset "$1" \
    --json_file "data/${1}_meta.json" \
    --video_dir "$vdir" \
    --output_file "data/${1}_meta.json" \
    --sample_per_sec 2