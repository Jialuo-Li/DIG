DATASET_NAME="${1:?Usage: $0 <dataset_name>}"
MODEL_NAME="${MODEL_NAME:?Env var MODEL_NAME missing}"
DATA_FILE="rewards/${MODEL_NAME##*/}_${DATASET_NAME}.json"
[ -f "$DATA_FILE" ] || exit 1

mkdir -p keyframes

for k in 8 16 32 64 128 192 256; do
    echo "k=$k..."
    python pipeline/video_refinement.py \
        --data_file "$DATA_FILE" \
        --output_file "keyframes/$(basename "$DATA_FILE" .json)_${k}.json" \
        --k "$k"
done