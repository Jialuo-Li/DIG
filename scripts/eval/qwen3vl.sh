# ================= Configuration =================
export HF_HOME="${HF_HOME:-~/.cache/huggingface}"
export DECORD_EOF_RETRY_MAX=20480

# Colors for logging
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Helper Functions
log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_err()  { echo -e "${RED}[ERROR]${NC} $1"; }

usage() {
    echo "Usage: bash $0 {mlvu|longvideobench|videomme} {DIG|UNI}"
    exit 1
}

# ================= Argument Parsing =================
TARGET=$1
MODE=$2
FRAMES_LIST=(8 16 32 64 128 192 256)

if [[ -z "$TARGET" || -z "$MODE" ]]; then
    log_err "Missing arguments."
    usage
fi

if [ -z "$MODEL_NAME" ]; then
    log_err "MODEL_NAME environment variable is not set."
    exit 1
fi

MODEL_BASENAME="${MODEL_NAME##*/}"
mkdir -p logs

# ================= Task Setup =================
case "$TARGET" in
    "mlvu")           TASK_NAME="mlvu_dev";           FILE_KEY="mlvu" ;;
    "longvideobench") TASK_NAME="longvideobench_val_v"; FILE_KEY="longvideobench" ;;
    "videomme")       TASK_NAME="videomme";           FILE_KEY="videomme" ;;
    *)                log_err "Invalid target: $TARGET"; usage ;;
esac

if [[ "$MODE" != "DIG" && "$MODE" != "UNI" ]]; then
    log_err "Invalid mode: $MODE. Choose 'DIG' or 'UNI'."
    usage
fi

log_info "Starting evaluation | Target: $TARGET ($TASK_NAME) | Mode: $MODE | Model: $MODEL_BASENAME"

# ================= Execution Loop =================
for FRAMES in "${FRAMES_LIST[@]}"; do
    log_info "Running evaluation for frames: ${FRAMES}..."

    # Base arguments common to both modes
    BASE_ARGS="pretrained=${MODEL_NAME},max_pixels=262144,min_pixels=25088,max_num_frames=${FRAMES},attn_implementation=flash_attention_2,interleave_visuals=False"

    # Mode-specific configuration
    if [[ "$MODE" == "DIG" ]]; then
        DATA_PATH="keyframes/${MODEL_BASENAME}_${FILE_KEY}_${FRAMES}.json"
        MODEL_ARGS="${BASE_ARGS},use_uniform=False,data_path=${DATA_PATH}"
        LOG_FILE="logs/${MODEL_BASENAME}_DIG_${FRAMES}_${FILE_KEY}.log"
    else # UNI
        MODEL_ARGS="${BASE_ARGS},use_uniform=True"
        LOG_FILE="logs/${MODEL_BASENAME}_UNI_${FRAMES}_${FILE_KEY}.log"
    fi

    # Run Inference
    accelerate launch --num_processes=8 -m lmms_eval \
        --model qwen3_vl \
        --force_simple \
        --model_args "${MODEL_ARGS}" \
        --tasks "${TASK_NAME}" \
        --batch_size 1 > "${LOG_FILE}" 2>&1

    log_info "Finished ${MODE} evaluation for ${FRAMES} frames. Log: ${LOG_FILE}"
done

log_info "All tasks completed successfully."