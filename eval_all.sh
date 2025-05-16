set -e

# Create a temporary file to track GPU usage
GPU_STATUS_FILE=$(mktemp /tmp/gpu_status.XXXXXX)
NUM_GPUS=7

# Initialize GPU status (0 = free, 1 = busy)
for i in $(seq 0 $((NUM_GPUS - 1))); do
	echo "0" >"${GPU_STATUS_FILE}_$i"
done

cleanup() {
	# Clean up temp files
	for i in $(seq 0 $((NUM_GPUS - 1))); do
		rm -f "${GPU_STATUS_FILE}_$i"
	done
	rm -f "$GPU_STATUS_FILE"
}

trap cleanup EXIT

run() {
	model=$1
	quant_config=$2
	gpu_id=$3
	job_id=$4

	echo "Running $model with quant config $quant_config on GPU $gpu_id (job $job_id)"

	# Mark GPU as busy
	echo "1" >"${GPU_STATUS_FILE}_$gpu_id"

	# Run the command
	CUDA_VISIBLE_DEVICES=$gpu_id OMP_NUM_THREADS=8 torchrun --nnodes=1 --nproc_per_node=1 --master-port $((24501 + gpu_id)) ./exps/eval_llm.py \
		--tasks arc_easy,arc_challenge,boolq,piqa,social_iqa,hellaswag,openbookqa,winogrande \
		--model_name $model \
		$quant_config

	# Mark GPU as free when done
	echo "0" >"${GPU_STATUS_FILE}_$gpu_id"
	echo "Job $job_id completed on GPU $gpu_id"
}

# Find next available GPU
get_free_gpu() {
	while true; do
		for i in $(seq 0 $((NUM_GPUS - 1))); do
			if [[ $(cat "${GPU_STATUS_FILE}_$i") -eq 0 ]]; then
				echo $i
				return
			fi
		done
		sleep 1
	done
}

models=(
	"facebook/MobileLLM-125M"
	"facebook/MobileLLM-600M"
	"meta-llama/Llama-3.2-1B"
	"meta-llama/Llama-3.2-3B"
)

quant_configs=(
	"--label BF16-baseline"
	"--weight_qtype float --weight_qbits 4 --weight_format e2m1 --weight_alg minmax --weight_block_size channel"
	"--weight_qtype float --weight_qbits 4 --weight_format e2m1 --weight_alg iterative --weight_block_size channel"
	"--weight_qtype float --weight_qbits 4 --weight_format e2m1 --weight_alg normal --weight_block_size channel"
	"--weight_qtype float --weight_qbits 4 --weight_format e2m1 --weight_alg lsq --weight_block_size channel"
	"--weight_qtype uniform --weight_qbits 4 --weight_alg minmax --weight_block_size channel"
	"--weight_qtype uniform --weight_qbits 4 --weight_alg iterative --weight_block_size channel"
	"--weight_qtype uniform --weight_qbits 4 --weight_alg normal --weight_block_size channel"
	"--weight_qtype uniform --weight_qbits 4 --weight_alg lsq --weight_block_size channel"
	"--weight_qtype float --weight_qbits 4 --weight_format e2m1 --weight_alg minmax"
	"--weight_qtype float --weight_qbits 4 --weight_format e2m1 --weight_alg iterative"
	"--weight_qtype float --weight_qbits 4 --weight_format e2m1 --weight_alg normal"
	"--weight_qtype float --weight_qbits 4 --weight_format e2m1 --weight_alg lsq"
	"--weight_qtype uniform --weight_qbits 4 --weight_alg minmax"
	"--weight_qtype uniform --weight_qbits 4 --weight_alg iterative"
	"--weight_qtype uniform --weight_qbits 4 --weight_alg normal"
	"--weight_qtype uniform --weight_qbits 4 --weight_alg lsq"
)

job_id=0

for model in "${models[@]}"; do
	# Run once without quantconfig for bf16 baseline
	for quant_config in "${quant_configs[@]}"; do
		gpu=$(get_free_gpu)
		run "$model" "$quant_config" $gpu $job_id &
		job_id=$((job_id + 1))
	done
done
wait
