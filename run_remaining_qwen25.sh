cd ~/Language-Reasoning-Disentangle
export PYTHONPATH=~/Language-Reasoning-Disentangle:~/Language-Reasoning-Disentangle/FreeEvalLM:$PYTHONPATH

mkdir -p results/qwen25_7b_xwinograd_baseline
python mlrs/src/main_generate_steering_task.py \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --vector_path ./vector/qwen25-7b-instruct/parallel_multilingual_space/vector.pt \
    --steering_strength "[0,0]" --task xwinograd_for_mlrs --sample -1 \
    --output_dir ./results/qwen25_7b_xwinograd_baseline \
    --prompt_key prompt --reasoning_key reasoning --reasoning_parser deepseek_r1 \
    --temperature 0.6 --top_p 0.95 --max_tokens 2048 --tensor_parallel_size 1 \
    --steering_method Mlrs --steering_layers "[10,11,12,13,14,15,16,17,18,19]" \
    --steering_layers2 "[20,21,22,23,24,25,26,27]"

mkdir -p results/qwen25_7b_xwinograd_steering
python mlrs/src/main_generate_steering_task.py \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --vector_path ./vector/qwen25-7b-instruct/parallel_multilingual_space/vector.pt \
    --steering_strength "[0.3,-0.1]" --task xwinograd_for_mlrs --sample -1 \
    --output_dir ./results/qwen25_7b_xwinograd_steering \
    --prompt_key prompt --reasoning_key reasoning --reasoning_parser deepseek_r1 \
    --temperature 0.6 --top_p 0.95 --max_tokens 2048 --tensor_parallel_size 1 \
    --steering_method Mlrs --steering_layers "[10,11,12,13,14,15,16,17,18,19]" \
    --steering_layers2 "[20,21,22,23,24,25,26,27]"

mkdir -p results/qwen25_7b_mmmlu_baseline
python mlrs/src/main_generate_steering_task.py \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --vector_path ./vector/qwen25-7b-instruct/parallel_multilingual_space/vector.pt \
    --steering_strength "[0,0]" --task mmmlu_for_mlrs --sample -1 \
    --output_dir ./results/qwen25_7b_mmmlu_baseline \
    --prompt_key input --reasoning_key reasoning --reasoning_parser deepseek_r1 \
    --temperature 0.6 --top_p 0.95 --max_tokens 3072 --tensor_parallel_size 1 \
    --steering_method Mlrs --steering_layers "[10,11,12,13,14,15,16,17,18,19]" \
    --steering_layers2 "[20,21,22,23,24,25,26,27]"

mkdir -p results/qwen25_7b_mmmlu_steering
python mlrs/src/main_generate_steering_task.py \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --vector_path ./vector/qwen25-7b-instruct/parallel_multilingual_space/vector.pt \
    --steering_strength "[0.3,-0.1]" --task mmmlu_for_mlrs --sample -1 \
    --output_dir ./results/qwen25_7b_mmmlu_steering \
    --prompt_key input --reasoning_key reasoning --reasoning_parser deepseek_r1 \
    --temperature 0.6 --top_p 0.95 --max_tokens 3072 --tensor_parallel_size 1 \
    --steering_method Mlrs --steering_layers "[10,11,12,13,14,15,16,17,18,19]" \
    --steering_layers2 "[20,21,22,23,24,25,26,27]"

echo "ALL QWEN2.5 RUNS COMPLETE"
