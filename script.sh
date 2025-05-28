# legacy
#CUDA_VISIBLE_DEVICES=5,6 python main.py --model gpt4o --dataset advbench --jailbreak text_autodan --text_evaluation_model sorrybench
#CUDA_VISIBLE_DEVICES=1 python main.py --evaluate_gpu_id 0 --model gpt4o --dataset advbench --jailbreak text_beast --text_evaluation_model sorrybench
#CUDA_VISIBLE_DEVICES=2 python main.py --evaluate_gpu_id 0 --model gpt4o --dataset advbench --jailbreak text_gcg --text_evaluation_model sorrybench
#CUDA_VISIBLE_DEVICES=8 python main.py --model gpt4o --dataset advbench --jailbreak speechgptaudio --text_evaluation_model sorrybench
#CUDA_VISIBLE_DEVICES=8 python main.py --model gpt4o --dataset advbench --jailbreak none --text_evaluation_model sorrybench
#CUDA_VISIBLE_DEVICES=0,2 python main.py --model qwen2 --evaluate_gpu_id 1 --dataset advbench --jailbreak text_autodan --text_evaluation_model sorrybench
#CUDA_VISIBLE_DEVICES=3,4 python main.py --model qwen2 --evaluate_gpu_id 1 --dataset advbench --jailbreak text_beast --text_evaluation_model sorrybench
#CUDA_VISIBLE_DEVICES=5,6 python main.py --model qwen2 --evaluate_gpu_id 1 --dataset advbench --jailbreak text_gcg --text_evaluation_model sorrybench
#CUDA_VISIBLE_DEVICES=7,8 python main.py --model qwen2 --evaluate_gpu_id 1 --dataset advbench --jailbreak none --text_evaluation_model sorrybench

# speechgpt attack
#CUDA_VISIBLE_DEVICES=0,1 python main.py --model speechgpt --evaluate_gpu_id 1 --dataset advbench --jailbreak text_gcg --text_evaluation_model sorrybench
#CUDA_VISIBLE_DEVICES=3,4 python main.py --model speechgpt --evaluate_gpu_id 1 --dataset advbench --jailbreak text_beast --text_evaluation_model sorrybench
#CUDA_VISIBLE_DEVICES=4,5 python main.py --model speechgpt --evaluate_gpu_id 1 --dataset advbench --jailbreak text_autodan --text_evaluation_model sorrybench
#CUDA_VISIBLE_DEVICES=5,6 python main.py --model speechgpt --evaluate_gpu_id 1 --dataset advbench --jailbreak none --text_evaluation_model sorrybench
#CUDA_VISIBLE_DEVICES=7 python main.py --model speechgpt --idstr sota --start 0 --end 100 --evaluate_gpu_id 0 --dataset advbench --jailbreak audio_ours --text_evaluation_model sorrybench
#CUDA_VISIBLE_DEVICES=2 python main.py --model speechgpt --idstr sota --start 100 --end 200 --evaluate_gpu_id 0 --dataset advbench --jailbreak audio_ours --text_evaluation_model sorrybench
#CUDA_VISIBLE_DEVICES=0 python main.py --model speechgpt --idstr sota --start 267 --end 300 --evaluate_gpu_id 0 --dataset advbench --jailbreak audio_ours --text_evaluation_model sorrybench
#CUDA_VISIBLE_DEVICES=5 python main.py --model speechgpt --idstr sota --start 300 --end 400 --evaluate_gpu_id 0 --dataset advbench --jailbreak audio_ours --text_evaluation_model sorrybench
#CUDA_VISIBLE_DEVICES=6 python main.py --model speechgpt --idstr sota --start 400 --end 500 --evaluate_gpu_id 0 --dataset advbench --jailbreak audio_ours --text_evaluation_model sorrybench
#CUDA_VISIBLE_DEVICES=0,1 python compute_score.py --model speechgpt --start 0 --end 100 --evaluate_gpu_id 1 --dataset advbench --jailbreak audio_ours --text_evaluation_model sorrybench
#CUDA_VISIBLE_DEVICES=2,3 python compute_score.py --model speechgpt --start 100 --end 200 --evaluate_gpu_id 1 --dataset advbench --jailbreak audio_ours --text_evaluation_model sorrybench
#CUDA_VISIBLE_DEVICES=4,5 python compute_score.py --model speechgpt --start 200 --end 300 --evaluate_gpu_id 1 --dataset advbench --jailbreak audio_ours --text_evaluation_model sorrybench
#CUDA_VISIBLE_DEVICES=6,7 python compute_score.py --model speechgpt --start 300 --end 400 --evaluate_gpu_id 1 --dataset advbench --jailbreak audio_ours --text_evaluation_model sorrybench
#CUDA_VISIBLE_DEVICES=8,9 python compute_score.py --model speechgpt --start 400 --end 500 --evaluate_gpu_id 1 --dataset advbench --jailbreak audio_ours --text_evaluation_model sorrybench

# speechgpt universal attack
#CUDA_VISIBLE_DEVICES=1,2 python main.py --model speechgpt --num_universal 1 --idstr sota --start 7 --end 8 --evaluate_gpu_id 1 --dataset advbench --jailbreak audio_ours_universal --text_evaluation_model sorrybench
#CUDA_VISIBLE_DEVICES=3,2 python main.py --model speechgpt --num_universal 1 --idstr sota --start 7 --end 8 --evaluate_gpu_id 1 --dataset advbench --jailbreak audio_ours_universal --text_evaluation_model sorrybench
#CUDA_VISIBLE_DEVICES=4,6 python main.py --model speechgpt --num_universal 1 --idstr sota --start 200 --end 300 --evaluate_gpu_id 1 --dataset advbench --jailbreak audio_ours_universal --text_evaluation_model sorrybench
#CUDA_VISIBLE_DEVICES=7,6 python main.py --model speechgpt --num_universal 1 --idstr sota --start 300 --end 400 --evaluate_gpu_id 1 --dataset advbench --jailbreak audio_ours_universal --text_evaluation_model sorrybench
#CUDA_VISIBLE_DEVICES=8,9 python main.py --model speechgpt --num_universal 1 --idstr sota --start 400 --end 520 --evaluate_gpu_id 1 --dataset advbench --jailbreak audio_ours_universal --text_evaluation_model sorrybench

#CUDA_VISIBLE_DEVICES=6,2 python main.py --model speechgpt --idstr test --start 400 --end 520 --evaluate_gpu_id 1 --dataset advbench --jailbreak audio_ours_universal --text_evaluation_model sorrybench

#CUDA_VISIBLE_DEVICES=2 python main.py --model speechgpt --idstr sota --start 267 --end 300 --evaluate_gpu_id 1 --dataset advbench --jailbreak audio_ours_universal --text_evaluation_model sorrybench
#CUDA_VISIBLE_DEVICES=3 python main.py --model speechgpt --idstr sota --start 300 --end 400 --evaluate_gpu_id 1 --dataset advbench --jailbreak audio_ours_universal --text_evaluation_model sorrybench
#CUDA_VISIBLE_DEVICES=4 python main.py --model speechgpt --idstr sota --start 400 --end 500 --evaluate_gpu_id 1 --dataset advbench --jailbreak audio_ours_universal --text_evaluation_model sorrybench

# qwen attack
#CUDA_VISIBLE_DEVICES=0 python main.py --model qwen2 --idstr maintable --advnoise_control carhorn --start 300 --end 310 --evaluate_gpu_id 1 --dataset advbench --jailbreak audio_ours --text_evaluation_model sorrybench
#CUDA_VISIBLE_DEVICES=1 python main.py --model qwen2 --idstr maintable --advnoise_control carhorn --start 310 --end 320 --evaluate_gpu_id 1 --dataset advbench --jailbreak audio_ours --text_evaluation_model sorrybench
#CUDA_VISIBLE_DEVICES=3 python main.py --model qwen2 --idstr maintable --advnoise_control carhorn --start 320 --end 330 --evaluate_gpu_id 1 --dataset advbench --jailbreak audio_ours --text_evaluation_model sorrybench
#CUDA_VISIBLE_DEVICES=7 python main.py --model qwen2 --idstr maintable --advnoise_control carhorn --start 330 --end 340 --evaluate_gpu_id 1 --dataset advbench --jailbreak audio_ours --text_evaluation_model sorrybench
#CUDA_VISIBLE_DEVICES=5 python main.py --model qwen2 --idstr maintable --advnoise_control carhorn --start 340 --end 350 --evaluate_gpu_id 1 --dataset advbench --jailbreak audio_ours --text_evaluation_model sorrybench
#CUDA_VISIBLE_DEVICES=6 python main.py --model qwen2 --idstr maintable --advnoise_control carhorn --start 250 --end 300 --evaluate_gpu_id 1 --dataset advbench --jailbreak audio_ours --text_evaluation_model sorrybench
#CUDA_VISIBLE_DEVICES=7 python main.py --model qwen2 --idstr maintable --advnoise_control carhorn --start 350 --end 400 --evaluate_gpu_id 1 --dataset advbench --jailbreak audio_ours --text_evaluation_model sorrybench
#CUDA_VISIBLE_DEVICES=8 python main.py --model qwen2 --idstr maintable --advnoise_control carhorn --start 400 --end 450 --evaluate_gpu_id 1 --dataset advbench --jailbreak audio_ours --text_evaluation_model sorrybench
#CUDA_VISIBLE_DEVICES=9 python main.py --model qwen2 --idstr maintable --advnoise_control carhorn --start 450 --end 520 --evaluate_gpu_id 1 --dataset advbench --jailbreak audio_ours --text_evaluation_model sorrybench

#CUDA_VISIBLE_DEVICES=2,3 python main.py --model qwen2 --idstr ablation_sure --advnoise_control carhorn --start 0 --end 260 --evaluate_gpu_id 1 --dataset advbench --jailbreak audio_ours --text_evaluation_model sorrybench
#CUDA_VISIBLE_DEVICES=4,5 python main.py --model qwen2 --idstr ablation_sure --advnoise_control carhorn --start 260 --end 520 --evaluate_gpu_id 1 --dataset advbench --jailbreak audio_ours --text_evaluation_model sorrybench
#CUDA_VISIBLE_DEVICES=6,7 python main.py --model qwen2 --idstr gcg_target --advnoise_control carhorn --start 0 --end 260 --evaluate_gpu_id 1 --dataset advbench --jailbreak audio_ours --text_evaluation_model sorrybench
#CUDA_VISIBLE_DEVICES=8,9 python main.py --model qwen2 --idstr gcg_target --advnoise_control carhorn --start 260 --end 520 --evaluate_gpu_id 1 --dataset advbench --jailbreak audio_ours --text_evaluation_model sorrybench


# omni_speech
#CUDA_VISIBLE_DEVICES=0 python main.py --model omni_speech --idstr maintable --advnoise_control carhorn --start 0 --end 75 --evaluate_gpu_id 1 --dataset advbench --jailbreak audio_ours --text_evaluation_model sorrybench
#CUDA_VISIBLE_DEVICES=1 python main.py --model omni_speech --idstr maintable --advnoise_control carhorn --start 75 --end 150 --evaluate_gpu_id 1 --dataset advbench --jailbreak audio_ours --text_evaluation_model sorrybench
#CUDA_VISIBLE_DEVICES=2 python main.py --model omni_speech --idstr maintable --advnoise_control carhorn --start 150 --end 225 --evaluate_gpu_id 1 --dataset advbench --jailbreak audio_ours --text_evaluation_model sorrybench
#CUDA_VISIBLE_DEVICES=3 python main.py --model omni_speech --idstr maintable --advnoise_control carhorn --start 225 --end 300 --evaluate_gpu_id 1 --dataset advbench --jailbreak audio_ours --text_evaluation_model sorrybench
#CUDA_VISIBLE_DEVICES=4 python main.py --model omni_speech --idstr maintable --advnoise_control carhorn --start 300 --end 375 --evaluate_gpu_id 1 --dataset advbench --jailbreak audio_ours --text_evaluation_model sorrybench
#CUDA_VISIBLE_DEVICES=6 python main.py --model omni_speech --idstr maintable --advnoise_control carhorn --start 375 --end 450 --evaluate_gpu_id 1 --dataset advbench --jailbreak audio_ours --text_evaluation_model sorrybench
#CUDA_VISIBLE_DEVICES=7 python main.py --model omni_speech --idstr maintable --advnoise_control carhorn --start 450 --end 520 --evaluate_gpu_id 1 --dataset advbench --jailbreak audio_ours --text_evaluation_model sorrybench

# blackbox attack against gpt-4o
#CUDA_VISIBLE_DEVICES=7 python main.py --start 100 --end 125 --model gpt4o --dataset advbench --jailbreak blackbox --text_evaluation_model sorrybench


# optimization targets search
#CUDA_VISIBLE_DEVICES=5 python optimization_target_search.py --model qwen2 --dataset advbench --K_modify 10 --num_targets 5
#CUDA_VISIBLE_DEVICES=9 python optimization_target_search.py --model speechgpt --dataset advbench --K_modify 10 --num_targets 5
#CUDA_VISIBLE_DEVICES=7 python optimization_target_search.py --model omni_speech --dataset advbench --K_modify 10 --num_targets 5


# Test
model=speechgpt #qwen2 omni_speech speechgpt
data=speechgptaudio #qwen2_transfer llama_omni_transfer speechgptaudio text_gcg text_beast text_autodan none
judge=llama3

#CUDA_VISIBLE_DEVICES=0,1 python main.py --model $model --start 80 --end 500 --idstr qwne_test --evaluate_gpu_id 1 --dataset advbench --jailbreak speechgptaudio --text_evaluation_model $judge
#CUDA_VISIBLE_DEVICES=2,3 python main.py --model $model --start 0 --end 520 --idstr qwne_test --evaluate_gpu_id 1 --dataset advbench --jailbreak text_autodan --text_evaluation_model $judge
#CUDA_VISIBLE_DEVICES=4,5 python main.py --model $model --start 0 --end 520 --idstr qwne_test --evaluate_gpu_id 1 --dataset advbench --jailbreak text_beast --text_evaluation_model $judge
#CUDA_VISIBLE_DEVICES=6,7 python main.py --model $model --start 0 --end 520 --idstr qwne_test --evaluate_gpu_id 1 --dataset advbench --jailbreak text_gcg --text_evaluation_model $judge
#CUDA_VISIBLE_DEVICES=8,9 python main.py --model $model --start 0 --end 520 --idstr qwne_test --evaluate_gpu_id 1 --dataset advbench --jailbreak none --text_evaluation_model $judge


CUDA_VISIBLE_DEVICES=1 python audio_stealthiness_judge.py