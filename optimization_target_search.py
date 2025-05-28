import os
from models_audio import models_audio
from models_audio import GenerationConfig
from dataload.text2audio import prompt2audio
from collect_benign.gpt import modify_unsafe_prompt, summarize_pattern
from tqdm import tqdm
import argparse
from dataload.dataload import get_audio_list
import json
import re

def extract_text_between_tags(text, tag1='[SpeechGPT] :', tag2='<eoa>'):
    pattern = f'{re.escape(tag1)}(.*?){re.escape(tag2)}'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        response = match.group(1)
    else:
        response = ""
    return response
def target_search(input_prompt: str, model, K: int = 3, cache_dir='./cache', K_modify=10, args=None):
    # TODO: search K optimization targets for model given the input prompt and input audio file
    # potential way: transform the input prompt with multiple benign prompts and extract common ways of answering
    # how to extract the common ways can be a challenge (can first leverage LLM and see results?)

    # print('Modifying input unsafe prompt...')
    modified_safe_prompts = modify_unsafe_prompt(input_prompt, sampling_number=K_modify)
    # print('*****' * 20)
    # print('Modified safe prompts:')
    # for i, modified_safe_prompt in enumerate(modified_safe_prompts):
    #     print(modified_safe_prompt)

    # Some useful function below
    # text-to-speech
    safe_audio_cache_dir = os.path.join(cache_dir, 'audio_cache')
    os.makedirs(safe_audio_cache_dir, exist_ok=True)
    prompt_audio_files = [os.path.join(safe_audio_cache_dir, input_prompt, f"{i:02}.wav") for i in range(len(modified_safe_prompts))]
    # print('Generating audio files for modified prompts...')
    os.makedirs(os.path.join(safe_audio_cache_dir, input_prompt), exist_ok=True)
    for i, modified_safe_prompt in enumerate(modified_safe_prompts):
        prompt2audio(modified_safe_prompt, prompt_audio_files[i])

    # audio model forward
    responses = []
    # print('Generating audio model responses...')
    for i, modified_safe_prompt in enumerate(modified_safe_prompts):
        response = model.forward([prompt_audio_files[i]], 0, modified_safe_prompt)
        if isinstance(response, tuple):
            response = response[0]
        if args.model == "speechgpt":
            if '[ta]' in response and '[ua]' in response:
                response = extract_text_between_tags(response, '[ta]', '[ua]')
            elif '[ta]' in response:
                response = extract_text_between_tags(response + '<eoa>', '[ta]', '<eoa>')
            else:
                response = ""
        responses.append(response)
    # print('*****' * 20)
    # print('Model responses:')
    # for i, response in enumerate(responses):
    #     print(response)
    #     print('-----' * 20)

    # print('Summarizing patterns...')
    optimization_targets = summarize_pattern(modified_safe_prompts, responses, input_prompt, sampling_number=K)
    # print('*****' * 20)
    # print('Possible optimization targets:')
    # for i, optimization_target in enumerate(optimization_targets):
    #     print(optimization_target)

    return optimization_targets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=["gpt4o", "qwen2", "speechgpt", "omni_speech"])
    parser.add_argument('--dataset', type=str, choices=["advbench"])
    parser.add_argument('--K_modify', type=int, default=30)
    parser.add_argument('--num_targets', type=int, default=10)
    parser.add_argument('--jailbreak', type=str, default="none")
    parser.add_argument('--advnoise_control', type=str, default="")
    args = parser.parse_args()

    save_dir = f"./output/{args.dataset}/optimization_target_search/{args.model}"
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "optimization_targets.jsonl")
    config = GenerationConfig(temperature=0.8, savedir=save_dir)

    audio_list, text_list = get_audio_list(args.dataset, "none")
    model = models_audio(args.model, config, args)

    results = []
    for unsafe_prompt in tqdm(text_list):
        optimization_targets = target_search(unsafe_prompt, model, K=args.num_targets, K_modify=args.K_modify, args=args)
        record = {"prompt": unsafe_prompt, "optimization_targets": optimization_targets}
        results.append(record)
        with open(log_path, 'w') as jsonl_file:
            for item in results:
                jsonl_file.write(json.dumps(item) + '\n')

