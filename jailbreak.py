from models_audio import models_audio
from models_audio import GenerationConfig
from evaluation_model import evaluation_model
import argparse
import os
from dataload.dataload import get_audio_list
from tqdm import tqdm
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=["qwen2"])
    parser.add_argument('--dataset', type=str, choices=["advbench"])
    parser.add_argument("--jailbreak", type=str, choices=["audio_ours"])
    args = parser.parse_args()

    save_dir = os.path.join("output", args.dataset, args.jailbreak, args.model)
    config = GenerationConfig(temperature=0.8, savedir=save_dir)
    model = models_audio(args.model, config)

    audio_list, text_list = get_audio_list(args.dataset, "none")
    all_score_keyword, all_score_llm_eval, all_score_audio = 0., 0., 0.
    logs = []
    log_path = os.path.join(save_dir,'log.jsonl')
    for id, audio_clip in tqdm(enumerate(audio_list)):
        response_text = model.forward([audio_clip], id)
        print(response_text)
        break