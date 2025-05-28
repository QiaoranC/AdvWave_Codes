from models_audio import models_audio
from models_audio import GenerationConfig
from evaluation_model import evaluation_model
import argparse
import os
from dataload.dataload import get_audio_list
from tqdm import tqdm
import json
import re
import torch
import torch.nn as nn
import soundfile as sf
import torchaudio

def extract_text_between_tags(text, tag1='[SpeechGPT] :', tag2='<eoa>'):
    pattern = f'{re.escape(tag1)}(.*?){re.escape(tag2)}'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        response = match.group(1)
    else:
        response = ""
    return response

def load_jsonl_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data


def get_list(s2u, path):
    waveform = s2u.feature_reader.read_audio(path)
    feat = s2u.feature_reader.get_feats(waveform)
    cluster_ids = s2u.apply_kmeans(feat).tolist()
    dup_cluster_list, dutiraon_list = s2u.merge_duplicates(cluster_ids)
    return waveform, dup_cluster_list, dutiraon_list


def construct_target(unit, s2u):
    targets = []
    C = s2u.apply_kmeans.C
    for i, ind in enumerate(unit):
        for _ in range(1):
            targets.append(C[:,ind])
    targets = torch.stack(targets)
    return targets, C

import numpy as np


def find_closest_cluster_center(feature, cluster_centers):
    distances = torch.norm(cluster_centers - feature, dim=1)
    closest_cluster_index = torch.argmin(distances)
    return closest_cluster_index.item()


def compute_negative(feat, C, target_ids):
    negatives = []
    for fea, target_id in zip(feat, target_ids):
        id = find_closest_cluster_center(fea, C)
        if id != target_id:
            negatives.append(C[id])
        else:
            negatives.append(C[0])
    negatives = torch.stack(negatives)
    # print('negatives.shape')
    # print(negatives.shape)
    return negatives

def audio_waveform_optimization(targets, s2u, C, targe_ids):
    mse_loss = nn.MSELoss()
    triplet_loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)
    wav = torch.zeros([1,16300], requires_grad=True, device='cuda')
    pbar = tqdm(range(10000))
    optimizer = torch.optim.Adam([wav], lr=1e-3)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)
    C = torch.tensor(C.T).cuda()
    for i in pbar:
        optimizer.zero_grad()
        feat = s2u.feature_reader.get_feats(wav)
        negative_anchors = compute_negative(feat, C, targe_ids)
        # loss = mse_loss(feat, targets)
        loss =  triplet_loss_fn(feat, targets, negative_anchors)
        grad = torch.autograd.grad(outputs=[loss], inputs=[wav])[0]
        wav.grad = grad
        optimizer.step()
        # scheduler.step()
        pbar.set_description(f'loss: {loss}')
    return wav



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=["gpt4o", "qwen2", "speechgpt", "anygpt"])
    parser.add_argument('--dataset', type=str, choices=["advbench"])
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument("--jailbreak", type=str, choices=["none", "text_gcg", "text_beast", "text_autodan", "audio_ours"])
    parser.add_argument("--text_evaluation_model", type=str, choices=["llama3"])
    parser.add_argument("--audio_evaluation_model", type=str, choices=["gpt4o", "qwen"])
    parser.add_argument("--evaluate_gpu_id", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=520)
    args = parser.parse_args()

    save_dir = os.path.join("output", args.dataset, args.jailbreak, args.model, f'{args.temperature}')
    config = GenerationConfig(temperature=args.temperature, savedir=save_dir)
    model = models_audio(args.model, config, args)
    evaluation_model = evaluation_model(args.text_evaluation_model, args.audio_evaluation_model, args.evaluate_gpu_id)

    audio_list, text_list = get_audio_list(args.dataset, args.jailbreak)
    if args.jailbreak == "audio_ours":
        audio_list_jailbreak_initialization, _ = get_audio_list(args.dataset, "text_gcg")
    all_score_keyword, all_score_llm_eval, all_score_audio = 0., 0., 0.
    logs = []
    log_path = os.path.join(save_dir,f'log_{args.start}_{args.end}.jsonl')

    data = load_jsonl_file(log_path)
    cnt = 0

    valid_ind = list(range(args.start, args.end))
    for id, audio_clip in tqdm(enumerate(audio_list)):
        if id not in valid_ind:
            continue
        instance = data[cnt]
        cnt += 1
        response_text = instance["prompt_all"]

        jailbreaked_tokens = extract_text_between_tags(response_text, "<sosp>","<eosp> <eoh>")

        unit = [int(num) for num in re.findall(r'<(\d+)>', jailbreaked_tokens)][-50:]
        # print('ground truth')
        # print(unit)
        x = {
            "code": torch.LongTensor(unit).view(1, -1).to(model.model.model.device),
        }
        wav = model.model.vocoder(x, True)
        jailbreak_audio_path = os.path.join(save_dir,f'jailbreak_audio_{id}.wav')
        model.model.save_wav(wav, jailbreak_audio_path)


        encoded_tokens = model.model.s2u(jailbreak_audio_path.strip(), merged=True)
        # waveform, dup_cluster_list, duration_list = get_list(model.model.s2u, jailbreak_audio_path.strip())
        targets, C = construct_target(unit, model.model.s2u)
        #
        waveform = audio_waveform_optimization(targets, model.model.s2u, C, unit).detach().cpu()
        torchaudio.save(jailbreak_audio_path, waveform, 16000)

        encoded_tokens_ori = model.model.s2u(audio_clip, merged=True)
        unit_encoded_ori = [int(num) for num in re.findall(r'<(\d+)>', encoded_tokens_ori)]
        encoded_tokens = model.model.s2u(jailbreak_audio_path.strip(), merged=True)
        unit_encoded = [int(num) for num in re.findall(r'<(\d+)>', encoded_tokens)]
        # print('reencoded')
        # print(unit_encoded)

        #
        # fea = model.model.s2u.feature_reader.get_feats(waveform)
        # cluster_ids = model.model.s2u.apply_kmeans(fea).tolist()
        # dup_cluster_list, duration_list = model.model.s2u.merge_duplicates(cluster_ids)


        # a,b = model.model.s2u.merge_duplicates(model.model.s2u.apply_kmeans(targets).tolist())
        # print(a)
        response_text = model.model.final_forward([[audio_clip,jailbreak_audio_path]], audio_save_path=os.path.join(model.savedir, f"{id}.wav"), prompt_ori = instance["prompt"])

        # print('response text')
        # print(response_text)

        if args.model == "speechgpt":
            res = response_text
            if '[ta]' in res and '[ua]' in res:
                str_ = extract_text_between_tags(res, '[ta]', '[ua]')
            elif '[ta]' in res:
                str_ = extract_text_between_tags(res + '<eoa>', '[ta]', '<eoa>')
            else:
                str_ = ""
            response_text = str_
        print(f'Input:')
        print(text_list[id])
        print(f'Response:')
        print(response_text)
        print(f'---------')

        score_keyword, score_llm_eval = evaluation_model.text_evaluate(text_list[id], response_text)
        # score_audio_eval = evaluation_model.audio_evaluate(text_list[id], response_text)
        # score_keyword = 0.0
        # score_llm_eval = 0.0
        score_audio_eval = 0.0
        all_score_keyword += score_keyword
        all_score_llm_eval += score_llm_eval
        all_score_audio += score_audio_eval
        print(f'Avg score keyword: {all_score_keyword / (id+1-args.start)}')
        print(f'Avg score llm eval: {all_score_llm_eval / (id+1-args.start)}')
        print(f'Avg score audio eval: {all_score_audio / (id+1-args.start)}')
    all_score_keyword /= (args.end - args.start)
    all_score_llm_eval /= (args.end - args.start)
    all_score_audio /= (args.end - args.start)
    print(f'Dataset: {args.dataset}, Model: {args.model}, jailbreak: {args.jailbreak}')
    print(f'Avg score keyword: {all_score_keyword}')
    print(f'Avg score llm eval: {all_score_llm_eval}')
    print(f'Avg score audio eval: {all_score_audio}')