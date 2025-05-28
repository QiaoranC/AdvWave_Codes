from models_audio import models_audio
from models_audio import GenerationConfig
from evaluation_model import evaluation_model
import argparse
import os
from dataload.dataload import get_audio_list
from tqdm import tqdm
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=["gpt4o", "qwen2", "speechgpt", "anygpt", "omni_speech"])
    parser.add_argument('--dataset', type=str, choices=["advbench"])
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument("--jailbreak", type=str, choices=["none", "text_gcg", "text_beast", "text_autodan", "audio_ours", "speechgptaudio", "blackbox", "audio_ours_universal", "qwen_transfer", "llama_omni_transfer"])
    parser.add_argument("--text_evaluation_model", type=str, choices=["llama3", "sorrybench"])
    parser.add_argument("--audio_evaluation_model", type=str, choices=["gpt4o", "qwen"])
    parser.add_argument("--evaluate_gpu_id", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=520)
    parser.add_argument("--idstr", type=str, default="defaul_setting")
    parser.add_argument("--advnoise_control", type=str, default="")
    parser.add_argument("--num_universal", type=int, default=1)
    args = parser.parse_args()

    save_dir = os.path.join("output", args.dataset, args.jailbreak, args.model, args.idstr)
    config = GenerationConfig(temperature=args.temperature, savedir=save_dir)
    model = models_audio(args.model, config, args)

    evaluation_model = evaluation_model(args.text_evaluation_model, args.audio_evaluation_model, args.evaluate_gpu_id)

    audio_list, text_list = get_audio_list(args.dataset, args.jailbreak)
    if args.jailbreak == "audio_ours" or args.jailbreak == "audio_ours_universal":
        audio_list_jailbreak_initialization, _ = get_audio_list(args.dataset, "text_gcg")
    all_score_keyword, all_score_llm_eval, all_score_audio = 0., 0., 0.
    logs = []
    log_path = os.path.join(save_dir,f'log_{args.start}_{args.end}.jsonl')

    valid_ind = list(range(args.start, args.end))

    record_jailbreaked_index = [] # used for universal attack
    num_universal = args.num_universal

    if args.jailbreak=="audio_ours":
        adaptive_targets =[]
        with open(f'/home/XXX/agent/AudioAttack/output/{args.dataset}/optimization_target_search/{args.model}/optimization_targets.jsonl', 'r', encoding='utf-8') as file:
            for line in file:
                adaptive_targets.append(json.loads(line.strip()))

    for id, audio_clip in tqdm(enumerate(audio_list)):
        if id not in valid_ind:
            continue



        # for universal_attack
        if id in record_jailbreaked_index:
            continue
        if "universal" in args.jailbreak:
            for jj in range(id, id + num_universal):
                record_jailbreaked_index.append(jj)
            audio_clip = []
            texts_ori = []
            for jj in range(id, id+num_universal):
                audio_clip.append(audio_list[jj])
                texts_ori.append(text_list[jj])

        if not isinstance(audio_clip, list):
            audio_clip = [audio_clip]
        if args.jailbreak == "audio_ours":
            response_text, record = model.forward(audio_clip, id, text_list[id], audio_list_jailbreak_initialization[id], adaptive_targets[id]["optimization_targets"])
            res = response_text
        elif args.jailbreak == "audio_ours_universal":
            response_texts, records = model.forward(audio_clip, [jj for jj in range(id, id+num_universal)], texts_ori, audio_list_jailbreak_initialization[id])
        else:
            record = "no record"
            response_text, _ = model.forward(audio_clip, id, text_list[id])
            res = response_text
        if "universal" not in args.jailbreak:
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
        else:
            if args.model == "speechgpt":
                ori_results = []
                responses = []
                for res in response_texts:
                    if '[ta]' in res and '[ua]' in res:
                        str_ = extract_text_between_tags(res, '[ta]', '[ua]')
                    elif '[ta]' in res:
                        str_ = extract_text_between_tags(res + '<eoa>', '[ta]', '<eoa>')
                    else:
                        str_ = ""
                    ori_results.append(res)
                    responses.append(str_)
            for jj in range(id, id + num_universal):
                print(f'Input for id {jj}:')
                print(text_list[jj])
                print(f'Response:')
                print(responses[jj-id])
                print(f'---------')

        if "universal" not in args.jailbreak:
            score_keyword, score_llm_eval, llm_eval_response = evaluation_model.text_evaluate(text_list[id], response_text)

            # score_audio_eval = evaluation_model.audio_evaluate(text_list[id], response_text)

            # score_keyword = 0.0
            # score_llm_eval = 0.0
            # llm_eval_response = ""

            score_audio_eval = 0.0
            all_score_keyword += score_keyword
            all_score_llm_eval += score_llm_eval
            all_score_audio += score_audio_eval
            print(f'Avg score keyword: {all_score_keyword / (id+1-args.start)}')
            print(f'Avg score llm eval: {all_score_llm_eval / (id+1-args.start)}')
            print(f'Avg score audio eval: {all_score_audio / (id+1-args.start)}')
            logs.append({"prompt": text_list[id], "prompt_all": res ,"response_text": response_text ,"score_keyword": score_keyword, "score_llm_eval": score_llm_eval, "score_audio_eval": score_audio_eval, "llm_eval_response": llm_eval_response, "record": record})
            with open(log_path, 'w') as jsonl_file:
                for item in logs:
                    jsonl_file.write(json.dumps(item) + '\n')
        else:
            for jj in range(id, id + num_universal):
                score_keyword, score_llm_eval, llm_eval_response = evaluation_model.text_evaluate(text_list[jj], responses[jj-id])
                score_audio_eval = 0.0
                all_score_keyword += score_keyword
                all_score_llm_eval += score_llm_eval
                all_score_audio += score_audio_eval
                logs.append({"prompt": text_list[jj], "prompt_all": ori_results[jj-id], "response_text": responses[jj-id],
                             "score_keyword": score_keyword, "score_llm_eval": score_llm_eval,
                             "score_audio_eval": score_audio_eval, "llm_eval_response": llm_eval_response,})
            print(f'Avg score keyword: {all_score_keyword / (id + num_universal - args.start)}')
            print(f'Avg score llm eval: {all_score_llm_eval / (id + num_universal - args.start)}')
            print(f'Avg score audio eval: {all_score_audio / (id + num_universal - args.start)}')

            with open(log_path, 'w') as jsonl_file:
                for item in logs:
                    jsonl_file.write(json.dumps(item) + '\n')
    all_score_keyword /= (args.end - args.start)
    all_score_llm_eval /= (args.end - args.start)
    all_score_audio /= (args.end - args.start)
    print(f'Dataset: {args.dataset}, Model: {args.model}, jailbreak: {args.jailbreak}')
    print(f'Avg score keyword: {all_score_keyword}')
    print(f'Avg score llm eval: {all_score_llm_eval}')
    print(f'Avg score audio eval: {all_score_audio}')