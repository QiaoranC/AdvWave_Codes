import json
from tqdm import tqdm
import numpy as np
import os
from .text2audio import prompt2audio
def get_audio_list(dataset, jailbreak, voice_id=0, speed=1):

    if dataset == 'advbench':
        if jailbreak == "text_gcg" or jailbreak == "none" or jailbreak == "audio_ours" or jailbreak == "speechgptaudio" or jailbreak == "blackbox" or jailbreak == "audio_ours_universal":
            input_file = '/home/XXX/agent/AudioAttack/data/text_jailbreak/gcg_advbench_behaviour.jsonl'
        elif jailbreak == "text_autodan":
            input_file = '/home/XXX/agent/AudioAttack/data/text_jailbreak/autodan_advbench_behaviour.jsonl'
        elif jailbreak == "text_beast":
            input_file = '/home/XXX/agent/AudioAttack/data/text_jailbreak/beast_advbench_behaviour.jsonl'
        else:
            input_file = '/home/XXX/agent/AudioAttack/data/text_jailbreak/gcg_advbench_behaviour.jsonl'

    text_data = []
    if not os.path.exists(input_file):
        input_file = input_file.replace("XXX", "ec2-user")
    with open(input_file, 'r') as infile:
        for line in infile:
            json_object = json.loads(line)
            text_data.append(json_object)

    audio_dir = f'/home/XXX/agent/AudioAttack/data/audio/{dataset}/{jailbreak}/{voice_id}/{speed}'
    if "ec2-user" in input_file:
        audio_dir = audio_dir.replace("XXX", "ec2-user")
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)

    audio_list = []
    text_list = []
    for id, instance in tqdm(enumerate(text_data)):
        text_list.append(instance["message"])
        if jailbreak == "text_gcg":
            jailbreak_string = instance["result"]["best_string"]
            jailbreak_string = instance["message"] + jailbreak_string
        elif jailbreak == "text_autodan":
            best_index = np.argsort(np.array(instance["result"]["loss_stored"]))[0]
            jailbreak_string = instance["result"]["optim_strings"][best_index]
            jailbreak_string = jailbreak_string.replace("[REPLACE]", instance["message"])
        elif jailbreak == "text_beast":
            best_index = np.argsort(np.array(instance["result"]["scores_stored"]))[-1]
            jailbreak_string = instance["result"]["optim_strings"][best_index]
            jailbreak_string = instance["message"] + jailbreak_string
        elif jailbreak == "none" or jailbreak == "audio_ours" or jailbreak == "blackbox" or jailbreak == "audio_ours_universal":
            jailbreak_string = instance["message"]

        if jailbreak == "speechgptaudio":
            audio_ori = os.path.join(f'/home/XXX/agent/AudioAttack/data/audio/{dataset}/none/{voice_id}/{speed}', f"{id}.wav")
            audio_suffix = os.path.join(f'/home/XXX/agent/AudioAttack/output/{dataset}/audio_ours/speechgpt/0.8', f"jailbreak_audio_{id}.wav")
            audio_list.append([[audio_ori, audio_suffix]])
            # audio_list.append([audio_ori])
        elif jailbreak == "qwen_transfer":
            audio_jailbreak = os.path.join(f'/home/XXX/agent/AudioAttack/output/{dataset}/audio_ours/qwen2/maintable',f"{id}.wav")
            audio_list.append([audio_jailbreak])
        elif jailbreak == "llama_omni_transfer":
            audio_jailbreak = os.path.join(f'/home/XXX/agent/AudioAttack/output/{dataset}/audio_ours/omni_speech/maintable',f"{id}.wav")
            audio_list.append([audio_jailbreak])
        else:
            audio_file = os.path.join(f'/home/XXX/agent/AudioAttack/data/audio/{dataset}/{jailbreak}/{voice_id}/{speed}', f"{id}.wav")
            if "ec2-user" in input_file:
                audio_file = audio_file.replace("XXX", "ec2-user")
            if not os.path.exists(audio_file):
                prompt2audio(jailbreak_string, audio_file)
            audio_list.append(audio_file)

    return audio_list, text_list