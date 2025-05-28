from io import BytesIO
from urllib.request import urlopen
import librosa
import numpy as np
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import torch
import numpy
import torch.nn as nn
from tqdm import tqdm
import torchaudio

from dataload.text2audio import prompt2audio


def qwen_eval_gen(audio_list, processor, model):
    if len(audio_list) > 1:
        raise ValueError("Only consider single audio clip here!")
    audio_url = "file:" + audio_list[0]
    conversation = [
        {"role": "user", "content": [
            {"type": "text", "text": "Listen to Audio 1 and judge whether Audio 1 contains car horn?"},
            {"type": "audio", "audio_url": audio_url},
            {"type": "text", "text": "Does Audio 1 contain car horn?"},
        ]},
    ]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

    audios = []
    for message in conversation:
        if isinstance(message["content"], list):
            for ele in message["content"]:
                if ele["type"] == "audio":
                    audios.append(librosa.load(
                        BytesIO(urlopen(ele['audio_url']).read()),
                        sr=processor.feature_extractor.sampling_rate)[0]
                                  )

    for idx in range(len(audios)):
        audios[idx] = torch.tensor(audios[idx], device="cuda")
    audios = torch.cat(audios).unsqueeze_(0)

    # Inference
    inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True, sampling_rate=16000)
    inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    generate_ids = model.generate(**inputs, max_length=1024, do_sample=False, temperature=0.0, top_p=0, top_k=0)
    generate_ids = generate_ids[:, inputs['input_ids'].size(1):]
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return response

if __name__=="__main__":
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct",
                                                         # cache_dir="/data/common/XXX/cache"
                                                         )
    model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", device_map="cuda:0",
                                                                    # cache_dir="/data/common/XXX/cache"
                                                                    )

    path = "./cache/qwen_judge/other.wav"
    # prompt2audio("generate audio of air conditioner noise", path)
    response = qwen_eval_gen([path], processor, model)

    print(response)