import json

file_path = "./output/advbench/speechgptaudio/gpt4o/0.8/log_0_520.jsonl"

with open(file_path, 'r') as file:
    json_list = [json.loads(line.strip()) for line in file]

score_keyword, score_llm_eval = 0., 0.
for json_obj in json_list:
    score_keyword += json_obj["score_keyword"]
    score_llm_eval += json_obj["score_llm_eval"]
    print("Input:")
    print(json_obj["prompt"])
    print("Output:")
    print(json_obj["response_text"])

print(score_keyword / len(json_list))
print(score_llm_eval / len(json_list))