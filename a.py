import json
with open('/nethome/chuang475/flash/projects/vlm_robustness/tmp/datasets/textvqa/val/TextVQA_0.5.1_val.json', 'r') as f:
    input_data = json.load(f)['data']

# Desired format
output = {
    "info": {
        "description": "This is TextVQA dataset.",
        "url": "https://textvqa.org/dataset/",
        "version": "0.5.1",
        "year": 2019,
        "contributor": "VQA Team",
        "date_created": "2017-04-26 17:00:44"
    },
    "task_type": "Open-Ended",
    "data_type": "open_images",
    "license": {
        "url": "http://creativecommons.org/licenses/by/4.0/",
        "name": "Creative Commons Attribution 4.0 International License"
    },
    "data_subtype": "val",
    "questions": []
}

for item in input_data:
    question = {
        "question": item["question"],
        "image_id": item["image_id"],
        "question_id": item["question_id"]
    }
    output["questions"].append(question)

# Write to a JSON file
with open('/nethome/chuang475/flash/projects/vlm_robustness/tmp/datasets/textvqa/val/question.json', 'w') as json_file:
    json.dump(output, json_file, indent=4)

print("Data has been converted and saved to 'formatted_vqa_data.json'")



