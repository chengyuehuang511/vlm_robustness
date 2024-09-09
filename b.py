import json
with open('/nethome/chuang475/flash/projects/vlm_robustness/tmp/datasets/textvqa/val/TextVQA_0.5.1_val.json', 'r') as f:
    input_data = json.load(f)['data']

# Desired format
output_data = {
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
    "annotations": []
}

for item in input_data:
    # the most frequent answer in the list - item["answers"] is the multiple_choice_answer
    multiple_choice_answer = max(item["answers"], key=item["answers"].count)
    answers = []
    for i, answer in enumerate(item["answers"]):
        answers.append({"answer": answer, "answer_confidence": "yes", "answer_id": i+1})
    
    annotation = {
        "question_type": "unknown",
        "multiple_choice_answer": multiple_choice_answer,
        "answers": answers,
        "image_id": item["image_id"],
        "answer_type": "other",
        "question_id": item["question_id"]
    }
    output_data["annotations"].append(annotation)

with open('/nethome/chuang475/flash/projects/vlm_robustness/tmp/datasets/textvqa/val/annotation.json', 'w') as f:
    json.dump(output_data, f, indent=4)

print("Combined data saved to 'output_data.json'.")