import json 
import os 
f = "/nethome/bmaneech3/flash/LAVIS/lavis/output/PALIGEMMA/VQAVS/ZS/20240716130/result/test_vqa_result.json"

    # print(type(data))

file_list = [
    "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KO/test_questions.json", 
    "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KOP/test_questions.json",
    "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KW/test_questions.json", 
    "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KW+KO/test_questions.json", 
    "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KWP/test_questions.json", 
    "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT/test_questions.json", 
    "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT+KO/test_questions.json",
    "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT+KW/test_questions.json",
    "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT+KW+KO/test_questions.json"
] 

for file in file_list : 

    with open(f, 'r') as k : 
        result_list = json.load(k)


    with open(file, 'r') as k: 
        data = json.load(k)

    data = data["questions"]

    question_id = [d["question_id"] for d in data]


    modif_result = [d for d in result_list if d["question_id"] in question_id]

    path = os.path.dirname(file)
    path = os.path.join(path, "test_result.json")
    with open(path, 'w') as k : 
        json.dump(modif_result, k)

