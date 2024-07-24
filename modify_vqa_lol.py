import json 
from tqdm import tqdm 
f = "flash/vlm_robustness/tmp/datasets/vqalol/train/train_vqa_lol_2.json"
#list 

combined_sample = [] 
#question_id', 'question', 'answer', 'image', 'dataset

with open(f, 'r') as file : 
    data = json.load(file)



    for i in tqdm(range(len(data)), desc="build annotations") : 
        #question id = i 
        #question 
        #answer 
        #image 
        #dataset 
        answer= [] 
        split = data[i]["img_id"].split("_")[1]
        image = f"{split}/{data[i]['img_id']}.jpg"
        for ans, freq in data[i]["label"].items() : 
            answer.extend([ans] * freq)


        comb_sample = {
            "question_id" : i,
            "question" : data[i]["sent"], 
            "answer": answer, 
            "image" : image,
            "dataset": "vqalol"
        } 


        combined_sample.append(comb_sample)

        
with open("/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqalol/train/combined_data.json", 'w') as file : 
    json.dump(combined_sample, file)    

print("length", len(combined_sample))



