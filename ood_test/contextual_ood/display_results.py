import json 
f = "/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/maha_score_dict.json"



#performance 
with open(f, 'r') as file : 
    data = json.load(file)

    samples = data["vqa_v2_train"]

    for data_split, results in samples.items() : 

        print("Current split: ", data_split) 
        for concept, val in results.items() : 
            print(f"{concept} : {val}")


        print("\n\n")




import matplotlib.pyplot as plt
import pandas as pd
from adjustText import adjust_text
import json 
from scipy.stats import pearsonr

# df = pd.DataFrame(data)
# correlation = df['ShiftMetric'].corr(df['Accuracy'])

# plt.figure(figsize=(10, 6))
# plt.scatter(df['ShiftMetric'], df['Accuracy'])

# texts = []
# for i in range(len(df)):
#     texts.append(plt.text(df['ShiftMetric'][i], df['Accuracy'][i], df['Dataset'][i], fontsize=9, ha='right'))

# # Automatically adjust text positions to minimize overlap
# adjust_text(texts)

# plt.title('Shift Metric vs. Accuracy')
# plt.xlabel('Shift Metric')
# plt.ylabel('Accuracy')

# plt.grid(True)
# plt.show()

# # Save the plot
# plt.savefig('/nethome/bmaneech3/flash/vlm_robustness/result_output/ood_perf_pzy_avg.png')


# print(f"Correlation : {correlation}")
with open("/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/maha_score_dict.json", 'r') as file:
    combined_ood_perf = json.load(file)["vqa_v2_train"]

plot_types = ["pt_joint", "pt_image"]


#pretrained performance
perf_dict = { 
    "vqa_v2_val" : 54.42, 
    "ivvqa_test": 63.95, 
    "cvvqa_test": 44.72, 
    "vqa_rephrasings_test": 50.1, 
    "vqa_ce_test" : 30.68,
    "advqa_test" : 30.46, 
    "textvqa_test": 14.86, 
    "vizwiz_test": 16.84,
    "okvqa_test" : 28.6,
    "vqa_cp_test" : 54.29
}


#Finetuned performance 

# perf_dict = { 
#     "vqa_v2_val" : 90.9, 
#     "ivvqa_test": 98.68, 
#     "cvvqa_test": 70.49, 
#     "vqa_rephrasings_test": 85.55, 
#     "vqa_ce_test" : 79.81,
#     "advqa_test" : 56.86, 
#     "textvqa_test": 44.25, 
#     "vizwiz_test": 24.62,
#     "okvqa_test" : 53.84,
#     "vqa_cp_test" : 90.42
# }

#modify pt and ft key values 





title_dict = {
    "image" : "Vision Shift", 
    "joint" : "V+Q Joint Shift", 
    "vqa" : "V+Q+A Joint Shift", 
    "q_mid" :"Question Mid layer Shift", 
    "q_last" : "Question Last layer shift",
    "pt_joint" : "Pretrain Joint Shift", 
    "pt_image" : "Pretrain Image Shift"

}

for plot_type in plot_types : 
    shift_values = [] 
    perf_values = [] 
    labels = [] 
    texts = [] 
    concepts = [] 
    print("PLOT_TYPE", plot_type)

    for data_split, results in combined_ood_perf.items():

        # if data_split == "vqa_cp_test" or data_split == "vqa_vs_id_val" : 
        if  data_split == "vqa_vs_id_val" or data_split == "vqa_lol_test" or data_split == "vqa_v2_train": 
            continue 


        print(data_split, end = ", ")
        
        perf_value = perf_dict[data_split]
        shift_value = results[plot_type]

        shift_values.append(shift_value)
        perf_values.append(perf_value)
        labels.append(f'{data_split}')
        

    plt.figure(figsize=(10, 6))
    plt.scatter(shift_values, perf_values, alpha=0.7)

    for i, label in enumerate(labels):
        texts.append(
            plt.text(
                shift_values[i], perf_values[i], label,
                fontsize=8, ha='center', va='center'
            )
        )

    adjust_text(texts, arrowprops=dict(arrowstyle="-", color='grey', lw=0.5))

    plt.title(title_dict[plot_type])
    plt.xlabel('Shift Score')
    plt.ylabel('Accuracy')
    plt.grid(True)
    # plt.legend()
    plt.savefig(f"/nethome/bmaneech3/flash/vlm_robustness/result_output/ft_{plot_type}_ood_perf.jpg")


    if len(shift_values) > 1 and len(perf_values) > 1:
        correlation, p_value = pearsonr(shift_values, perf_values)
        print(f"Correlation for {plot_type}: {correlation:.2f} (p-value: {p_value:.4f})")
    else:
        print(f"Not enough data to calculate correlation for {plot_type}")