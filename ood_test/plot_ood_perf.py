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
with open('/nethome/bmaneech3/flash/vlm_robustness/result_output/comb_shift_perf_dict.json', 'r') as file:
    combined_ood_perf = json.load(file)

plot_types = ["min", "max", "avg", "KO","KOP","KW","KW+KO","KWP","QT","QT+KO","QT+KW","QT+KW+KO"]

for plot_type in plot_types : 
    shift_values = [] 
    perf_values = [] 
    labels = [] 
    texts = [] 
    concepts = [] 
    print("PLOT_TYPE", plot_type)

    for k, data in combined_ood_perf.items():
        train_split, test_split = k.split(",") 
        opt_concept = None 
        
        if isinstance(data[plot_type],list) : 
            shift_value,opt_concept = data.get(plot_type, 0)[0], data.get(plot_type, 0)[1]
        else : 
            shift_value = data.get(plot_type, 0)

        perf_value = data['perf'] 
        if shift_value != 0 and perf_value != 0: 
            
            # print(plot_type)
            # print("shift_values ", shift_values)
            
            shift_values.append(shift_value)
            perf_values.append(perf_value)
            labels.append(f'{train_split}-{test_split}')
            if opt_concept != None : 
                concepts.append(f"{train_split}, {test_split} : {opt_concept}")



    if plot_type in ['max', 'min','avg'] : 
     #add display outside graph showing the concept for each train split that causes either max, min, avg
     #should be in format (train, test split) : concept 
        for i in concepts : 
            print(i)


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

    plt.title(f'Shift vs Performance for : {plot_type}')
    plt.xlabel('Shift Value')
    plt.ylabel('Performance Value')
    plt.grid(True)
    # plt.legend()
    plt.savefig(f"/nethome/bmaneech3/flash/vlm_robustness/result_output/{plot_type}_ood_perf.jpg")


    if len(shift_values) > 1 and len(perf_values) > 1:
        correlation, p_value = pearsonr(shift_values, perf_values)
        print(f"Correlation for {plot_type}: {correlation:.2f} (p-value: {p_value:.4f})")
    else:
        print(f"Not enough data to calculate correlation for {plot_type}")

    
