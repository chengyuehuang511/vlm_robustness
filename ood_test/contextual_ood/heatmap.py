import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Reconstructing the data from the latest image to ensure completeness and accuracy.
# This is based on a careful manual check to match the exact rows and columns.

data_complete_23_rows = [
    
    # [26.21, 26.02, 27.21, 26.45, 26.06, 26.54, 26.15, 26.40, 29.67, 32.76, 26.83, 29.75, 27.52],
    # [25.53, 24.64, 25.91, 25.79, 25.40, 26.03, 25.56, 25.56, 31.97, 35.31, 26.19, 31.16, 27.42],
    # [26.21, 25.39, 26.46, 26.43, 26.06, 26.59, 26.15, 26.18, 31.36, 33.25, 26.73, 30.45, 27.61],
    # [26.21, 26.02, 27.21, 26.45, 26.06, 26.54, 26.15, 26.40, 29.67, 32.76, 26.83, 29.75, 27.52],
    # [26.40, 25.55, 26.56, 26.53, 26.19, 26.72, 26.23, 26.30, 31.05, 32.90, 26.83, 30.26, 27.62],
    # [25.25, 24.47, 25.50, 25.52, 25.13, 25.76, 25.24, 25.27, 31.64, 34.34, 25.91, 30.63, 27.06],
    # [26.06, 25.25, 26.27, 26.28, 25.95, 26.44, 26.01, 26.03, 30.95, 32.69, 26.58, 30.07, 27.38],

    [27.66, 26.55, 27.80, 28.09, 27.37, 28.17, 28.03, 27.67, 34.81, 35.88, 28.84, 33.18, 29.50],
    [38.12, 37.19, 37.72, 38.20, 37.94, 38.41, 38.04, 37.92, 43.06, 42.49, 38.60, 41.38, 39.07],
    # [35.80, 35.02, 35.32, 36.29, 35.62, 37.31, 36.12, 35.95, 48.49, 51.86, 37.23, 45.86, 39.25],
    [27.84, 27.17, 28.67, 27.94, 27.92, 27.91, 27.66, 27.88, 28.93, 32.92, 27.98, 29.94, 28.57],
    [38.11, 37.18, 37.72, 38.20, 37.94, 38.41, 37.96, 37.90, 43.01, 42.50, 38.60, 41.37, 39.06],
    [31.25, 31.01, 30.71, 31.19, 31.19, 31.57, 30.89, 31.09, 33.38, 35.19, 31.48, 33.35, 31.85],
    [35.65, 33.80, 36.24, 35.84, 35.51, 35.99, 35.38, 35.46, 38.14, 41.22, 36.62, 38.66, 36.53],
    [33.36, 32.48, 35.39, 33.37, 33.30, 33.29, 33.20, 33.51, 32.23, 36.98, 33.64, 34.28, 33.77],

    [27.07, 27.02, 21.55, 31.78, 27.17, 26.46, 29.36, 27.22, 34.07, 36.51, 33.12, 34.57, 29.67],
    [36.94, 36.36, 29.26, 38.92, 38.04, 38.33, 38.63, 36.59, 47.97, 42.39, 44.86, 45.08, 39.42],
    # [37.44, 35.42, 25.39, 40.16, 36.53, 38.88, 41.14, 36.25, 52.94, 50.95, 48.75, 50.88, 41.13],
    [38.70, 37.08, 28.70, 40.84, 38.84, 40.19, 40.68, 37.72, 47.10, 46.25, 48.20, 47.18, 40.85],
    [37.25, 35.83, 28.31, 39.32, 38.20, 39.32, 37.38, 36.39, 47.45, 41.76, 44.81, 43.77, 38.85],
    [39.41, 39.09, 30.01, 42.69, 39.58, 40.32, 41.70, 38.73, 48.65, 41.07, 47.48, 46.82, 41.43],
    [36.01, 33.81, 24.74, 38.00, 35.65, 37.78, 37.96, 34.66, 47.46, 45.19, 46.25, 46.30, 40.31],
    [38.38, 36.53, 27.61, 40.49, 38.46, 40.01, 40.19, 37.22, 47.36, 46.39, 48.24, 47.33, 40.59],

    [39.76, 38.79, 36.72, 39.94, 39.65, 41.31, 39.68, 39.35, 50.38, 48.53, 42.40, 47.10, 41.93],
    # [33.96, 32.02, 28.80, 34.88, 33.73, 37.17, 36.79, 33.90, 52.89, 49.07, 40.72, 47.56, 38.48],
    [34.81, 32.91, 29.55, 35.26, 34.81, 37.10, 34.97, 34.10, 45.72, 45.69, 40.91, 44.11, 37.44],
    [37.05, 35.14, 31.67, 38.38, 37.09, 39.05, 37.40, 36.46, 50.59, 48.89, 43.02, 47.50, 40.71],
    [37.62, 36.38, 32.87, 39.44, 37.51, 39.40, 39.04, 37.44, 50.07, 48.59, 43.12, 47.26, 40.71],
    [35.03, 32.38, 28.29, 35.51, 34.94, 38.24, 35.71, 34.18, 49.97, 48.39, 42.19, 46.85, 38.40],
    [34.58, 32.82, 29.12, 35.02, 34.59, 36.97, 34.89, 33.90, 45.72, 45.39, 40.86, 43.99, 37.27],
]

# Convert the data to a DataFrame
df_complete_23_rows = pd.DataFrame(data_complete_23_rows)

# Adding the row labels as specified in the table for clarity
# row_labels = [
#     r"$f(v)$", r"$f_{pt}(v)$", r"$f_{fft}(v)$", r"$f_{vanilla\_ft}(v)$", r"$f_{linear\_prob}(v)$", r"$f_{lp\_ft}(v)$", r"$f_{tp}(v)$", r"$f_{spd}(v)$",
#     r"$f(q)$", r"$f_{pt}(q)$", r"$f_{fft}(q)$", r"$f_{vanilla\_ft}(q)$", r"$f_{linear\_prob}(q)$", r"$f_{lp\_ft}(q)$", r"$f_{tp}(q)$", r"$f_{spd}(q)$",
#     r"$f_{pt}(v, q)$", r"$f_{fft}(v, q)$", r"$f_{vanilla\_ft}(v, q)$", r"$f_{linear\_prob}(v, q)$", r"$f_{lp\_ft}(v, q)$", 
#     r"$f_{ftp}(v, q)$", r"$f_{spd}(v, q)$"
# ]
row_labels = [
    r"$f(v)$", r"$f_{pt}(v)$", r"$f_{vanilla\_ft}(v)$", r"$f_{linear\_prob}(v)$", r"$f_{lp\_ft}(v)$", r"$f_{ftp}(v)$", r"$f_{spd}(v)$",
    r"$f(q)$", r"$f_{pt}(q)$", r"$f_{vanilla\_ft}(q)$", r"$f_{linear\_prob}(q)$", r"$f_{lp\_ft}(q)$", r"$f_{ftp}(q)$", r"$f_{spd}(q)$",
    r"$f_{pt}(v, q)$", r"$f_{vanilla\_ft}(v, q)$", r"$f_{linear\_prob}(v, q)$", r"$f_{lp\_ft}(v, q)$", r"$f_{ftp}(v, q)$", r"$f_{spd}(v, q)$"
]


dataset_names = [
    "VQAv2 (val)", "IV-VQA", "CV-VQA", "VQA-Rep.", "VQA-CP", "VQA-CE", 
    "AdVQA", "Near OOD Avg.", "TextVQA", "VizWiz", "OK-VQA", "Far OOD Avg.", "OOD Avg."
]

# Create DataFrame with the specified row and column labels
df_complete = pd.DataFrame(data_complete_23_rows, index=row_labels, columns=dataset_names)

# Moving x-axis labels to the top
plt.figure(figsize=(14, 8))
sns.heatmap(df_complete, annot=True, cmap="Blues", cbar_kws={'label': 'Mahalanobis Distance'}, fmt=".2f", annot_kws={"size": 10})

# Adding a horizontal line between specified rows (row index 9 and 10)
# plt.axhline(y=2, color='white', linewidth=4)  # Adjust `y` to specify the row index for the line
plt.axhline(y=7, color='white', linewidth=5)  # Adjust `y` to specify the row index for the line
plt.axhline(y=14, color='white', linewidth=5)  # Adjust `y` to specify the row index for the line

# Adding a vertical line between specified columns (column index 7 and 8)
plt.axvline(x=1, color='white', linewidth=5)  # Adjust `x` to specify the column index for the line
plt.axvline(x=8, color='white', linewidth=5)  # Adjust `x` to specify the column index for the line
plt.axvline(x=12, color='white', linewidth=5)  # Adjust `x` to specify the column index for the line

# Move x-axis labels to the top and apply a 90-degree rotation
plt.gca().xaxis.tick_top()
plt.xticks(rotation=45, ha='center', fontsize=10)  # Ensure 90-degree rotation with centered alignment
plt.yticks(fontsize=10)
# plt.title("Final Complete OOD Distance Heatmap with Custom Row and Column Labels (Lighter is ID, Darker is OOD)")

# Increase the font size of the color bar legend
cbar = plt.gca().collections[0].colorbar
cbar.ax.tick_params(labelsize=12)
cbar.set_label('Mahalanobis Distance', fontsize=14)

# # Plotting the heatmap with dataset names on top
# plt.figure(figsize=(14, 10))
# sns.heatmap(df_complete, annot=True, cmap="Blues", cbar_kws={'label': 'OOD Distance'}, fmt=".2f")
# plt.xticks(rotation=90, ha='left')
# plt.title("Final Complete OOD Distance Heatmap with Custom Row and Column Labels (Lighter is ID, Darker is OOD)")
plt.tight_layout(pad=0.1)
plt.savefig("ood_distance_heatmap.png", dpi=500, bbox_inches="tight")
