
from glob import glob 
import numpy as np 
import os 
import pandas as pd
from tkinter import Tk, messagebox, filedialog
from numpy.lib.npyio import load
from sklearn.metrics import matthews_corrcoef, accuracy_score

##========================================================
Tk().withdraw()
direc = filedialog.askdirectory(initialdir="./", title='Select directory of results (npz)')
model_files = sorted(glob(os.path.join(direc, '*.npz')))

excel_file_path = filedialog.askopenfilename(initialdir=direc, title="Select Excel metadata file", filetypes=[("Excel files", "*.xlsx")])
if os.path.exists(excel_file_path):
    df_excel = pd.read_excel(excel_file_path)
else:
    df_excel = pd.DataFrame()

NUM_LABEL_CLASSES = 5

def calculate_percentile(array):
    results = array.copy()
    results[results == 5] = 3

    percentages = {}
    for cls in range(NUM_LABEL_CLASSES):
        cls_count = np.sum(results == cls)
        percentages[cls] = 100 * cls_count / results.size if results.size > 0 else 0.0

    return percentages

all_data = []

for file in model_files:
    with load(file, allow_pickle=True) as data:
        if "grey_label" in data:
            pred_results = data['grey_label']
        elif "label" in data:
            pred_results = data['label']
        else:
            pred_results = data['arr_1']

    if pred_results.ndim == 3:
        pred_results = np.argmax(pred_results, axis=-1)

    percentages = calculate_percentile(pred_results)

    row = {'file': os.path.basename(file)}
    for cls_result, val_result in percentages.items():
        row[f'class_{cls_result}'] = val_result

    # Sand-to-coarse ratio: sand (0) / (gravel 1 + cobble 2)
    sand = percentages.get(0, 0)
    gravel = percentages.get(1, 0)
    cobble = percentages.get(2, 0)
    coarse_total = gravel + cobble
    row['sand_to_coarse_ratio'] = sand / coarse_total if coarse_total > 0 else np.nan

    all_data.append(row)

df_update = pd.DataFrame(all_data)

# Merge with existing Excel file, updating by filename
if not df_excel.empty:
    df_merged = pd.concat([df_excel, df_update], ignore_index=True)
    df_merged.drop_duplicates(subset=['file'], keep='last', inplace=True)
else:
    df_merged = df_update

# Save back to Excel
df_merged.to_excel(excel_file_path, index=False)
print(f"Excel file updated: {excel_file_path}")
