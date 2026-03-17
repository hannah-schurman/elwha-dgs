import os
import json
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tkinter import filedialog, Tk
from doodleverse_utils.imports import *
from doodleverse_utils.prediction_imports import do_seg
from transformers import TFSegformerForSemanticSegmentation


root = Tk()
print("Select BINARY weights:")
bin_weights = filedialog.askopenfilename(initialdir="./", title="Binary weights (.h5)", filetypes=[("Weights files", "*.h5")])
root.withdraw()

root = Tk()
print("\nSelect MULTICLASS weights:")
multi_weights = filedialog.askopenfilename(initialdir=os.path.dirname(bin_weights), title="Multiclass weights (.h5)", filetypes=[("Weights files", "*.h5")])
root.withdraw()

root = Tk()
print("\nSelect IMAGES to Segment")
seg_directory = filedialog.askdirectory(initialdir=os.path.dirname(bin_weights), title="Select directory of images to segment")

print("Segmenting images in:", seg_directory)
root.withdraw()


# Load config files
multi_configfile = multi_weights.replace('_fullmodel.h5','.json').replace('weights', 'config')
bin_configfile   = bin_weights.replace('_fullmodel.h5','.json').replace('weights', 'config')

with open(multi_configfile) as f:
    multi_config = json.load(f)

with open(bin_configfile) as f:
    bin_config = json.load(f)

# create dict for model info for binary and multiclass
metadatadict_mc = {
    "model_weights": [multi_weights],
    "config_files": [multi_configfile],
    "model_types": [multi_config["MODEL"]],
}

metadatadict_bin = {
    "model_weights": [bin_weights],
    "config_files": [bin_configfile],
    "model_types": [bin_config["MODEL"]],
}


# Load SegFormer models
def get_model(NCLASSES):
    model = TFSegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b0",
        num_labels=NCLASSES,
        ignore_mismatched_sizes=True
    )
    model.compile(optimizer='adam')
    return model


binary_model = get_model(bin_config["NCLASSES"])
binary_model.load_weights(bin_weights)

multiclass_model = get_model(multi_config["NCLASSES"])
multiclass_model.load_weights(multi_weights)


# Wrapper for do_seg
def run_do_seg(fpath, model, metadatadict, config, sample_dir, out_dir_name):
    out_dir = os.path.join(sample_dir, out_dir_name)
    os.makedirs(out_dir, exist_ok=True)

    do_seg(fpath, [model], metadatadict, config["MODEL"], sample_dir, config["NCLASSES"], config["N_DATA_BANDS"], config["TARGET_SIZE"],
        config.get("TESTTIMEAUG", False), config.get("WRITE_MODELMETADATA", False), config.get("OTSU_THRESHOLD", False), 
        out_dir_name=out_dir_name, profile="meta",
    )


# Prepare output directories
out_multi = os.path.join(seg_directory, "out_multi")
out_binary = os.path.join(seg_directory, "out_binary")
out_final = os.path.join(seg_directory, "out_final")

os.makedirs(out_multi, exist_ok=True)
os.makedirs(out_binary, exist_ok=True)
os.makedirs(out_final, exist_ok=True)

# Process all JPG/png images
for f in os.listdir(seg_directory):
    if f.endswith(".JPG"):
        os.rename(
            os.path.join(seg_directory, f),
            os.path.join(seg_directory, f.replace(".JPG", ".jpg"))
        )
    if f.endswith(".png"):
        os.rename(
            os.path.join(seg_directory, f),
            os.path.join(seg_directory, f.replace(".png", ".jpg"))
        )

jpg_files = sorted(tf.io.gfile.glob(os.path.join(seg_directory, "*.jpg")))
print(f"Found {len(jpg_files)} images")

for f in tqdm(jpg_files):
    basefile = os.path.basename(f).replace(".jpg", "")

    # Run do_seg
    run_do_seg(f, multiclass_model, metadatadict_mc, multi_config, seg_directory, "out_multi")
    run_do_seg(f, binary_model, metadatadict_bin, bin_config, seg_directory, "out_binary")

    mc_npz = os.path.join(out_multi, f"{basefile}_res.npz")
    bin_npz = os.path.join(out_binary, f"{basefile}_res.npz")

    # Load label outputs
    mc_data = np.load(mc_npz, allow_pickle=True)
    bin_data = np.load(bin_npz, allow_pickle=True)
    
    mc_lab = mc_data["grey_label"]
    bin_lab = bin_data["grey_label"]

    # Merge: scale bar binary = 1 -> multiclass = 4
    merged_lab = mc_lab.copy()
    binary_mask = (bin_lab == 1)    
    merged_lab[binary_mask] = 4

    mc_meta = {}
    for key in mc_data.files:
        if key == "grey_label":
            mc_meta[key] = merged_lab
        else:
            mc_meta[key] = mc_data[key]    
            
    # Save everything
    out_npz = os.path.join(out_final, f"{basefile}_merged_res.npz")
    np.savez_compressed(out_npz, **mc_meta)

    # clean up multi and binary so only final is saved - saves space
    for tmp in [mc_npz, bin_npz]:
        if os.path.exists(tmp):
            os.remove(tmp)

print("Finished Segmentation")
