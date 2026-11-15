# Data

Dataset: **LGG MRI Segmentation** (Kaggle, `mateuszbuda/lgg-mri-segmentation`)
110 TCGA patients, 3,929 `.tif` slices with matching `*_mask.tif` tumor masks.

Put it here so the path is `data/kaggle_3m/<patient_folder>/*.tif`.

Option A (manual): download from Kaggle, unzip, then move the `kaggle_3m` folder into `data/`.

Option B (kagglehub, e.g. on Colab):

```python
import kagglehub, shutil
path = kagglehub.dataset_download("mateuszbuda/lgg-mri-segmentation")
shutil.copytree(f"{path}/kaggle_3m", "data/kaggle_3m")
```

Or skip copying and pass the path directly: `python scripts/train.py --data_dir <path>/kaggle_3m`.
