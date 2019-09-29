# ImageComparison
Determine cosine similarity of a human picture to

### Step 1: Download Repo
```
git clone https://github.com/whs2k/ImageSimilarity.git
```
### Step 2: Setup Environment

```conda create --name mxnet
conda activate mxnet
conda install jupyter
onda install -c anaconda pandas requests tqdm pil flask pip
pip install --upgrade mxnet gluoncv
conda install -c conda-forge opencv
```
### Step 3: Run Flask App
```
cd {insert_path_here}/src
python src/0.2.1-whs-invoSho_yolo.py
```

## App Logic
1. App Initialization (loading of DenseNet Model, YOLO Model, and dataframe of previously vectorized dog photos)
2. Human Uploads an Image and is saved to the "Static" folder
3. Yolo Model extractts and crops just the "human" portion of uploaded image
4. Densenet Model transforms the cropped image to a vector
5. Cosine Similarity is calculated for tthe uploaded image vector and all previously vectorized dog photos
6. Top three results are returned 

## Notes
- File "data/processed/0.0.6-whs-dogVectors.h5" is managed with Git LFS: https://help.github.com/en/articles/installing-git-large-file-storage
