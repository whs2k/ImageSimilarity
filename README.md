# ImageComparison
Determine cosine similarity of a human picture to

### Step 1: Download Repo

### Step 2: Setup Environment

```conda create --name mxnet
conda activate mxnet
conda install jupyter
conda install -c anaconda mxnet
conda install -c anaconda pandas 
conda install -c anaconda requests 
conda install -c conda-forge tqdm
conda install -c anaconda pil
conda install -c anaconda flask 
pip install --upgrade mxnet gluoncv
conda install -c conda-forge opencv
```
### Step 3: Run Flask App
```cd {insert_path_here}/src
python src/0.2.0-whs-invoSho.py```

## App Logic
1. App Initialization (loading of DenseNet Model, YOLO Model, and dataframe of previously vectorized dog photos)
2. Human Uploads an Image and is saved to the "Static" folder
3. Yolo Model extractts and crops just the "human" portion of uploaded image
4. Densenet Model transforms the cropped image to a vector
5. Cosine Similarity is calculated for tthe uploaded image vector and all previously vectorized dog photos
6. Top three results are returned 