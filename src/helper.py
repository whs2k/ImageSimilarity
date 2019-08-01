import mxnet as mx
from mxnet.gluon.model_zoo import vision
import os
import numpy as np
import glob
import pandas as pd
#from scipy.spatial.distance import cosine
#from IPython.display import Image 
from PIL import Image    
from tqdm import tqdm

# set the context on CPU, switch to GPU if there is one available
ctx = mx.cpu()
pd.set_option('display.max_colwidth', -1)

import glob2
from tqdm import tqdm

'''def cropNormFit(fnx):
    "
    accepts an mx decoded image
    returns an mxnet array ready for transformation image
    "
    image = mx.image.imdecode(open(fnx, 'rb').read()).astype(np.float32)
    resized = mx.image.resize_short(image, 224) #minimum 224x224 images
    cropped, crop_info = mx.image.center_crop(resized, (224, 224))
    normalized = mx.image.color_normalize(cropped/255,
                                          mean=mx.nd.array([0.485, 0.456, 0.406]),
                                          std=mx.nd.array([0.229, 0.224, 0.225])) 
    # the network expect batches of the form (N,3,224,224)
    flipped_axis = normalized.transpose((2,0,1))  # Flipping from (224, 224, 3) to (3, 224, 224)
    batchified = flipped_axis.expand_dims(axis=0) # change the shape from (3, 224, 224) to (1, 3, 224, 224)
    return batchified

def vectorize(batchified, preloaded_model):
    "
    accepts a preprocessed vector
    returns a numpy transformation
    "
    return preloaded_model.features(batchified)[0].asnumpy()

def cosineSimilarity(u, v):
    similarity = np.dot(u,v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return float(similarity)


def get_image_sims(fn_image_to_compare, trained_model, fn_df_save):
    batchified_image = cropNormFit(fn_image_to_compare)
    img_vec = vectorize(batchified_image ,preloaded_model=trained_model)
    df_corpus = pd.read_pickle(fn_df_save).reset_index(drop=True)
    df_corpus['ref_vec'] = None
    df_corpus['ref_cosim'] = None

    for index in tqdm(range(df_corpus.count()[0])):
        try:
            cos_sim = cosineSimilarity(u = df_corpus['vector'].loc[index],
                                        v = img_vec)
            df_corpus['ref_cosim'].loc[index] = cos_sim
        except:
            df_corpus['ref_cosim'].loc[index] = 0
            continue
    return df_corpus'''

def load_model():
    print("we're loading densenet model: \
        https://modelzoo.co/model/densely-connected-convolutional-networks-2")
    densenetX = vision.densenet201(pretrained=True)
    print("we just loaded: ")
    type(densenetX)
    return densenetX

def yolo_extraction(fnx):
    x, img = data.transforms.presets.yolo.load_test(fn, short=512)
    class_IDs, scores, bounding_boxs = net(x)
    for index, bbox in enumerate(bounding_boxs[0]):
        class_ID = int(class_IDs[0][index].asnumpy()[0])
        class_name = net.classes[class_ID]
        class_score = scores[0][index].asnumpy()
        if (class_name == 'person') & (class_score > 0.9):
            #print('index: ', index)
            #print('class_ID: ', class_ID)
            #print('class_name: ', class_name)
            #print('class_score: ',class_score)
            #print('bbox: ', bbox.asnumpy())
            xmin, ymin, xmax, ymax = [int(x) for x in bbox.asnumpy()]
            xmin = max(0, xmin)
            xmax = min(x.shape[3], xmax)
            ymin = max(0, ymin)
            ymax = min(x.shape[2], ymax)
            im_fname_save = fnx.replace('.jpg','_humanCrop.jpg')
            plt.imsave(im_fname_save, img[ymin:ymax,xmin:xmax,:])
            return 
def cropNormFit(fnx):
    '''
    accepts an mx decoded a filename of an image
    returns an mxnet array ready for transformation image
    '''
        
    image = mx.image.imdecode(open(fn, 'rb').read()).astype(np.float32)
    resized = mx.image.resize_short(image, 224) #minimum 224x224 images
    cropped, crop_info = mx.image.center_crop(resized, (224, 224))
    normalized = mx.image.color_normalize(cropped/255,
                                          mean=mx.nd.array([0.485, 0.456, 0.406]),
                                          std=mx.nd.array([0.229, 0.224, 0.225])) 
    # the network expect batches of the form (N,3,224,224)
    flipped_axis = normalized.transpose((2,0,1))  # Flipping from (224, 224, 3) to (3, 224, 224)
    batchified = flipped_axis.expand_dims(axis=0) # change the shape from (3, 224, 224) to (1, 3, 224, 224)
    return batchified

def vectorize(batchified, preloaded_model):
    '''
    accepts a preprocessed vector
    returns a numpy transformation
    '''
    return preloaded_model.features(batchified)[0].asnumpy()

def cosineSimilarity(u, v):
    similarity = np.dot(u,v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return similarity

def get_image_sims(fn_image_to_compare, trained_model, fn_df_save):
    batchified_image = cropNormFit(fn_image_to_compare)
    img_vec = vectorize(batchified_image ,preloaded_model=trained_model)
    df_corpus = pd.read_hdf(fn_df_save, key='df').reset_index(drop=True)
    df_corpus['cosim'] = df_corpus['vector'].apply(lambda x: helper.cosineSimilarity(x, image_vect))
    df_corpus = df_corpus.sort_values('cosim', ascending=False).reset_index(drop=True)
    return df_corpus
    
    

def createResultsHTML(df_html, upload_image, fn_to_save):
    '''
    Input: dataframe of similarities, the full path of the uploaded image, 
        and the relative /templates .html results page
    Ouptput: Saves a .html file in the /tempates folder
    '''
    df_html_final = df_html.to_html().replace('<table border="1" class="dataframe">',
                        '''
                        <head><link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css"></head>
                        <table border="1" class="dataframe">''')
    list_of_topX_links = df_html.fn.tolist()[0:3] 
    list_of_perfect_links = df_html[df_html['ref_cosim'] > 0.9].fn.tolist()[0:3] 
    html_string = '''
    <!DOCTYPE html>
        <html lang="en">
        <head>
          <title>CCR Tweets</title>
          <meta charset="utf-8">
          <meta name="viewport" content="width=device-width, initial-scale=1">
          <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css">
          <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js"></script>
          <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js"></script>
        </head>
        <body>
            <div class="container">
            <h2>Your Upload: </h2>  
            <h3>{}</h3>
            <img src={} alt="Girl in a jacket" width="500" height="600">
            </div>
            <div class="container">
            <h2>Top Three: </h2> 
            XXX>
            </div>
            <div class="container">
            <h2>Perfect Matches: </h2> 
            YYY
            </div>
        </body>
        </html>
    '''.format(upload_image, upload_image)

    print('Helper Saving: ',fn_to_save)
    with open(fn_to_save, "w") as f:
        f.write(html_string)



def init():
    fn_to_compare = input("gimme the fn of a pic please! ")
    print("if this isn't your image; give up now")    
    with Image.open(fn_to_compare) as img:
        img.show()
    #img = Image.open(fn_to_compare)
    #img.show() 

    batchified_image = cropNormFit(fn_to_compare)
    #densenet_model = load_model
    print("we're loading densenet model: \
        https://modelzoo.co/model/densely-connected-convolutional-networks-2")
    densenet_model = vision.densenet201(pretrained=True)
    print("we just loaded: ")
    print(type(densenet_model))
    img_vec = vectorize(batchified_image ,preloaded_model=densenet_model)

    fn_df_save = os.path.join(os.path.dirname(os.getcwd()), 'data', 'processed','0.0.4-whs-dogVectors.pickle')
    df_corpus = pd.read_pickle(fn_df_save)
    df_corpus['ref_vec'] = None
    df_corpus['ref_cosim'] = None

    for index in tqdm(range(df_corpus.count()[0])):
        try:
            cos_sim = cosineSimilarity(u = df_corpus['vector'].loc[index],
                                        v = img_vec)
            df_corpus['ref_cosim'].loc[index] = cos_sim
        except:
            df_corpus['ref_cosim'].loc[index] = 0
            continue
    df_corpus = df_corpus.sort_values('ref_cosim', ascending=False).reset_index(drop=True)
    for index in range(3):
        print(df_corpus['fn'].loc[index])
        print(df_corpus['ref_cosim'].loc[index])
        #img = Image.open(fn_to_compare)
        #img.show() 
        with Image.open(df_corpus['fn'].loc[index]) as img:
            img.show()

if __name__ == "__main__":
    init()
