import mxnet as mx
from mxnet.gluon.model_zoo import vision
from gluoncv import model_zoo, data, utils #For YOLO
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

import glob2
from tqdm import tqdm
import matplotlib.pyplot as plt

from Config import FN_DF_TRANSFORMED



def yolo_human_extraction(fnx, yolo_netX):
    x, img = data.transforms.presets.yolo.load_test(fnx, short=512)
    class_IDs, scores, bounding_boxs = yolo_netX(x)
    for index, bbox in enumerate(bounding_boxs[0]):
        class_ID = int(class_IDs[0][index].asnumpy()[0])
        class_name = yolo_netX.classes[class_ID]
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
            return im_fname_save

def cropNormFit(fnx):
    '''
    accepts an mx decoded image
    returns an mxnet array ready for transformation image
    '''
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
    '''
    accepts a preprocessed vector
    returns a numpy transformation
    '''
    return preloaded_model.features(batchified)[0].asnumpy()

def cosineSimilarity(u, v):
    similarity = np.dot(u,v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return float(similarity)

def load_model():
    print("we're loading densenet model: \
        https://modelzoo.co/model/densely-connected-convolutional-networks-2")
    densenetX = vision.densenet201(pretrained=True)
    print("we just loaded: ")
    type(densenetX)
    print("Now we're loading YOLO")
    yolo_netX = model_zoo.get_model('yolo3_darknet53_voc', pretrained=True)
    type(yolo_netX)
    return densenetX, yolo_netX

def init():
    densenet_model, yolo_net = load_model()

    fn_to_compare = input("gimme the fn of a pic please! ")
    print("if this isn't your image; give up now")    
    with Image.open(fn_to_compare) as img:
        img.show()
    #img = Image.open(fn_to_compare)
    #img.show() 

    fn_to_compare_crop = yolo_human_extraction(fn_to_compare, yolo_net)
    print("we are extractiong just the human there: ")
    print("if this isn't your image; give up now")    
    with Image.open(fn_to_compare_crop) as img:
        img.show()
    batchified_image = cropNormFit(fn_to_compare_crop)
    #densenet_model = load_model
    
    img_vec = vectorize(batchified_image ,preloaded_model=densenet_model)

    fn_df_load = FN_DF_TRANSFORMED
    df_corpus = pd.read_hdf(FN_DF_TRANSFORMED, key='df')
    #df_corpus['ref_vec'] = img_vec
    df_corpus['ref_cosim'] = df_corpus['vector'].apply(lambda x: cosineSimilarity(u = x,
                                        v = img_vec))

    
    '''
    for index in tqdm(range(df_corpus.count()[0])):
        try:
            cos_sim = cosineSimilarity(u = df_corpus['vector'].loc[index],
                                        v = img_vec)
            df_corpus['ref_cosim'].loc[index] = cos_sim
        except:
            df_corpus['ref_cosim'].loc[index] = 0
            continue'''
    df_corpus = df_corpus.sort_values('ref_cosim', ascending=False).reset_index(drop=True)
    print('There are {} fns to compare: '.format(df_corpus.count()[0]))
    print(df_corpus.head())
    for index in range(3):
        print(df_corpus['fn'].loc[index])
        print(df_corpus['ref_cosim'].loc[index])
        #img = Image.open(fn_to_compare)
        #img.show() 
        with Image.open(df_corpus['fn'].loc[index]) as img:
            img.show()

if __name__ == "__main__":
    init()
