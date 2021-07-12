#!/usr/bin/env python
# coding: utf-8

# ### Bibliotecas

# In[1]:


##  clases genéricas
import numpy as np
import os
#import pickle
import pandas as pd


# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dropout, Dense, Input, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model

from sklearn.metrics import  confusion_matrix, roc_auc_score, f1_score,  precision_score, recall_score #, roc_curve, auc


from keras.preprocessing import image as keras_image
import keras


# ### Dataset

# In[2]:


# from PIL import Image
# def load_image(df_image_path, inputShape):
#     """Load grayscale image
#     :param image_path: path to image to load
#     :return: loaded image
#     """
    
#     imagens = []
#     for i in range(df_image_path.shape[0]):
#         image_path = df_image_path[i]   
#         img = Image.open(image_path)
#         img = img.convert('RGB')
#         img = img.resize(inputShape)
#         img = np.asarray(img)
    
        
#         #img = keras_image.load_img(image_path, target_size = inputShape, interpolation = 'bicubic', color_mode = "rgb")
#         #img = 255.0*np.array(img) / (1.0*np.nanmax(img))
#         imagens.append(img)
#     return imagens   #  [0,255]


# In[3]:


### Definir paths em meu computador                 

Path_open_csv = 'CSV_Files/'
Path_images = 'Covid19_RadiograpyDataset/'
Path_save_result = 'Result/'  ## definir pasta onde salvar pesos e log
Path_Open_peso = 'Experiment/'  ## definir pasta onde salvar pesos e log

term_imagem = '.png'

### Definir parâmetros
width       = 224 
height      = 224
depth       = 3                 # 3 se RGB, 1 se grayscale (weights_ini= None)
inputShape  = [height, width, depth]



################################################################################################
#                   carregar e montar csv de test
################################################################################################
data_test = pd.DataFrame()
data_test_temp = pd.read_csv(Path_open_csv + 'test.csv')
data_test_temp['Image_name'] = data_test_temp['folder'] + data_test_temp['image_name'] + term_imagem
data_test = data_test_temp[['Image_name', 'label']]


#### montar vector covid=1 normal=0
data_test['classVector'] =  data_test['label']
data_test['Path'] =  Path_images + data_test['Image_name'] 


# In[4]:


# test_images = load_image(data_test['Path'], inputShape[0:2])
# label_test = data_test['classVector']


# ### Dados de salvamento

# In[5]:


# def create_folders(path):
#     if not os.path.exists(path):
#         os.makedirs(path)


# In[6]:


# def save_object(obj, filename):
#     with open(filename, 'wb') as output:  # Overwrites any existing file.
#         pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


# In[7]:


rede = 'VGG19'
#'NASNetLarge' #'DenseNet201', 'MobileNetV2', 'ResNet50V2', 'VGG19', 'VGG16', 'EfficientNetB0', 'EfficientNetB4'
#'Xception', 'InceptionV3', 'InceptionResNetV2'
weights_ini = 'imagenet'  # 'imagenet'  None
NUM_EPOCHS = 80
BATCH_SIZE  = 16


# In[8]:


################################################################################################
#                                Dados do treino 
################################################################################################

l_nome_proceso = 'CNN_mlp'
experimento = l_nome_proceso + '_WH_' + str(width) 

### definir pasta onde esta o peso treinado
Path_Open_peso_bes   = Path_Open_peso + '/Models/' + experimento +  '/'

### Criar pastas onde estão os pesos
#create_folders(Path_Open_peso)

### definir nomes de arquivos de saida
palavra_s = 'ensaio_' + rede + '_' + weights_ini + '_WH_' + str(width)       
arquivo_best_peso = os.path.sep.join([Path_Open_peso_bes, palavra_s + ".hdf5"]) 


# ### Image Generator

# In[9]:


class Data_GeneratorRx(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data_path_img, data_lab, index_b, batch_size=BATCH_SIZE, inputShape=inputShape, aug = 0, shuffle=True):            
        'Initialization'
        if aug == 1:
            data_gen = ImageDataGenerator(  
                                        rotation_range     = 15,
                                        # zoom_range         = 0.05,
                                        width_shift_range  = 0.1,
                                        height_shift_range = 0.1,
                                        # shear_range        = 0.01,
                                        horizontal_flip    = True,
                                        fill_mode          = "nearest",
                                        preprocessing_function = preprocess_input,
                                        )
        else:
            data_gen = ImageDataGenerator(
                                    preprocessing_function = preprocess_input, # se usar isto a imagem entrada deve ficar 0-255
                                 )            
        self.data_path_img  =  data_path_img   
        self.labels         =  data_lab   
        self.index_b        =  index_b    
        self.batch_size     =  batch_size # int
        self.inputShape     =  inputShape               # tuple int)
        self.data_gen       =  data_gen
        self.shuffle        =  shuffle
        self.aug            =  aug
        self.on_epoch_end()
        
####################################################################################################################################
    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch        """
        return int(np.ceil( len(self.data_path_img)  /  float(self.batch_size) ))
    
####################################################################################################################################
    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        ## Generate indexes of the batch
        indexes = self.index_b[index * self.batch_size:(index + 1) * self.batch_size]
        ## Find list of IDs (paths)
        list_IDs_temp = [self.data_path_img[k] for k in indexes]
        ## Generate data
        X = self._generate_X(list_IDs_temp)
        y = self.labels[indexes]                
        ## Generate Augmented data
        iterat = self.data_gen.flow(X, y, shuffle=self.shuffle)
        X0,y0  = iterat.next()
        y_mat  = np.array(np.uint32 (y0.tolist()) )
        return X0,y_mat
    
####################################################################################################################################
    def _generate_X(self, list_IDs_temp):
        """Generates data containing batch_size images
        :param list_IDs_temp: list of image paths to load
        :return: batch of images
        """
        # Initialization
        X = np.empty( (len(list_IDs_temp), *self.inputShape) )
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            X[i,] = self._load_image(ID)
        return X
    
####################################################################################################################################
    def _load_image(self, image_path):
        """Load grayscale image
        :param image_path: path to image to load
        :return: loaded image
        """
        img = keras_image.load_img(image_path ,target_size=self.inputShape,interpolation='bicubic',color_mode = "rgb")
        img = 255.0*np.array(img) / (1.0*np.nanmax(img))
        return img   #  [0,255]
    
####################################################################################################################################
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.index_b = np.arange(len(self.data_path_img))
        if self.shuffle == True:
            np.random.shuffle(self.index_b)      


# ### Rede

# In[10]:


def get_model(model_name, model_weights_ini, inputShape):        
    if model_name   == 'ResNet50V2':                 
        base_model = ResNet50V2(include_top=False,  weights = model_weights_ini, input_tensor=Input(shape=inputShape))
    
    elif model_name == 'VGG16':          
        base_model = VGG16(include_top=False,  weights = model_weights_ini, input_tensor=Input(shape=inputShape))
    
    elif model_name == 'VGG19':            
        base_model = VGG19(include_top=False,  weights = model_weights_ini, input_tensor=Input(shape=inputShape))
        
    elif model_name == 'EfficientNetB4':        
        base_model = EfficientNetB4(include_top=False,  weights = model_weights_ini, input_tensor=Input(shape=inputShape))   
        
    elif model_name == 'EfficientNetB0':  
        base_model = EfficientNetB0(include_top=False,  weights = model_weights_ini, input_tensor=Input(shape=inputShape))
        
    elif model_name == 'MobileNetV2':             
        base_model = MobileNetV2(include_top=False,  weights = model_weights_ini, input_tensor=Input(shape=inputShape)) 
    
    elif model_name == 'DenseNet201':             
        base_model = DenseNet201(include_top=False,  weights = model_weights_ini, input_tensor=Input(shape=inputShape))   
    
    elif model_name == 'NASNetLarge':             
        base_model = NASNetLarge(include_top=False,  weights = model_weights_ini, input_tensor=Input(shape=inputShape)) 
        
    elif model_name == 'Xception':             
        base_model = Xception(include_top=False,  weights = model_weights_ini, input_tensor=Input(shape=inputShape))   
        
    elif model_name == 'InceptionV3':             
        base_model = InceptionV3(include_top=False,  weights = model_weights_ini, input_tensor=Input(shape=inputShape))   
        
    elif model_name == 'InceptionResNetV2':             
        base_model = InceptionResNetV2(include_top=False,  weights = model_weights_ini, input_tensor=Input(shape=inputShape))   
    return base_model


# In[11]:


if rede == 'ResNet50V2': 
    from tensorflow.keras.applications import ResNet50V2
    from tensorflow.keras.applications.resnet import preprocess_input
elif rede == 'VGG16': 
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.applications.vgg16 import preprocess_input
elif rede == 'VGG19': 
    from tensorflow.keras.applications import VGG19
    from tensorflow.keras.applications.vgg19 import preprocess_input
elif rede == 'EfficientNetB4': 
    from keras_efficientnets import  EfficientNetB4
    from keras_efficientnets import  preprocess_input  
elif rede == 'EfficientNetB0':
    from keras_efficientnets import  EfficientNetB0
    from keras_efficientnets import  preprocess_input  
elif rede == 'MobileNetV2': 
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
elif rede == 'DenseNet201': 
    from tensorflow.keras.applications import DenseNet201
    from tensorflow.keras.applications.densenet import preprocess_input
elif rede == 'NASNetLarge': 
    from tensorflow.keras.applications import NASNetLarge
    from tensorflow.keras.applications.nasnet import preprocess_input    
elif rede == 'Xception': 
    from tensorflow.keras.applications import Xception
    from tensorflow.keras.applications.xception import preprocess_input    
elif rede == 'InceptionV3': 
    from tensorflow.keras.applications import InceptionV3
    from tensorflow.keras.applications.inception_v3 import preprocess_input    
elif rede == 'InceptionResNetV2': 
    from tensorflow.keras.applications import InceptionResNetV2
    from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input    


# In[12]:


### definir dictionario para apontador dentro do generator
paramsB = {
              'batch_size': BATCH_SIZE,
              'inputShape': inputShape,
          } 


# In[13]:


################################################################################################
#                  Montar CONV net
################################################################################################
##  carregar modelo base
base_model = get_model(rede, weights_ini, inputShape)
## construir MLP
headModel = base_model.output
headModel = GlobalAveragePooling2D()(headModel)
headModel = BatchNormalization()(headModel)
headModel = Dense(256, activation="relu")(headModel)
headModel = Dropout(0.3)(headModel)                      # orig = 0.5
headModel = Dense(1, activation="sigmoid", name="pred")(headModel)

## place the head FC model on top of the base model (this will become the actual model we will train)
model = Model(inputs=base_model.input, outputs=headModel, name="Classifica")

## carregar pesos
model.load_weights(arquivo_best_peso)


# In[14]:


################################################################################################
#                  Montar generator
################################################################################################  
test_generator = Data_GeneratorRx(data_path_img = data_test['Path']  , data_lab = data_test['classVector'], index_b = np.arange(len(data_test)), **paramsB, aug = 0, shuffle = False)


# In[15]:


################################################################################################
#                  predecir
################################################################################################
y_pred = model.predict(test_generator,    verbose=1  )
y_true = np.array(np.uint32 (data_test['classVector'].tolist()) ).reshape(len(data_test['classVector']),1)
      
## Assumir  threshold
threshold = [0.5]
AUC     = roc_auc_score(y_true, y_pred)  

## Find prediction  applying threshold
y_pred_bin = np.array([1 if prob >= threshold else 0 for prob in y_pred])
y_pred_bin = y_pred_bin.reshape([len(y_pred_bin),1])

## Calcular metricas
tn0, fp0, fn0, tp0 = confusion_matrix(y_true , y_pred_bin).ravel()
f1    = f1_score(y_true, y_pred_bin, average='binary')
acc   = (tp0 + tn0) / (tp0 + tn0 + fp0+ fn0)
ps    = precision_score(y_true , y_pred_bin)
rec   = recall_score(y_true , y_pred_bin)

## salvar metricas
metricas = np.zeros([y_pred.shape[1],12])
metricas[:] = [0,0, threshold[0],tp0, fp0, tn0, fn0, AUC, f1 , acc, ps, rec]
salida   = pd.DataFrame(metricas).rename(columns={0:'rede', 1:'Base',  2:'threshold',3:'TP', 4:'FP', 5:'TN', 6:'FN', 7:'AUROC', 8:'F1', 9:'acc', 10:'prec', 11: 'rec'})
salida['Base'] = 'exemplo cnn'
salida['rede'] = rede

salida.to_csv(Path_save_result + 'res_' + rede + '_' + weights_ini + '_WH_' + str(width) + '.csv' , index = False)


# In[ ]:





# In[ ]:




