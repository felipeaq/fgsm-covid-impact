#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


# ### Bibliotecas

# In[2]:


import cleverhans
import tensorflow as tf
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from tensorflow.keras.preprocessing.image import ImageDataGenerator
np.random.seed(4)

# In[3]:


from tensorflow.keras.callbacks import EarlyStopping , ModelCheckpoint, ReduceLROnPlateau,TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, BatchNormalization, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.optimizers import  Adam  #, SGD , Adam

from keras.preprocessing import image as keras_image
import keras


# ### Dataset

# In[4]:


from PIL import Image
def load_image(df_image_path, inputShape):
    """Load grayscale image
    :param image_path: path to image to load
    :return: loaded image
    """
    
    imagens = []
    for i in range(df_image_path.shape[0]):
        image_path = df_image_path[i]   
        img = Image.open(image_path)
        img = img.convert('RGB')
        img = img.resize(inputShape)
        img = np.asarray(img)
    
        
        #img = keras_image.load_img(image_path, target_size = inputShape, interpolation = 'bicubic', color_mode = "rgb")
        #img = 255.0*np.array(img) / (1.0*np.nanmax(img))
        imagens.append(img)
    return imagens   #  [0,255]


# In[5]:


### Definir paths em meu computador                 

Path_open_csv = 'CSV_Files/'
Path_images = 'Covid19_RadiograpyDataset/'
Path_save =  'Experiment/'

term_imagem = '.png'

### Definir parâmetros
width       = 224 
height      = 224
depth       = 3                 # 3 se RGB, 1 se grayscale (weights_ini= None)
inputShape  = [height, width, depth]


################################################################################################
#                   carregar e montar csv de treino e validação
################################################################################################
data_train = pd.DataFrame()
data_train_temp = pd.read_csv(Path_open_csv + 'train.csv')
data_train_temp['Image_name'] = data_train_temp['folder'] + data_train_temp['image_name'] + term_imagem
data_train = data_train_temp[['Image_name', 'label']]


data_val = pd.DataFrame()
data_val_temp = pd.read_csv(Path_open_csv + 'val.csv')
data_val_temp['Image_name'] =  data_val_temp['folder'] + data_val_temp['image_name'] + term_imagem
data_val = data_val_temp[['Image_name', 'label']]
del data_train_temp, data_val_temp

#### montar vector covid=1 normal=0
data_val['classVector']   =  data_val['label']
data_train['classVector'] =  data_train['label']
   
data_train['Path'] =  Path_images + data_train['Image_name'] 
data_val['Path']   =  Path_images + data_val['Image_name']  


# In[6]:


train_images = load_image(data_train['Path'], inputShape[0:2])
val_images = load_image(data_val['Path'], inputShape[0:2])


# ### Dados de salvamento

# In[7]:


def create_folders(path):
    if not os.path.exists(path):
        os.makedirs(path)


# In[8]:


def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


# In[10]:


rede = sys.argv[1] #'ResNet50V2'
#'NASNetLarge' #'DenseNet201', 'MobileNetV2', 'ResNet50V2', 'VGG19', 'VGG16', 'EfficientNetB0', 'EfficientNetB4'
#'Xception', 'InceptionV3', 'InceptionResNetV2'
weights_ini = 'imagenet'  # 'imagenet'  None
NUM_EPOCHS = 80
BATCH_SIZE  = 16


# In[11]:


################################################################################################
#                                Dados do treino 
################################################################################################

l_nome_proceso = 'CNN_mlp'
n_salv         = 'CNN_mlp_temp'

experimento = l_nome_proceso + '_WH_' + str(width) 
### Estabelecer path para salvar resultados e pesos do experimento
Path_save_peso  =  Path_save + '/Models/' + experimento +  '/'
Path_save_logs  =  Path_save + '/logs/'   + experimento +  '/'
### Criar pastas onde serão salvados os pesos do experimento
create_folders(Path_save_peso)

### definir nomes de arquivos de saida
palavra_s            = 'ensaio_' + rede +  '_'  + weights_ini + '_WH_' + str(width)       
arquivo_saida        = os.path.sep.join([Path_save_peso, "R_"+ palavra_s + ".hdf5"]) 
arquivo_saida_best_w = os.path.sep.join([Path_save_peso, "W_"+ palavra_s + "best_W.hdf5"]) 
arquivo_saida_histo  = os.path.sep.join([Path_save_peso, "H_"+ palavra_s + "Hist.pkl"]) 

### definir pasta para armazenar o Log
log_dir3  = Path_save_logs + palavra_s + '/'
create_folders(log_dir3)


# ### Image Genarator

# In[12]:


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

# In[13]:


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


# In[14]:


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


# In[15]:


### definir dictionario para apontador dentro do generator
paramsB = {
              'batch_size': BATCH_SIZE,
              'inputShape': inputShape,
          } 


# In[16]:


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


## Treinar toda a base menos as layer BatchNormalization
for layer in base_model.layers:
    if not isinstance(layer, layers.BatchNormalization):
        layer.trainable = True    

## compile our model (this needs to be done after our setting our layers to being non-trainable
print("[INFO] compiling model...")
model.compile( 
                loss      = 'binary_crossentropy',  
                optimizer = Adam(lr = 1e-4),
                metrics   = ['accuracy', 'mae'] 
             )     
model.summary()


# In[17]:


### Montar generator para leitura dos dados      
n_treino = len(data_train)
index_t  = np.arange(n_treino)

np.random.shuffle(index_t)
np.random.shuffle(index_t)
np.random.shuffle(index_t)

training_generator   = Data_GeneratorRx(data_path_img = data_train['Path'], data_lab = data_train['classVector'], index_b = index_t, **paramsB, aug = 1)
validation_generator = Data_GeneratorRx(data_path_img = data_val['Path'], data_lab = data_val['classVector'], index_b = np.arange(len(data_val)), **paramsB, aug = 0,shuffle = False)


# In[18]:


### create callback
Arquivo_temp_peso = Path_save_peso  + palavra_s + '.hdf5'
es        = EarlyStopping(  monitor='val_loss', patience= 10, mode='auto', verbose=1)
mc        = ModelCheckpoint(Arquivo_temp_peso , monitor='val_loss', save_best_only=True, mode='auto', verbose=1)
tensorboard_callback = TensorBoard(log_dir=log_dir3, histogram_freq=1)
Rp        = ReduceLROnPlateau(
                        monitor="val_loss",  factor=0.2,   patience=7,
                        verbose=1,           mode="auto",  min_lr=1e-6,                        
                            )


# In[19]:


### Treinar
H = model.fit(
            training_generator,
            validation_data = validation_generator,
            epochs          = NUM_EPOCHS,
            callbacks       = [ mc, es , tensorboard_callback , Rp],
            steps_per_epoch = np.ceil(len(index_t) / BATCH_SIZE),
            verbose         = 1,
            shuffle         = True
        )


# In[ ]:





# In[ ]:





# In[ ]:




