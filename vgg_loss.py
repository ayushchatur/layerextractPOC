from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
import numpy as np

# use the entire model with imagenet weights and exclude last dense layer of classification
base_model = VGG16(weights='imagenet',include_top = False)

# select the layer block4_pool
model = Model(inputs= base_model.input,outputs=base_model.get_layer('block4_pool').output)
# select the layer block1_conv2
model2 = Model(inputs= base_model.input,outputs=base_model.get_layer('block1_conv2').output)
base_model.summary()
model2.summary()

img_path = 'C:/Users/ayush/Downloads/images.jpg'
img_path2 = 'C:/Users/ayush/Downloads/BIMCV_139_image_0.tif'
img = image.load_img(img_path,target_size=(224,224))
x = image.img_to_array(img)
x = np.expand_dims(x,axis=0)
x = preprocess_input(x)

features = model.predict(x)
#
print(features.shape)
print(features)
#
# features = model2.predict(x)
#
# print(features.shape)
# print(features)


base_model2 = VGG16()

from matplotlib import pyplot

for layer in base_model2.layers:
    if 'conv' not in layer.name:
        continue

    filters,biases = layer.get_weights()
    print(layer.name,filters.shape)

filters,bias = base_model2.layers[1].get_weights()

f_min, f_max = filters.min(),filters.max()
filters = (filters - f_min) / (f_max- f_min)

n_filters , ix = 6,1

for i in range(n_filters):
    f = filters[:,:,:,i]
    for j in range(3):
        ax = pyplot.subplot(n_filters,3,ix)
        ax.set_xticks([])
        ax.set_yticks([])
        pyplot.imshow(f[:,:,j],cmap='gray')
        ix +=1


pyplot.show()