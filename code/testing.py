# -*- coding: utf-8 -*-
"""testing.py
   This testing script predicts the calss for particular fruit or vegetable
"""

from cv2 import rotate
from matplotlib.ft2font import BOLD, HORIZONTAL
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import requests
import matplotlib
from keras_preprocessing import image

# definiing 20 classes
class_names = ['Apple Braeburn', 'Apple Golden', 'Avocado', 'Banana', 'Beetroot', 'Cocos', 'Eggplant', 'Ginger Root', 'Guava', 'Lemon', 'Lychee', 'Onion Red', 'Onion White', 'Orange', 'Pepper Green', 'Pineapple', 'Potato White', 'Strawberry', 'Watermelon', 'tomato']

print("\n")
print(class_names)
print("\n")

model_path = "/Users/priya/Desktop/final_model/model2/final_model_modified.h5"
cnn_model = tf.keras.models.load_model(model_path)
 
# Image prediction
image_path = "/Users/priya/Desktop/final_model/samples/tomato2.jpg"

# Loading image with size 150x150
sample_img = image.load_img(image_path, target_size = (150, 150))

# Converting image into an array
array = image.img_to_array(sample_img)

# Expanding the shape of an array
x = np.expand_dims(array, axis=0)

# This is to stack arrays in sequence vertically (row wise)
vimage = np.vstack([x])

# predicting the array of an image
res = cnn_model.predict(vimage)

# Getting the class position where output is 1
result = np.where(res[0] == 1)
# print(result)

print("-----------------------------")
print("The given fruit or vegetable is: "+class_names[result[0][0]])
print("-----------------------------")
imge = mpimg.imread(image_path)
plt.imshow(imge)
plt.title("Predicted: "+class_names[result[0][0]], size=20)

# displaying the image for which prediction is done 
plt.show()

# Plotting bar graph
matplotlib.use("MacOSX")

y_pos = np.arange(len(class_names))
fig = plt.figure(figsize = (16, 6))
plt.barh(class_names, res[0]*100, color = 'maroon')
plt.yticks(y_pos, class_names)

plt.xlabel('Prediction %')
plt.ylabel('Class names')
plt.title('Fruit and Vegetable Prediction...')
plt.show()


# Rest api call for recipe information
predicted_class_name = class_names[result[0][0]]
api_id = 'ad0e79fc'
api_key = '3c3d8e868b0dafc7ded261225a725325'
api_url = "https://api.edamam.com/search?q="+predicted_class_name+"&app_id="+api_id+"&app_key="+api_key+"&from=0&to=5"
api_response = requests.get(api_url)
data = api_response.json()

recipe_1 = data['hits'][0]
ingredients = recipe_1['recipe']['ingredientLines']

# printing the information from rest api response
if api_response.status_code == 200:
  print("Recipe Name:\n",recipe_1['recipe']['label'])
  print("\nIngredients:")
  print('\n'.join(str(x) for x in ingredients))
  print("\nSell full recipe:")
  print(recipe_1['recipe']['url'])
  print("\n")
else:
  print("Recipe not found")


