# This script is for graphical user interface for fruits and vegetable classification
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from turtle import left
from PIL import ImageTk, Image
from keras_preprocessing import image
from rsa import verify
import tensorflow as tf
import numpy as np
import requests
import urllib, io
import ssl
import webbrowser
ssl._create_default_https_context = ssl._create_unverified_context

# definiing 20 classes
classes = ['Apple Braeburn', 'Apple Golden', 'Avocado', 'Banana', 'Beetroot', 'Cocos', 'Eggplant', 'Ginger Root', 'Guava', 'Lemon', 'Lychee', 'Onion Red', 'Onion White', 'Orange', 'Pepper Green', 'Pineapple', 'Potato White', 'Strawberry', 'Watermelon', 'tomato']


# Loading two CNN models
cnn_model_2 = tf.keras.models.load_model('/Users/priya/Desktop/final_model/model2/final_model_modified.h5')
cnn_model_1 = tf.keras.models.load_model('/Users/priya/Desktop/final_model/model3_keep_As+old/final_model_modified.h5')


#start GUI
top=tk.Tk()
top.geometry('1400x1200')
top.title('Fruits and Vegetable Image Classifier')
top.configure(background='#a09dbf')
label=Label(top,background='#a09dbf', font=('arial',10,'bold'))
sign_image = Label(top)

def classify_model_2(file_path):
    # classifying given image using model 2
    global label_packed
    global name
    
    img = image.load_img(file_path, target_size = (150, 150))
    array = image.img_to_array(img)
    x = np.expand_dims(array, axis=0)
    vimage = np.vstack([x])
    pred = cnn_model_2.predict(vimage)
    
    result = np.where(pred[0] == 1)
    print(result)
    print("The fruit is: "+classes[result[0][0]])
    name = classes[result[0][0]]

    sign = classes[result[0][0]]
    label.configure(foreground='#011638', text="The given fruit is: " + sign, font=('arial',30, 'bold'))
    label.place(relx=0.69,rely=0.66) 
    show_recipe()

def openNewWindow(name):
     
    # New window for recipe
    newWindow = Toplevel(top)
    
    newWindow.title("Recipe")
    newWindow.configure(background='#a09dbf')
    label=Label(newWindow,background='#a09dbf', font=('arial',10,'bold'))

    label.pack()
 
    # setting the geometry for window
    newWindow.geometry("1000x800")
 
    # Label widget for showing in toplevel
    title = Label(newWindow,
          text ="This is a recipe for: "+ name+"\n", font=('arial',25,'bold'))
    title.configure(background='#a09dbf',foreground='black')
    title.pack()
    
    # api call and displaying information in gui
    # api used: edamam recipe api
    api_id = 'ad0e79fc'
    api_key = '3c3d8e868b0dafc7ded261225a725325'
    api_url = "https://api.edamam.com/search?q="+name+"&app_id="+api_id+"&app_key="+api_key+"&from=0&to=5"
    api_response = requests.get(api_url)
    data = api_response.json()

    recipe_1 = data['hits'][0]
    ingredients = recipe_1['recipe']['ingredientLines']

    if api_response.status_code == 200:
        
        image_url = recipe_1['recipe']['image']

        # Recipe image
        data = urllib.request.urlopen(image_url)
        image = ImageTk.PhotoImage(data=data.read())
        tk.Label(newWindow, image=image).pack()

        # Recipe name
        recipe = Label(newWindow,
          text ="\nRecipe name:", font=('arial',20, 'bold'))
        recipe.configure(background='#a09dbf',foreground='black')
        recipe.pack()

        recipe_title = Label(newWindow,
          text =recipe_1['recipe']['label'], font=('arial',20, 'bold'))
        recipe_title.configure(background='#a09dbf',foreground='purple')
        recipe_title.pack()
        
        # Indredients in recipe
        ing = Label(newWindow,
          text ="\nIngredients:", font=('arial',20, 'bold'))
        ing.configure(background='#a09dbf',foreground='black')
        ing.pack()

        ingredient = Label(newWindow,
          text ='\n'.join(str(x) for x in ingredients), font=('arial',20, 'bold'))
        ingredient.configure(background='#a09dbf',foreground='purple')
        ingredient.pack()

        # Link to see more about recipe
        link = Label(newWindow,
          text ="\nSell full recipe:", font=('arial',20, 'bold'))
        link.configure(background='#a09dbf',foreground='black')
        link.pack()

        full_recipe = Label(newWindow,
          text =recipe_1['recipe']['url'], font=('arial',20, 'bold underline'))
        full_recipe.configure(background='#a09dbf',foreground='purple')
        full_recipe.pack()
        full_recipe.bind("<Button-1>", lambda e: callback(recipe_1['recipe']['url']))
    else:
        print("Recipe not found")
    newWindow.mainloop()

def callback(url):
    # for hyperlink
    webbrowser.open_new(url)

def classify_model_1(file_path):
    # classifying given image using model 1
    global label_packed
    global name
    
    img = image.load_img(file_path, target_size = (150, 150))
    array = image.img_to_array(img)
    x = np.expand_dims(array, axis=0)
    vimage = np.vstack([x])
    pred = cnn_model_1.predict(vimage)
    
    result = np.where(pred[0] == 1)
    print(result)
    print("-----------------------------")
    print("The fruit is: "+classes[result[0][0]])
    print("-----------------------------")
    name = classes[result[0][0]]

    sign = classes[result[0][0]]
    label.configure(foreground='#011638', text="The given fruit is: " + sign, font=('arial',30, 'bold'))
    label.place(relx=0.69,rely=0.66)
    show_recipe()


def show_classify_button_2(file_path):
    classify_b=Button(top,text="Classify Image Using model 2",
    command=lambda: classify_model_2(file_path),padx=10,pady=10)
    classify_b.configure(background='#1f0c38', foreground='black', font=('arial',20,'bold'))
    classify_b.place(relx=0.69,rely=0.56)

def show_classify_button_1(file_path):
    classify_b=Button(top,text="Classify Image Using model 1",
    command=lambda: classify_model_1(file_path),padx=10,pady=10)
    classify_b.configure(background='#1f0c38', foreground='black', font=('arial',20,'bold'))
    classify_b.place(relx=0.69,rely=0.46)

def show_recipe():
    classify_b=Button(top,text="Show Recipe",
    command=lambda: openNewWindow(name),padx=10,pady=10)
    classify_b.configure(background='#1f0c38', foreground='black', font=('arial',20,'bold'))
    classify_b.place(relx=0.69,rely=0.36)

def upload_image():
    # for uploading the image of fruit or vegetable
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),
        (top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')

        # after uploading, we can classify the image using two models
        show_classify_button_1(file_path)
        show_classify_button_2(file_path)
    except:
        pass

# inserting the logo in gui
img= (Image.open("logo.png"))
resized_image= img.resize((180,180), Image.ANTIALIAS)
new_image= ImageTk.PhotoImage(resized_image)
logo_img = Label(image = new_image, borderwidth=0)
logo_img.place(x=-50,y=10)
logo_img.pack(pady=20)

# upload image button
upload=Button(top,text="Upload an image",command=upload_image, bg = 'black', fg = 'black',
  padx=30,pady=15)
upload.configure(background='#1f0c38', foreground='#000000',
    font=('arial',20,'bold'))
upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)

# heading
heading = Label(top, text="Fruit and Vegetable Image Classifier",pady=20, font=('arial',25,'bold'))
heading.configure(background='#a09dbf',foreground='#364156')
heading.pack()

top.mainloop()