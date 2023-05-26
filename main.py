from tensorflow import keras
import tkinter as tk
from tkinter import *
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import os
import tensorflow as tf

# Get the current directory
current_directory = os.path.dirname(os.path.realpath(__file__))

# Create the full path to the model file
model_path = os.path.join(current_directory, "my_deneme_model.h5")

# Load the model
model = tf.keras.models.load_model(model_path)


window = Tk()

window.title("Predicting Number")

# Tuvali oluştur
canvas = Canvas(window, width=280, height=280, bg='white')
canvas.grid(row=0, column=0, pady=2, sticky=W)

# ImageDraw ile tuvali takip edecek bir image oluştur
img = Image.new('L', (280, 280), 'white')
draw = ImageDraw.Draw(img)

def draw_digit(event):
    x = event.x
    y = event.y
    r = 5  # radius
    canvas.create_oval(x-r, y-r, x+r, y+r, fill='black')
    draw.ellipse([(x-r, y-r), (x+r, y+r)], 'black')

canvas.bind('<B1-Motion>', draw_digit)

def predict_digit():
    # Yüksek çözünürlüklü resmi MNIST boyutuna (28,28) dönüştür
    img_resized = img.resize((28,28))
    img_inverted = ImageOps.invert(img_resized)

    # Resmi model için uygun bir numpy dizisine dönüştür
    img_array = np.array(img_inverted).reshape(1, 28, 28, 1)
    img_array = img_array / 255.0

    # Modeli kullanarak tahmin yap
    prediction = model.predict(img_array)
    label = np.argmax(prediction)

    # Tahmini ekrana yazdır
    label_text.config(text=str(label))

    # Çizim alanını temizle
    canvas.delete('all')
    draw.rectangle([(0, 0), (280, 280)], fill='white')


button = Button(window, text='Predict!', command=predict_digit)
button.grid(row=1, column=0, pady=2)

label_text = Label(window)
label_text.grid(row=2, column=0, pady=2)

# GUI'yi başlat
window.mainloop()
