from flask import Flask, render_template, request, redirect
import joblib
from multiprocessing import Process
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

app = Flask(__name__)

# Charger le modèle pré-entraîné
model = joblib.load('src/best_knn_model.pkl')

def extract_contour_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 0)
    _, thresholded_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(thresholded_image, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.erode(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    

    # Somme des coordonnées de tous les points de tous les contours
    sum_x = 0
    sum_y = 0
    for contour in contours:
        sum_x += np.sum(contour[:, :, 0])
        sum_y += np.sum(contour[:, :, 1])

    return sum_x, sum_y

@app.route('/', methods=['GET'])
def home():
 return render_template('home.html')


@app.route("/", methods=['POST'])
def predict():
    imagefile = request.files['image']
    image_path= "src/images/" + imagefile.filename
    imagefile.save(image_path)
   
    
    image = cv2.imread(image_path)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 0)
    _, thresholded_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(thresholded_image, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.erode(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    

    # Dessiner les contours sur l'image originale
    image_with_contours = image.copy()
    cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)

    # Afficher l'image résultante avec les contours
    plt.imshow(cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB))
    plt.title("Contour Detection:")
    # Save the output image
    output_path= "src/static/output/" + imagefile.filename  # Use the same filename for simplicity
    output_path2= "static/output/" + imagefile.filename 
    plt.savefig(output_path)
    print(output_path);
    plt.close()

    
    # Appliquer la prédiction
    x, y = extract_contour_features(image)
    
    features = np.array([[x, y]])

    # Faire la prédiction
    prediction = model.predict(features)  # Assurez-vous d'obtenir la première prédiction (et non un tableau)

    # Appliquer la prédiction ou d'autres opérations sur l'image
    return render_template('home.html', prediction=prediction, image_path=image_path, contour=output_path2)


if __name__ == '__main__':
  
    # Démarrer l'application Flask
    app.run(port=3000, debug=True)
