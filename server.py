from flask import Flask, jsonify, request, render_template, redirect, send_file
from models import Customer, session
from sphereface import get_embedding
from config import UPLOAD_FOLDER, FACES_FOLDER, EMBEDDINGS_FOLDER

import numpy as np
import os
import cv2

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app = Flask(__name__, template_folder='templates')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['FACES'] = FACES_FOLDER
app.config['EMBEDDINGS'] = EMBEDDINGS_FOLDER


@app.route('/upload', methods=["POST"])
def registration():

    customer_photo = request.files["photo"]
    customer_photo_filename = str(session.query(Customer).count()+1) + ".jpeg"
    print("FILE saved to ", os.path.join(app.config['UPLOAD_FOLDER'], customer_photo_filename))
    customer_photo.save(os.path.join(app.config['UPLOAD_FOLDER'], customer_photo_filename))

    img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], customer_photo_filename))
    face, embedding = get_embedding(img)

    # didn't find face
    if face is None:
        return redirect("/")

    # save face image
    customer_face_filename = str(session.query(Customer).count()+1) + ".jpeg"
    cv2.imwrite(os.path.join(app.config["FACES"], customer_face_filename), face)

    # save embedding
    customer_embedding_filename =  str(session.query(Customer).count()+1) + ".npy"
    np.save(os.path.join(app.config['EMBEDDINGS'], customer_embedding_filename), embedding)

    email = request.form.get('email')
    phone = request.form.get('phone')
    name = request.form.get('name')
    description = request.form.get('description')

    customer = Customer(name, email, phone, customer_face_filename, customer_embedding_filename, description)
    session.add(customer)
    session.commit()

    return redirect("/")


@app.route('/photo_<id>', methods=["GET"])
def customer_photo(id):
    filename = os.path.join(app.config['FACES'], id + ".jpeg")
    return send_file(filename, mimetype="image/jpeg")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3001, debug=True)

