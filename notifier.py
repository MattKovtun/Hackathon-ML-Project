
import cv2
import os
import requests
import numpy as np
from threading import Thread

from models import Customer, session
from sphereface import get_embeddings, compare_embeddings
from config import EMBEDDINGS_FOLDER, EMBEDDINGS_TRESHOLD

FPS = 10
# [embedding, id]
embeddings_ids = [filename.split('.')[0] for filename in os.listdir(EMBEDDINGS_FOLDER)]
embeddings = np.array([np.load(os.path.join(EMBEDDINGS_FOLDER, i + ".npy")) for i in embeddings_ids])


class ImgProcessor(Thread):

    def __init__(self, img):
        Thread.__init__(self)
        self.img = img

    def run(self):
        imgs_and_embeddings = get_embeddings(self.img)
        if imgs_and_embeddings is None:
            print("NOT FACE AT ALL!")
            return

        for i, (img, embedding) in enumerate(imgs_and_embeddings):
            cosdistances = compare_embeddings(embedding, embeddings)
            idx = np.argmax(cosdistances)

            if cosdistances[idx] < EMBEDDINGS_TRESHOLD:
                d = { "type": "IN"}
                print(d, cosdistances[idx])
                r = requests.post("http://localhost:3000/detect", json=d)
                print(r.status_code)
            else:
                customer_id = embeddings_ids[idx]
                customer = session.query(Customer).filter_by(id=customer_id).first()
                d = { "type": "IN", "user": {"id":customer_id, "name": customer.name, "picture": "http://localhost:3001/photo_"+customer_id,
                                                "description": customer.description, "preferences": customer.preferences}}
                print(d, cosdistances[idx])
                r = requests.post("http://localhost:3000/detect", json=d)
                print(r.status_code)

camera = cv2.VideoCapture(0)

exit = False

while not exit:

    # we work with each 10th frame
    for i in range(FPS):
        ret, frame = camera.read()
        cv2.imshow("camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit = True
            break

    if len(embeddings_ids) != len(os.listdir(EMBEDDINGS_FOLDER)):
        embeddings_ids = [filename.split('.')[0] for filename in os.listdir(EMBEDDINGS_FOLDER)]
        embeddings = np.array([np.load(os.path.join(EMBEDDINGS_FOLDER, i + ".npy")) for i in embeddings_ids])

    if len(embeddings_ids) > 0 :
        thread = ImgProcessor(frame)
        thread.start()



camera.release()
cv2.destroyAllWindows()