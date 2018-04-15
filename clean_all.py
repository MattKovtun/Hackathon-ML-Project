import os, shutil
from config import UPLOAD_FOLDER, FACES_FOLDER, EMBEDDINGS_FOLDER
from models import session

def clean_folder(folder):
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(e)

for folder in [UPLOAD_FOLDER, FACES_FOLDER, EMBEDDINGS_FOLDER]:
    clean_folder(folder)

session.execute("DROP TABLE customers;")
session.commit()