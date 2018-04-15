import torch
from torch.autograd import Variable

from matlab_cp2tform import get_similarity_transform_for_cv2
import net_sphere
import numpy as np
import cv2
import time

from mtcnn.mtcnn import MTCNN

torch.set_num_threads(8)

MODEL = 'sphere20a'
MODEL_PATH = 'sphere20a.pth'


def alignment(src_img, src_pts):
    min_y = min(src_pts["keypoints"]["left_eye"][1], src_pts["keypoints"]["right_eye"][1])
    max_y = max(src_pts["keypoints"]["mouth_left"][1], src_pts["keypoints"]["mouth_right"][1])
    min_x = min(src_pts["keypoints"]["left_eye"][0], src_pts["keypoints"]["mouth_left"][0])
    max_x = max(src_pts["keypoints"]["right_eye"][0], src_pts["keypoints"]["mouth_right"][0])
    # print(src_pts["keypoints"]["right_eye"])
    height = max_y - min_y
    width = max_x - min_x

    db_img = src_img[max(0, int(min_y-height*1.8)): min(src_img.shape[0], int(max_y+height*1.5)),
             max(0, int(min_x-width*1.2)): min(src_img.shape[1], int(max_x+width*1.2))]

    #cv2.imshow("windows", db_img)
    #cv2.waitKey(0)
    ref_pts = [[30.2946, 51.6963], [65.5318, 51.5014],
               [48.0252, 71.7366], [33.5493, 92.3655], [62.7299, 92.2041]]
    crop_size = (96, 112)

    src_pts = [src_pts["keypoints"]["left_eye"], src_pts["keypoints"]["right_eye"], src_pts["keypoints"]["nose"],
               src_pts["keypoints"]["mouth_left"], src_pts["keypoints"]["mouth_right"]]
    src_pts = np.array(src_pts)

    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)

    tfm = get_similarity_transform_for_cv2(s, r)
    face_img = cv2.warpAffine(src_img, tfm, crop_size)
    return face_img, db_img

def get_embeddings(img):

    start = time.time()
    pts = detector.detect_faces(img)
    end = time.time()

    print(end-start)

    imgs = []
    processed_imgs = []
    if not pts:
        return None

    for pt in pts:
        face, db_img = alignment(img, pt)
        imgs.append(db_img)
        processed_img = face.transpose(2, 0, 1).reshape((1, 3, 112, 96))
        processed_img = (processed_img - 127.5) / 128.0
        processed_imgs.append(processed_img)

    processed_img = np.vstack(processed_imgs)
    processed_img = Variable(torch.from_numpy(processed_img).float(), volatile=True)
    output = net(processed_img)
    embeddings = output.data.numpy()
    print(len(imgs), len(embeddings))
    return list(zip(imgs, embeddings))


def get_embedding(img):
    embeddings = get_embeddings(img)
    if embeddings is None:
        return None, None
    return embeddings[0]


def compare_embeddings(embedding, embeddings):
    embedding_norm = np.linalg.norm(embedding)
    divider = (np.linalg.norm(embeddings, axis=1) * embedding_norm + 1e-5)[:, None]
    cosdistances = np.dot(embeddings / divider, embedding)
    return cosdistances

detector = MTCNN()
net = getattr(net_sphere, 'sphere20a')()
net.load_state_dict(torch.load('sphere20a.pth'))
net.eval()
net.feature = True

if __name__ == "__main__":
    faces = ["oles3.jpg", "anton1.jpg", "anton3.jpg", "matt1.jpg", "matt2.jpg", "oles1.jpg", "oles2.jpg", "oles3.jpg", "oleh1.jpg", "oleh2.jpg", "harry2.jpg", "harry3.jpg"]
    emebeddings = []
    #for f in faces:
    #    print(f)
    img, e = get_embedding(cv2.imread("static/upload/5.jpeg"))
    emebeddings.append(e)
    #    break

#    for i, f in enumerate(faces):
#        print(f, end="\t")
#        for j, f in enumerate(faces):
#            if emebeddings[i] is None or emebeddings[j] is None:
#                print("---", end="\t")
#                continue
#            cosdistance = compare_embeddings(emebeddings[i], emebeddings[j])
#            print(round(cosdistance, 3), end="\t")
#        print()
    #img1, e1 = get_embedding("old/images/daniel-radcliffe_2.jpg")
    #img2, e2 = get_embedding("old/images/daniel_radcliffe.jpg")
    #cosdistance = compare_embeddings(e1, e2)
    #print(cosdistance)