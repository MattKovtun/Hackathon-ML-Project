import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN


def main():
    cap = cv2.VideoCapture(0)
    detector = MTCNN()

    while(True):
        ret, frame = cap.read()
        faces = detector.detect_faces(frame)
        # TODO: do something if there is more not one faces
        if len(faces) == 1:
            box = faces[0]["box"]
            face = frame[box[1]:box[1] + box[3],box[0]:box[0] + box[2]]
            cv2.imwrite('data/oleh1.jpg', face)
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()