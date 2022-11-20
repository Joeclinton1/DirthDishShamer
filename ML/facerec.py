import face_recognition
import cv2
import os
import numpy as np

class FaceRec():
    def __init__(self) -> None:
        self.known_face_encodings = []
        self.known_face_names =  ["josh", "joes", "joec", "ming"]
        here = os.path.dirname(__file__)
        for name in self.known_face_names:
            img = face_recognition.load_image_file(os.path.join(here, "../faces/{}prop.jpg".format(name)))
            self.known_face_encodings.append(face_recognition.face_encodings(img)[0])

    def face_rec(self, frame):
        # Initialize some variables
        face_locations = []
        face_encodings = []
        face_names = []

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            face_names.append(name)

        return face_names