"""
Face detection on an image using multi-task cascaded convolutional networks (MTCNN)
"""
import argparse

import cv2
from facenet_pytorch import MTCNN

class FaceDetector(object):
    """
    Create a face detector class
    Args:
        object (_type_): _description_
    """
    def __init__(self, mtcnn):
        self.mtcnn = mtcnn
        
    def _draw(self, image, faces, probs):
        """
        Draw landmarks and boxes around each detected face
        Args:
            image (cv image): _description_
            faces (list(list(floats))): _description_
            probs (list(floats)): _description_
        """
        try:
            for face, prob in zip(faces, probs):
                if prob > 0.9:
                    # Draw rectangle around detected face and show confidence
                    cv2.rectangle(image, 
                                (int(face[0]), int(face[1])), 
                                (int(face[2]), int(face[3])), 
                                (0, 0, 255), 
                                thickness=2
                                )
                    prob_str = str(round(prob*100, 2))  # convert confidence to string to display 
                    cv2.putText(image, prob_str, 
                            (int(face[2]), int(face[3])), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        except:
            pass
        
        return image
        
    def run(self, image):
        """
        Run the facedetector and draw boxes around detected faces 
        Returns:
            _type_: _description_
        """
        try:
            # detect faces and their probabilities respectively
            faces, probs = self.mtcnn.detect(image)
            # draw on image
            self._draw(image, faces, probs)
        except:
            pass
        
        # Show the image with detected faces
        cv2.imshow('Face Detection', image)
        cv2.waitKey(0) 
        cv2.destroyAllWindows()
        
# Run the app
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Input image path")
args = vars(ap.parse_args())

# Load input image
ip_image = cv2.imread(args["image"])

print("[INFO] Loading MTCNN model...")
mtcnn = MTCNN()
face_detector = FaceDetector(mtcnn)
face_detector.run(ip_image)
    
            
            