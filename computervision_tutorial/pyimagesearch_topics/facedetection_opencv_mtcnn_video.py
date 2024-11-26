"""
Face detection on a live webcam feed using multi-task cascaded convolutional networks (MTCNN)
"""
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
        
    def _draw(self, frame, faces, probs):
        """
        Draw landmarks and boxes around each detected face
        Args:
            frame (_type_): _description_
            faces (list(list(floats))): _description_
            probs (list(floats)): _description_
        """
        try:
            for face, prob in zip(faces, probs):
                if prob > 0.9: 
                    # Draw rectangle around detected face and show confidence
                    cv2.rectangle(frame, 
                                    (int(face[0]), int(face[1])), 
                                    (int(face[2]), int(face[3])), 
                                    (0, 0, 255), 
                                    thickness=2
                                    )
                    prob_str = str(round(prob*100, 2))  # convert confidence to string to display
                    cv2.putText(frame, prob_str, 
                            (int(face[2]), int(face[3])), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        except:
            pass
        
        return frame
        
    def run(self):
        """
        Run the facedetector and draw boxes around detected faces 
        Returns:
            _type_: _description_
        """
        capture = cv2.VideoCapture(0)
        
        while True:
            _, frame = capture.read()
            try:
                # detect faces and their probabilities respectively
                boxes, probs = self.mtcnn.detect(frame)
                # draw on frame
                self._draw(frame, boxes, probs)
            except:
                pass
            
            # Show the frame with detected faces
            cv2.imshow('Face Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        capture.release()
        cv2.destroyAllWindows()
        
# Run the app
print("[INFO] Loading MTCNN model...")
mtcnn = MTCNN()

# Initialize the video stream and allow the camera senson to warm up
face_detector = FaceDetector(mtcnn)
face_detector.run()
    
            
            