from mtcnn import MTCNN
import numpy as np
import face_detection_rotate

def align_image(img):
    detector=MTCNN()
    data=detector.detect_faces(img)    # detect_face function, gives box, keypoints of a face
    for face in data:
        box=face['box']                # box give the x,y co-ordinate of bottom_left face along with height and width of a human face
        keypoints=face['keypoints']    # 5 key points such as left_eye, right_eye, left_mouth, right_mouth, nose
        left_eye=keypoints['left_eye']
        right_eye=keypoints['right_eye'] 
        lx,ly=left_eye        
        rx,ry=right_eye
        dx=rx-lx
        dy=ry-ly
        tan=dy/dx
        theta=np.arctan(tan)
        theta=np.degrees(theta)          # theta determines in which angle rotation will be performed
        img=face_detection_rotate.rotate_bound(img, theta)
        return (True,img)