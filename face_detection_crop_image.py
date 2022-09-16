from mtcnn import MTCNN
def crop_image(img):
    detector=MTCNN()
    data=detector.detect_faces(img)   # detect_face function, gives box, keypoints of a face
    for face in data:
        box=face['box']               # box give the x,y co-ordinate of bottom_left face along with height and width of a human face
    box[0]= 0 if box[0]<0 else box[0]
    box[1]= 0 if box[1]<0 else box[1]
    img=img[box[1]: box[1]+box[3],box[0]: box[0]+ box[2]]
    return (True, img) 
