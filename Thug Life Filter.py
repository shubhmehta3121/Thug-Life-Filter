import cv2
import math
import cvzone
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
previous_time = 0
angle_deg=0
mp_drawing = mp.solutions.drawing_utils
mp_drawing_style = mp.solutions. drawing_styles
landmark_drawing_spec = mp_drawing.DrawingSpec(color=(0,0,255),thickness=1,circle_radius=1)
connection_drawing_spec = mp_drawing.DrawingSpec(color=(0,0,255))
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=2,refine_landmarks=True)

image_front_glass = cv2.imread('glasses.png',cv2.IMREAD_UNCHANGED)
image_front_glass = cv2.cvtColor(image_front_glass,cv2.COLOR_BGR2BGRA)
image_front_weed = cv2.imread('weed.png',cv2.IMREAD_UNCHANGED)
image_front_weed = cv2.cvtColor(image_front_weed,cv2.COLOR_BGR2BGRA)
image_front_chain = cv2.imread('chain1.png',cv2.IMREAD_UNCHANGED)
image_front_chain = cv2.cvtColor(image_front_chain,cv2.COLOR_BGR2BGRA)
image_front_cap= cv2.imread('cap.png',cv2.IMREAD_UNCHANGED)
image_front_cap = cv2.cvtColor(image_front_cap,cv2.COLOR_BGR2BGRA)
image_glass_scaling_factor=1.25
image_chain_width_scaling_factor=1.1
image_chain_height_scaling_factor=1.28
image_cap_width_scaling_factor=1.65
image_cap_height_scaling_factor=0.65

def length(p1,p2):
    x1,y1=p1[0],p1[1]
    x2,y2=p2[0],p2[1]
    dist = ((x2-x1)**2+(y2-y1)**2)**(0.5)
    return dist

def angle(p1,p2):
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]
    dx=x2-x1
    dy=y2-y1
    angle_rad = math.atan2(dy,dx)
    angle_deg = math.degrees(angle_rad)
    if angle_deg<0:
        angle_deg+=360
    return angle_deg
while True:
    success, frame = cap.read()
    frame=cv2.flip(frame,1)
    current_time = time.time()
    fps = 1/(current_time-previous_time)
    previous_time = current_time
    if not success:
        break
    frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    all_landmarks = results.multi_face_landmarks
    if all_landmarks: #1
        faces=[]
        for face_landmarks in all_landmarks:#2
            face = []
            for id,individual_landmark in enumerate(face_landmarks.landmark):
                frame_height,frame_width,frame_channel = frame.shape
                x=int(individual_landmark.x*frame_width)
                y=int(individual_landmark.y*frame_height)
                face.append([x,y])#3
            faces.append(face)#4
        for individual_face in faces:
            left_eye = individual_face[156]
            right_eye = individual_face[383]
            angle_deg =  angle(left_eye, right_eye)
            if (angle_deg < 35) or (angle_deg > 335):

                # glasses
                glass_width = image_front_glass.shape[1]
                glass_height = image_front_glass.shape[0]
                lr_dist = length(left_eye, right_eye)
                new_glass_width = math.ceil(lr_dist * image_glass_scaling_factor)
                new_glass_height = math.ceil(image_glass_scaling_factor * glass_height * (new_glass_width / glass_width))
                new_glass_front = cv2.resize(image_front_glass, (new_glass_width, new_glass_height))
                center = (new_glass_front.shape[1] // 2, new_glass_front.shape[0] // 2)
                cv2.putText(frame, f'ANGLE : { int(angle_deg)}', (50, 80), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
                rotation_matrix = cv2.getRotationMatrix2D(center, (-1*angle_deg) % 360, scale=1)
                rotated_image = cv2.warpAffine(new_glass_front, rotation_matrix,(new_glass_front.shape[1], new_glass_front.shape[0]))
                new_rotated_image = cv2.resize(rotated_image, (new_glass_width, new_glass_height))
                xminus = 48 * (new_glass_width / glass_width)
                yminus = 219 * (new_glass_height / glass_height)
                frame = cvzone.overlayPNG(frame, rotated_image,(left_eye[0] - math.ceil(xminus), left_eye[1] - math.ceil(yminus)))

            #chain
                top_head = individual_face[10]
                bottom_head = individual_face[152]
                face_v_dist = length(top_head, bottom_head)
                chain_left = individual_face[93]
                chain_right = individual_face[323]
                chain_dist = length(chain_left,chain_right)
                new_chain_width = math.ceil(chain_dist*image_chain_width_scaling_factor)
                new_chain_height=math.ceil(new_chain_width*image_chain_height_scaling_factor)
                new_image_front_chain = cv2.resize(image_front_chain,(new_chain_width,new_chain_height))
                yplus=math.ceil((80*face_v_dist)/132)
                frame = cvzone.overlayPNG(frame, new_image_front_chain,(chain_left[0],chain_left[1]+yplus))

            #weed
            mouth = individual_face[14]
            head_top = individual_face[10]
            head_bottom = individual_face[152]
            face_dist = length(head_bottom,head_top)
            new_image_front_weed = cv2.resize(image_front_weed,(math.ceil(face_dist*0.4),math.ceil(face_dist*0.4)))
            frame = cvzone.overlayPNG(frame,new_image_front_weed,(mouth[0]-new_image_front_weed.shape[1],mouth[1]))

            # #cap
            # left_head = individual_face[21]
            # right_head=individual_face[251]
            # top_head=individual_face[10]
            # head_dist = length(left_head,right_head)
            # cap_width = image_front_cap.shape[1]
            # cap_height=image_front_cap.shape[0]
            # new_cap_width = math.ceil(head_dist * image_cap_width_scaling_factor)
            # new_cap_height=math.ceil(new_cap_width*image_cap_height_scaling_factor)
            # new_image_front_cap = cv2.resize(image_front_cap,(new_cap_width,new_cap_height))
            # xminus = 94*(new_cap_width/cap_width)+17
            # yminus=880*(new_cap_height/cap_height)-7
            # frame = cvzone.overlayPNG(frame,new_image_front_cap,(left_head[0]-math.ceil(xminus),top_head[1]-math.ceil(yminus)))

    cv2.putText(frame,f'FPS : {int(fps)}',(50,50),cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),3)
    cv2.imshow('Screen',frame)
    cv2.waitKey(1)
"""
1
this is a list of 468 dictionaries (not dicts, but NormalizedLandmarkLists), 1 for each landmark. 1 list is for 1 persons face landmarks
all landmarks looks like this [[468 objects for 1 face],[468 objects for 2nd face],....]
"""
"""
2
face landmarks = [landmark 1 {x: ,y: ,z: }, landmark 2 {x: ,y: ,z: }, .....]
face_landmarks is a normalized Landmark List with 468 landmarks
"""
"""
3
468 landmarks appended to the face list which we can extract from their position
"""
"""
4            
a list of 468 landmarks in from of pixels appended for 1 face
"""
