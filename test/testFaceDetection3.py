import cv2
import mediapipe as mp
import threading
import csv

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# # For static images:
# IMAGE_FILES = []
# drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
# with mp_face_mesh.FaceMesh(
#     static_image_mode=True,
#     max_num_faces=1,
#     refine_landmarks=True,
#     min_detection_confidence=0.5) as face_mesh:
#   for idx, file in enumerate(IMAGE_FILES):
#     image = cv2.imread(file)
#     # Convert the BGR image to RGB before processing.
#     results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

#     # Print and draw face mesh landmarks on the image.
#     if not results.multi_face_landmarks:
#       continue
#     annotated_image = image.copy()
#     for face_landmarks in results.multi_face_landmarks:
#       print('face_landmarks:', face_landmarks)
#       mp_drawing.draw_landmarks(
#           image=annotated_image,
#           landmark_list=face_landmarks,
#           connections=mp_face_mesh.FACEMESH_TESSELATION,
#           landmark_drawing_spec=None,
#           connection_drawing_spec=mp_drawing_styles
#           .get_default_face_mesh_tesselation_style())
#       mp_drawing.draw_landmarks(
#           image=annotated_image,
#           landmark_list=face_landmarks,
#           connections=mp_face_mesh.FACEMESH_CONTOURS,
#           landmark_drawing_spec=None,
#           connection_drawing_spec=mp_drawing_styles
#           .get_default_face_mesh_contours_style())
#       mp_drawing.draw_landmarks(
#           image=annotated_image,
#           landmark_list=face_landmarks,
#           connections=mp_face_mesh.FACEMESH_IRISES,
#           landmark_drawing_spec=None,
#           connection_drawing_spec=mp_drawing_styles
#           .get_default_face_mesh_iris_connections_style())
#     cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)

# For webcam input:

def testFaceDetection3(camera):
    cap = cv2.VideoCapture()
    print(f"Connecting to camera: {camera['Name']}")
    cap.open(f"rtsp://{camera['username']}:{camera['password']}@{camera['ipAddr']}:{camera['port']}")

    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

# To improve performance, optionally mark the image as not writeable to
# pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)

# Draw the face mesh annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                    )
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                    )
                    
    # Flip the image horizontally for a selfie-view display.
            cv2.imshow(camera['Name'], image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        cap.release
        cv2.destroyAllWindows()
       


with open('cameraList.csv', 'r') as f:
    cameraList = csv.reader(f, delimiter=':', lineterminator='\n')
    cameraDict = {}
    for cameras in cameraList:
        camera = {
            'ipAddr': cameras[1], 
            'port': cameras[2],
            'Name': cameras[3], 
            'username': cameras[4], 
            'password': cameras[5]
        }
        tmp = {}
        for serialNumber in [cameras[0]]:
            tmp["camera_%s" % serialNumber] = camera
            cameraDict.update(tmp)


threads = list()
for cameraNumber, cameraDetails in cameraDict.items():
    th = threading.Thread(target=testFaceDetection3, args=(cameraDetails,))
    threads.append(th)

for th in threads:
    th.start()

for th in threads:
    th.join()