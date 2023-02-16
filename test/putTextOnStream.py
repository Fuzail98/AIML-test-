import cv2

def putTextOnStream(image, fps, result):
    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(
        image, 
        f'fps = {int(fps)}', 
        (20, 100), 
        font, 1, 
        (255, 0, 0), 2
    )
    cv2.putText(
        image,
        f"Emo:{result[0]['dominant_emotion']}",
        (20, 200),
        font, 1,
        (0, 0, 255), 2
    )
    cv2.putText(
        image,
        f"Gen:{result[0]['dominant_gender']}",
        (20, 300),
        font, 1,
        (0, 0, 255), 2
    )
    cv2.putText(
        image,
        f"Age:{result[0]['age']}",
        (20, 400),
        font, 3,
        (0, 0, 255), 2
    )
