import mediapipe as MP
import cv2
import time

mp_drawing = MP.solutions.drawing_utils
mp_holistic = MP.solutions.holistic
model_holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)
capture = cv2.VideoCapture(0)


def get_holisitc_model(results, image):
    # Facial landmarks
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
    )

    # Draw Right Hand Land Marks
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS
    )

    # Draw Left Hand Land Marks
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS
    )


def get_fps():
    cT, pT = 0, 0
    cT = time.time()
    fps = 1 / (cT - pT)
    pT = cT
    return fps


def output_holistic_model(image, results):
    fps = get_fps()
    get_holisitc_model(results, image)
    cv2.putText(
        image,
        f"{int(fps)} FPS",
        (10, 70),
        cv2.FONT_HERSHEY_COMPLEX,
        1,
        (0, 255, 0),
        2,
    )
    cv2.imshow("Landmarks on face and hands based on holistic model", image)


while capture.isOpened():
    success, frame = capture.read()    
    frame = cv2.resize(frame, (800, 600))
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model_holistic.process(image)
    image.flags.writeable = True

    # Converting back the RGB image to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    output_holistic_model(image, results)