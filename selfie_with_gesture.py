import cv2
import mediapipe as MP
from libs.coordinates_cv import (
    calculate_coordinates,
    check_if_hand_is_open_and_straight,
    all_fingers_up,
)
from holistic_model import get_fps

cap = cv2.VideoCapture(0)
mpHands = MP.solutions.hands
mp_Holistic = MP.solutions.holistic
model_holistic = mp_Holistic.Holistic(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)
hands = mpHands.Hands()
coord_list = []
id_list = []
Take_photo = True

while cap.isOpened():
    success, img = cap.read()

    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    h, w, c = img.shape
    frame = cv2.resize(img, (800, 600))
    image2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results2 = model_holistic.process(image2)    

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                id_list.append(id)
                cx_cy_list = calculate_coordinates(lm, img, h, w)
                coord_list.append(cx_cy_list)

            coord_dict = dict(zip(id_list, coord_list))
            if (
                check_if_hand_is_open_and_straight(coord_dict) is True
                and all_fingers_up(coord_dict) is True
            ):
                Take_photo = True

    if Take_photo:
        fps = get_fps()
        cv2.putText(
            img,
            f"{int(fps)} FPS",
            (10, 70),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.imwrite("photo.jpg", img)
        Take_photo = False

    cv2.imshow("Image", img)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
