import cv2

id_list = list(range(21))
id_dict = {
    "wrist": id_list[0],
    "thumb": id_list[1:5],
    "index_finger": id_list[5:9],
    "middle_finger": id_list[9:13],
    "ring_finger": id_list[13:17],
    "pinky_finger": id_list[17:21],
}


def calculate_coordinates(landmark, image, height, width):
    x_cor = landmark.x * width
    y_cor = landmark.y * height
    cv2.circle(image, (int(x_cor), int(y_cor)), 4, (255, 255, 255), cv2.FILLED)
    return [x_cor, y_cor]


def get_coordinates_from_id(self, dict_coordinates, id):
    coordinates_arr = list(dict_coordinates[id])
    x_value = coordinates_arr[0]
    y_value = coordinates_arr[1]
    return x_value, y_value


def verify_thumps_up(coordinate_dict):
    return verify_if_the_fingers_are_up(id_dict["thumb"], coordinate_dict)


def index_finger_up(coordinate_dict):
    return verify_if_the_fingers_are_up(id_dict["index_finger"], coordinate_dict)


def middle_finger_up(coordinate_dict):
    return verify_if_the_fingers_are_up(id_dict["middle_finger"], coordinate_dict)


def ring_finger_up(coordinate_dict):
    return verify_if_the_fingers_are_up(id_dict["ring_finger"], coordinate_dict)


def pinky_finger_up(coordinate_dict):
    return verify_if_the_fingers_are_up(id_dict["pinky_finger"], coordinate_dict)


def verify_if_the_fingers_are_up(index_arr, coordinate_dict):
    y_arr = []
    for id in index_arr:
        coordinate_list = list(coordinate_dict[id])
        cy = coordinate_list[1]
        y_arr.append(cy)
    return all(y_arr[i] >= y_arr[i + 1] for i in range(len(y_arr) - 1))


def all_fingers_up(coordinate_dict):
    return (
        verify_thumps_up(coordinate_dict) is True
        and index_finger_up(coordinate_dict) is True
        and middle_finger_up(coordinate_dict) is True
        and ring_finger_up(coordinate_dict) is True
        and pinky_finger_up(coordinate_dict) is True
    )


def check_if_hand_is_open_and_straight(coordinate_dict):
    c_y_pip_arr = []
    id_arr = [
        0,
        3,
        6,
        10,
        14,
        19,
    ]  # comparing the pip of every finger are above the wrist
    for id in id_arr:
        c_pip_list = list(coordinate_dict[id])
        c_y_pip_arr.append(c_pip_list[1])
    return min(c_y_pip_arr) < c_y_pip_arr[0]
