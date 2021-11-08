import cv2


def convert_gray_to_rgb(image_path, output_name):
    """
    Converts an image to rgb grayscale
    """
    gray = cv2.imread(image_path, 0)
    backtorgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(output_name, backtorgb)
