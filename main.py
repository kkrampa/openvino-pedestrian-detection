import time

import cv2 as cv

from detector import PedestrianDetector


def detect_pedestrians(image):
    print(image.shape)
    image = cv.resize(image, (544, 320))
    return image


def process_video(video_path):
    detector = PedestrianDetector()
    video = cv.VideoCapture(video_path)

    if not video.isOpened():
        exit(-1)

    color = (0, 0, 255)
    while True:
        _, image = video.read()
        x_scale = image.shape[1]
        y_scale = image.shape[0]
        if image is not None:
            result = detector.detect(image)
            for (image_id, label, conf, x_min, y_min, x_max, y_max) in result[0][0]:
                if label != 0 and conf > 0.2:
                    x = (int(x_min * x_scale), int(y_min * y_scale))
                    y = (int(x_max * x_scale), int(y_max * y_scale))
                    image = cv.rectangle(image, x, y, color)
            cv.imshow('image', image)
            if cv.waitKey(22) & 0xFF == ord('q'):
                break
        else:
            break
    video.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    process_video('http://www.robots.ox.ac.uk/ActiveVision/Research/Projects/2009bbenfold_headpose/Datasets/TownCentreXVID.avi')
