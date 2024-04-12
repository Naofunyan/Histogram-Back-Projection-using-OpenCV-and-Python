import cv2
import numpy as np

target = cv2.imread('Sample3.png')
cap = cv2.VideoCapture('Lab2c.mp4')

output_width, output_height = 640, 480
output_fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 30.0
output_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video_mp4', output_fourcc, output_fps, (output_width, output_height))

hsvt = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)

M = cv2.calcHist([hsvt], [0, 1], None, [180, 256], [0, 180, 0, 256])
I = cv2.calcHist([hsvt], [0, 1], None, [180, 256], [0, 180, 0, 256])
R = M / (I + 1)

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (output_width, output_height))
    hsvr = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    B = R[hsvr[:,:,0], hsvr[:,:,1]]
    B = np.minimum(B, 1)
    B = B.reshape(hsvr.shape[:2])
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cv2.filter2D(B, -1, disc, B)
    B = np.uint8(B * 255)

    thresh = cv2.threshold(B, 50, 255, 0)[1]
    thresh_3ch = cv2.merge([thresh, thresh, thresh]).astype(np.uint8)

    black_image = np.zeros_like(frame)
    target = cv2.bitwise_and(frame, frame, mask=thresh)

    res = cv2.add(black_image, target)
    out.write(res)
    cv2.imshow('Output', res)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
