from flask import Flask, Response
import cv2
import numpy as np
import time

app = Flask(__name__)

@app.route('/invisibility_cloak')
def invisibility_cloak():
    cap = cv2.VideoCapture(0)
    time.sleep(3)

    background = 0
    for i in range(30):
        ret, background = cap.read()

    background = np.flip(background, axis = 1)

    def generate():
        while cap.isOpened():
            ret, img = cap.read()
            if not ret:
                break
            img = np.flip(img, axis=1)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            value = (35, 35)
            blurred = cv2.GaussianBlur(hsv, value, 0)

            lower_red1 = np.array([0, 120, 70])
            upper_red1 = np.array([10, 255, 255])
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

            lower_red2 = np.array([170, 120, 70])
            upper_red2 = np.array([180, 255, 255])
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

            mask = mask1 + mask2
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

            img[np.where(mask == 255)] = background[np.where(mask == 255)]
            _, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
