import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# Set up the default output directory
output_directory = r"C:\Users\khkha\OneDrive\Bureau\Vehicle-Recognition-System\Deep-Learning"
os.makedirs(output_directory, exist_ok=True)

# Initialize the parameters for detection
confThreshold = 0.5
nmsThreshold = 0.4
inpWidth = 416
inpHeight = 416

# Load class names
classes_file = r"C:\Users\khkha\OneDrive\Bureau\Vehicle-Recognition-System\Deep-Learning\Licence_plate_detection\classes.names"
with open(classes_file, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Load YOLO configuration and weights
model_config = r"C:\Users\khkha\OneDrive\Bureau\Vehicle-Recognition-System\Deep-Learning\Licence_plate_detection\darknet-yolov3.cfg"
model_weights = r"C:\Users\khkha\OneDrive\Bureau\Vehicle-Recognition-System\Deep-Learning\Licence_plate_detection\lapi.weights"
net = cv2.dnn.readNetFromDarknet(model_config, model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def getOutputsNames(net):
    layersNames = net.getLayerNames()
    output_layers = net.getUnconnectedOutLayers()
    if isinstance(output_layers, np.ndarray):
        output_layers = output_layers.flatten()
    elif isinstance(output_layers, list):
        output_layers = [item for sublist in output_layers for item in sublist]
    return [layersNames[i - 1] for i in output_layers]


def drawPred(classId, conf, left, top, right, bottom, frame):
    global LP_extracted
    LP_extracted = frame[top + 6:bottom - 6, left + 6:right - 6]
    cv2.imwrite(os.path.join(output_directory, "Licence_Plate_extracted.jpg"), LP_extracted)
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
    label = f'{classes[classId]}: {conf:.2f}'
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(frame, (left, top - round(1.5 * labelSize[1])),
                  (left + round(1.5 * labelSize[0]), top + baseLine), (0, 0, 255), cv2.FILLED)
    cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)


def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if detection[4] > confThreshold and confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    if len(indices) > 0:
        indices = indices.flatten()
        for i in indices:
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            drawPred(classIds[i], confidences[i], left, top, left + width, top + height, frame)


def histogram_of_pixel_projection(img):
    caracrter_list_image = list()
    BLACK = [0, 0, 0]
    img = cv2.copyMakeBorder(img, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=BLACK)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    nb = np.array(gray)
    nb[nb > 120] = 255
    nb[nb < 120] = 0
    x_sum = cv2.reduce(nb, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S)
    y_sum = cv2.reduce(nb, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S)
    x_sum = x_sum.transpose()
    x = gray.shape[1]
    y = gray.shape[0]
    x_sum = x_sum / y
    y_sum = y_sum / x
    x_arr = np.arange(x)
    y_arr = np.arange(y)
    z = np.array(x_sum)
    w = np.array(y_sum)
    z[z < 15] = 0
    z[z > 15] = 1
    w[w < 20] = 0
    w[w > 20] = 1
    test = z.transpose() * nb
    test = w * test
    f = 0
    ff = z[0]
    t1 = list()
    t2 = list()
    for i in range(z.size):
        if z[i] != ff:
            f += 1
            ff = z[i]
            t1.append(i)
    rect_h = np.array(t1)
    f = 0
    ff = w[0]
    for i in range(w.size):
        if w[i] != ff:
            f += 1
            ff = w[i]
            t2.append(i)
    rect_v = np.array(t2)
    rectv = []
    rectv.append(rect_v[0])
    rectv.append(rect_v[1])
    max_diff = int(rect_v[1]) - int(rect_v[0])
    for i in range(len(rect_v) - 1):
        diff2 = int(rect_v[i + 1]) - int(rect_v[i])
        if diff2 > max_diff:
            rectv[0] = rect_v[i]
            rectv[1] = rect_v[i + 1]
            max_diff = diff2
    for i in range(len(rect_h) - 1):
        diff1 = int(rect_h[i + 1]) - int(rect_h[i])
        if (diff1 > 5) and (z[rect_h[i]] == 1):
            caracrter_list_image.append(nb[int(rectv[0]):int(rectv[1]), rect_h[i]:rect_h[i + 1]])
            cv2.rectangle(img, (rect_h[i], rectv[0]), (rect_h[i + 1], rectv[1]), (0, 255, 0), 1)
    return caracrter_list_image


def fix_dimension(img):
    new_img = np.zeros((28, 28, 3))
    for i in range(3):
        new_img[:, :, i] = img
    return new_img


model = load_model(
    r'C:\Users\khkha\OneDrive\Bureau\Vehicle-Recognition-System\Deep-Learning\Licence_Plate_Recognition\ocrmodel.h5')


def show_results(char):
    dic = {}
    characters = '0123456789T'
    for i, c in enumerate(characters):
        dic[i] = c
    output = []
    for i, ch in enumerate(char):
        img_ = cv2.resize(ch, (28, 28))
        img = fix_dimension(img_)
        img = img.reshape(1, 28, 28, 3)
        y_ = np.argmax(model.predict(img), axis=-1)[0]
        character = dic[y_]
        if (character == "T"):
            output.append("Tunisie")
        else:
            output.append(character)
    plate_number = ''.join(output)
    return output


def draw_text_on_image(img, title, text, x=30, y=50, w=300, h=100):
    sub_img = img[y:y + h, x:x + w]
    white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
    res = cv2.addWeighted(sub_img, 0.6, white_rect, 0.5, 0)
    img[y:y + h, x:x + w] = res
    cv2.putText(img, title, (x + 10, y + 30), cv2.FONT_HERSHEY_DUPLEX, 1, (128, 190, 82), 2)
    cv2.putText(img, text, (x + 10, y + 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (118, 82, 26), 2)
    return img


def process_frame(frame):
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    outs = net.forward(getOutputsNames(net))
    postprocess(frame, outs)
    if 'LP_extracted' in globals():
        plate_image = LP_extracted
        character_images = histogram_of_pixel_projection(plate_image)
        recognized_characters = show_results(character_images)
        plate_number = ''.join(recognized_characters)
        draw_text_on_image(frame, 'Plate Number', plate_number, 30, 30)
    return frame


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)  # Open the default webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        processed_frame = process_frame(frame)
        cv2.imshow("Real-Time License Plate Recognition", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()
