import cv2 as cv
import numpy as np
import os

# Set up the default output directory
output_directory = r"C:\Users\khkha\OneDrive\Bureau\Vehicle-Recognition-System\Deep-Learning"
os.makedirs(output_directory, exist_ok=True)

# Initialize the parameters
confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4  # Non-maximum suppression threshold
inpWidth = 416  # Width of network's input image
inpHeight = 416 # Height of network's input image

# Load class names
classes_file = r"C:\Users\khkha\OneDrive\Bureau\Vehicle-Recognition-System\Deep-Learning\Licence_plate_detection\classes.names"
with open(classes_file, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Load YOLO configuration and weights
model_config = r"C:\Users\khkha\OneDrive\Bureau\Vehicle-Recognition-System\Deep-Learning\Licence_plate_detection\darknet-yolov3.cfg"
model_weights = r"C:\Users\khkha\OneDrive\Bureau\Vehicle-Recognition-System\Deep-Learning\Licence_plate_detection\lapi.weights"
net = cv.dnn.readNetFromDarknet(model_config, model_weights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the indices of the output layers
    output_layers = net.getUnconnectedOutLayers()

    # Handle different formats for output_layers
    try:
        if isinstance(output_layers, np.ndarray):
            output_layers = output_layers.flatten()  # Ensure it's a 1D array
        elif isinstance(output_layers, list):
            output_layers = [item for sublist in output_layers for item in sublist]  # Flatten if needed

        return [layersNames[i - 1] for i in output_layers]  # Adjust indexing to match layer indices
    except Exception as e:
        print(f"Error getting output layers: {e}")
        raise

def drawPred(classId, conf, left, top, right, bottom, frame):
    """Draw the predicted bounding box."""
    global LP_extracted
    LP_extracted = frame[top + 6:bottom - 6, left + 6:right - 6]
    cv.imwrite(os.path.join(output_directory, "Licence_Plate_extracted.jpg"), LP_extracted)

    # Draw bounding box
    cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
    label = f'{classes[classId]}: {conf:.2f}'

    # Display label
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])),
                 (left + round(1.5 * labelSize[0]), top + baseLine), (0, 0, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds = []
    confidences = []
    boxes = []

    # Scan through all the bounding boxes output from the network
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if detection[4] > confThreshold:
                if confidence > confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

    # Perform non-maximum suppression to eliminate redundant overlapping boxes
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    if len(indices) > 0:
        indices = indices.flatten()  # Convert to 1D array if it's 2D
        for i in indices:
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            drawPred(classIds[i], confidences[i], left, top, left + width, top + height, frame)

def LP_detection(image_path):
    # Load the image from the given path
    frame = cv.imread(image_path)

    # Check if the image is loaded correctly
    if frame is None:
        raise FileNotFoundError(f"Could not load the image from {image_path}")

    # Prepare the input blob for the YOLO model
    blob = cv.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
    net.setInput(blob)

    # Perform forward pass to get the outputs
    outs = net.forward(getOutputsNames(net))

    # Post-process the outputs to extract bounding boxes and license plates
    postprocess(frame, outs)

    # Save or return processed image if needed for visualization
    processed_image_path = os.path.join(output_directory, "Processed_Image.jpg")
    cv.imwrite(processed_image_path, frame)

    # Check if 'LP_extracted' was set during post-processing
    if 'LP_extracted' not in globals():
        raise ValueError("No license plate detected.")

    return LP_extracted, frame

# Directly call LP_detection with a hardcoded image path for testing
image_path = r"C:\Users\khkha\OneDrive\Bureau\Vehicle-Recognition-System\Documentation\Screenshots\221.jpg"
LP_detection(image_path)
