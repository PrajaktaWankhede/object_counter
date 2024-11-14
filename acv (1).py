import cv2
import numpy as np
import streamlit as st

# Paths to your YOLO files
yolo_cfg_path = r"C:\Users\praja\Desktop\c\yolo_files\yolov3.cfg"  # Update with your path
yolo_weights_path = r"C:\Users\praja\Desktop\c\yolo_files\yolov3.weights"  # Update with your path
coco_names_path = r"C:\Users\praja\Desktop\c\yolo_files\coco.names"  # Update with your path

# Load YOLO model
def load_yolo_model():
    # Load YOLO network
    net = cv2.dnn.readNet(yolo_weights_path, yolo_cfg_path)
    
    # Load the class names (COCO dataset)
    try:
        with open(coco_names_path, "r") as f:
            classes = f.read().strip().split("\n")
    except FileNotFoundError:
        st.error(f"coco.names file not found at {coco_names_path}")
        return None, None, None
    
    # Get YOLO layer names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    return net, output_layers, classes

# Object detection and counting
def detect_objects(image, net, output_layers):
    height, width, channels = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    
    # Process detections
    for out in outputs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # You can adjust the threshold here
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Coordinates for the bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Non-maximum Suppression (to remove duplicate boxes)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    return class_ids, boxes, indices

# Displaying the results
def run_app():
    st.title("Object Detection with Count")

    # Load YOLO model
    net, output_layers, classes = load_yolo_model()
    if not net:
        return
    
    uploaded_image = st.file_uploader("Choose an image...", type="jpg")
    
    if uploaded_image is not None:
        # Read the uploaded image
        image = np.array(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        class_ids, boxes, indices = detect_objects(image, net, output_layers)

        # Draw bounding boxes and labels
        count_dict = {}
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if label not in count_dict:
                count_dict[label] = 1
            else:
                count_dict[label] += 1

            # Draw the bounding box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display count of detected objects
        st.image(image, channels="BGR")
        for label, count in count_dict.items():
            st.write(f"{label}: {count} detected")
    
if __name__ == "__main__":
    run_app()
