#!/usr/bin/env python
# coding: utf-8

# In[27]:


import cv2
import numpy as np


# In[28]:


# Load YOLO
def load_yolo():
    net = cv2.dnn.readNet("/Users/medhavijam/Desktop/HACKTX24/yolov3.weights",
                           "/Users/medhavijam/Desktop/HACKTX24/yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers

net, output_layers = load_yolo()
print ('hi')


# In[29]:


with open("/Users/medhavijam/Desktop/HACKTX24/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]


# In[30]:


def detect_objects(img):
    height, width = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids, confidences, boxes = [], [], []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Confidence threshold
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    return boxes, confidences, class_ids


# In[31]:


def draw_labels(boxes, confidences, class_ids, img):
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    detected_food = []

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            detected_food.append(label)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return detected_food


# In[32]:


# Load and process an image
image_path = '/Users/medhavijam/Desktop/HACKTX24/foods7.jpg' 
image = cv2.imread(image_path)
image_resized = cv2.resize(image, None, fx=0.4, fy=0.4)  # Resize for faster processing

boxes, confidences, class_ids = detect_objects(image_resized)
detected_items = draw_labels(boxes, confidences, class_ids, image_resized)

# Show the image with detections
cv2.imshow("Image", image_resized)
cv2.waitKey(3000)
cv2.destroyAllWindows()

print("Detected food items:", detected_items)


# In[33]:


import random

def prioritize_food_items(detected_items):
    freshness_scores = {}
    for item in detected_items:
        freshness_scores[item] = random.uniform(0, 1)
    sorted_items = sorted(freshness_scores.items(), key=lambda x: x[1])
    return sorted_items

temp = prioritize_food_items(detected_items)


# In[34]:


#!pip install mosaicml


# In[35]:


from langchain_community.llms import Ollama
llm = Ollama(model="llama3") 


# In[36]:


detected_items= []
for i in temp:
    detected_items.append(i[0])


# In[38]:


hardcoded_ingredients = ['milk', 'sugar', 'salt', 'butter', 'flour', 'pepper', 'eggs']

# detected_items = ['banana', 'apple', 'apple', 'carrot']

all_ingredients = hardcoded_ingredients + detected_items

top_n = 2
top_items = detected_items[:top_n]
remaining_items = detected_items[top_n:]


prompt = (f"Create a recipe using the following ingredients. Please prioritize using the first {top_n} detected "
          f"ingredients: {', '.join(top_items)}. After that, you can include the remaining detected ingredients: "
          f"{', '.join(remaining_items)}. You may also use some of these additional ingredients if needed: "
          f"{', '.join(hardcoded_ingredients)}. Make sure the recipe makes sense.")

llm(prompt)


# In[ ]:




