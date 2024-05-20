# from ultralytics import YOLO

# model = YOLO("best.pt")

# while True:
#     result = model.predict(show=True, source="123.jpg")
# #  source="0")



from ultralytics import YOLO
import easyocr
import cv2

# Initialize YOLO model
model = YOLO("best.pt")

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Source image
source = "bien1.jpg"

# Start prediction loop
while True:
    results = model.predict(show=True, source=source)
    
    # Load the image
    image = cv2.imread(source)
    
    # Process results
    for result in results:
        boxes = result.boxes.xyxy  # Bounding box coordinates (x1, y1, x2, y2)
        confidences = result.boxes.conf  # Confidence scores
        
        for box, confidence in zip(boxes, confidences):
            if confidence > 0.25:  # Ensure the detection is confident enough
                x1, y1, x2, y2 = map(int, box)
                
                # Crop the detected license plate area
                cropped_plate = image[y1:y2, x1:x2]
                
                # Use EasyOCR to read text from the cropped license plate
                text_results = reader.readtext(cropped_plate)
                
                # Extract and print the detected text
                for (bbox, text, prob) in text_results:
                    if prob > 0.5:  # Adjust the probability threshold as needed
                        print("Detected license plate text:", text)
                        
                        # Optionally, you can draw the bounding box and text on the image
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Display the result
    cv2.imshow("License Plate Detection", image)
                
    # Exit condition for the loop (e.g., pressing 'q')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()

