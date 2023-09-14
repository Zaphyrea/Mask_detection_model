import cv2
import numpy as np
from keras.models import load_model

# Charger the pre-trained Keras model
model = load_model("Initiation_Traitement_Image/mask_tests/test_model_webcam/best_model")

# Launch webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Return mirror image
    frame = cv2.flip(frame, 1)

    # Resize image
    img = cv2.resize(frame, (224, 224))
    img = img / 255.0

    # Add batch dimension
    prediction = model.predict(np.expand_dims(img, axis=0))[0]

    # Get higher prediction index
    predicted_class_index = np.argmax(prediction)

    # Determine class label
    if predicted_class_index == 0:
        label = "Sans masque"
    else:
        label = "Avec masque"

    # Display label
    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display webcam
    cv2.imshow("Webcam", frame)

    # Exit with q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# let's release the webcam
cap.release()
cv2.destroyAllWindows()
