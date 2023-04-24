import numpy as np
import cv2
import os
from pathlib import Path

MIN_NEIGHBOURS = 5


def fr(scale_factor):
    '''
    Main face recognition function, uses the images directory as the dataset, trains a model
    on the available images and assigns a label to the input image based on the trained model.
    The labels are obtained from the file names of the images available in the images folder itself.

    @param scale_factor: the scale factor to be used for the face detection algorithm
    '''

    try:
        # get current working directory using Path
        current_working_directory = Path.cwd()
        images_directory = os.path.join(
            current_working_directory.parent, 'images')

        # Create a list to store the known people's names (images to train the model on)
        known_people = []
        for file_name in os.listdir(images_directory):
            person_name = os.path.splitext(file_name)[0]
            known_people.append(person_name)

        # Create LBPH recognizer, list to store data and labels
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        training_data = []
        labels = []

        # Loop through known people
        for person_name in known_people:

            img_path = os.path.join(images_directory, f"{person_name}.jpg")
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # Extract the face from the image
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(
                img, scaleFactor=scale_factor, minNeighbors=MIN_NEIGHBOURS, minSize=(30, 30))

            # add face to training data list and label to known labels
            for (x, y, w, h) in faces:
                face = img[y:y + h, x:x + w]
                training_data.append(face)
                labels.append(known_people.index(person_name))

        # Train the LBPH model with the training data and labels
        recognizer.train(training_data, np.array(labels))

        # Load the input image
        input_image_path = os.path.join(
            current_working_directory.parent, 'input.jpg')
        input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

        # Detect faces in the input image and loop through them
        faces = face_cascade.detectMultiScale(
            input_image, scaleFactor=scale_factor, minNeighbors=MIN_NEIGHBOURS, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face = input_image[y:y + h, x:x + w]

            # Predict the label for the detected face using the trained LBPH model
            label, loss = recognizer.predict(face)
            confidence = 100 - loss

            # Check if the predicted label is in the known people list
            if label < len(known_people):
                person_name = known_people[label]
                cv2.putText(input_image, f"{person_name}: {confidence:.2f}",
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # If the predicted label is not in the known people list, display 'Unidentified'
            else:
                cv2.putText(input_image, "Unidentified", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Display the input image with predicted labels
        # cv2.imshow("Input Image", input_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # print(
        #     f'Prediction: {person_name} with a confidence of: {confidence:.2f}%')

        return (person_name, confidence)

    except Exception as e:
        # print('Error:', e)
        return ('undefined', 0)
