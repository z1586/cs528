# run_system.py
import cv2
import numpy as np
import time
import webbrowser
import mediapipe as mp
from setup import mediapipe_detection, draw_styled_landmarks, extract_keypoints
from model import load_trained_model, actions

colors = [
    (245, 117, 16),  # color for 'hello'
    (117, 245, 16),  # color for 'workout'
    (16, 117, 245),  # color for 'fly'
]

action_to_url = {
    'hello': 'https://translate.google.com/',
    'workout': 'https://www.youtube.com/watch?v=V9eik0pseGU',
    'fly': 'https://www.google.com/travel/flights'
}

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame

def main():
    model = load_trained_model()
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.5
    detected_action = None
    detected_action_start_time = None

    cap = cv2.VideoCapture(0)
    with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture image")
                break

            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predictions.append(np.argmax(res))

                if np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > threshold:
                        if detected_action != actions[np.argmax(res)]:
                            detected_action = actions[np.argmax(res)]
                            detected_action_start_time = time.time()
                        else:
                            if time.time() - detected_action_start_time >= 3:
                                webbrowser.open(action_to_url[detected_action])
                                detected_action_start_time = None
                                detected_action = None
                    else:
                        detected_action = None

                if len(sentence) > 5:
                    sentence = sentence[-5:]

                image = prob_viz(res, actions, image, colors)

            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('OpenCV Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
