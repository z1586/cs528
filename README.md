# CS 528 Final Project

## Description
This project involves developing a computer vision-based machine learning model to detect and classify gestures in real-time using a webcam feed. The project leverages OpenCV, MediaPipe, and TensorFlow to process video frames, detect keypoints, and train a Long Short-Term Memory (LSTM) network for action recognition. The detected actions trigger predefined responses, such as opening specific URLs.

## Key Features:

- Utilizes OpenCV and MediaPipe for capturing and processing webcam frames in real-time
- Employs a trained LSTM model to classify gestures such as 'hello', 'workout', and 'fly'.
- Displays detected landmarks and probabilities on the screen for user feedback.
- Opens corresponding web pages when gestures are recognized consistently over a set duration.
  
## Libraries and Tools Used:
- OpenCV
- MediaPipe
- TensorFlow
- NumPy
- Matplotlib

## Setup and Usage:
Install the required dependencies.

```
 !pip install tensorflow opencv-python mediapipe scikit-learn matplotlib
```

Run the script to start the webcam feed and begin gesture detection.
Perform the predefined gestures to trigger actions.

### Predefined Gestures:
- **Hello**: Opens Google Translate.
- **Workout**: Opens a workout video on YouTube.
- **Fly**: Opens Google Flights.




## Future Development: Gesture-Controlled OS for the Visually Impaired

This project has the potential to evolve into a comprehensive gesture-controlled operating system tailored for visually impaired individuals. By harnessing the power of computer vision and machine learning, the system can interpret hand gestures to perform various tasks, offering a hands-free, intuitive, and accessible way to interact with technology.

### Advantages:

1. **Accessibility:** Provides an inclusive solution for blind or visually impaired users, enabling them to interact with digital devices without the need for visual feedback.
2. **Intuitive Control:** Uses natural hand gestures for command input, making it easier for users to learn and adapt to the system.
3. **Hands-Free Operation:** Allows users to perform tasks without touching the screen or using a mouse and keyboard, which can be particularly beneficial in various environments and for individuals with limited mobility.

### Proposed Features:

1. **Comprehensive Gesture Library:** Expand the set of detectable gestures to cover a wide range of commands, such as opening applications, controlling media playback, adjusting settings, and navigating through interfaces.
2. **Voice Feedback Integration:** Integrate text-to-speech technology to provide audio feedback for actions performed, ensuring that users receive immediate confirmation and guidance.
3. **Customizable Gestures:** Allow users to customize gestures according to their preferences and needs, enhancing the user experience and adaptability of the system.
4. **Seamless Integration:** Develop compatibility with various operating systems and devices, including smartphones, tablets, and desktop computers, to offer a versatile and scalable solution.

### Implementation Steps:

1. **Expand Data Collection:** Collect and annotate a larger dataset of hand gestures, including those commonly used in everyday tasks, to improve the accuracy and reliability of gesture recognition.
2. **Model Enhancement:** Train more advanced machine learning models, potentially incorporating convolutional neural networks (CNNs) and recurrent neural networks (RNNs), to handle a broader range of gestures with higher precision.
3. **User Testing and Feedback:** Conduct extensive testing with visually impaired users to gather feedback and make iterative improvements, ensuring the system meets their needs and expectations.
4. **Integration with Assistive Technologies:** Collaborate with developers of existing assistive technologies to integrate gesture control capabilities, enhancing the overall ecosystem of tools available to visually impaired individuals.

### Potential Use Cases:

- **Navigation:** Use gestures to navigate through operating systems, open and close applications, and switch between different tasks.
- **Communication:** Enable gestures to control communication tools, such as making phone calls, sending messages, and managing contacts.
- **Entertainment:** Control media playback, adjust volume, and browse through content libraries using intuitive hand movements.
- **Productivity:** Execute commands to write documents, browse the internet, and manage files, enhancing productivity and independence.

By advancing this project into a full-fledged gesture-controlled OS, we can significantly enhance the quality of life for blind and visually impaired individuals, offering them greater autonomy and ease of use in their interactions with digital devices.
