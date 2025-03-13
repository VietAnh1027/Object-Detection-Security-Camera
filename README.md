# Object Detection Security Camera Documentation

## Feature
- Device: You can choose to use CPU or GPU (available for CUDA only) to run the program (recommend GPU, just use CPU for testing)
- Object: By default the system will detect all objects. If you choose the object, the system will sound an alarm if detected

## Custom code
- Model: You can change YOLO model by pass model name into self.model = YOLO("model name")
- Sound: Put your sound alarm into source folder, remember to rename the sound to "sound.mp3"


![image](https://github.com/user-attachments/assets/112789f0-2368-45ff-a2d9-b99c74a55327)
