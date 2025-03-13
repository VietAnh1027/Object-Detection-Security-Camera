# Object Detection Security Camera Documentation

## Feature
- Device: You can choose to use CPU or GPU (available for CUDA only) to run the program (recommend GPU, just use CPU for testing)
- Object: By default the system will detect all objects. If you choose the object, the system will sound an alarm if detected

## Custom code
- Model: You can change YOLO model by pass model name into self.model = YOLO("model name")
- Sound: Put your sound alarm into source folder, remember to rename the sound to "sound.mp3"


![image](https://github.com/user-attachments/assets/a6ea036b-c57b-4218-8a1f-d0aa33aac917)
