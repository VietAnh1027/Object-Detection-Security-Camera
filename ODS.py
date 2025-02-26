from PyQt6 import QtGui
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt6.uic import loadUi
import sys
import cv2 as cv
from ultralytics import YOLO
import pygame

class GUI_detect(QMainWindow):
    def __init__(self):
        # Tải UI lên
        super().__init__()
        loadUi("GUI/finalUI.ui",self)
        self.setWindowTitle("Real-time Object Detection Camera")

        # Thiết lập chức năng các nút
        self.radioCPU.setChecked(True)
        self.radioCPU.toggled.connect(self.change_to_cpu)
        self.radioGPU.toggled.connect(self.change_to_gpu)

        self.objectList = set()
        self.checkBoxBi.stateChanged.connect(self.bicycle)
        self.checkBoxCar.stateChanged.connect(self.car)
        self.checkBoxMotor.stateChanged.connect(self.motor)
        self.checkBoxPerson.stateChanged.connect(self.person)
        self.checkBoxUmb.stateChanged.connect(self.umbrella)

        self.pushButton.clicked.connect(self.state_cam)
        self.is_playing = False

        self.cap = None

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Mô hình Detect
        self.model = YOLO("yolov8s.pt")
        self.model.to("cpu")
        
        # Khởi tạo âm thanh
        pygame.mixer.init()
        self.sound = pygame.mixer.Sound("source/sound.mp3")

        # 0: webcam máy tính; 1,2,3,...: Các camera bên ngoài
        self.typeCam = 0


    def state_cam(self):
        if self.is_playing:
            self.timer.stop()
            self.cap.release()
            self.cap = None
            self.pushButton.setText("Start")
            self.label.setPixmap(QtGui.QPixmap("source/wait.png"))
        else:
            self.cap = cv.VideoCapture(self.typeCam)
            self.timer.start(30)
            self.pushButton.setText("Stop")
        self.is_playing = not self.is_playing

    def update_frame(self):
        if self.cap is None:
            return

        ret, frame = self.cap.read()
        if ret:
            if self.objectList:
                result = self.model.predict(frame, conf=0.5, classes=list(self.objectList))
                class_ids = result[0].boxes.cls.cpu().numpy().astype('int')
                if any(element in self.objectList for element in class_ids):
                    self.sound.play()
                else: self.sound.stop()
            else:
                result = self.model.predict(frame, classes=None)

            # result là danh sách có 1 phần tử nên phải result[0]
            annotated_frame = result[0].plot()

            rgb_frame = cv.cvtColor(annotated_frame, cv.COLOR_BGR2RGB)

            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            real_frame = QPixmap.fromImage(qt_img)

            self.label.setPixmap(real_frame)


    def change_to_cpu(self):
        if self.radioCPU.isChecked():
            self.model.to("cpu")
            print("Using CPU")

    def change_to_gpu(self):
        if self.radioGPU.isChecked():
            try:
                self.model.to("cuda")
                print("Using GPU")
            except:
                self.show_popup()


    def bicycle(self):
        if self.checkBoxBi.isChecked():
            self.objectList.add(1)

        else:
            self.objectList.discard(1)


    def car(self):
        if self.checkBoxCar.isChecked():
            self.objectList.add(2)

        else:
            self.objectList.discard(2)


    def motor(self):
        if self.checkBoxMotor.isChecked():
            self.objectList.add(3)

        else:
            self.objectList.discard(3)


    def person(self):
        if self.checkBoxPerson.isChecked():
            self.objectList.add(0)

        else:
            self.objectList.discard(0)


    def umbrella(self):
        if self.checkBoxUmb.isChecked():
            self.objectList.add(25)

        else:
            self.objectList.discard(25)


    def show_popup(self):
        msg = QMessageBox()
        msg.setWindowTitle("Warning")
        msg.setText("GPU is not availabel, use CPU instead!")
        msg.setIcon(QMessageBox.Icon.Critical)
        msg.exec()

    def closeEvent(self, a0):
        if self.is_playing and self.cap:
            self.cap.release()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = GUI_detect()
    win.show()
    sys.exit(app.exec())