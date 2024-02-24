"""
script designed for interface that runs eeg data collection from participants < 18yr old
"""
from fn_libs import *
from window_pptys import *
from fn_datacollection import stimseq
from fn_data_analysis import postprocess_generate_report_2


class Worker(QThread):
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, file1, tone_duration, isi_tone, word_duration, isi_word, ibi, collection_duration, parent=None):
        super().__init__()
        self.file1 = file1
        self.tone_duration = tone_duration
        self.isi_tone = isi_tone
        self.word_duration = word_duration
        self.isi_word = isi_word
        self.ibi = ibi
        self.collection_duration = collection_duration

    def run(self):
        try:
            stimseq(
                self.file1,
                self.tone_duration,
                self.isi_tone,
                self.word_duration,
                self.isi_word,
                self.ibi,
                self.collection_duration
            )
        except Exception as e:
            self.error.emit(str(e))
        else:
            self.finished.emit()

class PostProcessingWorker(QThread):
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def run(self):
        try:
            postprocess_generate_report_2()
        except Exception as e:
            self.error.emit(str(e))
        else:
            self.finished.emit()

class AdminDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle('Admin Panel')

        layout = QVBoxLayout()

        # existing file and timer inputs
        self.dir_line_1 = QLineEdit(self)
        self.dir_line_1.setPlaceholderText("Enter new file path 1")
        self.dir_line_2 = QLineEdit(self)
        self.dir_line_2.setPlaceholderText("Enter new file path 2")
        self.dir_line_3 = QLineEdit(self)
        self.dir_line_3.setPlaceholderText("Enter new file path 3")
        self.time_line = QLineEdit(self)
        self.time_line.setPlaceholderText("Enter new timer time (mm:ss)")

        # new inputs
        self.tone_duration_line = QLineEdit(self)
        self.tone_duration_line.setPlaceholderText("Enter tone duration")
        self.isi_tone_line = QLineEdit(self)
        self.isi_tone_line.setPlaceholderText("Enter ISI Tone")
        self.word_duration_line = QLineEdit(self)
        self.word_duration_line.setPlaceholderText("Enter word duration")
        self.isi_word_line = QLineEdit(self)
        self.isi_word_line.setPlaceholderText("Enter ISI Word")
        self.ibi_line = QLineEdit(self)
        self.ibi_line.setPlaceholderText("Enter IBI")
        self.collection_duration_line = QLineEdit(self)
        self.collection_duration_line.setPlaceholderText("Enter collection duration")

        self.save_button = QPushButton('Save Changes', self)
        self.save_button.setStyleSheet("QPushButton { color : black; }")
        self.save_button.clicked.connect(self.saveChanges)

        # existing widgets added to layout
        layout.addWidget(self.dir_line_1)
        layout.addWidget(self.dir_line_2)
        layout.addWidget(self.dir_line_3)
        layout.addWidget(self.time_line)

        # new widgets added to layout
        layout.addWidget(self.tone_duration_line)
        layout.addWidget(self.isi_tone_line)
        layout.addWidget(self.word_duration_line)
        layout.addWidget(self.isi_word_line)
        layout.addWidget(self.ibi_line)
        layout.addWidget(self.collection_duration_line)

        layout.addWidget(self.save_button)

        self.setLayout(layout)

    def showEvent(self, event):
        self.dir_line_1.setText(self.parent().file1)
        self.dir_line_2.setText(self.parent().file2)
        self.dir_line_3.setText(self.parent().file3)
        self.time_line.setText(self.parent().time_total.toString('mm:ss'))

        # load the existing values into the dialog
        self.tone_duration_line.setText(str(self.parent().tone_duration))
        self.isi_tone_line.setText(str(self.parent().isi_tone))
        self.word_duration_line.setText(str(self.parent().word_duration))
        self.isi_word_line.setText(str(self.parent().isi_word))
        self.ibi_line.setText(str(self.parent().ibi))
        self.collection_duration_line.setText(str(self.parent().collection_duration))
        #self.collection_duration_line.setText(str(self.parent().collectionDuration))

    def saveChanges(self):
        new_file1 = self.dir_line_1.text()
        new_file2 = self.dir_line_2.text()
        new_file3 = self.dir_line_3.text()
        new_time = self.time_line.text()

        # get new settings from dialog
        new_tone_duration = self.tone_duration_line.text()
        new_isi_tone = self.isi_tone_line.text()
        new_word_duration = self.word_duration_line.text()
        new_isi_word = self.isi_word_line.text()
        new_ibi = self.ibi_line.text()
        new_collection_duration = self.collection_duration_line.text()

        # create new settings dictionary
        settings = {
            'file 1': new_file1,
            'file 2': new_file2,
            'file 3': new_file3,
            'time': new_time, 
            'tone_duration': new_tone_duration, 
            'isi_tone': new_isi_tone, 
            'word_duration': new_word_duration, 
            'isi_word': new_isi_word, 
            'ibi': new_ibi, 
            'collection_duration': new_collection_duration
        }
        try:
            with open(self.parent().settings_file, 'w') as file:
                json.dump(settings, file)
        except Exception as e:
            #QMessageBox.critical(self, 'Error', f'Could not save settings: {str(e)}')
            self.error_dialog = ErrorDialog('Error', f'Could not save settings: {str(e)}')
            self.error_dialog.okClicked.connect(self.error_dialog.close)
            self.error_dialog.show()
            return

        self.parent().file1 = new_file1
        self.parent().file2 = new_file2
        self.parent().file3 = new_file3
       # print(f"New file path: {new_file}")

        new_time_qtime = QTime.fromString(new_time, 'mm:ss')
        if not new_time_qtime.isValid():
            QMessageBox.critical(self, 'Invalid time', 'The input time is not valid. Time should be in "mm:ss" format.')
        else:
            self.parent().time_total = new_time_qtime
            self.parent().time_left = new_time_qtime  # Also update the time_left
           # print(f"New time total: {new_time_qtime.toString('mm:ss')}")

        self.parent().tone_duration = new_tone_duration
      #  print(f"New tone duration: {new_tone_duration}")
        self.parent().isi_tone = new_isi_tone
      #  print(f"New ISI tone: {new_isi_tone}")
        self.parent().word_duration = new_word_duration
      #  print(f"New word duration: {new_word_duration}")
        self.parent().isi_word = new_isi_word
      #  print(f"New ISI word: {new_isi_word}")
        self.parent().ibi = new_ibi
      #  print(f"New IBI: {new_ibi}")
        self.parent().collection_duration = new_collection_duration
       # print(f"New collection duration: {new_collection_duration}")
        self.close()

class PasswordDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle('Admin Access')

        layout = QVBoxLayout()

        self.label = QLabel("Enter password:")
        self.edit = QLineEdit()
        self.edit.setEchoMode(QLineEdit.Password)  # The line should be here

        # Create OK and Cancel buttons
        self.ok_button = QPushButton('OK', self)
        self.ok_button.setStyleSheet("QPushButton { color : black; }")  # Set the button color to black
        self.ok_button.clicked.connect(self.accept)

        self.cancel_button = QPushButton('Cancel', self)
        self.cancel_button.setStyleSheet("QPushButton { color : black; }")  # Set the button color to black
        self.cancel_button.clicked.connect(self.reject)

        # Add buttons to a horizontal layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)

        layout.addWidget(self.label)
        layout.addWidget(self.edit)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def getPassword(self):
        return self.edit.text()

class ErrorDialog(QDialog):
    okClicked = pyqtSignal()

    def __init__(self, error_message, parent=None):
        super().__init__(parent)

        self.setWindowTitle('Error')

        layout = QVBoxLayout()

        self.label = QLabel(error_message)
        layout.addWidget(self.label)

        self.ok_button = QPushButton('OK', self)
        self.ok_button.setStyleSheet("QPushButton { color : black; }")
        self.ok_button.clicked.connect(self.accept)
        layout.addWidget(self.ok_button)

        self.setLayout(layout)

    def accept(self):
        self.okClicked.emit()
        super().accept()

class RunScanWidget_2(QWidget):
    def __init__(self):
        super().__init__()

        self.worker = None
        self.pp_worker = None
        self.file1 = r'..\assets\txt\debug.txt'  # Default file
        self.file2 = r'..\assets\txt\debug.txt'  # Default file
        self.file3 = r'..\assets\txt\debug.txt'  # Default file
        self.settings_file = r'..\assets\txt\admin_settings.txt'  # Path to the settings file
        self.time_total = QTime(0, 3, 50)  # Default time total
        # Default values
        self.tone_duration = '0.1'
        self.isi_tone = '0.25'
        self.word_duration = '1.0'
        self.isi_word = '1.0'
        self.ibi = '1.5'
        self.collection_duration = '360'
        self.loadSettings()  # Load the settings at the start of the program

        self.initUI()
        self.initTimer()

    def loadSettings(self):
        try:
            with open(self.settings_file, 'r') as file:
                settings = json.load(file)

            # existing settings
            self.file1 = settings.get('file 1', self.file1)
            self.file2 = settings.get('file 2', self.file2)
            self.file3 = settings.get('file 3', self.file3)
            time_str = settings.get('time', '')
            loaded_time = QTime.fromString(time_str, 'mm:ss') if time_str else self.time_total
            if loaded_time.isValid():
                self.time_total = loaded_time
            self.time_left = self.time_total

            # new settings
            self.tone_duration = settings.get('tone_duration', '')
            self.isi_tone = settings.get('isi_tone', '')
            self.word_duration = settings.get('word_duration', '')
            self.isi_word = settings.get('isi_word', '')
            self.ibi = settings.get('ibi', '')
            self.collection_duration = settings.get('collection_duration', '')

        except Exception as e:
            #print(f'Error loading settings: {str(e)}')
            self.error_dialog = ErrorDialog(f'Error loading settings: {str(e)}')
            self.error_dialog.okClicked.connect(self.error_dialog.close)
            self.error_dialog.show()
     
    def initUI(self):
        self.setGeometry(300, 300, 800, 800)
        self.setWindowTitle('SSR')

        layout = QVBoxLayout()

        self.header = QLabel('Stimulus Sequence Run', self)
        self.header.setFont(QFont('Default', 30, QFont.Bold))
        self.header.setStyleSheet("QLabel { color : #5a189a; }")
        self.header.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.header)

        self.admin_button = QPushButton('Admin', self)
        #self.admin_button.setStyleSheet("QPushButton { color : white; }")
        self.admin_button.setFont(QFont('Default', 8))
        self.admin_button.setStyleSheet("QPushButton { color : black; }")
        self.admin_button.clicked.connect(self.adminAccess)


        admin_layout = QHBoxLayout()
        admin_layout.addStretch()
        admin_layout.addWidget(self.admin_button)
        admin_layout.addStretch()

        layout.addLayout(admin_layout)

        self.timer_label = QLabel('Ready', self)
        self.timer_label.setFont(QFont('Default', 17, QFont.Bold))
        self.timer_label.setStyleSheet("QLabel { color : #5a189a; }")
        self.timer_label.setAlignment(Qt.AlignCenter)

        self.run_button_1 = QPushButton('Run 1', self)
        self.run_button_2 = QPushButton('Run 2', self)
        self.run_button_3 = QPushButton('Run 3', self)
        self.pp_button = QPushButton('Post-Process', self)

        self.run_button_1.clicked.connect(lambda: self.startTimer(self.run_button_1))
        self.run_button_2.clicked.connect(lambda: self.startTimer(self.run_button_2))
        self.run_button_3.clicked.connect(lambda: self.startTimer(self.run_button_3))

        button_style = """
            QPushButton {
                background-color: #5a189a;
                color: white;
                font: bold;
                font-size: 20px;
                height: 60px;
                border: none;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #7b2cbf;
            }
            QPushButton:disabled {
                background-color: #7b2cbf;
                color: #bab0ce;
            }
        """
        pp_button_style = """
            QPushButton {
                background-color: #51FF00;
                color: white;
                font: bold;
                font-size: 20px;
                height: 60px;
                border: none;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #28610D;
            }
            QPushButton:disabled {
                background-color: #163E03;
                color: #bab0ce;
            }
        """
        self.run_button_1.setStyleSheet(button_style)
        self.run_button_2.setStyleSheet(button_style)
        self.run_button_3.setStyleSheet(button_style)
        self.pp_button.setStyleSheet(pp_button_style)
        #self.run_button.setStyleSheet(button_style)
        #self.pp_button.setStyleSheet(pp_button_style)

        self.pp_button.setEnabled(True)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.run_button_1)
        button_layout.addWidget(self.run_button_2)
        button_layout.addWidget(self.run_button_3)
        button_layout.addWidget(self.pp_button)
        #button_layout.addWidget(self.run_button)
        #button_layout.addWidget(self.pp_button)

        self.pp_button.clicked.connect(self.startPostProcessing)

        spacer1 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        spacer2 = QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding)

        layout.addWidget(self.header)
        layout.addItem(spacer1)
        layout.addWidget(self.timer_label)
        layout.addItem(spacer2)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def adminAccess(self):
        dialog = PasswordDialog(self)
        result = dialog.exec_()
        if result == QDialog.Accepted:
            password = dialog.getPassword()
            if password == "123456":  # Replace with the actual password
                self.admin_dialog = AdminDialog(self)
                self.admin_dialog.show()
            else:
                self.error_dialog = ErrorDialog('The password is incorrect.', self)
                self.error_dialog.okClicked.connect(self.error_dialog.close)
                self.error_dialog.show()

    def startPostProcessing(self):
        if self.pp_worker is not None and self.pp_worker.isRunning():
            self.pp_worker.terminate()
            self.pp_worker.wait()

        self.pp_worker = PostProcessingWorker()

        self.pp_worker.finished.connect(self.on_pp_worker_finished)
        self.pp_worker.error.connect(self.on_pp_worker_error)

        self.pp_worker.start()

    def on_pp_worker_finished(self):
        pass

    def on_pp_worker_error(self, error_message):
        self.error_dialog = ErrorDialog(error_message, self)
        self.error_dialog.okClicked.connect(self.error_dialog.close)
        self.error_dialog.show()

    def initTimer(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateTimer)
        # Initialize time_left and time_total with the loaded setting
        self.time_left = self.time_total

    def startTimer(self, button):
        self.current_button = button
        self.current_button.setEnabled(False)
        self.timer.start(1000)

        if self.worker is not None and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()

        if button == self.run_button_1:
            self.worker = Worker(
                self.file1,
                self.tone_duration,
                self.isi_tone,
                self.word_duration,
                self.isi_word,
                self.ibi,
                self.collection_duration,
            )

        elif button == self.run_button_2:
            self.worker = Worker(
                self.file2,
                self.tone_duration,
                self.isi_tone,
                self.word_duration,
                self.isi_word,
                self.ibi,
                self.collection_duration,
            )

        elif button == self.run_button_3:
            self.worker = Worker(
                self.file3,
                self.tone_duration,
                self.isi_tone,
                self.word_duration,
                self.isi_word,
                self.ibi,
                self.collection_duration,
            )
            
        if self.worker is not None:
            self.worker.finished.connect(self.on_worker_finished)
            self.worker.error.connect(self.on_worker_error)
            self.worker.start()

    def on_worker_finished(self):
        #print("Long-running function finished execution.")
        if self.worker is not None:
            self.worker.wait()

    def on_worker_error(self, error_message):
        self.timer.stop()
        self.time_left = self.time_total
        self.error_dialog = ErrorDialog(error_message, self)
        self.error_dialog.okClicked.connect(self.error_dialog.close)
        self.error_dialog.show()
        # self.run_button.setEnabled(True)
        self.run_button_1.setEnabled(True)
        self.run_button_2.setEnabled(True)
        self.run_button_3.setEnabled(True)

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
        event.accept()

    def updateTimer(self):
        self.time_left = self.time_left.addSecs(-1)
        self.timer_label.setText(self.time_left.toString('mm:ss'))

        if self.time_left == QTime(0, 0, 0):
            self.timer.stop()
            self.time_left = self.time_total  # Reset the time

        self.update()

    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)
        self.drawRectProgressBar(qp)
        qp.end()

    def drawRectProgressBar(self, qp):
        size = self.size()
        timer_label_pos = self.timer_label.pos()

        rect_width = self.width() / 2
        rect_height = 20  

        rect_x = (self.width() - rect_width) / 2  
        rect_y = timer_label_pos.y() + self.timer_label.height() + 10

        total_secs = self.time_total.minute() * 60 + self.time_total.second()
        remaining_secs = self.time_left.minute() * 60 + self.time_left.second()
        progress_fraction = remaining_secs / total_secs

        pen = QPen()
        pen.setColor(QColor('#5a189a'))
        pen.setWidth(1)
        qp.setPen(pen)

        qp.drawRect(rect_x, rect_y, rect_width, rect_height)

        qp.fillRect(rect_x, rect_y, rect_width * progress_fraction, rect_height, QColor('#5a189a'))



#if __name__ == '__main__':

#    app = QApplication(sys.argv)
#    win = RunScanWidget_2()
#    win.show()
#    sys.exit(app.exec_())
