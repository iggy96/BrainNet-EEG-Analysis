from fn_libs import *

win_width, win_height = 900,1000
font_style = "Default"

welcome_text = """
Enjoy an audio sequence for brain response stimulation.

It consists of three scans with brief breaks.

For accurate EEG data:

1. Secure connections and apply gels.
2. Keep still and don’t remove the cap.
3. Use a quiet and dim room.
4. Breathe normally; minimize blinks.
5. Clean cap and electrodes after use.
6. Data is saved automatically.

Get ready for the experience!
"""

class Line(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)
        self.setLineWidth(3)
        self.setStyleSheet("background-color: #5a189a;")
        self.setFixedHeight(3)
        self.setFixedWidth(50)

class CircleButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.init_ui()

    def init_ui(self):
        self.setStyleSheet("""
            QPushButton {
                background-color: #5a189a;
                color: white;
                font-size: 20px;
                font-weight: bold;
                border: none;
                border-radius: 50px;
                box-shadow: 3px 3px 3px rgba(0, 0, 0, 0.3);
            }
            QPushButton:hover {
                background-color: #7b2cbf;
            }
        """)

class mainWindowTitleBar(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.start_drag = False
        self.start_drag_pos = None

    def init_ui(self):
        layout = QHBoxLayout()

        self.label = QLabel("BNN")
        self.label.setFont(QFont(font_style, 10, QFont.Bold))
        self.label.setStyleSheet("color: #5a189a;")

        self.minimize_button = QPushButton("–")
        self.minimize_button.setFont(QFont(font_style, 10, QFont.Bold))
        self.minimize_button.setStyleSheet("""
            QPushButton {
                background-color: #5a189a;
                color: white;
                border-radius: 15px;
                width: 30px;
                height: 30px;
            }
            QPushButton:hover {
                background-color: #39FF14;
            }
        """)
        self.minimize_button.setFixedSize(30, 30)
        self.minimize_button.clicked.connect(self.parent().showMinimized)

        self.close_button = QPushButton("×")
        self.close_button.setFont(QFont(font_style, 12, QFont.Bold))
        self.close_button.setStyleSheet("""
            QPushButton {
                background-color: #5a189a;
                color: white;
                border-radius: 15px;
                width: 30px;
                height: 30px;
            }
            QPushButton:hover {
                background-color: #FD1C03;
            }
        """)
        self.close_button.setFixedSize(30, 30)
        self.close_button.clicked.connect(self.parent().close)

        layout.addWidget(self.label)
        layout.addStretch(1)
        layout.addWidget(self.minimize_button)
        layout.addWidget(self.close_button)

        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.setLayout(layout)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start_drag = True
            self.start_drag_pos = event.globalPos()
            self.window_start_position = self.parent().pos()
            event.accept()
        else:
            event.ignore()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and self.start_drag:
            move_distance = event.globalPos() - self.start_drag_pos
            new_position = self.window_start_position + move_distance
            self.parent().move(new_position)
            event.accept()
        else:
            event.ignore()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start_drag = False
            event.accept()
        else:
            event.ignore()

class loginWindowTitleBar(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.start_drag = False
        self.start_drag_pos = None

    def init_ui(self):
        layout = QHBoxLayout()

        # Removed the BNN label
        # self.label = QLabel("BNN")
        # self.label.setFont(QFont(font_style, 10, QFont.Bold))
        # self.label.setStyleSheet("color: #5a189a;")

        self.minimize_button = QPushButton("–")
        self.minimize_button.setFont(QFont(font_style, 10, QFont.Bold))
        self.minimize_button.setStyleSheet("""
            QPushButton {
                background-color: #5a189a;
                color: white;
                border-radius: 15px;
                width: 30px;
                height: 30px;
            }
            QPushButton:hover {
                background-color: #39FF14;
            }
        """)
        self.minimize_button.setFixedSize(30, 30)
        self.minimize_button.clicked.connect(self.parent().showMinimized)

        self.close_button = QPushButton("×")
        self.close_button.setFont(QFont(font_style, 12, QFont.Bold))
        self.close_button.setStyleSheet("""
            QPushButton {
                background-color: #5a189a;
                color: white;
                border-radius: 15px;
                width: 30px;
                height: 30px;
            }
            QPushButton:hover {
                background-color: #FD1C03;
            }
        """)
        self.close_button.setFixedSize(30, 30)
        self.close_button.clicked.connect(self.parent().close)

        # Removed the addition of the BNN label to the layout
        # layout.addWidget(self.label)
        layout.addStretch(1)
        layout.addWidget(self.minimize_button)
        layout.addWidget(self.close_button)

        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.setLayout(layout)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start_drag = True
            self.start_drag_pos = event.globalPos()
            self.window_start_position = self.parent().pos()
            event.accept()
        else:
            event.ignore()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and self.start_drag:
            move_distance = event.globalPos() - self.start_drag_pos
            new_position = self.window_start_position + move_distance
            self.parent().move(new_position)
            event.accept()
        else:
            event.ignore()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start_drag = False
            event.accept()
        else:
            event.ignore()
