from fn_libs import *
from window_pptys import *

class LoginWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        #self.setWindowTitle("BNN")
        self.setFixedSize(win_width, win_height)
        self.setWindowFlags(Qt.FramelessWindowHint)

        layout = QVBoxLayout()

        # Custom title bar
        self.title_bar = loginWindowTitleBar(self)
        layout.addWidget(self.title_bar)

        # Add a header
        header = QLabel("BrainNet")
        header_font = QFont(font_style, 70, QFont.Bold)
        header.setFont(header_font)
        header.setStyleSheet("color: #5a189a;")
        header.setAlignment(Qt.AlignTop | Qt.AlignCenter)
        header.setFixedHeight(300)
        layout.addWidget(header)

        form_layout = QVBoxLayout()

        # Center the form layout
        form_layout.addStretch()

        label_font = QFont(font_style, 15)
        self.username_label = QLabel("Username:")
        self.username_label.setFont(label_font)
        self.username_input = QLineEdit()
        self.username_input.setFont(label_font)
        self.username_input.setFixedWidth(200)
        username_layout = QHBoxLayout()
        username_layout.addStretch()
        username_layout.addWidget(self.username_label)
        username_layout.addWidget(self.username_input)
        username_layout.addStretch()
        form_layout.addLayout(username_layout)

        self.password_label = QLabel("Password:")
        self.password_label.setFont(label_font)
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setFont(label_font)
        self.password_input.setFixedWidth(200)
        password_layout = QHBoxLayout()
        password_layout.addStretch()
        password_layout.addWidget(self.password_label)
        password_layout.addWidget(self.password_input)
        password_layout.addStretch()
        form_layout.addLayout(password_layout)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        self.login_button = QPushButton("Login")
        self.login_button.setFixedSize(100,60)
        self.login_button.setFont(label_font)
        self.login_button.setStyleSheet("""
            QPushButton {
                background-color: #5a189a;
                color: white;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #3a0868;
            }
        """)
        self.login_button.clicked.connect(self.check_credentials)
        button_layout.addWidget(self.login_button)
        button_layout.addStretch()
        form_layout.addLayout(button_layout)

        form_layout.addStretch()

        layout.addLayout(form_layout)

        # Text at the bottom of the window
        bottom_text = QLabel("Custom Data Acquisition and Analysis Software")
        bottom_text.setAlignment(Qt.AlignCenter)
        bottom_text.setFont(QFont(font_style, 10))
        bottom_text.setStyleSheet("color: #5a189a;")
        bottom_text.setFixedHeight(200)
        layout.addWidget(bottom_text)


        # make window edges round
        bitmap = QBitmap(self.size())
        bitmap.clear()
        painter = QPainter(bitmap)
        painter.setRenderHints(QPainter.Antialiasing)
        painter.setBrush(Qt.black)
        painter.drawRoundedRect(bitmap.rect(), 20, 20)
        painter.end()
        self.setMask(bitmap)

        self.setLayout(layout)

    def check_credentials(self):
        username = self.username_input.text()
        password = self.password_input.text()

        if username == "brainnet" and password == "123456":
            self.login_button.setStyleSheet("background-color: green; color: white; border-radius: 10px;")
            
            # Clear previous impedances
            with open(r"..\assets\utils\imps.txt", "w") as file:
                file.write("")

            self.accept()
        else:
            self.login_button.setStyleSheet("background-color: red; color: white; border-radius: 10px;")
            QMessageBox.warning(self, "Error", "Incorrect username or password.", QMessageBox.NoButton)