from fn_libs import *
from window_pptys import *
from widget_biodata import *
from widget_impedancecheck import *
from widget_datacollection_1 import *
from widget_datacollection_2 import *

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        font_style = 'Arial' 
        self.setFixedSize(win_width, win_height)
        self.setWindowFlags(Qt.FramelessWindowHint)

        layout = QVBoxLayout()

        # Custom title bar
        self.title_bar = mainWindowTitleBar(self)
        layout.addWidget(self.title_bar)

        # Dot indicators layout
        self.dot_layout = QHBoxLayout()
        self.dot_layout.setAlignment(Qt.AlignCenter)
        self.dots = []
        for i in range(5):
            dot = QLabel()
            dot.setFixedSize(10, 10)
            dot.setStyleSheet("background-color: grey; border-radius: 5px;")
            self.dot_layout.addWidget(dot)
            self.dots.append(dot)
        layout.addLayout(self.dot_layout)

        # Stacked widget to switch between pages
        self.stacked_widget = QStackedWidget()
        layout.addWidget(self.stacked_widget, alignment=Qt.AlignCenter)

        # This is your single button for navigation and checking impedance
        self.next_button = QPushButton("Next")
        self.next_button.setFixedSize(200, 40)
        self.next_button.setFont(QFont(font_style, 13))
        self.next_button.setStyleSheet("""
            QPushButton {
                background-color: #5a189a;
                color: white;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #3a0868;
            }
        """)
        self.next_button.clicked.connect(self.go_to_next_page)
        layout.addWidget(self.next_button, alignment=Qt.AlignCenter)

###### Page 1 (White canvas with welcome text)
        page1 = QWidget()
        page1_layout = QVBoxLayout()
        page1.setStyleSheet("background-color: white; border: 3px")

        # Set the size policy of page1 to expanding so that it takes up available space
        page1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Add a header
        header = QLabel("Welcome!")
        header_font = QFont(font_style, 50, QFont.Bold)
        header.setFont(header_font)
        header.setStyleSheet("color: #5a189a;")
        header.setAlignment(Qt.AlignTop)
        page1_layout.addWidget(header)
        page1_layout.addSpacing(100)

        # Add a welcome text
        welcome_label = QLabel(welcome_text)
        welcome_label.setStyleSheet("color: black; font-size: 20px;")
        welcome_label.setWordWrap(True)
        welcome_label.setAlignment(Qt.AlignLeft)
        page1_layout.addWidget(welcome_label)

        # Remove layout margins and align the layout to center
        page1_layout.setContentsMargins(0, 0, 0, 0)
        page1_layout.setAlignment(Qt.AlignCenter)
        page1.setLayout(page1_layout)
        self.stacked_widget.addWidget(page1)



###### Page 2 (Another white canvas)
        page2 = QWidget()
        page2.setStyleSheet("background-color: white; border-radius: 10px; border: 3px")
        page2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.stacked_widget.addWidget(page2)
        page2_layout = QVBoxLayout()

        # Create a QGraphicsView to display the image
        graphics_view = QGraphicsView()
        #graphics_view.setFixedSize(700,700)
        graphics_view.setStyleSheet("background-color: transparent; border: none;")

        # Create a QGraphicsScene to add the pixmap
        graphics_scene = QGraphicsScene()

        # Load the pixmap
        image_path = r"..\assets\imgs\device_config.png"
        pixmap = QPixmap(image_path)

        if pixmap.isNull():
            pass
           # print("Failed to load image")
        else:
            # Scale the pixmap to 760x800
            #pixmap = pixmap.scaled(700, 700, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            # Add the pixmap to the QGraphicsScene
            graphics_scene.addPixmap(pixmap)
            # Set the scene for the QGraphicsView
            graphics_view.setScene(graphics_scene)

        # Add the QGraphicsView to the layout
        page2_layout.addWidget(graphics_view, alignment=Qt.AlignCenter)

        # Create a back button with "<" icon
        label_font = QFont(font_style, 8, QFont.Bold)
        back_button = QPushButton("<<")
        back_button.setFixedSize(40, 40)
        back_button.setFont(label_font)
        back_button.setStyleSheet("""
            QPushButton {
                background-color: #5a189a;
                color: white;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #3a0868;
            }
        """)
        back_button.clicked.connect(self.go_to_previous_page)

        # Add the back button to the layout (top left corner)
        back_button_layout = QHBoxLayout()
        back_button_layout.addWidget(back_button)
        back_button_layout.addStretch()
        page2_layout.insertLayout(0, back_button_layout)
        page2.setLayout(page2_layout)



###### Page 3 (Another white canvas)
        page3 = QWidget()
        page3.setStyleSheet("background-color: white; border-radius: 10px;")
        page3.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        #page3.setFixedSize(900, 800)  # Set dimensions of the white page
        page3_layout = QVBoxLayout(page3)

        # Create a back button with "<" icon
        label_font = QFont(font_style, 8, QFont.Bold)
        back_button = QPushButton("<<")
        back_button.setFixedSize(40, 40)
        back_button.setFont(label_font)
        back_button.setStyleSheet("""
            QPushButton {
                background-color: #5a189a;
                color: white;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #3a0868;
            }
        """)
        back_button.clicked.connect(self.go_to_previous_page)

        # Add the back button to its own layout (top right corner)
        back_button_layout = QHBoxLayout()
        back_button_layout.addWidget(back_button)
        back_button_layout.addStretch()
        page3_layout.addLayout(back_button_layout)

        # Create an inner layout for PersonalInfoForm contents
        form_layout = QVBoxLayout()
        self.personal_info_form = PersonalInfoForm()
        form_layout.addWidget(self.personal_info_form)
        form_layout.setAlignment(Qt.AlignCenter)  # centralize the widgets in the layout
        #form_layout.setContentsMargins(-50,0,120,10)  # set margins (left, top, right, bottom)

        # Add form_layout to the main page3_layout
        page3_layout.addLayout(form_layout)
        page3_layout.addStretch()  # Add stretch to allow spacing for the back button

        # Add page3 to the stacked widget
        self.stacked_widget.addWidget(page3)




###### Page 4 (Another white canvas)
        page4 = QWidget()
        page4_layout = QVBoxLayout()
        page4.setStyleSheet("background-color: white; border-radius: 10px;")
        page4.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Add ElectrodeImpedanceStatusWidget to page 4
        electrode_widget = ElectrodeImpedanceStatusWidget()
        page4_layout.addWidget(electrode_widget)

        # Create a back button with "<" icon
        label_font = QFont(font_style, 8, QFont.Bold)
        back_button = QPushButton("<<")
        back_button.setFixedSize(40, 40)
        back_button.setFont(label_font)
        back_button.setStyleSheet("""
            QPushButton {
                background-color: #5a189a;
                color: white;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #3a0868;
            }
        """)
        back_button.clicked.connect(self.go_to_previous_page)

        # Add the back button to the layout (top left corner)
        back_button_layout = QHBoxLayout()
        back_button_layout.addWidget(back_button)
        back_button_layout.addStretch()
        page4_layout.insertLayout(0, back_button_layout)
        page4.setLayout(page4_layout)
        self.stacked_widget.addWidget(page4)

        

        ###### Page 5 (Another white canvas)
        self.page5 = QWidget()
        self.page5_layout = QVBoxLayout(self.page5)
        self.page5.setStyleSheet("background-color: white; border-radius: 10px;")
        self.page5.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Create a back button with "<" icon
        font_style = 'Arial'  # Replace with the actual style if it's different
        label_font = QFont(font_style, 8, QFont.Bold)
        back_button = QPushButton("<<")
        back_button.setFixedSize(40, 40)
        back_button.setFont(label_font)
        back_button.setStyleSheet("""
            QPushButton {
                background-color: #5a189a;
                color: white;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #3a0868;
            }
        """)
        back_button.clicked.connect(self.go_to_previous_page)

        # Add the back button to its own layout (top right corner)
        back_button_layout = QHBoxLayout()
        back_button_layout.addWidget(back_button)
        back_button_layout.addStretch()
        self.page5_layout.insertLayout(0, back_button_layout)
        self.page5.setLayout(self.page5_layout)
        self.stacked_widget.addWidget(self.page5)


# Other window pptys
        self.set_active_dot(0)

        # Apply rounded edges
        bitmap = QBitmap(self.size())
        bitmap.clear()
        painter = QPainter(bitmap)
        painter.setRenderHints(QPainter.Antialiasing)
        painter.setBrush(Qt.black)
        painter.drawRoundedRect(bitmap.rect(), 20, 20)
        painter.end()
        self.setMask(bitmap)

        self.setLayout(layout)

    def setup_page5(self):
        """Sets up the contents of page 5 based on biodata.txt."""
        with open(r"..\assets\utils\biodata.txt", "r") as file:
            lines = [line.strip() for line in file.readlines()]
        for line in lines:
            if "Age:" in line:
                age = int(line.replace("Age:", "").strip())

        # Decide which RunScanWidget variant to use based on age
        if age > 18:
            run_scan_widget = RunScanWidget_1()
        else:
            run_scan_widget = RunScanWidget_2()

        # Clear the previous widget (if any) from the layout and add the new one.
        for i in reversed(range(self.page5_layout.count())):
            widget = self.page5_layout.itemAt(i).widget()
            if widget is not None:
                self.page5_layout.removeWidget(widget)
                widget.deleteLater()

        self.page5_layout.addWidget(run_scan_widget, 1)

    def go_to_next_page(self):
        current_index = self.stacked_widget.currentIndex()
        if current_index + 1 < self.stacked_widget.count():
            next_index = current_index + 1
            if next_index == 4:  # Transitioning to page 5 (0-based index)
                self.setup_page5()
            self.stacked_widget.setCurrentIndex(next_index)
            self.set_active_dot(next_index)
            # Assuming you have a next_button, update its text
            self.next_button.setText("Next")

    def go_to_previous_page(self):
        current_index = self.stacked_widget.currentIndex()
        if current_index > 0:
            self.stacked_widget.setCurrentIndex(current_index - 1)
            self.set_active_dot(current_index - 1)  # Update the dot indicators

    def set_active_dot(self, index):
        # Set the dot at the given index as active, and others as inactive
        for i, dot in enumerate(self.dots):
            if i == index:
                dot.setStyleSheet("background-color: #5a189a; border-radius: 5px;")
            else:
                dot.setStyleSheet("background-color: grey; border-radius: 5px;")
