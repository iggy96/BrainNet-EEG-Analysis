from fn_libs import *
from window_pptys import *




class ElectrodeImpedanceStatusWidget(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)
        self.setLayout(layout)

        # Image Label
        self.image_label = QLabel(self)
        layout.addWidget(self.image_label)

        # Create QPixmap
        self.pixmap = QPixmap(800, 800)
        self.pixmap.fill(QColor("white"))

        # Initialize pill colors to white
        self.pill_colors = {name: "white" for names in self.get_pill_names() for name in names}

        # Draw the initial state
        self.redraw_pixmap()

        # Create a circle shaped button with text "Start Check"
        start_check_button = QPushButton("Start")
        start_check_button.setFixedSize(50, 40)
        start_check_button.setFont(QFont("Arial", 10))  # Change font here if needed
        start_check_button.setStyleSheet("""
            QPushButton {
                background-color: #5a189a;
                color: white;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #3a0868;
            }
        """)

        # Start the script and the timer
        start_check_button.clicked.connect(self.run_fn_impedancecheck_script)

        # Add the button to the layout
        layout.addWidget(start_check_button, alignment=Qt.AlignCenter)

        # Create a timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.check_impedance_colors)
        self.color_index = 0


    def run_fn_impedancecheck_script(self):
       # print("Starting script...")
        script_path = r"fn_impedancecheck.py"  # Replace with the actual path to your script
        subprocess.run(["python", script_path])
        # Start the timer with 5 second interval
        self.timer.start(1000)  # 5000 ms = 5 seconds

    def check_impedance_colors(self):
        try:
            with open(r"..\assets\utils\imps.txt", "r") as file:
                colors = file.readlines()

                # Use color_index to determine which colors to use
                if self.color_index < len(colors) - 2:
                    fz_color = colors[self.color_index].strip().lower() # Convert to lowercase
                    cz_color = colors[self.color_index + 1].strip().lower() # Convert to lowercase
                    pz_color = colors[self.color_index + 2].strip().lower() # Convert to lowercase

                    # Update pill colors
                    self.update_pill_color("FZ", fz_color)
                    self.update_pill_color("CZ", cz_color)
                    self.update_pill_color("PZ", pz_color)

                    # Move to the next set of colors
                    self.color_index += 3
                else:
                    # Stop the timer if we reached the end of the file
                    #self.timer.stop()
                    pass
        except Exception as e:
            pass
           # print("Error reading impedance colors:", e)

    def update_pill_color(self, pill_name, color):
        self.pill_colors[pill_name] = color
        self.redraw_pixmap()

    def redraw_pixmap(self):
        # Clear pixmap
        self.pixmap.fill(QColor("white"))

        # Create QPainter to draw on QPixmap
        painter = QPainter(self.pixmap)
        font = QFont("Arial", 10, QFont.Bold)  # Change font here if needed
        painter.setFont(font)

        # Draw header
        painter.setFont(QFont("Arial", 20, QFont.Bold))  # Change font here if needed
        painter.setPen(QColor("#5a189a"))
        painter.drawText(0, 10, self.pixmap.width(), 90, Qt.AlignCenter, "Electrode Impedance Status")
        painter.setFont(QFont("Arial", 10, QFont.Bold))  # Change font here if needed

        # Draw pill shapes
        y = 160
        image_width = self.pixmap.width()
        pill_names = self.get_pill_names()
        for row in pill_names:
            total_width_of_row = len(row) * 60 - 10
            x = (image_width - total_width_of_row) // 2
            for name in row:
                # Get color from pill_colors or default to white if not present
                color = self.pill_colors.get(name, "white")

                # Set brush color for filling the shape
                painter.setBrush(QColor(color))

                # Draw pill shape with the color
                painter.setPen(QPen(QColor("black"), 3))
                painter.drawRoundedRect(x, y, 50, 20, 10, 10)

                # Draw text inside centered (always in black)
                painter.setPen(QColor("black"))
                painter.drawText(x, y, 50, 20, Qt.AlignCenter, name)

                x += 60
            y += 40

        # Draw image at the bottom
        image_path = r"..\assets\imgs\es_bar.png"
        image_pixmap = QPixmap(image_path)
        if not image_pixmap.isNull():
            image_pixmap = image_pixmap.scaled(400, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            image_x = (self.pixmap.width() - image_pixmap.width()) // 2
            painter.drawPixmap(image_x, self.pixmap.height() - 200, image_pixmap)

        # End painting
        painter.end()

        # Refresh the label's pixmap
        self.image_label.setPixmap(self.pixmap)

    def get_pill_names(self):
        return [
            ("FP1", "FP2"),
            ("F7", "F8"),
            ("F3", "FZ", "F4"),
            ("FC5", "FC1", "FC2", "FC6"),
            ("C3", "CZ", "C4"),
            ("T3", "T4"),
            ("CP5", "CP1", "CP2", "CP6"),
            ("P7", "P8"),
            ("P3", "PZ", "P4"),
            ("PO7", "PO3", "PO4", "PO8"),
            ("O1", "OZ", "O2")
        ]
    



