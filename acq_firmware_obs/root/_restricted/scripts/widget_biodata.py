from fn_libs import *
from window_pptys import *

class PersonalInfoForm(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.header = QLabel("""<html>
            <body>
                <p style="font-size: 20pt; font-weight: bold;">Personal Information</p>
                <p style="font-size: 10pt;">Fill subject details where necessary below</p>
            </body>
        </html>""")
        self.header.setWordWrap(True)
        self.header.setStyleSheet("""
            QLabel {
                color: #5a189a;
                font-family: Helvetica;
                padding: 10px;
            }
        """)

        self.header.setAlignment(Qt.AlignTop | Qt.AlignCenter)
        layout.addWidget(self.header)

        fields_widget = QWidget()
        fields_layout = QVBoxLayout(fields_widget)
        layout.addWidget(fields_widget, alignment=Qt.AlignCenter)

        input_fields_width,input_fields_height = 600,20
        alt_input_fields_width,alt_input_fields_height = 300,20

        # Create text fields
        name_layout = QHBoxLayout()
        self.name_field = QLineEdit()
        self.name_field.setPlaceholderText("")
        self.name_field.setFixedSize(input_fields_width, input_fields_height)
        name_label = QLabel("Name*")
        name_layout.addWidget(name_label)
        name_layout.addWidget(self.name_field)
        fields_layout.addLayout(name_layout)

        # Create a horizontal layout for the ID field
        id_layout = QHBoxLayout()
        self.id_field = QLineEdit()
        self.id_field.setPlaceholderText("")
        self.id_field.setFixedSize(input_fields_width, input_fields_height)
        id_label = QLabel("ID*")
        id_layout.addWidget(id_label)
        id_layout.addWidget(self.id_field)
        fields_layout.addLayout(id_layout)

        # Create a horizontal layout for the Sex field
        sex_layout = QHBoxLayout()
        sex_label = QLabel("Sex")
        self.sex_combo = QComboBox()
        self.sex_combo.addItems(["Male", "Female"])
        self.sex_combo.setCurrentIndex(-1)
        self.sex_combo.setFixedSize(input_fields_width, input_fields_height)
        sex_layout.addWidget(sex_label)
        sex_layout.addWidget(self.sex_combo)
        fields_layout.addLayout(sex_layout)

        # Create a horizontal layout for the Age field
        age_layout = QHBoxLayout()
        age_label = QLabel("Age*")
        self.age_field = QLineEdit()
        self.age_field.setPlaceholderText("")
        self.age_field.setFixedSize(input_fields_width, input_fields_height)
        age_layout.addWidget(age_label)
        age_layout.addWidget(self.age_field)
        fields_layout.addLayout(age_layout)

        # Create a horizontal layout for the Date (DDMMYYYY) field
        date_layout = QHBoxLayout()
        date_label = QLabel("Date (DDMMYYYY)*")
        self.date_field = QLineEdit()
        self.date_field.setPlaceholderText("")
        self.date_field.setFixedSize(input_fields_width, input_fields_height)
        date_layout.addWidget(date_label)
        date_layout.addWidget(self.date_field)
        fields_layout.addLayout(date_layout)

        # Create a horizontal layout for the Scan Type field
        scantype_layout = QHBoxLayout()
        scantype_label = QLabel("Scan Type")
        self.scantype_combo = QComboBox()
        self.scantype_combo.addItems(["Initial Scan","Follow-up Scan"])
        self.scantype_combo.setCurrentIndex(-1)
        self.scantype_combo.setFixedSize(input_fields_width, input_fields_height)
        scantype_layout.addWidget(scantype_label)
        scantype_layout.addWidget(self.scantype_combo)
        fields_layout.addLayout(scantype_layout)

        # Create a horizontal layout for the Scan Location field
        scanlocation_layout = QHBoxLayout()
        scanlocation_label = QLabel("Scan Location")
        self.scanlocation_combo = QComboBox()
        self.scanlocation_combo.addItems(["Hospital","Home","Sports Center","Other"])
        self.scanlocation_combo.setCurrentIndex(-1)
        self.scanlocation_combo.setFixedSize(input_fields_width, input_fields_height)
        scanlocation_layout.addWidget(scanlocation_label)
        scanlocation_layout.addWidget(self.scanlocation_combo)
        fields_layout.addLayout(scanlocation_layout)

        # Create a horizontal layout for the Neurological Disorders field
        neuro_disorders_layout = QHBoxLayout()
        neuro_disorders_label = QLabel("Have you had any Neurological Disorders")   
        self.neuro_disorders_field = QLineEdit()
        self.neuro_disorders_field.setPlaceholderText("")
        self.neuro_disorders_field.setFixedSize(alt_input_fields_width, alt_input_fields_height)
        neuro_disorders_layout.addWidget(neuro_disorders_label)
        neuro_disorders_layout.addWidget(self.neuro_disorders_field)
        fields_layout.addLayout(neuro_disorders_layout)

        # Create a horizontal layout for the mood field
        mood_layout = QHBoxLayout()
        mood_label = QLabel("How is your mood today")
        self.mood_combo = QComboBox()
        self.mood_combo.addItems(["Good", "Average", "Bad"])
        self.mood_combo.setCurrentIndex(-1)
        self.mood_combo.setFixedSize(alt_input_fields_width, alt_input_fields_height)
        mood_layout.addWidget(mood_label)
        mood_layout.addWidget(self.mood_combo)
        fields_layout.addLayout(mood_layout)

        # Create a horizontal layout for the sleep field
        sleep_layout = QHBoxLayout()
        sleep_label = QLabel("How many hours of sleep did you have last night")
        self.sleep_combo = QComboBox()
        self.sleep_combo.addItems(["0-4", "4-6", "6-10"])
        self.sleep_combo.setCurrentIndex(-1)
        self.sleep_combo.setFixedSize(alt_input_fields_width, alt_input_fields_height)
        sleep_layout.addWidget(sleep_label)
        sleep_layout.addWidget(self.sleep_combo)
        fields_layout.addLayout(sleep_layout)

        # Create a horizontal layout for the caffeine field
        caffeine_layout = QHBoxLayout()
        caffeine_label = QLabel("Have you consumed any form of caffeine or alcohol in the last 24 hours")
        self.caffeine_combo = QComboBox()
        self.caffeine_combo.addItems(["Yes", "No"])
        self.caffeine_combo.setCurrentIndex(-1)
        self.caffeine_combo.setFixedSize(alt_input_fields_width, alt_input_fields_height)
        caffeine_layout.addWidget(caffeine_label)
        caffeine_layout.addWidget(self.caffeine_combo)
        fields_layout.addLayout(caffeine_layout)

        # Create a horizontal layout for medication field
        medication_layout = QHBoxLayout()
        medication_label = QLabel("Are you on any medications")
        self.medication_combo = QComboBox()
        self.medication_combo.addItems(["None",
            "Antibiotics", "Painkillers", "Anesthetics", "Antipsychotics",
            "Antidementia agents", "Bipolar agents", "Antidepressants", "Blood glucose regulators", "Hormone suppressant (thyroid)"
        ])
        self.medication_combo.setCurrentIndex(-1)
        self.medication_combo.setFixedSize(alt_input_fields_width, alt_input_fields_height)
        medication_layout.addWidget(medication_label)
        medication_layout.addWidget(self.medication_combo)
        fields_layout.addLayout(medication_layout)

        # Create a horizontal layout for the other location
        other_location_layout = QHBoxLayout()
        other_location_label = QLabel("Location*")
        self.other_location_field = QComboBox()
        self.other_location_field.addItems(["America","Africa","Asia","Europe","Australia"])
        self.other_location_field.setCurrentIndex(-1)
        self.other_location_field.setFixedSize(input_fields_width, input_fields_height)
        other_location_layout.addWidget(other_location_label)
        other_location_layout.addWidget(self.other_location_field)
        fields_layout.addLayout(other_location_layout)

        # Create a horizontal layout for language field
        language_layout = QHBoxLayout()
        language_label = QLabel("Language*")
        self.language_field = QComboBox()
        self.language_field.addItems(["English","Chichenwa","French","Spanish","German","Chinese","Japanese","Korean"])
        self.language_field.setCurrentIndex(-1)
        self.language_field.setFixedSize(input_fields_width, input_fields_height)
        language_layout.addWidget(language_label)
        language_layout.addWidget(self.language_field)
        fields_layout.addLayout(language_layout)


        # if America is selected, create a line variable with integer 60Hz
        # if Africa is selected, create a line variable with integer 50Hz
        # if Asia is selected, create a line variable with integer 50Hz
        # if Europe is selected, create a line variable with integer 50Hz
        # if Australia is selected, create a line variable with integer 50Hz

        # Set dark purple font color and reduced width for Save button
        save_button = QPushButton("Save")
        save_button.clicked.connect(self.save_data)
        save_button.setFont(QFont(font_style, 10))
        save_button.setStyleSheet("""
            QPushButton {
                background-color: #5a189a;
                color: white;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #3a0868;
            }
        """)
        save_button.setFixedSize(100,30)

        layout.addWidget(save_button, alignment=Qt.AlignCenter)

        self.setLayout(layout)

    def save_data(self):
        biodata = [
            f"Name:{self.name_field.text()}",
            f"ID:{self.id_field.text()}",
            f"Sex:{self.sex_combo.currentText()}",
            f"Age:{self.age_field.text()}",
            f"Date:{self.date_field.text()}",
            f"Scan Type:{self.scantype_combo.currentText()}",
            f"Scan Location:{self.scanlocation_combo.currentText()}",
            f"Neurological Disorders:{self.neuro_disorders_field.text()}",
            f"How is your mood today:{self.mood_combo.currentText()}",
            f"How many hours of sleep did you have last night:{self.sleep_combo.currentText()}",
            f"Have you consumed any form of caffeine in the last 24 hours: {self.caffeine_combo.currentText()}",
            f"Are you on any medications:{self.medication_combo.currentText()}",
            f"Location:{self.other_location_field.currentText()}",
            f"Language:{self.language_field.currentText()}"
        ]

        # Save the biodata list to a text file
        output_dir = r"..\assets\utils"
        os.makedirs(output_dir, exist_ok=True) # This ensures that the directory exists
        output_file = os.path.join(output_dir, "biodata.txt")

        with open(output_file, "w") as f:
            for line in biodata:
                f.write(line + "\n")

        # 1. Disable the save button
        self.sender().setEnabled(False)

        # 2. Replace the "save" text with "saved"
        self.sender().setText("Saved")

        # 3. Make the disabled button have green background color
        self.sender().setStyleSheet("""
            QPushButton {
                background-color: #00C800;
                color: white;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #00C800;
            }
            QPushButton:disabled {
                background-color: #00C800;
                color: white;
            }
        """)




def main():
    app = QApplication([])
    widget = PersonalInfoForm()
    widget.show()
    app.exec_()

if __name__ == '__main__':
    main()