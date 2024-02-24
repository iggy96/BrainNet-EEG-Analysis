import sys
import os 
import json
import numpy as np
import pandas as pd
import time
import wave
import json
import threading
import serial
import subprocess
import time
import numpy as np
import pygds as g
import threading
import ftd2xx as FTD2XX
import numpy as np
import warnings
from datetime import datetime
import winsound
import pygds as g
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QLabel, QWidget, QPushButton, QLineEdit, QFormLayout, QTimeEdit)
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import QTimer, QTime
from PyQt5.QtCore import QThread
import json
import os

class FileDetails(QWidget):
    def __init__(self, parent=None):
        super(FileDetails, self).__init__(parent)
        
        layout = QFormLayout()
        self.id_input = QLineEdit(self)
        self.date_input = QLineEdit(self)
        self.scan_number_input = QLineEdit(self)
        layout.addRow("ID", self.id_input)
        layout.addRow("Date of Scan", self.date_input)
        layout.addRow("Scan number", self.scan_number_input)
        
        self.save_button = QPushButton("Save", self)
        self.save_button.clicked.connect(self.save_to_json)  # Connect the save button to our new function
        
        layout.addWidget(self.save_button)
        self.setLayout(layout)

    def save_to_json(self):
        # Collect the details from the input fields
        details = {
            'ID': self.id_input.text(),
            'Date of Scan': self.date_input.text(),
            'Scan number': self.scan_number_input.text()
        }

        # Create the directory if it doesn't exist
        directory = r"_restricted_\utils"
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save the details to a JSON file
        with open(os.path.join(directory, 'details.json'), 'w') as json_file:
            json.dump(details, json_file, indent=4)

class ImpedanceCheckerWidget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QHBoxLayout()
        self.labels = [QLabel("FZ"), QLabel("CZ"), QLabel("PZ")]
        for label in self.labels:
            layout.addWidget(label)
            label.setStyleSheet('background-color: white; border: 1px solid black; width: 50px; text-align: center;')
        self.check_impedance_button = QPushButton("Check Impedance", self)
        self.check_impedance_button.setDisabled(True)
        layout.addWidget(self.check_impedance_button)
        self.setLayout(layout)

class RunScanLogicWorker(QThread):
    def __init__(self, collect_data_instance):
        super().__init__()
        self.collect_data_instance = collect_data_instance
    
    def run(self):
        self.collect_data_instance.run_scan_logic()

class CollectData(QWidget):
    tone_duration = 0.4
    isi_tone = 0.2
    word_duration = 1
    isi_word = 0.5
    ibi = 0.7
    display_time = "03:40"
    eeg_collection_duration = 210
    TXT_FILE = "_restricted_\\stims_struct\\"
    STIM_DIR = r"_restricted_\audio_files\\"

    def __init__(self):
        super().__init__()

        # File details setup
        self.file_details = FileDetails(self)
        
        layout = QVBoxLayout()

        buttons_layout = QHBoxLayout()
        self.run_scan_button = QPushButton("Run Scan", self)
        self.timer_label = QLabel("Timer")
        self.time_display = QTimeEdit(self)
        self.time_display.setReadOnly(True)
        
        buttons_layout.addWidget(self.run_scan_button)
        buttons_layout.addWidget(self.timer_label)
        buttons_layout.addWidget(self.time_display)
        
        layout.addLayout(buttons_layout)
        self.setLayout(layout)
        
        # Countdown Timer Setup
        minutes, seconds = map(int, self.display_time.split(':'))
        self.remaining_time = QTime(0,int(minutes),int(seconds))
        self.time_display.setDisplayFormat("mm:ss")
        self.time_display.setTime(self.remaining_time)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_timer_display)

        self.run_scan_button.clicked.connect(self.start_countdown)
        self.run_scan_button.clicked.connect(self.start_run_scan_logic)
        
    def update_timer_display(self):
        self.remaining_time = self.remaining_time.addSecs(-1)
        self.time_display.setTime(self.remaining_time)

        if self.remaining_time == QTime(0, 0):
            self.timer.stop()

    def start_countdown(self):
        self.timer.start(1000)

    def start_run_scan_logic(self):
        self.worker = RunScanLogicWorker(self)
        self.worker.start()

    def high_res_sleep(self, milliseconds):
        start_time = time.perf_counter()
        while True:
            current_time = time.perf_counter()
            elapsed_time = (current_time - start_time) * 1000
            if elapsed_time >= milliseconds:
                break

    def write_trigger_value(self, dev, trigger_value):
        trigger = bytes([trigger_value])
        dev.write(trigger)
        self.high_res_sleep(80)
        trigger = bytes([0])
        dev.write(trigger)

    def collect_eeg_data(self, d, samples, duration):
        more = lambda s: samples.append(s.copy()) or len(samples) < duration
        data = d.GetData(d.SamplingRate, more)
        samples_data = np.concatenate(samples, axis=0)

    def play_audio_for_duration(self, audio_file, desired_duration):
        winsound.PlaySound(audio_file, winsound.SND_ASYNC)
        time.sleep(desired_duration)

    def play_block(self, dev, start_index, stimuli, trigger_vals):
        for i in range(4):
            self.play_audio_for_duration(self.STIM_DIR + stimuli[start_index + i], self.tone_duration)
            self.write_trigger_value(dev, trigger_vals[start_index + i])
            self.high_res_sleep(self.isi_tone * 1000)

        self.high_res_sleep(0.5 * 1000)

        for i in range(2):
            self.play_audio_for_duration(self.STIM_DIR + stimuli[start_index + 4 + i], self.word_duration)
            self.write_trigger_value(dev, trigger_vals[start_index + 4 + i])
            self.high_res_sleep(self.isi_word * 1000)

        self.high_res_sleep(self.ibi * 1000)

    def play_blocks(self, dev, stimuli, trigger_vals):
        for i in range(0, len(stimuli), 6):
            self.play_block(dev, i, stimuli, trigger_vals)

    def find_ftdi_device_by_description(self, target_description):
        num_devices = FTD2XX.createDeviceInfoList()
        for index in range(num_devices):
            device_info = FTD2XX.getDeviceInfoDetail(index)
            if device_info['description'] == target_description:
                return index
        return None

    def read_json_file(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        return data

    def run_scan_logic(self):
        try:
            # Load the details from the JSON file
            details_path = r"_restricted_\utils\details.json"
            details = self.read_json_file(details_path)

            # Extracting the values
            id_value = details["ID"]
            date_value = details["Date of Scan"]
            scan_number_value = details["Scan number"]

            stimuli, trigger_vals = [], []

            if scan_number_value in ["1", "2", "3"]:
                self.TXT_FILE = f"_restricted_\stims_struct\chn_{scan_number_value}.txt"
            else:
                error_dialog = QMessageBox(self)
                error_dialog.setIcon(QMessageBox.Warning)
                error_dialog.setWindowTitle('Warning!')
                error_dialog.setText("Invalid scan number. Please enter 1, 2, or 3.")
                error_dialog.exec_()
                return

            with open(self.TXT_FILE, 'r') as file:
                for line in file:
                    elements = line.strip().split(', ')
                    if len(elements) == 0 or len(elements) == 1:
                        elements = line.strip().split(',')
                    for i, element in enumerate(elements):
                        if i % 2 == 0:
                            stimuli.append(element + ".wav")
                        else:
                            trigger_vals.append(int(element))

            target_description = b'TTL232RG-VSW3V3'
            device_index = self.find_ftdi_device_by_description(target_description)
            ftd = FTD2XX.open(device_index)
            self.high_res_sleep(50)
            ftd.setBitMode(0xFF, 1)
            self.high_res_sleep(50)

            # EEG device configuration
            d = g.GDS()
            minf_s = sorted(d.GetSupportedSamplingRates()[0].items())[1]
            d.SamplingRate, d.NumberOfScans = minf_s
            d.Trigger = 1
            for ch in d.Channels:
                ch.Acquire = True
                ch.BipolarChannel = -1
                ch.BandpassFilterIndex = -1
                ch.NotchFilterIndex = -1
            d.SetConfiguration()

            samples = []

            eeg_thread = threading.Thread(target=self.collect_eeg_data, args=(d, samples, self.eeg_collection_duration))
            eeg_thread.start()

            self.high_res_sleep(1000)
            play_blocks_thread = threading.Thread(target=self.play_blocks, args=(ftd, stimuli, trigger_vals))
            play_blocks_thread.start()

            eeg_thread.join()
            play_blocks_thread.join()

            ftd.close()
            samples_data = np.concatenate(samples, axis=0)

            path = r"user_access"
            os.makedirs(path, exist_ok=True)
            filename = f'{id_value}_{date_value}_{scan_number_value}.npy'
            full_path = os.path.join(path, filename)
            np.save(full_path, samples_data)

            d.Close()
            del d

        except Exception as e:
            error_dialog = QMessageBox(self)
            error_dialog.setIcon(QMessageBox.Critical)
            error_dialog.setWindowTitle('Error!')
            error_dialog.setText(f"An error occurred: {str(e)}")
            error_dialog.exec_()

class CustomAcqEEG(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        main_layout = QVBoxLayout()

        # File Details section
        self.file_details = FileDetails()
        main_layout.addWidget(QLabel("File Details"))
        main_layout.addWidget(self.file_details)

        # Impedance Status section
        self.impedance_checker_widget = ImpedanceCheckerWidget()
        main_layout.addWidget(QLabel("Impedance Status"))
        main_layout.addWidget(self.impedance_checker_widget)

        # Collect Data section
        self.collect_data = CollectData()
        main_layout.addWidget(QLabel("Collect Data"))
        main_layout.addWidget(self.collect_data)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    # Initially, only the Save button is enabled. All other buttons are disabled.
        self.file_details.save_button.setEnabled(True)
        self.impedance_checker_widget.check_impedance_button.setDisabled(True)
        self.collect_data.run_scan_button.setDisabled(True)

        # Connecting Save button signal to activate impedance check and then disable Save button
        self.file_details.save_button.clicked.connect(self.activate_impedance_check)
        self.file_details.save_button.clicked.connect(lambda: self.file_details.save_button.setDisabled(True))

        # Connecting Check Impedance button signal to the impedance check function
        self.impedance_checker_widget.check_impedance_button.clicked.connect(self.check_impedance)

        # Window properties
        self.setGeometry(100, 100, 500, 300)
        self.setWindowTitle('CustomAcqEEG')
        self.show()

    def activate_impedance_check(self):
        # Enable impedance checking button after Save button is clicked
        self.impedance_checker_widget.check_impedance_button.setEnabled(True)

    def check_impedance(self):
        imp_colors = self.param()

        # Update labels with the impedance colors
        for label, color in zip(self.impedance_checker_widget.labels, imp_colors):
            label.setStyleSheet(f'background-color: {color}; border: 1px solid black; width: 50px; text-align: center;')

        valid_colors = {"green", "orange"}

        # Stop impedance checking when the impedance colors are all orange, all green or a mix of green and orange
        if len(set(imp_colors).difference(valid_colors)) == 0:
            # Once impedance check is complete, enable Run Scan and Post Process buttons and disable impedance check button
            self.collect_data.run_scan_button.setEnabled(True)
            self.impedance_checker_widget.check_impedance_button.setDisabled(True)

        # Save impedance_colors to text file
        with open(r"_restricted_\utils\imps.txt", 'a') as file:
            for color in imp_colors:
                file.write(f"{color}\n")

    def param(self):
        impedance_colors = []

        try:
            # Initialize GDS
            d = g.GDS()
            imps = d.GetImpedance()
            imps = imps[0]

            imp_values = [imps[0], imps[1], imps[3]]
            imp_values = [imp / 1000 for imp in imp_values]

            for imp in imp_values:
                if imp < 5:
                    color = "green"
                elif imp >= 5 and imp <= 20:
                    color = "orange"
                else:
                    color = "red"
                impedance_colors.append(color)

            d.Close()
            del d

        except Exception as e:
            impedance_colors = ["blue", "blue", "blue"]
            error_dialog = QMessageBox(self)
            error_dialog.setIcon(QMessageBox.Critical)
            error_dialog.setWindowTitle('Error!')
            error_dialog.setText(f"An error occurred: {str(e)}")
            error_dialog.exec_()
            # Optionally, remove the following line if you don't want to delete 'd' again.
            del d
        return impedance_colors




app = QApplication(sys.argv)
checker = CustomAcqEEG()
sys.exit(app.exec_())
