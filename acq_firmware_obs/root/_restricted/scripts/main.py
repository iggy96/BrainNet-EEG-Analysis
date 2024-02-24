from fn_libs import *
from window_login import *
from window_pages import *

def main():
    current_process = psutil.Process(os.getpid())
    current_process.nice(psutil.REALTIME_PRIORITY_CLASS)
    app = QApplication(sys.argv)
    QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    icon_path = r"..\assets\imgs\bnnICON.ico"
    app.setWindowIcon(QIcon(icon_path))
    # Set a global light theme stylesheet
    app.setStyleSheet("""
    QWidget {
        background-color: ivory;
        color: black;
    }
   
    QLineEdit {
        background-color: #f0f0f0;
        border: 1px solid #cccccc;
        padding: 0 8px;
        selection-background-color: #3daee9;
    }
    QPushButton {
        background-color: #5a189a;
        border: 1px solid #5a189a;
        padding: 5px;
        color: white;
    }
    QPushButton:hover {
        background-color: #7b2cbf;
        border: 1px solid #7b2cbf;
    }
    QPushButton:pressed {
        background-color: #8c43e6;
        border: 1px solid #8c43e6;
    }
    QLabel {
        background-color: transparent;
    }
    QMessageBox {
        background-color: white;
        color: black;
    }
    """)

    login_window = LoginWindow()
    if login_window.exec_() == QDialog.Accepted:
        main_window = MainWindow()
        main_window.show()
        sys.exit(app.exec_())

if __name__ == '__main__':
    main()