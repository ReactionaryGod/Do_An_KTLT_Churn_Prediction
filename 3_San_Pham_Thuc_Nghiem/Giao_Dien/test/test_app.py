import sys
import os
from PyQt6.QtWidgets import QApplication

current_dir = os.path.dirname(os.path.abspath(__file__))
giao_dien_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(giao_dien_dir)

from ui.MainWindowEx import MainWindowEx

def main():
    app = QApplication(sys.argv)
    window = MainWindowEx()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()