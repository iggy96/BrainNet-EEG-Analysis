# Run this script through terminal/command prompt
import os
import sys
import time
import pygds as g
import psutil


def param():
    # Set the current process priority to real-time
    current_process = psutil.Process(os.getpid())
    current_process.nice(psutil.REALTIME_PRIORITY_CLASS)

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

    except:
        impedance_colors = ["blue", "blue", "blue"]

    return impedance_colors

while True:
    imp_colors = param()

    # Save impedance_colors to text file
    with open(r"..\assets\utils\imps.txt", 'a') as file:
        for color in imp_colors:
            file.write(f"{color}\n")

    # Check termination conditions
    if "red" not in imp_colors and "blue" not in imp_colors:
        break
    else:
        time.sleep(5)  # optional sleep for 5 seconds
        os.execv(sys.executable, ['python'] + sys.argv)

