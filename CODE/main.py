"""
    main.py

    This file contains the master control for the motherhive architecture.
    It contains an UI that will allow the user to comuniate with the SST
    machine. 
    
    It allows the user to preset a time for runining the monitoring code,
    as well as an overide option to run the device whenever the user want.
    It displays the output for the latest diagnosis and an option to clear
    the last output.

    REQIRED LIBRARIES:
        SST*
        tkinter

    *SST has library requirements of its own.

    Jose Antonio Kautau Toffoli 
    2022-11-18
"""

### IMPORTS
import sys
import os
import time

import SST as sst
import tkinter as tk 

### USER INTERFACE 
class Window(tk.Frame):

    def __init__(self, root):

        global run_time, results

        run_time = (0,0)
        results = list()

        tk.Frame.__init__(self, root)
        self.root = root

        ### SET TIME
        settingButton = tk.Button(root, text="Set Run Time", command=self.set_run_time)
        settingButton.pack(padx = 50, pady = 10)

        self.run_time_text = tk.Label(root, text=f"Run Time 00:00")
        self.run_time_text.pack(padx = 40, pady = 5)
        self.run_time_text.place()

        ### OUTPUT TABLE
        outButton = tk.Button (root, text="Output", command=self.display_output_table)
        outButton.pack (padx = 50, pady = 10)

        ### RUN NOW
        runNowButton = tk.Button (root, text="Run Now", command=self.run_now)
        runNowButton.pack (padx = 60, pady = 10)

        ### CLEAR OUTPUT TABLE
        outButton = tk.Button (root, text="Clear Output", command=self.clear_output_table)
        outButton.pack (padx = 60, pady = 10)

    ### ACTIONS
    def set_run_time (self):

        global run_time

        self.settings()
        self.run_time_text.config(text=f"Run Time {run_time[0]}:{run_time[1]}")
    
    def run_now (self):
        
        global results

        print("Running")

        #results = [["Gary is cute", 1], ["Spidermite", 0.7]]
        results = sst.run_sst()

    def display_output_table (self):

        global results
        
        self.outTblWindow = tk.Toplevel()

        self.outTblWindow.title("Output")
        self.outTblWindow.geometry("450x500")

        lst = ["Layer", "Diagnosis", "Probability"]

        for i in range (3):
            self.headerLabel = tk.Label(self.outTblWindow, width = 10, fg='blue', font=('Arial',16,'bold'), text=str(lst[i]), borderwidth=1 ).place(x=150*i,y=0)

        for r in range(len(results)):
            self.layerLabel = tk.Label(self.outTblWindow, width = 10, fg='blue', font=('Arial',16,'bold'), text=str(r+1), borderwidth=1 ).place(x=0,y=30*(r+1))
            self.diagnosisLabel = tk.Label(self.outTblWindow, width = 10, fg='blue', font=('Arial',16,'bold'), text=str(results[r][0][0]), borderwidth=1 ).place(x=150*(1),y=30*(r+1))
            self.probLabel = tk.Label(self.outTblWindow, width = 10, fg='blue', font=('Arial',16,'bold'), text=str(results[r][0][1]), borderwidth=1 ).place(x=150*(2),y=30*(r+1))

    def clear_output_table (self):

        global results

        results.clear()
        self.layerLabel.destroy()
        self.diagnosisLabel.destroy()
        self.probLabel.destroy()


    ### SUPPORT
    def settings (self):
        
        global run_time

        settingsWindow = tk.Toplevel()
        settingsWindow.title("Run Time Setting")
        settingsWindow.geometry("100x150")
        tk.Label(settingsWindow, text ="Run Time").grid(row=0)

        hours = tk.StringVar()
        note = tk.Entry(settingsWindow, width=10, textvariable=hours).grid(row=5,padx=25,pady=5)

        minutes = tk.StringVar()
        note = tk.Entry(settingsWindow, width=10, textvariable=minutes).grid(row=6,padx=25,pady=5)

        Button = tk.Button(settingsWindow, text ="Submit Data", command = settingsWindow.destroy).grid(row=7)

        settingsWindow.wait_window()

        run_time = ( int(hours.get()), int(minutes.get())) 

def check_start ():
            
    global run_time, results
    
    current_time = (int(time.strftime("%H")), int(time.strftime("%M")))

    if (run_time[0] == current_time[0] and run_time[1] == current_time[1]):
        #print ("Running")
        results = sst.run_sst()

### MAIN CODE STARTS HERE
def main (args):    
    
    global run_time

    root = tk.Tk()
    root.geometry("400x225")
    root.title("SSTv1.0")

    mainWindow = Window(root).pack(side="top", fill="both", expand=True)

    while True:
        
        check_start()

        root.update_idletasks()
        root.update()

if __name__ == '__main__':

    if (not(os.path.exists('/home/pi/Desktop/SSTV1.0/pineapple.jpg'))):
        quit()
        
    sys.exit(main(sys.argv))