#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 16:41:28 2020

@author: nicklauersdorf
"""

import tkinter as tk
from tkinter import filedialog, Text
import os
import rangeslider

from tkcalendar import Calendar, DateEntry

def example1():
    def print_sel():
        print(cal.selection_get())

    top = tk.Toplevel(root)

    cal = Calendar(top,
                   font="Arial 14", selectmode='day',
                   cursor="hand1", year=2018, month=2, day=5)
    cal.pack(fill="both", expand=True)
    tk.Button(top, text="ok", command=print_sel).pack()

def example2():
    top = tk.Toplevel(root)

    tk.Label(top, text='Choose date').pack(padx=10, pady=10)

    cal = DateEntry(top, width=12, background='darkblue',
                    foreground='black', borderwidth=2)
    cal.pack(padx=10, pady=10)



root=tk.Tk()
apps = []


def addApp():
    
    for widget in frame.winfo_children():
        widget.destroy()
        
    filename=filedialog.askopenfilename(initialdir="/",title="Select File", 
                                        filetypes=(("CSV Files", "*.csv"), ("all files", "*.*")))
    apps.append(filename)
    for app in apps:
        label = tk.Label(frame, text=app, bg="gray")
        label.pack()
    
canvas=tk.Canvas(root, height=700, width=1200, bg="#263D42")
                 
canvas.pack()

# create the main sections of the layout, 
# and lay them out
top = tk.Frame(root, bg="white")
bottom = tk.Frame(root, bg="white")
top.pack(side=tk.TOP)
bottom.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

frame=tk.Frame(root, bg="white")
frame.place(relwidth=0.75, relheight=0.6, relx=0.19, rely=0.05)

frame=tk.Frame(root, bg="white")
frame.place(relwidth=0.75, relheight=0.6, relx=0.19, rely=0.05)

openFile = tk.Button(root, text="Open File", padx=10, pady=5, fg="black", bg="#263D42", command=addApp)
openFile.pack(in_=top, side=tk.LEFT)

runApps = tk.Button(root, text="Run Apps", padx=10, pady=5, fg="black", bg="#263D42")
runApps.pack(in_=top, side=tk.LEFT)

lb = tk.Listbox(root, height=8)
lb.pack(side=tk.RIGHT, fill=tk.BOTH)
lb.place(relx=0.09, rely=0.14, anchor=tk.CENTER)
lb.insert(tk.END,"Total Concurrent Users")
lb.insert(tk.END,"Avg Concurrent Users")
lb.insert(tk.END,"Total Entries")
lb.insert(tk.END,"Avg Entries")
lb.insert(tk.END,"Visitation Purpose")
lb.insert(tk.END,"Visitation Growth")
lb.insert(tk.END,"Visits per Week")
lb.insert(tk.END,"Entry/Exit Times")

end_date=tk.Button(root, text='Set Ending Date', command=example2)
end_date.pack(in_=bottom, side=tk.RIGHT)

start_date=tk.Button(root, text='Set Starting Date', command=example2)
start_date.pack(in_=bottom, side=tk.RIGHT)

var1 = tk.IntVar()
checkbut1=tk.Checkbutton(root, text="male", variable=var1)
checkbut1.pack(in_=bottom, side=tk.RIGHT)

	
'''

rangeslider.RangeSlider(root, 
						 lowerBound = 0, upperBound = 31, 
						 initialLowerBound = 0, initialUpperBound = 31);
date_range = tk.Scale(root, from_=0, to=100)
date_range.pack()
date_range.place(relx=0.09, rely=0.8, anchor=tk.CENTER)
'''
root.mainloop()