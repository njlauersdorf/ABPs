#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 15:52:02 2020

@author: nicklauersdorf
"""

# This is the import cell
import sys
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches


def minute(s):
    return s[-2:]

def hour(s):
    return s[-5:-3]

def year(s):
    return s[-8:-6]

def day(s):
    return s[-11:-9]

def month(s):
    return s[-14:-12]


first = True

# Here are my rc parameters for matplotlibf
fsize = 20
mpl.rc('font', serif='Helvetica Neue') 
mpl.rcParams.update({'font.size': fsize})
mpl.rcParams['figure.figsize'] = 3.2, 2.8
mpl.rcParams['figure.dpi'] = 2000
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['axes.linewidth'] = 1.5
# Set x tick params
mpl.rcParams['xtick.major.size'] = 4.5
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['xtick.minor.size'] = 3.
mpl.rcParams['xtick.minor.width'] = 1.25
# Set y tick params
mpl.rcParams['ytick.major.size'] = 4.5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['ytick.minor.size'] = 3.
mpl.rcParams['ytick.minor.width'] = 1.25
# Load LaTeX and amsmath
# mpl.rc('text', usetex=True)
# mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]


import csv
import numpy as np

#Open saved swipe database
with open('/Volumes/External/BEAMCurrent.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    
    #Define array to translate loaded integer values of facility to string name
    facility_desc_dict={'Murray Hall': 1,
                        'Hanes Art Center': 2,
                        'Carmichael Dorm': 3,
                        'Kenan Science Library': 4,
                        'Test Facility': 5,
                        'CHANL Cleanroom': 6}
    
    #Define array to translate loaded integer values of purpose to string name
    purpose_desc_dict={'Workshop': 1,
                        'Personal project': 2,
                        'Product/business development': 3,
                        'Research': 4,
                        'Class assignment': 5,
                        'Training': 6,
                        'Consultation': 7}
    
    #Initiate line counter
    line_count = 0
    
    #Crate list of rows
    list_rows=list((spamreader))
    total_rows=(len(list_rows))
    PID_arr=np.zeros(len(list_rows))
    date_enter_arr = ["" for i in range(total_rows)]
    date_enter_arr_day = ["" for i in range(total_rows)]
    date_enter_arr_month = ["" for i in range(total_rows)]
    date_enter_arr_year = ["" for i in range(total_rows)]
    date_enter_arr_minute = ["" for i in range(total_rows)]
    date_enter_arr_hour = ["" for i in range(total_rows)]
    
    date_exit_arr = ["" for i in range(total_rows)]
    date_exit_arr_day = ["" for i in range(total_rows)]
    date_exit_arr_month = ["" for i in range(total_rows)]
    date_exit_arr_year = ["" for i in range(total_rows)]
    date_exit_arr_minute = ["" for i in range(total_rows)]
    date_exit_arr_hour = ["" for i in range(total_rows)]
    facility_id_arr=np.zeros(len(list_rows))
    purpose_id_arr=np.zeros(len(list_rows))
    visit_id_arr = ["" for i in range(total_rows)]

    for row in list_rows:
        if line_count == 0:
            line_count += 1
        else:
            PID_arr[line_count-1]=row[1]
            date_enter_arr[line_count-1]=row[2]
            date_exit_arr[line_count-1]=row[3]
            facility_id_arr[line_count-1]=row[6]
            purpose_id_arr[line_count-1]=row[10]
            visit_id_arr[line_count-1]=row[14]
            line_count += 1
            
            date_enter_arr_minute[line_count-1]=minute(row[2])
            date_enter_arr_hour[line_count-1]=hour(row[2])
            date_enter_arr_day[line_count-1]=day(row[2])
            date_enter_arr_year[line_count-1]=year(row[2])
            date_enter_arr_month[line_count-1]=month(row[2])
            
            date_exit_arr_minute[line_count-1]=minute(row[3])
            date_exit_arr_hour[line_count-1]=hour(row[3])
            date_exit_arr_day[line_count-1]=day(row[3])
            date_exit_arr_year[line_count-1]=year(row[3])
            date_exit_arr_month[line_count-1]=month(row[3])
            
            
    stop
