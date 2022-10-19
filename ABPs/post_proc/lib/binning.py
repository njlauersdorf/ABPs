import sys, os
from gsd import hoomd
import freud, numpy as np, math
from scipy.interpolate import interp1d
from scipy import interpolate
from scipy import ndimage
import numpy as np, matplotlib
import matplotlib.pyplot as plt
import matplotlib.collections
from matplotlib.patches import Circle
from matplotlib import pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Ellipse
from matplotlib import collections as mc
import matplotlib.colors as colors
import matplotlib.patches as mpatches
from matplotlib import cm
import matplotlib.patches as patches
import matplotlib.ticker as tick
from scipy.optimize import curve_fit
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))
import theory, utility

class binning:

    def __init__(self, l_box, partNum, NBins, peA, peB, typ, eps):
        self.theory_functs = theory.theory()
        self.l_box = l_box
        self.h_box = self.l_box / 2
        utility_functs = utility.utility(self.l_box)
        self.partNum = partNum
        self.min_size = int(self.partNum / 8)
        try:
            self.NBins = int(NBins)
        except:
            print('NBins must be either a float or an integer')
        self.sizeBin = utility_functs.roundUp(self.l_box / self.NBins, 6)
        self.peA = peA
        self.peB = peB
        self.r_cut=2**(1/6)                  #Cut off interaction radius (Per LJ Potential)
        self.eps = eps
        self.typ = typ

        self.utility_functs = utility.utility(self.l_box)


    def create_bins(self):
        pos_box_x_mid = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        pos_box_y_mid = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        pos_box_x_left = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        pos_box_y_bot = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        for ix in range(0, self.NBins):
            for iy in range(0, self.NBins):
                pos_box_x_mid[ix][iy] = (ix + 0.5) * self.sizeBin
                pos_box_y_mid[ix][iy] = (iy + 0.5) * self.sizeBin
                pos_box_x_left[ix][iy] = ix * self.sizeBin
                pos_box_y_bot[ix][iy] = iy * self.sizeBin
        pos_dict = {'bottom left':{'x':pos_box_x_left,
          'y':pos_box_y_bot},
         'mid point':{'x':pos_box_x_mid,  'y':pos_box_y_mid}}
        return pos_dict

    def bin_parts(self, pos, ids, clust_size):
        binParts = [[[] for b in range(self.NBins)] for a in range(self.NBins)]
        typParts = [[[] for b in range(self.NBins)] for a in range(self.NBins)]
        occParts = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        for k in range(0, len(ids)):
            tmp_posX = pos[k][0] + self.h_box
            tmp_posY = pos[k][1] + self.h_box
            x_ind = int(tmp_posX / self.sizeBin)
            y_ind = int(tmp_posY / self.sizeBin)
            binParts[x_ind][y_ind].append(k)
            typParts[x_ind][y_ind].append(self.typ[k])
            if clust_size[ids[k]] >= self.min_size:
                occParts[x_ind][y_ind] = 1
        part_dict = {'clust':occParts,  'typ':typParts,  'id':binParts}
        return part_dict

    def bin_align(self, orient_dict):
        p_avg_x = orient_dict['bin']['all']['x']
        p_avg_y = orient_dict['bin']['all']['y']
        p_avg_xA = orient_dict['bin']['A']['x']
        p_avg_yA = orient_dict['bin']['A']['y']
        p_avg_xB = orient_dict['bin']['B']['x']
        p_avg_yB = orient_dict['bin']['B']['y']
        align_avg_x = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        align_avg_y = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        align_avg_mag = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        align_norm_x = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        align_norm_y = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        align_norm_xA = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        align_norm_yA = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        align_norm_xB = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        align_norm_yB = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        align_avg_xA = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        align_avg_yA = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        align_avg_magA = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        align_avg_xB = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        align_avg_yB = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        align_avg_magB = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        align_avg_magDif = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        align_avg_num = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        align_tot_x = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        align_tot_y = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        align_tot_xA = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        align_tot_yA = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        align_tot_xB = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        align_tot_yB = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        for ix in range(0, self.NBins):
            if ix + 2 == self.NBins:
                lookx = [
                 ix - 1, ix - 1, ix, ix + 1, 0]
            else:
                if ix + 1 == self.NBins:
                    lookx = [
                     ix - 2, ix - 1, ix, 0, 1]
                else:
                    if ix == 0:
                        lookx = [
                         self.NBins - 2, self.NBins - 1, ix, ix + 1, ix + 2]
                    else:
                        if ix == 1:
                            lookx = [
                             self.NBins - 1, ix - 1, ix, ix + 1, ix + 2]
                        else:
                            lookx = [
                             ix - 2, ix - 1, ix, ix + 1, ix + 2]
            for iy in range(0, self.NBins):
                if iy + 2 == self.NBins:
                    looky = [
                     iy - 1, iy - 1, iy, iy + 1, 0]
                else:
                    if iy + 1 == self.NBins:
                        looky = [
                         iy - 2, iy - 1, iy, 0, 1]
                    else:
                        if iy == 0:
                            looky = [
                             self.NBins - 2, self.NBins - 1, iy, iy + 1, iy + 2]
                        else:
                            if iy == 1:
                                looky = [
                                 self.NBins - 1, iy - 1, iy, iy + 1, iy + 2]
                            else:
                                looky = [
                                 iy - 2, iy - 1, iy, iy + 1, iy + 2]
                for indx in lookx:
                    for indy in looky:
                        align_tot_x[ix][iy] += p_avg_x[indx][indy]
                        align_tot_y[ix][iy] += p_avg_y[indx][indy]
                        align_avg_num[ix][iy] += 1
                        align_tot_xA[ix][iy] += p_avg_xA[indx][indy]
                        align_tot_yA[ix][iy] += p_avg_yA[indx][indy]
                        align_tot_xB[ix][iy] += p_avg_xB[indx][indy]
                        align_tot_yB[ix][iy] += p_avg_yB[indx][indy]

                    if align_avg_num[ix][iy] > 0:
                        align_avg_x[ix][iy] = align_tot_x[ix][iy] / align_avg_num[ix][iy]
                        align_avg_y[ix][iy] = align_tot_y[ix][iy] / align_avg_num[ix][iy]
                        align_avg_xA[ix][iy] = align_tot_xA[ix][iy] / align_avg_num[ix][iy]
                        align_avg_yA[ix][iy] = align_tot_yA[ix][iy] / align_avg_num[ix][iy]
                        align_avg_xB[ix][iy] = align_tot_xB[ix][iy] / align_avg_num[ix][iy]
                        align_avg_yB[ix][iy] = align_tot_yB[ix][iy] / align_avg_num[ix][iy]
                        align_avg_mag[ix][iy] = (align_avg_x[ix][iy] ** 2 + align_avg_y[ix][iy] ** 2) ** 0.5
                        align_avg_magA[ix][iy] = (align_avg_xA[ix][iy] ** 2 + align_avg_yA[ix][iy] ** 2) ** 0.5
                        align_avg_magB[ix][iy] = (align_avg_xB[ix][iy] ** 2 + align_avg_yB[ix][iy] ** 2) ** 0.5
                        align_avg_magDif[ix][iy] = align_avg_magB[ix][iy] - align_avg_magA[ix][iy]
        align_dict = {'bin': {'all':{'x':align_avg_x, 'y':align_avg_y, 'mag':align_avg_mag},
         'A':{'x':align_avg_xA, 'y':align_avg_yA, 'mag':align_avg_magA},  'B':{'x':align_avg_xB, 'y':align_avg_yB, 'mag':align_avg_magB},  'avg dif':{'mag': align_avg_magDif}}}
        return align_dict

    def bin_vel(self, pos, prev_pos, part_dict, dt):

        occParts = part_dict['clust']
        typParts = part_dict['typ']
        binParts = part_dict['id']

        v_all_x = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        v_all_y = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        v_A_x = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        v_A_y = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        v_B_x = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        v_B_y = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        v_avg_x = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        v_avg_y = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        v_avg_xA = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        v_avg_yA = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        v_avg_xB = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        v_avg_yB = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        v_avg_mag = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        v_avg_magA = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        v_avg_magB = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        v_x_part = np.zeros(self.partNum)
        v_y_part = np.zeros(self.partNum)
        v_mag_part = np.zeros(self.partNum)

        for ix in range(0, self.NBins):
            for iy in range(0, self.NBins):
                typ0_temp = 0
                typ1_temp = 0
                if len(binParts[ix][iy]) != 0:
                    for h in binParts[ix][iy]:
                        difx = self.utility_functs.sep_dist(pos[h,0], prev_pos[h,0])

                        dify = self.utility_functs.sep_dist(pos[h,1], prev_pos[h,1])

                        vx = difx/dt
                        vy = dify/dt

                        v_all_x[ix][iy] += vx
                        v_all_y[ix][iy] += vy

                        if self.typ[h] == 0:
                            typ0_temp += 1
                            v_A_x[ix][iy] += vx
                            v_A_y[ix][iy] += vy
                        elif self.typ[h] == 1:
                            typ1_temp += 1
                            v_B_x[ix][iy] += vx
                            v_B_y[ix][iy] += vy

                        v_x_part[h] = vx
                        v_y_part[h] = vy
                        v_mag_part[h] = (vx**2 + vy**2 )**0.5
                if len(binParts[ix][iy])>0:
                    v_avg_x[ix][iy] = v_all_x[ix][iy] / len(binParts[ix][iy])
                    v_avg_y[ix][iy] = v_all_y[ix][iy] / len(binParts[ix][iy])

                    if typ0_temp > 0:
                        v_avg_xA[ix][iy] = v_A_x[ix][iy] / typ0_temp
                        v_avg_yA[ix][iy] = v_A_y[ix][iy] / typ0_temp

                    if typ1_temp > 0:
                        v_avg_xB[ix][iy] = v_B_x[ix][iy] / typ1_temp
                        v_avg_yB[ix][iy] = v_B_y[ix][iy] / typ1_temp

                v_avg_mag[ix][iy] = (v_avg_x[ix][iy] ** 2 + v_avg_y[ix][iy] ** 2) ** 0.5
                v_avg_magA[ix][iy] = (v_avg_xA[ix][iy] ** 2 + v_avg_yA[ix][iy] ** 2) ** 0.5
                v_avg_magB[ix][iy] = (v_avg_xB[ix][iy] ** 2 + v_avg_yB[ix][iy] ** 2) ** 0.5

        vel_dict = {'bin': {'all':{'x':v_avg_x,
          'y':v_avg_y,  'mag':v_avg_mag},
         'A':{'x':v_avg_xA,  'y':v_avg_yA,  'mag':v_avg_magA},  'B':{'x':v_avg_xB,  'y':v_avg_yB,  'mag':v_avg_magB}}, 'part': {'x': v_x_part, 'y': v_y_part, 'mag': v_mag_part}}

        return vel_dict

    def bin_ang_vel(self, ang, prev_ang, part_dict, dt):
        occParts = part_dict['clust']
        typParts = part_dict['typ']
        binParts = part_dict['id']

        ang_v_all = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        ang_v_A = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        ang_v_B = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        ang_v_avg = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        ang_v_avg_A = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        ang_v_avg_B = [[0 for b in range(self.NBins)] for a in range(self.NBins)]

        ang_v_part = np.zeros(self.partNum)
        for ix in range(0, self.NBins):
            for iy in range(0, self.NBins):
                typ0_temp = 0
                typ1_temp = 0
                if len(binParts[ix][iy]) != 0:
                    for h in binParts[ix][iy]:
                        ang_v = ang[h] - prev_ang[h]
                        ang_v_abs = np.abs(ang_v)
                        if ang_v_abs >= np.pi:
                            if ang_v < -np.pi:
                                ang_v += 2*np.pi
                            else:
                                ang_v -= 2*np.pi
                        print(ang_v)
                        ang_v_all[ix][iy] += ang_v/dt
                        if self.typ[h] == 0:
                            typ0_temp += 1
                            ang_v_A[ix][iy] += ang_v/dt
                        elif self.typ[h] == 1:
                            typ1_temp += 1
                            ang_v_B[ix][iy] += ang_v/dt

                        ang_v_part[h] = (ang_v/dt)* (180/np.pi)

                if len(binParts[ix][iy])>0:
                    ang_v_avg[ix][iy] = (ang_v_all[ix][iy] / len(binParts[ix][iy]))

                    if typ0_temp > 0:
                        ang_v_avg_A[ix][iy] = (ang_v_A[ix][iy] / typ0_temp)

                    if typ1_temp > 0:
                        ang_v_avg_B[ix][iy] = (ang_v_B[ix][iy] / typ1_temp)


        ang_vel_dict = {'bin': {'all': ang_v_avg,
         'A': ang_v_avg_A,  'B': ang_v_avg_B}, 'part': ang_v_part}
        return ang_vel_dict

    def bin_activity(self, part_dict):
        occParts = part_dict['clust']
        binParts = part_dict['id']
        typParts = part_dict['typ']
        pe_avg = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        for ix in range(0, self.NBins):
            for iy in range(0, self.NBins):
                typ0_temp = 0
                typ1_temp = 0
                pe_sum = 0
                if len(binParts[ix][iy]) != 0:
                    for h in range(0, len(binParts[ix][iy])):
                        if typParts[ix][iy][h] == 0:
                            pe_sum += self.peA
                        else:
                            pe_sum += self.peB
                    pe_avg[ix][iy] = pe_sum / len(binParts[ix][iy])

        activ_dict = {'avg': pe_avg}
        return activ_dict

    def bin_area_frac(self, part_dict):
        occParts = part_dict['clust']
        typParts = part_dict['typ']
        binParts = part_dict['id']
        area_frac = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        area_frac_dif = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        fast_frac_arr = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        area_frac_A = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        area_frac_B = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        for ix in range(0, self.NBins):
            for iy in range(0, self.NBins):
                typ0_temp = 0
                typ1_temp = 0
                if len(binParts[ix][iy]) != 0:
                    for h in range(0, len(binParts[ix][iy])):
                        if typParts[ix][iy][h] == 0:
                            typ0_temp += 1
                        elif typParts[ix][iy][h] == 1:
                            typ1_temp += 1
                    area_frac[ix][iy] = len(binParts[ix][iy]) / self.sizeBin ** 2 * (math.pi / 4)
                    area_frac_A[ix][iy] = typ0_temp / self.sizeBin ** 2 * (math.pi / 4)
                    area_frac_B[ix][iy] = typ1_temp / self.sizeBin ** 2 * (math.pi / 4)
                    fast_frac_arr[ix][iy] = area_frac_B[ix][iy] / area_frac[ix][iy]
                    if self.peB >= self.peA:
                        area_frac_dif[ix][iy] = area_frac_B[ix][iy] - area_frac_A[ix][iy]
                    else:
                        area_frac_dif[ix][iy] = area_frac_A[ix][iy] - area_frac_B[ix][iy]

        area_frac_dict = {'bin': {'all':area_frac,
         'A':area_frac_A,  'B':area_frac_B,  'dif':area_frac_dif,  'fast frac':fast_frac_arr}}
        return area_frac_dict

    def bin_orient(self, part_dict, pos, ang, com):
        occParts = part_dict['clust']
        typParts = part_dict['typ']
        binParts = part_dict['id']
        com_tmp_posX = com['x']
        com_tmp_posY = com['y']
        p_all_x = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        p_all_y = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        p_A_x = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        p_A_y = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        p_B_x = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        p_B_y = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        p_avg_x = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        p_avg_y = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        p_avg_mag = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        p_avg_xA = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        p_avg_yA = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        p_avg_magA = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        p_avg_xB = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        p_avg_yB = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        p_avg_magB = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        bin_part_x = np.zeros(self.partNum)
        bin_part_y = np.zeros(self.partNum)
        for ix in range(0, self.NBins):
            for iy in range(0, self.NBins):
                typ0_temp = 0
                typ1_temp = 0
                if len(binParts[ix][iy]) != 0:
                    for h in range(0, len(binParts[ix][iy])):
                        x_pos = pos[binParts[ix][iy]][h][0] + self.h_box
                        y_pos = pos[binParts[ix][iy]][h][1] + self.h_box
                        difx = self.utility_functs.sep_dist(x_pos, com_tmp_posX)
                        dify = self.utility_functs.sep_dist(y_pos, com_tmp_posY)

                        difr = (difx ** 2 + dify ** 2) ** 0.5
                        px = np.sin(ang[binParts[ix][iy][h]])
                        py = -np.cos(ang[binParts[ix][iy][h]])
                        r_dot_p = -difx * px + -dify * py
                        p_all_x[ix][iy] += px
                        p_all_y[ix][iy] += py
                        if typParts[ix][iy][h] == 0:
                            typ0_temp += 1
                            p_A_x[ix][iy] += px
                            p_A_y[ix][iy] += py
                        elif typParts[ix][iy][h] == 1:
                            typ1_temp += 1
                            p_B_x[ix][iy] += px
                            p_B_y[ix][iy] += py
                        bin_part_x[binParts[ix][iy][h]]=px
                        bin_part_y[binParts[ix][iy][h]]=py
                    p_avg_x[ix][iy] = p_all_x[ix][iy] / len(binParts[ix][iy])
                    p_avg_y[ix][iy] = p_all_y[ix][iy] / len(binParts[ix][iy])
                    if typ0_temp > 0:
                        p_avg_xA[ix][iy] = p_A_x[ix][iy] / typ0_temp
                        p_avg_yA[ix][iy] = p_A_y[ix][iy] / typ0_temp
                    else:
                        p_avg_xA[ix][iy] = 0.0
                        p_avg_yA[ix][iy] = 0.0
                    if typ1_temp > 0:
                        p_avg_xB[ix][iy] = p_B_x[ix][iy] / typ1_temp
                        p_avg_yB[ix][iy] = p_B_y[ix][iy] / typ1_temp
                    else:
                        p_avg_xB[ix][iy] = 0.0
                        p_avg_yB[ix][iy] = 0.0
                    p_avg_mag[ix][iy] = (p_avg_x[ix][iy] ** 2 + p_avg_y[ix][iy] ** 2) ** 0.5
                    p_avg_magA[ix][iy] = (p_avg_xA[ix][iy] ** 2 + p_avg_yA[ix][iy] ** 2) ** 0.5
                    p_avg_magB[ix][iy] = (p_avg_xB[ix][iy] ** 2 + p_avg_yB[ix][iy] ** 2) ** 0.5

        orient_dict = {'bin': {'all':{'x':p_avg_x,
          'y':p_avg_y,  'mag':p_avg_mag},
         'A':{'x':p_avg_xA,  'y':p_avg_yA,  'mag':p_avg_magA},  'B':{'x':p_avg_xB,  'y':p_avg_yB,  'mag':p_avg_magB}}, 'part': {'x': bin_part_x, 'y': bin_part_y}}
        return orient_dict

    def bin_active_press(self, align_dict, area_frac_dict):
        align_mag = align_dict['bin']['all']['mag']
        align_A_mag = align_dict['bin']['A']['mag']
        align_B_mag = align_dict['bin']['B']['mag']
        area_frac = area_frac_dict['bin']['all']
        area_frac_A = area_frac_dict['bin']['A']
        area_frac_B = area_frac_dict['bin']['B']
        press = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        press_A = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        press_B = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        press_dif = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        for ix in range(0, self.NBins):
            for iy in range(0, self.NBins):
                press[ix][iy] = area_frac[ix][iy] * align_mag[ix][iy]
                press_A[ix][iy] = area_frac_A[ix][iy] * align_A_mag[ix][iy]
                press_B[ix][iy] = area_frac_B[ix][iy] * align_B_mag[ix][iy]
                press_dif[ix][iy] = area_frac_B[ix][iy] * align_B_mag[ix][iy] - area_frac_A[ix][iy] * align_A_mag[ix][iy]
        press_dict = {'bin':{'all':press,
         'A':press_A,  'B':press_B,  'dif':press_dif}}
        return press_dict
    def bin_active_fa(self, orient_dict, part_dict, phaseBin):
        orient_x = orient_dict['part']['x']
        orient_y = orient_dict['part']['y']
        binParts = part_dict['id']
        fa_x = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        fa_y = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        fa_mag = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        for ix in range(0, self.NBins):
            for iy in range(0, self.NBins):
                if phaseBin[ix][iy]==1:
                    for h in binParts[ix][iy]:
                        if self.typ[h]==0:
                            fa_x[ix][iy] += orient_x[h] * self.peA
                            fa_y[ix][iy] += orient_y[h] * self.peA
                        else:
                            fa_x[ix][iy] += orient_x[h] * self.peB
                            fa_y[ix][iy] += orient_y[h] * self.peB
                    fa_mag[ix][iy] = ( (fa_x[ix][iy]) ** 2 + (fa_y[ix][iy]) ** 2 ) ** 0.5

        fa_x = fa_x / np.max(fa_mag)
        fa_y = fa_y / np.max(fa_mag)

        act_force_dict = {'bin':{'x': fa_x, 'y': fa_y}}
        return act_force_dict

    def bin_normal_active_fa(self, align_dict, area_frac_dict, activ_dict):
        align_mag = align_dict['bin']['all']['mag']
        align_A_mag = align_dict['bin']['A']['mag']
        align_B_mag = align_dict['bin']['B']['mag']
        area_frac = area_frac_dict['bin']['all']
        area_frac_A = area_frac_dict['bin']['A']
        area_frac_B = area_frac_dict['bin']['B']
        fa = activ_dict['avg']
        press = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        press_A = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        press_B = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        press_dif = [[0 for b in range(self.NBins)] for a in range(self.NBins)]
        for ix in range(0, self.NBins):
            for iy in range(0, self.NBins):
                press[ix][iy] = area_frac[ix][iy] * align_mag[ix][iy] * fa[ix][iy]
                press_A[ix][iy] = area_frac_A[ix][iy] * align_A_mag[ix][iy] * self.peA
                press_B[ix][iy] = area_frac_B[ix][iy] * align_B_mag[ix][iy] * self.peB
                press_dif[ix][iy] = area_frac_B[ix][iy] * align_B_mag[ix][iy] * self.peB - area_frac_A[ix][iy] * align_A_mag[ix][iy] * self.peA
        act_press_dict = {'bin':{'all':press,
         'A':press_A,  'B':press_B,  'dif':press_dif}}
        return act_press_dict

    def bin_interpart_press(self, pos, part_dict):

        binParts = part_dict['id']
        pressure_vp = [[0 for b in range(self.NBins)] for a in range(self.NBins)]

        press_num = [[0 for b in range(self.NBins)] for a in range(self.NBins)]

        for ix in range(0, self.NBins):
            for iy in range(0, self.NBins):
                if len(binParts[ix][iy]) != 0:
                    if ix==0:
                        ix_new_range = [self.NBins-1, 0, 1]
                    elif ix==self.NBins-1:
                        ix_new_range = [self.NBins-2, self.NBins-1, 0]
                    else:
                        ix_new_range = [ix-1, ix, ix+1]

                    if iy==0:
                        iy_new_range = [self.NBins-1, 0, 1]
                    elif iy==self.NBins-1:
                        iy_new_range = [self.NBins-2, self.NBins-1, 0]
                    else:
                        iy_new_range = [iy-1, iy, iy+1]
                    for h in range(0, len(binParts[ix][iy])):
                        #lat_temp = 10000
                        for ix2 in ix_new_range:
                            for iy2 in iy_new_range:
                                if len(binParts[ix2][iy2])!=0:
                                    for h2 in range(0,len(binParts[ix2][iy2])):
                                        if binParts[ix2][iy2][h2] != binParts[ix][iy][h]:

                                            difx = self.utility_functs.sep_dist(pos[binParts[ix][iy]][h][0], pos[binParts[ix2][iy2]][h2][0])

                                            dify = self.utility_functs.sep_dist(pos[binParts[ix][iy]][h][1], pos[binParts[ix2][iy2]][h2][1])

                                            difr=(difx**2+dify**2)**0.5
                                            #if difr2 < lat_temp:
                                            #    lat_temp = difr2
                                            #else:
                                            #    pass
                                            if 0.1<=difr<=self.r_cut:
                                                fx, fy = self.theory_functs.computeFLJ(difr, pos[binParts[ix][iy]][h][0], pos[binParts[ix][iy]][h][1], pos[binParts[ix2][iy2]][h2][0], pos[binParts[ix2][iy2]][h2][1], self.eps)
                                                                        # Compute the x force times x distance
                                                sigx = fx * (difx)
                                                                        # Likewise for y
                                                sigy = fy * (dify)

                                                press_num[ix][iy] += 1
                                                pressure_vp[ix][iy] += ((sigx + sigy) / 2.)


        pressure_vp_avg = [[0 for b in range(self.NBins)] for a in range(self.NBins)]

        for ix in range(0, self.NBins):
            for iy in range(0, self.NBins):
                pressure_vp_avg[ix][iy]=pressure_vp[ix][iy]/(2*self.sizeBin**2)

        press_dict = {'tot': pressure_vp, 'num': press_num, 'avg': pressure_vp_avg}
        return press_dict

    def curl_and_div(self, input_dict):
        all_input_x = input_dict['bin']['all']['x']
        A_input_x = input_dict['bin']['A']['x']
        B_input_x = input_dict['bin']['B']['x']
        all_input_y = input_dict['bin']['all']['y']
        A_input_y = input_dict['bin']['A']['y']
        B_input_y = input_dict['bin']['B']['y']
        tot_combined = np.zeros((self.NBins, self.NBins, 2))
        tot_A_combined = np.zeros((self.NBins, self.NBins, 2))
        tot_B_combined = np.zeros((self.NBins, self.NBins, 2))
        for ix in range(0, self.NBins):
            for iy in range(0, self.NBins):
                tot_combined[ix][iy][0] = all_input_x[ix][iy]
                tot_combined[ix][iy][1] = all_input_y[ix][iy]
                tot_A_combined[ix][iy][0] = A_input_x[ix][iy]
                tot_A_combined[ix][iy][1] = A_input_y[ix][iy]
                tot_B_combined[ix][iy][0] = B_input_x[ix][iy]
                tot_B_combined[ix][iy][1] = B_input_y[ix][iy]
        totx_grad = np.gradient(tot_combined, axis=0)
        toty_grad = np.gradient(tot_combined, axis=1)
        totx_A_grad = np.gradient(tot_A_combined, axis=0)
        toty_A_grad = np.gradient(tot_A_combined, axis=1)
        totx_B_grad = np.gradient(tot_B_combined, axis=0)
        toty_B_grad = np.gradient(tot_B_combined, axis=1)
        dx_over_dx = totx_grad[:, :, 0]
        dx_over_dy = totx_grad[:, :, 1]
        dy_over_dx = toty_grad[:, :, 0]
        dy_over_dy = toty_grad[:, :, 1]
        dx_over_dx_A = totx_A_grad[:, :, 0]
        dx_over_dy_A = totx_A_grad[:, :, 1]
        dy_over_dx_A = toty_A_grad[:, :, 0]
        dy_over_dy_A = toty_A_grad[:, :, 1]
        dx_over_dx_B = totx_B_grad[:, :, 0]
        dx_over_dy_B = totx_B_grad[:, :, 1]
        dy_over_dx_B = toty_B_grad[:, :, 0]
        dy_over_dy_B = toty_B_grad[:, :, 1]
        div = dx_over_dx + dy_over_dy
        curl = -dy_over_dx + dx_over_dy
        div_A = dx_over_dx_A + dy_over_dy_A
        curl_A = -dy_over_dx_A + dx_over_dy_A
        div_B = dx_over_dx_B + dy_over_dy_B
        curl_B = -dy_over_dx_B + dx_over_dy_B

        grad_dict = {'div': {'all': div, 'A': div_A, 'B': div_B}, 'curl': {'all': curl, 'A': curl_A, 'B': curl_B} }
        return grad_dict
    def decrease_bin_size(self, phaseBin, phasePart, binParts, pos, typ, factor = 4):

        end_size = self.sizeBin / factor

        if end_size < self.r_cut:
            raise ValueError('input factor must be large enough so end_size is at least equal to the LJ cut off distance (r_cut)')

        NBins = self.NBins * factor

        phaseBin_new = [[0 for b in range(NBins)] for a in range(NBins)]
        binParts_new = [[[] for b in range(NBins)] for a in range(NBins)]
        typParts_new = [[[] for b in range(NBins)] for a in range(NBins)]

        pos_box_x_mid = np.array([])
        pos_box_y_mid = np.array([])
        id_box_x = np.array([], dtype=int)
        id_box_y = np.array([], dtype=int)

        for ix in range(0, NBins):
            for iy in range(0, NBins):
                pos_box_x_mid = np.append(pos_box_x_mid, (ix + 0.5)* end_size)
                pos_box_y_mid = np.append(pos_box_y_mid, (iy * 0.5) * end_size)

                id_box_x = np.append(id_box_x, ix)
                id_box_y = np.append(id_box_y, iy)

        for ix in range(0, self.NBins):
            for iy in range(0, self.NBins):
                pos_box_x_min = (ix) * self.sizeBin
                pos_box_x_max = (ix + 1) * self.sizeBin
                pos_box_y_min = (iy) * self.sizeBin
                pos_box_y_max = (iy + 1) * self.sizeBin


                bin_loc = np.where(((pos_box_x_mid>=pos_box_x_min) & (pos_box_x_mid<=pos_box_x_max)) & ((pos_box_y_mid>=pos_box_y_min) & (pos_box_y_mid<=pos_box_y_max)))[0]

                for id in bin_loc:
                    phaseBin_new[id_box_x[int(id)]][id_box_y[int(id)]] = phaseBin[ix][iy]

                for h in binParts[ix][iy]:

                    tmp_posX = pos[h][0] + self.h_box
                    tmp_posY = pos[h][1] + self.h_box

                    x_ind = int(tmp_posX / end_size)
                    y_ind = int(tmp_posY / end_size)

                    binParts_new[x_ind][y_ind].append(h)
                    typParts_new[x_ind][y_ind].append(typ[h])
        """
        for ix in range(0, self.NBins):
            for iy in range(0, self.NBins):
                pos_box_x_left = np.append(pos_box_x_left, ix * sizeBin)
                pos_box_y_bot = np.append(pos_box_y_bot, iy * sizeBin)

                id_box_x = np.append(pos_box_x_mid, ix)
                id_box_y = np.append(pos_box_y_mid, iy)

        for ix in range(0, NBins):
            for iy in range(0, NBins):
                pos_box_x_mid_val = (ix + 0.5) * sizeBin
                pos_box_y_mid_val = (iy + 0.5) * sizeBin

                bin_loc = np.where(((pos_dict['bottom left']['x']<=pos_box_x_mid_val) & (pos_box_x_mid_val<=pos_dict['bottom left']['x']+self.sizeBin)) & ((pos_dict['bottom left']['y']<=pos_box_y_mid_val) & (pos_box_y_mid_val<=pos_dict['bottom left']['y']+self.sizeBin)))[0]

                phaseBin_new[ix][iy] = phaseBin[id_box_x[bin_loc]][id_box_y[bin_loc]]
                for h in range(0, len(binParts[id_box_x[bin_loc]][id_box_y[bin_loc]])):
                    tmp_posX = self.pos[h][0] + self.h_box
                    tmp_posY = self.pos[h][1] + self.h_box

                    x_ind = int(tmp_posX / end_size)
                    y_ind = int(tmp_posY / end_size)

                    binParts_new[x_ind][y_ind].append(binParts[ix][iy][h])
                    typParts_new[x_ind][y_ind].append(typ[binParts[ix][iy][k])

        """
        part_dict = {'typ':typParts_new,  'id':binParts_new}
        phase_dict = {'bin': phaseBin_new, 'part': phasePart}
        return phase_dict, part_dict
