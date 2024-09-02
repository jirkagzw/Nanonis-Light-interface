# -*- encoding: utf-8 -*-
'''
@Time    :   2023/03/05 15:51:09
@Author  :   Shixuan Shan 
'''
import pickle
import pandas as pd
from os import mkdir
from os.path import exists
import time
import numpy as np

class photon_meas:
    def __init__(self, connect,connect2): #connect2 = andor
        self.connect = connect
        return
    
    
    def v(self, bias_mV):
        """
        Set the bias voltage.
    
        Parameters
        ----------
        bias_mV : float
            The bias voltage in millivolts to set.
    
        Notes
        -----
        This function updates the bias voltage by sending a string formatted with the value and unit.
        """
        self.connect.BiasSet(str(bias_mV) + 'm')
    
    
    def i(self, current_nA):
        """
        Set the current.
    
        Parameters
        ----------
        current_nA : float
            The current in nanoamperes to set.
    
        Notes
        -----
        This function updates the current by sending a string formatted with the value and unit.
        """
        self.connect.ZCtrlSetpntSet(str(current_nA) + 'n')
    
    
    def pix(self, *pixs):
        """
        Set the pixel position for scanning.
    
        Parameters
        ----------
        *pixs : float
            One or two values representing the x and y coordinates of the pixel. If only one value is
            provided, it will be used for both x and y coordinates.
    
        Notes
        -----
        This function retrieves the current scan buffer, and sets the pixel position for scanning
        based on the provided coordinates.
        """
        for i in range(len(pixs)):
            if i == 0:
                pix_x = pixs[0]
                pix_y = pixs[0]
            elif i == 1:
                pix_y = pixs[1]
    
        df = self.connect.ScanBufferGet()
        num_chs = int(df.values[0][0])
        ch_idx = df.values.tolist()[1]
        self.connect.ScanBufferSet(num_chs, ch_idx, int(pix_x), int(pix_y))
    
    
    def dim(self, *dims, **kwargs):
        """
        Set the scan dimensions.
    
        Parameters
        ----------
        *dims : float
            One or two values representing the dimensions in nanometers. If only one value is provided,
            it will be used for both x and y dimensions.
        
        **kwargs : keyword arguments, optional
            - angle : float
                The angle in degrees for the scan frame. If not provided, it will default to the current angle.
    
        Notes
        -----
        This function updates the scan frame dimensions and angle based on the provided values.
        """
        for i in range(0, len(dims)):
            if i == 0:
                dim_x_nm = dims[0]
                dim_y_nm = dims[0]
            elif i == 1:
                dim_y_nm = dims[1]
    
        df = self.connect.ScanFrameGet()
        center_x = df.values[0][0]
        center_y = df.values[1][0]
        angle = kwargs.get("angle", df.values[4][0])
        self.connect.ScanFrameSet(center_x, center_y, 1e-9 * dim_x_nm, 1e-9 * dim_y_nm, angle)
    
    
    def dpos(self, dx_nm, dy_nm):
        """
        Move the scan frame by a delta position.
    
        Parameters
        ----------
        dx_nm : float
            The displacement in nanometers along the x-axis.
        
        dy_nm : float
            The displacement in nanometers along the y-axis.
    
        Notes
        -----
        This function calculates the new position of the scan frame considering the current angle and updates it.
        """
        df = self.connect.ScanFrameGet()
        center_x = df.values[0][0]
        center_y = df.values[1][0]
        angle = df.values[4][0]
        dx_rot = 1e-9 * (dx_nm * np.cos(-angle) + dy_nm * np.sin(-angle))
        dy_rot = 1e-9 * (-dx_nm * np.sin(-angle) + dy_nm * np.cos(-angle))
        new_center_x = center_x + dx_rot
        new_center_y = center_y + dy_rot
        self.connect.ScanFrameSet(new_center_x, new_center_y, df.values[2][0], df.values[3][0], angle)
    
    
    def mv(self, dx_nm, dy_nm, **kwargs):
        """
        Move the scan frame by a delta position and optionally wait.
    
        Parameters
        ----------
        dx_nm : float
            The displacement in nanometers along the x-axis.
        
        dy_nm : float
            The displacement in nanometers along the y-axis.
    
        **kwargs : keyword arguments, optional
            - wait : bool, optional
                If True (default), the function will wait until the movement is complete. If False, it will not wait.
    
        Notes
        -----
        This function calculates the new position of the scan frame considering the current angle and updates it.
        The `wait` parameter controls whether the function should block until the movement is complete.
        """
        df = self.connect.ScanFrameGet()
        center_x = df.values[0][0]
        center_y = df.values[1][0]
        angle = df.values[4][0]
        dx_rot = 1e-9 * (dx_nm * np.cos(-angle) + dy_nm * np.sin(-angle))
        dy_rot = 1e-9 * (-dx_nm * np.sin(-angle) + dy_nm * np.cos(-angle))
        new_center_x = center_x + dx_rot
        new_center_y = center_y + dy_rot
        wait_num = int(kwargs.get("wait", True))
        self.connect.FolMeXYPosSet(new_center_x, new_center_y, wait_num)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def check_dirs(self, dirName):
        if not exists(dirName):
            mkdir(dirName)
            print("Directory ", dirName,  " Created ")
        return dirName
    
    #################################### BIAS SPECTROSCOPY ###################################3
    def bias_spectr_par_get(self):
        bias_par = {'Bias': self.connect.BiasGet(),
                    'BiasSpectrChs': self.connect.BiasSpectrChsGet(),
                    'BiasSpectrProps': self.connect.BiasSpectrPropsGet(),
                    'BiasSpectrAdvProps': self.connect.BiasSpectrAdvPropsGet(),
                    'BiasSpectrLimits': self.connect.BiasSpectrLimitsGet(),
                    'BiasSpectrTiming': self.connect.BiasSpectrTimingGet(),
                    'BiasSpectrTTLSync': self.connect.BiasSpectrTTLSyncGet(),
                    'BiasSpectrAltZCtrl': self.connect.BiasSpectrAltZCtrlGet(),
                    'BiasSpectrMLSLockinPerSeg': self.connect.BiasSpectrMLSLockinPerSegGet(),
                    'BiasSpectrMLSMode': self.connect.BiasSpectrMLSModeGet(),
                    'BiasSpectrMLSVals': self.connect.BiasSpectrMLSValsGet(),
                    'BiasSpectrMore': pd.DataFrame({'Auto save': 'Yes/On', 'Save dialog': 'No/Off', 'Basename' : 'STS_%Y%m%d_'}, index=[0]).T,
                    'LockInModAmp1': self.connect.LockInModAmpGet(1),
                    'LockInModFreq1': self.connect.LockInModPhasFreqGet(1),
                    'LockInOnOff1': self.connect.LockInModOnOffGet(1),
                    }
        return bias_par
    
    def bias_spectr_par_save(self, bias_par, fdir, fname = ''):
        with open(fdir + '/BiasSpectr' + fname + '.par', 'wb') as handle:
            pickle.dump(bias_par, handle)
        print(f'".par" file created in {fdir}')

    def bias_spectr_par_load(self, fdir, fname):
        with open(fdir + '/' + fname, 'rb') as handle:
            bias_par = pickle.load(handle)
        for keys in bias_par:
            bias_par[keys] = bias_par[keys].replace(['No change', 'Yes/On', 'No/Off', 'False/Off', 'True/On'], [0, 1, 2, 0, 1])
        return bias_par

    def bias_spectr(self, par, data_folder, basename = '%Y%m%d_', run = True):
        self.connect.BiasSpectrOpen()
        props = (int(par['BiasSpectrProps'].loc['Save all', 0]),
                 int(par['BiasSpectrProps'].loc['Number of sweeps']),
                 par['BiasSpectrProps'].loc['Backward sweep', 0],
                 int(par['BiasSpectrProps'].loc['Number of points']),
                 float(par['BiasSpectrTiming'].loc['Z offset (m)']),
                 par['BiasSpectrMore'].loc['Auto save', 0],
                 par['BiasSpectrMore'].loc['Save dialog', 0])
        self.connect.BiasSet(*par['Bias'].values)
        self.connect.BiasSpectrChsSet(*par['BiasSpectrChs'].values.tolist())
        self.connect.BiasSpectrPropsSet(*props)
        self.connect.BiasSpectrAdvPropsSet(*par['BiasSpectrAdvProps'].values)
        self.connect.BiasSpectrLimitsSet(*par['BiasSpectrLimits'].values)
        self.connect.BiasSpectrTimingSet(*par['BiasSpectrTiming'].values)
        self.connect.BiasSpectrTTLSyncSet(*par['BiasSpectrTTLSync'].values)
        self.connect.BiasSpectrAltZCtrlSet(*par['BiasSpectrAltZCtrl'].values)
        self.connect.BiasSpectrMLSLockinPerSegSet(*par['BiasSpectrMLSLockinPerSeg'].values)
        self.connect.BiasSpectrMLSModeSet(*par['BiasSpectrMLSMode'].values)
        self.connect.BiasSpectrMLSValsSet(*par['BiasSpectrMLSVals'].values)

        self.connect.LockInModAmpSet(*par['LockInModAmp1'].values)
        self.connect.LockInModPhasFreqSet(*par['LockInModFreq1'].values)

        if run:
            self.connect.LockInModOnOffSet(*par['LockInOnOff1'].values)
            sess_path = self.connect.UtilSessionPathGet().loc['Session path', 0]
            bias_spectr_path = self.check_dirs(sess_path + '\\' + data_folder)
            self.connect.UtilSessionPathSet(bias_spectr_path, 0)
            data, parameters = self.connect.BiasSpectrStart(1, basename)
            self.connect.LockInModOnOffSet(1, 0)
            self.connect.UtilSessionPathSet(sess_path, 0)
            return data, parameters
        
    ##################################### PICK UP ATOMS ##################################    
    def atom_pickup(self, radius = 1e-9, num_aqui = 3):
        def multiple_z_get():
            z_list = []
            for i in range(num_aqui):
                time.sleep(0.2)
                z = self.connect.ZCtrlZPosGet()
                z_list.append(z.loc['Z position of the tip (m)', 0])
            return np.mean(z_list)

        def meas_dz():
            z_cen = multiple_z_get()
            surrounding_xy_list_ = [[x-radius, y], [x, y-radius], [x+radius,y], [x, y+radius]]
            surrounding_z_list = []

            for i in range(len(surrounding_xy_list_)):
                self.connect.FolMeXYPosSet(*surrounding_xy_list_[i], 1)
                surrounding_z = multiple_z_get()
                surrounding_z_list.append(surrounding_z)
            self.connect.FolMeXYPosSet(x, y, 1)
            z_sur = np.mean(surrounding_z_list)
            return z_cen - z_sur

        # get the original bias and tiplift values
        bias_ini = self.connect.BiasGet()
        tiplift_ini = self.connect.ZCtrlTipLiftGet()
        self.connect.ZCtrlTipLiftSet(0) # set tiplift to 0


        # tracking the atom for 3s
        self.connect.AtomTrackCtrlSet(0,1)
        self.connect.AtomTrackCtrlSet(1,1)
        print('Wait atom tracking for 4 seconds...')
        time.sleep(4)
        self.connect.AtomTrackCtrlSet(0,0)
        self.connect.AtomTrackCtrlSet(1,0)

        xy = self.connect.FolMeXYPosGet(1)
        x = xy.loc['X (m)']
        y = xy.loc['Y (m)']
        dz1 = meas_dz() # calculate the height of the atom before picking it up
        self.connect.BiasSet('50u')
        self.connect.ZCtrlOnOffSet(0)
        self.connect.BiasPulse(1, '150m', '650m', 1, 0)
        self.connect.ZCtrlOnOffSet(1)
        self.connect.BiasSet(bias_ini.loc['Bias (V)', 0])
        self.connect.ZCtrlTipLiftSet(tiplift_ini.loc['TipLift (m)', 0]) # set tiplift to 0
        dz2 = meas_dz()
        delta_z = dz1 - dz2
        print(f'delta z (pm): {delta_z*1e12}')
        if abs(delta_z) > 90e-12:
            print('Atom picked up.')
        else:
            print('Atom not picked up. Try again!')
        return
    
######################################## datalog measurements #############################################
