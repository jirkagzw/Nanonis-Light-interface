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
import threading
from datetime import datetime
import sys
import math

class photon_meas:
    def __init__(self, connect,connect2): #connect2 = andor
        self.connect = connect
        self.connect2=connect2
        return
    def clear_line(self):
        sys.stdout.write("\033[K") 
        sys.stdout.flush()
    
    def rotate(self,dx_nm, dy_nm, angle_deg):
        """Rotate (dx_nm, dy_nm) by a given angle in degrees. and returns outpur in (m) unit compatible with Nanonis"""
        angle = np.radians(angle_deg) #angle in radians
        factor = 1e-9  # Conversion factor for nanometers to meters
        dx_rot = factor * (dx_nm * np.cos(angle) + dy_nm * np.sin(angle))  # Rotate dx
        dy_rot = factor * (-dx_nm * np.sin(angle) + dy_nm * np.cos(angle))  # Rotate dy
        return dx_rot, dy_rot
    
    def write_initial_file(self,file_path, header, data):
        # Open the file in binary write mode
        with open(file_path, 'wb') as f:
            # Encode the ASCII header to bytes and write it
            f.write(header.encode())
            # Write the binary data
            f.write(data)

    # Function to update the End time
    def update_end_time(self,file_path):
        # Get the current end time
        end_time = datetime.now().strftime('%d.%m.%Y %H:%M:%S.%f')[:-3]
        end_time_str = f'End time="{end_time}"\n'.encode()
        
        with open(file_path, 'r+b') as f:
            # Seek to the start of the file and overwrite the placeholder
            f.seek(0)
            f.write(end_time_str)
    
    def v(self, bias_mV,protection=True):
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
        lims=(-5000,5000)
        if protection:
            if bias_mV < lims[0] or bias_mV > lims[1]:
                print(f"Value {bias_mV} not in range [{lims[0]}, {lims[1]}], coerced.")
                bias_mV = max(lims[0], min(bias_mV, lims[1]))
        self.connect.ZCtrlSetpntSet(1e-9*bias_mV)
        self.connect.BiasSet(1e-3*bias_mV)
    
    
    def i(self, current_nA, protection=True):
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
        lims=(5e-4,8)
        if protection:
            if current_nA < lims[0] or current_nA > lims[1]:
                print(f"Value {current_nA} not in range [{lims[0]}, {lims[1]}], coerced.")
                current_nA = max(lims[0], min(current_nA, lims[1]))
        self.connect.ZCtrlSetpntSet(1e-9*current_nA)
        
    
    
    def pix(self, *pixs, protection=True):
        """
        Set the pixel position for scanning, with optional range protection.
    
        Parameters
        ----------
        *pixs : int
            One or two values representing the x and y coordinates of the pixel. If only one value is
            provided, it will be used for both x and y coordinates. The x value has to be always divisible by 16, otherwise rounded by nanonis.
        protection : bool, optional
            If True (default), the pixel values will be coerced to the range [16, 1024].
            If False, no value coercion will be applied.
    
        Notes
        -----
        This function retrieves the current scan buffer, and sets the pixel position for scanning
        based on the provided coordinates. If `pix_x` and `pix_y` are equal, they are both rounded 
        to the nearest value divisible by 16, unless already divisible by 16.
        """
        for i in range(len(pixs)):
            if i == 0:
                pix_x = pixs[0]
                pix_y = pixs[0]
            elif i == 1:
                pix_y = pixs[1]
        
        # Coerce values if protection is enabled
        if protection:
            if pix_x < 16 or pix_x > 1024:
                print(f"Value {pix_x} not in range [16, 1024], coerced.")
                pix_x = max(16, min(pix_x, 1024))
            if pix_y < 16 or pix_y > 1024:
                print(f"Value {pix_y} not in range [16, 1024], coerced.")
                pix_y = max(16, min(pix_y, 1024))
                
        # Check if pix_x == pix_y, and ensure divisibility by 16
        if int(pix_x) == int(pix_y):
            if pix_x % 16 != 0:
                # Round both values to the nearest int divisible by 16
                pix_x = round(pix_x / 16) * 16
                pix_y = round(pix_y / 16) * 16
                print(f"Values were not divisible by 16, rounded to {pix_x}, {pix_y}.")
                
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
    
    def pos(self, x_nm, y_nm, width_nm=None,height_nm=None, angle=None):
        """
        Change to position of a scan frame.
    
        Parameters
        ----------
        x_nm : float
            The displacement in nanometers along the x-axis.
        
        y_nm : float
            The displacement in nanometers along the y-axis.
    
        Notes
        -----
        This function calculates the new position of the scan frame considering the current angle and updates it.
        """
        df = self.connect.ScanFrameGet()
        center_x = df.values[0][0]
        center_y = df.values[1][0]
        if angle==None:
            angle = df.values[4][0]
        if width_nm==None:
            width_nm = df.values[2][0]
        if height_nm==None:
            height_nm = df.values[3][0]
        x_rot = 1e-9 * (x_nm * np.cos(-angle) + y_nm * np.sin(-angle))
        y_rot = 1e-9 * (-x_nm * np.sin(-angle) + y_nm * np.cos(-angle))
        self.connect.ScanFrameSet(x_rot, y_rot, width_nm, height_nm , angle)
        
    def dpos(self, dx_nm, dy_nm, width_nm=None,height_nm=None, angle=None):
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
        if angle==None:
            angle = df.values[4][0]
        if width_nm==None:
            width_nm = df.values[2][0]
        if height_nm==None:
            height_nm = df.values[3][0]
        dx_rot = 1e-9 * (dx_nm * np.cos(-angle) + dy_nm * np.sin(-angle))
        dy_rot = 1e-9 * (-dx_nm * np.sin(-angle) + dy_nm * np.cos(-angle))
        new_center_x = center_x + dx_rot
        new_center_y = center_y + dy_rot
        self.connect.ScanFrameSet(new_center_x, new_center_y, width_nm, height_nm , angle)
    
    
    def mv(self, dx_nm, dy_nm, wait=True):
        """
        Change the probe XY position relative to the scan frame center.
    
        Parameters
        ----------
        dx_nm : float
            The displacement in nanometers along the x-axis.
        
        dy_nm : float
            The displacement in nanometers along the y-axis.
    
        wait=True : bool, optional
                If True (default), the function will wait until the movement is complete. If False, it will not wait.
    
        Notes
        -----
        This function calculates move to the new position onsidering the current angle.t.
        The `wait` parameter controls whether the function should block until the movement is complete.
        """
        df = self.connect.ScanFrameGet()
        center_x = df.values[0][0]
        center_y = df.values[1][0]
        angle_deg = df.values[4][0]
        angle = np.radians(angle_deg) #angle in radians
        dx_rot = 1e-9 * (dx_nm * np.cos(angle) + dy_nm * np.sin(angle))
        dy_rot = 1e-9 * (-dx_nm * np.sin(angle) + dy_nm * np.cos(angle))
        new_center_x = center_x + dx_rot
        new_center_y = center_y + dy_rot
        self.connect.FolMeXYPosSet(new_center_x, new_center_y, wait)
        
    def dmv(self, dx_nm, dy_nm, wait=True):
        """
        Change the probe XY position relative to the current probe position.
    
        Parameters
        ----------
        dx_nm : float
            The displacement in nanometers along the x-axis.
        
        dy_nm : float
            The displacement in nanometers along the y-axis.
    
        wait=True : bool, optional
                If True (default), the function will wait until the movement is complete. If False, it will not wait.
    
        Notes
        -----
        This function calculates move to the new position considering the current scan frame angle
        The `wait` parameter controls whether the function should block until the movement is complete.
        """
        df = self.connect.FolMeXYPosGet(False)
        x = df.values[0][0]
        y = df.values[1][0]
        angle_deg = self.connect.ScanFrameGet().values[4][0]
        angle = np.radians(angle_deg) #angle in radians
        dx_rot = 1e-9 * (dx_nm * np.cos(angle) + dy_nm * np.sin(angle))
        dy_rot = 1e-9 * (-dx_nm * np.sin(angle) + dy_nm * np.cos(angle))
        new_x = x + dx_rot
        new_y = y + dy_rot
        self.connect.FolMeXYPosSet(new_x, new_y, wait)
        
    def mv_spd(self,spd=None,cus_spd=True,protection=True):
        """
        Sets the linear speed in Follow me.
        Parameters
        ----------
        spd : float
            Follow me speed in nm/s
        cus_speed: bool 
            use follow me speed (True) or scan speed (False) for moving 
        """
        if spd==None:
            spd=1e9*self.connect.FolMeSpeedGet().values[0][0]
        cus_spd_int = 1
        if cus_spd in [True, 1]:
            cus_spd_int = 1
        elif cus_spd in [False, 0]:
            cus_spd_int=0
        lims=(0.001,400)
        if protection:
            if spd < lims[0] or spd > lims[1]:
                print(f"Value {spd} not in range [{lims[0]}, {lims[1]}], coerced.")
                spd = max(lims[0], min(spd, lims[1]))

        self.connect.FolMeSpeedSet( 1e-9*spd, cus_spd_int)
        
        
    def spd(self,spd, spd_bw=None,protection=True):
        """
        Sets the linear speed in scan constrol
        Parameters
        ----------
        spd_nm : float
            The displacement in nanometers along the x-axis.
        """
        lims=(0.001,400)
        if protection:
            if spd < lims[0] or spd > lims[1]:
                print(f"Value {spd} not in range [{lims[0]}, {lims[1]}], coerced.")
                spd = max(lims[0], min(spd, lims[1]))        
        if spd_bw!=None:
            if protection:
                if spd_bw < lims[0] or spd_bw > lims[1]:
                    print(f"Value {spd_bw} not in range [{lims[0]}, {lims[1]}], coerced.")
                    spd_bw = max(lims[0], min(spd_bw, lims[1]))
            ratio=spd_bw/spd
        else:
            ratio=1
            spd_bw=spd
        
        self.connect.ScanSpeedSet(1e-9*spd, 1e-9*spd, "10m", "10m", 1, ratio)
        
    def lnt(self,lnt, lnt_bw=None,protection=True):
        """
        Sets the linear speed in scan constrol
        Parameters
        ----------
        lnt : float
            Linetime in seconds.
        """
        lims=(0.005,10)
        if protection:
            if lnt < lims[0] or lnt > lims[1]:
                print(f"Value {lnt} not in range [{lims[0]}, {lims[1]}], coerced.")
                lnt = max(lims[0], min(lnt, lims[1]))        
        if lnt_bw!=None:
            if protection:
                if lnt_bw < lims[0] or lnt_bw > lims[1]:
                    print(f"Value {lnt_bw} not in range [{lims[0]}, {lims[1]}], coerced.")
                    lnt_bw = max(lims[0], min(lnt_bw, lims[1]))
            ratio=lnt/lnt_bw
        else:
            ratio=1
            lnt_bw=lnt
        
        self.connect.ScanSpeedSet("10n", "10n", lnt, lnt_bw, 2, ratio)
                
        
    def drift(self,Vx_nm=None,Vy_nm=None,Vz_nm=None,switch_on=None):
        """
        Sets the drift compensiation in the piezo configuration module. If keyword arguments are not given, no change in the argument will be made.
    
        Parameters
        ----------
        Vx_nm (nm/s) : float
            Linear speed applied to the X piezo to compensate the drift 
        Vy_nm : float
            Linear speed applied to the Y piezo to compensate the drift
        Vz_nm : float
            Linear speed applied to the Z piezo to compensate the drift
        wait=True : bool, optional
                If True (default), the function will wait until the movement is complete. If False, it will not wait.

        Return arguments:  None
        """
        if switch_on==None:
            compensation=-1
        elif switch_on==True:
            compensation=1
        elif switch_on==False:
            compensation=0
        df=self.connect.PiezoDriftCompGet()    
        if Vx_nm==None:
            Vx_nm=df.values[1][0]*1e9
        if Vy_nm==None:
            Vy_nm=df.values[2][0]*1e9
        if Vz_nm==None:
            Vz_nm=df.values[2][0]*1e9
        self.connect.PiezoDriftCompSet(compensation,Vx_nm*1e-9,Vy_nm*1e-9,Vz_nm*1e-9,10)
    
    def ddrift(self,dVx_nm=None,dVy_nm=None,dVz_nm=None,switch_on=False):
        """
        Adds values to the current drift compensiation in the piezo configuration module. If keyword arguments are not given, no change in the argument will be made.
    
        Parameters
        ----------
        dVx_nm (nm/s) : float
            Linear speed applied to the X piezo to compensate the drift
        dVy_nm : float
            Linear speed applied to the Y piezo to compensate the drift
        dVz_nm : float
            Linear speed applied to the Z piezo to compensate the drift
        wait=True : bool, optional
                If True (default), the function will wait until the movement is complete. If False, it will not wait.

        Return arguments:  None
        """
        if switch_on==None:
            compensation=-1
        elif switch_on==True:
            compensation=1
        elif switch_on==False:
            compensation=0
        df=self.connect.PiezoDriftCompGet()    
        if dVx_nm==None:
            Vx_nm=df.values[1][0]*1e9
        else:
            Vx_nm=df.values[1][0]*1e9+dVx_nm
            
        if dVy_nm==None:
            Vy_nm=df.values[2][0]*1e9
        else:
            Vx_nm=df.values[2][0]*1e9+dVy_nm
            
        if dVz_nm==None:
            Vz_nm=df.values[3][0]*1e9
        else:
            Vz_nm=df.values[3][0]*1e9+dVz_nm
            
        self.connect.PiezoDriftCompSet(compensation,Vx_nm*1e-9,Vy_nm*1e-9,Vz_nm*1e-9,10)
        
    def scan(self, direction="up", wait=True):
        """
        Perform a scan in the specified direction and wait until completion or interruption.
        
        Parameters
        ----------
        direction : str or bool or int, optional
            The direction to scan. Can be 'up', 'down', True, False, 0, or 1. 
            - 'up', True, and 0 correspond to scanning in the up direction.
            - 'down', False, and 1 correspond to scanning in the down direction.
            Default is 'up'.
        
        wait : bool, optional
            If True, the function will wait for the scan to complete. If False, it will return immediately (iscan).
            Default is True.
        
        Returns
        -------
        tuple
            A tuple containing:
            - A 3D NumPy array of forward scan data.
            - A 3D NumPy array of backward scan data.
            - A list of channel indices recorded in the scan.
        
        Raises
        ------
        ValueError
            If an invalid direction is provided.
        
        Notes
        -----
        This method initiates a scan action based on the specified direction and waits for the scan to complete,
        handling interruptions gracefully. It retrieves the scan data for both forward and backward directions,
        and stacks the data into 3D NumPy arrays for further processing.
        """
        # Initialize variables
        if direction in ["up", True, 0]:
            direction = 1
        elif direction in ["down", False, 1]:
            direction = 0
        else:
            raise ValueError("Invalid direction. Use 'up', 'down', True, False, 0, or 1.")   
        
        self.connect.ScanAction(0, direction)
        self.connect.ScanFrameGet()
        
        try:
            while wait:
                df = self.connect.ScanWaitEndOfScan(1)
                # Process the current iteration
                if float(df.loc['Timeout status', 0]) == 0:
                    wait = False
                
        except KeyboardInterrupt:
            print("Interrupt received. Finishing the current iteration...")
            self.connect.ScanAction(1, direction)
            print("Iteration finished. Exiting gracefully.")
        finally:
            channels = self.connect.ScanBufferGet().iloc[1, 0]
            data_fw, data_bw = [], []
            for channel in channels:
                data_fw.append(self.connect.ScanFrameDataGrab(channel, 0)[0].to_numpy())
                data_bw.append(self.connect.ScanFrameDataGrab(channel, 1)[0].to_numpy())
        
        return np.stack(data_fw), np.stack(data_bw), channels
    
    def scan_pause(self):
        """
        pauses the current scan and returns the data
        
        Returns
        tuple
            A tuple containing:
            - A 3D NumPy array of forward scan data.
            - A 3D NumPy array of backward scan data.
            - A list of channel indices recorded in the scan.
        
        Raises
        ------
        ValueError
            If an invalid direction is provided.
        
        Notes
        -----
        This method pauses a scan. It retrieves the scan data for both forward and backward directions,
        and stacks the data into 3D NumPy arrays for further processing.
        """
        self.connect.ScanAction(2, 0)
        self.connect.ScanFrameGet()
        channels = self.connect.ScanBufferGet().iloc[1, 0]
        data_fw, data_bw = [], []
        for channel in channels:
            data_fw.append(self.connect.ScanFrameDataGrab(channel, 0)[0].to_numpy())
            data_bw.append(self.connect.ScanFrameDataGrab(channel, 1)[0].to_numpy())
        
        return np.stack(data_fw), np.stack(data_bw), channels
    
    def scan_wait(self):
        """
        Locks the scan until the scan is not finished 
        
        Returns
        tuple
            A tuple containing:
            - A 3D NumPy array of forward scan data.
            - A 3D NumPy array of backward scan data.
            - A list of channel indices recorded in the scan.
        
        Raises
        ------
        ValueError
            If an invalid direction is provided.
        
        Notes
        -----
        This method pauses a scan. It retrieves the scan data for both forward and backward directions,
        and stacks the data into 3D NumPy arrays for further processing.
        """
        wait=True
        try:
            while wait:
                df = self.connect.ScanWaitEndOfScan(1)
                # Process the current iteration
                if float(df.loc['Timeout status', 0]) == 0:
                    wait = False
                
        except KeyboardInterrupt:
            print("Interrupt received. Finishing the current iteration...")
            self.connect.ScanAction(1, 0)
            print("Iteration finished. Exiting gracefully.")
        finally:
            channels = self.connect.ScanBufferGet().iloc[1, 0]
            data_fw, data_bw = [], []
            for channel in channels:
                data_fw.append(self.connect.ScanFrameDataGrab(channel, 0)[0].to_numpy())
                data_bw.append(self.connect.ScanFrameDataGrab(channel, 1)[0].to_numpy())
        
        return np.stack(data_fw), np.stack(data_bw), channels
    
    def scan_stop(self):
        """

        Returns
        -------
        Stops the current scan and returns the data

        tuple
            A tuple containing:
            - A 3D NumPy array of forward scan data.
            - A 3D NumPy array of backward scan data.
            - A list of channel indices recorded in the scan.
        
        Raises
        ------
        ValueError
            If an invalid direction is provided.
        
        Notes
        -----
        This method stops a scan. It retrieves the scan data for both forward and backward directions,
        and stacks the data into 3D NumPy arrays for further processing.
        """
        self.connect.ScanAction(1, 0)
        self.connect.ScanFrameGet()
        channels = self.connect.ScanBufferGet().iloc[1, 0]
        data_fw, data_bw = [], []
        for channel in channels:
            data_fw.append(self.connect.ScanFrameDataGrab(channel, 0)[0].to_numpy())
            data_bw.append(self.connect.ScanFrameDataGrab(channel, 1)[0].to_numpy())
        
        return np.stack(data_fw), np.stack(data_bw), channels
    
    def scan_volume(self, z_range, n_steps, return_to_start=False, direction="up"):
        """
        Perform a series of CH scans (n_steps) in z_range  in the specified direction and wait until completion or interruption.
        
        Parameters
        ----------
        z_range : float in nm
        n_steps : int -number of scans in the volume
        
        direction : str or bool or int, optional
            The direction to scan. Can be 'up', 'down', True, False, 0, or 1. 
            - 'up', True, and 0 correspond to scanning in the up direction.
            - 'down', False, and 1 correspond to scanning in the down direction.
            Default is 'up'.
        return_to_start : bool, optional False - return to the original heigth after finishing 
        
        Returns
        -------
        tuple
            A tuple containing:
            - A 4D NumPy array of forward scan data.
            - A 4D NumPy array of backward scan data.
            - A list of channel indices recorded in the scan.
        
        Raises
        ------
        ValueError
            If an invalid direction is provided.
        
        Notes
        -----
        This method initiates a scan action based on the specified direction and waits for the scan to complete,
        handling interruptions gracefully. It retrieves the scan data for both forward and backward directions,
        and stacks the data into 3D NumPy arrays for further processing.
        """
        # Initialize variables
        if direction in ["up", True, 0]:
            direction = 1
        elif direction in ["down", False, 1]:
            direction = 0
        else:
            raise ValueError("Invalid direction. Use 'up', 'down', True, False, 0, or 1.")   
        
        feed=self.connect.tcp.bistate_cvt(self.connect.ZCtrlOnOffGet().values[0][0])
        if feed==True:
            self.connect.ZCtrlOnOffSet(0)
            time.sleep(0.05)
            
        
            
        try:
            for i in range(n_steps):
                if i!=0:
                    self.dz(z_range/(n_steps-1))
                time.sleep(0.05)
                self.connect.ScanAction(0, direction)
                wait=True
                while wait:
                    df = self.connect.ScanWaitEndOfScan(1)
                    # Process the current iteration
                    if float(df.loc['Timeout status', 0]) == 0:
                        wait = False
                
        except KeyboardInterrupt:
            print("Interrupt received. Finishing the current iteration...")
            self.connect.ScanAction(1, direction)
            print("Iteration finished. Exiting gracefully.")
        finally:
            if return_to_start==True:
                self.dz(-z_range)
            channels = self.connect.ScanBufferGet().iloc[1, 0]
            data_fw, data_bw = [], []
            for channel in channels:
                data_fw.append(self.connect.ScanFrameDataGrab(channel, 0)[0].to_numpy())
                data_bw.append(self.connect.ScanFrameDataGrab(channel, 1)[0].to_numpy())
        
        return np.stack(data_fw), np.stack(data_bw), channels
    
    def dz(self, dz_nm, switch_off_feed=True):
        """
        Change the z position by dz and optionally switch off the feedback before.
    
        Parameters
        ----------
        dz_nm : float
            The displacement in nanometers along the x-axis - positive = retract, negative = approach!!!.
        
        switch_off_feed : bool
            The displacement in nanometers along the y-axis.
    
        """
        if switch_off_feed==True:
            self.connect.ZCtrlOnOffSet(0)
            time.sleep(0.05)
            
        z=self.connect.ZCtrlZPosGet().loc['Z position of the tip (m)', 0]
        z_new=z+dz_nm*1E-9
        self.connect.ZCtrlZPosSet(z_new)
        
    def df0(self):
        df_old=self.connect.PLLCenterFreqGet().values[0][0]
        self.connect.PLLFreqShiftAutoCenter()
        df_new=self.connect.PLLCenterFreqGet().values[0][0]
        return((df_new,df_new-df_old))
        
        
    def spectrum_simple(self,acqtime=10, acqnum=1, name="LS-man"):
        name="AA"+name
        self.connect2.acqtime_set(acqtime)
        
        for i in range(0, int(acqnum)):
            if i==0:
                data=self.connect2.acquisition_set()
            else:
                if i==1:
                    data.columns = [data.columns[0], f"Counts {i}"]
            
                data_new=self.connect2.acquisition_set()
                data[f"Counts {i+1}"]=data_new["Counts"]
            
        return(data)

    def acquire_data_from_connect(self, signal_values, acquisition_complete, stop_time,signal_range):
        start_time = time.time()
        while not acquisition_complete.is_set() and (time.time() - start_time) < stop_time:
            # Collect signal values from the connect device
            signal_values.append(self.connect.SignalsValsGet(np.arange(0, signal_range, 1), 1))
            # Sleep briefly to avoid busy-waiting
           # time.sleep(0.1)
        # Signal that acquisition is complete
        acquisition_complete.set()
        
    def acquire_data_from_connect_new(self, signal_values, acquisition_complete, stop_time,signal_range):
        start_time = time.time()
        while not acquisition_complete.is_set() and (time.time() - start_time) < stop_time:
            # Collect signal values from the connect device
            val=(self.connect.SignalsValsGet(np.arange(0, signal_range, 1), 1)).iloc[:,1].to_numpy() # extract 2nd column and convert to np array
            signal_values.append(val)
            # Sleep briefly to avoid busy-waiting
            #time.sleep(0.1)
        # Signal that acquisition is complete
        acquisition_complete.set()



    def acquire_data_from_connect2(self, data_storage, acquisition_complete):
        try:
            # Start acquisition on connect2 (blocking call)
            data_storage['data'] = self.connect2.acquisition_set(prt=False)
        finally:
            # Signal that acquisition is complete
            acquisition_complete.set()
            
    
    def save_params_connect(self,list_of_dfs, signal_names_df, signal_names=None):
        if signal_names is None:
            signal_names = [
                "Bias (V)", "X (m)", "Y (m)", "Z (m)", "Current (A)", 
                "LI Demod 1 Y (A)", "LI Demod 2 Y (A)", "Counter 1 (Hz)"
            ]

        # Fetch valid signal names from the connect object
        if signal_names_df is None or 'Signal names' not in signal_names_df.columns:
            raise ValueError("The DataFrame returned by SignalsNamesGet does not contain 'Signal names' column.")
        
        valid_signal_names = set(signal_names_df['Signal names'])
        signal_col = signal_names_df['Signal names']

        # Create a mask for valid signal names (this mask is based on the set of valid signal names)
        valid_signal_mask = lambda signal_col: signal_col.isin(valid_signal_names)

        # Initialize a dictionary to store lists of values for each signal
        signal_values = {signal: [] for signal in signal_names if signal in valid_signal_names}

        for df_list in list_of_dfs:
            for df in df_list:
                if df.shape[1] < 2:
                    continue

                # Extract the relevant columns

                values_col = df.iloc[:, 1]

                # Apply the precomputed mask
                mask = valid_signal_mask(signal_col)
                filtered_signals = signal_names_df['Signal names'][mask]
                filtered_values = values_col[mask]

                # Append values to the appropriate signal lists
                for signal, value in zip(filtered_signals, filtered_values):
                    if signal in signal_values:
                        signal_values[signal].append(value)

        # Calculate the mean for each signal
        averages = {signal: np.mean(values) if values else None
                    for signal, values in signal_values.items()}
        
      #  print(averages)
        return averages
    """
    def save_params_connect_new(self,list_of_dfs, signal_names_df, signal_names=None):
        if signal_names is None:
            signal_names = [
                "Bias (V)", "X (m)", "Y (m)", "Z (m)", "Current (A)", 
                "LI Demod 1 Y (A)", "LI Demod 2 Y (A)", "Counter 1 (Hz)"
            ]

        # Fetch valid signal names from the connect object
        if signal_names_df is None or 'Signal names' not in signal_names_df.columns:
            raise ValueError("The DataFrame returned by SignalsNamesGet does not contain 'Signal names' column.")
        
        signal_names_array = signal_names_df['Signal names'].to_numpy()
        
        valid_signal_names = set(signal_names_array)

        # Create a mask for valid signal names (this mask is based on the set of valid signal names)
        valid_signal_mask = lambda arr: np.isin(arr, valid_signal_names)

        # Initialize a dictionary to store lists of values for each signal
        signal_values = {signal: [] for signal in signal_names if signal in valid_signal_names}

        for df_list in list_of_dfs:
            for df in df_list:
                if len(df) != len(signal_names_array):
                    raise ValueError("Length of df does not match length of signal_names_array.")
               # Apply the precomputed mask
                mask = valid_signal_mask(signal_names_array)
                filtered_signals = signal_names_array[mask]
                filtered_values = df[mask]

                # Append values to the appropriate signal lists
                for signal, value in zip(filtered_signals, filtered_values):
                    if signal in signal_values:
                        signal_values[signal].append(value)

        # Calculate the mean for each signal
        averages = {signal: np.mean(values) if values else None
                    for signal, values in signal_values.items()}
        
     #   print(averages)
        return averages
    """

    def save_params_connect_new(self,list_of_dfs, signal_names=None):
        signal_names_df=self.connect.SignalsNamesGet()
        # Handle default signal names
        if signal_names is None:
            signal_names = [
                "Bias (V)", "X (m)", "Y (m)", "Z (m)", "Current (A)", 
                "LI Demod 1 Y (A)", "LI Demod 2 Y (A)", "Counter 1 (Hz)"
            ]
            
            
        # Fetch valid signal names from the connect object
        if signal_names_df is None or 'Signal names' not in signal_names_df.columns:
            raise ValueError("The DataFrame returned by SignalsNamesGet does not contain 'Signal names' column.")
        
        signal_names_array = signal_names_df['Signal names'].to_numpy()
        valid_signal_names = set(signal_names_array)
        
       
    
        signal_values = {signal: [] for signal in signal_names if signal in valid_signal_names}
    
        for df_list in list_of_dfs:
            for df in df_list:
                if len(df) != len(signal_names_array):
                    raise ValueError("Length of df does not match length of signal_names_array.")
                
                mask = np.isin(signal_names_array, valid_signal_names)
                
                print(f"Mask: {mask}")
    
                filtered_signals = signal_names_array[mask]
                filtered_values = df[mask]
    
             #   print(f"Filtered signals: {filtered_signals}")
              #  print(f"Filtered values: {filtered_values}")
    
                for signal, value in zip(filtered_signals, filtered_values):
                    if signal in signal_values:
                        signal_values[signal].append(value)
    
        averages = {signal: np.mean(values) if values else None
                    for signal, values in signal_values.items()}
        
        print(f"Averages: {averages}")
    
        return averages
    
    
    
    


    def spectrum(self, acqtime=10, acqnum=1, name="LS-man", user="Jirka",signal_names=None):
        # Initialize variables
        self.connect2.acqtime_set(acqtime)
        folder=self.connect.UtilSessionPathGet().loc['Session path', 0]
        settings=self.connect2.settings_get()
        signal_names_df=self.connect.SignalsNamesGet()

        sigvals = []
        data_dict = {}
        try:
            for i in range(int(acqnum)):
                # Create events to signal when acquisitions are complete
                acquisition_complete_connect = threading.Event()
                acquisition_complete_connect2 = threading.Event()
                stop_signal = threading.Event()  # Create a stop signal
                
                # Storage for data from connect2
                data_storage = {}
                
                # Start the thread to acquire data from connect2
                acquire_thread2 = threading.Thread(target=self.acquire_data_from_connect2, args=(data_storage, acquisition_complete_connect2))
                acquire_thread2.start()
                
                # Start a thread to acquire data from connect with a time limit
                signal_values = []
                acquire_thread_connect = threading.Thread(target=self.acquire_data_from_connect, args=(signal_values, acquisition_complete_connect, acqtime,len(signal_names_df))) 
                acquire_thread_connect.start()
                
                # Wait for the acquisition to complete on connect2
                acquisition_complete_connect2.wait()  # This will block until acquisition_complete_connect2 is set
                
                # Ensure the connect thread has finished
                acquire_thread_connect.join()
                acquire_thread2.join()
                
                # Process the acquired signal values
                sigvals.append(signal_values)
                
                # Update the DataFrame with new data from connect2
                data_new = data_storage['data']
                if i == 0:
                    data_dict['Wavelength (nm)'] = data_new['Wavelength (nm)']
                    data_dict[f"Counts {i+1}"] = data_new['Counts']
                else:
                    data_dict[f"Counts {i+1}"] = data_new['Counts']
                    
              #   Optionally, set the stop signal here if you want to stop after each iteration
              #  stop_signal.set() # Uncomment if you want to stop after each response from connect2
        
        except KeyboardInterrupt:
            print("Acquisition interrupted.")
        finally:
            # Ensure that `data` is created even if interrupted
            data = pd.DataFrame(data_dict)
          #  print("Sigvals before processing:", sigvals)
          #  print(signal_names_df)
          
            sigvals=self.save_params_connect(sigvals,signal_names_df,signal_names=signal_names)
            
            formatted_date_str = datetime.now().strftime('%d.%m.%Y %H:%M:%S')

            # Data to prepend
            prepend_data = {
                'Column1': ['Experiment', 'Date', 'User'],
                'Column2': ['LS', formatted_date_str, user]
            }
            
            # Create DataFrames
            prepend_df = pd.DataFrame(prepend_data)
            sigvals_df = pd.DataFrame(list(sigvals.items()), columns=['Column1', 'Column2'])
       #     start_time = time.perf_counter()  # Use time.perf_counter() for high-resolution timing
            # Define filenames and data
            filename = self.connect.get_next_filename(name,extension='.dat',folder=folder)
            print(filename)
            combined_df = pd.concat([prepend_df, sigvals_df], ignore_index=True)
            settings_df = settings
            data_df = data
            
            # Format the DataFrame in one go
            combined_df = combined_df.applymap(lambda x: '{:.7E}'.format(x) if isinstance(x, float) else x)
            
            # Write all data to a file in one go
            with open(filename, 'w') as f:
                # Write the formatted DataFrame
                combined_df.to_csv(f, sep='\t', header=False, index=False, lineterminator="\n")
                settings_df.to_csv(f, sep='\t', header=False, index=False, lineterminator="\n") # Write additional settings
                
                # Write section header and additional data
                f.write("\n[DATA]\n")
                data_df.to_csv(f, sep='\t', header=True, index=False, lineterminator="\n")  
              #  end_time = time.perf_counter()
              #  elapsed_time = end_time - start_time
               # print(f"Time taken for pix command: {elapsed_time:.4f} seconds")
                     
       # return data, sigvals, settings
    

    def spectrum_old(self, acqtime=10, acqnum=1, name="LS-man"):
        # Initialize variables
        signal_names_df=self.connect.SignalsNamesGet()
        df_deep_copy = signal_names_df.copy(deep=True)
        length=len(df_deep_copy)
        sigvals = []
        data_dict = {}
        try:
            for i in range(int(acqnum)):
                # Create events to signal when acquisitions are complete
                acquisition_complete_connect = threading.Event()
                acquisition_complete_connect2 = threading.Event()
                stop_signal = threading.Event()  # Create a stop signal
                
                # Storage for data from connect2
                data_storage = {}
                
                # Start the thread to acquire data from connect2
                acquire_thread2 = threading.Thread(target=self.acquire_data_from_connect2, args=(data_storage, acquisition_complete_connect2))
                acquire_thread2.start()
                
                # Start a thread to acquire data from connect with a time limit
                signal_values = []
                acquire_thread_connect = threading.Thread(target=self.acquire_data_from_connect_new, args=(signal_values, acquisition_complete_connect, acqtime,length)) 
                acquire_thread_connect.start()
                
                # Wait for the acquisition to complete on connect2
                acquisition_complete_connect2.wait()  # This will block until acquisition_complete_connect2 is set
                
                # Ensure the connect thread has finished
                acquire_thread_connect.join()
                acquire_thread2.join()
                
                # Process the acquired signal values
                sigvals.append(signal_values)
                
                # Update the DataFrame with new data from connect2
                data_new = data_storage['data']
                if i == 0:
                    data_dict['Wavelength (nm)'] = data_new['Wavelength (nm)']
                    data_dict[f"Counts {i+1}"] = data_new['Counts']
                else:
                    data_dict[f"Counts {i+1}"] = data_new['Counts']
                    
              #   Optionally, set the stop signal here if you want to stop after each iteration
              #  stop_signal.set() # Uncomment if you want to stop after each response from connect2
        
        except KeyboardInterrupt:
            print("Acquisition interrupted.")
        finally:
            # Ensure that `data` is created even if interrupted
            data = pd.DataFrame(data_dict)
          #  print("Sigvals before processing:", sigvals)
            sigvals2=self.save_params_connect_new(sigvals)
                        
        return data, sigvals2  
        
    def photon_map(self, acqtime=10, acqnum=1, pix=(10, 10), dim=None, name="LS-man", user="Jirka", signal_names=None,savedat=False,direction="up",backward=False):
        self.connect2.acqtime_set(acqtime)
        # Initialize variables
        if direction in ["up", True, 0]:
            direction = "up"
        elif direction in ["down", False, 1]:
            direction = "down"
        else:
            raise ValueError("Invalid direction. Use 'up', 'down', True, False, 0, or 1.")   
        start_time_scan = time.perf_counter()
        folder = self.connect.UtilSessionPathGet().loc['Session path', 0]
        signal_names_df = self.connect.SignalsNamesGet()
        SF = self.connect.ScanFrameGet()  # Retrieve scan frame
        settings=self.connect2.settings_get()
        if dim is None:
            dim=(1e9*SF.values[2][0],1e9*SF.values[3][0])
        cx, cy, angle = SF.values[0][0], SF.values[1][0], SF.values[4][0]  # Extract center and angle
        wait_num = True  # Wait flag
    
        channels_of_interest = ['Z (m)', 'Current (A)', 'LI Demod 1 Y (A)', 'LI Demod 2 Y (A)', 'Counter 1 (Hz)']
        data_ar=[]
        sigval_ar=[]
        counter,count_write=0,0
        
        # .3ds header creation
        filename_3ds = self.connect.get_next_filename("G"+name, extension='.3ds', folder=folder)
    
        bias_voltage = self.connect.BiasGet().iloc[0, 0]  # Bias (V) as float
        grid_settings = np.array([cx,cy,1e-9*dim[0],1e-9*dim[1],angle])  # Grid settings as np.array
        sweep_signal = "Wavelength (nm)"  # Sweep signal as string
        count_write=0
        
        # Prepare the header
        start_time = datetime.now().strftime('%d.%m.%Y %H:%M:%S.%f')[:-3]
        header = f'''End time="{start_time}"
Start time="{start_time}"
Delay before measuring (s)=0
Comment=
Bias (V)={bias_voltage:.6E}
Experiment=Experiment
Date="{start_time.split()[0]}"
User=
Grid dim="{pix[0]} x {pix[1]}"
Grid settings={";".join([f'{val:.6E}' for val in grid_settings])}

'''
        if backward==False:
            col_lst=list(range(pix[0]))
            bw_fact=1
        else:
            col_lst=list(range(pix[0])) + list(range(pix[0]-1, -1, -1))
            filename_3ds_bw = self.connect.get_next_filename("G"+name+'_bw', extension='.3ds', folder=folder)
            f_bw = open(filename_3ds_bw, 'wb')
            f_bw.write(header.encode())
            bw_fact=2
        f = open(filename_3ds, 'wb')
        f.write(header.encode())
        try:
            for row in range(pix[1]):
                for index, column in enumerate(col_lst):
                    counter+=1
                    if direction == "up":
                        dx_nm = (-(dim[0] / 2) + (dim[0] / pix[0]) * 0.5) + column * (dim[0] / pix[0])
                        dy_nm = (-(dim[1] / 2) + (dim[1] / pix[1]) * 0.5) + row * (dim[1] / pix[1])
                    elif direction == "down":
                        dx_nm = (-(dim[0] / 2) + (dim[0] / pix[0]) * 0.5) + column * (dim[0] / pix[0])
                        dy_nm = (-(dim[1] / 2) + (dim[1] / pix[1]) * 0.5) + (pix[1] - 1 - row) * (dim[1] / pix[1])
                        
                    dx_rot, dy_rot = self.rotate(dx_nm, dy_nm, angle)
                    self.connect.FolMeXYPosSet(cx + dx_rot, cy + dy_rot, wait_num)  # Set new position
                    
                    sigvals = []
                    data_dict = {}
                    
                    for i in range(int(acqnum)):
                        swrite=time.perf_counter() # start meas time
                        # Create events to signal when acquisitions are complete
                        acquisition_complete_connect = threading.Event()
                        acquisition_complete_connect2 = threading.Event()
                        stop_signal = threading.Event()  # Create a stop signal
                        
                        # Storage for data from connect2
                        data_storage = {}
                        
                        # Start the thread to acquire data from connect2
                        acquire_thread2 = threading.Thread(target=self.acquire_data_from_connect2, args=(data_storage, acquisition_complete_connect2))
                        acquire_thread2.start()
                        
                        # Start a thread to acquire data from connect with a time limit
                        signal_values = []
                        acquire_thread_connect = threading.Thread(target=self.acquire_data_from_connect, args=(signal_values, acquisition_complete_connect, acqtime, len(signal_names_df)))
                        acquire_thread_connect.start()
                        
                        # Wait for the acquisition to complete on connect2
                        acquisition_complete_connect2.wait()  # This will block until acquisition_complete_connect2 is set
                        
                        # Ensure the connect thread has finished
                        acquire_thread_connect.join()
                        acquire_thread2.join()
                        
                        count_write+=time.perf_counter()-swrite # stop meas time
                        
                        # Process the acquired signal values
                        sigvals.append(signal_values)
                        
                        # Update the DataFrame with new data from connect2
                        data_new = data_storage['data']
                        if i == 0:
                            data_dict['Wavelength (nm)'] = data_new['Wavelength (nm)']
                            data_dict[f"Counts {i+1}"] = data_new['Counts']
                        else:
                            data_dict[f"Counts {i+1}"] = data_new['Counts']
                        
                        # Optionally, set the stop signal here if you want to stop after each iteration
                        # stop_signal.set() # Uncomment if you want to stop after each response from connect2
                    
                    # SAVE FILES
                    data = pd.DataFrame(data_dict)
                    # print("Sigvals before processing:", sigvals)
                    # print(signal_names_df)
                    
                    sigvals = self.save_params_connect(sigvals, signal_names_df, signal_names=signal_names)
                    
                    formatted_date_str = datetime.now().strftime('%d.%m.%Y %H:%M:%S')
                    
                    # Data to prepend
                    prepend_data = {
                        'Column1': ['Experiment', 'Date', 'User'],
                        'Column2': ['LS', formatted_date_str, user]
                    }
                    
                    # Create DataFrames
                    prepend_df = pd.DataFrame(prepend_data)
                    sigvals_df = pd.DataFrame(list(sigvals.items()), columns=['Column1', 'Column2'])
                    combined_df = pd.concat([prepend_df, sigvals_df], ignore_index=True)
                    settings_df = settings
                    
                    if savedat==True:
                        # Define filenames and data
                        filename = self.connect.get_next_filename("AA"+name, extension='.dat', folder=folder)
                        print(filename)
                        

                        
                        # Format the DataFrame in one go
                        combined_df = combined_df.applymap(lambda x: '{:.7E}'.format(x) if isinstance(x, float) else x)
                        
                        # Write all data to a file in one go
                        with open(filename, 'w') as f_text:
                            # Write the formatted DataFrame
                            combined_df.to_csv(f, sep='\t', header=False, index=False, lineterminator="\n")
                            settings_df.to_csv(f, sep='\t', header=False, index=False, lineterminator="\n")  # Write additional settings
                            
                            # Write section header and additional data
                            f_text.write("\n[DATA]\n")
                            data.to_csv(f, sep='\t', header=True, index=False, lineterminator="\n")
                    else:
                        pass
                    
                    ### saves data 
                    selected_columns = data.iloc[:, 1:].values
                    
                    # Step 2: Compute the row-wise average
                    row_average=np.mean(selected_columns, axis=1)
                    data_ar.append(row_average)
                    
                    # Filter the dataframe based on the channels of interest
                    filtered_sigvals_df = sigvals_df[sigvals_df['Column1'].isin(channels_of_interest)]
                    sigval_ar.append(filtered_sigvals_df['Column2'].tolist())
                    filtered_sigvals_list=filtered_sigvals_df['Column2'].tolist()
                    num_signals = len(filtered_sigvals_df)
                    num_pixels=selected_columns.shape[0]
                    
                    # write to .3ds file
# start problematic section
                    if row==0 and index==0:
                        fixed_parameters = ["Sweep Start", "Sweep End"]+filtered_sigvals_df['Column1'].tolist()
                            #print(fixed_parameters)
                        header2=f'''Sweep Signal="{sweep_signal}"
Fixed parameters="{';'.join(fixed_parameters)}"
Experiment parameters=
# Parameters (4 byte)={len(fixed_parameters)}
Experiment size (bytes)=4096
Points={num_pixels}
Channels=Counts
'''
                        f.write(header2.encode())
                        andor_chan_names= data.iloc[:, 0].values.tolist()
                        chnames = [f"wl {i}=" + str(item) for i, item in enumerate(andor_chan_names)]
                        f.write(('\n'.join(chnames)+"\n").encode())
                        f.write((':HEADER_END:\n').encode())
                    if backward==True and row==0 and index==pix[0]:
                        f_bw.write(header2.encode())
                        f_bw.write(('\n'.join(chnames)+"\n").encode())
                        f_bw.write((':HEADER_END:\n').encode())
                    res_list=[float(andor_chan_names[0]),float(andor_chan_names[-1])]+filtered_sigvals_list
                 #   print(res_list)
                    swrite=time.perf_counter()
                    # write only forward scan
                    if index<pix[0]: 
                        np.array(res_list).astype(">f4").tofile(f)
                        row_average.astype(">f4").tofile(f)
                    else:
                        np.array(res_list).astype(">f4").tofile(f_bw)
                        row_average.astype(">f4").tofile(f_bw)
                    
                    #timing and print in terminal remaining time
                    elapsed = time.perf_counter() - start_time_scan
                    remaining = (elapsed / counter) * (math.prod(pix)*bw_fact - counter)
                    sys.stdout.write(f"\rExecuted {counter}/{math.prod(pix)*bw_fact} | Remaining: {int(remaining // 60):02d}:{int(remaining % 60):02d} | Overhead {count_write-acqtime}") #(column+1+pix[0]*row)*
                    sys.stdout.flush()
                   # print(f"Executed {counter}/{math.prod(pix)}", end="\r")  # Updates in-place
                 
            
        except KeyboardInterrupt:
            print("Acquisition interrupted.")
            for i in range(len(sigval_ar),int(len(col_lst)*pix[1])):
                sigval_ar.append([np.NaN] * num_signals)
                data_ar.append(np.full(num_pixels, np.NaN))
        finally:
            f.close()
            if backward==True:
                f_bw.close()
            end_time_scan = time.perf_counter()
            elapsed_time_scan="{:.1f}".format(end_time_scan-start_time_scan)
            filename_sxm = self.connect.get_next_filename("M"+name,extension='.sxm',folder=folder)
            
            nanonis_const= dict(zip(combined_df.T.iloc[0], combined_df.T.iloc[1].astype(str)))
            settings_dict=(dict(zip(settings.T.iloc[0], settings.T.iloc[1].astype(str))))
            scan_par = {
                "REC_DATE": datetime.now().strftime('%d.%m.%Y'),
                "REC_TIME":  datetime.now().strftime('%H:%M:%S'),
                "ACQ_TIME": str(elapsed_time_scan),
                "SCAN_PIXELS": f"{pix[0]}\t{pix[1]}",
                "SCAN_FILE": filename_sxm,
                "SCAN_RANGE": f"{1e-9 * dim[0]:.6E}\t{1e-9 * dim[1]:.6E}",
                "SCAN_OFFSET": f"{cx:.6E}\t{cy:.6E}",
                "SCAN_ANGLE": str(angle),
                "SCAN_DIR": direction,
                "BIAS": str(sigvals_df[sigvals_df['Column1'].isin(["Bias (V)"])]['Column2'].iloc[0])
                }   
            andor_chan_names= data.iloc[:, 0].values.tolist()
            nanonis_chan_names=filtered_sigvals_df['Column1'].tolist()
            
            nanononis_data_to_sxm=np.array(sigval_ar).T
            andor_data_to_sxm=np.array(data_ar).T
            combined_data = np.vstack((nanononis_data_to_sxm, andor_data_to_sxm))
            
                        # Define constants
            scan_dir = 'both'
            default_value1 = '1.000E+0'
            default_value2 = '0.000E+0'
            
            # Initialize lists to store the final output
            final_list = []
            
            # Process nanonis_chan_names
            for i, name in enumerate(nanonis_chan_names, start=1):
                base_name, unit = name.split(' (')
                unit = unit.strip(')')
                final_list.append([i, f'{base_name}_avg.', unit, scan_dir, default_value1, default_value2])
            
            # Process andor_chan_names starting from index 128
            for i, name in enumerate(andor_chan_names, start=128):
                final_list.append([i, name, 'nm', scan_dir, default_value1, default_value2])

            data_sxm=self.connect.writesxm(backward,filename_sxm, settings_dict, scan_par, final_list, combined_data)

            # Ensure that `data` is created even if interrupted
     
              #  end_time = time.perf_counter()
              #  elapsed_time = end_time - start_time
               # print(f"Time taken for pix command: {elapsed_time:.4f} seconds")
                          
            return data_sxm,combined_data
        
    
    
    def nanonis_map(self, acqtime=10, acqnum=1, pix=(10, 10), dim=None, name="LS-man", user="Jirka", signal_names=None,savedat=False,direction="up"):
        # Initialize variables
        name="AA"+name
        if direction in ["up", True, 0]:
            direction = "up"
        elif direction in ["down", False, 1]:
            direction = "down"
        else:
            raise ValueError("Invalid direction. Use 'up', 'down', True, False, 0, or 1.")
            
        start_time_scan = time.perf_counter()
        folder = self.connect.UtilSessionPathGet().loc['Session path', 0]
        signal_names_df = self.connect.SignalsNamesGet()
        SF = self.connect.ScanFrameGet()  # Retrieve scan frame
        if dim is None:
            dim=(1e9*SF.values[2][0],1e9*SF.values[3][0])
        cx, cy, angle = SF.values[0][0], SF.values[1][0], SF.values[4][0]  # Extract center and angle
        wait_num = True  # Wait flag
    
        channels_of_interest = ['Z (m)', 'Current (A)', 'LI Demod 1 Y (A)', 'LI Demod 2 Y (A)', 'Counter 1 (Hz)']
        # Initialize sigval_ar with NaN values
        sigval_ar = []    
        
        
        # .3ds header creation
        filename_3ds = self.connect.get_next_filename(name, extension='.3ds', folder=folder)
    
        bias_voltage = self.connect.BiasGet().iloc[0, 0]  # Bias (V) as float
        grid_settings = np.array([cx,cy,1e-9*dim[0],1e-9*dim[1],angle])  # Grid settings as np.array
        sweep_signal = "Wavelength (nm)"  # Sweep signal as string
        count_write=0
        
        # Prepare the header
        start_time = datetime.now().strftime('%d.%m.%Y %H:%M:%S.%f')[:-3]
        header = f'''End time="{start_time}"
Start time="{start_time}"
Delay before measuring (s)=0
Comment=
Bias (V)={bias_voltage:.6E}
Experiment=Experiment
Date="{start_time.split()[0]}"
User=
Grid dim="{pix[0]} x {pix[1]}"
Grid settings={";".join([f'{val:.6E}' for val in grid_settings])}
'''
        
    #    with open(filename_3ds, 'wb') as f:
            # Write the ASCII header encoded as bytes
        f = open(filename_3ds, 'wb')
        f.write(header.encode())
        try:
            for row in range(pix[1]):
                for column in range(pix[0]):
                    if direction == "up":
                        dx_nm = (-(dim[0] / 2) + (dim[0] / pix[0]) * 0.5) + column * (dim[0] / pix[0])
                        dy_nm = (-(dim[1] / 2) + (dim[1] / pix[1]) * 0.5) + row * (dim[1] / pix[1])
                    elif direction == "down":
                        dx_nm = (-(dim[0] / 2) + (dim[0] / pix[0]) * 0.5) + column * (dim[0] / pix[0])
                        dy_nm = (-(dim[1] / 2) + (dim[1] / pix[1]) * 0.5) + (pix[1] - 1 - row) * (dim[1] / pix[1])
                        
                    dx_rot, dy_rot = self.rotate(dx_nm, dy_nm, angle)
                    self.connect.FolMeXYPosSet(cx + dx_rot, cy + dy_rot, wait_num)  # Set new position
    
                    sigvals = []
    
                    for i in range(int(acqnum)):
                        signal_values = []
                        # Sequential data acquisition
                        start_time = time.time()
                        while (time.time() - start_time) < acqtime:
                            # Collect signal values from the connect device
                          #  signal_values.append(self.connect.SignalsValsGet(np.arange(0, len(signal_names_df), 1), 1))   
                            signal_values.append(self.connect.SignalsValsGet(np.arange(0, 31, 1), 1))   
                        # Process the acquired signal values
                        sigvals.append(signal_values)
    
                    # SAVE FILES
                    swrite=time.perf_counter()
    
                    sigvals = self.save_params_connect(sigvals, signal_names_df, signal_names=signal_names)
                    formatted_date_str = datetime.now().strftime('%d.%m.%Y %H:%M:%S')
    
                    # Prepend metadata
                    prepend_data = {
                        'Column1': ['Experiment', 'Date', 'User'],
                        'Column2': ['LS', formatted_date_str, user]
                    }
                    prepend_df = pd.DataFrame(prepend_data)
                    sigvals_df = pd.DataFrame(list(sigvals.items()), columns=['Column1', 'Column2'])
                    if savedat==True:
                        
                        # Define filenames and data
                        filename = self.connect.get_next_filename(name, extension='.dat', folder=folder)
                        print(filename)
        
                        # Write all data to a file in one go
                        with open(filename, 'w') as f:
                            combined_df = pd.concat([prepend_df, sigvals_df], ignore_index=True)
                            combined_df.to_csv(f, sep='\t', header=False, index=False, lineterminator="\n")
                    else:
                        pass
        
                    # Process and store data
                    filtered_sigvals_df = sigvals_df[sigvals_df['Column1'].isin(channels_of_interest)]
                    bias_df=sigvals_df[sigvals_df['Column1'].isin(["Bias (V)"])]
                    num_signals = len(filtered_sigvals_df)
                    filtered_sigvals_list=filtered_sigvals_df['Column2'].tolist()
                    sigval_ar.append(filtered_sigvals_list)
                    
                    # write to .3ds file
                   # print(row, column)
# start problematic section
                    if row==0 and column==0:
                        #with open(filename_3ds, 'ab') as f:
                            # Write the ASCII header encoded as bytes
                        fixed_parameters = ["Sweep Start", "Sweep End"]+filtered_sigvals_df['Column1'].tolist()
                            #print(fixed_parameters)
                        header2=f'''Sweep Signal="{sweep_signal}"
Fixed parameters="{';'.join(fixed_parameters)}"
Experiment parameters=
# Parameters (4 byte)={len(fixed_parameters)}
Experiment size (bytes)=4096
Points=1024
Channels=Integer
'''
                        f.write(header2.encode())
                        chnames=[str(i) for i in range(1, 1025)]
                        f.write(('\n'.join(chnames)+"\n").encode())
                        f.write((':HEADER_END:\n').encode())
                    res_list=[float(1),float(1024)]+filtered_sigvals_list

                    #with open(filename_3ds, 'ab') as f:
                    np.array(res_list).astype(">f4").tofile(f)
                    np.arange(len(chnames)).astype(">f4").tofile(f)
                 #   print(row, column,"after")
                    count_write+=time.perf_counter()-swrite
                    sys.stdout.write(f"\rTotal write time {count_write}")
                    sys.stdout.flush()
#end problematic setion
                        

    
        except KeyboardInterrupt:
            print("Acquisition interrupted.")
            for i in range(len(sigval_ar),int(pix[0]*pix[1])):
                sigval_ar.append([np.NaN] * num_signals)
        finally:
            f.close()
            end_time_scan = time.perf_counter()
            elapsed_time_scan = "{:.1f}".format(end_time_scan - start_time_scan)
            filename_sxm = self.connect.get_next_filename(name, extension='.sxm', folder=folder)
    
            nanonis_const = dict(zip(sigvals_df.T.iloc[0], sigvals_df.T.iloc[1].astype(str)))
            scan_par = {
                "REC_DATE": datetime.now().strftime('%d.%m.%Y'),
                "REC_TIME":  datetime.now().strftime('%H:%M:%S'),
                "ACQ_TIME": str(elapsed_time_scan),
                "SCAN_PIXELS": f"{pix[0]}\t{pix[1]}",
                "SCAN_FILE": filename_sxm,
                "SCAN_RANGE": f"{1e-9 * dim[0]:.6E}\t{1e-9 * dim[1]:.6E}",
                "SCAN_OFFSET": f"{cx:.6E}\t{cy:.6E}",
                "SCAN_ANGLE": str(angle),
                "SCAN_DIR": direction
            }
            settings_dict = {
                "BIAS": str(bias_df['Column2'].iloc[0])
            }
            nanonis_chan_names = filtered_sigvals_df['Column1'].tolist()
            
                        # Define constants
            scan_dir = 'both'
            default_value1 = '1.000E+0'
            default_value2 = '0.000E+0'
            
            # Initialize lists to store the final output
            final_list = []
            
            # Process nanonis_chan_names
            for i, name in enumerate(nanonis_chan_names, start=1):
                base_name, unit = name.split(' (')
                unit = unit.strip(')')
                final_list.append([i, f'{base_name}_avg.', unit, scan_dir, default_value1, default_value2])
    
            nanonis_data = np.array(sigval_ar).T
    
            # Save the final data to an SXM file
            final_data=self.connect.writesxm(False,filename_sxm,settings_dict, scan_par, final_list, nanonis_data)
            # update.3ds
            self.update_end_time(filename_3ds)
            return final_data
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
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
