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
from scipy.ndimage import median_filter
import requests
from queue import Queue
from io import StringIO  # Import StringIO for in-memory text handling
from .log_utils import apply_logging, init_logger
@apply_logging
class photon_meas:
    def __init__(self, connect,connect2=None, connect3=None, logging=True): #connect2 = andor
        self.connect = connect
        self.connect2=connect2
        self.connect3=connect3
        self.logging_enabled = logging
        try:
            session_path = self.connect.UtilSessionPathGet().loc['Session path', 0]
            init_logger(session_path)
        except Exception as e:
            print(f"Failed to initialize logger: {e}")

        self.signal_names = self.connect.SignalsNamesGet() 
        # Initialize URL placeholders
        self.url_cal = None
        self.kinser_dat = None
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
    
    def cr_remove(self,spectra, filter_size=3, offset=300):
        """
        Identify and remove cosmic ray outliers from spectral data and average the acquisitions.
    
        Parameters:
        - spectra: 2D NumPy array where each row is an acquisition (shape: [num_acquisitions, 1024])
        - filter_size: Size of the horizontal median filter (default is 3)
        - offset: Offset value to be subtracted from the spectra (default is 300)
    
        Returns:
        - cleaned_average: 1D NumPy array of the averaged spectra after removing outliers
        """
        
        # Ensure spectra is 2D: If it's 1D, reshape to (1, 1024)
        if spectra.ndim == 1:
            spectra = spectra.reshape(1, -1)  # Reshape to (1, 1024)
            
        # Apply a 2D median filter to the spectra data
        filtered_spectra = median_filter(spectra, size=(spectra.shape[0], filter_size))
    
        # Calculate the threshold based on the error estimate
        adjusted_spectra = filtered_spectra - offset
        threshold = 5 * np.sqrt(np.maximum(1, np.abs(adjusted_spectra)))
    
        # Apply a median filter to the threshold to smooth it out
        smoothed_threshold = median_filter(threshold, size=(spectra.shape[0], filter_size))
    
        # Identify outliers: points where the original spectra differ significantly from the filtered spectra
        outliers = np.abs(spectra - filtered_spectra) > smoothed_threshold
        
        # Replace outliers with the median filtered values
        cleaned_spectra = np.where(outliers, filtered_spectra, spectra)
    
        # Average the cleaned acquisitions
        cleaned_average = np.mean(cleaned_spectra, axis=0)
    
        return cleaned_average
    
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
    
    def ao(self,index,value_mV):
        self.connect.UserOutValSet(index,value_mV*1E-3)
        
        
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
        
    def acquire_data_from_connect_relevant(self, signal_values, acquisition_complete, stop_time,relevant_indices):
        start_time = time.time()
        while not acquisition_complete.is_set() and (time.time() - start_time) < stop_time:
            # Collect signal values from the connect device
            signal_values.append(self.connect.SignalsValsGet(relevant_indices, 1))
            # Sleep briefly to avoid busy-waiting
           # time.sleep(0.1)
        # Signal that acquisition is complete
        acquisition_complete.set()
        
    def acquire_data_from_connect_relevant_2(self, signal_values,acquisition_complete, relevant_indices):
        while not acquisition_complete.is_set():
            # Collect signal values from the connect device
            signal_values.append(self.connect2.SignalsValsGet(relevant_indices,0))
           # time.sleep(0.1)  # Adjust sampling frequency as desired
        
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
    
    def save_params_connect_relevant(self, list_of_dfs):
        # Initialize a dictionary to store lists of values for each signal name
        signal_values = {}
    
        for df_list in list_of_dfs:
            for df in df_list:
                # Skip DataFrames with fewer than 2 columns
                if df.shape[1] < 2:
                    continue
    
                # Assume first column is 'Signal names' and second column contains 'Values'
                signal_names_col = df.iloc[:, 0]
                values_col = df.iloc[:, 1]
    
                # Collect values for each signal name
                for signal, value in zip(signal_names_col, values_col):
                    # Initialize list if the signal name is encountered for the first time
                    if signal not in signal_values:
                        signal_values[signal] = []
                    signal_values[signal].append(value)
    
        # Calculate the mean for each signal and store in the averages dictionary
        averages = {signal: np.mean(values) if values else None for signal, values in signal_values.items()}
    
        return averages

    def save_params_connect_new(self,list_of_dfs, signal_names=None):
        signal_names_df=self.signal_names 
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
    
    
    
    def extract_relevant_indices(self,signal_names_df, signal_names_for_save=None):
        """
        Extracts indices of relevant signal names from signal_names_df.
    
        Parameters:
        signal_names_df : DataFrame
            A DataFrame containing available signal names.
        signal_names_for_save : list
            A list of signal names to extract indices for.
    
        Returns:
        list
            A list of indices corresponding to relevant signal names.
        """
        if signal_names_for_save is None:
            signal_names_for_save = [
                "Bias (V)", "X (m)", "Y (m)", "Z (m)", "Current (A)", 
                "LI Demod 1 Y (A)", "LI Demod 2 Y (A)", "Counter 1 (Hz)"
            ]
        # Ensure that signal_names_df has the correct column
        if 'Signal names' not in signal_names_df.columns:
            raise ValueError("The DataFrame must contain a 'Signal names' column.")
    
        # Extract relevant indices for the signals to acquire
        relevant_indices = [
        int(signal_names_df[signal_names_df['Signal names'] == name].index[0])
        for name in signal_names_for_save if name in signal_names_df['Signal names'].values
        ]

        matching_signals = [name for name in signal_names_for_save if name in signal_names_df['Signal names'].values]
        return relevant_indices, matching_signals


    def spectrum_list(self, acqtime=10, acqnum=1, name="LS-man", user="Jirka",signal_names=None): # spectrum with saving lists
        name="AA"+name
        # Initialize variables
        self.connect2.acqtime_set(acqtime)
        folder=self.connect.UtilSessionPathGet().loc['Session path', 0]
        settings=self.connect2.settings_get()
        signal_names_df=self.signal_names 

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
                    data_dict[f"Counts nf {i+1}"] = data_new['Counts']
                else:
                    data_dict[f"Counts nf {i+1}"] = data_new['Counts']
                    
              #   Optionally, set the stop signal here if you want to stop after each iteration
              #  stop_signal.set() # Uncomment if you want to stop after each response from connect2
        
        except KeyboardInterrupt:
            print("Acquisition interrupted.")
        finally:
            # Ensure that `data` is created even if interrupted
            counts_columns = np.array([data_dict[f"Counts nf {i + 1}"].to_numpy() for i in range(acqnum)])
            data_dict["Counts"] = self.cr_remove(counts_columns,filter_size=5,offset=305).tolist()
            
            # Reorder data_dict to place "Counts" after "Wavelength (nm)"
            data_dict = {
    **{k: v for k, v in data_dict.items() if k == "Wavelength (nm)"},
        "Counts": data_dict["Counts"],
        **{k: v for k, v in data_dict.items() if k not in ["Wavelength (nm)", "Counts"]}
    }
            
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
        signal_names_df=self.signal_names 
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

    def spectrum(self, acqtime=10, acqnum=1, name="LS-man", user="Jirka",signal_names=None,readmode=0):# spectrum with only saving the relevant channels and using np array to store nanonis data
        name="AA"+name
        # Initialize variables
        self.connect2.acqtime_set(acqtime)
        folder=self.connect.UtilSessionPathGet().loc['Session path', 0]
        settings=self.connect2.settings_get()
        signal_names_df=self.signal_names 
        relevant_indices,matching_signals=self.extract_relevant_indices(signal_names_df, signal_names_for_save=signal_names)
        
        nanonis_shape,andor_shape = (acqnum,len(relevant_indices)),(acqnum,1024)  # For example, if you want to concatenate 5 arrays
        nanonis_array = np.full(nanonis_shape,np.nan, dtype=np.float64)
        #andor_array = np.full(andor_shape,np.nan, dtype=np.int64)
        
        # check if spectrograph setting is corrent and eventually change it
        for index, row in settings.iterrows():
            code = row['Code']
            value = row['Value']
            if code == 'GRM' and (value == 4 or readmode not in [0, "FVB"] or value != readmode):
                print(f"Camera in image mode!: GRM with value {value}, setting it to FVB mode.")
                self.connect2.readmode_set(readmode)
                settings.at[index, 'Value'] = readmode  # Update settings DataFrame
            elif code == 'GAM' and value != 1:
                print(f"Acq. mode with value {value} invalid, setting it to single (1) mode.")
                self.connect2.acqmode_set(1)
                settings.at[index, 'Value'] = 1  # Update settings DataFrame
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
                acquire_thread_connect = threading.Thread(target=self.acquire_data_from_connect_relevant, args=(signal_values, acquisition_complete_connect, acqtime,relevant_indices)) 
                acquire_thread_connect.start()
                
                # Wait for the acquisition to complete on connect2
                acquisition_complete_connect2.wait()  # This will block until acquisition_complete_connect2 is set
                
                # Ensure the connect thread has finished
                acquire_thread_connect.join()
                acquire_thread2.join()
                
                # Process the acquired signal values
                nanonis_array[i,:]=np.nanmean(np.stack([df.iloc[:, 1].values for df in signal_values]), axis=0)
                #del signal_values
                # Update the DataFrame with new data from connect2
                data_new = data_storage['data']
                if i == 0:
                    data_dict['Wavelength (nm)'] = data_new['Wavelength (nm)']
                    data_dict[f"Counts nf {i+1}"] = data_new['Counts']
                else:
                    data_dict[f"Counts nf {i+1}"] = data_new['Counts']
                    
              #   Optionally, set the stop signal here if you want to stop after each iteration
              #  stop_signal.set() # Uncomment if you want to stop after each response from connect2
        
        except KeyboardInterrupt:
            print("Acquisition interrupted.")
        finally:
            # Ensure that `data` is created even if interrupted
            counts_columns = np.array([data_dict[f"Counts nf {i + 1}"].to_numpy() for i in range(acqnum)])
            data_dict["Counts"] = self.cr_remove(counts_columns,filter_size=5,offset=305).tolist()
            
            # Reorder data_dict to place "Counts" after "Wavelength (nm)"
            data_dict = {
**{k: v for k, v in data_dict.items() if k == "Wavelength (nm)"},
    "Counts": data_dict["Counts"],
    **{k: v for k, v in data_dict.items() if k not in ["Wavelength (nm)", "Counts"]}
}
            
            data = pd.DataFrame(data_dict)
                        
            formatted_date_str = datetime.now().strftime('%d.%m.%Y %H:%M:%S')

            # Data to prepend
            prepend_data = {
                'Signal names': ['Experiment', 'Date', 'User'],
                'Value': ['LS', formatted_date_str, user]
            }
            
            # Create DataFrames
            prepend_df = pd.DataFrame(prepend_data)
            
            nanonis_mean_array = np.nanmean(nanonis_array, axis=0)
            
            sigvals_df = pd.DataFrame({
                'Signal names': matching_signals,
                'Value': nanonis_mean_array
                })
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
                     
        return  sigvals_df,combined_df, prepend_df

    def photon_map_v2(self, acqtime=10, acqnum=1, pix=(10, 10), dim=None, name="LS-man", user="Jirka", signal_names=None,savedat=False,direction="up",backward=False):
        self.connect2.acqtime_set(acqtime)
        self.connect2.acqnum_set(acqnum)
        # Initialize variables
        if direction in ["up", True, 0]:
            direction = "up"
        elif direction in ["down", False, 1]:
            direction = "down"
        else:
            raise ValueError("Invalid direction. Use 'up', 'down', True, False, 0, or 1.")   
        start_time_scan = time.perf_counter()
        folder = self.connect.UtilSessionPathGet().loc['Session path', 0]
        signal_names_df = self.signal_names 
        SF = self.connect.ScanFrameGet()  # Retrieve scan frame
        settings=self.connect2.settings_get()
        if dim is None: 
            dim=(1e9*SF.values[2][0],1e9*SF.values[3][0]) # gets the dimension from the scan frame
        cx, cy, angle = SF.values[0][0], SF.values[1][0], SF.values[4][0]  # Extract center and angle
        wait_num = True  # Wait flag
    
       # channels_of_interest = ['Z (m)', 'Current (A)', 'LI Demod 1 Y (A)', 'LI Demod 2 Y (A)', 'Counter 1 (Hz)']
        data_ar=[]
        sigval_ar=[]
        counter,count_write=0,0
        
        relevant_indices,matching_signals=self.extract_relevant_indices(signal_names_df, signal_names_for_save=signal_names) #filter only selected channels to save 
        
        nanonis_shape,andor_shape = (acqnum,len(relevant_indices)),(acqnum,1024)  # For example, if you want to concatenate 5 arrays
        nanonis_array = np.full(nanonis_shape,np.nan, dtype=np.float64)
        
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
                    
                   # sigvals = []
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
                        acquire_thread2 = threading.Thread(target=self.acquire_datdaa_from_connect2, args=(data_storage, acquisition_complete_connect2))
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
             #           sigvals.append(signal_values)

                        nanonis_array[i,:]=np.mean(np.stack([df.iloc[:, 1].values for df in signal_values]), axis=0)
                        del signal_values
                        
                        # Update the DataFrame with new data from connect2
                        data_new = data_storage['data']
                        if i == 0:
                            data_dict['Wavelength (nm)'] = data_new['Wavelength (nm)']
                            data_dict[f"Counts nf {i+1}"] = data_new['Counts']
                        else:
                            data_dict[f"Counts nf {i+1}"] = data_new['Counts']
                        
                        # Optionally, set the stop signal here if you want to stop after each iteration
                        # stop_signal.set() # Uncomment if you want to stop after each response from connect2
                    
                    # SAVE FILES
                    
                    counts_columns = np.array([data_dict[f"Counts nf {i + 1}"].to_numpy() for i in range(acqnum)])
                    data_dict["Counts"] = self.cr_remove(counts_columns,filter_size=5,offset=305).tolist()
                    
                    # Reorder data_dict to place "Counts" after "Wavelength (nm)"
                    data_dict = {
        **{k: v for k, v in data_dict.items() if k == "Wavelength (nm)"},
            "Counts": data_dict["Counts"],
            **{k: v for k, v in data_dict.items() if k not in ["Wavelength (nm)", "Counts"]}
        }
                    
                    # Convert to DataFrame
                    data = pd.DataFrame(data_dict)
                    del data_dict

                    # print("Sigvals before processing:", sigvals)
                    # print(signal_names_df)
                    
                #    sigvals = self.save_params_connect(sigvals, signal_names_df, signal_names=signal_names)
                    
                    formatted_date_str = datetime.now().strftime('%d.%m.%Y %H:%M:%S')
                    
                    # Data to prepend
                    prepend_data = {
                        'Signal names': ['Experiment', 'Date', 'User'],
                        'Value': ['LS', formatted_date_str, user]
                    }
                    
                    # Create DataFrames
                    prepend_df = pd.DataFrame(prepend_data)
                    
                    nanonis_mean_array = np.nanmean(nanonis_array, axis=0)
                    sigvals_df = pd.DataFrame({
                        'Signal names': matching_signals,
                        'Value': nanonis_mean_array
                        })
                    
                 #   sigvals_df = pd.DataFrame(list(sigvals.items()), columns=['Column1', 'Column2'])
                    
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
                   # filtered_sigvals_df = sigvals_df[sigvals_df['Column1'].isin(channels_of_interest)]
                    sigval_ar.append(sigvals_df['Value'].tolist())
                  #  filtered_sigvals_list=filtered_sigvals_df['Column2'].tolist()
                    num_signals = len(sigvals_df)
                    num_pixels=selected_columns.shape[0]
                    
                    # write to .3ds file
# start problematic section
                    if row==0 and index==0:
                        fixed_parameters = ["Sweep Start", "Sweep End"]+sigvals_df['Signal names'].tolist()
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
                    res_list=[float(andor_chan_names[0]),float(andor_chan_names[-1])]+matching_signals
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
                "SCAN_TIME": f"{acqtime*pix[0]:.6E}\t{acqtime*pix[0]:.6E}",
                "SCAN_RANGE": f"{1e-9 * dim[0]:.6E}\t{1e-9 * dim[1]:.6E}",
                "SCAN_OFFSET": f"{cx:.6E}\t{cy:.6E}",
                "SCAN_ANGLE": str(angle),
                "SCAN_DIR": direction,
                "BIAS": str(sigvals_df[sigvals_df['Column1'].isin(["Bias (V)"])]['Column2'].iloc[0])
                }   
            andor_chan_names= data.iloc[:, 0].values.tolist()
            nanonis_chan_names=sigvals_df['Signal names'].tolist()
            
            nanononis_data_to_sxm = np.array(sigval_ar, dtype=np.float32).T
            andor_data_to_sxm = np.array(data_ar, dtype=np.float32).T
            combined_data = np.concatenate((nanononis_data_to_sxm, andor_data_to_sxm), axis=0)
            del nanononis_data_to_sxm, andor_data_to_sxm #delete intermediate data
            
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
            del combined_data

            # Ensure that `data` is created even if interrupted
     
              #  end_time = time.perf_counter()
              #  elapsed_time = end_time - start_time
               # print(f"Time taken for pix command: {elapsed_time:.4f} seconds")
                          
            #return data_sxm,combined_data
        
        
    def photon_map(self, acqtime=10, acqnum=1, pix=(10, 10), dim=None, name="LS-man", user="Jirka", signal_names=None,savedat=False,direction="up",backward=False,readmode=0):
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
        signal_names_df = self.signal_names 
        SF = self.connect.ScanFrameGet()  # Retrieve scan frame
        
        settings=self.connect2.settings_get()
        # check if spectrograph setting is corrent and eventually change it
        for index, row in settings.iterrows():
            code = row['Code']
            value = int(row['Value'])
            if code == 'GRM' and (value == 4 or readmode not in [0, "FVB"]):
                print(f"Camera in image mode!: GRM with value {value}, setting it to FVB mode.")
                self.connect2.readmode_set(readmode)
            elif code == 'GAM' and value != 1:
                print(f"Acq. mode with value {value} invalid, setting it to single (1) mode.")
                self.connect2.acqmode_set(1)
                
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
                            data_dict[f"Counts nf {i+1}"] = data_new['Counts']
                        else:
                            data_dict[f"Counts nf {i+1}"] = data_new['Counts']
                        
                        # Optionally, set the stop signal here if you want to stop after each iteration
                        # stop_signal.set() # Uncomment if you want to stop after each response from connect2
                    
                    # SAVE FILES
                    
                    counts_columns = np.array([data_dict[f"Counts nf {i + 1}"].to_numpy() for i in range(acqnum)])
                    data_dict["Counts"] = self.cr_remove(counts_columns,filter_size=5,offset=305).tolist()
                    
                    # Reorder data_dict to place "Counts" after "Wavelength (nm)"
                    data_dict = {
        **{k: v for k, v in data_dict.items() if k == "Wavelength (nm)"},
            "Counts": data_dict["Counts"],
            **{k: v for k, v in data_dict.items() if k not in ["Wavelength (nm)", "Counts"]}
        }
                    
                    # Convert to DataFrame
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
            
            settings_dict=(dict(zip(settings.T.iloc[0], settings.T.iloc[1].astype(str))))
            scan_par = {
                "REC_DATE": datetime.now().strftime('%d.%m.%Y'),
                "REC_TIME":  datetime.now().strftime('%H:%M:%S'),
                "ACQ_TIME": str(elapsed_time_scan),
                "SCAN_PIXELS": f"{pix[0]}\t{pix[1]}",
                "SCAN_FILE": filename_sxm,
                "SCAN_TIME": f"{acqtime*pix[0]:.6E}\t{acqtime*pix[0]:.6E}",
                "SCAN_RANGE": f"{1e-9 * dim[0]:.6E}\t{1e-9 * dim[1]:.6E}",
                "SCAN_OFFSET": f"{cx:.6E}\t{cy:.6E}",
                "SCAN_ANGLE": str(angle),
                "SCAN_DIR": direction,
                "BIAS": str(sigvals_df[sigvals_df['Column1'].isin(["Bias (V)"])]['Column2'].iloc[0])
                }   
            andor_chan_names= data.iloc[:, 0].values.tolist()
            nanonis_chan_names=filtered_sigvals_df['Column1'].tolist()
            
            nanonis_data_to_sxm=np.array(sigval_ar).T
            andor_data_to_sxm=np.array(data_ar).T
            combined_data = np.vstack((nanonis_data_to_sxm, andor_data_to_sxm))
            
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
        
        def photon_map_kinetic_v1(self, acqtime=10, acqnum=1, pix=(10, 10), dim=None, name="LS-man", user="Jirka", signal_names=None,savedat=False,direction="up",backward=False,readmode=0):
            # Events to coordinate between threads
            acquisition_complete = threading.Event()
            andor_ready = threading.Event()
            position_ready = threading.Event()
            stop_signal = threading.Event()  # Overall stop signal for acquisition
            
            # Placeholder function for Andor command
            def send_andor_command():
                print("Thread 1: Sending Andor command...")
                # Simulate waiting for response from Andor
                # Code for Andor command and response goes here
                # Once response is received, signal readiness to other threads
                andor_ready.set()
                print("Thread 1: Andor command response received.")
            
            # Placeholder function to monitor port and set position
            def monitor_port():
                print("Thread 2: Waiting for Andor readiness...")
                andor_ready.wait()  # Wait for the first thread to signal readiness
                print("Thread 2: Andor is ready. Monitoring port...")
                
                while not stop_signal.is_set():
                    # Code for readport and checking condition
                    port_status = readport()  # Custom function returns 1 or 0
                    if port_status == 1:
                        print("Thread 2: Port condition met, setting position...")
                        # Code for FolMeXYPosSet and waiting for response
                        position_ready.set()  # Notify acquisition can be stopped
                        break
                print("Thread 2: Position set, stopping acquisition.")
            
            # Placeholder function for signal acquisition
            def acquire_data_from_connect(signal_values):
                print("Thread 3: Starting data acquisition...")
                while not acquisition_complete.is_set():
                    # Code for acquiring signals goes here
                    pass
                print("Thread 3: Acquisition stopped.")
            
            # Start threads
            andor_thread = threading.Thread(target=send_andor_command)
            port_thread = threading.Thread(target=monitor_port)
            acquisition_thread = threading.Thread(target=acquire_data_from_connect, args=(signal_values,))
            
            # Start all threads
            andor_thread.start()
            port_thread.start()
            acquisition_thread.start()
            
            # Synchronize threads as per completion of tasks
            position_ready.wait()  # Wait for port thread to complete its task
            acquisition_complete.set()  # Stop the acquisition thread
            
            # Join threads to ensure they complete
            andor_thread.join()
            port_thread.join()
            acquisition_thread.join()
            
            print("All tasks completed.")
            
       
    
    def zig_zag_move_acq(self, acqtime=10, acqnum=1, pix=(10, 10), dim=None, name="LS-man", user="Jirka", signal_names=None,savedat=False,direction="up",backward=False,readmode=0):       
        def bin_average_stacked(array, n):
            m = array.shape[0]  # Number of rows (measurements)
            bin_size = m / n
            result = np.empty((n, array.shape[1]))
            print(m,bin_size,n)
            
            for i in range(n):
                start = int(i * bin_size)
                end = int((i + 1) * bin_size)
                result[i,:] = np.mean(array[start:end,:], axis=0)
            return result

        # Retrieve scan frame and set default dimensions if not provided
        SF = self.connect.ScanFrameGet()
        dim = (1e9 * SF.values[2][0], 1e9 * SF.values[3][0]) if dim is None else dim
        cx, cy, angle = SF.values[0][0], SF.values[1][0], SF.values[4][0]
        signal_names_df=self.signal_names 
        relevant_indices,matching_signals=self.extract_relevant_indices(signal_names_df, signal_names_for_save=signal_names)
        
        # Initial setup for the first point in the bottom-left corner
        initial_dx_nm, initial_dy_nm = (-dim[0] / 2, -dim[1] / 2) if direction == "up" else (-dim[0] / 2, dim[1] / 2)
        initial_dx_rot, initial_dy_rot = self.rotate(initial_dx_nm, initial_dy_nm, angle)
        
        # Move to the first point
        #print(f"Moving to initial start point (dx_nm: {initial_dx_nm}, dy_nm: {initial_dy_nm})")
        self.connect.FolMeXYPosSet(cx + initial_dx_rot, cy + initial_dy_rot,True)
        
        mv_spd=1e-9*(np.sqrt(dim[0]**2+(dim[1]/pix[1]/2)**2)/(acqtime*pix[0]))
        self.connect.FolMeSpeedSet(mv_spd,1)
        
        # Set scan direction parameters
        row_range = range(2 * pix[1])# if direction == "up" else range(2 * pix[1] - 1, -1, -1)
        dy_sign = -1 if direction == "up" else 1
        
        # Create the signal_array with NaN values
        #signal_array = np.empty(len(row_range), dtype=object)
        signal_array=np.full((len(row_range),pix[1],len(relevant_indices)), np.nan, dtype=np.float32)
        print(signal_array.shape)
        s_time=time.perf_counter()
        # Start zigzag pattern)
        for index, row in enumerate(row_range):
        # You can now use 'index' to access signal_array

            # Calculate y-coordinate for the current row
            dy_nm = (dy_sign * dim[1] / 2) + (row + 1) * (dim[1] / pix[1]) / 2 * (-dy_sign)
        
            # Determine end x-coordinate based on row direction
            end_dx_nm = dim[0] / 2 if row % 2 == 0 else -dim[0] / 2
        
            # Rotate the end coordinates and move to the end point of the current row
            end_dx_rot, end_dy_rot = self.rotate(end_dx_nm, dy_nm, angle)
           # print(f"{direction.capitalize()}ward scan - Moving to end point (dx_nm: {end_dx_nm}, dy_nm: {dy_nm})")
           
            # Event to stop acquisition
            acquisition_complete = threading.Event()
            signal_values = []            
            time1 = time.perf_counter()
            acquire_thread = threading.Thread(target=self.acquire_data_from_connect_relevant_2, args=(signal_values,acquisition_complete, relevant_indices))
            acquire_thread.start()
            time2 = time.perf_counter()
            # Start move and wait for completion
            self.connect.FolMeXYPosSet(cx + end_dx_rot, cy + end_dy_rot, True)
            
            # Signal acquisition to stop once movement is complete
            acquisition_complete.set()
            time3 = time.perf_counter()
            # Wait for the move to complete and stop acquisition thread
            acquire_thread.join()  # Ensure acquisition stops exactly when movement completes

            
            stacked_data = np.stack([df.iloc[:, 1].values for df in signal_values])
            

            signal_array[index]=bin_average_stacked(stacked_data,pix[1])
            #signal_array[index]=bin_average_stacked(stacked_data,pix[0])
            time4 = time.perf_counter()
            print(len(signal_values), "{:.5f}".format(time4-time3),"{:.5f}".format(time3-time2),"{:.5f}".format(time2-time1))
        e_time=time.perf_counter()
        print("tot time", "{:.5f}".format(e_time-s_time))
        self.connect.FolMeSpeedSet(mv_spd,0)
        #return(np.stack(signal_array,axis=0))
        return(signal_array)


    def zig_zag_move_acq_folme(self, acqtime=10, acqnum=1, pix=(10, 10), dim=None, name="LS-man", user="Jirka", signal_names=None,savedat=False,direction="up",backward=False,readmode=0):       
        def bin_average_stacked(array, n):
            m = array.shape[0]  # Number of rows (measurements)
            bin_size = m / n
            result = np.empty((n, array.shape[1]))
            print(m,bin_size,n)
            
            for i in range(n):
                start = int(i * bin_size)
                end = int((i + 1) * bin_size)
                result[i,:] = np.mean(array[start:end,:], axis=0)
            return result

        # Retrieve scan frame and set default dimensions if not provided
        SF = self.connect.ScanFrameGet()
        dim = (1e9 * SF.values[2][0], 1e9 * SF.values[3][0]) if dim is None else dim
        cx, cy, angle = SF.values[0][0], SF.values[1][0], SF.values[4][0]
        
        signals_slots=self.connect.SignalsInSlotsGet(prt=False)

        if signal_names is None:
            signal_names = [
                "Bias (V)", "X (m)", "Y (m)", "Z (m)", "Current (A)", 
                "LI Demod 1 Y (A)", "LI Demod 2 Y (A)", "Counter 1 (Hz)"
            ]
        # Filter rows in `signals_slots` where the first column matches any item in `signal_names`
        signals =signals_slots.iloc[:, 0].tolist()
        matching_indices = [i for i, signal in enumerate(signals) if signal in signal_names]
        unmatched_items = [item for item in signal_names if item not in signals]
        
        scan_buffer=self.connect.ScanBufferGet()
        num_chs, ch_idx, pxs, lines = scan_buffer.iloc[:, 0]
        self.connect.ScanBufferSet(len(matching_indices),matching_indices,pxs, lines) 
             
        # Initial setup for the first point in the bottom-left corner
        initial_dx_nm, initial_dy_nm = (-dim[0] / 2, -dim[1] / 2) if direction == "up" else (-dim[0] / 2, dim[1] / 2)
        initial_dx_rot, initial_dy_rot = self.rotate(initial_dx_nm, initial_dy_nm, angle)
        
        # Move to the first point
        #print(f"Moving to initial start point (dx_nm: {initial_dx_nm}, dy_nm: {initial_dy_nm})")
        self.connect.FolMeXYPosSet(cx + initial_dx_rot, cy + initial_dy_rot,True)
        
        mv_spd=1e-9*(np.sqrt(dim[0]**2+(dim[1]/pix[1]/2)**2)/(acqtime*pix[0]))
        num_points=pix[0]*20
        spl_rate=20/acqtime
        ovs=max(1, min(2E4/spl_rate, 1000))
        num_points=2E4/ovs*acqtime*pix[0]+200 # 200 margin
        self.connect.FolMeOversamplSet(ovs,prt=False)
        self.connect.TipRecBufferSizeSet(num_points,prt=False)
        self.connect.FolMeSpeedSet(mv_spd,1)
        
        # Set scan direction parameters
        row_range = range(2 * pix[1])# if direction == "up" else range(2 * pix[1] - 1, -1, -1)
        dy_sign = -1 if direction == "up" else 1
        
        # Create the signal_array with NaN values
        #signal_array = np.empty(len(row_range), dtype=object)
        signal_array=np.full((len(row_range),pix[1],len(matching_indices)), np.nan, dtype=np.float32)
        print(signal_array.shape)
        s_time=time.perf_counter()
        # Start zigzag pattern)
        for index, row in enumerate(row_range):
        # You can now use 'index' to access signal_array

            # Calculate y-coordinate for the current row
            dy_nm = (dy_sign * dim[1] / 2) + (row + 1) * (dim[1] / pix[1]) / 2 * (-dy_sign)
        
            # Determine end x-coordinate based on row direction
            end_dx_nm = dim[0] / 2 if row % 2 == 0 else -dim[0] / 2
        
            # Rotate the end coordinates and move to the end point of the current row
            end_dx_rot, end_dy_rot = self.rotate(end_dx_nm, dy_nm, angle)
           # print(f"{direction.capitalize()}ward scan - Moving to end point (dx_nm: {end_dx_nm}, dy_nm: {dy_nm})")
            time1 = time.perf_counter()
            # Start move and wait for completion
            self.connect.FolMeXYPosSet(cx + end_dx_rot, cy + end_dy_rot, True)
            time2 = time.perf_counter()
            chan,data=self.connect.TipRecDataGet()
            self.connect.TipRecBufferClear()           
            time3 = time.perf_counter()
            signal_array[index]=bin_average_stacked(np.transpose(data),pix[1])
            #signal_array[index]=bin_average_stacked(stacked_data,pix[0])
            time4 = time.perf_counter()
            print( "{:.5f}".format(time4-time3),"{:.5f}".format(time3-time2),"{:.5f}".format(time2-time1))
        e_time=time.perf_counter()
        print("tot time", "{:.5f}".format(e_time-s_time))
        
        self.connect.FolMeSpeedSet(mv_spd,0)
        self.connect.ScanBufferSet(*scan_buffer.iloc[:, 0].values)
        #return(np.stack(signal_array,axis=0))
        return(signal_array)
            
    
    def fwbw_move_acq_folme(self, acqtime=10, acqnum=1, pix=(10, 10), dim=None, name="LS-man", user="Jirka", signal_names=None,savedat=False,direction="up",backward=False,readmode=0):       
        def bin_average_stacked(array, n):
            m = array.shape[1]  # Number of rows (measurements)
            bin_size = m / n
            result = np.empty((n, array.shape[0]))
            
            for i in range(n):
                start = int(i * bin_size)
                end = int((i + 1) * bin_size)
                result[i,:] = np.mean(array[:,start:end], axis=1)
            return result
        
        def calculate_parameters(acqtime, pix, backward=True):
            # Initial calculation of sampling rate and ovs
            spl_rate = 20 / acqtime
            ovs = int(min(max(int(20000 / spl_rate), 2), 30))  # Ensure ovs is within the range [2, 30]

            # Calculate num_points based on the current ovs
            num_points = (2 * (20000 / ovs * acqtime * pix[0]) + 20) if backward else (20000 / ovs * acqtime * pix[0] + 20)

            # Adjust ovs if num_points exceeds 200,000
            while num_points >= 200000 and ovs < 300:
                ovs += 1
                num_points = int((2 * (20000 / ovs * acqtime * pix[0]) + 20) if backward else (20000 / ovs * acqtime * pix[0] + 20))+1
                print(ovs)

            # Final check to ensure num_points constraint is met
            if num_points >= 200000:
                print("Warning: Cannot satisfy num_points < 200000 even with adjusted ovs. Use map with single points.")
            
            return num_points / pix[0], int(ovs), num_points
        
        # Retrieve scan frame and set default dimensions if not provided
        SF = self.connect.ScanFrameGet()
        dim = (1e9 * SF.values[2][0], 1e9 * SF.values[3][0]) if dim is None else dim
        cx, cy, angle = SF.values[0][0], SF.values[1][0], SF.values[4][0]
        bw_ratio=10
        signals_slots=self.connect.SignalsInSlotsGet(prt=False)

        if signal_names is None:
            signal_names = [
                "Bias (V)", "X (m)", "Y (m)", "Z (m)", "Current (A)", 
                "LI Demod 1 Y (A)", "LI Demod 2 Y (A)", "Counter 1 (Hz)"
            ]
        # Filter rows in `signals_slots` where the first column matches any item in `signal_names`

        signals =signals_slots.iloc[:, 0].tolist()
        matching_indices = [i for i, signal in enumerate(signals) if signal in signal_names]
        matching_items = [item for item in signal_names if item in signals]
        unmatched_items = [item for item in signal_names if item not in signals]
        print(f'Signals {unmatched_items} not in slots')
        print(f'Signals {matching_items} in slots')
        
        scan_buffer=self.connect.ScanBufferGet()
        num_chs, ch_idx, pxs, lines = scan_buffer.iloc[:, 0]
        self.connect.ScanBufferSet(len(matching_indices),matching_indices,pxs, lines) 
             
        # Initial setup for the first point in the bottom-left corner
        initial_dx_nm, initial_dy_nm = (-dim[0] / 2, -dim[1] / 2 + (dim[1] / pix[1])/2) if direction == "up" else (-dim[0] / 2, dim[1] / 2 - (dim[1] / pix[1])/2)
        initial_dx_rot, initial_dy_rot = self.rotate(initial_dx_nm, initial_dy_nm, angle)
        
        # Move to the first point
        self.connect.FolMeXYPosSet(cx + initial_dx_rot, cy + initial_dy_rot,True)
        
        mv_spd=1e-9*(dim[0]/(acqtime*pix[0]))
        #num_points=pix[0]*20
       # spl_rate=20/acqtime
       # ovs=max(3, min(2E4/spl_rate, 60))
        #num_points = (2E4 / ovs * acqtime * pix[0] + 200) if not backward else (2 * (2E4 / ovs * acqtime * pix[0]) + 200) # 200 margin
        
        pppix, ovs, num_points = calculate_parameters(acqtime, pix, backward)
        self.connect.FolMeOversamplSet(ovs,prt=False)
        self.connect.TipRecBufferSizeSet(num_points,prt=False)
        self.connect.TipRecBufferClear()
        self.connect.FolMeSpeedSet(mv_spd,1)
        
        # Set scan direction parameters
        row_range = range(2 * pix[1])
        dy_sign = -1 if direction == "up" else 1
        dy_plus=0
        
        # Create the signal_array with NaN values
        signal_array=np.full((pix[0],len(row_range),len(matching_indices)), np.nan, dtype=np.float32)
        print(signal_array.shape)
        s_time=time.perf_counter()
        # Start zigzag pattern)
        for index, row in enumerate(row_range):
        # You can now use 'index' to access signal_array

            # Calculate y-coordinate for the current row
            if row % 2 == 0 and row!=0:
                dy_plus+=(dim[1] / pix[1]) * (-dy_sign)
                
            dy_nm=initial_dy_nm+dy_plus
        
            # Determine end x-coordinate based on row direction
            end_dx_nm = dim[0] / 2 if row % 2 == 0 else -dim[0] / 2
        
            # Rotate the end coordinates and move to the end point of the current row
            end_dx_rot, end_dy_rot = self.rotate(end_dx_nm, dy_nm, angle)

            # Start move and wait for completion
            
            if backward==False:
                if row % 2 == 0 and row!=0: 
                    dx_n_rot,dy_n_rot=self.rotate(-end_dx_nm,dy_nm, angle)
                    self.connect.FolMeXYPosSet(cx + dx_n_rot, cy + dy_n_rot, True) # move one pixel up after finishing fw bw line
                    self.connect.FolMeSpeedSet(mv_spd,1)
                    self.connect.TipRecBufferClear()
                    
                self.connect.FolMeXYPosSet(cx + end_dx_rot, cy + end_dy_rot, True)  #regular scan move
    
                if row % 2 == 0:
                    _,data=self.connect.TipRecDataGet()   # save data containing only fw 
                    temp_data = bin_average_stacked(data, pix[0]) # analyse to fw and bw movement and make avarage of pixels
                    print(data.shape,temp_data.shape,index,index+1)
                    signal_array[:,index,:], signal_array[:,index + 1,:] = temp_data, temp_data[::-1,:] # write first fw an later fw reversed
                    self.connect.FolMeSpeedSet(bw_ratio*mv_spd,1)
                
                
            else:
                if row % 2 == 0 and row!=0:
                    dx_n_rot,dy_n_rot=self.rotate(-end_dx_nm,dy_nm, angle)
                    self.connect.FolMeXYPosSet(cx + dx_n_rot, cy + dy_n_rot, True) # move one pixel up after finishing fw bw line
                    self.connect.TipRecBufferClear()
                    
                self.connect.FolMeXYPosSet(cx + end_dx_rot, cy + end_dy_rot, True) #regular scan move
    
                if row % 2 == 1:
                    _,data=self.connect.TipRecDataGet()   # save data containing fw and bw
                    temp_data = bin_average_stacked(data, 2 * pix[0]) # analyse to fw and bw movement and make avarage of pixels
                    print(data.shape,temp_data[:pix[0], :].shape,temp_data[pix[0]:, :].shape,index-1,index)
                    signal_array[:,index - 1,:], signal_array[:,index,:] = temp_data[:pix[0], :], temp_data[pix[0]:, :] # write first fw an later bw
           # print("loop time", "{:.5f}".format((time.perf_counter()-s_time)/(index+1)),"theoretic loop time","{:.5f}".format(acqtime*pix[0]) )
        e_time=time.perf_counter()
        print("tot time", "{:.5f}".format(e_time-s_time))
        
        self.connect.FolMeSpeedSet(mv_spd,0)
        self.connect.ScanBufferSet(*scan_buffer.iloc[:, 0].values)
        #return(np.stack(signal_array,axis=0))
        return(signal_array)
    
    
    def bin_average_stacked(self,array, n):
        m = array.shape[1]  # Number of rows (measurements)
        bin_size = m / n
        result = np.empty((n, array.shape[0]))
        
        for i in range(n):
            start = int(i * bin_size)
            end = int((i + 1) * bin_size)
            result[i,:] = np.mean(array[:,start:end], axis=1)
        return result
    
    def calculate_parameters(self,acqtime, pix, backward=True):
        # Initial calculation of sampling rate and ovs
        spl_rate = 20 / acqtime
        ovs = int(min(max(int(20000 / spl_rate), 2), 30))  # Ensure ovs is within the range [2, 30]

        # Calculate num_points based on the current ovs
        num_points = (2 * (20000 / ovs * acqtime * pix[0]) + 20) if backward else (20000 / ovs * acqtime * pix[0] + 20)

        # Adjust ovs if num_points exceeds 200,000
        while num_points >= 200000 and ovs < 300:
            ovs += 1
            num_points = int((2 * (20000 / ovs * acqtime * pix[0]) + 20) if backward else (20000 / ovs * acqtime * pix[0] + 20))+1
            print(ovs)

        # Final check to ensure num_points constraint is met
        if num_points >= 200000:
            print("Warning: Cannot satisfy num_points < 200000 even with adjusted ovs. Use map with single points.")
        
        return num_points / pix[0], int(ovs), num_points
    
    def fetch_data_from_queue_v2(self, fetch_queue: Queue, delta: float, n: int):
        """
        Fetches data from an external source and processes it.
    
        Args:
            fetch_queue (Queue): Queue to receive notifications for fetching data.
            delta (float): Time delay between each fetch.
            n (int): Desired number of rows for reshaping the data.
    
        """
        port = str(self.connect2.tcp.server_addr[1])
        path = "/shm/andor.dat"
        address = self.connect2.tcp.server_addr[0]
        
        # Construct the URL
        url = f"http://{address}:{port}{path}"
    
        while True:
            # Wait for a notification to fetch data
            item = fetch_queue.get()  # Block until there's a new item in the queue
            if item is None:  # Check for shutdown signal
                break
    
            # Wait for the specified delta time
            time.sleep(delta)
    
            try:
                # Send an HTTP GET request to fetch the data file
                response = requests.get(url)
    
                # Check if the request was successful
                if response.status_code == 200:
                    # Use np.fromstring to convert the entire response text into an array of floats
                    data_array = np.fromstring(response.text, sep='\n', dtype=float)
    
                    # Ensure the data array length matches the expected shape
                    m = data_array.size // n
                    if data_array.size != n * m:
                        print("Warning: The data length is not divisible by the specified rows.")
                        data_array = data_array[:n * m]  # Trim any extra values if needed
    
                    # Reshape the data array
                    data_array = data_array.reshape((n, m))
    
                    # Further processing of data_array can be done here
                    print("Data fetched and reshaped successfully.")
                    print(data_array)
    
                else:
                    print(f"Failed to fetch data. Status code: {response.status_code}")
            
            except Exception as e:
                print(f"An error occurred: {e}")
            
            finally:
                fetch_queue.task_done()  # Mark the task as done
                
    def andor_thread_open(self,port=2,index=0,wait_time=None):
        andor_thread=threading.Thread(target=self.connect2.kinser_start)
        andor_thread.start()
        if wait_time==None:
           while True:
               if self.connect.DigLines_TTLValGet(port=port)[index] == 0: # low or high input
                   break
        else:
            time.sleep(wait_time)
        return(andor_thread)
    
    def build_urls(self):
        # Only build URLs for kinetic series when needed
        TCP_IP, PORT = self.connect2.tcp.server_addr
        PORT=1234
        self.url_cal = f"http://{TCP_IP}:{PORT}/shm/andor.cal"
        self.kinser_dat = f"http://{TCP_IP}:{PORT}/shm/andor.dat"

    def get_cal(self): #maybe move to andor_meas later
        if self.url_cal is None or self.kinser_dat is None:
            self.build_urls()
        try:
            response = requests.get(self.url_cal)  # Send GET request to URL
            if response.status_code == 200:  # Check if request is successful
                data = response.text  # Get the response text
                split_index = data.rfind("\n\n") + 1  # Find the last empty line to split data
                cal = np.genfromtxt(StringIO(data[:split_index]), dtype=float, invalid_raise=False)  # Load numeric data into NumPy array
                return cal, len(cal)  # Return data and pixel count
            print(f"Failed: {response.status_code}")  # Print error if request failed
        except Exception as e:  # Catch any exceptions
            print(f"Error: {e}")  # Print exception error
            return None, None  # Return None if an error occurred
        
    def read_kinser(self,n): #maybe move to andor_meas later
        """Fetch data, reshape it to 1024 x n, and return."""
        try:
            response = requests.get(self.kinser_dat)  # Fetch data
            if response.status_code != 200: raise ValueError(f"Error: {response.status_code}")
            
            data = np.loadtxt(StringIO(response.text))  # Load data
            if data.size % n != 0: raise ValueError("Invalid data size")
            
            return data.reshape(data.size // n, n)  # Reshape and return
        except Exception as e:
            raise RuntimeError(f"An error occurred: {e}")  # Error handling
            
    def fetch_data_from_queue(self, backward,cal,file,matching_signals,signal_array,andor_settings,fetch_queue: Queue, delta: float, andor_array: np.ndarray,file_bw=None):
        """
        Fetches data from an external source and processes it.
    
        Args:
            fetch_queue (Queue): Queue to receive notifications for fetching data.
            delta (float): Time delay between each fetch.
            n (int): Desired number of rows for reshaping the data.
    
        """
        if self.url_cal is None or self.kinser_dat is None:
            self.build_urls()
        i=0
        bw_fact = 2 if backward else 1
        while True:
            # Wait for a notification to fetch data
            item = fetch_queue.get()  # Block until there's a new item in the queue
            if item is None:  # Check for shutdown signal
                break
    
            # Wait for the specified delta time
            time.sleep(delta)
            
            if i==0:
                calib,n=self.get_cal() # get calibration
                cal.append(calib.tolist())
                sweep_signal = "Wavelength (nm)"  # Sweep signal as string
                fixed_parameters = ["Sweep Start", "Sweep End"]+matching_signals
                            #print(fixed_parameters)
                header2=f'''Sweep Signal="{sweep_signal}"
Fixed parameters="{';'.join(fixed_parameters)}"
Experiment parameters=
# Parameters (4 byte)={len(fixed_parameters)}
Experiment size (bytes)=4096
Points={n}
Channels=Counts
'''
                file.write(header2.encode())
                
                settings_str = andor_settings.apply(lambda row: f"{row['Code']}={row['Value']}",axis=1).str.cat(sep='\n') + '\n'
                file.write(settings_str.encode())
                
                andor_chan_names= calib.tolist()
                chnames = [f"wl {i}=" + str(item) for i, item in enumerate(andor_chan_names)]
                file.write(('\n'.join(chnames)+"\n").encode())
                file.write((':HEADER_END:\n').encode())
                if backward==True:
                    file_bw.write(header2.encode())
                    file_bw.write(settings_str.encode())
                    file_bw.write(('\n'.join(chnames)+"\n").encode())
                    file_bw.write((':HEADER_END:\n').encode())
                
            try:
                #print(self.kinser_dat)
                response = requests.get(self.kinser_dat)  # Fetch data
                if response.status_code != 200: raise ValueError(f"Error: {response.status_code}")
                
                data = np.loadtxt(StringIO(response.text))  # Load data
                if data.size % n != 0: raise ValueError("Invalid data size")
                
                if backward:  # Case for backward==True
                    # Split the data into two halves
                    half_size = data.size // 2
                    andor_array[2*i, :, :] = data[:half_size].reshape(half_size // n, n)  # First half
                    andor_array[2*i+1, :, :] = data[half_size:].reshape(half_size // n, n)  # Second half
                    for j in range(signal_array.shape[1]-1,-1):
                        nanonis_data_bw = np.array([float(calib[0]), float(calib[-1])] + list(signal_array[2*i, j, :]))
                        andor_data_bw = andor_array[2*i+1, j, :]
                        
                        # Convert to the correct dtype and write to file
                        nanonis_data_bw.astype(">f4").tofile(file_bw)
                        andor_data_bw.astype(">f4").tofile(file_bw)
                    
                else:  # Case for backward == False
                    andor_array[i, :, :] = data.reshape(data.size // n, n)  # Reshape and assign
                    
                for j in range(signal_array.shape[1]):
                    # Construct arrays of nanonis data (with start and end wavelength) and andor data 
                    nanonis_data = np.array([float(calib[0]), float(calib[-1])] + list(signal_array[2*i, j, :]))
                    andor_data = andor_array[bw_fact*i, j, :]
                    
                    # Convert to the correct dtype and write to file
                    nanonis_data.astype(">f4").tofile(file)
                    andor_data.astype(">f4").tofile(file)
                
            except Exception as e:
                raise RuntimeError(f"An error occurred: {e}")  # Error handling
                
            finally:
                fetch_queue.task_done()  # Mark the task as done
                i+=1
    
    def photon_map_k(self, acqtime=10, acqnum=1, pix=(10, 10), dim=None, name="LS-man", user="Jirka", signal_names=None,direction="up",backward=False,bw_ratio=10,readmode=0,wait_time=None):     
        """
 Perform a photon mapping scan for a given experimental setup.

 This function sets up the acquisition parameters, prepares the Andor camera, and processes 
 the data based on the provided configuration. It includes setting the scan dimensions, 
 preparing the scan buffer, and managing the scan direction. The data is collected, 
 reshaped, and written to a file in a structured format.

 Args:
     acqtime (float): Acquisition time per pixel (in seconds). Default is 10 seconds.
     pix (tuple): Tuple of two integers specifying the number of pixels in the x and y dimensions.
     dim (tuple, optional): Tuple specifying the dimensions of the scan (width, height in nanometers). 
                            If not provided, dimensions are determined from the scan frame.
     name (str): Name for the experiment, used in file naming. Default is "LS-man".
     user (str): Name of the user performing the experiment. Default is "Jirka".
     signal_names (list, optional): List of signal names to retrieve from the system. Default is None.=
     direction (str): Direction of the scan. Can be "up" or "down". Default is "up".
     backward (bool): Whether the scan is backward (zigzag pattern). Default is False.
     bw_ratio (float): Ratio to adjust the backward scan speed. Default is 10.
     readmode (int): Mode for the camera acquisition. Default is 0.
     wait_time (float, optional): Time in seconds to wait before starting the next acquisition. Default is None.

 Returns:
     None: The function doesn't return any value but writes the scanned data to a file.

 Raises:
     RuntimeError: If an error occurs during setup, data fetching, or file writing.
 
 Notes:
     - The function automatically adjusts settings based on the connected camera's configuration.
     - It prepares the camera for image mode (if necessary) and sets the acquisition mode.
     - The function assumes a zigzag scanning pattern, with the option for backward scanning.
     - Data is fetched from the Andor camera, processed, and saved in a 3D file format.
 """
        s_time=time.perf_counter()
        bw_fact = 2 if backward else 1
        #first try communication with andor
        try:
            self.connect2.acqtime_set(acqtime)
            settings=self.connect2.settings_get()
            andor=True
            for index, row in settings.iterrows():
                code = row['Code']
                value = int(row['Value'])
                if code == 'GRM' and (value == 4 or readmode not in [0, "FVB"]):
                    print(f"Camera in image mode!: GRM with value {value}, setting it to FVB mode.")
                    self.connect2.readmode_set(readmode)
                elif code == 'GAM' and value != 3:
                    print(f"Acq. mode with value {value} invalid, setting it to single (1) mode.")
                    self.connect2.acqmode_set(3)
                elif code == 'GKT':
                    acqtime=float(value)
    
            self.connect2.tcp.cmd_send(f"SKN {pix[0]*bw_fact}") #make it a function
            response_and = self.connect2.tcp.recv_until() 
            self.connect2.tcp.cmd_send("AQP")
            response_and = self.connect2.tcp.recv_until()
            settings=self.connect2.settings_get()
        except Exception as e:
            #raise RuntimeError(f"An error occurred: {e}")  # Error handling
            print(e)
            andor=False
            settings=[]
        print("andor",andor)
        # Retrieve scan frame and set default dimensions if not provided
        SF = self.connect.ScanFrameGet()
        dim = (1e9 * SF.values[2][0], 1e9 * SF.values[3][0]) if dim is None else dim
        cx, cy, angle = SF.values[0][0], SF.values[1][0], SF.values[4][0]

        if signal_names is None:
            signal_names = [
                "Bias (V)", "Z (m)", "Current (A)", 
                "LI Demod 1 Y (A)", "LI Demod 2 Y (A)", "Counter 1 (Hz)"
            ]
        # Filter rows in `signals_slots` where the first column matches any item in `signal_names`

        if self.connect.version<13000:
            signals_slots=self.connect.SignalsInSlotsGet(prt=False)
            signals =signals_slots.iloc[:, 0].tolist()
        else:
            signals =self.signal_names.iloc[:, 0].tolist()
        # Get matching indices and signals in the order based on signals
        matching_indices = [i for i, signal in enumerate(signals) if signal in signal_names]
        matching_signals = [signals[i] for i in matching_indices]
        
        # Get unmatched items
        unmatched_items = [item for item in signal_names if item not in signals]
        print(f'Signals {unmatched_items} not in slots')
        # define the channels recorded
        scan_buffer=self.connect.ScanBufferGet()
        num_chs, ch_idx, pxs, lines = scan_buffer.iloc[:, 0]
        self.connect.ScanBufferSet(len(matching_indices),matching_indices,pxs, lines) 
             
        # Initial setup for the first point in the bottom-left corner
        initial_dx_nm, initial_dy_nm = (-dim[0] / 2, -dim[1] / 2 + (dim[1] / pix[1])/2) if direction == "up" else (-dim[0] / 2, dim[1] / 2 - (dim[1] / pix[1])/2)
        initial_dx_rot, initial_dy_rot = self.rotate(initial_dx_nm, initial_dy_nm, angle)
        
                
        #prepare Andor, set acqtime anf get acqtime, set it for range(2 * pix[1]) or range pix[1] accumulations
        
        # Move to the first point
        self.connect.FolMeXYPosSet(cx + initial_dx_rot, cy + initial_dy_rot,True)
        # set bw and fw move speed 
        mv_spd=1e-9*(dim[0]/(acqtime*pix[0]))
        bw_ratio=(min(bw_ratio, 100*1e-9/mv_spd)) #limit bw_spped to max 100 nm/s
        # define the Tip Rec parameters 
        pppix, ovs, num_points = self.calculate_parameters(acqtime, pix, backward)
        self.connect.FolMeOversamplSet(ovs,prt=False)
        self.connect.TipRecBufferSizeSet(num_points,prt=False)
        self.connect.TipRecBufferClear()
        self.connect.FolMeSpeedSet(mv_spd,1)
        
        # Set scan direction parameters
        row_range = range(2 * pix[1])
        dy_sign = -1 if direction == "up" else 1
        dy_plus=0
        
        # Create the signal_array with NaN values
        signal_array=np.full((len(row_range),pix[0],len(matching_indices)), np.nan, dtype=np.float32)

        # Start zigzag pattern)
        g1_time,g2_time=0,0

        
        # .3ds header creation
        folder = self.connect.UtilSessionPathGet().loc['Session path', 0]
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
        if backward==True:
           # filename_3ds_bw = self.connect.get_next_filename("G"+name+'_bw', extension='.3ds', folder=folder)
            filename_3ds_bw = filename_3ds.replace(name, f"{name}_bw")
            f_bw = open(filename_3ds_bw, 'wb')
            f_bw.write(header.encode())

        f = open(filename_3ds, 'wb')
        f.write(header.encode())
        if andor:
            cal=[]
            andor_array = np.full((bw_fact*pix[1], pix[0], 1024), np.nan, dtype=np.float32)
            #start andor queue for data downloading and processing
            fetch_queue = Queue()  # Create a queue for URLs to fetch
            delta = 0.05  # Set your desired delay time
            if backward==True:
                fetch_thread = threading.Thread(target=self.fetch_data_from_queue, args=(backward,cal,f,matching_signals,signal_array,settings,fetch_queue, delta,andor_array,f_bw))
            else:
                fetch_thread = threading.Thread(target=self.fetch_data_from_queue, args=(backward,cal,f,matching_signals,signal_array,settings,fetch_queue, delta,andor_array))
            fetch_thread.start()  # Start the fetch thread
        

        try:
            counter=0
            for index, row in enumerate(row_range):                    
            # You can now use 'index' to access signal_array
                # Calculate y-coordinate for the current row
                if row % 2 == 0 and row!=0:
                    dy_plus+=(dim[1] / pix[1]) * (-dy_sign)
                    
                dy_nm=initial_dy_nm+dy_plus
            
                # Determine end x-coordinate based on row direction
                end_dx_nm = dim[0] / 2 if row % 2 == 0 else -dim[0] / 2
            
                # Rotate the end coordinates and move to the end point of the current row
                end_dx_rot, end_dy_rot = self.rotate(end_dx_nm, dy_nm, angle)
    
                # Start move and wait for completion
                
                if backward==False:
                    if andor==True and row ==0:   #for row 0 start the acquisition; starts andor kinetic series acquisition and waits for TTL pulse from andor
                        andor_thread=self.andor_thread_open(wait_time=wait_time)

                    if row % 2 == 0 and row!=0: 
                        dx_n_rot,dy_n_rot=self.rotate(-end_dx_nm,dy_nm, angle)
                        self.connect.FolMeXYPosSet(cx + dx_n_rot, cy + dy_n_rot, True) # move one pixel up after finishing fw bw line
                        self.connect.FolMeSpeedSet(mv_spd,1)
                        self.connect.TipRecBufferClear()
                        
                        if andor:
                            andor_thread.join()  # for all beginnings of new lines, finish the old one and start the new acquisition
                            fetch_queue.put("start")
                            andor_thread=self.andor_thread_open(wait_time=wait_time)
                        
                    self.connect.FolMeXYPosSet(cx + end_dx_rot, cy + end_dy_rot, True)  #regular scan move
        
                    if row % 2 == 0:
                        _,data=self.connect.TipRecDataGet()   # save data containing only fw 
                        temp_data = self.bin_average_stacked(data,pix[0]) # analyse to fw and bw movement and make avarage of pixels
                        signal_array[index,:,:], signal_array[index + 1,:,:] = temp_data, temp_data[::-1,:] # write first fw an later fw reversed
                        self.connect.FolMeSpeedSet(bw_ratio*mv_spd,1)
                        
                    if index == len(row_range) - 1 and andor==True: #Terminate in last row
                        andor_thread.join()
                        fetch_queue.put("start")
                    
                else:
                    
                    if andor==True and row ==0:   #for row 0 start the acquisition; starts andor kinetic series acquisition and waits for TTL pulse from andor
                        andor_thread=self.andor_thread_open(wait_time=wait_time)
                            
                    if row % 2 == 0 and row!=0:
                        dx_n_rot,dy_n_rot=self.rotate(-end_dx_nm,dy_nm, angle)
                        self.connect.FolMeXYPosSet(cx + dx_n_rot, cy + dy_n_rot, True) # move one pixel up after finishing fw bw line
                        self.connect.TipRecBufferClear()
                        if andor:
                            andor_thread.join()  # for all beginnings of new lines, finish the old one and start the new acquisition
                            fetch_queue.put("start")
                            andor_thread=self.andor_thread_open(wait_time=wait_time)
                        
                    self.connect.FolMeXYPosSet(cx + end_dx_rot, cy + end_dy_rot, True) #regular scan move
        
                    if row % 2 == 1:
                        _,data=self.connect.TipRecDataGet()   # save data containing fw and bw
                        temp_data = self.bin_average_stacked(data,2 * pix[0]) # analyse to fw and bw movement and make avarage of pixels
                    #    print(data.shape,temp_data[pix[0]:, :].shape,index-1,index)
                        signal_array[index - 1,:,:], signal_array[index,:,:] = temp_data[:pix[0], :], temp_data[pix[0]:, :] # write first fw an later bw
                        
                    if index == len(row_range) - 1 and andor==True: #Terminate in last row
                        andor_thread.join()
                        fetch_queue.put("start")
                counter+=pix[0]        
                elapsed = time.perf_counter() - s_time
                remaining = (elapsed / counter) * (math.prod(pix)*2 - counter)
                sys.stdout.write(f"\rExecuted {int(counter*bw_fact/2)}/{math.prod(pix)*bw_fact} | Remaining: {int(remaining // 60):02d}:{int(remaining % 60):02d}") #(column+1+pix[0]*row)*
                sys.stdout.flush()
             #   print("loop time", "{:.5f}".format((time.perf_counter()-sl_time-g2_time+g1_time)),"theoretic loop time","{:.5f}".format(acqtime*pix[0]))
                      
        except KeyboardInterrupt:
            pass
            
        finally:
            if andor:
                # Ensure all remaining items in the queue are processed
                fetch_queue.join()  # Wait until all items in the queue have been processed
                fetch_queue.put(None)  # Send a shutdown signal to the fetch thread
                fetch_thread.join()  # Wait for the fetch thread to finish
                self.connect2.acqmode_set(1)
              #  print("Shutdown complete.")
            
            end_time_scan = time.perf_counter()
            elapsed_time_scan="{:.1f}".format(end_time_scan-s_time)
            
            end_time = datetime.now().strftime('%d.%m.%Y %H:%M:%S.%f')[:-3]
            end_time_line = f'End time="{end_time}" \n'
            end_time_line = end_time_line.ljust(len(header.splitlines()[0]) + 1)
            
            # Seek to the beginning and overwrite the first line
            f.seek(0)
            f.write(end_time_line.encode())
            f.close()
            if backward==True:
                # Seek to the beginning and overwrite the first line
                f_bw.seek(0)
                f_bw.write(end_time_line.encode())
                f_bw.close()
                f_bw.close()
            filename_sxm = self.connect.get_next_filename("M"+name,extension='.sxm',folder=folder)
           ### settings_dict=(dict(zip(settings.T.iloc[0], settings.T.iloc[1].astype(str))))
            scan_par = {
                "REC_DATE": datetime.now().strftime('%d.%m.%Y'),
                "REC_TIME":  datetime.now().strftime('%H:%M:%S'),
                "ACQ_TIME": str(elapsed_time_scan),
                "SCAN_PIXELS": f"{pix[0]}\t{pix[1]}",
                "SCAN_FILE": filename_sxm,
                "SCAN_TIME": f"{acqtime*pix[0]:.6E}\t{acqtime*pix[0]:.6E}",
                "SCAN_RANGE": f"{1e-9 * dim[0]:.6E}\t{1e-9 * dim[1]:.6E}",
                "SCAN_OFFSET": f"{cx:.6E}\t{cy:.6E}",
                "SCAN_ANGLE": str(angle),
                "SCAN_DIR": direction,
                "BIAS": str(bias_voltage)
                }   
            andor_chan_names= []
            settings_dict={}
            nanonis_chan_names=matching_signals
            if andor:
                #print(cal,"calibration")
                andor_chan_names=[item for sublist in cal for item in sublist]
                
            
            #andor_data_to_sxm=np.array(data_ar).T
            #combined_data = np.vstack((nanonis_data_to_sxm, andor_data_to_sxm))
            if andor:
                if backward==True:
                    combined_data =np.vstack((signal_array.transpose(2, 0, 1).reshape(signal_array.shape[2], -1),andor_array.transpose(2, 0, 1).reshape(andor_array.shape[2], -1)))
                else:
                    print(andor_array.shape,signal_array.shape)
                    combined_data =np.vstack((signal_array[::2,:,:].transpose(2, 0, 1).reshape(signal_array.shape[2], -1),andor_array.transpose(2, 0, 1).reshape(andor_array.shape[2], -1)))
            else:
                if backward==True:
                    combined_data =signal_array.transpose(2, 0, 1).reshape(signal_array.shape[2], -1)
                else:
                    combined_data =signal_array[::2,:,:].transpose(2, 0, 1).reshape(signal_array.shape[2], -1)
            
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
            #print(andor_chan_names)
            #print(cal,"cal")
            #print(nanonis_chan_names)
            for i, name in enumerate(andor_chan_names, start=128):
                final_list.append([i, name, 'nm', scan_dir, default_value1, default_value2])

            data_sxm=self.connect.writesxm(backward,filename_sxm, settings_dict, scan_par, final_list, combined_data)
            
            

        
        # reset speed and recorded channels in scan window to the original value before the map acquisition
        self.connect.FolMeSpeedSet(mv_spd,0)
        self.connect.ScanBufferSet(*scan_buffer.iloc[:, 0].values)
        #return(np.stack(signal_array,axis=0))
        e_time=time.perf_counter()
        print("tot time", "{:.5f}".format(e_time-s_time))
        if andor:
            return(signal_array,andor_array)
        else:
            return(signal_array)
    
    def fw_bw_map_chatGPT(self, acqtime=10, acqnum=1, pix=(10, 10), dim=None, name="LS-man", user="Jirka", signal_names=None, savedat=False, direction="up", backward=False, readmode=0):       
        def bin_average_stacked(array, n):
            """Average rows of an array in bins of size n."""
            bin_size = array.shape[0] // n
            return np.array([np.mean(array[i * bin_size:(i + 1) * bin_size], axis=0) for i in range(n)])
        
        # Retrieve scan frame and set default dimensions if not provided
        SF = self.connect.ScanFrameGet()
        dim = (1e9 * SF.values[2][0], 1e9 * SF.values[3][0]) if dim is None else dim
        cx, cy, angle = SF.values[0][0], SF.values[1][0], SF.values[4][0]
        
        # Get available signal slots and set signal names
        signals_slots = self.connect.SignalsInSlotsGet(prt=False)
        if signal_names is None:
            signal_names = [
                "Bias (V)", "X (m)", "Y (m)", "Z (m)", "Current (A)", 
                "LI Demod 1 Y (A)", "LI Demod 2 Y (A)", "Counter 1 (Hz)"
            ]
        
        # Find matching indices of required signals
        signals = signals_slots.iloc[:, 0].tolist()
        matching_indices = [i for i, signal in enumerate(signals) if signal in signal_names]
    
        # Set scan buffer and initialize settings
        scan_buffer = self.connect.ScanBufferGet()
        self.connect.ScanBufferSet(len(matching_indices), matching_indices, scan_buffer.iloc[:, 2], scan_buffer.iloc[:, 3]) 
                  
        # Initial setup for movement
        initial_dy_nm = -dim[1] / 2 + (dim[1] / pix[1]) / 2 if direction == "up" else dim[1] / 2 - (dim[1] / pix[1]) / 2
        initial_dx_rot, initial_dy_rot = self.rotate(-dim[0] / 2, initial_dy_nm, angle)
        self.connect.FolMeXYPosSet(cx + initial_dx_rot, cy + initial_dy_rot, True)
        
        # Movement and acquisition setup
        mv_spd = 1e-9 * (np.sqrt(dim[0]**2 + (dim[1] / pix[1] / 2)**2) / (acqtime * pix[0]))
        ovs = max(1, min(2E4 / (20 / acqtime), 1000))
        num_points = int(2E4 / ovs * acqtime * pix[0] + 200)
        self.connect.FolMeOversamplSet(ovs, prt=False)
        self.connect.TipRecBufferSizeSet(num_points, prt=False)
        self.connect.TipRecBufferClear()
        self.connect.FolMeSpeedSet(mv_spd, 1)
        
        # Initialize signal array for data storage
        signal_array = np.full((2 * pix[1], pix[1], len(matching_indices)), np.nan, dtype=np.float32)
        
        # Zigzag scanning
        s_time = time.perf_counter()
        dy_sign = -1 if direction == "up" else 1
        dy_plus = 0
        for index, row in enumerate(range(2 * pix[1])):
            # Update y-coordinate for new rows
            if row % 2 == 0 and row != 0:
                dy_plus += (dim[1] / pix[1]) * -dy_sign
            
            dy_nm = initial_dy_nm + dy_plus
            end_dx_nm = dim[0] / 2 if row % 2 == 0 else -dim[0] / 2
            end_dx_rot, end_dy_rot = self.rotate(end_dx_nm, dy_nm, angle)
            
            # Move to new row and record data
            if row % 2 == 0 and row != 0:
                dx_n_rot, dy_n_rot = self.rotate(-end_dx_nm, dy_nm, angle)
                self.connect.FolMeXYPosSet(cx + dx_n_rot, cy + dy_n_rot, True)
                self.connect.TipRecBufferClear()
            
            self.connect.FolMeXYPosSet(cx + end_dx_rot, cy + end_dy_rot, True)
            if row % 2 == 1:
                _, data = self.connect.TipRecDataGet()
                temp_data = bin_average_stacked(np.transpose(data), 2 * pix[1])
                signal_array[:,index - 1,:], signal_array[index] = temp_data[:pix[1], :], temp_data[pix[1]:, :][::-1, :]
        
        e_time = time.perf_counter()
        print("Total time:", "{:.5f}".format(e_time - s_time))
        
        # Restore original settings and return data
        self.connect.FolMeSpeedSet(mv_spd, 0)
        self.connect.ScanBufferSet(*scan_buffer.iloc[:, 0].values)
        return signal_array
    
    def zig_zag_move(self, acqtime=10, acqnum=1, pix=(10, 10), dim=None, name="LS-man", user="Jirka", signal_names=None,savedat=False,direction="up",backward=False,readmode=0):       
       
        # Retrieve scan frame and set default dimensions if not provided
        SF = self.connect.ScanFrameGet()
        dim = (1e9 * SF.values[2][0], 1e9 * SF.values[3][0]) if dim is None else dim
        cx, cy, angle = SF.values[0][0], SF.values[1][0], SF.values[4][0]
        
        # Initial setup for the first point in the bottom-left corner
        initial_dx_nm, initial_dy_nm = (-dim[0] / 2, -dim[1] / 2) if direction == "up" else (-dim[0] / 2, dim[1] / 2)
        initial_dx_rot, initial_dy_rot = self.rotate(initial_dx_nm, initial_dy_nm, angle)
        
        # Move to the first point
        #print(f"Moving to initial start point (dx_nm: {initial_dx_nm}, dy_nm: {initial_dy_nm})")
        self.connect.FolMeXYPosSet(cx + initial_dx_rot, cy + initial_dy_rot,True)
        
        # Set scan direction parameters
        row_range = range(2 * pix[1])# if direction == "up" else range(2 * pix[1] - 1, -1, -1)
        dy_sign = -1 if direction == "up" else 1
        
        # Start zigzag pattern
        for row in row_range:
            # Calculate y-coordinate for the current row
            dy_nm = (dy_sign * dim[1] / 2) + (row + 1) * (dim[1] / pix[1]) / 2 * (-dy_sign)
        
            # Determine end x-coordinate based on row direction
            end_dx_nm = dim[0] / 2 if row % 2 == 0 else -dim[0] / 2
        
            # Rotate the end coordinates and move to the end point of the current row
            end_dx_rot, end_dy_rot = self.rotate(end_dx_nm, dy_nm, angle)
           # print(f"{direction.capitalize()}ward scan - Moving to end point (dx_nm: {end_dx_nm}, dy_nm: {dy_nm})")
            self.connect.FolMeXYPosSet(cx + end_dx_rot, cy + end_dy_rot, True)
      
    def zig_zag_move_old(self, acqtime=10, acqnum=1, pix=(10, 10), dim=None, name="LS-man", user="Jirka", signal_names=None,savedat=False,direction="up",backward=False,readmode=0):       
    
        SF=self.connect.ScanFrameGet()  # Retrieve scan frame
        if dim is None:
            dim = (1e9 * SF.values[2][0], 1e9 * SF.values[3][0])  # Set dimensions if not provided
        cx, cy, angle = SF.values[0][0], SF.values[1][0], SF.values[4][0]  # Extract center and angle
        wait_num = True  # Wait flag
        
        # Initial setup for the first point in the bottom-left corner
        initial_dx_nm = - dim[0]/2
        initial_dy_nm = - dim[1]/2
        initial_dx_rot, initial_dy_rot = self.rotate(initial_dx_nm, initial_dy_nm, angle)
        
        # Move to the first point
        print(f"Moving to initial start point (dx_nm: {initial_dx_nm}, dy_nm: {initial_dy_nm})")
        self.connect.FolMeXYPosSet(cx + initial_dx_rot, cy + initial_dy_rot, wait_num)
        
        # Start zigzag pattern
        if direction == "up":
            for row in range(2*pix[1]):
                # Calculate y-coordinate for the current row
                dy_nm = - dim[1]/2 + (row+1) * (dim[1] / pix[1]) / 2
        
                # Determine end x-coordinate based on row direction
                if row % 2 == 0:  # Left-to-right for even rows
                    end_dx_nm = + dim[0]/2
                else:  # Right-to-left for odd rows
                    end_dx_nm = - dim[0]/2
        
                # Rotate the end coordinates
                end_dx_rot, end_dy_rot = self.rotate(end_dx_nm, dy_nm, angle)
        
                # Move to end point of the current row
                print(f"Upward scan - Moving to end point (dx_nm: {end_dx_nm}, dy_nm: {dy_nm})")
                self.connect.FolMeXYPosSet(cx + end_dx_rot, cy + end_dy_rot, wait_num)
    
        elif direction == "down":
            for row in range(2*pix[1] - 1, -1, -1):
                # Calculate y-coordinate for the current row
                dy_nm = + dim[1]/2 - (row+1) * (dim[1] / pix[1]) / 2
        
                # Determine end x-coordinate based on row direction
                if row % 2 == 0:  # Left-to-right for even rows
                    end_dx_nm = + dim[0]/2
                else:  # Right-to-left for odd rows
                    end_dx_nm = - dim[0]/2
        
                # Rotate the end coordinates
                end_dx_rot, end_dy_rot = self.rotate(end_dx_nm, dy_nm, angle)
        
                # Move to end point of the current row
                print(f"Downward scan - Moving to end point (dx_nm: {end_dx_nm}, dy_nm: {dy_nm})")
                self.connect.FolMeXYPosSet(cx + end_dx_rot, cy + end_dy_rot, wait_num) 

                
    def nanonis_map(self, acqtime=10, acqnum=1, pix=(10, 10), dim=None, name="LS-man", user="Jirka", signal_names=None,savedat=False,direction="up"):
        # Initialize variables

        if direction in ["up", True, 0]:
            direction = "up"
        elif direction in ["down", False, 1]:
            direction = "down"
        else:
            raise ValueError("Invalid direction. Use 'up', 'down', True, False, 0, or 1.")
            
        start_time_scan = time.perf_counter()
        folder = self.connect.UtilSessionPathGet().loc['Session path', 0]
        signal_names_df = self.signal_names
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
                "SCAN_TIME": f"{acqtime*pix[0]:.6E}\t{acqtime*pix[0]:.6E}",
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
            print("shape",nanonis_data.shape)
    
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
                   # 'BiasSpectrTTLSync': self.connect.BiasSpectrTTLSyncGet(), #doesnt work with version in lab
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
        with open(fdir + '/' + fname + '.par', 'wb') as handle:
            pickle.dump(bias_par, handle)
        print(f'".par" file created in {fdir}')

    def bias_spectr_par_load(self, fdir, fname):
        with open(fdir + '/' + fname, 'rb') as handle:
            bias_par = pickle.load(handle)
        for key in bias_par:
            if key in ["BiasSpectrMLSLockinPerSeg",'BiasSpectrMore','LockInOnOff1']:
                try:
                    bias_par[key] = bias_par[key].replace(['No change', 'Yes/On', 'No/Off', 'False/Off', 'True/On'], [0, 1, 2, 0, 1])
                except:
                    pass
            else:
                try:
                    bias_par[key] = bias_par[key].replace(['No change', 'Yes/On', 'No/Off', 'False/Off', 'True/On'], [0, 1, 2, 2, 1])
                except:
                    pass
            
            # Ensure `bias_par[key]` is a pandas Series or DataFrame before using infer_objects
            if isinstance(bias_par[key], (pd.Series, pd.DataFrame)):
                bias_par[key] = bias_par[key].infer_objects(copy=False)
        return bias_par

    def bias_spectr(self, par, data_folder, basename = '%Y%m%d_', run = True):
        self.connect.BiasSpectrOpen()
        props = (int(par['BiasSpectrProps'].loc['Save all', 0]),
                 int(par['BiasSpectrProps'].loc['Number of sweeps',0]),
                 par['BiasSpectrProps'].loc['Backward sweep', 0],
                 int(par['BiasSpectrProps'].loc['Number of points',0]),
                 float(par['BiasSpectrTiming'].loc['Z offset (m)',0]),
                 par['BiasSpectrMore'].loc['Auto save', 0],
                 par['BiasSpectrMore'].loc['Save dialog', 0])
        self.connect.BiasSet(*par['Bias'].values)
        self.connect.BiasSpectrChsSet(*par['BiasSpectrChs'].values.tolist())
        self.connect.BiasSpectrPropsSet(*props)
        self.connect.BiasSpectrAdvPropsSet(*par['BiasSpectrAdvProps'].values)
        self.connect.BiasSpectrLimitsSet(*par['BiasSpectrLimits'].values)
        self.connect.BiasSpectrTimingSet(*par['BiasSpectrTiming'].values)
       # self.connect.BiasSpectrTTLSyncSet(*par['BiasSpectrTTLSync'].values)
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
