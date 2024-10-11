# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 14:54:20 2024

@author: jirka
"""

import pickle
import pandas as pd
from os import mkdir
from os.path import exists
import time
import numpy as np

class andor_meas:
    # Class variables
    # To change the value of class variable in your script, use this: 
    #   import andor_measnanonis_tcp as tcp
    #   tcp.andor_meas.if_print = True
    if_print = False
    def __init__(self, tcp):
        self.tcp = tcp
        # self.f_print = False
        
        return
    
    def wl_set(self, wl, prt=if_print):
        """
        Sets the center wavelength of a current grating on spectrograph to the specified value..

        Parameters:
            wavelength (float): The center wavelength to set (in nm). 0 means zeroth order reflection.
            prt (bool): Whether to print the output (default is `if_print`).

        Raises:
            ValueError: If the wavelength is below zero .

        Returns:
            response from Andor
        """
        if wl< 0:
            body="{:.5f}".format(0)
            raise ValueError('The minimum allowed wavelength is 0 (zeroth order reflection). Please check your input! Center wavelength has been set to zero')    
        
        body="{:.5f}".format(wl)
        header = 'SWL '
        cmd = header + body + self.tcp.termination_char

        self.tcp.cmd_send(cmd)
        result= self.tcp.recv_until()

     #   self.tcp.print_err(res_err)
      #  bias_df = pd.DataFrame({'Bias (V)': bias}, index=[0]).T
        
        if prt: 
            print('\n' + result)
        return result 
    
    def grating_set(self, number, prt=if_print):
        """
        Sets the grating on spectrograph to the specified value.. 

        Parameters:
            number (float): The grating number (1 to 4)
            Gr n.1 150 grooves/mm. (500nm blaze) 500 nm range 2.5 nm resolution
            Gr n.2 600 grooves/mm. (500nm blaze) 125 nm range 1 nm resolution
            Gr n.3 600 grooves/mm. (1000nm blaze) 125 nm range 1 nm resolution
            Gr n.4 1200 grooves/mm. (500nm blaze) 60 nm range 0.3 nm resolution

            prt (bool): Whether to print the output (default is `if_print`).

        Raises:
            ValueError: If the wavelength is out of range 

        Returns:
            response from Andor
        """
        if number<1 and number>4:
            raise ValueError('Only numbers from 1 to 4 are allowed')    
        else:
            
            body="{:.0f}".format(number)
            header = 'SGR '
            cmd = header + body +self.tcp.termination_char
    
            self.tcp.cmd_send(cmd)
            result= self.tcp.recv_until()
        
            if prt: 
                print('\n' + result)
            return result 
    
    def acqtime_set(self, acqtime, prt=if_print):
        """
        Sets the acquisition time of the camera to the specified value in seconds. 

        Raises:
            ValueError: If the time is out of range 

        Returns:
            response from Andor
        """
        if acqtime<=0 and acqtime>300:
            body="{:.0f}".format(1)
            raise ValueError('Only >0 and < 300 s are allowed')    

        body="{:.4f}".format(acqtime)
        header = 'SET '
        cmd = header + body +self.tcp.termination_char

        self.tcp.cmd_send(cmd)
        result= self.tcp.recv_until()
        
        if prt: 
            print('\n' + result)
        return result 
    
    def acqnum_set(self, acqnum, prt=if_print):
        """
        Sets the number of acquisitions - always use 1. 

        Raises:
            ValueError: If the time is out of range 

        Returns:
            response from Andor
        """
        if acqnum<=0 and acqnum>10:
            body="{:.0f}".format(1)
            raise ValueError('Only >0 and < 10 are allowed')    

        body="{:.0f}".format(acqnum)
        header = 'SAN '
        cmd = header + body +self.tcp.termination_char

        self.tcp.cmd_send(cmd)
        result= self.tcp.recv_until()
        
        if prt: 
            print('\n' + result)
        return result 
    
    def acqmode_set(self, mode, prt=if_print):
        """
        Sets the mode of acquisition - 0  or "FVB" or True means full vertical binning,  0  or "FVB" True means full vertical binning
                                       4  or "IMG" or False 2d image mode,  recommended frequency 1MHz or 2MHz

        Raises:
            ValueError: If the time is out of range 

        Returns:
            response from Andor
        """
        
        if mode in ["FVB", 0,True]:
            mode = 0
        elif mode in ["IMG", 4,False]:
            mode = 4
        else:
            raise ValueError("Invalid mode. Use 'FVB', 'IMG', 0 or 4, True or False") 

        body="{:.0f}".format(mode)
        header = 'SRM '
        cmd = header + body +self.tcp.termination_char

        self.tcp.cmd_send(cmd)
        result= self.tcp.recv_until()
        
        if prt: 
            print('\n' + result)
        return result
    
    def acqfreq_set(self, freq, prt=if_print):
        """
        Sets the freq of acquisition in kHz - 0  or "2000" means 2 MHz
                                           1 or "1000" means 
                                           

        Raises:
            ValueError: If the time is out of range 

        Returns:
            response from Andor
        """
        
        if freq in ["50", 2, 50]:
            freq = 2
        elif freq in ["1000", 1, 1000]:
            freq = 1
        elif freq in ["2000", 0,2000]:
            freq = 0
        else:
            raise ValueError("Invalid freq. Use '50', '1000', '2000' or  2, 1, 0 or 50, 1000, 2000") 

        body="{:.0f}".format(freq)
        header = 'SHS '
        cmd = header + body +self.tcp.termination_char

        self.tcp.cmd_send(cmd)
        result= self.tcp.recv_until()
        
        if prt: 
            print('\n' + result)
        return result
        
    
    def acquisition_set(self, prt=if_print):
        """
        Starts the acquisition with the preselected parameters and returns the data as a DataFrame with two columns.
    
        Raises:
            ValueError: If the response "OK AQD" is not received.
        
        Returns:
            pandas.DataFrame: A DataFrame with columns:
                'Wavelength (nm)' (float) and 'Counts' (int).
        """
        header = 'AQD '
        body = ""
        cmd = header + body + self.tcp.termination_char
    
        self.tcp.cmd_send(cmd)
        result = self.tcp.recv_until()
    
        elements = result.split()
        response = " ".join(elements[:2])
        expected_string = "OK AQD"
        
        if response != expected_string:
            raise ValueError(f"Error: Expected '{expected_string}', but got '{response}'.")
    
        ar_length = int(elements[2])
        column_1 = elements[3:(3 + ar_length)]
        column_2 = elements[3 + ar_length:]
    
        # Convert column_1 to floats and column_2 to integers
        column_1 = [float(x) for x in column_1]
        column_2 = [int(float(x)) for x in column_2]  # Convert to float first, then to int
    
        # Create a DataFrame with two columns
        df = pd.DataFrame({
            'Wavelength (nm)': column_1,
            'Counts': column_2
        })
    
        if prt:
            print('\n' + response)
        
        return df

    def settings_get(self, prt=if_print):
        """
        Gets the settings of the spectrograph.
    
        Raises:
            ValueError: If the response "OK GST" is not received.
        
        Returns:
            pandas.DataFrame: A DataFrame with columns:
                '3 DIGIT code' (STRING) and 'Value' (float).
        """
        header = 'GST '
        body = ""
        cmd = header + body + self.tcp.termination_char
    
        self.tcp.cmd_send(cmd)
        result = self.tcp.recv_until()
    
        elements = result.split()
        response = " ".join(elements[:2])
        expected_string = "OK GST"
        
        if response != expected_string:
            raise ValueError(f"Error: Expected '{expected_string}', but got '{response}'.")
    
        column_1 = elements[1::2]
        column_2 = elements[2::2]
    
        # Convert column_1 to floats and column_2 to integers
        column_2 = [float(x) for x in column_2]  # Convert to float first, then to int
    
        # Create a DataFrame with two columns
        df = pd.DataFrame({
            'Code': column_1,
            'Value': column_2
        })
    
        if prt:
            print('\n' + response)
        
        return df
