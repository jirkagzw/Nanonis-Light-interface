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
            raise ValueError('The minimum allowed wavelength is 0 (zeroth order reflection) 10V. Please check your input! Center wavelength has been set to zero')    
        
        body="{:.5f}".format(wl)
        header = 'SWL '
        cmd = header + body + '\n'#self.tcp.termination_char

        self.tcp.cmd_send(cmd)
        result= self.tcp.recv_until()

     #   self.tcp.print_err(res_err)
      #  bias_df = pd.DataFrame({'Bias (V)': bias}, index=[0]).T
        
        if prt: 
            print('\n' + result)
        return result 
    