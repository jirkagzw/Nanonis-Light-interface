# -*- encoding: utf-8 -*-
'''
@Time    :   2023/03/04 01:54:34
@Author  :   Shixuan Shan 
'''
import pandas as pd
import numpy as np
import os
import re

class nanonis_ctrl:
    # Class variables
    # To change the value of class variable in your script, use this: 
    #   import nanonis_esr_tcp as tcp
    #   tcp.nanonis_ctrl.if_print = True
    if_print = False
    # Functions
    def __init__(self, tcp,PLL_modulator_index=1):
        self.tcp = tcp
        # self.f_print = False
        self.mod_index=PLL_modulator_index

# it is recommended to construct body first so that you don't need to calculate the body size by yourself
# SI units are used in this module
######################################## Bias Module #############################################




    def get_next_filename(self,base_name, extension=".dat", folder="."):
        # Pattern to match filenames with number
        pattern = re.compile(rf'{re.escape(base_name)}(\d{{5}})\{extension}')
        
        # Get all files in the directory with the relevant pattern
        numbers = [
            int(match.group(1)) 
            for file in os.listdir(folder)
            if (match := pattern.match(file)) is not None
        ]
        
        # Determine the next number (starting from 1 if no files exist)
        next_number = max(numbers, default=0) + 1
        # Format number with leading zeros
        filename_number = f'{next_number:05}'
        
        # Construct new filename
        new_filename = f'{base_name}{filename_number}{extension}'
        return os.path.join(folder,new_filename)
    
    def writesxm(self,backward,pathname, header, scan_par, channels, data2d):
        """
        Writes a Nanonis file from map data. The file is saved with the given filename at the specified path.
        
        Parameters:
        - path (str): Directory path where the file will be saved.
        - filename (str): Name of the file to be created.
        - header (dict): Dictionary containing header information to be written to the file.
        - scan_par (dict): Dictionary containing scan parameters to be written to the file.
        - channels (list of lists): 2D list representing channel data to be written to the file.
        - *argv: Additional numpy arrays to be written to the file.
        """
        scan_pixels_str = scan_par["SCAN_PIXELS"]
        pixels = tuple(map(int, scan_pixels_str.split("\t")))  # Converts "10\t10" to (10, 10)

        # Set the direction flag
        direction = True if scan_par["SCAN_DIR"] == "down" else False
        
        # Open the file in binary write mode
        fn = open(pathname, mode='wb')
        
        # Write the Nanonis version information
        fn.write((":NANONIS_VERSION:\n").encode('utf-8'))
        fn.write(("2\n").encode('utf-8'))  # Assuming version 2 is being used
        
        # Write the scan type information
        fn.write((":SCANIT_TYPE:\n").encode('utf-8'))
        fn.write(("\tFLOAT\tMSBFIRST\n").encode('utf-8'))
        
        # Write the scan parameters
        for key in scan_par:
            print(key)  # Print the current key to the console (for debugging or logging)
            fn.write((":" + key + ":\n").encode('utf-8'))
            fn.write((scan_par[key] + "\n").encode('utf-8'))
        
        # Write the header information
        for key in header:
            fn.write((":" + key + ":\n").encode('utf-8'))
            fn.write((header[key] + "\n").encode('utf-8'))
        
        # Write data information header
        fn.write((":DATA_INFO:\n").encode('utf-8'))
        fn.write(("\tChannel\tName\tUnit\tDirection\tCalibration\tOffset\n").encode('utf-8'))
        
        C = data2d.shape[0]  # Number of channels (C)
        L = data2d.shape[1]  # Number of elements (L), which should be equal to pix_tuple[0] * pix_tuple[1]

        # Reshape the data to (C x pix_tuple[0] x pix_tuple[1])
        if backward==False:
            data = np.reshape(data2d, (C, pixels[1], pixels[0]))
        else:
            data = np.reshape(data2d, (C, pixels[1],2*pixels[0]))
            array1 = data[:,:,:pixels[0]]
            array2 = data[:,:,pixels[0]:]
            data = np.stack([array1,array2], axis=-1)
        # Write the channel data
        for j in range(len(channels)):
            for i in range(len(channels[0])):
                fn.write(("\t").encode('utf-8'))
                fn.write((str(channels[j][i])).encode('utf-8'))
            fn.write(("\n").encode('utf-8'))   
        
        # End of the data section
        fn.write(("\n").encode('utf-8'))
        fn.write((":SCANIT_END:\n\n\n").encode('utf-8'))
        
        # Write control characters
        fn.write(bytes([26]))  # ASCII control character for 'substitute' (often used as EOF marker)
        fn.write(bytes([4]))   # ASCII control character for 'end of transmission'
        
        # Write additional data arrays
    
        for i in range(0,len(data)):
           # if direction==False:
            if backward==False:
                data[i, :, :].astype(">f4").tofile(fn)  # Convert array to big-endian 32-bit float and write to file
                data[i, :, ::-1].astype(">f4").tofile(fn)
            else:
                data[i, :, :,0].astype(">f4").tofile(fn)  # Convert array to big-endian 32-bit float and write to file
                data[i, :, :,1].astype(">f4").tofile(fn)
           # else:
            #    data[i, ::-1, :].astype(">f4").tofile(fn)  # Convert array to big-endian 32-bit float and write to file
             #   data[i, ::-1, ::-1].astype(">f4").tofile(fn)
        
        # Close the file
        fn.close()
        return (data)

    def BiasSet(self, bias, prt=if_print):
        """
        Sets the Bias voltage to the specified value..

        Parameters:
            bias (float32): The bias voltage to set (in volts).
            prt (bool): Whether to print the output (default is `if_print`).

        Raises:
            ValueError: If the bias exceeds the maximum allowed value of 10V.

        Returns:
            pd.DataFrame: A DataFrame containing the set bias voltage.
        """
        bias = self.tcp.unit_cvt(bias)
        if bias > 10:
            raise ValueError('The maximum allowed bias is 10V. Please check your input! Bias has been set to 0 to protect the tip!')    
        
        body = self.tcp.dtype_cvt(bias, 'float32', 'bin')
        header = self.tcp.header_construct('Bias.Set', body_size=len(body))
        cmd = header + body

        self.tcp.cmd_send(cmd)
        _, _, res_err = self.tcp.res_recv()

        self.tcp.print_err(res_err)
        bias_df = pd.DataFrame({'Bias (V)': bias}, index=[0]).T
        
        if prt: 
            print('\n' + bias_df.to_string(header=False) + '\n\nBias set.')
        else:
            print('output suppressed')
        return bias_df 

    def BiasGet(self, prt=if_print):
        """
        Returns the Bias voltage value.

        Parameters:
            prt (bool): Whether to print the output (default is `if_print`).

        Returns:
            pd.DataFrame: A DataFrame containing the current bias voltage.
        """
        header = self.tcp.header_construct('Bias.Get', body_size=0)

        self.tcp.cmd_send(header)
        _, res_arg, res_err = self.tcp.res_recv('float32')

        self.tcp.print_err(res_err)
        bias_df = pd.DataFrame({'Bias (V)': res_arg[0]}, index=[0]).T
        
        if prt: 
            print('\n' + bias_df.to_string(header=False) + '\n\nBias returned.')
        return bias_df 
    
    def BiasRangeSet(self, bias_ran_idx, prt=if_print): 
        """
        Set the bias range index.

        Parameters:
            bias_ran_idx (int): The index of the bias range to set.
            prt (bool): Whether to print the output (default is `if_print`).

        Returns:
            pd.DataFrame: A DataFrame containing the set bias range index.
        """
        body  = self.tcp.dtype_cvt(bias_ran_idx, 'uint16', 'bin')

        header = self.tcp.header_construct('Bias.RangeSet', len(body))
        cmd = header + body

        self.tcp.cmd_send(cmd)
        _, _, res_err = self.tcp.res_recv()

        self.tcp.print_err(res_err)
        bias_range_df = pd.DataFrame({'Bias range index': bias_ran_idx}, index=[0]).T
        
        if prt: 
            print('\n' + bias_range_df.to_string(header=False) + '\n\nBias range set.')
        return bias_range_df

    def BiasRangeGet(self, prt=if_print):
        """
        Get the current bias range settings.

        Parameters:
            prt (bool): Whether to print the output (default is `if_print`).

        Returns:
            pd.DataFrame: A DataFrame containing the bias range settings.
        """
        header = self.tcp.header_construct('Bias.RangeGet', body_size=0)

        self.tcp.cmd_send(header)
        _, res_arg, res_err = self.tcp.res_recv('int', 'int', '1dstr', 'uint16')

        self.tcp.print_err(res_err)
        bias_range_df = pd.DataFrame({'Bias ranges size': res_arg[0],
                                      'Number of ranges': res_arg[1],
                                      'Bias ranges': res_arg[2].tolist(), 
                                      'Bias range index': res_arg[3]}, index=[0]).T
        
        if prt: 
            print('\n' + bias_range_df.to_string(header=False) + '\n\nBias range returned.')
        return bias_range_df
    
    def BiasCalibrSet(self, calibr, offset, prt=if_print):
        """
        Set the bias calibration and offset values.

        Parameters:
            calibr (float): The calibration value.
            offset (float): The offset value.
            prt (bool): Whether to print the output (default is `if_print`).

        Returns:
            pd.DataFrame: A DataFrame containing the set calibration and offset values.
        """
        body  = self.tcp.dtype_cvt(calibr, 'float32', 'bin')
        body += self.tcp.dtype_cvt(offset, 'float32', 'bin')

        header = self.tcp.header_construct('Bias.CalibrSet', len(body))
        cmd = header + body

        self.tcp.cmd_send(cmd)
        _, _, res_err = self.tcp.res_recv()

        self.tcp.print_err(res_err)
        bias_calibr_df = pd.DataFrame({'Calibration': calibr,
                                      'Offset': offset}, index=[0]).T
        if prt: 
            print('\n' + bias_calibr_df.to_string(header=False) + '\n\nBias calibration set.')
        return bias_calibr_df

    def BiasCalibrGet(self, prt=if_print):
        """
        Get the current bias calibration and offset values.

        Parameters:
            prt (bool): Whether to print the output (default is `if_print`).

        Returns:
            pd.DataFrame: A DataFrame containing the current calibration and offset values.
        """
        header = self.tcp.header_construct('Bias.CalibrGet', body_size=0)

        self.tcp.cmd_send(header)
        _, res_arg, res_err = self.tcp.res_recv('float32', 'float32')

        self.tcp.print_err(res_err)
        bias_calibr_df = pd.DataFrame({'Calibration': res_arg[0],
                                       'Offset': res_arg[1]}, index=[0]).T
        if prt: 
            print('\n' + bias_calibr_df.to_string(header=False) + '\n\nBias calibration returned.')
        return bias_calibr_df
    
    def BiasPulse(self, wait_until_done, bias_pulse_width, bias_value, zctrl_on_hold, pulse_abs_rel, prt=if_print):
        """
        Set a bias pulse with specified parameters.

        Parameters:
            wait_until_done (int): Whether to wait until the pulse is done.
            bias_pulse_width (float): The width of the bias pulse (in seconds).
            bias_value (float): The value of the bias pulse (in volts).
            zctrl_on_hold (int): Z-controller hold status.
            pulse_abs_rel (int): Pulse absolute/relative status.
            prt (bool): Whether to print the output (default is `if_print`).

        Returns:
            pd.DataFrame: A DataFrame containing the bias pulse settings.
        """
        bias_pulse_width = self.tcp.unit_cvt(bias_pulse_width)
        bias_value = self.tcp.unit_cvt(bias_value)
        if bias_value > 10:
            raise ValueError('The maximum allowed bias is 10V. Please check your input! Bias has been set to 0 to protect the tip!') 
        
        body  = self.tcp.dtype_cvt(wait_until_done, 'uint32', 'bin')
        body += self.tcp.dtype_cvt(bias_pulse_width, 'float32', 'bin')
        body += self.tcp.dtype_cvt(bias_value, 'float32', 'bin')
        body += self.tcp.dtype_cvt(zctrl_on_hold, 'uint16', 'bin')
        body += self.tcp.dtype_cvt(pulse_abs_rel, 'uint16', 'bin')
        header = self.tcp.header_construct('Bias.Pulse', body_size=len(body))
        cmd = header + body

        self.tcp.cmd_send(cmd)
        _, _, res_err = self.tcp.res_recv()

        self.tcp.print_err(res_err)
        bias_pulse_df = pd.DataFrame({'Wait until done': self.tcp.bistate_cvt(wait_until_done),
                                      'Bias pulse width (s)': bias_pulse_width,
                                      'Bias value (V)': bias_value,
                                      'Z-Controller on hold': self.tcp.tristate_cvt(zctrl_on_hold),
                                      'Pulse absolute/relative': pulse_abs_rel}, index=[0]).T
        if prt: 
            print('\n' + bias_pulse_df.to_string(header=False) + '\n\nBias pulse set.')
        return bias_pulse_df 

    ######################################## Bias Spectroscopy Module #############################################

    def BiasSpectrOpen(self, prt = if_print):
        """
        Open the Bias Spectroscopy window.

        Parameters:
            prt (bool, optional): Whether to print a confirmation message. Default is `if_print`.
        """
        header = self.tcp.header_construct('BiasSpectr.Open', body_size=0)

        self.tcp.cmd_send(header)
        _, _, res_err = self.tcp.res_recv()

        self.tcp.print_err(res_err)
        if prt: 
            print('Bias Spectroscopy window opened.')

    def BiasSpectrStart(self, get_data, save_base_name, prt = if_print):
        """
        Start a Scanning Tunneling Spectroscopy (STS) measurement.

        Parameters:
            get_data (array-like): Data to be sent for measurement.
            save_base_name (str): Base name for saving data.
            prt (bool, optional): Whether to print the results. Default is `if_print`.

        Returns:
            tuple: A tuple containing two DataFrames - the spectroscopy data and parameters.
        """
        save_base_name_size = len(save_base_name)

        print('Scanning tunneling spectroscopy (STS) launched. Please wait...')
        body  = self.tcp.dtype_cvt(get_data, 'uint32', 'bin')
        body += self.tcp.dtype_cvt(save_base_name_size, 'int', 'bin')
        body += self.tcp.dtype_cvt(save_base_name, 'str', 'bin')
        header = self.tcp.header_construct('BiasSpectr.Start', body_size = len(body))
        cmd = header + body

        self.tcp.cmd_send(cmd)
        _, res_arg, res_err = self.tcp.res_recv('int', 'int', '1dstr', 'int', 'int', '2dfloat32', 'int', '1dfloat32')
        
        self.tcp.print_err(res_err)
        bias_spectr_df = pd.DataFrame(res_arg[5].T, columns = res_arg[2][0])

        bias_spectr_param_df = pd.DataFrame(res_arg[7].T)
        print('STS done!')
        if prt: 
            print(bias_spectr_df)
            print(bias_spectr_param_df)
        return bias_spectr_df, bias_spectr_param_df

    def BiasSpectrStop(self, prt = if_print):
        """
        Stop the Scanning Tunneling Spectroscopy (STS) measurement.

        Parameters:
            prt (bool, optional): Whether to print a confirmation message. Default is `if_print`.
        """
        header = self.tcp.header_construct('BiasSpectr.Stop', body_size=0)

        self.tcp.cmd_send(header)
        _, _, res_err = self.tcp.res_recv()

        self.tcp.print_err(res_err)
        if prt: 
            print('STS stopped.')

    def BiasSpectrStatusGet(self, prt = if_print):
        """
        Get the status of the Bias Spectroscopy.

        Parameters:
            prt (bool, optional): Whether to print the status. Default is `if_print`.

        Returns:
            pd.DataFrame: A DataFrame containing the Bias Spectroscopy status.
        """
        header = self.tcp.header_construct('BiasSpectr.StatusGet', body_size=0)

        self.tcp.cmd_send(header)
        _, res_arg, res_err = self.tcp.res_recv('uint32')

        self.tcp.print_err(res_err)
        status_df = pd.DataFrame({'Bias spectroscopy status': self.tcp.bistate_cvt(res_arg[0])}, index=[0]).T

        if prt: 
            print('\n'+
                status_df.to_string(header=False)+
                '\n\nBias Spectroscopy status returned.')
        return status_df

    def BiasSpectrChsSet(self, num_chs, ch_idx, prt = if_print):
        """
        Set the channels for Bias Spectroscopy.

        Parameters:
            num_chs (int): Number of channels to set.
            ch_idx (list of int): List of channel indexes.
            prt (bool, optional): Whether to print the result. Default is `if_print`.

        Returns:
            pd.DataFrame: A DataFrame with the number of channels and their indexes.
        """
        #print('To get the signal name and its corresponding index in the list of the 128 available signals in the Nanonis Controller, use the "Signal.NamesGet" function, or check the RT Idx value in the Signals Manager module.')

        body  = self.tcp.dtype_cvt(num_chs, 'int', 'bin')
        body += self.tcp.dtype_cvt(ch_idx, '1dint', 'bin')
        header = self.tcp.header_construct('BiasSpectr.ChsSet', body_size = len(body))
        cmd = header + body

        self.tcp.cmd_send(cmd)
        _, _, res_err = self.tcp.res_recv()

        self.tcp.print_err(res_err)
        chs_df = pd.DataFrame({'Number of channels': num_chs,
                               'Channel indexes': ch_idx},
                               index=[0]).T

        if prt: 
            print('\n'+
                chs_df.to_string(header=False)+
                '\n\nChannels set.')
        return chs_df

    def BiasSpectrChsGet(self, prt = if_print):
        """
        Get the current channels set for Bias Spectroscopy.

        Parameters:
            prt (bool, optional): Whether to print the result. Default is `if_print`.

        Returns:
            pd.DataFrame: A DataFrame with the number of channels and their indexes.
        """
        header = self.tcp.header_construct('BiasSpectr.ChsGet', body_size = 0)

        self.tcp.cmd_send(header)
        _, res_arg, res_err = self.tcp.res_recv('int', '1dint')

        self.tcp.print_err(res_err)
        chs_df = pd.DataFrame({'Number of channels': res_arg[0],
                               'Channel indexes': [res_arg[1]]},
                               index=[0]).T

        if prt: 
            print('\n'+
                chs_df.to_string(header=False)+
                '\n\n Channels returned.')
        return chs_df

    def BiasSpectrPropsSet(self, save_all, num_sweeps, bw_sweep, num_pts, z_offset, auto_save, show_save_dialog, prt = if_print):
        """
        Set the properties for Bias Spectroscopy.

        Parameters:
            save_all (bool): Whether to save all data.
            num_sweeps (int): Number of sweeps.
            bw_sweep (bool): Whether to perform a backward sweep.
            num_pts (int): Number of points.
            z_offset (float): Z offset in meters.
            auto_save (bool): Whether to enable auto save.
            show_save_dialog (bool): Whether to show the save dialog.
            prt (bool, optional): Whether to print the result. Default is `if_print`.

        Returns:
            pd.DataFrame: A DataFrame with the properties set for Bias Spectroscopy.
        """
        z_offset = self.tcp.unit_cvt(z_offset)

        body  = self.tcp.dtype_cvt(save_all, 'uint16', 'bin')
        body += self.tcp.dtype_cvt(num_sweeps, 'int', 'bin')
        body += self.tcp.dtype_cvt(bw_sweep, 'uint16', 'bin')
        body += self.tcp.dtype_cvt(num_pts, 'int', 'bin')
        body += self.tcp.dtype_cvt(z_offset, 'float32', 'bin')
        body += self.tcp.dtype_cvt(auto_save, 'uint16', 'bin')
        body += self.tcp.dtype_cvt(show_save_dialog, 'uint16', 'bin')
        header = self.tcp.header_construct('BiasSpectr.PropsSet', body_size = len(body))
        cmd = header + body

        self.tcp.cmd_send(cmd)
        _, _, res_err = self.tcp.res_recv()

        self.tcp.print_err(res_err)
        props_df = pd.DataFrame({'Save all': self.tcp.tristate_cvt(save_all), 
                                 'Number of sweeps': num_sweeps, 
                                 'Backward sweep': self.tcp.tristate_cvt(bw_sweep), 
                                 'Number of points': num_pts, 
                                 'Z offset (m)': z_offset,
                                 'Autosave': self.tcp.tristate_cvt(auto_save), 
                                 'Show save dialog': self.tcp.tristate_cvt(show_save_dialog)},
                                 index=[0]).T

        if prt: 
            print('\n'+
                props_df.to_string(header=False)+
                '\n\nBias spectroscopy properties set.')
        return props_df

    def BiasSpectrPropsGet(self, prt = if_print):
        """
            BiasSpectr.PropsGet
            Returns the Bias Spectroscopy parameters.
        
            This function retrieves various parameters related to the Bias Spectroscopy module, such as the number of sweeps, channels, and whether to save individual sweeps.
        
            Arguments: None
        
            Return arguments (if Send response back flag is set to True when sending request message):
            - Save all (unsigned int16): Indicates whether the data from individual sweeps are saved along with the average data (1 = Save, 0 = Don't Save). This parameter is relevant only when multiple sweeps are configured.
            - Number of sweeps (int): The number of sweeps to measure and average.
            - Backward sweep (unsigned int16): Indicates whether the backward sweep is performed (1 = Yes, 0 = No). Forward is always measured.
            - Number of points (int): The number of points to acquire over the sweep range.
            - Channels size (int): The size in bytes of the Channels string array.
            - Number of channels (int): The number of channels in the Channels string array.
            - Channels (1D array string): Names of the acquired channels in the sweep. The size of each string is defined by an integer 32 that precedes it.
            - Parameters size (int): The size in bytes of the Parameters string array.
            - Number of parameters (int): The number of elements in the Parameters string array.
            - Parameters (1D array string): Parameters of the sweep. The size of each string is defined by an integer 32 that precedes it.
            - Fixed parameters size (int): The size in bytes of the Fixed parameters string array.
            - Number of fixed parameters (int): The number of elements in the Fixed parameters string array.
            - Fixed parameters (1D array string): Fixed parameters of the sweep. The size of each string is defined by an integer 32 that precedes it.
            - Error: Described in the Response message>Body section.
        
            The function constructs a DataFrame with the retrieved properties and returns it. The DataFrame contains:
            - 'Save all': The save-all flag converted to a readable format.
            - 'Number of sweeps': Number of configured sweeps.
            - 'Backward sweep': Backward sweep flag converted to a readable format.
            - 'Number of points': Total number of points across the sweep range.
            - 'Number of channels': The number of channels involved in the sweep.
            - 'Channels': A list of the names of the channels.
            - 'Number of parameters': Total number of parameters involved in the sweep.
            - 'Parameters': A list of the parameters.
            - 'Number of fixed parameters': Total number of fixed parameters.
            - 'Fixed parameters': A list of the fixed parameters.
        
            If `prt` is set to True, the DataFrame is printed in a formatted manner.
        
            Parameters:
            - prt (bool): If True, prints the properties DataFrame (default is the value of if_print).
        
            Returns:
            - props_df (DataFrame): DataFrame containing the Bias Spectroscopy properties.
            """
        header = self.tcp.header_construct('BiasSpectr.PropsGet', body_size = 0)
    
        self.tcp.cmd_send(header)
        _, res_arg, res_err = self.tcp.res_recv('uint16', 'int', 'uint16', 'int', 'int', 'int', '1dstr', 'int', 'int', '1dstr', 'int', 'int', '1dstr')
    
        self.tcp.print_err(res_err)
        props_df = pd.DataFrame({'Save all': self.tcp.bistate_cvt(res_arg[0]), 
                                 'Number of sweeps': res_arg[1], 
                                 'Backward sweep': self.tcp.bistate_cvt(res_arg[2]), 
                                 'Number of points': res_arg[3], 
                                 'Number of channels': res_arg[5],
                                 'Channels': res_arg[6].tolist(), 
                                 'Number of parameters': res_arg[8],
                                 'Parameters': res_arg[9].tolist(),
                                 'Number of fixed parameters': res_arg[11],
                                 'Fixed parameters': res_arg[12].tolist()
                                 },
                                 index=[0]).T
        if prt: 
            print('\n'+
                props_df.to_string(header=False) + 
                '\n\nBias spectroscopy properties returned.')
        return props_df
    

    def BiasSpectrAdvPropsSet(self, reset_bias, z_ctrl_hold, rec_final_z, lock_in_run, prt = if_print):
        """
        Set advanced properties for Bias Spectroscopy.

        Parameters:
            reset_bias (bool): Whether to reset the bias.
            lockin_run (bool): Whether to start the lock-in run.
            prt (bool, optional): Whether to print the result. Default is `if_print`.
        """
        body  = self.tcp.dtype_cvt(reset_bias, 'uint16', 'bin')
        body += self.tcp.dtype_cvt(z_ctrl_hold, 'uint16', 'bin')
        body += self.tcp.dtype_cvt(rec_final_z, 'uint16', 'bin')
        body += self.tcp.dtype_cvt(lock_in_run, 'uint16', 'bin')
        header = self.tcp.header_construct('BiasSpectr.AdvPropsSet', body_size = len(body))
        cmd = header + body

        self.tcp.cmd_send(cmd)
        _, _, res_err = self.tcp.res_recv()

        self.tcp.print_err(res_err)
        props_df = pd.DataFrame({'Reset bias': self.tcp.tristate_cvt(reset_bias), 
                                 'Z-Controller hold': self.tcp.tristate_cvt(z_ctrl_hold), 
                                 'Record final Z': self.tcp.tristate_cvt(rec_final_z), 
                                 'Lockin Run': self.tcp.tristate_cvt(lock_in_run), 
                                 },
                                 index=[0]).T
        if prt: 
            print('\n'+
                props_df.to_string(header=False)+
                '\n\nBias spectroscopy advanced properties set.')
        return props_df
    

    def BiasSpectrAdvPropsGet(self, prt = if_print):
        """
        Get the advanced properties of Bias Spectroscopy.

        Parameters:
            prt (bool, optional): Whether to print the advanced properties. Default is `if_print`.

        Returns:
            pd.DataFrame: A DataFrame with the current advanced properties of Bias Spectroscopy.
        """
        header = self.tcp.header_construct('BiasSpectr.AdvPropsGet', body_size = 0)

        self.tcp.cmd_send(header)
        _, res_arg, res_err = self.tcp.res_recv('uint16', 'uint16', 'uint16', 'uint16')

        self.tcp.print_err(res_err)
        props_df = pd.DataFrame({'Reset bias': self.tcp.bistate_cvt(res_arg[0]), 
                                 'Z-Controller hold': self.tcp.bistate_cvt(res_arg[1]), 
                                 'Record final Z': self.tcp.bistate_cvt(res_arg[2]), 
                                 'Lockin Run': self.tcp.bistate_cvt(res_arg[3]), 
                                 },
                                 index=[0]).T
        if prt: 
            print('\n'+
                props_df.to_string(header=False)+
                '\n\nBias spectroscopy advanced properties returned.')
        return props_df
    

    def BiasSpectrLimitsSet(self, start_val, end_val, prt = if_print):
        """
        Set the bias limits for Bias Spectroscopy.

        Parameters:
            start_val (float): The lower bias limit.
            end_val(float): The upper bias limit.
            prt (bool, optional): Whether to print the result. Default is `if_print`.

        Returns:
            pd.DataFrame: A DataFrame with the bias limits set.
        """
        start_val = self.tcp.unit_cvt(start_val)
        end_val = self.tcp.unit_cvt(end_val)

        body  = self.tcp.dtype_cvt(start_val, 'float32', 'bin')
        body += self.tcp.dtype_cvt(end_val, 'float32', 'bin')
        header = self.tcp.header_construct('BiasSpectr.LimitsSet', body_size = len(body))
        cmd = header + body

        self.tcp.cmd_send(cmd)
        _, _, res_err = self.tcp.res_recv()

        self.tcp.print_err(res_err)
        limits_df = pd.DataFrame({'Start value (V)': start_val, 
                                 'Stop value (V)': end_val, 
                                 },
                                 index=[0]).T
        if prt: 
            print('\n'+
                limits_df.to_string(header=False)+
                '\n\nBias limits set.')
        return limits_df


    def BiasSpectrLimitsGet(self, prt = if_print):
        """
        Get the current bias limits for Bias Spectroscopy.

        Parameters:
            prt (bool, optional): Whether to print the bias limits. Default is `if_print`.

        Returns:
            pd.DataFrame: A DataFrame with the current bias limits.
        """
        header = self.tcp.header_construct('BiasSpectr.LimitsGet', body_size = 0)

        self.tcp.cmd_send(header)
        _, res_arg, res_err = self.tcp.res_recv('float32', 'float32')

        self.tcp.print_err(res_err)
        limits_df = pd.DataFrame({'Start value (V)': res_arg[0], 
                                 'Stop value (V)': res_arg[1]},
                                 index=[0]).T
        if prt: 
            print('\n'+
                limits_df.to_string(header=False)+
                '\n\nBias spectroscopy bias limits returned.')
        return limits_df
    
    def BiasSpectrTimingSet(self, z_avg_t, z_offset, init_settling_t, max_slew_rate, settling_t, inte_t, end_settling_t, z_ctrl_t, prt = if_print):
        """
        Set the timing parameters for Bias Spectroscopy.
    
        Parameters:
            z_avg_t (float): Z averaging time in seconds.
            z_offset (float): Z offset in meters.
            init_settling_t (float): Initial settling time in seconds.
            max_slew_rate (float): Maximum slew rate in volts per second (V/s).
            settling_t (float): Settling time in seconds.
            inte_t (float): Integration time in seconds.
            end_settling_t (float): End settling time in seconds.
            z_ctrl_t (float): Z control time in seconds.
            prt (bool, optional): Whether to print the result. Defaults to `if_print`.
    
        Returns:
            pd.DataFrame: A DataFrame containing the timing parameters that were set.
        """
        z_avg_t = self.tcp.unit_cvt(z_avg_t)
        z_offset = self.tcp.unit_cvt(z_offset)
        init_settling_t = self.tcp.unit_cvt(init_settling_t)
        max_slew_rate = self.tcp.unit_cvt(max_slew_rate)
        settling_t = self.tcp.unit_cvt(settling_t)
        inte_t = self.tcp.unit_cvt(inte_t)
        end_settling_t = self.tcp.unit_cvt(end_settling_t)
        z_ctrl_t = self.tcp.unit_cvt(z_ctrl_t)
    
        body  = self.tcp.dtype_cvt(z_avg_t, 'float32', 'bin')
        body += self.tcp.dtype_cvt(z_offset, 'float32', 'bin')
        body += self.tcp.dtype_cvt(init_settling_t, 'float32', 'bin')
        body += self.tcp.dtype_cvt(max_slew_rate, 'float32', 'bin')
        body += self.tcp.dtype_cvt(settling_t, 'float32', 'bin')
        body += self.tcp.dtype_cvt(inte_t, 'float32', 'bin')
        body += self.tcp.dtype_cvt(end_settling_t, 'float32', 'bin')
        body += self.tcp.dtype_cvt(z_ctrl_t, 'float32', 'bin')
        header = self.tcp.header_construct('BiasSpectr.TimingSet', len(body))
        cmd = header + body
    
        self.tcp.cmd_send(cmd)
        _, _, res_err = self.tcp.res_recv()
    
        self.tcp.print_err(res_err)
        timing_df = pd.DataFrame({'Z averaging time (s)': z_avg_t,
                                   'Z offset (m)': z_offset,
                                   'Initial settling time (s)': init_settling_t,
                                   'Maximum slew rate (V/s)': max_slew_rate,
                                   'Settling time (s)': settling_t,
                                   'Integration time (s)': inte_t,
                                   'End settling time (s)': end_settling_t,
                                   'Z control time (s)': z_ctrl_t},
                                 index=[0]).T
        if prt: 
            print('\n'+
                timing_df.to_string(header=False)+
                '\n\nBias spectroscopy timing set.')
        return timing_df


    def BiasSpectrTimingGet(self, prt = if_print):
        """
        Get the current timing parameters for Bias Spectroscopy.

        Parameters:
            prt (bool, optional): Whether to print the timing parameters. Default is `if_print`.

        Returns:
            pd.DataFrame: A DataFrame with the current timing parameters.
        """
        header = self.tcp.header_construct('BiasSpectr.TimingGet', 0)

        self.tcp.cmd_send(header)
        _, res_arg, res_err = self.tcp.res_recv('float32','float32','float32','float32','float32','float32','float32','float32')

        self.tcp.print_err(res_err)
        timing_df = pd.DataFrame({'Z averaging time (s)': res_arg[0],
                                  'Z offset (m)': res_arg[1],
                                  'Initial settling time (s)': res_arg[2],
                                  'Maximum slew rate (V/s)': res_arg[3],
                                  'Settling time (s)': res_arg[4],
                                  'Integration time (s)': res_arg[5],
                                  'End settling time (s)': res_arg[6],
                                  'Z control time (s)': res_arg[7]},
                                 index=[0]).T
        if prt: 
            print('\n'+
                timing_df.to_string(header=False)+
                '\n\nBias spectroscopy timing settings retured.')
        return timing_df
        
    
    def BiasSpectrTTLSyncSet(self, enable, ttl_line, ttl_polarity, t_2_on, on_duration, prt = if_print):
        """
        Set the TTL synchronization parameters for Bias Spectroscopy.

        Parameters:
            ttl_sync (bool): Whether to synchronize with TTL signals.
            prt (bool, optional): Whether to print the result. Default is `if_print`.

        Returns:
            pd.DataFrame: A DataFrame with the TTL synchronization settings.
        """
        t_2_on = self.tcp.unit_cvt(t_2_on)
        on_duration = self.tcp.unit_cvt(on_duration)

        body  = self.tcp.dtype_cvt(enable, 'uint16', 'bin')
        body += self.tcp.dtype_cvt(ttl_line, 'uint16', 'bin')
        body += self.tcp.dtype_cvt(ttl_polarity, 'uint16', 'bin')
        body += self.tcp.dtype_cvt(t_2_on, 'float32', 'bin')
        body += self.tcp.dtype_cvt(on_duration, 'float32', 'bin')
        header = self.tcp.header_construct('BiasSpectr.TTLSyncSet', len(body))
        cmd = header + body

        self.tcp.cmd_send(cmd)
        _, _, res_err = self.tcp.res_recv()

        self.tcp.print_err(res_err)
        ttl_df = pd.DataFrame({'Enable': self.tcp.tristate_cvt(enable),
                               'TTL line': ttl_line,
                               'TTL polarity': ttl_polarity,
                               'Time to on (s)': t_2_on,
                               'On duration (s)': on_duration},
                                index=[0]).T
        if prt: 
            print('\n'+
                ttl_df.to_string(header=False)+
                '\n\nTTL sychronizetion set.')
        return ttl_df

    def BiasSpectrTTLSyncGet(self, prt = if_print):
        """
        Get the current TTL synchronization parameters for Bias Spectroscopy.
        
        Parameters:
            prt (bool, optional): Whether to print the TTL synchronization settings. Default is `if_print`.
    
        Returns:
            pd.DataFrame: A DataFrame with the current TTL synchronization settings.
        """
        header = self.tcp.header_construct('BiasSpectr.TTLSyncGet', body_size = 0)
    
        self.tcp.cmd_send(header)
        _, res_arg, res_err = self.tcp.res_recv('uint16', 'uint16', 'uint16', 'float32', 'float32')
    
        self.tcp.print_err(res_err)
        ttl_df = pd.DataFrame({'Enable': res_arg[0],
                               'TTL line': res_arg[1],
                               'TTL polarity': res_arg[2],
                               'Time to on (s)': res_arg[3],
                               'On duration (s)': res_arg[4]},
                                index=[0]).T
        if prt: 
            print('\n'+
                ttl_df.to_string(header=False)+
                '\n\nTTL sychronizetion settings returned.')
        return ttl_df
        
    def BiasSpectrAltZCtrlSet(self, alt_z_ctrl_sp, sp, settling_t, prt = if_print):
        """
        BiasSpectr.AltZCtrlSet
        Sets the configuration of the alternate Z-controller setpoint in the Advanced section of the Bias Spectroscopy module.
        
        When switched on, the Z-controller setpoint is set to the setpoint right after starting the measurement. 
        After changing the setpoint, the settling time (in seconds) will be waited for the Z-controller to adjust to the modified setpoint.
        Then, Z averaging will start. The original Z-controller setpoint is restored at the end of the measurement, before restoring the Z-controller state.
    
        Arguments:
        - Alternate Z-controller setpoint (unsigned int16): 0 means no change, 1 means On, and 2 means Off
        - Setpoint (float32): The new setpoint for the Z-controller
        - Settling time (float32): Time (in seconds) to wait for the Z-controller to stabilize
    
        Return arguments (if Send response back flag is set to True when sending request message):
        - Error described in the Response message>Body section
        """
        sp = self.tcp.unit_cvt(sp)
        settling_t = self.tcp.unit_cvt(settling_t)
    
        body  = self.tcp.dtype_cvt(alt_z_ctrl_sp, 'uint16', 'bin')
        body += self.tcp.dtype_cvt(sp, 'float32', 'bin')
        body += self.tcp.dtype_cvt(settling_t, 'float32', 'bin')
        header = self.tcp.header_construct('BiasSpectr.AltZCtrlSet', len(body))
        cmd = header + body
    
        self.tcp.cmd_send(cmd)
        _, _, res_err = self.tcp.res_recv()
    
        self.tcp.print_err(res_err)
        alt_z_ctrl_df = pd.DataFrame({'Alternative Z-controller setpoint': self.tcp.tristate_cvt(alt_z_ctrl_sp),
                                      'Setpoint (A)': sp,
                                      'Settling time (s)': settling_t
                                      },
                                        index=[0]).T  
        if prt: 
            print('\n'+
                alt_z_ctrl_df.to_string(header=False)+
                '\n\nAlternative Z controller set.')
        return alt_z_ctrl_df   
    
    def BiasSpectrAltZCtrlGet(self, prt = if_print):
        """
        BiasSpectr.AltZCtrlGet
        Returns the configuration of the alternate Z-controller setpoint in the Advanced section of the Bias Spectroscopy module.
        
        When switched on, the Z-controller setpoint is set to the defined setpoint right after starting the measurement. 
        After changing the setpoint, the settling time will be waited for the Z-controller to adjust to the modified setpoint.
        Then, Z averaging will start, and the original setpoint is restored at the end of the measurement, before restoring the Z-controller state.
        
        Arguments: None
        
        Return arguments (if Send response back flag is set to True when sending request message):
        - Alternate Z-controller setpoint (unsigned int16): 0 means Off, 1 means On
        - Setpoint (float32): The current setpoint for the Z-controller
        - Settling time (float32): The configured settling time (in seconds)
        - Error described in the Response message>Body section
        """
        header = self.tcp.header_construct('BiasSpectr.AltZCtrlGet', 0)
    
        self.tcp.cmd_send(header)
        _, res_arg, res_err = self.tcp.res_recv('uint16', 'float32', 'float32')
    
        self.tcp.print_err(res_err)
        alt_z_ctrl_df = pd.DataFrame({'Alternative Z-controller setpoint': self.tcp.bistate_cvt(res_arg[0]),
                                      'Setpoint (A)': res_arg[1],
                                      'Settling time (s)': res_arg[2]
                                      },
                                      index=[0]).T  
        if prt: 
            print('\n'+
                alt_z_ctrl_df.to_string(header=False)+
                '\n\nAlternative Z controller settings returned.')
        return alt_z_ctrl_df   
    
    def BiasSpectrMLSLockinPerSegSet(self, lockin_per_seg, prt = if_print):
        """
        BiasSpectr.MLSLockinPerSegSet
        Sets the Lock-In per Segment flag in the Multi Line Segment editor.
    
        When enabled, the Lock-In can be configured for each segment in the Multi Line Segment editor. 
        Otherwise, the Lock-In is applied globally according to the flag in the Advanced section of Bias spectroscopy.
    
        Arguments:
        - Lock-In per segment (unsigned int32): 0 means Off, 1 means On
    
        Return arguments (if Send response back flag is set to True when sending request message):
        - Error described in the Response message>Body section
        """
        body  = self.tcp.dtype_cvt(lockin_per_seg, 'uint32', 'bin')
        header = self.tcp.header_construct('BiasSpectr.MLSLockinPerSegSet', len(body))
        cmd = header + body
    
        self.tcp.cmd_send(cmd)
        _, _, res_err = self.tcp.res_recv()
    
        self.tcp.print_err(res_err)
        lockin_per_seg_df = pd.DataFrame({'Lock-in per segment': self.tcp.bistate_cvt(lockin_per_seg)},
                                index=[0]).T
        if prt: 
            print('\n'+
                lockin_per_seg_df.to_string(header=False)+
                '\n\nLock-In per Segment flag in Multi line segment editor set.')
        return lockin_per_seg_df
    
    def BiasSpectrMLSLockinPerSegGet(self, prt = if_print):
        """
        BiasSpectr.MLSLockinPerSegGet
        Returns the Lock-In per Segment flag in the Multi Line Segment editor.
        
        When selected, the Lock-In can be defined per segment. Otherwise, it is set globally.
        
        Arguments: None
        
        Return arguments (if Send response back flag is set to True when sending request message):
        - Lock-In per segment (unsigned int32): 0 means Off, 1 means On
        - Error described in the Response message>Body section
        """
        header = self.tcp.header_construct('BiasSpectr.MLSLockinPerSegGet', 0)
    
        self.tcp.cmd_send(header)
        _, res_arg, res_err = self.tcp.res_recv('uint32')
    
        self.tcp.print_err(res_err)
        lockin_per_seg_df = pd.DataFrame({'Lock-in per segment': self.tcp.bistate_cvt(res_arg[0])},
                                index=[0]).T
        if prt: 
            print('\n'+
                lockin_per_seg_df.to_string(header=False)+
                '\n\nLock-In per Segment flag in Multi line segment editor settings returned.')
        return lockin_per_seg_df          
    
    def BiasSpectrMLSModeSet(self, sweep_mode, prt = if_print):
        """
        BiasSpectr.MLSModeSet
        Sets the Bias Spectroscopy sweep mode.
        
        Arguments:
        - Sweep mode (int): The number of characters in the sweep mode string. For example, 6 for 'Linear' and 3 for 'MLS'
        - Sweep mode (string): 'Linear' for Linear mode or 'MLS' for MultiSegment mode
        
        Return arguments (if Send response back flag is set to True when sending request message):
        - Error described in the Response message>Body section
        """
        # sweep mode: 'Linear' or 'MLS'
        sweep_mode_len = len(sweep_mode)
    
        body  = self.tcp.dtype_cvt(sweep_mode_len, 'int', 'bin')
        body += self.tcp.dtype_cvt(sweep_mode, 'str', 'bin')
        header = self.tcp.header_construct('BiasSpectr.MLSModeSet', body_size = len(body))
        cmd = header + body
    
        self.tcp.cmd_send(cmd)
        _, _, res_err = self.tcp.res_recv()
    
        self.tcp.print_err(res_err)
        mls_mode_df = pd.DataFrame({'Sweep mode': sweep_mode},
                                 index=[0]).T
        if prt: 
            print('\n'+
                mls_mode_df.to_string(header=False)+
                '\n\nBias spectroscopy sweep mode set.')
        return mls_mode_df
    
    def BiasSpectrMLSModeGet(self, prt = if_print):
        """
        BiasSpectr.MLSModeGet
        Returns the Bias Spectroscopy sweep mode.
    
        Arguments: None
    
        Return arguments (if Send response back flag is set to True when sending request message):
        - Sweep mode (int): Number of characters in the sweep mode string. 6 for 'Linear', 3 for 'MLS'
        - Sweep mode (string): 'Linear' for Linear mode or 'MLS' for MultiSegment mode
        - Error described in the Response message>Body section
        """
        header = self.tcp.header_construct('BiasSpectr.MLSModeGet', 0)
    
        self.tcp.cmd_send(header)
        _, res_arg, res_err = self.tcp.res_recv('int', 'str') 
    
        self.tcp.print_err(res_err)
        mls_mode_df = pd.DataFrame({'Sweep mode': res_arg[1]},
                                index=[0]).T
        if prt: 
            print('\n'+
                mls_mode_df.to_string(header=False)+
                '\n\nLock-In per Segment flag in Multi line segment editor settings returned.')
        return mls_mode_df 
    
    def BiasSpectrMLSValsSet(self, num_segs, bias_start, bias_end, init_settling_t, settling_t, inte_t, steps, lockin_run, prt = if_print):
        """
        BiasSpectr.MLSValsSet
        Sets the bias spectroscopy multiple line segment configuration for Multi Line Segment mode.
    
        Up to 16 distinct line segments may be defined. Any segments beyond the maximum allowed will be ignored.
    
        Arguments:
        - Number of segments (int): Number of segments configured in MLS mode. This value determines the size of the 1D arrays set afterwards
        - Bias start (V) (1D array float32): Start bias value (V) for each segment
        - Bias end (V) (1D array float32): End bias value (V) for each segment
        - Initial settling time (s) (1D array float32): Time to wait at the beginning of each segment after applying the Lock-In setting
        - Settling time (s) (1D array float32): Time to wait before measuring each data point within the segment
        - Integration time (s) (1D array float32): Time during which the data are acquired and averaged for each segment
        - Steps (1D array int): Number of steps to measure in each segment
        - Lock-In run (1D array unsigned int32): Indicates if Lock-In will run during the segment (requires the global Lock-In per Segment flag to be enabled)
    
        Return arguments (if Send response back flag is set to True when sending request message):
        - Error described in the Response message>Body section
        """
        bias_start = self.tcp.unit_cvt(bias_start)
        bias_end = self.tcp.unit_cvt(bias_end)
        init_settling_t = self.tcp.unit_cvt(init_settling_t)
        settling_t = self.tcp.unit_cvt(settling_t)
        inte_t = self.tcp.unit_cvt(inte_t)
    
        body  = self.tcp.dtype_cvt(num_segs, 'int', 'bin')
        body += self.tcp.dtype_cvt(bias_start, '1dfloat32', 'bin')
        body += self.tcp.dtype_cvt(bias_end, '1dfloat32', 'bin')
        body += self.tcp.dtype_cvt(init_settling_t, '1dfloat32', 'bin')
        body += self.tcp.dtype_cvt(settling_t, '1dfloat32', 'bin')
        body += self.tcp.dtype_cvt(inte_t, '1dfloat32', 'bin')
        body += self.tcp.dtype_cvt(steps, '1dint', 'bin')
        body += self.tcp.dtype_cvt(lockin_run, '1duint32', 'bin')
        header = self.tcp.header_construct('BiasSpectr.MLSValsSet', body_size = len(body))
        cmd = header + body
    
        self.tcp.cmd_send(cmd)
        _, _, res_err = self.tcp.res_recv()
    
        self.tcp.print_err(res_err)
        mls_df = pd.DataFrame({'Number of segments': num_segs,
                               'Bias start (V)': bias_start,
                               'Bias end (V)': bias_end,
                               'Initial settling time (s)': init_settling_t,
                               'Settling time (s)': settling_t,
                               'Integration time (s)': inte_t,
                               'Steps': steps,
                               'Lock-in run': lockin_run},
                                 index=[0]).T
        if prt: 
            print('\n'+
                mls_df.to_string(header=False)+
                '\n\nBias sepectroscopy line segment configuration for Multi Line Segment mode set.')
        return mls_df 
       
    def BiasSpectrMLSValsGet(self, prt = if_print): # might encounter issues when having multiple segaments
        """
        BiasSpectr.MLSValsGet
        Returns the bias spectroscopy multiple line segment configuration for Multi Line Segment mode.
    
        Up to 16 distinct line segments may be defined.
    
        Arguments: None
    
        Return arguments (if Send response back flag is set to True when sending request message):
        - Number of segments (int): Indicates the number of segments configured in MLS mode
        - Bias start (V) (1D array float32): Start bias value (V) for each segment
        - Bias end (V) (1D array float32): End bias value (V) for each segment
        - Initial settling time (s) (1D array float32): Time to wait at the beginning of each segment
        - Settling time (s) (1D array float32): Time to wait before each data point measurement
        - Integration time (s) (1D array float32): Time for data acquisition and averaging for each segment
        - Steps (1D array int): Number of steps to measure in each segment
        - Lock-In run (1D array unsigned int32): Lock-In status per segment
        - Error described in the Response message>Body section
        """
        header = self.tcp.header_construct('BiasSpectr.MLSValsGet', body_size = 0)
    
        self.tcp.cmd_send(header)
        _, res_arg, res_err = self.tcp.res_recv('int', '1dfloat32', '1dfloat32', '1dfloat32', '1dfloat32', '1dfloat32', '1dint', '1duint32')
    
        self.tcp.print_err(res_err)
        mls_df = pd.DataFrame({'Number of segments': res_arg[0],
                               'Bias start (V)': res_arg[1],
                               'Bias end (V)': res_arg[2],
                               'Initial settling time (s)': res_arg[3],
                               'Settling time (s)': res_arg[4],
                               'Integration time (s)': res_arg[5],
                               'Steps': res_arg[6],
                               'Lock-in run': res_arg[7]},
                                index=[0]).T
        if prt: 
            print('\n'+
                mls_df.to_string(header=False)+
                '\n\nBias sepectroscopy line segment configuration for Multi Line Segment mode settings returned.')
        return mls_df 
        
        
        
        
        
        
        
######################################## Current Module #############################################
    def CurrentGet(self, prt = if_print):
        """
        Returns the tunneling current value.

        Parameters:
            None,

        Returns:
            pd.DataFrame: A DataFrame with 
            
                Current value (A) (float32)
        """       
        header = self.tcp.header_construct('Current.Get', 0)

        self.tcp.cmd_send(header)
        _, res_arg, res_err = self.tcp.res_recv('float32') 

        self.tcp.print_err(res_err)
        current_df = pd.DataFrame({'Current value (A)': res_arg},
                                index=[0]).T
        if prt: 
            print('\n'+
                current_df.to_string(header=False)+
                '\n\nCurrent value returned.')
        return current_df 

    def Current100Get(self, prt = if_print):
        return

    def CurrentBEEMGet(self, prt = if_print):
        return

    def CurrentGainSet(self, current_gain_idx, prt = if_print):
        """
        Sets the calibration and offset of the selected gain in the Current module


        Parameters:
            current_gain_idx (int): The index of the current gain to set.
            prt (bool): Whether to print the output (default is `if_print`).

        Returns:
            pd.DataFrame: A DataFrame containing the set gain index.
        """
        body  = self.tcp.dtype_cvt(current_gain_idx , 'uint16', 'bin')

        header = self.tcp.header_construct('Current.GainSet', len(body))
        cmd = header + body

        self.tcp.cmd_send(cmd)
        _, _, res_err = self.tcp.res_recv()

        self.tcp.print_err(res_err)
        current_gain_df = pd.DataFrame({'Current gain index': current_gain_idx}, index=[0]).T
        
        if prt: 
            print('\n' + current_gain_df.to_string(header=False) + '\n\nCurrent gain set.')
        return current_gain_df
    

    def CurrentGainsGet(self, prt = if_print):
        """
        Get the selectable gains of the current amplifier and the index of the selected one.

        Parameters:
            prt (bool): Whether to print the output (default is `if_print`).

        Returns:
            pd.DataFrame: A DataFrame containing the gains settings.
                - Gains size (int32) is the size in bytes of the Gains array
                - Number of gains (int32) is the number of elements of the Gains array
                - Gains (1D list) returns an array of selectable gains. Each element of the array is preceded by its
                size in bytes
                - Gain index (uint) is the index out of the list of gains
        """
        header = self.tcp.header_construct('Current.GainsGet', 0)
        
        self.tcp.cmd_send(header)
        _, res_arg, res_err = self.tcp.res_recv('int', 'int', '1dstr','uint16')
        self.tcp.print_err(res_err)
        
        
        current_gains_df = pd.DataFrame({'Gains size': res_arg[0],
                                      'Number of gains': res_arg[1],
                                      'Gains': res_arg[2].tolist(), 
                                      'Gain index': res_arg[3]}, index=[0]).T

        if prt: 
            print('\n'+
                current_gains_df.to_string(header=False)+
                '\n\n Current gains returned.')
        return current_gains_df
        

    def CurrentCalibrSet(self, prt = if_print):
        return

    def CurrentCalibrGet(self, prt = if_print):
        header = self.tcp.header_construct('Current.CalibrGet', 0)

        self.tcp.cmd_send(header)
        _, res_arg, res_err = self.tcp.res_recv('float64', 'float64')
        self.tcp.print_err(res_err)
        at_props_df = pd.DataFrame({'Calibration': res_arg[0],
                                    'Offset': res_arg[1]},
                                 index=[0]).T
        if prt: 
            print('\n'+
                at_props_df.to_string(header=False)+
                '\n\nCalibration and offset of the selected gain returned.')
        return at_props_df

######################################## Z-controller Module #############################################
    def ZCtrlZPosSet(self, z_pos, prt=if_print):
        """
        Set the Z-position of the tip.

        Converts the Z-position to the appropriate unit, constructs the command,
        and sends it to the Z-controller. Waits for a response and handles errors.

        Parameters
        ----------
        z_pos : float
            The desired Z-position of the tip in meters.
        prt : bool, optional
            Whether to print the result (default is if_print).

        Returns
        -------
        pd.DataFrame
            DataFrame containing the set Z-position of the tip.

        Raises
        ------
        Exception
            If there is an error in communication with the Z-controller.
        """
        z_pos = self.tcp.unit_cvt(z_pos)
        body = self.tcp.dtype_cvt(z_pos, 'float32', 'bin')
        header = self.tcp.header_construct('ZCtrl.ZPosSet', len(body))
        cmd = header + body
        self.tcp.cmd_send(cmd)
        _, _, res_err = self.tcp.res_recv()
        self.tcp.print_err(res_err)
        z_pos_df = pd.DataFrame({'Z position of the tip (m)': z_pos}, index=[0]).T
        if prt: 
            print('\n' + z_pos_df.to_string(header=False) + '\n\nZ position of the tip set. Note: to change the Z-position of the tip, the Z-controller must be switched OFF!!!')
        return z_pos_df

    def ZCtrlZPosGet(self, prt=if_print):
        """
        Get the current Z-position of the tip.

        Constructs the command to request the Z-position and processes the response.

        Parameters
        ----------
        prt : bool, optional
            Whether to print the result (default is if_print).

        Returns
        -------
        pd.DataFrame
            DataFrame containing the current Z-position of the tip.

        Raises
        ------
        Exception
            If there is an error in communication with the Z-controller.
        """
        header = self.tcp.header_construct('ZCtrl.ZPosGet', 0)
        self.tcp.cmd_send(header)
        _, res_arg, res_err = self.tcp.res_recv('float32')
        self.tcp.print_err(res_err)
        z_pos_df = pd.DataFrame({'Z position of the tip (m)': res_arg[0]}, index=[0]).T
        if prt: 
            print('\n' + z_pos_df.to_string(header=False) + '\n\nZ position of the tip returned.')
        return z_pos_df

    def ZCtrlOnOffSet(self, z_ctrl_status, prt=if_print):
        """
        Set the on/off status of the Z-controller.

        Constructs the command to set the Z-controller status and processes the response.

        Parameters
        ----------
        z_ctrl_status : int
            The desired on/off status of the Z-controller (typically 0 or 1).
        prt : bool, optional
            Whether to print the result (default is if_print).

        Returns
        -------
        pd.DataFrame
            DataFrame containing the status of the Z-controller.

        Raises
        ------
        Exception
            If there is an error in communication with the Z-controller.
        """
        body = self.tcp.dtype_cvt(z_ctrl_status, 'uint32', 'bin')
        header = self.tcp.header_construct('ZCtrl.OnOffSet', len(body))
        cmd = header + body
        self.tcp.cmd_send(cmd)
        _, _, res_err = self.tcp.res_recv()
        self.tcp.print_err(res_err)
        z_ctrl_df = pd.DataFrame({'Z-controller status': self.tcp.bistate_cvt(z_ctrl_status)}, index=[0]).T
        if prt: 
            print('\n' + z_ctrl_df.to_string(header=False) + '\n\nZ-controller on/off set.')
        return z_ctrl_df

    def ZCtrlOnOffGet(self, prt=if_print):
        """
        Get the current on/off status of the Z-controller.

        Constructs the command to request the Z-controller status and processes the response.

        Parameters
        ----------
        prt : bool, optional
            Whether to print the result (default is if_print).

        Returns
        -------
        pd.DataFrame
            DataFrame containing the current status of the Z-controller.

        Raises
        ------
        Exception
            If there is an error in communication with the Z-controller.
        """
        header = self.tcp.header_construct('ZCtrl.OnOffGet', 0)
        self.tcp.cmd_send(header)
        _, res_arg, res_err = self.tcp.res_recv('uint32')
        self.tcp.print_err(res_err)
        z_ctrl_df = pd.DataFrame({'Z-controller status': self.tcp.bistate_cvt(res_arg[0])}, index=[0]).T
        if prt: 
            print('\n' + z_ctrl_df.to_string(header=False) + '\n\nZ-controller on/off status returned.')
        return z_ctrl_df

    def ZCtrlSetpntSet(self, z_ctrl_sp, prt=if_print):
        """
        Set the Z-controller setpoint.

        Converts the setpoint to the appropriate unit, constructs the command,
        and sends it to the Z-controller. Waits for a response and handles errors.

        Parameters
        ----------
        z_ctrl_sp : float
            The desired setpoint for the Z-controller in amperes.
        prt : bool, optional
            Whether to print the result (default is if_print).

        Returns
        -------
        pd.DataFrame
            DataFrame containing the set Z-controller setpoint.

        Raises
        ------
        Exception
            If there is an error in communication with the Z-controller.
        """
        z_ctrl_sp = self.tcp.unit_cvt(z_ctrl_sp)
        body = self.tcp.dtype_cvt(z_ctrl_sp, 'float32', 'bin')
        header = self.tcp.header_construct('ZCtrl.SetpntSet', len(body))
        cmd = header + body
        self.tcp.cmd_send(cmd)
        _, _, res_err = self.tcp.res_recv()
        self.tcp.print_err(res_err)
        z_ctrl_sp_df = pd.DataFrame({'Z-controller setpoint (A)': z_ctrl_sp}, index=[0]).T
        if prt: 
            print('\n' + z_ctrl_sp_df.to_string(header=False) + '\n\nZ-controller setpoint set.')
        return z_ctrl_sp_df

    def ZCtrlSetpntGet(self, prt=if_print):
        """
        Get the current Z-controller setpoint.

        Constructs the command to request the Z-controller setpoint and processes the response.

        Parameters
        ----------
        prt : bool, optional
            Whether to print the result (default is if_print).

        Returns
        -------
        pd.DataFrame
            DataFrame containing the current Z-controller setpoint.

        Raises
        ------
        Exception
            If there is an error in communication with the Z-controller.
        """
        header = self.tcp.header_construct('ZCtrl.SetpntGet', 0)
        self.tcp.cmd_send(header)
        _, res_arg, res_err = self.tcp.res_recv('float32')
        self.tcp.print_err(res_err)
        z_ctrl_sp_df = pd.DataFrame({'Z-controller setpoint (A)': res_arg[0]}, index=[0]).T
        if prt: 
            print('\n' + z_ctrl_sp_df.to_string(header=False) + '\n\nZ-controller setpoint returned.')
        return z_ctrl_sp_df

    def ZCtrlGainSet(self, prt=if_print):
        """
        Set the Z-controller gain parameters.

        This method is a placeholder and currently does not implement any functionality.

        Parameters
        ----------
        prt : bool, optional
            Whether to print the result (default is if_print).

        Returns
        -------
        None
        """
        return

    def ZCtrlGainGet(self, prt=if_print):
        """
        Get the Z-controller gain parameters.

        This method is a placeholder and currently does not implement any functionality.

        Parameters
        ----------
        prt : bool, optional
            Whether to print the result (default is if_print).

        Returns
        -------
        None
        """
        return

    def ZCtrlSwitchOffDelaySet(self, prt=if_print):
        """
        Set the delay time for switching off the Z-controller.

        This method is a placeholder and currently does not implement any functionality.

        Parameters
        ----------
        prt : bool, optional
            Whether to print the result (default is if_print).

        Returns
        -------
        None
        """
        return

    def ZCtrlSwitchOffDelayGet(self, prt=if_print):
        """
        Get the delay time for switching off the Z-controller.

        This method is a placeholder and currently does not implement any functionality.

        Parameters
        ----------
        prt : bool, optional
            Whether to print the result (default is if_print).

        Returns
        -------
        None
        """
        return

    def ZCtrlTipLiftSet(self, tiplift, prt=if_print):
        """
        Set the tip lift distance of the Z-controller.

        Converts the lift distance to the appropriate unit, constructs the command,
        and sends it to the Z-controller. Waits for a response and handles errors.

        Parameters
        ----------
        tiplift : float
            The desired tip lift distance in meters.
        prt : bool, optional
            Whether to print the result (default is if_print).

        Returns
        -------
        pd.DataFrame
            DataFrame containing the set tip lift distance.

        Raises
        ------
        Exception
            If there is an error in communication with the Z-controller.
        """
        tiplift = self.tcp.unit_cvt(tiplift)
        body = self.tcp.dtype_cvt(tiplift, 'float32', 'bin')
        header = self.tcp.header_construct('ZCtrl.TipLiftSet', len(body))
        cmd = header + body
        self.tcp.cmd_send(cmd)
        _, _, res_err = self.tcp.res_recv()
        self.tcp.print_err(res_err)
        z_ctrl_tiplift_df = pd.DataFrame({'TipLift (m)': tiplift}, index=[0]).T
        if prt: 
            print('\n' + z_ctrl_tiplift_df.to_string(header=False) + '\n\nZ-controller tiplift set.')
        return z_ctrl_tiplift_df

    def ZCtrlTipLiftGet(self, prt=if_print):
        """
        Get the current tip lift distance of the Z-controller.

        Constructs the command to request the tip lift distance and processes the response.

        Parameters
        ----------
        prt : bool, optional
            Whether to print the result (default is if_print).

        Returns
        -------
        pd.DataFrame
            DataFrame containing the current tip lift distance.

        Raises
        ------
        Exception
            If there is an error in communication with the Z-controller.
        """
        header = self.tcp.header_construct('ZCtrl.TipLiftGet', 0)
        self.tcp.cmd_send(header)
        _, res_arg, res_err = self.tcp.res_recv('float32')
        self.tcp.print_err(res_err)
        z_ctrl_tiplift_df = pd.DataFrame({'TipLift (m)': res_arg[0]}, index=[0]).T
        if prt: 
            print('\n' + z_ctrl_tiplift_df.to_string(header=False) + '\n\nZ-controller tiplift returned.')
        return z_ctrl_tiplift_df

    def ZCtrlHome(self, prt=if_print):
        """
        Home the Z-controller.

        This method is a placeholder and currently does not implement any functionality.

        Parameters
        ----------
        prt : bool, optional
            Whether to print the result (default is if_print).

        Returns
        -------
        None
        """
        return

    def ZCtrlHomePropsSet(self, prt=if_print):
        """
        Set the home properties of the Z-controller.

        This method is a placeholder and currently does not implement any functionality.

        Parameters
        ----------
        prt : bool, optional
            Whether to print the result (default is if_print).

        Returns
        -------
        None
        """
        return

    def ZCtrlHomePropsGet(self, prt=if_print):
        """
        Get the home properties of the Z-controller.

        This method is a placeholder and currently does not implement any functionality.

        Parameters
        ----------
        prt : bool, optional
            Whether to print the result (default is if_print).

        Returns
        -------
        None
        """
        return

    def ZCtrlActiveCtrlSet(self, prt=if_print):
        """
        Set the active control for the Z-controller.

        This method is a placeholder and currently does not implement any functionality.

        Parameters
        ----------
        prt : bool, optional
            Whether to print the result (default is if_print).

        Returns
        -------
        None
        """
        return

    def ZCtrlCtrlListGet(self, prt=if_print):
        """
        Get the control list for the Z-controller.

        This method is a placeholder and currently does not implement any functionality.

        Parameters
        ----------
        prt : bool, optional
            Whether to print the result (default is if_print).

        Returns
        -------
        None
        """
        return

    def ZCtrlWithdraw(self, prt=if_print):
        """
        Withdraw the Z-controller.

        This method is a placeholder and currently does not implement any functionality.

        Parameters
        ----------
        prt : bool, optional
            Whether to print the result (default is if_print).

        Returns
        -------
        None
        """
        return

    def ZCtrlWithdrawRateSet(self, prt=if_print):
        """
        Set the withdraw rate of the Z-controller.

        This method is a placeholder and currently does not implement any functionality.

        Parameters
        ----------
        prt : bool, optional
            Whether to print the result (default is if_print).

        Returns
        -------
        None
        """
        return

    def ZCtrlWithdrawRateGet(self, prt=if_print):
        """
        Get the withdraw rate of the Z-controller.

        This method is a placeholder and currently does not implement any functionality.

        Parameters
        ----------
        prt : bool, optional
            Whether to print the result (default is if_print).

        Returns
        -------
        None
        """
        return

    def ZCtrlLimitsEnabledSet(self, prt=if_print):
        """
        Enable or disable limits for the Z-controller.

        This method is a placeholder and currently does not implement any functionality.

        Parameters
        ----------
        prt : bool, optional
            Whether to print the result (default is if_print).

        Returns
        -------
        None
        """
        return

    def ZCtrlLimitsEnabledGet(self, prt=if_print):
        """
        Get the current status of limits for the Z-controller.

        This method is a placeholder and currently does not implement any functionality.

        Parameters
        ----------
        prt : bool, optional
            Whether to print the result (default is if_print).

        Returns
        -------
        None
        """
        return

    def ZCtrlLimitsSet(self, prt=if_print):
        """
        Set the limits for the Z-controller.

        This method is a placeholder and currently does not implement any functionality.

        Parameters
        ----------
        prt : bool, optional
            Whether to print the result (default is if_print).

        Returns
        -------
        None
        """
        return

    def ZCtrlLimitsGet(self, prt=if_print):
        """
        Get the current limits of the Z-controller.

        This method is a placeholder and currently does not implement any functionality.

        Parameters
        ----------
        prt : bool, optional
            Whether to print the result (default is if_print).

        Returns
        -------
        None
        """
        return

    def ZCtrlStatusGet(self, prt=if_print):
        """
        Get the status of the Z-controller.

        This method is a placeholder and currently does not implement any functionality.

        Parameters
        ----------
        prt : bool, optional
            Whether to print the result (default is if_print).

        Returns
        -------
        None
        """
        return
    ######################################## Piezos module #############################################
    def PiezoDriftCompSet(self,compensation,Vx,Vy,Vz,sat_limit, prt=if_print):
        """
        Configures the drift compensation parameters.
        Arguments:
        - Compensation on/off (int) activates or deactivates the drift compensation, where -1=no change, 0=Off,
        1=On
        - Vx (m/s) (float32) is the linear speed applied to the X piezo to compensate the drift
        - Vy (m/s) (float32) is the linear speed applied to the Y piezo to compensate the drift
        - Vz (m/s) (float32) is the linear speed applied to the Z piezo to compensate the drift
        - Saturation limit (%) (float32) is the drift saturation limit in percent of the full piezo range and it applies to
        all axes
        Return arguments (if Send response back flag is set to True when sending request message):
        - Error described in the Response message>Body section
        """
       # compensation=self.tcp.tristate_cvt_2(compensation)
        Vx = self.tcp.unit_cvt(Vx)
        Vy = self.tcp.unit_cvt(Vy)
        Vz = self.tcp.unit_cvt(Vz)
        sat_limit = self.tcp.unit_cvt(sat_limit)

        body  = self.tcp.dtype_cvt(compensation,'int', 'bin')
        body += self.tcp.dtype_cvt(Vx, 'float32', 'bin')
        body += self.tcp.dtype_cvt(Vy, 'float32', 'bin')
        body += self.tcp.dtype_cvt(Vz, 'float32', 'bin')
        body += self.tcp.dtype_cvt(sat_limit, 'float32', 'bin')
        
        header = self.tcp.header_construct('Piezo.DriftCompSet', len(body))
        cmd = header + body

        self.tcp.cmd_send(cmd)
        _, _, res_err = self.tcp.res_recv()
        self.tcp.print_err(res_err)
     #   if prt: 
    #        print('\n'+
    #            scan_frame_df.to_string(header=False)+
    #            '\n\nScan frame set.')
     #   return scan_frame_df
        
    def PiezoDriftCompGet(self, prt=if_print):
        """
        Configures the drift compensation parameters.
        Arguments:
        None
        Return arguments :
            - Compensation status (unsigned int32) indicates whether the drift compensation is On or Off
           - Vx (m/s) (float32) is the linear speed applied to the X piezo to compensate the drift
           - Vy (m/s) (float32) is the linear speed applied to the Y piezo to compensate the drift
           - Vz (m/s) (float32) is the linear speed applied to the Z piezo to compensate the drift
           - X saturated status (unsigned int32) indicates if the X drift correction reached 10% of the piezo range.
           When this happens, the drift compensation stops for this axis and its LED turns on. To reactivate the
           compensation, switch the drift compensation off and on
           - Y saturated status (unsigned int32) indicates if the Y drift correction reached 10% of the piezo range.
           When this happens, the drift compensation stops for this axis and its LED turns on. To reactivate the
           compensation, switch the drift compensation off and on
           - Z saturated status (unsigned int32) indicates if the Z drift correction reached 10% of the piezo range.
           When this happens, the drift compensation stops for this axis and its LED turns on. To reactivate the
           compensation, switch the drift compensation off and on
           - Saturation limit (%) (float32) is the drift saturation limit in percent of the full piezo range and it applies to
           all axes
           - Error described in the Response message>Body section
            if Send response back flag is set to True when sending request message):
        - Error described in the Response message>Body section
        """
        header = self.tcp.header_construct('Piezo.DriftCompGet', body_size = 0)
    
        self.tcp.cmd_send(header)
        
        _, res_arg, res_err = self.tcp.res_recv('uint32', 'float32','float32','float32', 'uint32', 'uint32', 'uint32')
        self.tcp.print_err(res_err)
    
       # self.tcp.print_err(res_err)
        set_df = pd.DataFrame({'Compensation status': self.tcp.bistate_cvt(res_arg[0]), 
                                 'Vx (m/s)': res_arg[1], 
                                 'Vy (m/s)': res_arg[2], 
                                 'Vz (m/s)': res_arg[3], 
                                 'X saturated status': self.tcp.bistate_cvt(res_arg[4]),
                                 'Y saturated status': self.tcp.bistate_cvt(res_arg[5]), 
                                 'Z saturated status': self.tcp.bistate_cvt(res_arg[6]),  },                            
                                 index=[0]).T
                               #  'Saturation limit (%)': res_arg[7]}, 
    
        if prt: 
            print('\n'+
                set_df.to_string(header=False)+
                '\n\nDrift compensation settings returned returned.')
        return set_df   
    
    def PiezoDriftCompGet_test(self, prt=if_print):
        """
        Configures the drift compensation parameters.
        Arguments:
        None
        Return arguments :
            - Compensation status (unsigned int32) indicates whether the drift compensation is On or Off
           - Vx (m/s) (float32) is the linear speed applied to the X piezo to compensate the drift
           - Vy (m/s) (float32) is the linear speed applied to the Y piezo to compensate the drift
           - Vz (m/s) (float32) is the linear speed applied to the Z piezo to compensate the drift
           - X saturated status (unsigned int32) indicates if the X drift correction reached 10% of the piezo range.
           When this happens, the drift compensation stops for this axis and its LED turns on. To reactivate the
           compensation, switch the drift compensation off and on
           - Y saturated status (unsigned int32) indicates if the Y drift correction reached 10% of the piezo range.
           When this happens, the drift compensation stops for this axis and its LED turns on. To reactivate the
           compensation, switch the drift compensation off and on
           - Z saturated status (unsigned int32) indicates if the Z drift correction reached 10% of the piezo range.
           When this happens, the drift compensation stops for this axis and its LED turns on. To reactivate the
           compensation, switch the drift compensation off and on
           - Saturation limit (%) (float32) is the drift saturation limit in percent of the full piezo range and it applies to
           all axes
           - Error described in the Response message>Body section
            if Send response back flag is set to True when sending request message):
        - Error described in the Response message>Body section
        """
        header = self.tcp.header_construct('Piezo.DriftCompGet', body_size = 0,res=0)
    
        self.tcp.cmd_send(header)
       
        return(self.tcp.sk.recv(self.tcp.buffersize))
    ######################################## Safe Tip Module #############################################
    ######################################## Auto Approach Module #############################################
    ######################################## Scan Module #############################################

    def ScanAction(self, scan_act, scan_dir, prt=if_print):
        """
        Perform a scan action.

        Constructs the command to perform a scan action and sends it to the scanner.
        Waits for a response and handles errors.

        Parameters
        ----------
        scan_act : int
            The scan action to perform (e.g., start, stop).
        scan_dir : int
            The direction of the scan.

        prt : bool, optional
            Whether to print the result (default is if_print).

        Returns
        -------
        pd.DataFrame
            DataFrame containing the scan action and direction.

        Raises
        ------
        Exception
            If there is an error in communication with the scanner.
        """
        body = self.tcp.dtype_cvt(scan_act, 'uint16', 'bin')
        body += self.tcp.dtype_cvt(scan_dir, 'uint32', 'bin')
        header = self.tcp.header_construct('Scan.Action', body_size=len(body))
        cmd = header + body
        self.tcp.cmd_send(cmd)
        _, _, res_err = self.tcp.res_recv()
        self.tcp.print_err(res_err)
        scan_act_df = pd.DataFrame({'Scan action': scan_act, 'Scan direction': scan_dir}, index=[0]).T
        if prt: 
            print('\n' + scan_act_df.to_string(header=False) + '\n\nScan action set.')
        return scan_act_df

    def ScanStatusGet(self, prt=if_print):
        """
        Get the current status of the scan.

        Constructs the command to request the scan status and processes the response.

        Parameters
        ----------
        prt : bool, optional
            Whether to print the result (default is if_print).

        Returns
        -------
        pd.DataFrame
            DataFrame containing the current scan status.

        Raises
        ------
        Exception
            If there is an error in communication with the scanner.
        """
        header = self.tcp.header_construct('Scan.StatusGet', 0)
        self.tcp.cmd_send(header)
        _, res_arg, res_err = self.tcp.res_recv('uint32')
        self.tcp.print_err(res_err)
        scan_status_df = pd.DataFrame({'Scan status': res_arg[0]}, index=[0]).T
        if prt: 
            print('\n' + scan_status_df.to_string(header=False) + '\n\nScan status returned.')
        return scan_status_df
    
    def ScanWaitEndOfScan(self, timeout, prt = if_print):
        timeout = int(self.tcp.unit_cvt(timeout)*1000)

        body  = self.tcp.dtype_cvt(timeout, 'int', 'bin')
        header = self.tcp.header_construct('Scan.WaitEndOfScan', body_size = len(body))
        cmd = header + body

        self.tcp.cmd_send(cmd)
        _, res_arg, res_err = self.tcp.res_recv('uint32', 'uint32', 'str')

        self.tcp.print_err(res_err)
        wait_scan_df = pd.DataFrame({'Timeout status': res_arg[0],
                                     'File path': res_arg[2]},
                                        index=[0]).T
        
        if prt: 
            print('\n'+
                wait_scan_df.to_string(header=False)+
                '\n\nScan status returned.')
        return wait_scan_df 
       
    def ScanFrameSet(self, center_x, center_y, w, h, angle, prt = if_print):
        center_x = self.tcp.unit_cvt(center_x)
        center_y = self.tcp.unit_cvt(center_y)
        w = self.tcp.unit_cvt(w)
        h = self.tcp.unit_cvt(h)
        angle = self.tcp.unit_cvt(angle)

        body  = self.tcp.dtype_cvt(center_x, 'float32', 'bin')
        body += self.tcp.dtype_cvt(center_y, 'float32', 'bin')
        body += self.tcp.dtype_cvt(w, 'float32', 'bin')
        body += self.tcp.dtype_cvt(h, 'float32', 'bin')
        body += self.tcp.dtype_cvt(angle, 'float32', 'bin')
        
        header = self.tcp.header_construct('Scan.FrameSet', len(body))
        cmd = header + body

        self.tcp.cmd_send(cmd)
        _, _, res_err = self.tcp.res_recv()

        self.tcp.print_err(res_err)
        scan_frame_df = pd.DataFrame({'Center X (m)': center_x,
                                      'Center Y (m)': center_y,
                                      'Width (m)': w,
                                      'Height (m)': h,
                                      'Angle (deg)': angle
                                      },
                                      index=[0]).T
        if prt: 
            print('\n'+
                scan_frame_df.to_string(header=False)+
                '\n\nScan frame set.')
        return scan_frame_df

    def ScanFrameGet(self, prt = if_print):
        header = self.tcp.header_construct('Scan.FrameGet', body_size = 0)

        self.tcp.cmd_send(header)
        _, res_arg, res_err = self.tcp.res_recv('float32', 'float32', 'float32', 'float32', 'float32')

        self.tcp.print_err(res_err)
        scan_frame_df = pd.DataFrame({'Center X (m)': res_arg[0],
                                      'Center Y (m)': res_arg[1],
                                      'Width (m)': res_arg[2],
                                      'Height (m)': res_arg[3],
                                      'Angle (deg)': res_arg[4]
                                      },
                                      index=[0]).T
        if prt: 
            print('\n'+
                scan_frame_df.to_string(header=False)+
                '\n\nScan frame settings returned.')
        return scan_frame_df

    def ScanBufferSet(self, num_chs, ch_idx, px, lines, prt = if_print):
        body  = self.tcp.dtype_cvt(num_chs, 'int', 'bin')
        body += self.tcp.dtype_cvt(ch_idx, '1dint', 'bin')
        body += self.tcp.dtype_cvt(px, 'int', 'bin')
        body += self.tcp.dtype_cvt(lines, 'int', 'bin')
        
        header = self.tcp.header_construct('Scan.BufferSet', len(body))
        cmd = header + body

        self.tcp.cmd_send(cmd)
        _, _, res_err = self.tcp.res_recv()

        self.tcp.print_err(res_err)
        scan_buffer_df = pd.DataFrame({'Number of channels': num_chs,
                                       'Channel indexes': [ch_idx],
                                       'Pixels': px,
                                       'Lines': lines
                                      },
                                      index=[0]).T
        if prt: 
            print('\n'+
                scan_buffer_df.to_string(header=False)+
                '\n\nScan buffer set.')
        return scan_buffer_df


    def ScanBufferGet(self, prt = if_print):
        header = self.tcp.header_construct('Scan.BufferGet', body_size = 0)
        
        self.tcp.cmd_send(header)
        _, res_arg, res_err = self.tcp.res_recv('int', '1dint', 'int', 'int')

        self.tcp.print_err(res_err)
        scan_buffer_df = pd.DataFrame({'Number of channels': res_arg[0],
                                      'Channel indexes': [res_arg[1]],
                                      'Pixels': res_arg[2],
                                      'Lines': res_arg[3]
                                      },
                                      index=[0]).T
        if prt: 
            print('\n'+
                scan_buffer_df.to_string(header=False)+
                '\n\nScan buffer settings returned.')
        return scan_buffer_df

    def ScanPropsSet(self, cont_scan, bouncy_scan, autosave, series_name, comment, prt = if_print):
        series_name_size = len(series_name)
        comment_size = len(comment)

        body  = self.tcp.dtype_cvt(cont_scan, 'uint32', 'bin')
        body += self.tcp.dtype_cvt(bouncy_scan, 'uint32', 'bin')
        body += self.tcp.dtype_cvt(autosave, 'uint32', 'bin')
        body += self.tcp.dtype_cvt(series_name_size, 'int', 'bin')
        body += self.tcp.dtype_cvt(series_name, 'str', 'bin')
        body += self.tcp.dtype_cvt(comment_size, 'int', 'bin')
        body += self.tcp.dtype_cvt(comment, 'str', 'bin')
        
        header = self.tcp.header_construct('Scan.PropsSet', len(body))
        cmd = header + body

        self.tcp.cmd_send(cmd)
        _, _, res_err = self.tcp.res_recv()

        self.tcp.print_err(res_err)
        scan_props_df = pd.DataFrame({'Continuous scan': cont_scan,
                                       'Bouncy scan':bouncy_scan,
                                       'Autosave': autosave,
                                       'Series name': series_name,
                                       'Comment': comment
                                      },
                                      index=[0]).T
        if prt: 
            print('\n'+
                scan_props_df.to_string(header=False)+
                '\n\nScan properties set.')
        return scan_props_df
        

    def ScanPropsGet(self, prt = if_print):
        header = self.tcp.header_construct('Scan.PropsGet', body_size = 0)
        
        self.tcp.cmd_send(header)
        _, res_arg, res_err = self.tcp.res_recv('uint32', 'uint32', 'uint32', 'int', 'str', 'int', 'str')

        self.tcp.print_err(res_err)
        scan_props_df = pd.DataFrame({'Continuous scan': res_arg[0],
                                       'Bouncy scan': res_arg[1],
                                       'Autosave': res_arg[2],
                                       'Series name': res_arg[4],
                                       'Comment': res_arg[6]
                                      },
                                      index=[0]).T
        if prt: 
            print('\n'+
                scan_props_df.to_string(header=False)+
                '\n\nScan properties returned.')
        return scan_props_df

    def ScanSpeedSet(self, fwd_li_spd, bwd_li_spd, fwd_t_per_line, bwd_t_per_line, keep_cst, spd_ratio, prt = if_print):
        fwd_li_spd = self.tcp.unit_cvt(fwd_li_spd)
        bwd_li_spd = self.tcp.unit_cvt(bwd_li_spd)
        fwd_t_per_line = self.tcp.unit_cvt(fwd_t_per_line)
        bwd_t_per_line = self.tcp.unit_cvt(bwd_t_per_line)


        body  = self.tcp.dtype_cvt(fwd_li_spd, 'float32', 'bin')
        body += self.tcp.dtype_cvt(bwd_li_spd, 'float32', 'bin')
        body += self.tcp.dtype_cvt(fwd_t_per_line, 'float32', 'bin')
        body += self.tcp.dtype_cvt(bwd_t_per_line, 'float32', 'bin')
        body += self.tcp.dtype_cvt(keep_cst, 'uint16', 'bin')
        body += self.tcp.dtype_cvt(spd_ratio, 'float32', 'bin')
        
        header = self.tcp.header_construct('Scan.SpeedSet', len(body))
        cmd = header + body

        self.tcp.cmd_send(cmd)
        _, _, res_err = self.tcp.res_recv()

        self.tcp.print_err(res_err)
        scan_spd_df = pd.DataFrame({'Forward linear speed (m/s)': fwd_li_spd,
                                    'Backward linear speed (m/s)':bwd_li_spd,
                                    'Forward time per line (s)': fwd_t_per_line,
                                    'Backward time per line (s)': bwd_t_per_line,
                                    'Keep parameter constant': keep_cst,
                                    'Speed ratio': spd_ratio
                                    },
                                    index=[0]).T
        if prt: 
            print('\n'+
              scan_spd_df.to_string(header=False)+
              '\n\nScan speed set.')
        return scan_spd_df

    def ScanSpeedGet(self, prt = if_print):
        header = self.tcp.header_construct('Scan.SpeedGet', body_size = 0)
        
        self.tcp.cmd_send(header)
        _, res_arg, res_err = self.tcp.res_recv('float32', 'float32', 'float32', 'float32', 'uint16', 'float32')

        self.tcp.print_err(res_err)
        scan_spd_df = pd.DataFrame({'Forward linear speed (m/s)': res_arg[0],
                                    'Backward linear speed (m/s)':res_arg[1],
                                    'Forward time per line (s)': res_arg[2],
                                    'Backward time per line (s)': res_arg[3],
                                    'Keep parameter constant': res_arg[4],
                                    'Speed ratio': res_arg[5]
                                    },
                                    index=[0]).T
        if prt: 
            print('\n'+
              scan_spd_df.to_string(header=False)+
              '\n\nScan speed settings returned.')
        return scan_spd_df

    def ScanFrameDataGrab(self, ch_idx, data_dir, prt = if_print):
        body  = self.tcp.dtype_cvt(ch_idx, 'uint32', 'bin')
        body += self.tcp.dtype_cvt(data_dir, 'uint32', 'bin')
        
        header = self.tcp.header_construct('Scan.FrameDataGrab', len(body))
        cmd = header + body

        self.tcp.cmd_send(cmd)
        _, res_arg, res_err = self.tcp.res_recv('int', 'str', 'int', 'int', '2dfloat32', 'uint32')

        self.tcp.print_err(res_err)
        scan_data_df = pd.DataFrame(res_arg[4])


        scan_frame_data_grab_df = pd.DataFrame({'Channels name size': res_arg[0],
                                       'Channel name':res_arg[1],
                                       'Scan data rows': res_arg[2],
                                       'Scan data columns': res_arg[3],
                                       'Scan direction': res_arg[5]
                                      },
                                      index=[0]).T
        if prt: 
            print('\n'+
              scan_frame_data_grab_df.to_string(header=False)+
              '\n\nScan data returned.')
        return scan_data_df, scan_frame_data_grab_df

######################################## Follow Me Module #############################################
    def FolMeXYPosSet(self, x, y, wait_end_of_mv, prt = if_print):
        x = self.tcp.unit_cvt(x)
        y = self.tcp.unit_cvt(y)

        body  = self.tcp.dtype_cvt(x, 'float64', 'bin')
        body += self.tcp.dtype_cvt(y, 'float64', 'bin')
        body += self.tcp.dtype_cvt(wait_end_of_mv, 'uint32', 'bin')
        
        header = self.tcp.header_construct('FolMe.XYPosSet', len(body))
        cmd = header + body

        self.tcp.cmd_send(cmd)
        _, _, res_err = self.tcp.res_recv()

        self.tcp.print_err(res_err)
        folme_xypos_df = pd.DataFrame({'X (m)': x,
                                      'Y (m)': y,
                                      'Wait end of move': wait_end_of_mv
                                      },
                                      index=[0]).T
        if prt: 
            print('\n'+
              folme_xypos_df.to_string(header=False)+
              '\n\nTip coordinates set.')
        return folme_xypos_df

    def FolMeXYPosGet(self, wait_for_new_data, prt = if_print):
        body  = self.tcp.dtype_cvt(wait_for_new_data, 'uint32', 'bin')
        
        header = self.tcp.header_construct('FolMe.XYPosGet', len(body))
        cmd = header + body

        self.tcp.cmd_send(cmd)
        _, res_arg, res_err = self.tcp.res_recv('float64', 'float64')

        self.tcp.print_err(res_err)
        folme_xypos_df = pd.DataFrame({'X (m)': res_arg[0],
                                      'Y (m)': res_arg[1],
                                      'Wait for newest data': wait_for_new_data
                                      },
                                      index=[0]).T
        if prt: 
            print('\n'+
              folme_xypos_df.to_string(header=False)+
              '\n\nTip coordinates returned.')
        return folme_xypos_df

    def FolMeSpeedSet(self, spd, cus_spd, prt = if_print):
        spd = self.tcp.unit_cvt(spd)

        body  = self.tcp.dtype_cvt(spd, 'float32', 'bin')
        body += self.tcp.dtype_cvt(cus_spd, 'uint32', 'bin')
        
        header = self.tcp.header_construct('FolMe.SpeedSet', len(body))
        cmd = header + body

        self.tcp.cmd_send(cmd)
        _, _, res_err = self.tcp.res_recv()

        self.tcp.print_err(res_err)
        speed_df = pd.DataFrame({'Speed (m/s)': spd,
                                       'Custom speed': cus_spd,
                                      },
                                      index=[0]).T
        if prt: 
            print('\n'+
              speed_df.to_string(header=False)+
              '\n\nTip speed set.')
        return speed_df

    def FolMeSpeedGet(self, prt = if_print):
        header = self.tcp.header_construct('FolMe.SpeedGet', 0)

        self.tcp.cmd_send(header)
        _, res_arg, res_err = self.tcp.res_recv('float32', 'uint32')

        self.tcp.print_err(res_err)
        speed_df = pd.DataFrame({'Speed (m/s)': res_arg[0],
                                'Custom speed': res_arg[1],
                                },
                                index=[0]).T
        if prt: 
            print('\n'+
              speed_df.to_string(header=False)+
              '\n\nTip speed returned.')
        return speed_df

    def FolMeOversamplSet(self, oversampling, prt = if_print):
        body  = self.tcp.dtype_cvt(oversampling, 'int', 'bin')
        
        header = self.tcp.header_construct('FolMe.OversamplSet', len(body))
        cmd = header + body

        self.tcp.cmd_send(cmd)
        _, _, res_err = self.tcp.res_recv()

        self.tcp.print_err(res_err)
        oversmp_df = pd.DataFrame({'Oversampling': oversampling,
                                },
                                index=[0]).T
        if prt: 
            print('\n'+
              oversmp_df.to_string(header=False)+
              '\n\nOversamping set.')
        return oversmp_df

    def FolMeOversamplGet(self, prt = if_print):
        header = self.tcp.header_construct('FolMe.OversamplGet', 0)

        self.tcp.cmd_send(header)
        _, res_arg, res_err = self.tcp.res_recv('int', 'float32')

        self.tcp.print_err(res_err)
        oversmp_df = pd.DataFrame({'Oversampling': res_arg[0],
                                   'Sampling rate (Samples/s)': res_arg[1]
                                    },
                                    index=[0]).T
        if prt: 
            print('\n'+
              oversmp_df.to_string(header=False)+
              '\n\nOversamping settings returned.')
        return oversmp_df

    def FolMeStop(self, prt = if_print):
        header = self.tcp.header_construct('FolMe.Stop', 0)

        self.tcp.cmd_send(header)
        _, _, res_err = self.tcp.res_recv()

        self.tcp.print_err(res_err)
        
        if prt: 
            print('\n'+
              '\n\nTip movement stopped.')
        return 

    def FolMePSOnOffGet(self, prt = if_print):
        header = self.tcp.header_construct('FolMe.PSOnOffGet', 0)

        self.tcp.cmd_send(header)
        _, res_arg, res_err = self.tcp.res_recv('uint32')

        self.tcp.print_err(res_err)
        ps01_df = pd.DataFrame({'Point & Shoot status': res_arg[0],
                                },
                                index=[0]).T
        if prt: 
            print('\n'+
              ps01_df.to_string(header=False)+
              '\n\nPoint & Shoot status returned.')
        return ps01_df

    def FolMePSOnOffSet(self, ps_01, prt = if_print):
        body  = self.tcp.dtype_cvt(ps_01, 'uint32', 'bin')

        header = self.tcp.header_construct('FolMe.PSOnOffSet', len(body))
        cmd = header + body

        self.tcp.cmd_send(cmd)
        _, _, res_err = self.tcp.res_recv('uint32')

        self.tcp.print_err(res_err)
        ps01_df = pd.DataFrame({'Point & Shoot status': ps_01,
                                },
                                index=[0]).T
        if prt: 
            print('\n'+
              ps01_df.to_string(header=False)+
              '\n\nPoint & Shoot set.')
        return ps01_df

    def FolMePSExpGet(self, prt = if_print):
        header = self.tcp.header_construct('FolMe.PSExpGet', 0)

        self.tcp.cmd_send(header)
        _, res_arg, res_err = self.tcp.res_recv('uint16', 'int', 'int', '1dstr')

        self.tcp.print_err(res_err)
        ps_exp_df = pd.DataFrame({'Point & Shoot experiment': res_arg[0],
                                'Size of the list of experiments': res_arg[1],
                                'Number of experiments': res_arg[2],
                                'List of experiments': res_arg[3].tolist()
                                },
                                index=[0]).T
        if prt: 
            print('\n'+
              ps_exp_df.to_string(header=False)+
              '\n\nPoint & Shoot experiment returned.')
        return ps_exp_df

    def FolMePSExpSet(self, ps_exp, prt = if_print):
        body  = self.tcp.dtype_cvt(ps_exp, 'uint16', 'bin')

        header = self.tcp.header_construct('FolMe.PSExpSet', len(body))
        cmd = header + body

        self.tcp.cmd_send(cmd)
        _, _, res_err = self.tcp.res_recv()

        self.tcp.print_err(res_err)
        ps_exp_df = pd.DataFrame({'Point & Shoot experiment': ps_exp,
                                },
                                index=[0]).T
        if prt: 
            print('\n'+
              ps_exp_df.to_string(header=False)+
              '\n\nPoint & Shoot experiment set.')
        return ps_exp_df

    def FolMePSPropsGet(self, prt = if_print):
        header = self.tcp.header_construct('FolMe.PSPropsGet', 0)

        self.tcp.cmd_send(header)
        _, res_arg, res_err = self.tcp.res_recv('uint32', 'uint32', 'int', 'str', 'int', 'str', 'float32')

        self.tcp.print_err(res_err)
        ps_props_df = pd.DataFrame({'Auto resume': res_arg[0],
                                    'Use own basename': res_arg[1],
                                    'Basename size': res_arg[2],
                                    'Basename': res_arg[3],
                                    'External VI path size': res_arg[4],
                                    'External VI path':res_arg[5],
                                    'Pre-measure delay (s)': res_arg[6]
                                    },
                                    index=[0]).T
        if prt: 
            print('\n'+
              ps_props_df.to_string(header=False)+
              '\n\nPoint & Shoot properties returned.')
        return ps_props_df

    def FolMePSPropsSet(self, auto_resume, use_own_basename, basename, ext_VI_path_size, ext_VI_path, pre_meas_delay, prt = if_print):
        pre_meas_delay = self.tcp.unit_cvt(pre_meas_delay)
        basename_size = len(basename)

        body  = self.tcp.dtype_cvt(auto_resume, 'uint32', 'bin')
        body += self.tcp.dtype_cvt(use_own_basename, 'uint32', 'bin')
        body += self.tcp.dtype_cvt(basename_size, 'int', 'bin') 
        body += self.tcp.dtype_cvt(basename, 'str', 'bin') 
        body += self.tcp.dtype_cvt(ext_VI_path_size, 'int', 'bin')
        body += self.tcp.dtype_cvt(ext_VI_path, 'str', 'bin')
        body += self.tcp.dtype_cvt(pre_meas_delay, 'float32', 'bin')
        header = self.tcp.header_construct('FolMe.PSPropsSet', len(body))
        cmd = header + body

        self.tcp.cmd_send(cmd)
        _, _, res_err = self.tcp.res_recv()

        self.tcp.print_err(res_err)
        ps_props_df = pd.DataFrame({'Auto resume': auto_resume,
                                         'Use own basename': use_own_basename,
                                         'Basename size': basename_size, 
                                         'Basename': basename, 
                                         'External VI path size': ext_VI_path_size, 
                                         'External VI path': ext_VI_path,
                                         'Pre-measure delay (s)': pre_meas_delay},
                                         index=[0]).T
        
        if prt: 
            print('\n'+
              ps_props_df.to_string(header=False)+
              '\n\nPoint & Shoot properties set.')
        return ps_props_df
######################################## Marks in Scan Module #############################################
    def MarksPointDraw(self, x, y, txt, color, prt = if_print):
        x = self.tcp.unit_cvt(x)
        y = self.tcp.unit_cvt(y)

        txt_size = len(txt)

        color_int = self.tcp.rgb_to_int(color)  

        body  = self.tcp.dtype_cvt(x, 'float32', 'bin')
        body += self.tcp.dtype_cvt(y, 'float32', 'bin')
        body += self.tcp.dtype_cvt(txt_size, 'int', 'bin')
        body += self.tcp.dtype_cvt(txt, 'str', 'bin')
        body += self.tcp.dtype_cvt(color_int, 'uint32', 'bin')
        
        header = self.tcp.header_construct('Marks.PointDraw', len(body))
        cmd = header + body

        self.tcp.cmd_send(cmd)
        _, _, res_err = self.tcp.res_recv()

        self.tcp.print_err(res_err)
        marks_piont_draw_df = pd.DataFrame({'X (m)': x,
                                            'Y (m)': y,
                                            'Text': txt,
                                            'Color': [color]
                                            },
                                            index=[0]).T
        if prt: 
            print('\n'+
              marks_piont_draw_df.to_string(header=False)+
              '\n\nPoint mark is drawn.')
        return marks_piont_draw_df

    def MarksPointsDraw(self, x_1d, y_1d, txt_1d, color_1d, prt = if_print):
        x_1d = [self.tcp.unit_cvt(x) for x in x_1d]
        y_1d = [self.tcp.unit_cvt(y) for y in y_1d]

        num_pts = len(x_1d)
        txt_1d_bin = self.tcp.dtype_cvt(txt_1d, '1dstr', 'bin')

        color_1d_int = [self.tcp.rgb_to_int(color) for color in color_1d]
        
        body  = self.tcp.dtype_cvt(num_pts, 'int', 'bin')
        body += self.tcp.dtype_cvt(x_1d, '1dfloat32', 'bin')
        body += self.tcp.dtype_cvt(y_1d, '1dfloat32', 'bin')
        body += self.tcp.dtype_cvt(len(txt_1d_bin), 'int', 'bin')
        body += txt_1d_bin
        body += self.tcp.dtype_cvt(color_1d_int, '1duint32', 'bin')
        header = self.tcp.header_construct('Marks.PointsDraw', len(body))
        cmd = header + body
        
        self.tcp.cmd_send(cmd)
        
        _, _, res_err = self.tcp.res_recv()

        self.tcp.print_err(res_err)
        marks_points_draw_df = pd.DataFrame({'Number of points': num_pts,
                                            'X (m)': [x_1d],
                                            'Y (m)': [y_1d],
                                            'Text': [txt_1d],
                                            'Color': [color_1d]
                                            },
                                            index=[0]).T
        if prt: 
            print('\n'+
              marks_points_draw_df.to_string(header=False)+
              '\n\nPoint marks are drawn.')
        return marks_points_draw_df

    def MarksLineDraw(self, x_start, y_start, x_end, y_end, color, prt = if_print):
        x_start = self.tcp.unit_cvt(x_start)
        y_start = self.tcp.unit_cvt(y_start)
        x_end = self.tcp.unit_cvt(x_end)
        y_end = self.tcp.unit_cvt(y_end)

        color_int = self.tcp.rgb_to_int(color)  


        body  = self.tcp.dtype_cvt(x_start, 'float32', 'bin')
        body += self.tcp.dtype_cvt(y_start, 'float32', 'bin')
        body += self.tcp.dtype_cvt(x_end, 'float32', 'bin')
        body += self.tcp.dtype_cvt(y_end, 'float32', 'bin')
        body += self.tcp.dtype_cvt(color_int, 'uint32', 'bin')
        
        header = self.tcp.header_construct('Marks.LineDraw', len(body))
        cmd = header + body

        self.tcp.cmd_send(cmd)
        _, _, res_err = self.tcp.res_recv()

        self.tcp.print_err(res_err)
        marks_line_draw_df = pd.DataFrame({'Start point X coordinate (m)': x_start,
                                            'Start point Y coordinate (m)': y_start,
                                            'End point X coordinate (m)': x_end,
                                            'End point Y coordinate (m)': y_end, 
                                            'Color': [color]
                                            },
                                            index=[0]).T
        if prt: 
            print('\n'+
              marks_line_draw_df.to_string(header=False)+
              '\n\nLine mark is drawn.')
        return marks_line_draw_df

    def MarksLinesDraw(self, x_start_1d, y_start_1d, x_end_1d, y_end_1d, color_1d, prt = if_print):
        num_lines = len(x_start_1d)
        x_start_1d = [self.tcp.unit_cvt(x_start) for x_start in x_start_1d]
        y_start_1d = [self.tcp.unit_cvt(y_start) for y_start in y_start_1d]
        x_end_1d = [self.tcp.unit_cvt(x_end) for x_end in x_end_1d]
        y_end_1d = [self.tcp.unit_cvt(y_end) for y_end in y_end_1d]

        color_1d_int = [self.tcp.rgb_to_int(color) for color in color_1d]


        body  = self.tcp.dtype_cvt(num_lines, 'int', 'bin')
        body += self.tcp.dtype_cvt(x_start_1d, '1dfloat32', 'bin')
        body += self.tcp.dtype_cvt(y_start_1d, '1dfloat32', 'bin')
        body += self.tcp.dtype_cvt(x_end_1d, '1dfloat32', 'bin')
        body += self.tcp.dtype_cvt(y_end_1d, '1dfloat32', 'bin')
        body += self.tcp.dtype_cvt(color_1d_int, '1duint32', 'bin')
        
        header = self.tcp.header_construct('Marks.LinesDraw', len(body))
        cmd = header + body

        self.tcp.cmd_send(cmd)
        _, _, res_err = self.tcp.res_recv()

        self.tcp.print_err(res_err)
        marks_lines_draw_df = pd.DataFrame({'Number of lines': num_lines,
                                           'Start point X coordinate (m)': [x_start_1d],
                                           'Start point Y coordinate (m)': [y_start_1d],
                                           'End point X coordinate (m)': [x_end_1d],
                                           'End point Y coordinate (m)': [y_end_1d], 
                                           'Color': [color_1d]
                                            },
                                            index=[0]).T
        if prt: 
            print('\n'+
              marks_lines_draw_df.to_string(header=False)+
              '\n\nLine marks are drawn.')
        return marks_lines_draw_df

    def MarksPointsErase(self, pt_idx, prt = if_print):
        body  = self.tcp.dtype_cvt(pt_idx, 'int', 'bin')

        header = self.tcp.header_construct('Marks.PointsErase', len(body))
        cmd = header + body

        self.tcp.cmd_send(cmd)
        _, _, res_err = self.tcp.res_recv()

        self.tcp.print_err(res_err)
        point_idx_df = pd.DataFrame({'Point index': pt_idx,
                                },
                                index=[0]).T
        if prt: 
            print('\n'+
              point_idx_df.to_string(header=False)+
              '\n\nPoint erased.')
        return point_idx_df

    def MarksLinesErase(self, line_idx, prt = if_print):
        body  = self.tcp.dtype_cvt(line_idx, 'int', 'bin')

        header = self.tcp.header_construct('Marks.LinesErase', len(body))
        cmd = header + body

        self.tcp.cmd_send(cmd)
        _, _, res_err = self.tcp.res_recv()

        self.tcp.print_err(res_err)
        line_idx_df = pd.DataFrame({'Line index': line_idx,
                                },
                                index=[0]).T
        if prt: 
            print('\n'+
              line_idx_df.to_string(header=False)+
              '\n\nLine erased.')
        return line_idx_df

    def MarksPointsVisibleSet(self, pt_idx, visib, prt = if_print):
        body  = self.tcp.dtype_cvt(pt_idx, 'int', 'bin')
        body += self.tcp.dtype_cvt(visib, 'uint16', 'bin')

        header = self.tcp.header_construct('Marks.PointsVisibleSet', len(body))
        cmd = header + body

        self.tcp.cmd_send(cmd)
        _, _, res_err = self.tcp.res_recv()

        self.tcp.print_err(res_err)
        point_visi_df = pd.DataFrame({'Point index': pt_idx,
                                    'Show/hide': visib,
                                },
                                index=[0]).T
        if prt: 
            print('\n'+
              point_visi_df.to_string(header=False)+
              '\n\nPoint visibility set.')
        return point_visi_df

    def MarksLinesVisibleSet(self, line_idx, visib, prt = if_print):
        body  = self.tcp.dtype_cvt(line_idx, 'int', 'bin')
        body += self.tcp.dtype_cvt(visib, 'uint16', 'bin')

        header = self.tcp.header_construct('Marks.LinesVisibleSet', len(body))
        cmd = header + body

        self.tcp.cmd_send(cmd)
        _, _, res_err = self.tcp.res_recv()

        self.tcp.print_err(res_err)
        line_visi_df = pd.DataFrame({'Line index': line_idx,
                                     'Show/hide': visib,
                                    },
                                    index=[0]).T
        if prt: 
            print('\n'+
              line_visi_df.to_string(header=False)+
              '\n\nLine visibility set.')
        return line_visi_df

    def MarksPointsGet(self, prt = if_print):
        header = self.tcp.header_construct('Marks.PointsGet', 0)

        self.tcp.cmd_send(header)
        #! This function is an special case for processing the response from Nanonis. The 'res_recv' function cannot process the command because the second 'int' argument is in the middle of the non-string arrays.
        _, res_arg, res_err = self.tcp.res_recv_MarksPointsGet('int', '1dfloat32', '1dfloat32', 'int', '1dstr', '1duint32', '1duint32')

        self.tcp.print_err(res_err)
        points_df = pd.DataFrame({'Number of points': res_arg[0],
                                 'X coordinate (m)': [res_arg[1]],
                                 'Y coordinate (m)': [res_arg[2]],
                                 'Text': [txt for txt in res_arg[4]],
                                 'Color': [res_arg[5]],
                                 'Visible':[res_arg[6]],
                                 },
                                 index=[0]).T
        if prt: 
            print('\n'+
              points_df.to_string(header=False)+
              '\n\nPoint list returned.')
        return points_df

    def MarksLinesGet(self, prt = if_print):
        header = self.tcp.header_construct('Marks.LinesGet', 0)
        self.tcp.cmd_send(header)
        _, res_arg, res_err = self.tcp.res_recv('int', '1dfloat32', '1dfloat32', '1dfloat32', '1dfloat32', '1duint32', '1duint32')

        self.tcp.print_err(res_err)
        lines_df = pd.DataFrame({'Number of lines': res_arg[0],
                                  'Start point X coordinate (m)': [res_arg[1]],
                                  'Start point Y coordinate (m)': [res_arg[2]],
                                  'End point X coordinate (m)':[res_arg[3]],
                                  'End point Y coordinate (m)':[res_arg[4]],
                                  'Color': [res_arg[5]],
                                  'Visible': [res_arg[6]],
                                  },
                                  index=[0]).T
        if prt: 
            print('\n'+
              lines_df.to_string(header=False)+
              '\n\nLine list returned.')
        return lines_df


######################################## Tip Shaper Module #############################################
    def TipShaperStart(self, wait_until_fin, timeout, prt = if_print):
        timeout = int(self.tcp.unit_cvt(timeout)*1000)

        body  = self.tcp.dtype_cvt(wait_until_fin, 'uint32', 'bin')
        body += self.tcp.dtype_cvt(timeout, 'int', 'bin')
        header = self.tcp.header_construct('TipShaper.Start', body_size = len(body))
        cmd = header + body

        self.tcp.cmd_send(cmd)
        _, _, res_err = self.tcp.res_recv()

        self.tcp.print_err(res_err)
        tipshaper_start_df = pd.DataFrame({'Wait until finished': self.tcp.bistate_cvt(wait_until_fin),
                                           'Timeout (ms)': timeout},
                                           index=[0]).T
        
        if prt: 
            print('\n'+
              tipshaper_start_df.to_string(header=False)+
              '\n\nTip shaping done.')
        return tipshaper_start_df

    def TipShaperPropsSet(self, switch_off_delay, change_bias, bias, tip_lift, lift_t1, bias_lift, bias_settling_t, lift_h, lift_t2, end_wait_t, restore_feedback, prt = if_print):
        switch_off_delay = self.tcp.unit_cvt(switch_off_delay)
        bias = self.tcp.unit_cvt(bias)
        tip_lift = self.tcp.unit_cvt(tip_lift)
        lift_t1 = self.tcp.unit_cvt(lift_t1)
        bias_lift = self.tcp.unit_cvt(bias_lift)
        bias_settling_t = self.tcp.unit_cvt(bias_settling_t)
        lift_h = self.tcp.unit_cvt(lift_h)
        lift_t2 = self.tcp.unit_cvt(lift_t2)
        end_wait_t = self.tcp.unit_cvt(end_wait_t)

        body  = self.tcp.dtype_cvt(switch_off_delay, 'float32', 'bin')
        body += self.tcp.dtype_cvt(change_bias, 'uint32', 'bin')
        body += self.tcp.dtype_cvt(bias, 'float32', 'bin')
        body += self.tcp.dtype_cvt(tip_lift, 'float32', 'bin')
        body += self.tcp.dtype_cvt(lift_t1, 'float32', 'bin')
        body += self.tcp.dtype_cvt(bias_lift, 'float32', 'bin')
        body += self.tcp.dtype_cvt(bias_settling_t, 'float32', 'bin')
        body += self.tcp.dtype_cvt(lift_h, 'float32', 'bin')
        body += self.tcp.dtype_cvt(lift_t2, 'float32', 'bin')
        body += self.tcp.dtype_cvt(end_wait_t, 'float32', 'bin')
        body += self.tcp.dtype_cvt(restore_feedback, 'uint32', 'bin')
        header = self.tcp.header_construct('TipShaper.PropsSet', len(body))
        cmd = header + body

        self.tcp.cmd_send(cmd)
        _, _, res_err = self.tcp.res_recv()

        self.tcp.print_err(res_err)
        tip_shaper_props_df = pd.DataFrame({'Switch off delay (s)': switch_off_delay,
                                            'Change bias': self.tcp.tristate_cvt(change_bias),
                                            'Bias (V)': bias,
                                            'Tip lift (m)': tip_lift,
                                            'Lift time 1 (s)': lift_t1,
                                            'Bias lift (V)': bias_lift,
                                            'Bias settling time (s)': bias_settling_t,
                                            'Lift height (m)': lift_h,
                                            'Lift time 2 (s)': lift_t2,
                                            'End wait time (s)': end_wait_t,
                                            'Restore feedback': self.tcp.tristate_cvt(restore_feedback),
                                            },
                                 index=[0]).T
        if prt: 
            print('\n'+
              tip_shaper_props_df.to_string(header=False)+
              '\n\nTip shaper procedure set.')
        return tip_shaper_props_df

    def TipShaperPropsGet(self, prt = if_print):
        header = self.tcp.header_construct('TipShaper.PropsGet', 0)

        self.tcp.cmd_send(header)
        _, res_arg, res_err = self.tcp.res_recv('float32', 'uint32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'uint32')

        self.tcp.print_err(res_err)
        tip_shaper_props_df = pd.DataFrame({'Switch off delay (s)': res_arg[0],
                                            'Change bias': self.tcp.tristate_cvt(res_arg[1]),
                                            'Bias (V)': res_arg[2],
                                            'Tip lift (m)': res_arg[3],
                                            'Lift time 1 (s)': res_arg[4],
                                            'Bias lift (V)': res_arg[5],
                                            'Bias settling time (s)': res_arg[6],
                                            'Lift height (m)': res_arg[7],
                                            'Lift time 2 (s)': res_arg[8],
                                            'End wait time (s)': res_arg[9],
                                            'Restore feedback': self.tcp.tristate_cvt(res_arg[10]),
                                            },
                                 index=[0]).T
        if prt: 
            print('\n'+
              tip_shaper_props_df.to_string(header=False)+
              '\n\nTip shaper procedure returned.')
        return tip_shaper_props_df

######################################## Generic Sweeper Module #############################################
    def GenSwpAcqChsSet(self, num_chs, ch_idx, prt = if_print):
        body  = self.tcp.dtype_cvt(num_chs, 'int', 'bin')
        body += self.tcp.dtype_cvt(ch_idx, '1dint', 'bin')
        header = self.tcp.header_construct('GenSwp.AcqChsSet', len(body))
        cmd = header + body

        self.tcp.cmd_send(cmd)
        _, _, res_err = self.tcp.res_recv()

        self.tcp.print_err(res_err)
        gen_swp_chs_df = pd.DataFrame({'Number of channels': num_chs,
                                 'Channel indexes': [ch_idx]},
                                 index=[0]).T
        if prt: 
            print('\n'+
              gen_swp_chs_df.to_string(header=False)+
              '\n\nThe recorded channels of the Generic Sweeper set.')
        return gen_swp_chs_df

    def GenSwpAcqChsGet(self, prt = if_print):
        header = self.tcp.header_construct('GenSwp.AcqChsGet', 0)

        self.tcp.cmd_send(header)
        _, res_arg, res_err = self.tcp.res_recv('int', '1dint')

        self.tcp.print_err(res_err)
        gen_swp_chs_df = pd.DataFrame({'Number of channels': res_arg[0],
                                 'Channel indexes': [res_arg[1]]},
                                 index=[0]).T
        if prt: 
            print('\n'+
              gen_swp_chs_df.to_string(header=False)+
              '\n\nThe recorded channels of the Generic Sweeper returned.')
        return gen_swp_chs_df
    
    def GenSwpSwpSignalSet(self, swp_ch_name_size, swp_ch_name, prt = if_print):
        body  = self.tcp.dtype_cvt(swp_ch_name_size, 'int', 'bin')
        body += self.tcp.dtype_cvt(swp_ch_name, 'str', 'bin')
        header = self.tcp.header_construct('GenSwp.SwpSignalSet', len(body))
        cmd = header + body

        self.tcp.cmd_send(cmd)
        _, _, res_err = self.tcp.res_recv()

        self.tcp.print_err(res_err)
        gen_swp_sgn_df = pd.DataFrame({'Sweep channel name size': swp_ch_name_size,
                                 'Sweep channel name': swp_ch_name},
                                 index=[0]).T
        if prt: 
            print('\n'+
              gen_swp_sgn_df.to_string(header=False)+
              '\n\nThe recorded channels of the Generic Sweeper set.')
        return gen_swp_sgn_df

    def GenSwpSwpSignalGet(self, prt = if_print):
        header = self.tcp.header_construct('GenSwp.SwpSignalGet', 0)

        self.tcp.cmd_send(header)
        _, res_arg, res_err = self.tcp.res_recv('int', 'str', 'int', 'int', '1dstr')

        self.tcp.print_err(res_err)
        gen_swp_sgn_df = pd.DataFrame({'Sweep channel name': res_arg[1], 
                                       'Channel names': res_arg[4].tolist()},
                                       index=[0]).T
        if prt: 
            print('\n'+
              gen_swp_sgn_df.to_string(header=False)+
              '\n\nThe recorded channels of the Generic Sweeper returned.')
        return gen_swp_sgn_df
    
    def GenSwpLimitsSet(self, lo_lmt, up_lmt, prt = if_print):
        lo_lmt = self.tcp.unit_cvt(lo_lmt)
        up_lmt = self.tcp.unit_cvt(up_lmt)

        body  = self.tcp.dtype_cvt(lo_lmt, 'float32', 'bin')
        body += self.tcp.dtype_cvt(up_lmt, 'float32', 'bin')
        header = self.tcp.header_construct('GenSwp.LimitsSet', len(body))
        cmd = header + body

        self.tcp.cmd_send(cmd)
        _, _, res_err = self.tcp.res_recv()

        self.tcp.print_err(res_err)
        gen_swp_lmt_df = pd.DataFrame({'Lower limit': lo_lmt,
                                       'Upper limit': up_lmt},
                                       index=[0]).T
        if prt: 
            print('\n'+
              gen_swp_lmt_df.to_string(header=False)+
              '\n\nThe limits of the Sweep signals set.')
        return gen_swp_lmt_df

    def GenSwpLimitsGet(self, prt = if_print):
        header = self.tcp.header_construct('GenSwp.LimitsGet', 0)

        self.tcp.cmd_send(header)
        _, res_arg, res_err = self.tcp.res_recv('float32', 'float32')

        self.tcp.print_err(res_err)
        gen_swp_lmt_df = pd.DataFrame({'Lower limit': res_arg[0],
                                       'Upper limit': res_arg[1]},
                                       index=[0]).T
        if prt: 
            print('\n'+
              gen_swp_lmt_df.to_string(header=False)+
              '\n\nThe limits of the Sweep signals returned.')
        return gen_swp_lmt_df

    # ! -1=no change, 0=Off, 1=On different from other functions!!!
    def GenSwpPropsSet(self, init_settling_t, max_slew_rate, num_steps, periods, autosave, save_dialog, settling_t, prt = if_print):
        init_settling_t = self.tcp.unit_cvt(init_settling_t)*1000
        periods = int(self.tcp.unit_cvt(periods)*1000)
        settling_t = self.tcp.unit_cvt(settling_t)*1000

        body  = self.tcp.dtype_cvt(init_settling_t, 'float32', 'bin')
        body += self.tcp.dtype_cvt(max_slew_rate, 'float32', 'bin')
        body += self.tcp.dtype_cvt(num_steps, 'int', 'bin') #* 0 means no change
        body += self.tcp.dtype_cvt(periods, 'uint16', 'bin') #* 0 means no change
        body += self.tcp.dtype_cvt(autosave, 'int', 'bin')
        body += self.tcp.dtype_cvt(save_dialog, 'int', 'bin')
        body += self.tcp.dtype_cvt(settling_t, 'float32', 'bin')
        header = self.tcp.header_construct('GenSwp.PropsSet', len(body))
        cmd = header + body

        self.tcp.cmd_send(cmd)
        _, _, res_err = self.tcp.res_recv()

        self.tcp.print_err(res_err)
        gen_swp_props_df = pd.DataFrame({'Initial Settling time (ms)': init_settling_t,
                                         'Maximum slew rate (units/s)': max_slew_rate,
                                         'Number of steps': num_steps, 
                                         'Period (ms)': periods, 
                                         'Autosave': self.tcp.tristate_cvt_2(autosave), 
                                         'Save dialog box': self.tcp.tristate_cvt_2(save_dialog),
                                         'Settling time (ms)': settling_t},
                                        index=[0]).T
        
        if prt: 
            print('\n'+
              gen_swp_props_df.to_string(header=False)+
              '\n\nGeneric sweeper parameters set.')
        return gen_swp_props_df

    def GenSwpPropsGet(self, prt = if_print):
        header = self.tcp.header_construct('GenSwp.PropsGet', 0)

        self.tcp.cmd_send(header)
        _, res_arg, res_err = self.tcp.res_recv('float32', 'float32', 'int', 'uint16', 'uint32', 'uint32', 'float32')

        self.tcp.print_err(res_err)
        gen_swp_props_df = pd.DataFrame({'Initial Settling time (ms)': res_arg[0],
                                         'Maximum slew rate (units/s)': res_arg[1],
                                         'Number of steps': res_arg[2], 
                                         'Period (ms)': res_arg[3], 
                                         'Autosave': self.tcp.bistate_cvt(res_arg[4]), 
                                         'Save dialog box': self.tcp.bistate_cvt(res_arg[5]),
                                         'Settling time (ms)': res_arg[6]},
                                 index=[0]).T
        if prt: 
            print('\n'+
              gen_swp_props_df.to_string(header=False)+
              '\n\nGeneric sweeper parameters returned.')
        return gen_swp_props_df

    def GenSwpStart(self, get_data, sweep_dir, save_base_name, reset_signal, prt = if_print):
        save_base_name_str_size = len(save_base_name)

        body  = self.tcp.dtype_cvt(get_data, 'uint32', 'bin')
        body += self.tcp.dtype_cvt(sweep_dir, 'uint32', 'bin')
        body += self.tcp.dtype_cvt(save_base_name_str_size, 'int', 'bin')
        body += self.tcp.dtype_cvt(save_base_name, 'str', 'bin')
        body += self.tcp.dtype_cvt(reset_signal, 'uint32', 'bin')
        header = self.tcp.header_construct('GenSwp.Start', len(body))
        cmd = header + body

        self.tcp.cmd_send(cmd)
        _, res_arg, res_err = self.tcp.res_recv('int', 'int', '1dstr', 'int', 'int', '2dfloat32')

        self.tcp.print_err(res_err)
        gen_swp_df = pd.DataFrame(res_arg[5].T, columns = res_arg[2][0])

        gen_swp_param_df = pd.DataFrame({'Get data': self.tcp.bistate_cvt(get_data),
                                         'Sweep direction': sweep_dir,
                                         'Save base name string size': save_base_name_str_size,
                                         'Save base name string': save_base_name,
                                         'Reset signal': self.tcp.bistate_cvt(reset_signal), 
                                         'Channels names size': res_arg[0],
                                         'Number of channels': res_arg[1],
                                         'Channels names': res_arg[2].tolist(),
                                         'Number of rows': res_arg[3],
                                         'Number of columns': res_arg[4]},
                                        index=[0]).T
        print(gen_swp_df)
        print('\n\n'+
              gen_swp_param_df.to_string(header=False)+
              '\n\nGeneric sweep done!')
        return gen_swp_df, gen_swp_param_df

    def GenSwpStop(self, prt = if_print):
        header = self.tcp.header_construct('GenSwp.Stop', 0)

        self.tcp.cmd_send(header)
        _, _, res_err = self.tcp.res_recv()

        self.tcp.print_err(res_err)

        if prt: 
            print('\n'+
              '\n\nGeneric sweeper stopped.')
        return 

    def GenSwpOpen(self, prt = if_print):
        header = self.tcp.header_construct('GenSwp.Open', 0)

        self.tcp.cmd_send(header)
        _, _, res_err = self.tcp.res_recv()

        self.tcp.print_err(res_err)

        if prt: 
            print('\n'+
              '\n\nGeneric sweep module opened.')
        return 

######################################## Atom Tracking Module #############################################
    def AtomTrackCtrlSet(self, at_ctrl, status, prt = if_print): #Modulation: 0; Controller: 1; Drift measurement:2
        body  = self.tcp.dtype_cvt(at_ctrl, 'uint16', 'bin')
        body += self.tcp.dtype_cvt(status,'uint16', 'bin')
        header = self.tcp.header_construct('AtomTrack.CtrlSet', len(body))
        cmd = header + body

        self.tcp.cmd_send(cmd)
        _, _, res_err = self.tcp.res_recv()

        self.tcp.print_err(res_err)
        at_ctrl_df = pd.DataFrame({'Modulation=0 Controller=1 Drift measurement=2': None,
                                   'Atom tracking control': at_ctrl,
                                   'Status': self.tcp.bistate_cvt(status)},
                                   index=[0]).T
        if prt: 
            print('\n'+
              at_ctrl_df.to_string(header=False)+
              '\n\nAtom tracking control set.')
        return at_ctrl_df

    def AtomTrackStatusGet(self, at_ctrl, prt = if_print):
        body  = self.tcp.dtype_cvt(at_ctrl, 'uint16', 'bin')
        header = self.tcp.header_construct('AtomTrack.CtrlGet', len(body))
        cmd = header + body

        self.tcp.cmd_send(cmd)
        _, res_arg, res_err = self.tcp.res_recv('uint16')

        self.tcp.print_err(res_err)
        at_ctrl_df = pd.DataFrame({'Modulation=0 Controller=1 Drift measurement=2': None,
                                   'Atom tracking control': at_ctrl,
                                   'Status': self.tcp.bistate_cvt(res_arg[0])},
                                   index=[0]).T
        if prt: 
            print('\n'+
              at_ctrl_df.to_string(header=False)+
              '\n\nAtom tracking control status returned.')
        return at_ctrl_df

    def AtomTrackPropsSet(self, inte_gain, freq, ampl, phase, switch_off_delay, prt = if_print):
        inte_gain = self.tcp.unit_cvt(inte_gain)
        freq = self.tcp.unit_cvt(freq)
        ampl = self.tcp.unit_cvt(ampl)
        switch_off_delay= self.tcp.unit_cvt(switch_off_delay)

        body  = self.tcp.dtype_cvt(inte_gain, 'float32', 'bin')
        body += self.tcp.dtype_cvt(freq, 'float32', 'bin')
        body += self.tcp.dtype_cvt(ampl, 'float32', 'bin')
        body += self.tcp.dtype_cvt(phase, 'float32', 'bin')
        body += self.tcp.dtype_cvt(switch_off_delay, 'float32', 'bin')
        header = self.tcp.header_construct('AtomTrack.PropsSet', len(body))
        cmd = header + body

        self.tcp.cmd_send(cmd)
        _, _, res_err = self.tcp.res_recv()
        self.tcp.print_err(res_err)
        at_props_df = pd.DataFrame({'Integral gain (m/(m/m.s))': inte_gain,
                                    'Frequency (Hz)': freq,
                                    'Amplitude (m)': ampl,
                                    'Phase (deg)': phase,
                                    'Switch off delay (s)': switch_off_delay,
                                    },
                                 index=[0]).T
        if prt: 
            print('\n'+
              at_props_df.to_string(header=False)+
              '\n\nAtom track parameters set.')
        return at_props_df

    def AtomTrackPropsGet(self, prt = if_print):
        header = self.tcp.header_construct('AtomTrack.PropsGet', 0)

        self.tcp.cmd_send(header)
        _, res_arg, res_err = self.tcp.res_recv('float32', 'float32', 'float32', 'float32', 'float32')
        self.tcp.print_err(res_err)
        at_props_df = pd.DataFrame({'Integral gain (m/(m/m.s))': res_arg[0],
                                    'Frequency (Hz)': res_arg[1],
                                    'Amplitude (m)': res_arg[2],
                                    'Phase (deg)': res_arg[3],
                                    'Switch off delay (s)': res_arg[4],
                                    },
                                 index=[0]).T
        if prt: 
            print('\n'+
              at_props_df.to_string(header=False)+
              '\n\nAtom track parameters returned.')
        return at_props_df

    def AtomTrackQuickCompStart(self, prt = if_print):
        return

    def AtomTrackDriftComp(self, prt = if_print):
        return

######################################## Lock-in Module #############################################
    def LockInModOnOffSet(self, modu_num, lockin_onoff, prt = if_print):
        body  = self.tcp.dtype_cvt(modu_num, 'int', 'bin')
        body += self.tcp.dtype_cvt(lockin_onoff, 'uint32', 'bin')
        header = self.tcp.header_construct('LockIn.ModOnOffSet', len(body))
        cmd = header + body

        self.tcp.cmd_send(cmd)
        _, _, res_err = self.tcp.res_recv()

        self.tcp.print_err(res_err)
        lockin_onoff_df = pd.DataFrame({'Modulator number': modu_num,
                                        'Lock-in on/off': self.tcp.bistate_cvt(lockin_onoff)},
                                        index=[0]).T
        
        if prt: 
            print('\n'+
              lockin_onoff_df.to_string(header=False)+
              '\n\nLock-in modulator status set.')
        return lockin_onoff_df

    def LockInModOnOffGet(self, modu_num, prt = if_print):
        body  = self.tcp.dtype_cvt(modu_num, 'int', 'bin')
        header = self.tcp.header_construct('LockIn.ModOnOffGet', len(body))
        cmd = header + body

        self.tcp.cmd_send(cmd)
        _, res_arg, res_err = self.tcp.res_recv('uint32')

        self.tcp.print_err(res_err)
        lockin_onoff_df = pd.DataFrame({'Modulator number': modu_num,
                                        'Lock-in on/off': self.tcp.bistate_cvt(res_arg[0])},
                                        index=[0]).T
        
        if prt: 
            print('\n'+
              lockin_onoff_df.to_string(header=False)+
              '\n\nLock-in modulator status returned.')
        return lockin_onoff_df

    # def LockInModSignalSet(self, prt = if_print):
    #     return

    # def LockInModSignalGet(self, prt = if_print):
    #     return

    # def LockInModPhasRegSet(self, prt = if_print):
    #     return

    # def LockInModPhasRegGet(self, prt = if_print):
    #     return

    # def LockInModHarmonicSet(self, prt = if_print):
    #     return

    # def LockInModHarmonicGet(self, prt = if_print):
    #     return

    def LockInModPhasSet(self, modu_num, phase, prt = if_print):
        body  = self.tcp.dtype_cvt(modu_num, 'int', 'bin')
        body += self.tcp.dtype_cvt(phase, 'float32', 'bin')
        header = self.tcp.header_construct('LockIn.ModPhasSet', body_size = len(body))
        cmd = header + body

        self.tcp.cmd_send(cmd)
        _, _, res_err = self.tcp.res_recv()

        self.tcp.print_err(res_err)
        print('currently not used in our system.')
        return

    # def LockInModPhasGet(self, prt = if_print):
    #     return

    def LockInModAmpSet(self, modu_num, ampl, prt = if_print):
        ampl = self.tcp.unit_cvt(ampl)

        body  = self.tcp.dtype_cvt(modu_num, 'int', 'bin')
        body += self.tcp.dtype_cvt(ampl, 'float32', 'bin')
        header = self.tcp.header_construct('LockIn.ModAmpSet', len(body))
        cmd = header + body

        self.tcp.cmd_send(cmd)
        _, _, res_err = self.tcp.res_recv()

        self.tcp.print_err(res_err)
        lockin_amp_df = pd.DataFrame({'Modulator number': modu_num,
                                      'Amplitude': ampl},
                                      index=[0]).T
        
        if prt: 
            print('\n'+
              lockin_amp_df.to_string(header=False)+
              '\n\nLock-in modulator amplitude set.')
        return lockin_amp_df

    def LockInModAmpGet(self, modu_num, prt = if_print):
        body  = self.tcp.dtype_cvt(modu_num, 'int', 'bin')
        header = self.tcp.header_construct('LockIn.ModAmpGet', len(body))
        cmd = header + body

        self.tcp.cmd_send(cmd)
        _, res_arg, res_err = self.tcp.res_recv('float32')

        self.tcp.print_err(res_err)
        lockin_amp_df = pd.DataFrame({'Modulator number': modu_num,
                                      'Amplitude': res_arg[0]},
                                      index=[0]).T
        
        if prt: 
            print('\n'+
              lockin_amp_df.to_string(header=False)+
              '\n\nLock-in modulator amplitude returned.')
        return lockin_amp_df

    def LockInModPhasFreqSet(self, modu_num, freq, prt = if_print):
        freq = self.tcp.unit_cvt(freq)

        body  = self.tcp.dtype_cvt(modu_num, 'int', 'bin')
        body += self.tcp.dtype_cvt(freq, 'float64', 'bin')
        header = self.tcp.header_construct('LockIn.ModPhasFreqSet', len(body))
        cmd = header + body

        self.tcp.cmd_send(cmd)
        _, _, res_err = self.tcp.res_recv()

        self.tcp.print_err(res_err)
        lockin_freq_df = pd.DataFrame({'Modulator number': modu_num,
                                      'Frequency (Hz)': freq},
                                      index=[0]).T
        
        if prt: 
            print('\n'+
              lockin_freq_df.to_string(header=False)+
              '\n\nLock-in modulator frequency set.')
        return lockin_freq_df

    def LockInModPhasFreqGet(self, modu_num, prt = if_print):
        body  = self.tcp.dtype_cvt(modu_num, 'int', 'bin')
        header = self.tcp.header_construct('LockIn.ModPhasFreqGet', len(body))
        cmd = header + body

        self.tcp.cmd_send(cmd)
        _, res_arg, res_err = self.tcp.res_recv('float64')

        self.tcp.print_err(res_err)
        lockin_freq_df = pd.DataFrame({'Modulator number': modu_num,
                                      'Frequency (Hz)': res_arg[0]},
                                      index=[0]).T
        
        if prt: 
            print('\n'+
              lockin_freq_df.to_string(header=False)+
              '\n\nLock-in modulator frequency returned.')
        return lockin_freq_df

    # def LockInDemodSignalSet(self, prt = if_print):
    #     return

    # def LockInDemodSignalGet(self, prt = if_print):
    #     return

    # def LockInDemodHarmonicSet(self, prt = if_print):
    #     return

    # def LockInDemodHarmonicGet(self, prt = if_print):
    #     return

    # def LockInDemodHPFilterSet(self, prt = if_print):
    #     return

    # def LockInDemodHPFilterGet(self, prt = if_print):
    #     return

    # def LockInDemodLPFilterSet(self, prt = if_print):
    #     return

    # def LockInDemodLPFilterGet(self, prt = if_print):
    #     return

    # def LockInDemodPhasRegSet(self, prt = if_print):
    #     return

    # def LockInDemodPhasRegGet(self, prt = if_print):
    #     return

    # def LockInDemodPhasSet(self, prt = if_print):
    #     return

    # def LockInDemodPhasGet(self, prt = if_print):
    #     return

    # def LockInDemodSyncFilterSet(self, prt = if_print):
    #     return

    # def LockInDemodSyncFilterGet(self, prt = if_print):
    #     return

    # def LockInDemodRTSignalsSet(self, prt = if_print):
    #     return

    # def LockInDemodRTSignalsGet(self, prt = if_print):
    #     return
######################################## Signals Module #############################################
    def SignalsNamesGet(self, prt = if_print):
        header = self.tcp.header_construct('Signals.NamesGet', 0)

        self.tcp.cmd_send(header)
        _, res_arg, res_err = self.tcp.res_recv('int', 'int', '1dstr')

        self.tcp.print_err(res_err)
        signal_name_df = pd.DataFrame({'Signal names': res_arg[2].flatten()})
        
        pd.set_option('display.max_rows', None)
        if prt: 
            print('\n'+
                signal_name_df.to_string()+
                '\n\nSignal name list returned.')
        return signal_name_df
    
    def SignalsValGet(self, signal_idx, wait_for_new, prt = if_print):
        body  = self.tcp.dtype_cvt(signal_idx, 'int', 'bin')
        body += self.tcp.dtype_cvt(wait_for_new, 'uint32', 'bin')
        header = self.tcp.header_construct('Signals.ValGet', len(body))
        cmd = header + body

        self.tcp.cmd_send(cmd)
        
        _,res_arg, res_err = self.tcp.res_recv('float32')

        self.tcp.print_err(res_err)
        signal_value_df = pd.DataFrame({'Signal index': signal_idx,'Signal valuee': res_arg[0]},
                                      index=[0]).T
        if prt: 
            print('\n'+
                signal_value_df.to_string()+
                '\n\nSignal value returned.')
        return signal_value_df
    
    def SignalsValsGet(self,signal_idxs, wait_for_new, prt = if_print):
        body  = self.tcp.dtype_cvt(len(signal_idxs), 'int', 'bin')
        body += self.tcp.dtype_cvt(signal_idxs, '1dint', 'bin')
        body += self.tcp.dtype_cvt(wait_for_new, 'uint32', 'bin')
        header = self.tcp.header_construct('Signals.ValsGet', len(body))
        cmd = header + body
    
        self.tcp.cmd_send(cmd)
        
        _, res_arg, res_err = self.tcp.res_recv('int','1dfloat32')
    
        self.tcp.print_err(res_err)
        signal_values_df = pd.DataFrame([signal_idxs, res_arg[1]]).T
        if prt: 
            print('\n'+
                signal_values_df.to_string()+
                '\n\nSignal values returned.')
        return signal_values_df
######################################## Data Logger Module #############################################
    def DataLogOpen(self, prt = if_print):
        header = self.tcp.header_construct('DataLog.Open', 0)

        self.tcp.cmd_send(header)
        _, _, res_err = self.tcp.res_recv()

        self.tcp.print_err(res_err)
        print('Data Logger opened.')
        
    def DataLogStart(self, prt = if_print):
        header = self.tcp.header_construct('DataLog.Start', 0)

        self.tcp.cmd_send(header)
        _, _, res_err = self.tcp.res_recv()

        self.tcp.print_err(res_err)
        print('Data Logger acquisition started.')
        
    def DataLogStop(self, prt = if_print):
        header = self.tcp.header_construct('DataLog.Stop', 0)

        self.tcp.cmd_send(header)
        _, _, res_err = self.tcp.res_recv()

        self.tcp.print_err(res_err)
        print('Data Logger acquisition stopped.')
        
    def DataLogStatusGet(self, prt = if_print):
        header = self.tcp.header_construct('DataLog.StatusGet', body_size = 0)

        self.tcp.cmd_send(header)
        _, res_arg, res_err = self.tcp.res_recv('int', 'str', 'uint16', 'uint16', 'float32', 'int', 'str', 'int', 'str', 'int')

        self.tcp.print_err(res_err)
        datalog_status_df = pd.DataFrame({'Start time': res_arg[1], 
                                 'Acquisition elapsed hours': res_arg[2], 
                                 'Acquisition elapsed minutes': res_arg[3], 
                                 'Acquisition elapsed seconds': res_arg[4], 
                                 'Stop time': res_arg[6],
                                 'Saved file path': res_arg[8], 
                                 'Points counter': res_arg[9],
                                 },
                                 index=[0]).T
        if prt: 
            print('\n'+
              datalog_status_df.to_string(header=False) + 
              '\n\nData Logger status returned.')
        return datalog_status_df
        
    def DataLogChsSet(self, ch_idx, prt = if_print):
        print('To get the signal name and its corresponding index in the list of the 128 available signals in the Nanonis Controller, use the "Signal.NamesGet" function, or check the RT Idx value in the Signals Manager module.')
        num_chs = len(ch_idx)
        body  = self.tcp.dtype_cvt(num_chs, 'int', 'bin')
        body += self.tcp.dtype_cvt(ch_idx, '1dint', 'bin')
        header = self.tcp.header_construct('DataLog.ChsSet', len(body))
        cmd = header + body

        self.tcp.cmd_send(cmd)
        _, _, res_err = self.tcp.res_recv()

        self.tcp.print_err(res_err)
        chs_df = pd.DataFrame({'Number of channels': num_chs,
                               'Channel indexes': [ch_idx]},
                               index=[0]).T

        if prt: 
            print('\n'+
              chs_df.to_string(header=False)+
              '\n\nData Logger channels set.')
        return chs_df
        
    def DataLogChsGet(self, prt = if_print):
        header = self.tcp.header_construct('DataLog.ChsGet', body_size = 0)

        self.tcp.cmd_send(header)
        _, res_arg, res_err = self.tcp.res_recv('int', '1dint')

        self.tcp.print_err(res_err)
        chs_df = pd.DataFrame({'Number of channels': res_arg[0],
                               'Channel indexes': [res_arg[1]]},
                               index=[0]).T

        if prt: 
            print('\n'+
              chs_df.to_string(header=False)+
              '\n\nData Logger channels returned.')
        return chs_df
    
    #! Attention!!!
    def DataLogPropsSet(self, acq_mode, acq_dura_h, acq_dura_m, acq_dura_s, avg, basename, cmt, lst_module, prt = if_print):
        
        
        basename_size = len(basename)
        cmt_size = len(cmt)
        lst_modules_size = sum(len(ele) for ele in lst_module)
        num_modules = len(lst_module)

        body  = self.tcp.dtype_cvt(acq_mode, 'uint16', 'bin')
        body += self.tcp.dtype_cvt(acq_dura_h, 'int', 'bin')
        body += self.tcp.dtype_cvt(acq_dura_m, 'int', 'bin')
        body += self.tcp.dtype_cvt(acq_dura_s, 'float32', 'bin')
        body += self.tcp.dtype_cvt(avg, 'int', 'bin')
        body += self.tcp.dtype_cvt(basename_size, 'int', 'bin')
        body += self.tcp.dtype_cvt(basename, 'str', 'bin')
        body += self.tcp.dtype_cvt(cmt_size, 'int', 'bin')
        body += self.tcp.dtype_cvt(cmt, 'str', 'bin')
        body += self.tcp.dtype_cvt(lst_modules_size, 'int', 'bin')
        body += self.tcp.dtype_cvt(num_modules, 'int', 'bin')
        body += self.tcp.dtype_cvt(lst_module, '1dstr', 'bin')

        header = self.tcp.header_construct('DataLog.PropsSet', len(body))
        cmd = header + body

        self.tcp.cmd_send(cmd)
        _, _, res_err = self.tcp.res_recv()

        self.tcp.print_err(res_err)
        datalog_props_df = pd.DataFrame({'Acquisition mode': acq_mode,
                                      'Acquisition duration (hours)':acq_dura_h,
                                      'Acquisition duration (minutes)': acq_dura_m,
                                      'Acquisition duration (seconds)': acq_dura_s,
                                      'Averaging': avg,
                                      'Basename': basename,
                                      'Comment':cmt,
                                      'Number of modules':num_modules, 
                                      'List of modules': [lst_module],
                                      },
                                      index=[0]).T
        if prt: 
            print('\n'+
              datalog_props_df.to_string(header=False)+
              '\n\nData Logger properties set.')
        return datalog_props_df
        
    def DataLogPropsGet(self, prt = if_print):
        header = self.tcp.header_construct('DataLog.PropsGet', body_size = 0)

        self.tcp.cmd_send(header)
        _, res_arg, res_err = self.tcp.res_recv('uint16', 'int', 'int', 'float32', 'int', 'int', 'str', 'int', 'str')

        self.tcp.print_err(res_err)
        datalog_props_df = pd.DataFrame({'Acquisition mode': res_arg[0],
                               'Acquisition duration (hours)': res_arg[1],
                               'Acquisition duration (minutes)': res_arg[2],
                               'Acquisition duration (seconds)': res_arg[3],
                               'Averaging': res_arg[4],
                               'Basename': res_arg[6],
                               'Comment': res_arg[8],
                               },
                               index=[0]).T

        if prt: 
            print('\n'+
              datalog_props_df.to_string(header=False)+
              '\n\nData Logger properties returned.')
        return datalog_props_df
    
######################################## User output module #############################################

    def UserOutValSet(self, index, value, prt = if_print):
        """
    Sets the value of the selected user output channel.
    Arguments:
    - Output index (int) sets the output to be used, where index could be any value from 1 to the number of
    available outputs
    - Output value (float32) is the value applied to the selected user output in physical units
    Return arguments (if Send response back flag is set to True when sending request message):
    - Error described in the Response message>Body section
        """
        body  = self.tcp.dtype_cvt(index, 'int', 'bin')
        body  += self.tcp.dtype_cvt(value, 'float32', 'bin')
        header = self.tcp.header_construct('UserOut.ValSet', len(body))
        cmd = header + body
        self.tcp.cmd_send(cmd)
        
        
    def UserOutCalibrSet(self, index, cal_per_volt, offset, prt = if_print):
        """
    Sets the calibration of the selected user output or monitor channel.
    Arguments:
    - Output index (int) sets the output to be used, where index could be any value from 1 to the number of
    available outputs
    - Calibration per volt (float32)
    - Offset in physical units (float32)
    Return arguments (if Send response back flag is set to True when sending request message):
    - Error described in the Response message>Body section
        """
        body  = self.tcp.dtype_cvt(index, 'int', 'bin')
        body  += self.tcp.dtype_cvt(cal_per_volt, 'float32', 'bin')
        body  += self.tcp.dtype_cvt(offset, 'float32', 'bin')
        header = self.tcp.header_construct('UserOut.CalibrSet', len(body))
        cmd = header + body
        self.tcp.cmd_send(cmd)       
        _, _, res_err = self.tcp.res_recv()
        self.tcp.print_err(res_err)
        return
        
######################################## PLL Module #############################################   
    def PLLCenterFreqGet(self, prt = if_print):
        """
        Returns the center frequency of the oscillation control module.
        Arguments:
        - Modulator index (int) specifies which modulator or PLL to control. The valid values start from 1
        Return arguments (if Send response back flag is set to True when sending request message):
        - Center frequency (Hz) (float64)
        - Error described in the Response message>Body section
        """
        body  = self.tcp.dtype_cvt(self.mod_index, 'int', 'bin')
        header = self.tcp.header_construct('PLL.CenterFreqGet', len(body))
        cmd = header + body
        self.tcp.cmd_send(cmd)
        
        _, res_arg, res_err = self.tcp.res_recv('float64')
        self.tcp.print_err(res_err)
        df = pd.DataFrame({'Center frequency (Hz)': res_arg[0]},
                                       index=[0]).T
        if prt: 
            print('\n'+
              df.to_string(header=False)+
              '\n\nCenter frequency returned.')
        return df
    
    def PLLCenterFreqSet(self, center_freq, prt = if_print):
        """
        Sets the center frequency of the oscillation control module.
        Arguments:
        - Modulator index (int) specifies which modulator or PLL to control. The valid values start from 1
        - Center frequency (Hz) (float64)
        Return arguments (if Send response back flag is set to True when sending request message):
        - Error described in the Response message>Body section
        """
        body  = self.tcp.dtype_cvt(self.mod_index, 'int', 'bin')
        body  += self.tcp.dtype_cvt(center_freq, 'float64', 'bin')
        header = self.tcp.header_construct('PLL.CenterFreqSet', len(body))
        cmd = header + body
        self.tcp.cmd_send(cmd)
        
        _, _, res_err = self.tcp.res_recv()
        self.tcp.print_err(res_err)
        return
    
    def PLLFreqShiftAutoCenter(self, prt = if_print):
        """
        Auto-centers frequency shift of the oscillation control module.
        It works like the corresponding button on the oscillation control module. It adds the current frequency shift to the
        center frequency and sets the frequency shift to zero.
        Arguments:
        - Modulator index (int) specifies which modulator or PLL to control. The valid values start from 1
        Return arguments (if Send response back flag is set to True when sending request message):
        - Error described in the Response message>Body section
        """
        body  = self.tcp.dtype_cvt(self.mod_index, 'int', 'bin')
        header = self.tcp.header_construct('PLL.FreqShiftAutoCenter', len(body))
        cmd = header + body
        self.tcp.cmd_send(cmd)
        
        _, _, res_err = self.tcp.res_recv()
        self.tcp.print_err(res_err)
        return

######################################## Utilities Module #############################################
    def UtilSessionPathGet(self, prt = if_print):
        header = self.tcp.header_construct('Util.SessionPathGet', 0)

        self.tcp.cmd_send(header)
        _, res_arg, res_err = self.tcp.res_recv('int', 'str')

        self.tcp.print_err(res_err)
        util_session_path_df = pd.DataFrame({'Session path': res_arg[1]},
                                       index=[0]).T
        if prt: 
            print('\n'+
              util_session_path_df.to_string(header=False)+
              '\n\nSession folder path returned.')
        return util_session_path_df
    

    def UtilSessionPathSet(self, sess_path, save_settings_to_prev, prt = if_print):
        sess_path_size = len(sess_path)

        body  = self.tcp.dtype_cvt(sess_path_size, 'int', 'bin')
        body += self.tcp.dtype_cvt(sess_path, 'str', 'bin')
        body += self.tcp.dtype_cvt(save_settings_to_prev, 'uint32', 'bin')
        header = self.tcp.header_construct('Util.SessionPathSet', len(body))
        cmd = header + body

        self.tcp.cmd_send(cmd)
        _, _, res_err = self.tcp.res_recv()

        self.tcp.print_err(res_err)
        util_session_path_df = pd.DataFrame({'Session path': sess_path,
                                             'Save settings to previous': self.tcp.bistate_cvt(save_settings_to_prev)}, 
                                             index=[0]).T
        
        if prt: 
            print('\n'+
              util_session_path_df.to_string(header=False)+
              '\n\nSession folder path set.')
        return util_session_path_df
    

    # def UtilSettingsLoad(self, prt = if_print):

    #     return

    # def UtilSettingsSave(self, prt = if_print):

    #     return

    # def UtilLayoutLoad(self, prt = if_print):

    #     return

    # def UtilLayoutSave(self, prt = if_print):

    #     return

    # def UtilLock(self, prt = if_print):

    #     return

    # def UtilUnLock(self, prt = if_print):

    #     return

    # def UtilRTFreqSet(self, prt = if_print):

    #     return

    # def UtilRTFreqGet(self, prt = if_print):

    #     return

    # def UtilAcqPeriodSet(self, prt = if_print):

    #     return

    # def UtilAcqPeriodGet(self, prt = if_print):

    #     return

    # def UtilRTOversamplSet(self, prt = if_print):

    #     return

    # def UtilRTOversamplGet(self, prt = if_print):

    #     return

    # def UtilQuit(self, prt = if_print):

    #     return
        
                          
        
