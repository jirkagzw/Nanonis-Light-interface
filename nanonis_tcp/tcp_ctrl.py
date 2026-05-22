# -*- encoding: utf-8 -*-
'''
@Time    :   2023/03/04 01:54:29
@Author  :   Shixuan Shan 
modified 2026/14/05 GPT
'''

# data types: 'str', 'int'(i), 'uint16'(H), 'uint32'(L), 'float32'(f), 'float64'(d), 'hex'
# big-endian encoded '>'
############################### packages ######################################
import socket
import threading
from collections import defaultdict
import struct as st
import pandas as pd
import numpy as np

class tcp_ctrl:
############################### functions #####################################
#################### basic functions for creating commands ####################
    # create a connection between tcp client and nanonis software
    def __init__(self, TCP_IP = '127.0.0.1', PORT = 6501, buffersize=50*1024*1024, version=999999): # buffer size = 50 MB enough for tip recorder 200k samples of 62 channels 
        """
       Parameters
       IP              : Listening IP address
       PORT            : Listening Port (check Nanonis File>Settings>TCP)
       max_buf_size    : maximum size of the response message. just make it big
       version         : Nanonis version. See Nanonis > help > info and take the RT Engine number.
                         Defaults to the latest version of Nanonis 
       """
        self.server_addr = (TCP_IP, PORT)
        self.sk = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sk.connect(self.server_addr)
        self.buffersize = buffersize
        self.version = version
        # Protect one TCP command/response transaction from interleaving with another thread.
        self._io_lock = threading.RLock()

    # close socket
    def socket_close(self):
        self.sk.close()

    # data type conversion. 
    '''
       - the arguments returned by this function are bytelike strings when converting to 'bin' and data in requested format and the length of the data when converting from 'bin'
       - arg is for 1d or 2d array conversions. 
            for 2d arrays: arg should be a tuple in the form of (num_rows, num_cols), eg. (41, 2)
            for 1d string array: put (1, size) instead of the size directly
            for a string: arg should be a integer
    '''
    def dtype_cvt(self, data, original_fmt, target_fmt, *arg):  
        self.supported_dtypes = ['bin', 'str', 'int', 'uint16', 'uint32', 'float32', 'float64',
                                 '1dstr', '1dint', '1duint32', '1dfloat32',
                                 '2dstr', '2dfloat32']
        
        ##############* TO BYTES ####################
        #* str to binary
        if original_fmt == 'str' and target_fmt == 'bin': 
            return np.array(data, '>S').tobytes()
        #* 1dstr to binary
        elif original_fmt == '1dstr' and target_fmt == 'bin': 
            str_array_in_bytes = b''
            for ele in data:
                # Each element of the array must be preceded by its size in bytes in 'int' format
                ele_bytes_size = np.array(len(ele), '>i').tobytes()
                ele_bytes = np.array(ele, '>S').tobytes()

                str_array_in_bytes += ele_bytes_size
                str_array_in_bytes += ele_bytes
            return str_array_in_bytes        
        #* int32 to binary
        elif original_fmt in ['int', '1dint'] and target_fmt == 'bin': 
            return np.array(data, '>i').tobytes()
        #* unsigned int16 to binary
        elif original_fmt == 'uint16' and target_fmt == 'bin': 
            return np.array(data, '>H').tobytes()
        #* unsigned int32 to binary
        elif original_fmt in ['uint32', '1duint32'] and target_fmt == 'bin': 
            return np.array(data, '>L').tobytes()
        #* float32 to binary
        elif original_fmt in ['float32', '1dfloat32', '2dfloat32'] and target_fmt == 'bin': 
            return np.array(data, '>f').tobytes()
        #* float64 to binary   
        elif original_fmt == 'float64' and target_fmt == 'bin': 
            return np.array(data, '>d').tobytes()
        
        #############* FROM BYTES ####################
        #* binary to string (expression after '%' gives the size of the string in bytes)
        #* for string, no need to put 'int' before 'str'
        elif original_fmt == 'bin' and target_fmt == 'str': 
            data_cvted = np.frombuffer(data, '>%dS' % arg)[0].decode('iso-8859-1') 
            return [data_cvted, len(data)]
        #* binary to 1d or 2d string
        elif original_fmt == 'bin' and target_fmt in ['1dstr', '2dstr']: 
            ele_idx = 0
            str_array= []
            for idx in range(np.prod(arg)):
                ele_size = np.frombuffer(data[ele_idx: ele_idx+4], '>i')[0]
                ele_idx += 4

                ele = np.frombuffer(data[ele_idx: ele_idx + ele_size], '>%dS' % ele_size)[0].decode('utf-8')
                ele_idx += ele_size

                str_array.append(ele)
            return [np.array(str_array).reshape(arg), len(data)]
        #* binary to int & 1d int
        elif original_fmt == 'bin' and target_fmt in ['int', '1dint']: 
            data_cvted = np.frombuffer(data, '>i')
            if len(data_cvted) == 1:
                data_cvted = data_cvted[0]
            return [data_cvted, len(data)] 
        #* binary to unsigned int16 
        elif original_fmt == 'bin' and target_fmt == 'uint16': 
            data_cvted = np.frombuffer(data, '>H')
            if len(data_cvted) == 1:
                data_cvted = data_cvted[0]
            return [data_cvted, len(data)]
        #* binary to unsigned int32 and 1d unsigned int32
        elif original_fmt == 'bin' and target_fmt in ['uint32', '1duint32']: 
            data_cvted = np.frombuffer(data, '>L')
            if len(data_cvted) == 1:
                data_cvted = data_cvted[0]
            return [data_cvted, len(data)]
        #* binary to float32 and 1d float32
        elif original_fmt == 'bin' and target_fmt in ['float32', '1dfloat32']: 
            data_cvted = np.frombuffer(data, '>f')
            if len(data_cvted) == 1:
                data_cvted = data_cvted[0]
            return [data_cvted, len(data)]
        #* binary to 2d float32
        elif original_fmt == 'bin' and target_fmt == '2dfloat32': 
            return [np.frombuffer(data, '>f').reshape(arg), len(data)]
        #* binary to float64
        elif original_fmt == 'bin' and target_fmt in ['float64', '1dfloat64']:
            data_cvted = np.frombuffer(data, '>d')
            if len(data_cvted) == 1:
                data_cvted = data_cvted[0] 
            return [data_cvted, len(data)]


    # unit conversion function
    def unit_cvt(self, data):
        unit_list = ['m', 'u', 'n', 'p', 'f']
        unit_conv ={
                    'm': 1e-3,
                    'u': 1e-6,
                    'n': 1e-9,
                    'p': 1e-12,
                    'f': 1e-15
                    }
        if type(data) == str:
            if data[-1] in unit_list:
                significand = float(''.join(char for char in data if char.isdigit() or char in ['.', '-']))
                return significand*unit_conv[data[-1]]
            elif data.isdigit():
                return float(data)
            else:
                print('An error occured! Please check if the input unit is one of the following: "m", "u", "n", "p", "f"')
        else:
            return data

    # construct header
    def header_construct(self,command_name, body_size, res = True):
        self.header_bin_rep =  bytes(command_name, 'utf-8').ljust(32, b'\x00')  # convert command name to binary representation and pad it to 32 bytes long with b'\x00'
        self.header_bin_rep += self.dtype_cvt(body_size, 'int', 'bin')         # boty size
        self.header_bin_rep += self.dtype_cvt(1 if res else 0, 'uint16', 'bin') # send response back (1) or not (0)
        self.header_bin_rep += b'\x00\x00'
        return self.header_bin_rep

    # send command to nanonis tcp server
    def cmd_send(self, data):
        self.sk.sendall(data)

    # receive and decode response message
        '''
        supported argument formats (arg_fmt) are: 
            'str', 'int', 'uint16', 'uint32', 'float32', 'float64', 
            '1dstr', '1dint', '1duint8'(not supported now), '1duint32', 
            '1dfloat32', '1dfloat64', '2dfloat32', '2dstr'
        '''

    def _recv_exact(self, nbytes):
        """Receive exactly nbytes from the TCP stream."""
        chunks = []
        remaining = int(nbytes)
        while remaining > 0:
            chunk = self.sk.recv(remaining)
            if chunk == b'':
                raise ConnectionError(
                    f"TCP socket closed while waiting for {remaining} more bytes "
                    f"of a {nbytes}-byte response."
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        return b''.join(chunks)

    def _recv_response_exact(self, debug=False):
        """Read one full Nanonis response: 40-byte header + declared body."""
        header = self._recv_exact(40)
        body_size = int(np.frombuffer(header[32:36], '>i')[0])
        if body_size < 0:
            raise ValueError(f"Invalid negative response body size: {body_size}")
        body = self._recv_exact(body_size)
        if debug:
            cmd = self.dtype_cvt(header[0:32], 'bin', 'str', 32)[0].replace('\x00', '')
            print(f"Nanonis response: command={cmd!r}, body_size={body_size}, received_body={len(body)}")
        return header + body

    def drain_socket(self, timeout_s=0.02, max_bytes=10_000_000):
        """
        Best-effort emergency drain for stale bytes in the socket.

        Use only before starting a new clean experiment, not between cmd_send() and
        res_recv(), because it will discard unread response bytes.
        """
        old_timeout = self.sk.gettimeout()
        self.sk.settimeout(timeout_s)
        drained = 0
        try:
            while drained < max_bytes:
                chunk = self.sk.recv(min(65536, max_bytes - drained))
                if not chunk:
                    break
                drained += len(chunk)
        except socket.timeout:
            pass
        finally:
            self.sk.settimeout(old_timeout)
        return drained

    def res_recv_MarksPointsGet(self, *varg_fmt, get_header = True, get_arg = True, get_err = True):
        res_bin_rep = self.sk.recv(self.buffersize)
        
        res_arg = []
        res_err = pd.DataFrame()
        res_header = pd.DataFrame()

        num_pts = None

        # parse the header of a response message
        if get_header:
            res_header['commmand name'] = self.dtype_cvt(res_bin_rep[0:32], 'bin', 'str', 32) # drop all '\x00' in the string
            res_header['body size'] = self.dtype_cvt(res_bin_rep[32:36], 'bin', 'int')  
        # parse the arguments values of a response message
        if get_arg:
            arg_byte_idx = 40   
            arg_size_dict = {'int': 4,'uint16': 2,'uint32': 4,'float32': 4,'float64': 8}
            for idx, arg_fmt in enumerate(varg_fmt):
                if arg_fmt == 'int':
                    arg, arg_size = self.dtype_cvt(res_bin_rep[arg_byte_idx: arg_byte_idx + arg_size_dict[arg_fmt]], 'bin', arg_fmt)
                    arg_byte_idx += arg_size
                    res_arg.append(arg)
                    if num_pts == None:
                        num_pts = arg

                elif arg_fmt == '1dstr':
                    num_rows = 1
                    num_cols = num_pts
                    
                    # calculate the total size of the string array
                    interal_byte_idx = arg_byte_idx
                    for ele_idx in range(num_rows*num_cols): 
                        ele_size, len_int = self.dtype_cvt(res_bin_rep[interal_byte_idx: interal_byte_idx + arg_size_dict['int']], 'bin', 'int')
                        interal_byte_idx += ele_size + len_int
                    array_size = interal_byte_idx - arg_byte_idx

                    arg, arg_size = self.dtype_cvt(res_bin_rep[arg_byte_idx: arg_byte_idx + array_size], 'bin', arg_fmt, num_rows, num_cols)
                    arg_byte_idx += arg_size

                    if array_size == arg_size:
                        res_arg.append(arg)
                    else:
                        print('There might be an error when parsing the string array. Possible causes could be: \n 1) the argument format (arg_fmt) input is wrong. \n 2) the previous arg_fmt is wrong. ' )
                        res_arg.append(arg)

                elif arg_fmt in ['1dint', '1duint32', '1dfloat32', '1dfloat64']:
                    num_rows = 1
                    num_cols = num_pts

                    array_size = num_rows * num_cols * arg_size_dict[arg_fmt[2:]]
                    arg, arg_size = self.dtype_cvt(res_bin_rep[arg_byte_idx: arg_byte_idx + array_size], 'bin', arg_fmt, num_rows, num_cols)
                    arg_byte_idx += arg_size
                    res_arg.append(arg)
                else: 
                    raise TypeError('Please check the data types! Supported data types are: \
                                    "bin", "str", "int", "uint16", "uint32", "float32", "float64", \
                                    "1dstr", "1dint", "1duint8" (currently unavailable), "1duint32", "1dfloat32", "1dfloat64", \
                                    "2dfloat32", "2dstr"')
                
            res_bin_rep = res_bin_rep[arg_byte_idx-1:] # for parsing the error in a request or a response

            # parse the error of a response message
            if get_err:
                res_err['error status'] = [self.dtype_cvt(res_bin_rep[0:4], 'bin', 'uint32')[0]] # error status
                res_err['error body size'] = [self.dtype_cvt(res_bin_rep[4:8], 'bin', 'int')[0]]# error description size
                res_err['error description'] = [self.dtype_cvt(res_bin_rep[8:], 'bin', 'str', len(res_bin_rep[8:]))[0]] # error description
            return res_header, res_arg, res_err

    def res_recv(self, *varg_fmt, get_header=True, get_arg=True, get_err=True, debug=False):
        """
        Receive and decode one complete Nanonis response.

        Fixes two failure modes of the older implementation:
        1) TCP is a byte stream, so one recv(buffersize) is not guaranteed to
           contain the whole response. We now read the 40-byte header and then
           exactly the body size declared in that header.
        2) The error block starts immediately after the decoded arguments. The
           old code used arg_byte_idx-1, shifting the error parser by one byte.
        """
        res_bin_rep = self._recv_response_exact(debug=debug)

        res_arg = []
        res_err = pd.DataFrame()
        res_header = pd.DataFrame()

        if get_header:
            res_header['commmand name'] = self.dtype_cvt(res_bin_rep[0:32], 'bin', 'str', 32)
            res_header['body size'] = self.dtype_cvt(res_bin_rep[32:36], 'bin', 'int')

        arg_byte_idx = 40
        arg_size_dict = {'int': 4, 'uint16': 2, 'uint32': 4, 'float32': 4, 'float64': 8}

        if get_arg:
            for idx, arg_fmt in enumerate(varg_fmt):
                if arg_fmt in arg_size_dict:
                    arg, arg_size = self.dtype_cvt(
                        res_bin_rep[arg_byte_idx:arg_byte_idx + arg_size_dict[arg_fmt]],
                        'bin', arg_fmt
                    )
                    arg_byte_idx += arg_size
                    res_arg.append(arg)

                elif arg_fmt == 'str':
                    str_size = int(res_arg[idx - 1])
                    if str_size != 0:
                        arg, _ = self.dtype_cvt(
                            res_bin_rep[arg_byte_idx:arg_byte_idx + str_size],
                            'bin', arg_fmt, str_size
                        )
                    else:
                        arg = 'EmptyString'
                    arg_byte_idx += str_size
                    res_arg.append(arg)

                elif arg_fmt in ['1dstr', '2dstr']:
                    num_rows = int(res_arg[idx - 2]) if arg_fmt == '2dstr' else 1
                    num_cols = int(res_arg[idx - 1])

                    internal_byte_idx = arg_byte_idx
                    for _ in range(num_rows * num_cols):
                        ele_size, len_int = self.dtype_cvt(
                            res_bin_rep[internal_byte_idx:internal_byte_idx + arg_size_dict['int']],
                            'bin', 'int'
                        )
                        internal_byte_idx += int(ele_size) + len_int
                    array_size = internal_byte_idx - arg_byte_idx

                    arg, arg_size = self.dtype_cvt(
                        res_bin_rep[arg_byte_idx:arg_byte_idx + array_size],
                        'bin', arg_fmt, num_rows, num_cols
                    )
                    arg_byte_idx += arg_size
                    res_arg.append(arg)

                elif arg_fmt in ['1dint', '1duint32', '1dfloat32', '1dfloat64', '2dfloat32']:
                    if arg_fmt == '2dfloat32':
                        # Nanonis 2D arrays are preceded by rows and columns.
                        num_rows = int(res_arg[idx - 2])
                        num_cols = int(res_arg[idx - 1])
                    else:
                        num_rows = 1
                        if varg_fmt[idx - 1] == 'int':
                            num_cols = int(res_arg[idx - 1])
                        else:
                            # Fallback for arrays whose length is defined by the latest int.
                            last_int_pos = len(varg_fmt) - 1 - varg_fmt[::-1].index('int')
                            num_cols = int(res_arg[last_int_pos])

                    base_fmt = arg_fmt[2:]
                    array_size = num_rows * num_cols * arg_size_dict[base_fmt]
                    arg, arg_size = self.dtype_cvt(
                        res_bin_rep[arg_byte_idx:arg_byte_idx + array_size],
                        'bin', arg_fmt, num_rows, num_cols
                    )
                    arg_byte_idx += arg_size
                    res_arg.append(arg)

                else:
                    raise TypeError(
                        'Please check the data types! Supported data types are: '
                        '"bin", "str", "int", "uint16", "uint32", "float32", "float64", '
                        '"1dstr", "1dint", "1duint32", "1dfloat32", "1dfloat64", '
                        '"2dfloat32", "2dstr"'
                    )

        if get_err:
            # Error block is uint32 status + int32 description size + description.
            # It starts exactly after the decoded arguments; no -1 offset.
            err_start = arg_byte_idx
            body_end = 40 + int(res_header['body size'][1] if False else np.frombuffer(res_bin_rep[32:36], '>i')[0])
            if err_start + 8 > len(res_bin_rep):
                raise ValueError(
                    f"Response ended before error block: err_start={err_start}, total={len(res_bin_rep)}"
                )
            err_status = self.dtype_cvt(res_bin_rep[err_start:err_start + 4], 'bin', 'uint32')[0]
            err_body_size = self.dtype_cvt(res_bin_rep[err_start + 4:err_start + 8], 'bin', 'int')[0]
            err_body_size = int(err_body_size)
            err_bytes = res_bin_rep[err_start + 8:err_start + 8 + max(err_body_size, 0)]
            if err_body_size > 0:
                err_description = self.dtype_cvt(err_bytes, 'bin', 'str', len(err_bytes))[0]
            else:
                err_description = ''

            res_err['error status'] = [err_status]
            res_err['error body size'] = [err_body_size]
            res_err['error description'] = [err_description]

            if debug:
                expected_total = err_start + 8 + max(err_body_size, 0)
                print(f"Decoded args end={arg_byte_idx}, error_size={err_body_size}, "
                      f"expected_total={expected_total}, actual_total={len(res_bin_rep)}")

        return res_header, res_arg, res_err

    def print_err(self, res_err):
        if not res_err.loc[0, 'error body size'] == 0:
            print(res_err.loc[0, 'error description'])

    def tristate_cvt(self, status):
        if status == 0:
            return 'No change'
        elif status == 1:
            return 'Yes/On'
        elif status == 2:
            return 'No/Off'
        elif status == 'No change':
            return 0
        elif status == 'Yes/On':
            return 1
        elif status == 'No/Off':
            return 2
        else:
            print("Error: A valid input should be either in [0, 1, 2] or in ['No change', 'Yes/On', 'No/Off']")
            return status
        
    def tristate_cvt_2(self, status):
        if status == -1:
            return 'No Change'
        elif status == 0:
            return 'Off/No'
        elif status == 1:
            return 'On/Yes'
        elif status == 'No Change':
            return -1
        elif status == 'Off/No':
            return 0
        elif status == 'On/Yes':
            return 1
        else:
            print("Error: A valid input should be either in [-1, 0, 1] or in ['No change', 'Yes/On', 'No/Off']")
            return status
        
    def bistate_cvt(self, status):
        if status == 0:
            return 'False/Off'
        elif status == 1:
            return 'True/On'
        elif status == 'False/Off':
            return 0
        elif status == 'True/On':
            return 1
        else:
            print(str(status)+" sent, Error: A valid input should be either in [0, 1] or in ['False/Off', ''True/On].")
            return status     
        
        
    def tri_bi_cvt(self,status):
        if status == 0:
            return 'Off/No'
        elif status == 1:
            return 'Yes/On'
        elif status == 'False/Off':
            return 2
        elif status == 'True/On':
            return 1
        else:
            print("Error: A valid input should be either in [0, 1] or in ['False/Off', 'True/On']")
            return status
        
    def rgb_to_int(self, rgb_lst):
        # Make sure the color components are within the valid range (0-255)
        r = max(0, min(255, rgb_lst[0]))
        g = max(0, min(255, rgb_lst[1]))
        b = max(0, min(255, rgb_lst[2]))
        
        # Combine the color components using bitwise left shifts
        color_int = (r << 16) + (g << 8) + b
        
        return color_int
