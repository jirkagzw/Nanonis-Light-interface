# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 14:57:32 2024

@author: jirka
"""

# data types: 'str', 'int'(i), 'uint16'(H), 'uint32'(L), 'float32'(f), 'float64'(d), 'hex'
# big-endian encoded '>'
############################### packages ######################################
import socket
from collections import defaultdict
import struct as st
import pandas as pd
import numpy as np

class tcp_andor_ctrl:
############################### functions #####################################
#################### basic functions for creating commands ####################
    # create a connection between tcp client and nanonis software
   # def __init__(self, TCP_IP = '192.168.236.88', PORT = 6501, buffersize = 1048576): # buffer size = 10 kb
    def __init__(self, TCP_IP = 'localhost', PORT = 8888, buffersize = 1048576,termination_char='\n'): # buffer size = 10 kb
        self.server_addr = (TCP_IP, PORT)
        self.sk = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sk.connect(self.server_addr)
        self.buffersize = buffersize
        self.termination_char = termination_char

    # close socket
    def socket_close(self):
        self.sk.close()
        
    def cmd_send(self, data):
        self.sk.sendall((data+self.termination_char).encode('utf-8'))
        
    def res_recv(self):  
        return(self.sk.recv(self.buffersize))
    
    
    def recv_until(self,termination_char='\n'):
        """Read from the socket until the termination character is found."""
        data = []
        while True:
            chunk = self.sk.recv(self.buffersize)
            if not chunk:
                # No more data from socket, connection may be closed
                break
            data.append(chunk.decode('utf-8'))
            if self.termination_char in chunk.decode('utf-8'):
                # Stop reading once the termination character is found
                break
        return ''.join(data)

    
    def clear_socket_buffer(self):
        """Clear the receive buffer of the given socket."""
    # Set a small timeout to avoid blocking indefinitely
        self.sk.settimeout(1.0)
    
        try:
            while True:
                # Attempt to read data from the buffer
                data = self.sk.recv(1024)
                if not data:
                    break  # If no data is received, the buffer is clear
        except self.sk.timeout:
            pass  # Timeout means no more data is available
        finally:
            # Reset the socket timeout to the original setting (e.g., 10 seconds)
            self.sk.settimeout(10.0)