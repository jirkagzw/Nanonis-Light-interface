# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 14:57:32 2024

@author: jirka

Updated: Andor TCP connection banner handling and query helpers.
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
    # create a connection between tcp client and Andor TCP server
    # def __init__(self, TCP_IP = '192.168.236.88', PORT = 6501, buffersize = 1048576): # buffer size = 10 kb
    def __init__(self, TCP_IP='localhost', PORT=8888, buffersize=1048576,
                 termination_char='\n', timeout=None, drain_on_connect=True,
                 verbose=False):
        """
        Open TCP connection to the Andor TCP server.

        Parameters
        ----------
        TCP_IP : str
            Andor TCP server IP address.
        PORT : int
            Andor TCP server port.
        buffersize : int
            Receive buffer size.
        termination_char : str
            Command/response termination character. Usually '\n'.
        timeout : float or None
            Socket timeout in seconds. None means infinite/blocking timeout.
            Default is None to keep the old behavior.
        drain_on_connect : bool
            If True, temporarily drains the initial 'OK CON' greeting sent by
            the newer Andor TCP server after connection.
        verbose : bool
            If True, print raw TX/RX strings for debugging.
        """
        self.server_addr = (TCP_IP, PORT)
        self.sk = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sk.settimeout(timeout)  # None = infinite/blocking, old behavior
        self.sk.connect(self.server_addr)
        self.buffersize = buffersize
        self.termination_char = termination_char
        self.verbose = verbose

        # New Andor TCP server sends e.g. 'OK CON \r\n' immediately after
        # connection. Drain it so the first real command does not read it.
        if drain_on_connect:
            drained = self.clear_socket_buffer(timeout=0.2)
            if self.verbose and drained:
                print('ANDOR drained on connect:', repr(drained))

    # close socket
    def socket_close(self):
        self.sk.close()

    def cmd_send(self, data):
        """
        Send command to Andor TCP server.

        Keeps old usage compatible, but avoids adding termination_char twice if
        the caller already included it.
        """
        if isinstance(data, bytes):
            payload = data
        else:
            payload = str(data).encode('utf-8')

        term = self.termination_char.encode('utf-8')
        if not payload.endswith(term):
            payload += term

        if self.verbose:
            print('ANDOR TX:', repr(payload))

        self.sk.sendall(payload)

    def res_recv(self):
        return self.sk.recv(self.buffersize)

    def recv_until(self, termination_char=None, timeout=None):
        """
        Read from the socket until the termination character is found.

        Parameters
        ----------
        termination_char : str or None
            Response terminator. If None, uses self.termination_char.
        timeout : float or None
            Optional temporary timeout only for this read. None keeps the
            current socket timeout, which is blocking/infinite by default.
        """
        if termination_char is None:
            termination_char = self.termination_char

        old_timeout = self.sk.gettimeout()
        if timeout is not None:
            self.sk.settimeout(timeout)

        data = []
        try:
            while True:
                chunk = self.sk.recv(self.buffersize)
                if not chunk:
                    # No more data from socket, connection may be closed
                    break
                text = chunk.decode('utf-8', errors='replace')
                data.append(text)
                if termination_char in text:
                    # Stop reading once the termination character is found
                    break
        finally:
            if timeout is not None:
                self.sk.settimeout(old_timeout)

        result = ''.join(data)
        if self.verbose:
            print('ANDOR RX:', repr(result))
        return result

    def clear_socket_buffer(self, timeout=0.1):
        """
        Clear the receive buffer of the socket.

        This temporarily uses a short timeout, then restores the previous
        timeout. Useful for removing the initial 'OK CON' connection banner.
        """
        old_timeout = self.sk.gettimeout()
        self.sk.settimeout(timeout)

        chunks = []
        try:
            while True:
                try:
                    data = self.sk.recv(self.buffersize)
                    if not data:
                        break
                    chunks.append(data)
                except socket.timeout:
                    break
        finally:
            self.sk.settimeout(old_timeout)

        result = b''.join(chunks).decode('utf-8', errors='replace')
        if self.verbose and result:
            print('ANDOR cleared:', repr(result))
        return result

    def cmd_query(self, data, termination_char=None, timeout=None,
                  ignore_prefixes=('OK CON',), drain_before=False):
        """
        Send a command and read one response.

        Parameters
        ----------
        data : str or bytes
            Command to send. termination_char is appended by cmd_send if needed.
        termination_char : str or None
            Response terminator. None means self.termination_char.
        timeout : float or None
            Optional temporary read timeout. None keeps current/infinite timeout.
        ignore_prefixes : tuple[str]
            Replies starting with these prefixes are ignored. This protects
            against a leftover 'OK CON' connection banner.
        drain_before : bool
            If True, clear the receive buffer just before sending the command.
        """
        if drain_before:
            self.clear_socket_buffer(timeout=0.1)

        self.cmd_send(data)

        while True:
            result = self.recv_until(termination_char=termination_char,
                                     timeout=timeout)
            stripped = result.strip()

            if not stripped:
                return result

            if any(stripped.startswith(prefix) for prefix in ignore_prefixes):
                if self.verbose:
                    print('ANDOR ignored reply:', repr(result))
                continue

            return result

    def cmd_query_slow(self, data, termination_char=None, timeout=None,
                       ignore_prefixes=('OK CON',), drain_before=False):
        """
        Same as cmd_query, intended for slow commands like SWL.

        timeout defaults to None, i.e. infinite/blocking, to keep your current
        requested behavior. You can pass timeout=60 later if desired.
        """
        return self.cmd_query(data, termination_char=termination_char,
                              timeout=timeout,
                              ignore_prefixes=ignore_prefixes,
                              drain_before=drain_before)
