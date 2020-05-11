#cython: language_level=3

import numpy as np

cimport numpy as np

from libc.string cimport memcpy

from zmq cimport libzmq, Socket

import zmq

cpdef dispatch_replies(Socket sock, list work_reqs, 
                       np.ndarray[np.float32_t, ndim=1, mode='c'] llhs):
    cdef void* c_sock = sock.handle

    cdef const char* llh_bytes = <const char *>&llhs[0] 

    cdef libzmq.zmq_msg_t msg
    
    # declare types for all loop variables
    cdef dict work_req
    cdef list header_frames
    cdef bytes header_frame
    cdef const char* data
    cdef int size
    cdef int start_ind
    cdef int stop_ind
    cdef int start_byte
    cdef int errno

    for work_req in work_reqs:
        header_frames = work_req["header_frames"]
        for header_frame in header_frames:
            data = header_frame
            size = len(header_frame)
            
            libzmq.zmq_msg_init_size(&msg, size)
            memcpy(libzmq.zmq_msg_data(&msg), data, size)
            libzmq.zmq_msg_send(&msg, c_sock, libzmq.ZMQ_SNDMORE)
            libzmq.zmq_msg_close(&msg)

        start_ind = work_req["start_ind"]
        stop_ind = work_req["stop_ind"]
        size = (stop_ind - start_ind)*sizeof(np.float32_t)
        start_byte = start_ind*sizeof(np.float32_t)
        
        libzmq.zmq_msg_init_size(&msg, size)
        memcpy(libzmq.zmq_msg_data(&msg), llh_bytes + start_byte, size)
        libzmq.zmq_msg_send(&msg, c_sock, 0)
        libzmq.zmq_msg_close(&msg)
