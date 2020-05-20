#cython: language_level=3

"""
llh cython:
cython accelerated send/recv functions for use with the LLH service and client
"""


__author__ = "Aaron Fienberg"

import numpy as np

cimport numpy as np

from libc.string cimport memcpy

from zmq cimport libzmq, Socket

import zmq

cdef inline int check_zmq_error(int retcode) except -1:
    cdef errno
    cdef const char* errstr
    if retcode < 0:
        errno = libzmq.zmq_errno()
        if errno == libzmq.ZMQ_EAGAIN:
            raise zmq.error.Again
        else:
            errstr = libzmq.zmq_strerror(errno)
            raise zmq.error.ZMQError(errno=errno,
                                     msg=f'ZMQ error {errno}: {errstr.decode()}')

    return 0

cdef inline int send_error_check(void* sock, libzmq.zmq_msg_t *msg, int flags) except -1:
    '''
    convenience function. Sends message, checks errors, and then closes the msg
    '''
    try:
        check_zmq_error(libzmq.zmq_msg_send(msg, sock, flags))
    finally:
        # is closing necessary? It's in the python example,
        # but the zmq guide suggests it's not necessary after sending
        libzmq.zmq_msg_close(msg)

    return 0

cpdef dispatch_replies(Socket sock, list work_reqs, 
                       np.ndarray[np.float32_t, ndim=1, mode='c'] llhs):
    '''
    dispatch replies from the LLH service to the clients
    '''

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

    for work_req in work_reqs:
        header_frames = work_req["header_frames"]
        for header_frame in header_frames:
            data = header_frame
            size = len(header_frame)
            
            libzmq.zmq_msg_init_size(&msg, size)
            memcpy(libzmq.zmq_msg_data(&msg), data, size)
            send_error_check(c_sock, &msg, libzmq.ZMQ_SNDMORE)
            
        start_ind = work_req["start_ind"]
        stop_ind = work_req["stop_ind"]
        size = (stop_ind - start_ind)*sizeof(np.float32_t)
        start_byte = start_ind*sizeof(np.float32_t)
        
        libzmq.zmq_msg_init_size(&msg, size)
        memcpy(libzmq.zmq_msg_data(&msg), llh_bytes + start_byte, size)
        send_error_check(c_sock, &msg, 0)
        
cpdef dispatch_request(Socket sock,
                       bytes req_id,  
                       np.ndarray[np.float32_t, ndim=1, mode='c'] x,
                       np.ndarray[np.float32_t, ndim=1, mode='c'] theta):
    '''
    dispatch a request from the LLH client to the service
    '''

    cdef void* c_sock = sock.handle
    cdef const char* req_id_bytes = req_id
    cdef const char* x_bytes = <const char *>&x[0] 
    cdef const char* theta_bytes = <const char *>&theta[0] 

    cdef libzmq.zmq_msg_t msg
    cdef int size

    # send header frames
    size = len(req_id)
    libzmq.zmq_msg_init_size(&msg, size)
    memcpy(libzmq.zmq_msg_data(&msg), req_id_bytes, size)
    send_error_check(c_sock, &msg, libzmq.ZMQ_SNDMORE)
    
    # send x
    size = len(x)*sizeof(np.float32_t)
    libzmq.zmq_msg_init_size(&msg, size)
    memcpy(libzmq.zmq_msg_data(&msg), x_bytes, size)
    send_error_check(c_sock, &msg, libzmq.ZMQ_SNDMORE)
    
    # send theta
    size = len(theta)*sizeof(np.float32_t)
    libzmq.zmq_msg_init_size(&msg, size)
    memcpy(libzmq.zmq_msg_data(&msg), theta_bytes, size)
    send_error_check(c_sock, &msg, 0)


cpdef receive_req(Socket sock):
    '''
    receive a multipart request
    to be used in the llh_service
    '''
    cdef void* c_sock = sock.handle

    cdef list outframes = []
    cdef bytes frame

    cdef libzmq.zmq_msg_t part
    cdef int ret
    cdef int errno
    cdef int more
    cdef size_t size
    cdef const char* data
    cdef const char* errstr
    while True:
        libzmq.zmq_msg_init(&part)
        try:
            check_zmq_error(libzmq.zmq_msg_recv(&part, c_sock, libzmq.ZMQ_DONTWAIT))
        except:
            libzmq.zmq_msg_close(&part)
            raise
            
        size = libzmq.zmq_msg_size(&part)
        data = <const char*>libzmq.zmq_msg_data(&part)
        frame = data[:size]
        outframes.append(frame)
        
        more = libzmq.zmq_msg_more(&part)
        libzmq.zmq_msg_close(&part)
        if more == 0:
            break

    return outframes



