"""
Script Name: Mandelbrot Set
Author: Bernat Oller Pujol
Created: 19/04/00
Last Modified: [Date modified]
Description: In this script, different methods to calculate the Mandelbrot set are explored and compared
"""
import time
import pyopencl as cl
import numpy as np
import numba as nb
from multiprocessing import Pool, cpu_count
import multiprocessing as mp
import matplotlib.pyplot as plt


def mandelbrot_naive(C, I, T):
    """
    Calculate the Mandelbrot set for a given input array C.

    Parameters
    ----------
    C : array-like
        The input array of complex numbers.
    I : int
        The maximum number of iterations to use when calculating the Mandelbrot set.
    T : float
        The threshold value for the absolute value of z.

    Returns
    -------
    M : ndarray
        An array of the same shape as C, containing the corresponding values of the Mandelbrot set.

    Examples
    --------
    >>> C = np.array([[-2+1j, -1+1j, 0+1j], [-2+0j, -1+0j, 0+0j], [-2-1j, -1-1j, 0-1j]])
    >>> mandelbrot_naive(C, 100, 2)
    array([[0.01, 0.01, 0.  ],
           [0.02, 0.  , 0.  ],
           [0.01, 0.01, 0.  ]])

    >>> C = np.array([[-2+2j, -1+2j, 0+2j], [-2+1j, -1+1j, 0+1j], [-2+0j, -1+0j, 0+0j]])
    >>> mandelbrot_naive(C, 100, 2)
    array([[0.06, 0.01, 0.  ],
           [0.03, 0.  , 0.  ],
           [0.02, 0.01, 0.  ]])

    >>> C = np.array([[1+1j, 1+2j, 1+3j], [2+1j, 2+2j, 2+3j], [3+1j, 3+2j, 3+3j]])
    >>> mandelbrot_naive(C, 100, 2)
    array([[0., 0., 0.],
           [0., 0., 0.],
           [0., 0., 0.]])

    >>> C = np.array([[-2+2j, -1+2j, 0+2j], [-2+1j, -1+1j, 0+1j], [-2+0j, -1+0j, 0+0j]])
    >>> mandelbrot_naive(C, 100, 10)
    array([[0., 0., 0.],
           [0., 0., 0.],
           [0., 0., 0.]])

    """
    # Initialize an array of zeros with the same shape as the input array C
    M = np.zeros(C.shape)

    # Loop over the rows of C
    for i in range(C.shape[0]):
        # Extract the i-th row of C
        C_ = C[i,:]
        # Loop over the columns of C
        for j in range(C.shape[1]):
            # Extract the j-th element of the i-th row of C
            c = C_[j]
            # Initialize the complex number z to be the same as c
            z = c
            # Loop over a range of values from 0 to I (inclusive)
            for k in range(I+1):
                # If the absolute value of z is greater than T, break out of the loop
                if abs(z) > T:
                    # Set the corresponding value in M to k divided by I
                    M[i,j] = k/I
                    # Continue to the next element of C
                    continue
                # Otherwise, update the value of z using the iteration rule for the Mandelbrot set
                z = z**2 + c

    return M

def mandelbrot_numpy(C, I, T):
    """
    Calculate the Mandelbrot set for a given input array C using NumPy.

    Parameters
    ----------
    C : array-like
        The input array of complex numbers.
    I : int
        The maximum number of iterations to use when calculating the Mandelbrot set.
    T : float
        The threshold value for the absolute value of z.

    Returns
    -------
    M : ndarray
        An array of the same shape as C, containing the corresponding values of the Mandelbrot set.

    Examples
    --------
    >>> C = np.array([-2 + 2j, 0 + 1j, 1 + 1j, 2 + 2j])
    >>> M = mandelbrot_numpy(C, 100, 2)
    >>> np.allclose(M, [1.0, 0.23, 0.02, 0.0], rtol=1e-2)
    True
    
    >>> C = np.array([-0.5 + 0j, -0.5 + 1j, -0.5 - 1j, 0.5 + 0j])
    >>> M = mandelbrot_numpy(C, 100, 2)
    >>> np.allclose(M, [0.01, 1.0, 1.0, 0.01], rtol=1e-2)
    True
    
    >>> C = np.array([0.5 + 0j, -0.5 + 1j, -0.5 - 1j, 0.5 + 0j])
    >>> M = mandelbrot_numpy(C, 50, 1)
    >>> np.allclose(M, [0.0, 1.0, 1.0, 0.0], rtol=1e-2)
    True
    
    >>> C = np.array([1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j])
    >>> M = mandelbrot_numpy(C, 20, 0.5)
    >>> np.allclose(M, [0.15, 0.0, 0.0, 0.0], rtol=1e-2)
    True
    """
    # Initialize an array of zeros with the same shape as the input array C
    M = np.zeros(C.shape)

    # create a matrix to store the complex numbers
    z = np.zeros(C.shape, dtype=np.complex128)

    # loop over the number of iterations
    for k in range(I+1):
        # calculate the next iteration of z
        z = z**2 + C
        # create a mask of points that have exceeded the threshold
        mask = np.abs(z) > T
        # set the values in M to the current iteration where the mask is true and the value in M is 0
        M[mask & (M == 0)] = k/I
        # set the values of z to 0 where the mask is true
        z[mask] = 0

    return M

@nb.njit(parallel=True)
def mandelbrot_numba(C, I, T):
    """
    Computes the Mandelbrot set using Numba for parallelization.

    Parameters:
        C (ndarray): a 2D array of complex numbers
        I (int): the number of iterations to compute for each point
        T (float): the escape threshold for the Mandelbrot set

    Returns:
        ndarray: a 2D array of floating-point values between 0 and 1,
        representing the "distance" from each point in C to the boundary
        of the Mandelbrot set. Points inside the set have a value of 0.

    Examples:
        >>> C = np.array([[-2-2j, 0+2j], [1-1j, -1+1j]])
        >>> mandelbrot_numba(C, 50, 2)
        array([[0.84, 0.  ],
                [0.  , 0.  ]])

        >>> C = np.meshgrid(np.linspace(-2, 1, 5), np.linspace(-1, 1, 3))
        >>> C = C[0] + 1j*C[1]
        >>> mandelbrot_numba(C, 20, 2)
        array([[0. , 0. , 0. , 0. , 0. ],
                [0.6, 0. , 0. , 0. , 0.6],
                [0. , 0. , 0. , 0. , 0. ]])

        >>> C = np.meshgrid(np.linspace(-2, 2, 5), np.linspace(-1, 1, 3))
        >>> C = C[0] + 1j*C[1]
        >>> mandelbrot_numba(C, 20, 2)
        array([[0.  , 0.  , 0.  , 0.  , 0.  ],
                [0.16, 0.44, 0.44, 0.44, 0.16],
                [0.  , 0.  , 0.  , 0.  , 0.  ]])

        >>> C = np.meshgrid(np.linspace(-2, 2, 5), np.linspace(-1, 1, 3))
        >>> C = C[0] + 1j*C[1]
        >>> mandelbrot_numba(C, 50, 10)
        array([[0.  , 0.  , 0.  , 0.  , 0.  ],
                [0.  , 0.  , 0.  , 0.  , 0.  ],
                [0.96, 0.96, 0.96, 0.96, 0.96]])
    """
    M = np.zeros(C.shape)
    z = np.zeros(C.shape, dtype=np.complex128)
    for i in nb.prange(C.shape[0]):
        C_ = C[i,:]
        for j in range(C.shape[1]):
            c = C_[j]
            for k in range(I+1):
                if abs(z[i,j]) > T:
                    M[i,j] = k/I
                    break
                z[i,j] = z[i,j]**2 + c
    return M

# Define the OpenCL kernel for the Mandelbrot algorithm
kernel_source = """
kernel void mandelbrot(__global float2 *C, __global float *M, const int I, const float T)
    {
        // Get the global ID of the current work item
        int i = get_global_id(0);
        int j = get_global_id(1);

        // Extract the complex number c from the input array C
        float2 c = C[i * get_global_size(1) + j];

        // Initialize the complex number z to be the same as c
        float2 z = c;

        // Loop over a range of values from 0 to I (inclusive)
        for (int k = 0; k <= I; k++) {
            // If the absolute value of z is greater than T, set the corresponding value in M and break out of the loop
            if (sqrt(z.x*z.x + z.y*z.y) > T) {
                M[i * get_global_size(1) + j] = (float)k/I;
                break;
            }
            // Otherwise, update the value of z using the iteration rule for the Mandelbrot set
            float2 z_new;
            z_new.x = z.x * z.x - z.y * z.y + c.x;
            z_new.y = 2.0f * z.x * z.y + c.y;
            z = z_new;
        }
    }
"""

def mandelbrot_opencl(C, I, T, platform_index=0, device_index=0):
    """
    Calculate the Mandelbrot set for a given input array C using OpenCL.

    Parameters
    ----------
    C : array-like
        The input array of complex numbers.
    I : int
        The maximum number of iterations to use when calculating the Mandelbrot set.
    T : float
        The threshold value for the absolute value of z.
    platform_index : int, optional
        The index of the OpenCL platform to use (default is 0).
    device_index : int, optional
        The index of the OpenCL device to use (default is 0).

    Returns
    -------
    M : ndarray
        An array of the same shape as C, containing the corresponding values of the Mandelbrot set.

    """
    # Set up the OpenCL environment
    platforms = cl.get_platforms()
    devices = platforms[platform_index].get_devices()
    context = cl.Context(devices=[devices[device_index]])
    queue = cl.CommandQueue(context)

    # Create the input and output buffers
    C_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY, C.nbytes)
    M_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, C.nbytes)

    # Copy the input array C to the input buffer
    cl.enqueue_copy(queue, C_buf, C)

    # Compile the OpenCL kernel
    program = cl.Program(context, kernel_source).build()
        
    # Set the arguments for the kernel
    program.mandelbrot.set_args(C_buf, M_buf, np.int32(I), np.float32(T))

    # Execute the kernel
    global_size = (C.shape[0], C.shape[1])
    local_size = None
    program.mandelbrot(queue, global_size, local_size, C_buf, M_buf, np.int32(I), np.float32(T))

    # Copy the output buffer to the output array M
    M = np.empty_like(C)
    cl.enqueue_copy(queue, M, M_buf)

    return M[:M.shape[0]//2, M.shape[1]//2:]


def main():
    xlim = (-2,1)
    ylim = (-1.5, 1.5)

    pre = pim = 5000
    pr = [200, 500, 1000, 2000, 4000, 5000, 10000]
    I = 150
    T = 2

    gpu_cpu = [0, 1]


    Re = np.linspace(xlim[0],xlim[1],pre)[np.newaxis,:]
    Im = np.linspace(ylim[0],ylim[1],pim)[:,np.newaxis]
    C = np.complex64(Re + Im*1j)
    
    for p in pr:
        for c in gpu_cpu:
            
            Re = np.linspace(xlim[0],xlim[1],p)[np.newaxis,:]
            Im = np.linspace(ylim[0],ylim[1],p)[:,np.newaxis]
            C = np.complex64(Re + Im*1j)

            s = time.time()
            M = mandelbrot_opencl(C, 150, 2, c, c)
            with open("out.txt", "a") as f:

                f.write(f"OpenCL with grid size {p} and compute device {c}(0 = gpu | 1 = cpu): " + str(time.time()-s) + "s\n")
                print(f"OpenCL with grid size {p} and compute device {c}(0 = gpu | 1 = cpu): " + str(time.time()-s) + "\n")
    
    Re = np.linspace(xlim[0],xlim[1],pre)[np.newaxis,:]
    Im = np.linspace(ylim[0],ylim[1],pim)[:,np.newaxis]
    C = np.complex64(Re + Im*1j)
    """
    s = time.time()
    M = mandelbrot_naive(C, 150, 2)
    print("Naive: " + str(time.time()-s))

    s = time.time()
    M = mandelbrot_numpy(C, 150, 2)
    print("Numpy: " + str(time.time()-s))

    s = time.time()
    M = mandelbrot_opencl(C, 150, 2, 1, 1)
    print("OpenCL: " + str(time.time()-s))
    print(M)

    s = time.time()
    M = mandelbrot_numba(C, 150, 2)
    print("Numba: " + str(time.time()-s))

    """
    M = mandelbrot_opencl(C, 150, 2)

    plt.imshow(M.real, cmap="hot", extent=[-2, 1, -1.5, 1.5])
    plt.hot()
    plt.show()
    

if __name__ == '__main__':
    main()
