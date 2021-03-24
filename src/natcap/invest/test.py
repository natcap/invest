import numpy
import scipy.signal

search_kernel = numpy.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0]
])

# initialize two identical arrays using empty() and full()
a = numpy.empty((50, 50))
a[:] = 10
b = numpy.full((50, 50), 10)
assert numpy.array_equal(a, b)
mask = numpy.full((50, 50), True)

# convolve the array initialized with emtpy()
a_convolved = scipy.signal.convolve(
    a,
    search_kernel,
    mode='same')
a_sum1 = numpy.full(a.shape, -1, dtype=float)
a_sum1[mask] = a_convolved[mask]
a_sum2 = a_convolved
# a_sum1 contains a bunch of 49s (incorrect)
print(numpy.array_equal(a_sum1, a_sum2))  # not equal

# repeat with the supposedly identical array initialized with full()
b_convolved = scipy.signal.convolve(
    b,
    search_kernel,
    mode='same')
b_sum1 = numpy.full(b.shape, -1)
b_sum1[mask] = b_convolved[mask]
b_sum2 = b_convolved
print(numpy.array_equal(b_sum1, b_sum2))  # equal

