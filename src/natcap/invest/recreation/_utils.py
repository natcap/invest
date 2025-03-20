from io import BytesIO

import numpy


def _numpy_dumps(numpy_array):
    """Safely pickle numpy array to string.

    Args:
        numpy_array (numpy.ndarray): arbitrary numpy array.
    Returns:
        A string representation of the array that can be loaded using
        `numpy_loads`.
    """
    with BytesIO() as file_stream:
        numpy.save(file_stream, numpy_array, allow_pickle=False)
        return file_stream.getvalue()


def _numpy_loads(queue_string):
    """Safely unpickle string to numpy array.

    Args:
        queue_string (str): binary string representing a pickled
            numpy array.
    Returns:
        A numpy representation of ``binary_numpy_string``.
    """
    with BytesIO(queue_string) as file_stream:
        return numpy.load(file_stream)
