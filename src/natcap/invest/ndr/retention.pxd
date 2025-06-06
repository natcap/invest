cdef extern from "retention.h":
    void calculate_retention[T](
        char*,
        char*,
        char*,
        char*,
        char*,
        char*) except +
