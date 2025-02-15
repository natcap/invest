cdef extern from "effective_retention.h":
    void run_effective_retention[T](
        char*,
        char*,
        char*,
        char*,
        char*,
        char*) except +
