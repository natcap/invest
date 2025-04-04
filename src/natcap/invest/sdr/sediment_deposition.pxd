cdef extern from "sediment_deposition.h":
    void run_sediment_deposition[T](
        char*,
        char*,
        char*,
        char*,
        char*) except +
