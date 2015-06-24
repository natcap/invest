'''
Affine Matrix Class
'''


class Affine(object):
    # any global variables here
    def __init__(self, a, b, c, d, e, f):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f

    def __repr__(self):
        rep = "\n| %f %f %f |" % (self.a, self.b, self.c)
        rep += "\n| %f %f %f |\n" % (self.d, self.e, self.f)
        return rep

    def __eq__(self, other):
        a = (self.a == other.a)
        b = (self.b == other.b)
        c = (self.c == other.c)
        d = (self.d == other.d)
        e = (self.e == other.e)
        f = (self.f == other.f)
        if all([a, b, c, d, e, f]):
            return True
        else:
            return False

    @classmethod
    def identity(self):
        return Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)

    @ classmethod
    def from_gdal(self, c, a, b, f, d, e):
        return Affine(a, b, c, d, e, f)

    def to_gdal(self):
        return (self.c, self.a, self.b, self.f, self.d, self.e)
