import sys

class Registrar(object):
    def __init__(self):
        object.__init__(self)
        self.map = {}

    def update_map(self, updates):
        self.map.update(updates)

    def eval(self, mapKey, opValues):
        try:
            return self.map[mapKey](opValues)
        except KeyError: #key not in self.map
            return None
        except ValueError as e:
            #This handles the case where a type is a numeric value but doens't cast 
            #correctly.  In that case what is the value of an empty string?  Perhaps
            #it should be NaN?  Here we're returning 0.  James and Rich arbitrarily
            #decided this on 5/16/2012, there's no other good reason.
            if mapKey in ['int', 'float']:
                return 0
            else:
                # Actually print out the exception information.
                raise sys.exc_info()[1], None, sys.exc_info()[2]

    def get_func(self, mapKey):
        return self.map[mapKey]

class DatatypeRegistrar(Registrar):
    def __init__(self):
        Registrar.__init__(self)

        updates = {'int': int,
                   'float': float,
                   'boolean': bool}
        self.update_map(updates)

    def eval(self, mapKey, opValues):
        cast_value = Registrar.eval(self, mapKey, opValues)

        if cast_value == None:
            return str(opValues)

        return cast_value
