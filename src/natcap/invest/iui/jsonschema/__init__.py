#!/usr/bin/env python
#:coding=utf-8:
#:tabSize=2:indentSize=2:noTabs=true:
#:folding=explicit:collapseFolds=1:

'''
A complete, full-featured validator for JSON Schema

JSON Schema validation is based on the specifications of the the 
JSON Schema Proposal Second Draft
(http://groups.google.com/group/json-schema/web/json-schema-proposal---second-draft).

jsonschema provides an API similar to simplejson in that validators can be
overridden to support special property support or extended functionality.

Parsing a simple JSON document

>>> import jsonschema
>>> jsonschema.validate("simplejson", {"type":"string"})

Parsing a more complex JSON document.

>>> import simplejson
>>> import jsonschema
>>> 
>>> data = simplejson.loads('["foo", {"bar":["baz", null, 1.0, 2]}]')
>>> schema = {
...   "type":"array", 
...   "items":[
...     {"type":"string"},
...     {"type":"object",
...      "properties":{
...        "bar":{
...          "items":[
...            {"type":"string"},
...            {"type":"any"},
...            {"type":"number"},
...            {"type":"integer"}
...          ]
...        }
...      }
...    }
...   ]
... }
>>> jsonschema.validate(data,schema)

Handling validation errors
ValueErrors are thrown when validation errors occur.

>>> import jsonschema
>>> try:
...     jsonschema.validate("simplejson", {"type":"string","minLength":15})
... except ValueError, e:
...     print e.message
... 
Length of value 'simplejson' for field '_data' must be more than or equal to 15.000000

Running from the command line

% echo '{"type":"string"}' > schema.json
% echo '"mystring"' | python -mjsonschema schema.json

% echo '"mystring"' > data.json
% python -mjsonschema schema.json data.json

'''

#TODO: Line numbers for error messages
#TODO: Add checks to make sure the schema itself is valid
#TODO: Support command line validation kind of like how simplejson allows 
#      encoding using the "python -m<modulename>" format.
#TODO: Support encodings other than utf-8

from jsonschema.validator import JSONSchemaValidator

__all__ = [ 'validate', 'JSONSchemaValidator' ]
__version__ = '0.1a'

def validate(data, schema, validator_cls=None, interactive_mode=True):
  '''
  Validates a parsed json document against the provided schema. If an
  error is found a ValueError is raised.
  
  ``data`` is a python dictionary object of parsed json data.
  
  ``schema`` is a python dictionary object of the parsed json schema.
  
  If ``validator_cls`` is provided that class will be used to validate
  the given ``schema`` against the given ``data``. The given class should
  be a subclass of the JSONSchemaValidator class.
  
  ``interactive_mode`` is a boolean value specifying if the data should
  be validated in interactive mode. Validating in interactive mode will
  allow the validator to make changes to the given json ``data`` object
  to put in place default values specified in the given ``schema``
  object.
  '''
  if validator_cls == None:
    validator_cls = JSONSchemaValidator
  v = validator_cls(interactive_mode)
  return v.validate(data,schema)

if __name__ == '__main__':
  import sys, simplejson
  if len(sys.argv) == 1:
    raise SystemExit("%s SCHEMAFILE [INFILE]" % (sys.argv[0],))
  elif len(sys.argv) == 2:
    if sys.argv[1] == "--help":
      raise SystemExit("%s SCHEMAFILE [INFILE]" % (sys.argv[0],))
    schemafile = open(sys.argv[1], 'rb')
    infile = sys.stdin
  elif len(sys.argv) == 3:
    schemafile = open(sys.argv[1], 'rb')
    infile = open(sys.argv[2], 'rb')
  else:
    raise SystemExit("%s SCHEMAFILE [INFILE]" % (sys.argv[0],))
  try:
    obj = simplejson.load(infile)
    schema = simplejson.load(schemafile)
    validate(obj, schema)
  except ValueError, e:
    raise SystemExit(e)
