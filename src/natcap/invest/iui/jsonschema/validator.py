#!/usr/bin/env python
#:coding=utf-8:
#:tabSize=2:indentSize=2:noTabs=true:
#:folding=explicit:collapseFolds=1:

#TODO: Support references
#TODO: Support inline schema

import types, sys, re, copy

class JSONSchemaValidator:
  '''
  Implementation of the json-schema validator that adheres to the 
  JSON Schema Proposal 2nd Draft.
  '''
  
  # Map of schema types to their equivalent in the python types module
  _typesmap = {
    "string": [types.StringType, types.UnicodeType],
    "integer": types.IntType,
    "number": [types.IntType, types.FloatType],
    "boolean": types.BooleanType,
    "object": types.DictType,
    "array": types.ListType,
    "null": types.NoneType,
    "any": None
  }
  
  # Default schema property values.
  _schemadefault = {
    "id": None,
    "type": None,
    "properties": None,
    "items": None,
    "optional": False,
    "additionalProperties": None,
    "requires": None,
    "identity": None,
    "minimum": None,
    "maximum": None,
    "minItems": None,
    "maxItems": None,
    "pattern": None,
    "maxLength": None,
    "minLength": None,
    "enum": None,
    "options": None,
    "readonly": None,
    "title": None,
    "description": None,
    "format": None,
    "default": None,
    "transient": None,
    "maxDecimal": None,
    "hidden": None,
    "disallow": None,
    "extends": None
  }
  
  _refmap = {}
  
  _interactive_mode = True
  
  def __init__(self, interactive_mode=True):
    self._interactive_mode = interactive_mode
  
  def validate_id(self, x, fieldname, schema, ID=None):
    '''
    Validates a schema id and adds it to the schema reference map
    '''
    if ID is not None:
      if ID == "$":
        raise ValueError("Reference id for field '%s' cannot equal '$'" % fieldname)
      self._refmap[ID] = schema
    return x
  
  def validate_type(self, x, fieldname, schema, fieldtype=None):
    '''
    Validates that the fieldtype specified is correct for the given
    data
    '''
    
    converted_fieldtype = self._convert_type(fieldtype)
    
    # We need to know if the field exists or if it's just Null
    fieldexists = True
    try:
      value = x[fieldname]
    except KeyError:
      fieldexists = False
    finally:
      value = x.get(fieldname)
    
    if converted_fieldtype is not None and fieldexists:
      if type(converted_fieldtype) == types.ListType:
        # Match if type matches any one of the types in the list
        datavalid = False
        for eachtype in converted_fieldtype:
          try:
            self.validate_type(x, fieldname, eachtype, eachtype)
            datavalid = True
            break
          except ValueError:
            pass
        if not datavalid:
          raise ValueError("Value %r for field '%s' is not of type %r" % (value, fieldname, fieldtype))
      elif type(converted_fieldtype) == types.DictType:
        try:
          self.__validate(fieldname, x, converted_fieldtype)
        except ValueError,e:
          raise e
      else:
        if type(value) != converted_fieldtype:
          raise ValueError("Value %r for field '%s' is not of type %r" % (value, fieldname, fieldtype))
    return x
  
  def validate_properties(self, x, fieldname, schema, properties=None):
    '''
    Validates properties of a JSON object by processing the object's
    schema recursively
    '''
    if properties is not None and x.get(fieldname) is not None:
      value = x.get(fieldname)
      if value is not None:
        if type(value) == types.DictType:
          if type(properties) == types.DictType:
            for eachProp in properties.keys():
              self.__validate(eachProp, value, properties.get(eachProp))
          else:
            raise ValueError("Properties definition of field '%s' is not an object" % fieldname)
    return x
  
  def validate_items(self, x, fieldname, schema, items=None):
    '''
    Validates that all items in the list for the given field match the
    given schema
    '''
    if items is not None and x.get(fieldname) is not None:
      value = x.get(fieldname)
      if value is not None:
        if type(value) == types.ListType:
          if type(items) == types.ListType:
            if len(items) == len(value):
              for itemIndex in range(len(items)):
                try:
                  self.validate(value[itemIndex], items[itemIndex])
                except ValueError, e:
                  raise ValueError("Failed to validate field '%s' list schema: %r" % (fieldname, e.message))
            else:
              raise ValueError("Length of list %r for field '%s' is not equal to length of schema list" % (value, fieldname))
          elif type(items) == types.DictType:
            for eachItem in value:
                try:
                  self._validate(eachItem, items)
                except ValueError, e:
                  raise ValueError("Failed to validate field '%s' list schema: %r" % (fieldname, e.message))
          else:
            raise ValueError("Properties definition of field '%s' is not a list or an object" % fieldname)
    return x
  
  def validate_optional(self, x, fieldname, schema, optional=False):
    '''
    Validates that the given field is present if optional is false
    '''
    # Make sure the field is present
    if fieldname not in x.keys() and not optional:
      raise ValueError("Required field '%s' is missing" % fieldname)
    return x
  
  def validate_additionalProperties(self, x, fieldname, schema, additionalProperties=None):
    '''
    Validates additional properties of a JSON object that were not
    specifically defined by the properties property
    '''
    if additionalProperties is not None:
      # If additionalProperties is the boolean value True then we accept any
      # additional properties.
      if type(additionalProperties) == types.BooleanType and additionalProperties == True:
        return x
      
      value = x.get(fieldname)
      if type(additionalProperties) == types.DictType \
       or type(additionalProperties) == types.BooleanType:
        properties = schema.get("properties")
        if properties is None:
          properties = {}
        for eachProperty in value.keys():
          if eachProperty not in properties:
            # If additionalProperties is the boolean value False then we 
            # don't accept any additional properties.
            if type(additionalProperties) == types.BooleanType and additionalProperties == False:
              raise ValueError("Additional properties not defined by 'properties' are not allowed in field '%s'" % fieldname)
            self.__validate(eachProperty, value, additionalProperties)
      else:
        raise ValueError("additionalProperties schema definition for field '%s' is not an object" % fieldname)
    return x
  
  def validate_requires(self, x, fieldname, schema, requires=None):
    if x.get(fieldname) is not None and requires is not None:
      if x.get(requires) is None:
        raise ValueError("Field '%s' is required by field '%s'" % (requires, fieldname))
    return x
  
  def validate_identity(self, x, fieldname, schema, unique=False):
    return x
  
  def validate_minimum(self, x, fieldname, schema, minimum=None):
    '''
    Validates that the field is longer than or equal to the minimum
    length if specified
    '''
    if minimum is not None and x.get(fieldname) is not None:
      value = x.get(fieldname)
      if value is not None:
        if type(value) in (types.IntType,types.FloatType) and value < minimum:
          raise ValueError("Value %r for field '%s' is less than minimum value: %f" % (value, fieldname, minimum))
        elif type(value) == types.ListType and len(value) < minimum:
          raise ValueError("Value %r for field '%s' has fewer values than the minimum: %f" % (value, fieldname, minimum))
    return x
  
  def validate_maximum(self, x, fieldname, schema, maximum=None):
    '''
    Validates that the field is shorter than or equal to the maximum
    length if specified.
    '''
    if maximum is not None and x.get(fieldname) is not None:
      value = x.get(fieldname)
      if value is not None:
        if type(value) in (types.IntType, types.FloatType) and value > maximum:
          raise ValueError("Value %r for field '%s' is greater than maximum value: %f" % (value, fieldname, maximum))
        elif type(value) == types.ListType and len(value) > maximum:
          raise ValueError("Value %r for field '%s' has more values than the maximum: %f" % (value, fieldname, maximum))
    return x
  
  def validate_minItems(self, x, fieldname, schema, minitems=None):
    '''
    Validates that the number of items in the given field is equal to or
    more than the minimum amount.
    '''
    if minitems is not None and x.get(fieldname) is not None:
      value = x.get(fieldname)
      if value is not None:
        if type(value) == types.ListType and len(value) < minitems:
          raise ValueError("Value %r for field '%s' must have a minimum of %d items" % (fieldname, fieldname, minitems))
    return x
  
  def validate_maxItems(self, x, fieldname, schema, maxitems=None):
    '''
    Validates that the number of items in the given field is equal to or
    less than the maximum amount.
    '''
    if maxitems is not None and x.get(fieldname) is not None:
      value = x.get(fieldname)
      if value is not None:
        if type(value) == types.ListType and len(value) > maxitems:
          raise ValueError("Value %r for field '%s' must have a maximum of %d items" % (value, fieldname, maxitems))
    return x
  
  def validate_pattern(self, x, fieldname, schema, pattern=None):
    '''
    Validates that the given field, if a string, matches the given
    regular expression.
    '''
    value = x.get(fieldname)
    if pattern is not None and \
       value is not None and \
       self._is_string_type(value):
      p = re.compile(pattern)
      if not p.match(value):
        raise ValueError("Value %r for field '%s' does not match regular expression '%s'" % (value, fieldname, pattern))
    return x
  
  def validate_maxLength(self, x, fieldname, schema, length=None):
    '''
    Validates that the value of the given field is shorter than or equal
    to the specified length if a string
    '''
    value = x.get(fieldname)
    if length is not None and \
       value is not None and \
       self._is_string_type(value) and \
       len(value) > length:
      raise ValueError("Length of value %r for field '%s' must be less than or equal to %f" % (value, fieldname, length))
    return x
    
  def validate_minLength(self, x, fieldname, schema, length=None):
    '''
    Validates that the value of the given field is longer than or equal
    to the specified length if a string
    '''
    value = x.get(fieldname)
    if length is not None and \
       value is not None and \
       self._is_string_type(value) and \
       len(value) < length:
      raise ValueError("Length of value %r for field '%s' must be more than or equal to %f" % (value, fieldname, length))
    return x
  
  def validate_enum(self, x, fieldname, schema, options=None):
    '''
    Validates that the value of the field is equal to one of the
    specified option values
    '''
    value = x.get(fieldname)
    if options is not None and value is not None:
      if not type(options) == types.ListType:
        raise ValueError("Enumeration %r for field '%s' is not a list type", (options, fieldname))
      if value not in options:
        raise ValueError("Value %r for field '%s' is not in the enumeration: %r" % (value, fieldname, options))
    return x
  
  def validate_options(self, x, fieldname, schema, options=None):
    return x
  
  def validate_readonly(self, x, fieldname, schema, readonly=False):
    return x
  
  def validate_title(self, x, fieldname, schema, title=None):
    if title is not None and \
       not self._is_string_type(title):
      raise ValueError("The title for field '%s' must be a string" % fieldname);
    return x
  
  def validate_description(self, x, fieldname, schema, description=None):
    if description is not None and \
       not self._is_string_type(description):
      raise ValueError("The description for field '%s' must be a string." % fieldname);
    return x
  
  def validate_format(self, x, fieldname, schema, format=None):
    '''
    Validates that the value of the field matches the predifined format
    specified.
    '''
    # No definitions are currently defined for formats
    return x
  
  def validate_default(self, x, fieldname, schema, default=None):
    '''
    Adds default data to the original json document if the document is
    not readonly
    '''
    if self._interactive_mode and fieldname not in x.keys() and default is not None:
      if not schema.get("readonly"):
        x[fieldname] = default
    return x
  
  def validate_transient(self, x, fieldname, schema, transient=False):
    return x
  
  def validate_maxDecimal(self, x, fieldname, schema, maxdecimal=None):
    '''
    Validates that the value of the given field has less than or equal
    to the maximum number of decimal places given
    '''
    value = x.get(fieldname)
    if maxdecimal is not None and value is not None:
      maxdecstring = str(value)
      if len(maxdecstring[maxdecstring.find(".")+1:]) > maxdecimal:
        raise ValueError("Value %r for field '%s' must not have more than %d decimal places" % (value, fieldname, maxdecimal))
    return x
  
  def validate_hidden(self, x, fieldname, schema, hidden=False):
    return x
  
  def validate_disallow(self, x, fieldname, schema, disallow=None):
    '''
    Validates that the value of the given field does not match the
    disallowed type.
    '''
    if disallow is not None:
      try:
        self.validate_type(x, fieldname, schema, disallow)
      except ValueError:
        return x
      raise ValueError("Value %r of type %s is disallowed for field '%s'" % (x.get(fieldname), disallow, fieldname))
    return x
  
  def validate_extends(self, x, fieldname, schema, extends=None):
    return x
  
  def _convert_type(self, fieldtype):
    if type(fieldtype) in (types.TypeType, types.DictType):
      return fieldtype
    elif type(fieldtype) == types.ListType:
      converted_fields = []
      for subfieldtype in fieldtype:
        converted_fields.append(self._convert_type(subfieldtype))
      return converted_fields
    elif fieldtype is None:
      return None
    else:
      fieldtype = str(fieldtype)
      if fieldtype in self._typesmap.keys():
        return self._typesmap[fieldtype]
      else:
        raise ValueError("Field type '%s' is not supported." % fieldtype)
  
  def validate(self, data, schema):
    '''
    Validates a piece of json data against the provided json-schema.
    '''
    
    #TODO: Validate the schema object here.
    
    self._refmap = {
      '$': schema
    }
    # Wrap the data in a dictionary
    self._validate(data, schema)
  
  def _validate(self, data, schema):
    self.__validate("_data", {"_data": data}, schema)
  
  def __validate(self, fieldname, data, schema):
    
    if schema is not None:
      if not type(schema) == types.DictType:
        raise ValueError("Schema structure is invalid.");
      
      # Produce a copy of the schema object since we will make changes to
      # it to process default values. Deep copy is not necessary since we will
      # produce a copy of sub items on the next recursive call.
      
      new_schema = copy.copy(schema)
      #Initialize defaults
      for schemaprop in self._schemadefault.keys():
        if schemaprop not in new_schema:
          new_schema[schemaprop] = self._schemadefault[schemaprop]
      
      for schemaprop in new_schema:
        
        validatorname = "validate_"+schemaprop
        
        try:
          validator = getattr(self, validatorname)
          # Pass the original schema object but the value of the property from
          # copy in order to validate default values.
          validator(data, fieldname, schema, new_schema.get(schemaprop))
        except AttributeError, e:
          raise ValueError("Schema property '%s' is not supported" % schemaprop)
      
    return data
  
  def _is_string_type(self, value):
    return type(value) in (types.StringType, types.UnicodeType)
  
__all__ = [ 'JSONSchemaValidator' ]
