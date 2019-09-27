import fs from 'fs';
import path from 'path';

// TODO: if invalid, also return a helpful message
// to use in Form.Control.Feedback.
// Will require a new place in state to hold the message.

function validate(value, type, required) {
  // This function validates a single input value given a rule.
  //
  // Parameters:
  //   value (string): the value to validate
  //   type (string):   
  //   required (boolean or string): 
  //        sometimes it's a bool, other times a string describing dependecies
  // Returns:
  //   boolean

  if (value === undefined || value === '') {
    // if required is a string, we assume that means some sort of conditional
    // requirement, and for simplicity we'll treat that as an optional input.
    if (typeof required === 'string') {
      required = false; 
    }
    return (required ? false : true)  // empty is valid for optional args
  } else {

    if (['csv', 'vector', 'raster'].includes(type)) {
      return fs.existsSync(value);
    }

    // if (type === 'directory') {
    //   return (fs.existsSync(value) && fs.lstatSync(value).isDirectory());
    // }

    if (type === 'directory') { // TODO: this really means 'workspace'! use above for generic dir
      const dirname = path.dirname(value);
      return (fs.existsSync(dirname) && fs.lstatSync(dirname).isDirectory());
    }

    if (type === 'number') {
      return Number.isInteger(parseInt(value));
    }

    if (type === 'freestyle_string') {
      return true; // for the results_suffix, anything goes?
    }

    if (type === 'option_string') {
      return true;  // dropdowns are always valid
    }

    if (type === 'boolean') {
      return true;  // boolean types should have an input that is always valid (e.g. dropdown, radio/checkbox)
    }
  }
  throw 'Validation rule is not defined';
}

export default validate;