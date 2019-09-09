import fs from 'fs';
import path from 'path';

// TODO: if invalid, also return a helpful message
// to use in Form.Control.Feedback.
// Will require a new place in state to hold the message.

function validate(value, rule) {
  // This function validates a single input value given a rule.
  //
  // Parameters:
  //   value (string): the value to validate
  //   rule (object):  {required: bool, rule: 'some rule'} 
  // Returns:
  //   boolean

  if (value === '') {
    return (rule.required ? false : true)  // empty is valid for optional args
  }

  if (rule.rule === 'filepath') {
    return fs.existsSync(value);
  }

  if (rule.rule === 'directory') {
    return (fs.existsSync(value) && fs.lstatSync(value).isDirectory());
  }

  if (rule.rule === 'workspace') {
    const dirname = path.dirname(value);
    return (fs.existsSync(dirname) && fs.lstatSync(dirname).isDirectory());
  }

  if (rule.rule === 'integer') {
    return Number.isInteger(parseInt(value));
  }

  if (rule.rule === 'string') {
    return true; // for the results_suffix, anything goes?
  }

  if (rule.rule === 'select') {
    return true;  // dropdowns are always valid
  }

  throw 'Validation rule is not defined';
}

export default validate;