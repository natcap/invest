import fs from 'fs';

function validate(value, rule) {
  // func to validate a single input value
  // returns boolean

  if (rule === 'filepath') {
    return fs.existsSync(value);
  }

  if (rule === 'directory') {
    // todo: invest workspace need not be pre-existing
    return (fs.existsSync(value) && fs.lstatSync(value).isDirectory());
  }

  if (rule === 'integer') {
    return Number.isInteger(parseInt(value));
  }

  if (rule === 'string') {
    return true; // for the results_suffix, anything goes?
  }

  if (['select', 'checkbox'].includes(rule)) {
    return true;  // dropdowns and checkboxes are always valid
  }

  throw 'Validation rule is not defined';
}

export default validate;