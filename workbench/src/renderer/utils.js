/** Convert a string representing a bool to an actual boolean.
 *
 * HTML inputs in this app must send string values, but for invest args of
 * type boolean, we want to convert that to a real boolean before passing
 * to invest's validate or execute.
 *
 * @param {string} val - such as "true", "True", "false", "False"
 * @returns {boolean} unless the input was not a string, then undefined
 */
export function boolStringToBoolean(val) {
  let valBoolean;
  try {
    const valString = val.toLowerCase();
    valBoolean = valString === 'true';
  } catch (e) {
    if (e instanceof TypeError) {
      valBoolean = undefined;
    } else {
      throw e;
    }
  }
  return valBoolean;
}

/** Create a JSON string with invest argument keys and values.
 *
 * @param {object} args - object keyed by invest argument keys and
 *   with each item including a `value` property, among others.
 * @returns {object} - invest argument key: value pairs as expected
 * by invest model `execute` and `validate` functions
 */
export function argsDictFromObject(args) {
  const argsDict = {};
  Object.keys(args).forEach((argname) => {
    argsDict[argname] = args[argname].value;
  });
  return argsDict;
}

/** Prevent the default case for onDragOver and set dropEffect to none.
*
*@param {Event} event - an ondragover event.
*/
export function dragOverHandlerNone(event) {
  event.preventDefault();
  event.stopPropagation();
  event.dataTransfer.dropEffect = 'none';
}
