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
