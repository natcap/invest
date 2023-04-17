/**
 * Assign a class based on text content.
 *
 * @param  {string} message - from a python logger
 * @param  {string} pyModuleName - e.g. 'natcap.invest.carbon'
 * @returns {string} - a class name or an empty string
 */
export default function markupMessage(message, pyModuleName) {
  const escapedPyModuleName = pyModuleName.replace(/\./g, '\\.');
  const patterns = {
    'invest-log-error': /(ERROR|CRITICAL)/,
    'invest-log-primary-warning': new RegExp(`${escapedPyModuleName}.*WARNING`),
    'invest-log-primary': new RegExp(escapedPyModuleName)
  };

  // eslint-disable-next-line
  for (const [cls, pattern] of Object.entries(patterns)) {
    if (pattern.test(message)) {
      return cls;
    }
  }
  return '';
}
