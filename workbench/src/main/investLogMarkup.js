/**
 * Assign a class based on text content.
 *
 * @param  {string} message - from a python logger
 * @returns {string} - a class name or an empty string
 */
export default function markupMessage(message) {
  if (/(ERROR|CRITICAL)/.test(message)) {
    return 'invest-log-error';
  }
  if (/natcap\.invest.*WARNING/.test(message)) {
    return 'invest-log-primary-warning';
  }
  if (/natcap\.invest/.test(message)) {
    return 'invest-log-primary';
  }
  return '';
}
