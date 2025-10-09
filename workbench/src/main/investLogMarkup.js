const datetimePrefix = /[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]\s+[0-9][0-9]:[0-9][0-9]:[0-9][0-9]\s+/;

/**
 * Assign a class based on text content.
 * 
 * Core invest models log from a logger name starting with natcap.invest
 * Plugins should model from a logger name starting with invest
 *
 * @param  {string} message - from a python logger
 * @returns {string} - a class name or an empty string
 */
export default function markupMessage(message) {
  if (/(ERROR|CRITICAL)/.test(message)) {
    return 'invest-log-error';
  }
  if (
    new RegExp(`^${datetimePrefix.source}natcap\.invest.*WARNING`).test(message)
    || new RegExp(`^${datetimePrefix.source}invest.*WARNING`).test(message)
  ) {
    return 'invest-log-primary-warning';
  }
  if (
    new RegExp(`^${datetimePrefix.source}natcap\.invest`).test(message)
    || new RegExp(`^${datetimePrefix.source}invest`).test(message)
  ) {
    return 'invest-log-primary';
  }
  return '';
}
