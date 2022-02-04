import sanitizeHtml from 'sanitize-html';

/**
 * Encapsulate text in html, assigning class based on text content.
 *
 * @param  {string} message - from a python logger
 * @param  {string} pyModuleName - e.g. 'natcap.invest.carbon'
 * @returns {string} - sanitized html
 */
export default function markupMessage(message, pyModuleName) {
  const escapedPyModuleName = pyModuleName.replace(/\./g, '\\.');
  const patterns = {
    'invest-log-error': /(ERROR|CRITICAL)/,
    'invest-log-primary-warning': new RegExp(`${escapedPyModuleName}.*WARNING`),
    'invest-log-primary': new RegExp(escapedPyModuleName)
  };

  const logTextTag = 'span';
  const allowedHtml = {
    allowedTags: [logTextTag],
    allowedAttributes: { [logTextTag]: ['class'] },
  };

  // eslint-disable-next-line
  for (const [cls, pattern] of Object.entries(patterns)) {
    if (pattern.test(message)) {
      const markup = `<${logTextTag} class="${cls}">${message}</${logTextTag}>`;
      return sanitizeHtml(markup, allowedHtml);
    }
  }
  return sanitizeHtml(message);
}
