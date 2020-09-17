import React from 'react';
import { remote } from 'electron'; // eslint-disable-line import/no-extraneous-dependencies
import PropTypes from 'prop-types';

import Button from 'react-bootstrap/Button';

/** Render a button that saves current args to a datastack json.
 * Opens an native OS filesystem dialog to browse to a save location.
 * Creates the JSON using datastack.py.
 */
export default class SaveFileButton extends React.Component {
  constructor(props) {
    super(props);
    this.browseSaveFile = this.browseSaveFile.bind(this);
  }

  async browseSaveFile(event) {
    const data = await remote.dialog.showSaveDialog(
      { defaultPath: this.props.defaultTargetPath }
    );
    if (data.filePath) {
      this.props.func(data.filePath);
    }
  }

  render() {
    return (
      <Button
        onClick={this.browseSaveFile}
        variant="link"
      >
        {this.props.title}
      </Button>
    );
  }
}

SaveFileButton.propTypes = {
  title: PropTypes.string.isRequired,
  defaultTargetPath: PropTypes.string.isRequired,
  func: PropTypes.func.isRequired,
};
