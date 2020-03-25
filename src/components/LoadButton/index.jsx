import React from 'react';
import Electron from 'electron'
import PropTypes from 'prop-types';

import Row from 'react-bootstrap/Row';
import Form from 'react-bootstrap/Form';
import Button from 'react-bootstrap/Button';
import Modal from 'react-bootstrap/Modal';
import Dropdown from 'react-bootstrap/Dropdown';

import { fetchDatastackFromFile } from '../../server_requests';

export class LoadButton extends React.Component {
  /** Render a button that loads args from a datastack, parameterset, or logfile.
  * Opens an native OS filesystem dialog to browse to a file.
  * Extracts the args using datastack.py.
  */

  constructor(props) {
    super(props);
    this.browseFile = this.browseFile.bind(this);
  }

  async browseFile(event) {
    const data = await Electron.remote.dialog.showOpenDialog()
    const payload = { 
      datastack_path: data.filePaths[0]
    }
    const datastack = await fetchDatastackFromFile(payload)
    console.log(datastack)
    const specLoaded = await this.props.investGetSpec(datastack.module_name)
    console.log('after promised getspec, before batch update')
    console.log(specLoaded)
    if (specLoaded) { this.props.batchUpdateArgs(datastack['args']) }
  }

  render() {
    return(
      <Button 
        onClick={this.browseFile}
        variant="primary">
        Load Parameters
      </Button>
    );
  }
}

LoadButton.propTypes = {
  argsToJsonFile: PropTypes.func,
  disabled: PropTypes.bool
}

