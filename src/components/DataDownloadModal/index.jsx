import path from 'path';
import React, { useEffect, useState } from 'react';
import PropTypes from 'prop-types';
import { ipcRenderer } from 'electron';
import fetch from 'node-fetch';

import Form from 'react-bootstrap/Form';
import Button from 'react-bootstrap/Button';
import Modal from 'react-bootstrap/Modal';
import Alert from 'react-bootstrap/Alert';
import ProgressBar from 'react-bootstrap/ProgressBar';

import pkg from '../../../package.json';
import sampledataRegistry from '../../sampledata_registry.json';

/** Render a dialog with a form for configuring global invest settings */
export class DataDownloadModal extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      allDataCheck: true,
      allLinksArray: [],
      selectedLinksArray: [],
      dataListCheckBoxes: {},
    };

    this.handleClose = this.handleClose.bind(this);
    this.handleSubmit = this.handleSubmit.bind(this);
    this.handleCheckAll = this.handleCheckAll.bind(this);
    this.handleCheckList = this.handleCheckList.bind(this);
  }

  componentDidMount() {
    const linksArray = [];
    const dataListCheckBoxes = {};
    Object.entries(sampledataRegistry.Models)
      .forEach(([modelName, data]) => {
        linksArray.push(data.url);
        dataListCheckBoxes[modelName] = true;
      });
    this.setState({
      allLinksArray: linksArray,
      selectedLinksArray: linksArray,
      dataListCheckBoxes: dataListCheckBoxes,
    });
  }

  handleClose() {
    // storing something sends the signal that the user declined
    // and doesn't need to be asked again on app startup.
    this.props.storeDownloadDir('');
  }

  async handleSubmit(event) {
    event.preventDefault();
    const allDataURL = sampledataRegistry.allData.url;
    // even though the idea is to save files, here we just want to chooose
    // a directory, so must use OpenDialog.
    const data = await ipcRenderer.invoke(
      'show-open-dialog',
      { properties: ['openDirectory'] }
    );
    if (data.filePaths.length) {
      if (this.state.allDataCheck) {
        ipcRenderer.send('download-url', [allDataURL], data.filePaths[0]);
      } else {
        ipcRenderer.send('download-url', this.state.selectedLinksArray, data.filePaths[0]);
      }
      this.props.storeDownloadDir(data.filePaths[0]);
    }
  }

  handleCheckAll(event) {
    const {
      dataListCheckBoxes,
      allLinksArray,
    } = this.state;
    const newCheckList = Object.fromEntries(
      Object.entries(dataListCheckBoxes).map(
        ([k, v]) => [k, event.target.checked]
      )
    );
    let selectedLinks;
    if (event.target.checked) {
      selectedLinks = allLinksArray;
    } else {
      selectedLinks = [];
    }
    this.setState({
      allDataCheck: event.target.checked,
      dataListCheckBoxes: newCheckList,
      selectedLinksArray: selectedLinks,
    });
  }

  handleCheckList(event, modelName) {
    let { selectedLinksArray, dataListCheckBoxes } = this.state;
    const { url } = sampledataRegistry.Models[modelName];
    if (event.target.checked) {
      selectedLinksArray.push(url);
      dataListCheckBoxes[modelName] = true;
    } else {
      selectedLinksArray = selectedLinksArray.filter((val) => val !== url);
      dataListCheckBoxes[modelName] = false;
    }
    this.setState({
      allDataCheck: false,
      selectedLinksArray: selectedLinksArray,
      dataListCheckBoxes: dataListCheckBoxes,
    });
  }

  render() {
    const { dataListCheckBoxes } = this.state;
    const DatasetCheckboxList = [];
    Object.keys(dataListCheckBoxes)
      .forEach((modelName) => {
        const filesize = parseFloat(
          `${sampledataRegistry.Models[modelName].filesize / 1000000}`
        ).toFixed(2) + ' MB';
        DatasetCheckboxList.push(
          <Form.Check
            key={modelName}
            id={modelName}
            type="checkbox"
            checked={dataListCheckBoxes[modelName]}
            onChange={(event) => this.handleCheckList(
              event, modelName
            )}
            label={`${modelName} ${filesize}`}
          />
        );
      });

    return (
      <Modal show={this.props.show} onHide={this.handleClose}>
        <Form>
          <Modal.Header>
            <Modal.Title>Download InVEST sample data</Modal.Title>
          </Modal.Header>
          <Modal.Body>
            <Form.Group>
              <Form.Label>
                Select All
              </Form.Label>
              <Form.Check
                id="all-sampledata"
                inline
                type="checkbox"
                checked={this.state.allDataCheck}
                onChange={this.handleCheckAll}
                name="all-sampledata"
              />
              <React.Fragment>
                {DatasetCheckboxList}
              </React.Fragment>
            </Form.Group>
          </Modal.Body>
          <Modal.Footer>
            <Button variant="secondary" onClick={this.handleClose}>
              Cancel
            </Button>
            <Button
              variant="primary"
              onClick={this.handleSubmit}
            >
              Download
            </Button>
          </Modal.Footer>
        </Form>
      </Modal>
    );
  }
}

function Expire(props) {
  const [visible, setVisible] = useState(true);

  useEffect(() => {
    setTimeout(() => {
      setVisible(false);
    }, props.delay);
  }, [props.delay]);

  return visible ? <div>{props.children}</div> : <div />;
}

export function DownloadProgressBar(props) {
  const [nComplete, nTotal] = props.downloadedNofN;
  if (nComplete === nTotal) {
    return (
      <Expire delay="5000">
        <Alert variant="success">Download Complete</Alert>
      </Expire>
    );
  }
  return (
    <ProgressBar
      animated
      max={1}
      now={nComplete / nTotal}
      label={`Downloading ${nComplete + 1} of ${nTotal}`}
    />
  );
}

// DataDownladModal.propTypes = {
//   saveSettings: PropTypes.func,
//   investSettings: PropTypes.shape({
//     nWorkers: PropTypes.string,
//     loggingLevel: PropTypes.string,
//   })
// };
