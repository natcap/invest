import React from 'react';
import PropTypes from 'prop-types';
import { ipcRenderer } from 'electron';

import Form from 'react-bootstrap/Form';
import Button from 'react-bootstrap/Button';
import Modal from 'react-bootstrap/Modal';
import Alert from 'react-bootstrap/Alert';
import ProgressBar from 'react-bootstrap/ProgressBar';

import Expire from '../Expire';
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
    // and doesn't need to be asked again on app startup. We need
    // something truthy that won't be confused for a real filepath.
    this.props.storeDownloadDir(1);
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
    const { dataListCheckBoxes, selectedLinksArray } = this.state;
    const downloadEnabled = Boolean(selectedLinksArray.length);
    const DatasetCheckboxList = [];
    Object.keys(dataListCheckBoxes)
      .forEach((modelName) => {
        const filesize = parseFloat(
          `${sampledataRegistry.Models[modelName].filesize / 1000000}`
        ).toFixed(2) + ' MB';
        const labelSuffix = sampledataRegistry.Models[modelName].labelSuffix || '';
        DatasetCheckboxList.push(
          <Form.Check
            className="pt-1"
            key={modelName}
            id={modelName}
          >
            <Form.Check.Input
              type="checkbox"
              checked={dataListCheckBoxes[modelName]}
              onChange={(event) => this.handleCheckList(
                event, modelName
              )}
            />
            <Form.Check.Label>
              {modelName}
              <em>{` ${labelSuffix} . . . ${filesize}`}</em>
            </Form.Check.Label>
          </Form.Check>
        );
      });

    return (
      <Modal
        show={this.props.show}
        onHide={this.handleClose}
        size="lg"
      >
        <Form>
          <Modal.Header>
            <Modal.Title>Download InVEST sample data</Modal.Title>
          </Modal.Header>
          <Modal.Body>
            <h5>
              <Form.Check
                type="checkbox"
                id="all-sampledata"
                checked={this.state.allDataCheck}
                onChange={this.handleCheckAll}
                name="all-sampledata"
                label="Select All"
              />
            </h5>
            <Form.Group>
              {DatasetCheckboxList}
            </Form.Group>
          </Modal.Body>
          <Modal.Footer>
            <Button variant="secondary" onClick={this.handleClose}>
              Cancel
            </Button>
            <Button
              variant="primary"
              onClick={this.handleSubmit}
              disabled={!downloadEnabled}
            >
              Download
            </Button>
          </Modal.Footer>
        </Form>
      </Modal>
    );
  }
}

DataDownloadModal.propTypes = {
  show: PropTypes.bool.isRequired,
  storeDownloadDir: PropTypes.func.isRequired,
};

export function DownloadProgressBar(props) {
  const [nComplete, nTotal] = props.downloadedNofN;
  if (nComplete === nTotal) {
    return (
      <Expire delay={props.expireAfter}>
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

DownloadProgressBar.propTypes = {
  downloadedNofN: PropTypes.arrayOf(PropTypes.number).isRequired,
  expireAfter: PropTypes.number.isRequired,
};
