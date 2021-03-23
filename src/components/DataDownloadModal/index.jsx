import path from 'path';
import React from 'react';
import PropTypes from 'prop-types';
import { remote, ipcRenderer } from 'electron';

import Col from 'react-bootstrap/Col';
import Row from 'react-bootstrap/Row';
import Form from 'react-bootstrap/Form';
import Button from 'react-bootstrap/Button';
import Modal from 'react-bootstrap/Modal';

import pkg from '../../../package.json';

/** Render a dialog with a form for configuring global invest settings */
export default class DataDownloadModal extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      allDataCheck: true,
      sampleDataRegistryArray: [],
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
    // TODO move this query to the build process and package this data in a file?
    const prefix = encodeURIComponent(`invest/${pkg.invest.version}/data`);
    const queryURL = `https://www.googleapis.com/storage/v1/b/${pkg.invest.bucket}/o?prefix=${prefix}`;
    const linksArray = [];
    const dataListCheckBoxes = {};
    fetch(queryURL)
      .then((response) => {
        if (response.status === 200) {
          return response.json();
        }
        console.log(response.status);
      })
      .then((data) => {
        data.items.forEach((item) => {
          linksArray.push(item.mediaLink);
          dataListCheckBoxes[item.name] = true;
        });
        console.log(linksArray);
        this.setState({
          sampleDataRegistryArray: data.items,
          allLinksArray: linksArray,
          selectedLinksArray: linksArray,
          dataListCheckBoxes: dataListCheckBoxes,
        });
      });
  }

  handleClose() {
    // storing something sends the signal that the user declined
    // and doesn't need to be asked again on app startup.
    this.props.storeDownloadDir('');
  }

  async handleSubmit(event) {
    event.preventDefault();
    // need two things here
    // 1. list of files to download
    // 2. downloadDir to save them
    // downloads in background? or keep modal open?
    // progress is important.
    const allDataURL = path.join(
      this.props.releaseDataURL, 'InVEST_3.9.0.post235+g296690d7_sample_data.zip'
    );
    ipcRenderer.send('download-url', allDataURL);
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
    console.log(event.target.checked);
    console.log(dataListCheckBoxes);
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

  handleCheckList(event, item) {
    let { selectedLinksArray, dataListCheckBoxes } = this.state;
    if (event.target.checked) {
      selectedLinksArray.push(item.mediaLink);
      dataListCheckBoxes[item.name] = true;
    } else {
      selectedLinksArray = selectedLinksArray.filter((val) => val !== item.mediaLink);
      dataListCheckBoxes[item.name] = false;
    }
    this.setState({
      allDataCheck: false,
      selectedLinksArray: selectedLinksArray,
      dataListCheckBoxes: dataListCheckBoxes,
    });
  }

  render() {
    console.log(this.state.selectedLinksArray);
    const DatasetCheckboxList = [];
    this.state.sampleDataRegistryArray
      .forEach((item) => {
        const name = path.basename(item.name);
        DatasetCheckboxList.push(
          <Form.Check
            key={name}
            id={name}
            type="checkbox"
            checked={this.state.dataListCheckBoxes[item.name]}
            onChange={(event) => this.handleCheckList(event, item)}
            label={name}
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
                Download All
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
              Download All
            </Button>
          </Modal.Footer>
        </Form>
      </Modal>
    );
  }
}

// DataDownladModal.propTypes = {
//   saveSettings: PropTypes.func,
//   investSettings: PropTypes.shape({
//     nWorkers: PropTypes.string,
//     loggingLevel: PropTypes.string,
//   })
// };
