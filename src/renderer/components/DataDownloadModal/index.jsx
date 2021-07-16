import React from 'react';
import PropTypes from 'prop-types';
import { ipcRenderer } from 'electron';

import Form from 'react-bootstrap/Form';
import Button from 'react-bootstrap/Button';
import Modal from 'react-bootstrap/Modal';
import Alert from 'react-bootstrap/Alert';
import ProgressBar from 'react-bootstrap/ProgressBar';
import Table from 'react-bootstrap/Table';

import Expire from '../Expire';
import sampledataRegistry from '../../sampledata_registry.json';
import { ipcMainChannels } from '../../../main/ipcMainChannels';

const logger = window.Workbench.getLogger(__filename.split('/').slice(-1)[0]);

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

    this.handleSubmit = this.handleSubmit.bind(this);
    this.handleCheckAll = this.handleCheckAll.bind(this);
    this.handleCheckList = this.handleCheckList.bind(this);
  }

  componentDidMount() {
    const linksArray = [];
    const dataListCheckBoxes = {};
    Object.entries(sampledataRegistry)
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

  async handleSubmit(event) {
    event.preventDefault();
    // even though the idea is to save files, here we just want to chooose
    // a directory, so must use OpenDialog.
    const data = await ipcRenderer.invoke(
      ipcMainChannels.SHOW_OPEN_DIALOG,
      { properties: ['openDirectory'] }
    );
    if (data.filePaths.length) {
      ipcRenderer.send(
        ipcMainChannels.DOWNLOAD_URL,
        this.state.selectedLinksArray,
        data.filePaths[0]
      );
      this.props.storeDownloadDir(data.filePaths[0]);
    }
    this.props.closeModal();
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
    const { url } = sampledataRegistry[modelName];
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
    const DatasetCheckboxRows = [];
    Object.keys(dataListCheckBoxes)
      .forEach((modelName) => {
        const filesize = parseFloat(
          `${sampledataRegistry[modelName].filesize / 1000000}`
        ).toFixed(2) + ' MB';
        const labelSuffix = sampledataRegistry[modelName].labelSuffix || '';
        DatasetCheckboxRows.push(
          <tr key={modelName}>
            <td>
              <Form.Check
                className="pt-1"
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
                </Form.Check.Label>
              </Form.Check>
            </td>
            <td><em>{labelSuffix}</em></td>
            <td>{filesize}</td>
          </tr>
        );
      });

    return (
      <Modal className="download-data-modal"
        show={this.props.show}
        onHide={this.props.closeModal}
        size="lg"
      >
        <Form>
          <Modal.Header>
            <Modal.Title>Download InVEST sample data</Modal.Title>
          </Modal.Header>
          <Modal.Body>
            <Table
              size="sm"
              borderless
              striped
            >
              <thead>
                <tr>
                  <th>
                    <Form.Check
                      type="checkbox"
                      id="all-sampledata"
                      checked={this.state.allDataCheck}
                      onChange={this.handleCheckAll}
                      name="all-sampledata"
                      label="Select All"
                    />
                  </th>
                </tr>
              </thead>
              <tbody className="table-body">
                {DatasetCheckboxRows}
              </tbody>
            </Table>
          </Modal.Body>
          <Modal.Footer>
            <Button
              variant="secondary"
              onClick={this.props.closeModal}
            >
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
  closeModal: PropTypes.func.isRequired,
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
      now={(nComplete + 1) / nTotal}
      label={`Downloading ${nComplete + 1} of ${nTotal}`}
    />
  );
}

DownloadProgressBar.propTypes = {
  downloadedNofN: PropTypes.arrayOf(PropTypes.number).isRequired,
  expireAfter: PropTypes.number.isRequired,
};
