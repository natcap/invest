import React from 'react';
import PropTypes from 'prop-types';

import Form from 'react-bootstrap/Form';
import Button from 'react-bootstrap/Button';
import Modal from 'react-bootstrap/Modal';
import Alert from 'react-bootstrap/Alert';
import Table from 'react-bootstrap/Table';
import {
  MdErrorOutline,
} from 'react-icons/md';
import { withTranslation } from 'react-i18next';

import sampledataRegistry from './sampledata_registry.json';
import { ipcMainChannels } from '../../../main/ipcMainChannels';

const { ipcRenderer } = window.Workbench.electron;
const { logger } = window.Workbench;

// A URL for sampledata to use in devMode, when the token containing the URL
// associated with a production build of the Workbench does not exist.
const BASE_URL = 'https://storage.googleapis.com/releases.naturalcapitalproject.org/invest/3.13.0/data';
const DEFAULT_FILESIZE = 0;

/** Render a dialog with a form for configuring global invest settings */
class DataDownloadModal extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      allDataCheck: true,
      allLinksArray: [],
      selectedLinksArray: [],
      modelCheckBoxState: {},
      dataRegistry: null,
      baseURL: BASE_URL,
      alertPath: '',
    };

    this.handleSubmit = this.handleSubmit.bind(this);
    this.handleCheckAll = this.handleCheckAll.bind(this);
    this.handleCheckList = this.handleCheckList.bind(this);
    this.closeDialog = this.closeDialog.bind(this);
    this.controller = new AbortController();
    this.signal = this.controller.signal;
  }

  async componentDidMount() {
    const registry = JSON.parse(JSON.stringify(sampledataRegistry));
    const tokenURL = await ipcRenderer.invoke(ipcMainChannels.CHECK_STORAGE_TOKEN);
    const baseURL = tokenURL || BASE_URL;
    let filesizes;
    try {
      const response = await window.fetch(
        `${baseURL}/registry.json`, { signal: this.signal, method: 'get' }
      );
      filesizes = await response.json();
    } catch (error) {
      if (error.name === 'AbortError') {
        // We aborted the fetch on unmount,
        // return before trying to setState.
        return;
      }
      logger.debug(error);
    }

    const linksArray = [];
    const modelCheckBoxState = {};
    Object.entries(registry).forEach(([modelName, data]) => {
      linksArray.push(`${baseURL}/${data.filename}`);
      modelCheckBoxState[modelName] = true;
      try {
        registry[modelName].filesize = filesizes[data.filename];
      } catch {
        registry[modelName].filesize = DEFAULT_FILESIZE;
      }
    });

    this.setState({
      allLinksArray: linksArray,
      selectedLinksArray: linksArray,
      modelCheckBoxState: modelCheckBoxState,
      dataRegistry: registry,
      baseURL: baseURL,
    });
  }

  componentWillUnmount() {
    this.controller.abort();
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
      const writable = await ipcRenderer.invoke(
        ipcMainChannels.CHECK_FILE_PERMISSIONS, data.filePaths[0]
      );
      if (!writable) {
        this.setState({ alertPath: data.filePaths[0] });
      } else {
        this.setState({ alertPath: '' });
        ipcRenderer.send(
          ipcMainChannels.DOWNLOAD_URL,
          this.state.selectedLinksArray,
          data.filePaths[0]
        );
        this.closeDialog();
      }
    }
  }

  handleCheckAll(event) {
    const {
      modelCheckBoxState,
      allLinksArray,
    } = this.state;
    const newCheckList = Object.fromEntries(
      Object.entries(modelCheckBoxState).map(
        ([k, v]) => [k, event.currentTarget.checked]
      )
    );
    let selectedLinks;
    if (event.currentTarget.checked) {
      selectedLinks = allLinksArray;
    } else {
      selectedLinks = [];
    }
    this.setState({
      allDataCheck: event.currentTarget.checked,
      modelCheckBoxState: newCheckList,
      selectedLinksArray: selectedLinks,
    });
  }

  handleCheckList(event, modelName) {
    let {
      selectedLinksArray,
      modelCheckBoxState,
      dataRegistry,
      baseURL,
    } = this.state;
    const url = `${baseURL}/${dataRegistry[modelName].filename}`;
    if (event.currentTarget.checked) {
      selectedLinksArray.push(url);
      modelCheckBoxState[modelName] = true;
    } else {
      selectedLinksArray = selectedLinksArray.filter((val) => val !== url);
      modelCheckBoxState[modelName] = false;
    }
    this.setState({
      allDataCheck: false,
      selectedLinksArray: selectedLinksArray,
      modelCheckBoxState: modelCheckBoxState,
    });
  }

  closeDialog() {
    this.setState({ alertPath: '' });
    this.props.closeModal();
  }

  render() {
    const {
      modelCheckBoxState,
      selectedLinksArray,
      dataRegistry,
    } = this.state;
    const { t } = this.props;
    // Don't render until registry is loaded, since it loads async
    if (!dataRegistry) { return <div />; }

    const downloadEnabled = Boolean(selectedLinksArray.length);
    const DatasetCheckboxRows = [];
    Object.keys(modelCheckBoxState)
      .forEach((modelName) => {
        const filesize = parseFloat(dataRegistry[modelName].filesize);
        const filesizeStr = `${(filesize / 1000000).toFixed(2)} MB`;
        const note = dataRegistry[modelName].note || '';
        DatasetCheckboxRows.push(
          <tr key={modelName}>
            <td>
              <Form.Check
                className="pt-1"
                id={modelName}
              >
                <Form.Check.Input
                  type="checkbox"
                  checked={modelCheckBoxState[modelName]}
                  onChange={(event) => this.handleCheckList(
                    event, modelName
                  )}
                />
                <Form.Check.Label>
                  {modelName}
                </Form.Check.Label>
              </Form.Check>
            </td>
            <td><em>{note}</em></td>
            <td>{filesizeStr}</td>
          </tr>
        );
      });

    return (
      <Modal className="download-data-modal"
        show={this.props.show}
        onHide={this.closeDialog}
        size="lg"
      >
        <Form>
          <Modal.Header>
            {
              (this.state.alertPath)
                ? (
                  <Alert
                    className="mb-0"
                    variant="danger"
                  >
                    <MdErrorOutline
                      size="2em"
                      className="pr-1"
                    />
                    {t('Please choose a different folder. '
                      + 'This application does not have permission to write to folder:')}
                    <p className="mb-0"><em>{this.state.alertPath}</em></p>
                  </Alert>
                )
                : <Modal.Title>{t("Download InVEST sample data")}</Modal.Title>
            }
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
              onClick={this.closeDialog}
            >
              {t("Cancel")}
            </Button>
            <Button
              variant="primary"
              onClick={this.handleSubmit}
              disabled={!downloadEnabled}
            >
              {t("Download")}
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
};

export default withTranslation()(DataDownloadModal);
