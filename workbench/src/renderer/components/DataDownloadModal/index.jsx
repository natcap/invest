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
function DataDownloadModal(props) {
  const [allDataCheck, setAllDataCheck] = React.useState(true);
  const [allLinksArray, setAllLinksArray] = React.useState([]);
  const [selectedLinksArray, setSelectedLinksArray] = React.useState([]);
  const [modelCheckBoxState, setModelCheckBoxState] = React.useState({});
  const [dataRegistry, setDataRegistry] = React.useState(null);
  const [baseURL, setBaseURL] = React.useState(BASE_URL);
  const [alertPath, setAlertPath] = React.useState('');
  const controller = new AbortController();
  const signal = controller.signal;

  React.useEffect(() => {
    componentMountFunc();
    return () => {
      controller.abort();
    }
  }, [])

  async function componentMountFunc() {
    const registry = JSON.parse(JSON.stringify(sampledataRegistry));
    const tokenURL = await ipcRenderer.invoke(ipcMainChannels.CHECK_STORAGE_TOKEN);
    const baseURL = tokenURL || BASE_URL;
    let filesizes;
    try {
      const response = await window.fetch(
        `${baseURL}/registry.json`, { signal: signal, method: 'get' }
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
    const modelCheckBoxDict = {};
    Object.entries(registry).forEach(([modelName, data]) => {
      linksArray.push(`${baseURL}/${data.filename}`);
      modelCheckBoxDict[modelName] = true;
      try {
        registry[modelName].filesize = filesizes[data.filename];
      } catch {
        registry[modelName].filesize = DEFAULT_FILESIZE;
      }
    });

    setAllLinksArray(linksArray);
    setSelectedLinksArray(linksArray);
    setModelCheckBoxState(modelCheckBoxDict);
    setDataRegistry(registry);
    setBaseURL(baseURL);
  }

  async function handleSubmit(event) {
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
        setAlertPath(data.filePaths[0]);
      } else {
        setAlertPath('');
        ipcRenderer.send(
          ipcMainChannels.DOWNLOAD_URL,
          selectedLinksArray,
          data.filePaths[0]
        );
        closeDialog();
      }
    }
  }

  function handleCheckAll(event) {
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
    setAllDataCheck(event.currentTarget.checked);
    setModelCheckBoxState(newCheckList);
    setSelectedLinksArray(selectedLinks);
  }

  function handleCheckList(event, modelName) {
    const modelCheckBoxDict = modelCheckBoxState;
    const url = `${baseURL}/${dataRegistry[modelName].filename}`;
    if (event.currentTarget.checked) {
      setSelectedLinksArray(selectedLinksArray => selectedLinksArray.push(url));
      modelCheckBoxDict[modelName] = true;
    } else {
      setSelectedLinksArray(selectedLinksArray => selectedLinksArray.filter((val) => val !== url));
      modelCheckBoxDict[modelName] = false;
    }
    setAllDataCheck(false);
    // setSelectedLinksArray(selectedLinksArray);
    setModelCheckBoxState(modelCheckBoxState);
  }

  function closeDialog() {
    setAlertPath('');
    props.closeModal();
  }

  const { t } = props;
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
                onChange={(event) => handleCheckList(
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
      show={props.show}
      onHide={closeDialog}
      size="lg"
    >
      <Form>
        <Modal.Header>
          {
            (alertPath)
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
                  <p className="mb-0"><em>{alertPath}</em></p>
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
                    checked={allDataCheck}
                    onChange={handleCheckAll}
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
            onClick={closeDialog}
          >
            {t("Cancel")}
          </Button>
          <Button
            variant="primary"
            onClick={handleSubmit}
            disabled={!downloadEnabled}
          >
            {t("Download")}
          </Button>
        </Modal.Footer>
      </Form>
    </Modal>
  );
}

DataDownloadModal.propTypes = {
  show: PropTypes.bool.isRequired,
  closeModal: PropTypes.func.isRequired,
};

export default withTranslation()(DataDownloadModal);
