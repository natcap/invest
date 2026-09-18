import React, { useEffect, useState } from 'react';
import PropTypes from 'prop-types';

import Button from 'react-bootstrap/Button';
import Col from 'react-bootstrap/Col';
import Modal from 'react-bootstrap/Modal';
import Nav from 'react-bootstrap/Nav';
import Row from 'react-bootstrap/Row';
import Tab from 'react-bootstrap/Tab';
import { useTranslation } from 'react-i18next';
import { IconContext } from "react-icons";
import {
  MdClose,
  MdOutlineWarningAmber
} from 'react-icons/md';

import { ipcMainChannels } from '../../../main/ipcMainChannels';

import AboutTab from './AboutTab';
import AdvancedSettingsTab from './AdvancedSettingsTab';
import InstalledPluginsTab from './InstalledPluginsTab';
import ManualInstallTab from './ManualInstallTab';
import PluginRegistryTab from './PluginRegistryTab';

const { getFilePath, ipcRenderer } = window.Workbench.electron;

export default function PluginModal(props) {
  const {
    updateInvestList,
    closeInvestModel,
    openJobs,
    show,
    closeModal,
    openModal,
  } = props;
  const [statusMessage, setStatusMessage] = useState('Installing...');
  const [needsMSVC, setNeedsMSVC] = useState(false);

  const [installLoading, setInstallLoading] = useState('');
  const [installErr, setInstallErr] = useState('');
  const [installErrMsg, setInstallErrMsg] = useState('');
  const [installSuccess, setInstallSuccess] = useState('');

  const [uninstallLoading, setUninstallLoading] = useState('');
  const [uninstallErr, setUninstallErr] = useState('');
  const [uninstallErrMsg, setUninstallErrMsg] = useState('');
  const [removalSuccess, setRemovalSuccess] = useState(false);

  const [plugins, setPlugins] = useState({});
  const [registryData, setRegistryData] = useState([]);
  const [activePluginKey, setActivePluginKey] = useState('');
  const [activePluginIndex, setActivePluginIndex] = useState(0);
  const [fetchError, setFetchError] = useState(false);

  const registryMetadataURL = "https://natcap.github.io/invest-plugin-registry/workbench_metadata.json";
  const dataCacheKey = "registryData";
  const cacheTimeout = 1000 * 60 * 60 * 24; // 24 hours

  const handleModalClose = () => {
    if (!installLoading) {
      setInstallErr('');
      setInstallErrMsg('');
      setUninstallErr('');
      setUninstallErrMsg('');
      setInstallSuccess('');
      setRemovalSuccess(false);
      closeModal();
    }
  };

  const resetManualInstallFormStatus = () => {
    setInstallErr('');
    setInstallLoading('');
  }

  const clearUninstallErrors = () => {
    setUninstallLoading('');
    setUninstallErr('');
    setUninstallErrMsg('');
  }

  function sortByName(a, b) {
    if (a.plugin_name > b.plugin_name) {
      return 1;
    }
    return -1;
  }

  async function fetchRegistryData() {
    let cacheJSON = null;
    let cacheStale = true;

    // Check if data is cached in localStorage
    const cachedData = localStorage.getItem(dataCacheKey);

    if (cachedData) {
      cacheJSON = JSON.parse(cachedData);
      if (Date.now() - cacheJSON.cacheDate < cacheTimeout) {
        cacheStale = false;
      }
    }

    if (cacheJSON && !cacheStale) {
        console.log('Using cached data');
        setRegistryData(cacheJSON.data);
        setFetchError(false);
    } else {
      console.log('Cache miss; fetching data...');
      try {
        // Fetch data from the Registry if not cached
        const response = await fetch(registryMetadataURL);
        if (!response.ok) {
          throw new Error(`Response status: ${response.status}`);
        }
        const pluginJSON = await response.json();
        const sortedPlugins = pluginJSON.data.sort(sortByName);

        const cacheData = Object({
          'data': sortedPlugins,
          'cacheDate': Date.now()
        });

        // Cache the data in localStorage
        localStorage.setItem(dataCacheKey, JSON.stringify(cacheData));

        setRegistryData(sortedPlugins);
        setFetchError(false);
      } catch (error) {
        console.log(error.message);
        setFetchError(true);
      }
    }
  }

  const handleRetryFetchRegistryData = () => {
    setFetchError(false);
    fetchRegistryData();
  }

  useEffect(() => {
    fetchRegistryData();
  }, []);

  useEffect(() => {
    if (Object.keys(registryData).length) {
      setActivePluginKey(registryData[0]['invest_package_name']);
      setActivePluginIndex(0);
    }
  }, [registryData]);

  function handlePluginClick(pluginKey) {
    setActivePluginKey(pluginKey);
    setActivePluginIndex(registryData.findIndex(i => i.invest_package_name === pluginKey));
  }

  const addPlugin = (pluginID, url, revision, path, sourceType) => {
    setInstallSuccess('');
    setRemovalSuccess(false);
    setInstallLoading(pluginID);
    ipcRenderer.invoke(
      ipcMainChannels.ADD_PLUGIN,
      url,       // git url (via manual install or registry)
      revision,  // revision (manual install) or version (registry)
      path,      // local path (manual local install)
      sourceType // 'local_path', 'git_url', or 'registry'
    ).then(() => {
      setInstallLoading('');
      updateInvestList();
      setInstallSuccess(pluginID);
    }).catch((err) => {
      setInstallErrMsg(err.toString());
      setInstallErr(pluginID);
      setInstallLoading('');
    });
  };

  const removePlugin = (pluginToRemove) => {
    setRemovalSuccess(false);
    setInstallSuccess('');
    setUninstallLoading(pluginToRemove);
    openJobs.forEach((job, tabID) => {
      if (job.modelID === pluginToRemove) {
        closeInvestModel(tabID);
      }
    });
    ipcRenderer.invoke(
      ipcMainChannels.REMOVE_PLUGIN, pluginToRemove
    ).then(() => {
      setRemovalSuccess(true);
      updateInvestList();
      clearUninstallErrors();
    }).catch((err) => {
      setUninstallLoading('');
      setUninstallErr(pluginToRemove);
      setUninstallErrMsg(err.toString());
    });
  };

  const downloadMSVC = () => {
    closeModal();
    ipcRenderer.invoke(ipcMainChannels.DOWNLOAD_MSVC).then(
      openModal()
    );
  };

  const selectDirectory = async (event) => {
    const data = await ipcRenderer.invoke(
      ipcMainChannels.SHOW_OPEN_DIALOG, { properties: ['openDirectory'] }
    );
    if (data.filePaths.length) {
      return data.filePaths[0];
    }
  };

  /**
   * Prevent the default case for onDragOver so onDrop event will be fired.
   *
   * @param {Event} event - dragover event
   */
  function dragOverHandler(event) {
    event.preventDefault();
    event.stopPropagation();
    if (event.currentTarget.disabled) {
      event.dataTransfer.dropEffect = 'none';
    } else {
      event.dataTransfer.dropEffect = 'copy';
    }
  }

  function getDroppedFilePath(event) {
    event.preventDefault();
    event.stopPropagation();
    event.currentTarget.classList.remove('input-dragging');

    if (event.currentTarget.disabled) {
      return undefined;
    }

    const fileList = event.dataTransfer.files;
    if (fileList.length !== 1) {
      alert(t('Only drop one file at a time.')); // eslint-disable-line no-alert
      return undefined;
    }

    event.currentTarget.focus();
    return getFilePath(fileList[0]);
  }

  const rejectDropHandler = (event) => {
    event.preventDefault();
    event.stopPropagation();
    event.dataTransfer.dropEffect = 'none';
    event.currentTarget.classList.remove('input-dragging');
  };

  function dragEnterHandler(event) {
    event.preventDefault();
    event.stopPropagation();
    if (event.currentTarget.disabled) {
      event.dataTransfer.dropEffect = 'none';
    } else {
      event.dataTransfer.dropEffect = 'copy';
      event.currentTarget.classList.add('input-dragging');
    }
  }

  function dragLeavingHandler(event) {
    event.preventDefault();
    event.stopPropagation();
    event.dataTransfer.dropEffect = 'copy';
    event.currentTarget.classList.remove('input-dragging');
  }

  const selectFile = async (event) => {
    const data = await ipcRenderer.invoke(
      ipcMainChannels.SHOW_OPEN_DIALOG, { properties: ['openFile'] }
    );
    if (data.filePaths.length) {
      return data.filePaths[0];
    }
  };

  useEffect(() => {
    ipcRenderer.on('plugin-install-status', (msg) => { setStatusMessage(msg); });
    if (show) {
      if (window.Workbench.OS === 'win32') {
        ipcRenderer.invoke(ipcMainChannels.HAS_MSVC).then((hasMSVC) => {
          setNeedsMSVC(!hasMSVC);
        });
      }
    }
    return () => { ipcRenderer.removeAllListeners('plugin-install-status'); };
  }, [show]);

  useEffect(() => {
    ipcRenderer.invoke(ipcMainChannels.GET_SETTING, 'plugins').then(
      (data) => {
        if (data) {
          setPlugins(data);
        }
      }
    );
  }, [installLoading, uninstallLoading]);

  const { t } = useTranslation();

  let modalBody = (
    <Modal.Body>
      <Tab.Container id="plugin-modal-tabs" defaultActiveKey="registry">
        <Row>
          <Col sm={2} className="plugin-modal-nav">
            <Nav variant="pills" className="flex-column">
              <Nav.Item className="plugin-modal-nav-item">
                <Nav.Link eventKey="registry">{t('Plugin Registry')}</Nav.Link>
              </Nav.Item>
              <Nav.Item className="plugin-modal-nav-item">
                <Nav.Link eventKey="installed">{t('Installed Plugins')}</Nav.Link>
              </Nav.Item>
              <Nav.Item className="plugin-modal-nav-item">
                <Nav.Link eventKey="manual">{t('Manual Install')}</Nav.Link>
              </Nav.Item>
              <Nav.Item className="plugin-modal-nav-item">
                <Nav.Link eventKey="advanced">{t('Advanced Settings')}</Nav.Link>
              </Nav.Item>
              <Nav.Item className="plugin-modal-nav-item">
                <Nav.Link eventKey="about">{t('About Plugins')}</Nav.Link>
              </Nav.Item>
            </Nav>
          </Col>
          <Col sm={10}>
            <Tab.Content>
              <Tab.Pane eventKey="registry">
                {fetchError ? (
                  <div className="registry-fetch-error">
                    <IconContext.Provider value={{ className: 'registry-warning-icon' }}>
                      <MdOutlineWarningAmber />
                    </IconContext.Provider>
                    <p>
                      {t(`An error occurred when loading the Plugin Registry data.
                        Please check your internet connection, then try again.
                        If the problem persists, consider reporting it on the NatCap Community Forum.`)}
                    </p>
                    <Button
                      className="me-2"
                      onClick={handleRetryFetchRegistryData}
                    >
                      {t('Retry')}
                    </Button>
                  </div>
                ) : (
                  <PluginRegistryTab
                    registryData={registryData}
                    activePluginKey={activePluginKey}
                    activePluginIndex={activePluginIndex}
                    handlePluginClick={handlePluginClick}
                    fetchError={fetchError}
                    installedPlugins={plugins}
                    addPlugin={addPlugin}
                    installLoading={installLoading}
                    installErr={installErr}
                    installErrMsg={installErrMsg}
                    installSuccess={installSuccess}
                    statusMessage={statusMessage}
                    needsMSVC={needsMSVC}
                    downloadMSVC={downloadMSVC}
                  />
                )}
              </Tab.Pane>
              <Tab.Pane eventKey="installed">
                <InstalledPluginsTab
                  plugins={plugins}
                  removePlugin={removePlugin}
                  uninstallLoading={uninstallLoading}
                  uninstallErr={uninstallErr}
                  uninstallErrMsg={uninstallErrMsg}
                  removalSuccess={removalSuccess}
                />
              </Tab.Pane>
              <Tab.Pane eventKey="manual">
                <ManualInstallTab
                  addPlugin={addPlugin}
                  resetManualInstallFormStatus={resetManualInstallFormStatus}
                  installLoading={installLoading}
                  installErr={installErr}
                  installErrMsg={installErrMsg}
                  installSuccess={installSuccess}
                  statusMessage={statusMessage}
                  needsMSVC={needsMSVC}
                  downloadMSVC={downloadMSVC}
                  dragOverHandler={dragOverHandler}
                  dragEnterHandler={dragEnterHandler}
                  dragLeavingHandler={dragLeavingHandler}
                  selectDirectory={selectDirectory}
                  getDroppedFilePath={getDroppedFilePath}
                  rejectDropHandler={rejectDropHandler}
                />
              </Tab.Pane>
              <Tab.Pane eventKey="advanced">
                <AdvancedSettingsTab
                  plugins={plugins}
                  dragOverHandler={dragOverHandler}
                  dragEnterHandler={dragEnterHandler}
                  dragLeavingHandler={dragLeavingHandler}
                  selectFile={selectFile}
                  selectDirectory={selectDirectory}
                  getDroppedFilePath={getDroppedFilePath}
                />
              </Tab.Pane>
              <Tab.Pane eventKey="about">
                <AboutTab />
              </Tab.Pane>
            </Tab.Content>
          </Col>
        </Row>
      </Tab.Container>
    </Modal.Body>
  );

  return (
    <Modal
      size="xl"
      show={show}
      onHide={handleModalClose}
      contentClassName="plugin-modal"
    >
      <Modal.Header>
        <Modal.Title>{t('Plugin Manager')}</Modal.Title>
        <Button
          variant="secondary-outline"
          onClick={handleModalClose}
          aria-label={t('Close modal')}
        >
          <MdClose />
        </Button>
      </Modal.Header>
      {modalBody}
    </Modal>
  );
}

PluginModal.propTypes = {
  show: PropTypes.bool.isRequired,
  closeModal: PropTypes.func.isRequired,
  openModal: PropTypes.func.isRequired,
  updateInvestList: PropTypes.func.isRequired,
  closeInvestModel: PropTypes.func.isRequired,
  openJobs: PropTypes.shape({
    modelID: PropTypes.string,
  }).isRequired,
};

