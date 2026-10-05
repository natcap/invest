import React, { useEffect, useState } from 'react';
import PropTypes from 'prop-types';

import Button from 'react-bootstrap/Button';
import Container from 'react-bootstrap/Container';
import Col from 'react-bootstrap/Col';
import Modal from 'react-bootstrap/Modal';
import Nav from 'react-bootstrap/Nav';
import Row from 'react-bootstrap/Row';
import Spinner from 'react-bootstrap/Spinner';
import Tab from 'react-bootstrap/Tab';
import { useTranslation } from 'react-i18next';
import { BsCheckCircle } from "react-icons/bs";
import {
  MdClose,
  MdOutlineWarningAmber
} from 'react-icons/md';

import { ipcMainChannels } from '../../../main/ipcMainChannels';
import { fetchRegistryData } from './services';

import AboutTab from './AboutTab';
import AdvancedSettingsTab from './AdvancedSettingsTab';
import InstalledPluginsTab from './InstalledPluginsTab';
import ManualInstallTab from './ManualInstallTab';
import PluginRegistryTab from './PluginRegistryTab';
import {
  opTypeInstall,
  opTypeUninstall,
  opStatusLoading,
  opStatusSuccess,
  opStatusFailure,
  manualInstallID,
} from './constants';

const { getFilePath, ipcRenderer } = window.Workbench.electron;
const { logger } = window.Workbench;

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

  const defaultAddRemoveState = {
    opType: null,      // install or uninstall
    opStatus: null,    // loading, success, or error
    opErrorMsg: null,  // error message
    opPluginID: null,  // pluginID associated with the op
  }
  const [addRemoveState, setAddRemoveState] = useState(defaultAddRemoveState);
  const [plugins, setPlugins] = useState({});
  const [registryData, setRegistryData] = useState([]);
  const [fetchError, setFetchError] = useState(false);
  const [registryDataLoading, setRegistryDataLoading] = useState(true);
  const [activePluginKey, setActivePluginKey] = useState('');
  const [tabKey, setTabKey] = useState('registry');

  const handleModalClose = () => {
    if (addRemoveState.opStatus !== opStatusLoading) {
      closeModal();
    }
  };

  async function handleFetchRegistryData() {
    try {
      setRegistryDataLoading(true);
      let data = await fetchRegistryData();
      setRegistryDataLoading(false);
      if (data !== null) {
        setRegistryData(data);
        setFetchError(false);
      } else {
        setFetchError(true);
      }
    } catch(error) {
      setRegistryDataLoading(false);
      setFetchError(true);
    }
  }

  const handleRetryFetchRegistryData = () => {
    setFetchError(false);
    handleFetchRegistryData();
  }

  useEffect(() => {
    handleFetchRegistryData();
  }, []);

  useEffect(() => {
    if (Object.keys(registryData).length) {
      setActivePluginKey(registryData[0]['invest_package_name']);
    }
  }, [registryData]);

  function handlePluginClick(pluginKey) {
    setActivePluginKey(pluginKey);
  }

  const addPlugin = (pluginID, url, revision, path, sourceType) => {
    setAddRemoveState({
      opType: opTypeInstall,
      opStatus: opStatusLoading,
      opErrorMsg: "",
      opPluginID: pluginID
    })
    ipcRenderer.invoke(
      ipcMainChannels.ADD_PLUGIN,
      url,       // git url (via manual install or registry)
      revision,  // revision (manual install) or version (registry)
      path,      // local path (manual local install)
      sourceType // 'local_path', 'git_url', or 'registry'
    ).then(() => {
      setAddRemoveState({
        opType: opTypeInstall,
        opStatus: opStatusSuccess,
        opErrorMsg: "",
        opPluginID: pluginID
      })
      updateInvestList();
    }).catch((err) => {
      setAddRemoveState({
        opType: opTypeInstall,
        opStatus: opStatusFailure,
        opErrorMsg: err.toString(),
        opPluginID: pluginID
      })
    });
  };

  const removePlugin = (pluginToRemove) => {
    setAddRemoveState({
      opType: opTypeUninstall,
      opStatus: opStatusLoading,
      opErrorMsg: "",
      opPluginID: pluginToRemove
    })
    openJobs.forEach((job, tabID) => {
      if (job.modelID === pluginToRemove) {
        closeInvestModel(tabID);
      }
    });
    ipcRenderer.invoke(
      ipcMainChannels.REMOVE_PLUGIN, pluginToRemove
    ).then(() => {
      setAddRemoveState({
        opType: opTypeUninstall,
        opStatus: opStatusSuccess,
        opErrorMsg: "",
        opPluginID: pluginToRemove
      })
      updateInvestList();
    }).catch((err) => {
      setAddRemoveState({
        opType: opTypeUninstall,
        opStatus: opStatusFailure,
        opErrorMsg: err.toString(),
        opPluginID: pluginToRemove
      })
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
  }, [addRemoveState]);

  function jumpToInstallMsg() {
    if (addRemoveState.opPluginID === manualInstallID) {
      setTabKey('manual');
    } else {
      // set ActivePluginKey so correct Registry plugin will display,
      // then jump to Registry tab
      setActivePluginKey(addRemoveState.opPluginID);
      setTabKey('registry');
    }
  }

  const { t } = useTranslation();

  const modalBody = (
    <Modal.Body>
      <Tab.Container
        id="plugin-modal-tabs"
        activeKey={tabKey}
        onSelect={(k) => setTabKey(k)}
      >
        <Container>
          <Row>
            <Col sm={2} className="plugin-modal-nav">
              <Nav variant="pills">
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
            <Col sm={10} className="plugin-modal-pane">
              <Tab.Content>
                <Tab.Pane eventKey="registry" className="registry-pane-with-tabs">
                  {registryDataLoading ? (
                    <div className="registry-fetch-status">
                      <Spinner animation="border" role="status" size="sm" className="plugin-spinner" />
                      <span>{t("Loading data from the Plugin Registry...")}</span>
                    </div>
                  ) : fetchError ? (
                    <div className="registry-fetch-status">
                      <MdOutlineWarningAmber className="registry-warning-icon" />
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
                  ) : registryData.length ? (
                    <PluginRegistryTab
                      registryData={registryData}
                      activePluginKey={activePluginKey}
                      handlePluginClick={handlePluginClick}
                      fetchError={fetchError}
                      installedPlugins={plugins}
                      addPlugin={addPlugin}
                      addRemoveState={addRemoveState}
                      statusMessage={statusMessage}
                      needsMSVC={needsMSVC}
                      downloadMSVC={downloadMSVC}
                    />
                  ) : (
                    <p>{t('No plugins found.')}</p>
                  )}
                </Tab.Pane>
                <Tab.Pane eventKey="installed">
                  <InstalledPluginsTab
                    plugins={plugins}
                    removePlugin={removePlugin}
                    addRemoveState={addRemoveState}
                  />
                </Tab.Pane>
                <Tab.Pane eventKey="manual">
                  <ManualInstallTab
                    addPlugin={addPlugin}
                    addRemoveState={addRemoveState}
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
        </Container>
      </Tab.Container>
    </Modal.Body>
  );

  let modalFooter;
  if (addRemoveState.opStatus === opStatusSuccess) {
    if (addRemoveState.opType === opTypeInstall) {
      modalFooter = (
        <>
          <BsCheckCircle className="plugin-modal-icons" />
          <span>
            {t("Installation Success! You can now close this modal and open the plugin from the list of models.")}
          </span>
          <Button
            className="plugin-submit-btn"
            onClick={jumpToInstallMsg}
          >{t("View Details")}</Button>
        </>
      );
    } else if (addRemoveState.opType === opTypeUninstall) {
      modalFooter = (
        <>
          <BsCheckCircle className="plugin-modal-icons" />
          <span>{t("Plugin successfully uninstalled.")}</span>
        </>
      );
    }
  } else if (addRemoveState.opStatus === opStatusFailure) {
    if (addRemoveState.opType === opTypeInstall) {
      modalFooter = (
        <>
          <MdOutlineWarningAmber className="plugin-modal-icons plugin-modal-icons-error" />
          <span>{t("An error occurred during installation.")}</span>
          <Button
            className="plugin-submit-btn"
            onClick={jumpToInstallMsg}
          >{t("View Details")}</Button>
        </>
      );
    } else if (addRemoveState.opType === opTypeUninstall) {
      modalFooter = (
        <>
          <MdOutlineWarningAmber className="plugin-modal-icons plugin-modal-icons-error" />
          <span>{t("An error occurred during uninstallation:")}</span>
          <div className="plugin-error plugin-install-remove-error">
            {addRemoveState.opErrorMsg}
          </div>
          <Button
            className="plugin-submit-btn"
            onClick={() => setTabKey("installed")}
          >{t("View Details")}</Button>
        </>
      );
    }
  } else if (addRemoveState.opStatus === opStatusLoading) {
    if (addRemoveState.opType === opTypeInstall) {
      modalFooter = (
        <>
          <Spinner animation="border" role="status" size="sm" className="plugin-spinner" />
          {t("Installation in progress: ")}{statusMessage}
          <Button
            className="plugin-submit-btn"
            onClick={jumpToInstallMsg}
          >{t("View Installing Plugin")}</Button>
        </>
      );
    } else if (addRemoveState.opType === opTypeUninstall) {
      modalFooter = (
        <>
          <Spinner animation="border" role="status" size="sm" className="plugin-spinner" />
          {t("Uninstallation in progress...")}
        </>
      );
    }
  } else {
    modalFooter = (
      <p>{t("No installation or uninstallation is currently in progress.")}</p>
    )
  }

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
      <Modal.Footer className="plugin-modal-footer">
        <p className="plugin-modal-footer-header">{t("Plugin Installation / Uninstallation Status:")}</p>
        {modalFooter}
      </Modal.Footer>
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
