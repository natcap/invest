import React, { useEffect, useState } from 'react';
import PropTypes from 'prop-types';

import { useTranslation } from 'react-i18next';

import Col from 'react-bootstrap/Col';
import Row from 'react-bootstrap/Row';

import PluginDetailPane from './PluginDetailPane';

export default function PluginRegistryTab(props) {
  const {
    registryData,
    activePluginKey,
    activePluginIndex,
    handlePluginClick,
    fetchError,
    installedPlugins,
    addPlugin,
    installLoading,
    installErr,
    installErrMsg,
    installSuccess,
    statusMessage,
    needsMSVC,
    downloadMSVC,
  } = props;
  const [installedPluginNames, setInstalledPluginNames] = useState([]);
  const [installedPluginNamesVersions, setInstalledPluginNamesVersions] = useState([]);
  const [plugins, setPlugins] = useState({});

  const { t } = useTranslation();

  useEffect(() => {
    let installedPluginNameList = [];
    let installedPluginNameVersionList = [];
    for (const pluginID in installedPlugins) {
      let p = installedPlugins[pluginID];
      if (p.hasOwnProperty('packageName')) {
        installedPluginNameVersionList.push(p.packageName + "@" + p.version);
        installedPluginNameList.push(p.packageName);
      }
    };
    setInstalledPluginNames(installedPluginNameList);
    setInstalledPluginNamesVersions(installedPluginNameVersionList);
  }, [installedPlugins]);

  const pluginList = [];
  for (const pluginInfo of registryData) {
    let pluginID = pluginInfo.invest_package_name;
    const listItem = (
      <button
        key={pluginID}
        size="lg"
        className={`registry-list-group-item plugin-registry-button ${activePluginKey === pluginID ? 'active' : ''}`}
        onClick={(e) => handlePluginClick(pluginID)}
      >
        {pluginInfo.plugin_name}
      </button>
    )
    pluginList.push(listItem);
  }

  function getInstallStatus(plugin) {
    if (installedPluginNamesVersions.includes(plugin.invest_package_name + '@' + plugin.version)) {
      return "thisVersionInstalled";
    } else if (installedPluginNames.includes(plugin.invest_package_name)) {
      return "anotherVersionInstalled";
    } else {
      return "notInstalled"
    }
  }

  return (
    <Row>
      <Col sm={3} className="registry-list registry-list-group">
        {pluginList.length
          ? (
            <div className="d-grid">
              {pluginList}
            </div>
          )
          : (
            <p>{t('No plugins found')}</p>
          )
        }
      </Col>
      <Col sm={9} className="registry-pane">
        {activePluginKey.length &&
          <PluginDetailPane
            key={activePluginKey}
            pluginID={activePluginKey}
            plugin={registryData[activePluginIndex]}
            installStatus={getInstallStatus(registryData[activePluginIndex])}
            addPlugin={addPlugin}
            installLoading={installLoading === activePluginKey}
            installErr={installErr === activePluginKey}
            installErrMsg={installErrMsg}
            installSuccess={installSuccess === activePluginKey}
            installDisabled={installLoading && installLoading !== activePluginKey}
            statusMessage={statusMessage}
            needsMSVC={needsMSVC}
            downloadMSVC={downloadMSVC}
          />
        }
      </Col>
    </Row>
  );
}