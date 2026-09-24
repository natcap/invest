import React, { useEffect, useState } from 'react';

import Col from 'react-bootstrap/Col';
import Nav from 'react-bootstrap/Nav';
import Row from 'react-bootstrap/Row';
import Tab from 'react-bootstrap/Tab';
import { useTranslation } from 'react-i18next';

import PluginDetailPane from './PluginDetailPane';

export const thisVersionInstalled = "thisVersionInstalled";
export const anotherVersionInstalled = "anotherVersionInstalled";
export const notInstalled = "notInstalled";

export default function PluginRegistryTab(props) {
  const {
    registryData,
    activePluginKey,
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

  function getInstallStatus(plugin) {
    if (installedPluginNamesVersions.includes(plugin.invest_package_name + '@' + plugin.version)) {
      return thisVersionInstalled;
    } else if (installedPluginNames.includes(plugin.invest_package_name)) {
      return anotherVersionInstalled;
    } else {
      return notInstalled;
    }
  }

  return (
    <Tab.Container id="plugin-registry-tabs" activeKey={activePluginKey}>
      <Row>
        <Col sm={3} className="plugin-modal-nav">
          <Nav variant="pills" className="flex-column">
            {registryData.map((pluginObject, index) =>
              <Nav.Item
                className="plugin-modal-nav-item"
                key={`${pluginObject.invest_package_name}-nav`}
              >
                <Nav.Link
                  eventKey={pluginObject.invest_package_name}
                  onClick={(e) => handlePluginClick(pluginObject.invest_package_name)}
                >
                  {pluginObject.plugin_name}
                </Nav.Link>
              </Nav.Item>
            )}
          </Nav>
        </Col>
        <Col sm={9} className="registry-pane">
          <Tab.Content>
            {registryData.map((pluginObject, index) =>
              <Tab.Pane
                eventKey={pluginObject.invest_package_name}
                key={`${pluginObject.invest_package_name}-tab`}
              >
                <PluginDetailPane
                  pluginID={pluginObject.invest_package_name}
                  plugin={pluginObject}
                  installStatus={getInstallStatus(pluginObject)}
                  addPlugin={addPlugin}
                  installLoading={installLoading === pluginObject.invest_package_name}
                  installErr={installErr === pluginObject.invest_package_name}
                  installErrMsg={installErrMsg}
                  installSuccess={installSuccess === pluginObject.invest_package_name}
                  installDisabled={installLoading && installLoading !== pluginObject.invest_package_name}
                  statusMessage={statusMessage}
                  needsMSVC={needsMSVC}
                  downloadMSVC={downloadMSVC}
                />
              </Tab.Pane>
            )}
          </Tab.Content>
        </Col>
      </Row>
    </Tab.Container>
  );
}
