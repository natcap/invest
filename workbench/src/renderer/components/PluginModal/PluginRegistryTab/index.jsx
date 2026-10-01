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
    addRemoveState,
    statusMessage,
    needsMSVC,
    downloadMSVC,
  } = props;
  const [installedPluginSources, setInstalledPluginSources] = useState([]);
  const [installedPluginSourcesVersions, setInstalledPluginSourcesVersions] = useState([]);
  const [plugins, setPlugins] = useState({});

  const { t } = useTranslation();

  useEffect(() => {
    let installedPluginSourceList = [];
    let installedPluginSourceVersionList = [];
    for (const pluginID in installedPlugins) {
      let p = installedPlugins[pluginID];
      if (p.hasOwnProperty('source') && p.source !== undefined) {
        if (p.source.startsWith("git")) {
          installedPluginSourceVersionList.push(p.source);
          installedPluginSourceList.push(p.source.split("@")[0]);
        }
      }
    };
    setInstalledPluginSources(installedPluginSourceList);
    setInstalledPluginSourcesVersions(installedPluginSourceVersionList);
  }, [installedPlugins]);

  function getInstallStatus(plugin) {
    if (installedPluginSourcesVersions.includes("git+" + plugin.repository_url + '@' + plugin.version)) {
      return thisVersionInstalled;
    } else if (installedPluginSources.includes("git+" + plugin.repository_url)) {
      return anotherVersionInstalled;
    } else {
      return notInstalled;
    }
  }

  return (
    <Tab.Container
      id="plugin-registry-tabs"
      activeKey={activePluginKey}
      onSelect={(k) => handlePluginClick(k)}
    >
      <Row className="plugin-modal-pane-height">
        <Col sm={3} className="plugin-modal-nav">
          <Nav variant="pills" className="flex-column">
            {registryData.map((pluginObject, index) =>
              <Nav.Item
                className="plugin-modal-nav-item"
                key={`${pluginObject.invest_package_name}-nav`}
              >
                <Nav.Link eventKey={pluginObject.invest_package_name}>
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
                  priorInstallationStatus={getInstallStatus(pluginObject)}
                  addPlugin={addPlugin}
                  addRemoveState={addRemoveState}
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
