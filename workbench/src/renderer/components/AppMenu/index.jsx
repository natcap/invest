import React from 'react';

import Dropdown from 'react-bootstrap/Dropdown';
import { useTranslation } from 'react-i18next';
import { GiHamburgerMenu } from 'react-icons/gi';

export default function AppMenu(props) {
  const { t } = useTranslation();

  return (
    <Dropdown>
      <Dropdown.Toggle
        className="app-menu-button"
        aria-label="menu"
        childBsPrefix="outline-secondary"
      >
        <GiHamburgerMenu />
      </Dropdown.Toggle>
      <Dropdown.Menu
        align="right"
        className="shadow"
      >
        <Dropdown.Item
          as="button"
          onClick={props.openPluginModal}
        >
          Manage Plugins
        </Dropdown.Item>
        <Dropdown.Item
          as="button"
          onClick={props.openDownloadModal}
        >
          Download Sample Data
        </Dropdown.Item>
        <Dropdown.Item
          as="button"
          onClick={props.openMetadataModal}
        >
          Configure Metadata
        </Dropdown.Item>
        <Dropdown.Item
          as="button"
          onClick={props.openChangelogModal}
        >
          View Changelog
        </Dropdown.Item>
        <Dropdown.Item
          as="button"
          onClick={props.openSettingsModal}
        >
          Settings
        </Dropdown.Item>
      </Dropdown.Menu>
    </Dropdown>
  );
}
