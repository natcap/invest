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
        aria-label={t('menu')}
        childBsPrefix="outline-secondary"
      >
        <GiHamburgerMenu />
      </Dropdown.Toggle>
      <Dropdown.Menu
        align="right"
        className="app-menu"
      >
        <Dropdown.Item
          as="button"
          onClick={props.openPluginModal}
        >
          {t('Manage Plugins')}
        </Dropdown.Item>
        <Dropdown.Item
          as="button"
          onClick={props.openDownloadModal}
        >
          {t('Download Sample Data')}
        </Dropdown.Item>
        <Dropdown.Item
          as="button"
          onClick={props.openMetadataModal}
        >
          {t('Configure Metadata')}
        </Dropdown.Item>
        <Dropdown.Item
          as="button"
          onClick={props.openChangelogModal}
        >
          {t('View Changelog')}
        </Dropdown.Item>
        <Dropdown.Item
          as="button"
          onClick={props.openSettingsModal}
        >
          {t('Settings')}
        </Dropdown.Item>
      </Dropdown.Menu>
    </Dropdown>
  );
}
