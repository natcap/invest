import React, { useEffect, useState } from 'react';

import { useTranslation } from 'react-i18next';

import Button from 'react-bootstrap/Button';
import Form from 'react-bootstrap/Form';
import OverlayTrigger from 'react-bootstrap/OverlayTrigger';
import Spinner from 'react-bootstrap/Spinner';
import Tooltip from 'react-bootstrap/Tooltip';
import { MdOpenInNew } from 'react-icons/md';

import { openLinkInBrowser } from '../../../utils';

export function InstallButton(props) {
  const {
    handleAddPluginClick,
    pluginID,
    installLoading,
    installDisabled,
    statusMessage,
  } = props;

  const { t } = useTranslation();

  return (
    <>
      <OverlayTrigger
        trigger={(installLoading || installDisabled) ? ['hover', 'focus'] : []}
        rootClose
        placement="top"
        overlay={
          <Tooltip>
            {t("An installation or uninstallation is in progress.")}
          </Tooltip>
        }
      >
        <Button
          className="plugin-submit-btn"
          aria-disabled={installLoading || installDisabled}
          onClick={handleAddPluginClick}
          aria-describedby={`${pluginID}-plugin-installation-disabled-notice ${pluginID}-plugin-installation-duration-notice`}
        >
          {installLoading
            ? (
              <div className="adding-button">
                <Spinner animation="border" role="status" size="sm" className="plugin-spinner">
                  <span className="visually-hidden">{t('Adding plugin')}</span>
                </Spinner>
                {t(statusMessage)}
              </div>
            )
            : t('Install')
          }
        </Button>
      </OverlayTrigger>
      {installDisabled
        ? (
          <Form.Text
            as="span"
            muted
            id={`${pluginID}-plugin-installation-disabled-notice`}
            className="plugin-form-text"
          >
            {t('An installation is currently in progress. Please wait for it to complete '
              + 'before installing another plugin.')}
          </Form.Text>
        )
        : (
          <Form.Text
            as="span"
            muted
            id={`${pluginID}-plugin-installation-duration-notice`}
            className="plugin-form-text"
          >
            {t('This may take several minutes.')}
          </Form.Text>
        )
      }
    </>
  )
}

export function NeedsMSVC(props) {
  const {
    downloadMSVC
  } = props;

  const { t } = useTranslation();

  return (
    <>
      <h5>
        {t('Microsoft Visual C++ Redistributable must be installed!')}
      </h5>
      <p>
        {t('Plugin features require the ')}
        <a
          href="https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist"
          title="https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist"
          onClick={openLinkInBrowser}
        >
          {t('Microsoft Visual C++ Redistributable')}
          <MdOpenInNew
            aria-label={t("(opens in web browser)")}
          />
        </a>
        {t('. You must download and install the redistributable before continuing.')}
      </p>
      <Button
        className="mt-3"
        onClick={downloadMSVC}
      >
        {t('Continue to download and install')}
      </Button>
    </>
  )
}
