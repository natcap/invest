import React from 'react';

import { useTranslation } from 'react-i18next';

import Button from 'react-bootstrap/Button';
import Form from 'react-bootstrap/Form';
import OverlayTrigger from 'react-bootstrap/OverlayTrigger';
import Spinner from 'react-bootstrap/Spinner';
import Tooltip from 'react-bootstrap/Tooltip';

export default function InstallButton(props) {
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
