import React from 'react';
import PropTypes from 'prop-types';

import Alert from 'react-bootstrap/Alert';
import Button from 'react-bootstrap/Button';
import { useTranslation } from 'react-i18next';

/* Render different Alert contents depending on an InVEST run status */
export default function ModelStatusAlert(props) {
  const { status } = props;
  const { t, i18n } = useTranslation();

  const WorkspaceButton = (
    <Button
      variant="outline-dark"
      onClick={props.handleOpenWorkspace}
      disabled={props.status === 'running'}
    >
      {t('Open Workspace')}
    </Button>
  );

  const CancelButton = (
    <Button
      variant="outline-dark"
      onClick={props.terminateInvestProcess}
    >
      {t('Cancel Run')}
    </Button>
  );

  if (status === 'running') {
    return (
      <Alert variant="secondary">
        {CancelButton}
      </Alert>
    );
  }

  let alertVariant;
  let alertMessage;
  if (status === 'success') {
    alertVariant = 'success';
    alertMessage = t('Model Complete');
  } else if (status === 'error') {
    alertVariant = 'danger';
    alertMessage = t('Error: see log for details');
  } else if (status === 'canceled') {
    alertVariant = 'danger';
    alertMessage = t('Run Canceled');
  }

  return (
    <Alert
      className="text-break"
      variant={alertVariant}
    >
      {alertMessage}
      {WorkspaceButton}
    </Alert>
  );
}

ModelStatusAlert.propTypes = {
  status: PropTypes.oneOf(
    ['running', 'error', 'success', 'canceled']
  ).isRequired,
  terminateInvestProcess: PropTypes.func.isRequired,
  handleOpenWorkspace: PropTypes.func.isRequired,
};
