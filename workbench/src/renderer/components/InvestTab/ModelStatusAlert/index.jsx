import React from 'react';
import PropTypes from 'prop-types';

import Alert from 'react-bootstrap/Alert';
import Button from 'react-bootstrap/Button';

/* Render different Alert contents depending on an InVEST run status */
export default function ModelStatusAlert(props) {
  const WorkspaceButton = (
    <Button
      variant="outline-dark"
      onClick={props.handleOpenWorkspace}
      disabled={props.status === 'running'}
    >
      {_("Open Workspace")}
    </Button>
  );

  const CancelButton = (
    <Button
      variant="outline-dark"
      onClick={props.terminateInvestProcess}
    >
      {_("Cancel Run")}
    </Button>
  );

  if (props.status === 'running') {
    return (
      <Alert variant="secondary">
        {CancelButton}
      </Alert>
    );
  }
  if (props.status === 'error') {
    return (
      <Alert
        className="text-break"
        variant="danger"
      >
        {props.finalTraceback}
        {WorkspaceButton}
      </Alert>
    );
  }
  if (props.status === 'success') {
    return (
      <Alert variant="success">
        {_("Model Complete")}
        {WorkspaceButton}
      </Alert>
    );
  }
  return null;
}

ModelStatusAlert.propTypes = {
  status: PropTypes.oneOf(['running', 'error', 'success']).isRequired,
  finalTraceback: PropTypes.string,
  terminateInvestProcess: PropTypes.func.isRequired,
  handleOpenWorkspace: PropTypes.func.isRequired,
};
