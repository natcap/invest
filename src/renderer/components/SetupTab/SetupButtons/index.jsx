import React from 'react';

import Button from 'react-bootstrap/Button';

import SaveFileButton from '../../SaveFileButton';

export function SaveParametersButtons(props) {
  return (
    <React.Fragment>
      <SaveFileButton
        title={_("Save to JSON")}
        defaultTargetPath="invest_args.json"
        func={props.saveJsonFile}
      />
      <SaveFileButton
        title={_("Save to Python script")}
        defaultTargetPath="execute_invest.py"
        func={props.savePythonScript}
      />
    </React.Fragment>
  );
}

export function RunButton(props) {
  return (
    <Button
      block
      variant="primary"
      size="lg"
      onClick={props.wrapInvestExecute}
      disabled={props.disabled}
    >
      {props.buttonText}
    </Button>
  );
}
