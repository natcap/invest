import React from 'react';

import Button from 'react-bootstrap/Button';

import SaveFileButton from '../../SaveFileButton';

/** Prevent the default case for onDragOver so onDrop event will be fired. */
function dragOverHandler(event) {
  event.preventDefault();
  event.stopPropagation();
  event.dataTransfer.dropEffect = 'none';
}

export function SaveParametersButtons(props) {
  return (
    <React.Fragment>
      <SaveFileButton
        title="Save to JSON"
        defaultTargetPath="invest_args.json"
        func={props.wrapArgsToJsonFile}
      />
      <SaveFileButton
        title="Save to Python script"
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
      onDragOver={dragOverHandler}
    >
      {props.buttonText}
    </Button>
  );
}
