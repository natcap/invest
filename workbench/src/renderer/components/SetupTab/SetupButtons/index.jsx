import React from 'react';

import Button from 'react-bootstrap/Button';
import OverlayTrigger from 'react-bootstrap/OverlayTrigger';
import Tooltip from 'react-bootstrap/Tooltip';

import SaveFileButton from '../../SaveFileButton';

function HoverText(props) {
  return (
    <OverlayTrigger
      placement="right"
      delay={{ show: 250, hide: 400 }}
      overlay={(
        <Tooltip>
          {props.hoverText}
        </Tooltip>
      )}
    >
      {/* the first child must be a DOM element, not a custom component,
      so that event handlers from OverlayTrigger make it to DOM
      https://github.com/react-bootstrap/react-bootstrap/issues/2208 */}
      <div>
        {props.children}
      </div>
    </OverlayTrigger>
  );
}

export function SaveParametersButtons(props) {
  return (
    <React.Fragment>
      <HoverText
        hoverText={_("Save model setup to a JSON file")}
      >
        <SaveFileButton
          title={_("Save to JSON")}
          defaultTargetPath="invest_args.json"
          func={props.saveJsonFile}
        />
      </HoverText>
      <HoverText
        hoverText={_("Save model setup to a Python script")}
      >
        <SaveFileButton
          title={_("Save to Python script")}
          defaultTargetPath="execute_invest.py"
          func={props.savePythonScript}
        />
      </HoverText>
      <HoverText
        hoverText={_("Export all input data to a compressed archive")}
      >
        <SaveFileButton
          title={_("Save datastack")}
          defaultTargetPath="invest_datastack.tgz"
          func={props.saveDatastack}
        />
      </HoverText>
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
