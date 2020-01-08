import path from 'path';
import React from 'react';
import Electron from 'electron';

import Button from 'react-bootstrap/Button';
import ButtonGroup from 'react-bootstrap/ButtonGroup';
import Col from 'react-bootstrap/Col';
import Row from 'react-bootstrap/Row';


export class ModelsTab extends React.Component {

  constructor(props) {
    super(props);
  }

  render () {
    // A button for each model
    const investJSON = this.props.investList;
    let investButtons = [];
    for (const model in investJSON) {
      investButtons.push(
        <Button key={model}
          value={investJSON[model]['internal_name']}
          onClick={this.props.investGetSpec}
          variant="outline-success">
          {model}
        </Button>
      );
    }

    return (
      <Row>
        <Col md={6}>
          <ButtonGroup vertical className="mt-2">
            {investButtons}
          </ButtonGroup>
        </Col>
        <Col md={6}>
          <Row className="mt-2">
            <LoadStateForm
              loadState={this.props.loadState}
              recentSessions={this.props.recentSessions}/>
          </Row>
        </Col>
      </Row>
    );
  }
}


class LoadStateForm extends React.Component {
  
  constructor(props) {
    super(props);
    this.state = {
      session_id_to_load: ''
    }
    this.selectFile = this.selectFile.bind(this);
  }

  selectFile(event) {
    const dialog = Electron.remote.dialog;
    // TODO: could add more filters based on argType (e.g. only show .csv)
    dialog.showOpenDialog({
      properties: ['openFile']
    }, (filepath) => {
      if (filepath[0]) {
        this.props.loadState(
          path.parse(path.basename(filepath[0])).name); // 0 is safe since we only allow 1 selection
      }
    })
  }

  render() {

    // Buttons to load each recently saved state
    let recentButtons = [];
    this.props.recentSessions.forEach(session => {
      recentButtons.push(
        <Button  className="text-left"
          key={session}
          value={session}
          onClick={this.handleLink}
          variant='outline-dark'>
          {session}
        </Button>
      );
    });
    // Also a button to browse to a cached state file if it's not in recent list
    recentButtons.push(
      <Button
        type="submit"
        variant="secondary"
        onClick={this.selectFile}>
        Browse for saved session
      </Button>
    );

    return (
      <div>
        <div>
          Select Recent Session:
        </div>
        <ButtonGroup vertical className="mt-2">
          {recentButtons}
        </ButtonGroup>
      </div>
    );
  }
}
