import path from 'path';
import fs from 'fs';
import React from 'react';
import Electron from 'electron';

import Button from 'react-bootstrap/Button';
import ButtonGroup from 'react-bootstrap/ButtonGroup';
import CardGroup from 'react-bootstrap/CardGroup';
import CardColumns from 'react-bootstrap/CardColumns';
import Card from 'react-bootstrap/Card';
import Spinner from 'react-bootstrap/Spinner';
import Col from 'react-bootstrap/Col';
import Row from 'react-bootstrap/Row';

const CACHE_DIR = 'cache' //  for storing state snapshot files
const STATUS_COLOR_MAP = {
  running: 'warning',
  error: 'danger',
  success: 'success'
}

export class HomeTab extends React.Component {

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
          <LoadStateForm
            loadState={this.props.loadState}
            recentSessions={this.props.recentSessions}/>
        </Col>
      </Row>
    );
  }
}


class LoadStateForm extends React.Component {
  
  constructor(props) {
    super(props);
    this.selectFile = this.selectFile.bind(this);
    this.handleClick = this.handleClick.bind(this);
  }

  selectFile(event) {
    const dialog = Electron.remote.dialog;
    // TODO: could add more filters to only show .json
    dialog.showOpenDialog({
      properties: ['openFile']
    }, (filepath) => {
      if (filepath[0]) {
        this.props.loadState(
          path.parse(path.basename(filepath[0])).name); // 0 is safe since we only allow 1 selection
      }
    })
  }

  handleClick(sessionName) {
    this.props.loadState(sessionName);
  }

  render() {

    // Buttons to load each recently saved state
    let recentButtons = [];
    this.props.recentSessions.forEach(session => {
      const name = session[0];
      const status = session[1]['status'];
      const description = session[1]['description'];
      const mtime = session[1]['mtime'];

      recentButtons.push(
        <Card className="text-left" style={{ width: '24rem' }}
          as="button"
          key={name}
          value={name} // TODO: send the actual filename with json ext
          onClick={() => this.handleClick(name)}
          border={STATUS_COLOR_MAP[status] || 'dark'}>
          <Card.Body>
            <Card.Title>
              {name}   
              {status === 'running' && 
               <Spinner as='span' animation='border' size='sm' role='status' aria-hidden='true'/>
              }
            </Card.Title>
            <Card.Text>{description}</Card.Text>
          </Card.Body>
          <Card.Footer className="text-muted">last modified: {mtime}</Card.Footer>
        </Card>
      );
    });
    // Also a button to browse to a cached state file if it's not in recent list
    recentButtons.push(
      <Button
        key="browse"
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
        <div>
          {recentButtons}
        </div>
      </div>
    );
  }
}
