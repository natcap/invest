import fs from 'fs';
import path from 'path';
import React from 'react';
import { spawn } from 'child_process';
import request from 'request';

import Button from 'react-bootstrap/Button';
import ButtonGroup from 'react-bootstrap/ButtonGroup';
import Form from 'react-bootstrap/Form';
import Col from 'react-bootstrap/Col';
import Row from 'react-bootstrap/Row';

const INVEST_EXE = process.env.INVEST
const CACHE_DIR = 'cache'

export class ModelsTab extends React.Component {

  constructor(props) {
    super(props);
    this.state = {
      models: {}, // stores result of `invest list`
      recentSessions: [],
    };

    this.onSaveClick = this.onSaveClick.bind(this);
  }

  componentDidMount() {
  	this.makeInvestList();
    const recentSessions = findRecentSessions(CACHE_DIR);
    this.setState({recentSessions: recentSessions});
  }

  makeInvestList() {
    request.get(
      'http://localhost:5000/models',
      (error, response, body) => {
        if (!error && response.statusCode == 200) {
          const models = JSON.parse(body);
          this.setState({models: models});
        } else {
          console.log('Status: ' + response.statusCode)
          console.log('Error: ' + error.message)
        }
      }
    );
  }

  onSaveClick(event) {
    event.preventDefault();
    this.props.saveState();
    // and append sessionID to list of recent sessions
    let recentSessions = Object.assign([], this.state.recentSessions);
    recentSessions.unshift(this.props.sessionID);
    this.setState({recentSessions: recentSessions});
  }

  render () {
    // A button for each model
    const investJSON = this.state.models;
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
            <SaveStateForm
              sessionID={this.props.sessionID}
              setSessionID={this.props.setSessionID}
              onSaveClick={this.onSaveClick}/>
          </Row>
          <Row className="mt-2">
            <LoadStateForm
              loadState={this.props.loadState}
              recentSessions={this.state.recentSessions}/>
          </Row>
        </Col>
      </Row>
    );
  }
}

function SaveStateForm(props) {
  return(
    <Form
      onSubmit={props.onSaveClick}>
      <Form.Control
        type="text"
        placeholder={props.sessionID}
        value={props.sessionID}
        onChange={props.setSessionID}
      />
      <Button
        onClick={props.onSaveClick}
        variant="primary">
        Save State
      </Button>
    </Form>
  );
}

class LoadStateForm extends React.Component {
  
  constructor(props) {
    super(props);
    this.state = {
      session_id_to_load: ''
    }
    this.handleTextChange = this.handleTextChange.bind(this);
    this.handleSubmit = this.handleSubmit.bind(this);
    this.handleLink = this.handleLink.bind(this);
  }

  handleTextChange(event) {
    const value = event.target.value;
    this.setState({session_id_to_load: value})
  }

  handleSubmit(event) {
    event.preventDefault();
    this.props.loadState(this.state.session_id_to_load);
  }

  handleLink(event) {
    event.preventDefault();
    const value = event.target.value;
    this.setState(
      {session_id_to_load: value},
      this.props.loadState(value));
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

    return (
      <div>
        <Form
          onSubmit={this.handleSubmit}>
          <Form.Control
            type="text"
            placeholder="Enter state name"
            value={this.state.session_id_to_load}
            onChange={this.handleTextChange}/>
          <Button
            type="submit">
            Load State
          </Button>
        </Form>
        <div>
          Recent Sessions:
        </div>
        <ButtonGroup vertical className="mt-2">
          {recentButtons}
        </ButtonGroup>
      </div>
    );
  }
}

function findRecentSessions(cache_dir) {
  // Populate recentSessions from list of files in cache dir
  // sorted by modified time.

  // TODO: check that files are actually state config files
  // before putting them on the array
  const files = fs.readdirSync(cache_dir);

  // reverse sort (b - a) based on last-modified time
  const sortedFiles = files.sort(function(a, b) {
    return fs.statSync(path.join(cache_dir, b)).mtimeMs -
         fs.statSync(path.join(cache_dir, a)).mtimeMs
  });
  // trim off extension, since that is how sessions
  // were named orginally
  return (sortedFiles
    .map(f => path.parse(f).name)
    .slice(0, 15) // max 15 items returned
  );
}