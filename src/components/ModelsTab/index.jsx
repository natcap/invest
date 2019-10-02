import React from 'react';
import { spawn } from 'child_process';

import Button from 'react-bootstrap/Button';
import ButtonGroup from 'react-bootstrap/ButtonGroup';
import Form from 'react-bootstrap/Form';
import Col from 'react-bootstrap/Col';
import Row from 'react-bootstrap/Row';

const INVEST_EXE = process.env.INVEST

export class ModelsTab extends React.Component {

  constructor(props) {
    super(props);
    this.state = {
      models: {},
      recentSessions: [],
    };

    this.saveSession = this.saveSession.bind(this);
  }

  componentDidMount() {
  	this.makeInvestList();
  }

  makeInvestList() {
    const options = {
      shell: true, // without true, IOError when datastack.py loads json
    };
    const cmdArgs = ['list', '--json']
    const proc = spawn(INVEST_EXE, cmdArgs, options);

    proc.stdout.on('data', (data) => {
      const results = JSON.parse(data.toString());
      this.setState({models:results});
    });

    proc.stderr.on('data', (data) => {
      console.log(`${data}`);
    });

    proc.on('close', (code) => {
      console.log(code);
    });
  }

  saveSession() {
    // write the snapshot to json
    this.props.saveState();
    // and append sessionID to list of recent sessions
    let recentSessions = this.state.recentSessions.slice();
    recentSessions.push(this.props.sessionID);
    this.setState({recentSessions: recentSessions});
  }

  render () {
    const investJSON = this.state.models;
    let buttonItems = [];
    for (const model in investJSON) {
      buttonItems.push(
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
        <Col>
          <ButtonGroup vertical className="mr-2">
            {buttonItems}
          </ButtonGroup>
        </Col>
        <Col>
          <Form.Control
            type="text"
            placeholder={this.props.sessionID}
            value={this.props.sessionID}
            onChange={this.props.setSessionID}
          />
          <Button
            onClick={this.saveSession}
            variant="primary">
            Save State
          </Button>
          <LoadStateForm
            loadState={this.props.loadState}
            recentSessions={this.state.recentSessions}/>
        </Col>
      </Row>
    );
  }
}

class LoadStateForm extends React.Component {
  
  constructor(props) {
    super(props);
    this.state = {
      sessionID: ''
    }
    this.handleTextChange = this.handleTextChange.bind(this);
    this.handleSubmit = this.handleSubmit.bind(this);
    this.handleLink = this.handleLink.bind(this);
  }

  handleTextChange(event) {
    const value = event.target.value;
    this.setState({sessionID: value})
  }

  handleSubmit(event) {
    event.preventDefault();
    this.props.loadState(this.state.sessionID);
  }

  handleLink(event) {
    event.preventDefault();
    const value = event.target.value;
    this.setState(
      {sessionID: value},
      this.props.loadState(value));
  }

  render() {

    let recentButtons = [];
    this.props.recentSessions.forEach(session => {
      recentButtons.push(
        <Button key={session}
          value={session}
          onClick={this.handleLink}
          variant='Link'>
          {session}
        </Button>
      );
    });

    return (
      <div>
        <Form
          onSubmit={this.handleSubmit}>
          <Form.Group>
            <Form.Control
              type="text"
              placeholder="Enter state name"
              value={this.state.sessionID}
              onChange={this.handleTextChange}
            />
          </Form.Group>
          <Button
            type="submit">
            Load State
          </Button>
        </Form>
        <ButtonGroup vertical className="mr-2">
          {recentButtons}
        </ButtonGroup>
      </div>
    );
  }
}