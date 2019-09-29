import React from 'react';
import { spawn } from 'child_process';

import Button from 'react-bootstrap/Button';
import ButtonGroup from 'react-bootstrap/ButtonGroup';
import Form from 'react-bootstrap/Form';


const INVEST_EXE = process.env.INVEST

export class ModelsTab extends React.Component {

  constructor(props) {
    super(props);
    this.state = {
      models: {},
      sessionIdToLoad: null
    };
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

  render () {
    const investJSON = this.state.models;
    let buttonItems = [];
    for (const model in investJSON) {
      buttonItems.push(
        <Button key={model}
          value={investJSON[model]['internal_name']}
          onClick={this.props.loadModelSpec}
          variant="outline-success">
          {model}
        </Button>
      );
    }

    return (
      <div>
      <Form.Label>Save Name</Form.Label>
        <Form.Control
          type="text" 
          value={this.props.sessionID}
          onChange={this.props.setSession}
        />
      <Button
        onClick={this.props.saveState}
        variant="primary">
        Save State
      </Button>
      <LoadStateForm
        loadState={this.props.loadState}/>
      <ButtonGroup vertical className="mr-2" aria-label="First group">
        {buttonItems}
      </ButtonGroup>
      </div>
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
  }

  handleTextChange(event) {
    const value = event.target.value;
    this.setState({sessionID: value})
  }

  handleSubmit(event) {
    event.preventDefault();
    this.props.loadState(this.state.sessionID);
  }

  render() {
    return (
      <Form
        onSubmit={this.handleSubmit}>
        <Form.Group>
          <Form.Label>Load Name</Form.Label>
          <Form.Control
            type="text" 
            value={this.state.sessionID}
            onChange={this.handleTextChange}
          />
        </Form.Group>
        <Button
          type="submit">
          Load State
        </Button>
      </Form>
    );
  }
}