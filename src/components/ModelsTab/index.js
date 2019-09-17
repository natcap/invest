import React from 'react';
import { spawn } from 'child_process';

import Button from 'react-bootstrap/Button';
import ButtonGroup from 'react-bootstrap/ButtonGroup';

const INVEST_EXE = process.env.INVEST

export class ModelsTab extends React.Component {

  constructor(props) {
    super(props);
    this.state = {models: {}};
    this.makeInvestList = this.makeInvestList.bind(this);
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
      console.log(results);
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
        <Button
          value={investJSON[model]['internal_name']}
          onClick={this.props.loadModelSpec}
          variant="outline-success">
          {model}
        </Button>
      );
    }

    return (
      <ButtonGroup vertical className="mr-2" aria-label="First group">
        {buttonItems}
      </ButtonGroup>
    );
  }
}