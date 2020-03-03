import React from 'react';
import { Tail } from 'tail';
import path from 'path';
import os from 'os';

import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import Container from 'react-bootstrap/Container';
import Alert from 'react-bootstrap/Alert';
import Button from 'react-bootstrap/Button';

const logStyle = {
  whiteSpace: 'pre-line',
  maxHeight: '700px',
  overflowY: 'scroll',
};

class LogDisplay extends React.Component {

  constructor(props) {
    super(props);
    this.content = React.createRef();
  }

  componentDidUpdate() {
    this.content.current.scrollTop = this.content.current.scrollHeight;
  }

  render() {
    return (
      <Col ref={this.content} style={logStyle}>
        {this.props.logdata}
      </Col>
      );
  }
}

export class LogTab extends React.Component {

  constructor(props) {
    super(props);
    this.state = {
      logdata: ''
    }
  }

  componentDidUpdate(prevProps) {
    if (prevProps.logfile !== this.props.logfile && this.props.logfile) {
      let tail = new Tail(this.props.logfile, {
        fromBeginning: true
      });
      let logdata = Object.assign('', this.state.logdata);
      tail.on('line', (data) => {
        logdata += `${data}` + os.EOL
        this.setState({ logdata: logdata })
      })
    } else if (this.state.logdata === '') {
      this.setState({logdata: 'Starting...'})
    }
  }

  render() {
    const current_err = this.props.logStdErr;
    
    let renderedAlert;
    if (current_err) {
      renderedAlert = <Alert variant={'danger'}>{current_err}</Alert>
    } else {
      if (this.props.sessionProgress === 'results') { // this was set if python exited w/o error
        renderedAlert = <Alert variant={'success'}>{'Model Completed'}</Alert>
      }
    }

    let killButton = 
      <Button
        variant="primary" 
        size="lg"
        onClick={this.props.investKill}>
        Kill Subprocess
      </Button>

    return (
      <Container>
        <Row>
          <LogDisplay logdata={this.state.logdata}/>
        </Row>
        <Row>{renderedAlert}</Row>
      </Container>
    );
  }
}