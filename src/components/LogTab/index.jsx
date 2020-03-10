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
    this.tail = null;
  }

  componentDidUpdate(prevProps) {
    // if there is a logfile and it's new, start tailing the file.
    if (this.props.logfile && prevProps.logfile !== this.props.logfile) {
      this.tail = new Tail(this.props.logfile, {
        fromBeginning: true
      });
      let logdata = Object.assign('', this.state.logdata);
      this.tail.on('line', (data) => {
        logdata += `${data}` + os.EOL
        this.setState({ logdata: logdata })
      })

    // No new logfile. No existing logdata.
    } else if (this.state.logdata === '') {
      this.setState({logdata: 'Starting...'})

    // No new logfile. Existing logdata. Invest process exited. 
    } else if (['success', 'error'].includes(this.props.jobStatus)) {
      this.tail.unwatch()
    }
  }

  render() {    
    let renderedAlert;
    if (this.props.jobStatus === 'error') {
      renderedAlert = <Alert variant={'danger'}>{this.props.logStdErr}</Alert>
    } else if (this.props.jobStatus === 'success') {
      renderedAlert = <Alert variant={'success'}>{'Model Completed'}</Alert>
    }

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