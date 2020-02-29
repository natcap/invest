import React from 'react';
import { Tail } from 'tail';

import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import Alert from 'react-bootstrap/Alert';
import Button from 'react-bootstrap/Button';

const logStyle = {
  whiteSpace: 'pre-line',
  maxHeight: '700px',
  overflowY: 'scroll',
};

export class LogTab extends React.Component {

  constructor(props) {
    super(props);
    this.content = React.createRef();
  }

  componentDidUpdate() {
    this.content.current.scrollTop = this.content.current.scrollHeight;
  }

  render() {
    const current_err = this.props.logStdErr;
    // Include the stderr in the main log even though it also gets an Alert
    // const current_out = this.props.logStdOut + current_err;
    let renderedLog;
    let renderedAlert;
    let killButton;
    let current_out = '';

    if (this.props.logfile) {
      console.log(this.props.logfile);
      tail = new Tail(this.props.logfile);
      tail.on('line', function(data) {
        current_out + data
      })
    }

    renderedLog =
        <Col ref={this.content} style={logStyle}>
          {current_out}
        </Col>

    if (current_err) {
      renderedAlert = <Alert variant={'danger'}>{current_err}</Alert>
    } else {
      if (this.props.sessionProgress === 'results') { // this was set if python exited w/o error
        renderedAlert = <Alert variant={'success'}>{'Model Completed'}</Alert>
      }
    }

    killButton = 
      <Button
        variant="primary" 
        size="lg"
        onClick={this.props.investKill}>
        Kill Subprocess
      </Button>

    return (
      <React.Fragment>
        <Row>{renderedLog}</Row>
        <Row>{renderedAlert}</Row>
      </React.Fragment>
    );
  }
}