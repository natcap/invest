import React from 'react';

import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import Alert from 'react-bootstrap/Alert';
import Button from 'react-bootstrap/Button';

const logStyle = {
  whiteSpace: 'pre-line',
  maxHeight: '700px',
  overflowY: 'scroll',
};

export class LogDisplay extends React.Component {

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
    const current_out = this.props.logStdOut + current_err;
    let renderedLog;
    let renderedAlert;
    let killButton;

    renderedLog =
        <Col ref={this.content} style={logStyle}>
          {current_out}
        </Col>

    if (current_err) {
      renderedAlert = <Alert variant={'danger'}>{current_err}</Alert>
    } else {
      if (this.props.sessionProgress === 'viz') { // this was set if python exited w/o error
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