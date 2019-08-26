import React from 'react';

import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import Alert from 'react-bootstrap/Alert';

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
    const jobStatus = this.props.jobStatus;
    const current_err = this.props.logStdErr;
    // Include the stderr in the main log even though it also gets an Alert
    const current_out = this.props.logStdOut + current_err;
    let renderedLog;
    let renderedAlert;

    renderedLog =
        <Col ref={this.content} style={logStyle}>
          {current_out}
        </Col>

    // todo: these states should be mutually exclusive, but I don't have a contract 
    if (current_err) {
      renderedAlert = <Alert variant={'danger'}>{current_err}</Alert>
    }
    if (jobStatus === 0) {
      renderedAlert = <Alert variant={'success'}>{'Model Completed'}</Alert>
    }

    return (
      <React.Fragment>
        <Row>{renderedLog}</Row>
        <Row>{renderedAlert}</Row>
      </React.Fragment>
    );
  }
}