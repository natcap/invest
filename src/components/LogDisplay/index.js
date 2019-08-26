import React from 'react';

import Row from 'react-bootstrap/Row';
import Alert from 'react-bootstrap/Alert';

const logStyle = {
  whiteSpace: 'pre-line',
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
    const job_status = this.props.jobStatus;
    const current_err = this.props.logStdErr;
    // Include the stderr in the main log even though it also gets an Alert
    const current_out = this.props.logStdOut + current_err;
    let renderedLog;
    let renderedErr;

    renderedLog = <Row ref={this.content} style={logStyle}>
        {current_out}
      </Row>

    if (current_err) {
      renderedErr = <Row>
          <Alert variant={'danger'}>{current_err}</Alert>
        </Row>
    }

    return (
      <React.Fragment>
        {renderedLog}
        {renderedErr}
      </React.Fragment>
    );
  }
}