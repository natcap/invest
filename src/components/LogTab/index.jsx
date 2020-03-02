import React from 'react';
import { Tail } from 'tail';
import { LazyLog } from 'react-lazylog';
import path from 'path';

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

export class LogTab extends React.Component {

  constructor(props) {
    super(props);
    this.content = React.createRef();
    this.state = {
      logdata: ''
    }
  }

  componentDidUpdate() {
    this.content.current.scrollTop = this.content.current.scrollHeight;
    if (this.props.logfile) {
      let tail = new Tail(this.props.logfile, {
        fromBeginning: true
      });
      tail.on('line', (data) => {
        logdata += `${data}`
        this.setState({ logdata: logdata })
      })
    }
  }

  render() {
    console.log('render log')
    const current_err = this.props.logStdErr;
    // Include the stderr in the main log even though it also gets an Alert
    let renderedLog;
    let renderedAlert;
    let killButton;
    let logdata;

    // if (this.props.logfile) {
    //   renderedLog = <LazyLog
    //     height={600}
    //     stream={true}
    //     url={path.resolve(this.props.logfile)}/>
    // }
    
    renderedLog =
        <Col ref={this.content} style={logStyle}>
          {this.state.logdata}
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
      <Container>
        <Row>
          <Col md="12">
            {renderedLog}
          </Col>
        </Row>
        <Row>{renderedAlert}</Row>
      </Container>
    );
  }
}