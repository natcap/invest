import React from 'react';
import PropTypes from 'prop-types';
import { Tail } from 'tail';
import os from 'os';
import { shell } from 'electron';

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

LogDisplay.propTypes = {
  logdata: PropTypes.string
}

export class LogTab extends React.Component {

  constructor(props) {
    super(props);
    this.state = {
      logdata: ''
    }
    this.tail = null;
    this.handleOpenWorkspace = this.handleOpenWorkspace.bind(this);
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

  handleOpenWorkspace() {
    shell.showItemInFolder(this.props.logfile)
  }

  render() {    
    let RenderedAlert;
    const WorkspaceButton = <Button className='float-right float-bottom'
      variant='outline-dark'
      onClick={this.handleOpenWorkspace}
      disabled={this.props.jobStatus === 'running'}>
      Open Workspace
    </Button>

    if (this.props.jobStatus === 'error') {
      RenderedAlert = <Alert className='py-4 mt-3'
        variant={'danger'}>
        {this.props.logStdErr}
        {WorkspaceButton}
      </Alert>
    } else if (this.props.jobStatus === 'success') {
      RenderedAlert = <Alert className='py-4 mt-3'
        variant={'success'}>
        <span>Model Completed</span>
        {WorkspaceButton}
      </Alert>
    }


    return (
      <Container>
        <Row>
          <LogDisplay logdata={this.state.logdata}/>
        </Row>
        <Row>
          <Col>
            {RenderedAlert}
          </Col>
        </Row>
      </Container>
    );
  }
}

LogTab.propTypes = {
  jobStatus: PropTypes.string,
  logfile: PropTypes.string,
  logStdErr: PropTypes.string
}