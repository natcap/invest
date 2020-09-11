import React from 'react';
import PropTypes from 'prop-types';
import { Tail } from 'tail';
import os from 'os';
import { shell } from 'electron'; // eslint-disable-line import/no-extraneous-dependencies

import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import Container from 'react-bootstrap/Container';
import Alert from 'react-bootstrap/Alert';
import Button from 'react-bootstrap/Button';

import { getLogger } from '../../logger';

const logger = getLogger(__filename.split('/').slice(-2).join('/'));

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
  logdata: PropTypes.string,
};

export default class LogTab extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      logdata: null,
    };
    this.tail = null;
    this.handleOpenWorkspace = this.handleOpenWorkspace.bind(this);
    this.tailLogfile = this.tailLogfile.bind(this);
  }

  componentDidMount() {
    const { logfile } = this.props;
    if (logfile) {
      this.tailLogfile(logfile);
    }
  }

  componentDidUpdate(prevProps) {
    // Re-executing a model will generate a new logfile
    // so need to update to tail the new file.
    const { logfile } = this.props;
    if (logfile && (prevProps.logfile !== logfile)) {
      this.tailLogfile(logfile);
    }
  }

  componentWillUnmount() {
    if (this.tail) {
      try {
        this.tail.unwatch();
      } catch (error) {
        logger.error(error.stack);
      }
    }
  }

  tailLogfile(logfile) {
    try {
      this.tail = new Tail(logfile, {
        fromBeginning: true,
      });
      let logdata = Object.assign('', this.state.logdata);
      this.tail.on('line', (data) => {
        logdata += `${data}${os.EOL}`;
        this.setState({ logdata: logdata });
      });
      this.tail.on('error', (error) => {
        logger.error(error);
      });
    } catch (error) {
      this.setState({
        logdata: `Logfile is missing: ${os.EOL}${logfile}`
      });
      logger.error(`Not able to read ${logfile}`);
      logger.error(error.stack);
    }
  }

  handleOpenWorkspace() {
    shell.showItemInFolder(this.props.logfile);
  }

  render() {
    const { jobStatus } = this.props;
    let ModelStatusAlert;
    const WorkspaceButton = (
      <Button
        className="float-right float-bottom"
        variant="outline-dark"
        onClick={this.handleOpenWorkspace}
        disabled={jobStatus === 'running'}
      >
        Open Workspace
      </Button>
    );

    if (jobStatus === 'running') {
      ModelStatusAlert = (
        <Alert className="py-4 mt-3" variant="secondary">
          <Button
            className="float-right float-bottom"
            variant="outline-dark"
            onClick={this.props.killInvestProcess}
          >
            Cancel Run
          </Button>
        </Alert>
      );
    } else if (jobStatus === 'error') {
      ModelStatusAlert = (
        <Alert className="py-4 mt-3" variant="danger">
          {this.props.logStdErr}
          {WorkspaceButton}
        </Alert>
      );
    } else if (jobStatus === 'success') {
      ModelStatusAlert = (
        <Alert className="py-4 mt-3" variant="success">
          <span>Model Completed</span>
          {WorkspaceButton}
        </Alert>
      );
    } else if (jobStatus === 'canceled') {
      ModelStatusAlert = (
        <Alert className="py-4 mt-3" variant="warning">
          <span>Run Canceled</span>
          {WorkspaceButton}
        </Alert>
      );
    }

    return (
      <Container>
        <Row>
          <LogDisplay logdata={this.state.logdata} />
        </Row>
        <Row>
          <Col>
            {ModelStatusAlert}
          </Col>
        </Row>
      </Container>
    );
  }
}

LogTab.propTypes = {
  jobStatus: PropTypes.string,
  logfile: PropTypes.string,
  logStdErr: PropTypes.string,
};
LogTab.defaultProps = {
  jobStatus: undefined,
  logfile: undefined,
  logStdErr: undefined,
};
