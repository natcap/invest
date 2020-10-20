import React from 'react';
import PropTypes from 'prop-types';
import { Tail } from 'tail';
import os from 'os';
import { shell } from 'electron'; // eslint-disable-line import/no-extraneous-dependencies
import sanitizeHtml from 'sanitize-html';

import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import Container from 'react-bootstrap/Container';
import Alert from 'react-bootstrap/Alert';
import Button from 'react-bootstrap/Button';

import Portal from '../Portal';
import { getLogger } from '../../logger';

const logger = getLogger(__filename.split('/').slice(-2).join('/'));
//2020-10-16 07:13:04,325
const INVEST_LOG_PATTERN = /^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2},[0-9]{3}/; 

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
      <Col id="log-display" ref={this.content}>
        <div
          id="log-text"
          dangerouslySetInnerHTML={{ __html: this.props.logdata }}
        />
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
    this.unwatchLogfile = this.unwatchLogfile.bind(this);
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
    const { logfile, jobStatus } = this.props;
    if (logfile && (prevProps.logfile !== logfile)) {
      this.tailLogfile(logfile);
    }
    if ((jobStatus !== 'running') && (prevProps.jobStatus !== jobStatus)) {
      // status changed from running to anything else
      this.unwatchLogfile();
    }
  }

  componentWillUnmount() {
    // This does not trigger on browser window close
    this.unwatchLogfile();
  }

  tailLogfile(logfile) {
    const primaryLogger = `${
      this.props.pyModuleName
    }`.split('.').slice(-1)[0];
    try {
      this.tail = new Tail(logfile, {
        fromBeginning: true,
      });
      let logdata = Object.assign('', this.state.logdata);
      this.tail.on('line', (data) => {
        const dataString = `${data}${os.EOL}`;
        let markup;
        if (INVEST_LOG_PATTERN.test(dataString)) {
          if (dataString.includes(`${primaryLogger}.`)) {
            markup = `<p class="invest-log-primary">${dataString}</p>`;
          } else {
            markup = dataString;
          }
        } else {
          markup = `<p class="invest-log-error">${dataString}</p>`;
        }
        logdata += sanitizeHtml(markup, {
          allowedTags: ['p'],
          allowedAttributes: {
            p: ['class']
          },
        });
        // logdata += `${data}${os.EOL}`;
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

  unwatchLogfile() {
    if (this.tail) {
      try {
        logger.debug(`unwatching file: ${this.tail.filename}`);
        this.tail.unwatch();
      } catch (error) {
        logger.error(error.stack);
      }
    }
  }

  render() {
    const { jobStatus } = this.props;
    let ModelStatusAlert;
    const WorkspaceButton = (
      <Button
        variant="outline-dark"
        onClick={this.handleOpenWorkspace}
        disabled={jobStatus === 'running'}
      >
        Open Workspace
      </Button>
    );

    const CancelButton = (
      <Button
        variant="outline-dark"
        onClick={this.props.terminateInvestProcess}
      >
        Cancel Run
      </Button>
    );

    if (jobStatus === 'running') {
      ModelStatusAlert = (
        <Alert variant="secondary">
          {CancelButton}
        </Alert>
      );
    } else if (jobStatus === 'error') {
      let lastCall = '';
      if (this.props.logStdErr) {
        let i = 1;
        while (!lastCall) {
          [lastCall] = `${this.props.logStdErr}`
            .split(`${os.EOL}`).splice(-1 * i);
          i += 1;
        }
      } else {
        // Placeholder for a recent job re-loaded, logStdErr data doesn't persist.
        lastCall = 'Error (see Log for details)';
      }

      ModelStatusAlert = (
        <Alert variant="danger">
          {lastCall}
          {WorkspaceButton}
        </Alert>
      );
    } else if (jobStatus === 'success') {
      ModelStatusAlert = (
        <Alert variant="success">
          Model Complete
          {WorkspaceButton}
        </Alert>
      );
    }
    return (
      <Container fluid>
        <Row>
          <LogDisplay logdata={this.state.logdata} />
        </Row>
        <Portal id="log-alert" elId={this.props.sidebarFooterElementId}>
          {ModelStatusAlert}
        </Portal>
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
