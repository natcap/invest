import React from 'react';

import Alert from 'react-bootstrap/Alert';
import Button from 'react-bootstrap/Button';

import { handleClickFindLogfiles } from '../../menubar/handlers';

const logger = window.Workbench.getLogger('ErrorBoundary');

export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    logger.error(error);
    logger.error(errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <Alert variant="dark">
          <h2>Something went wrong.</h2>
          <p>
            Please help us fix this by reporting the problem.
            Follow these steps:
          </p>
          <ol>
            <li>
              Find the Workbench log files using the button below.
              There may be multiple files with a ".log" extension.
            </li>
            <Button
              onClick={handleClickFindLogfiles}
            >
              Find My Logs
            </Button>
            <br />
            <br />
            <li>
              Create a post on our forum and upload all the log files, along with a
              brief description of what happened before you saw this message.
              <br />
              <a
                href="https://community.naturalcapitalproject.org/"
              >
                https://community.naturalcapitalproject.org
              </a>
            </li>
          </ol>
        </Alert>
      );
    }
    return this.props.children;
  }
}
