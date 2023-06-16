import React from 'react';

import Alert from 'react-bootstrap/Alert';
import Button from 'react-bootstrap/Button';
import { withTranslation } from 'react-i18next';

import { handleClickFindLogfiles } from '../../menubar/handlers';

const { logger } = window.Workbench;

class ErrorBoundary extends React.Component {
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
    const { t } = this.props;
    if (this.state.hasError) {
      return (
        <Alert className="error-boundary">
          <h2>{`\u{1F986}  ${t('Something went wrong')}`}</h2>
          <p>
            <em>
              {t('Please help us fix this by reporting the problem.' +
                  'You may follow these steps:')}
            </em>
          </p>
          <ol>
            <li>
              <b>{t('Find the Workbench log files using the button below.')}</b>
              {t('There may be multiple files with a ".log" extension.')}
            </li>
            <Button
              onClick={handleClickFindLogfiles}
            >
              {t('Find My Logs')}
            </Button>
            <br />
            <br />
            <li>
              <b>{t('Create a post on our forum ')}</b>
              {t('and upload all the log files, along with a brief description ' +
                 'of what happened before you saw this message.')}
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

export default withTranslation()(ErrorBoundary);
