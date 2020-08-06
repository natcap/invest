import React from 'react';
import PropTypes from 'prop-types';

import Button from 'react-bootstrap/Button';
import Table from 'react-bootstrap/Table';
import CardGroup from 'react-bootstrap/CardGroup';
import Card from 'react-bootstrap/Card';
import Spinner from 'react-bootstrap/Spinner';
import Container from 'react-bootstrap/Container';
import Col from 'react-bootstrap/Col';
import Row from 'react-bootstrap/Row';

// these are bootstrap codes for colors
// const STATUS_COLOR_MAP = {
//   running: 'warning',
//   error: 'danger',
//   success: 'success'
// }

// These are the same colors as above
const STATUS_COLOR_MAP = {
  running: 'rgba(23, 162, 184, 0.7)',
  error: 'rgba(220, 53, 69, 0.7)',
  success: '#148F68', // invest green
};

/**
 * Renders a table of buttons for each invest model and
 * a list of cards for each cached invest job.
 */
export default class HomeTab extends React.PureComponent {
  constructor(props) {
    super(props);
    this.handleClick = this.handleClick.bind(this);
  }

  handleClick(event) {
    const modelName = event.target.value;
    this.props.investGetSpec(modelName);
  }

  render() {
    const { investList, loadState, recentSessions } = this.props;
    // A button in a table row for each model
    const investButtons = [];
    Object.keys(investList).forEach((model) => {
      investButtons.push(
        <tr key={model}>
          <td>
            <Button
              className="invest-button"
              block
              size="lg"
              value={investList[model].internal_name}
              onClick={this.handleClick}
              variant="link"
            >
              {model}
            </Button>
          </td>
        </tr>
      );
    });

    return (
      <Row>
        <Col md={5}>
          <Table size="sm" className="invest-list-table">
            <tbody>
              {investButtons}
            </tbody>
          </Table>
        </Col>
        <Col md={7}>
          <RecentInvestJobs
            loadState={loadState}
            recentSessions={recentSessions}
          />
        </Col>
      </Row>
    );
  }
}

HomeTab.propTypes = {
  investList: PropTypes.objectOf(
    PropTypes.shape({
      internal_name: PropTypes.string,
    }),
  ).isRequired,
  investGetSpec: PropTypes.func.isRequired,
  loadState: PropTypes.func.isRequired,
  recentSessions: PropTypes.array,
};
HomeTab.defaultProps = {
  recentSessions: [],
};

/**
 * Renders a button for each recent invest job.
 */
class RecentInvestJobs extends React.PureComponent {
  render() {
    // Buttons to load each recently saved state
    const recentButtons = [];
    const { recentSessions } = this.props;
    recentSessions.forEach((session) => {
      // These properties are required, if they don't exist,
      // the session data was corrupted and should be skipped
      let name;
      let metadata;
      let model;
      let workspaceDir;
      try {
        [name, metadata] = session;
        model = metadata.model;
        workspaceDir = metadata.workspace.directory;
      } catch (error) {
        return;
      }

      // These are optional and the rest of the render method
      // should be robust to undefined values
      const { suffix } = metadata.workspace;
      const { status, description, humanTime } = metadata;

      const headerStyle = {
        backgroundColor: STATUS_COLOR_MAP[status] || 'rgba(23, 162, 184, 0.7)'
      }
      recentButtons.push(
        <Card
          className="text-left session-card"
          as="button"
          key={name}
          onClick={() => this.props.loadState(metadata.statefile)}
        >
          <Card.Body>
            <Card.Header as="h4" style={headerStyle}>
              {model}
              {status === 'running'
                && (
                  <Spinner
                    as="span"
                    animation="border"
                    size="sm"
                    role="status"
                    aria-hidden="true"
                  />
                )
              }
            </Card.Header>
            <Card.Title>
              <span className="text-heading">{'Workspace: '}</span>
              <span className="text-mono">{workspaceDir}</span>
            </Card.Title>
            <Card.Title>
              <span className="text-heading">{ suffix && 'Suffix: ' }</span>
              <span className="text-mono">{suffix}</span>
            </Card.Title>
            <Card.Text>{description || <em>no description</em>}</Card.Text>
            <Card.Footer className="text-muted">{humanTime}</Card.Footer>
          </Card.Body>
        </Card>
      );
    });

    return (
      <Container>
        <label htmlFor="session-card-group">
          <h4>Recent Sessions:</h4>
        </label>
        {recentButtons.length
          ? (
            <CardGroup
              id="session-card-group"
              className="session-card-group"
            >
              {recentButtons}
            </CardGroup>
          )
          : (
            <div>
              No recent sessions yet.
              <br />
              Try the <b>Load</b> button to load a sample data json file
            </div>
          )}
      </Container>
    );
  }
}

RecentInvestJobs.propTypes = {
  loadState: PropTypes.func.isRequired,
  recentSessions: PropTypes.array.isRequired,
};
