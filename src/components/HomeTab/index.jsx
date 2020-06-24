import React from 'react';
import PropTypes from 'prop-types';

import Button from 'react-bootstrap/Button';
import ButtonGroup from 'react-bootstrap/ButtonGroup';
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
  success: '#148F68' // invest green
}

export class HomeTab extends React.PureComponent {
  /** Renders a button for each invest model and for each cached invest job.
  */

  constructor(props) {
    super(props);
    this.handleClick = this.handleClick.bind(this);
  }

  handleClick(event) {
    const modelName = event.target.value;
    this.props.investGetSpec(modelName);
  }

  render () {
    // A button for each model
    const investJSON = this.props.investList;
    let investButtons = [];
    for (const model in investJSON) {
      investButtons.push(
        <tr>
          <td>
            <Button key={model} size="lg" block className="invest-button"
              value={investJSON[model]['internal_name']}
              onClick={this.handleClick}
              variant="link">
              {model}
            </Button>
          </td>
        </tr>
      );
    }

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
            loadState={this.props.loadState}
            recentSessions={this.props.recentSessions}/>
        </Col>
      </Row>
    );
  }
}

HomeTab.propTypes = {
  investList: PropTypes.object,
  investGetSpec: PropTypes.func,
  saveState: PropTypes.func,
  loadState: PropTypes.func,
  recentSessions: PropTypes.array
}


class RecentInvestJobs extends React.PureComponent {
  /** Renders a button for each recent invest job. Also displays job metadata.
  *
  * Recent job metadata is passed here via props, originally loaded from
  * a persistent file when the app is launched.
  */
  
  constructor(props) {
    super(props);
  }

  render() {

    // Buttons to load each recently saved state
    let recentButtons = [];
    this.props.recentSessions.forEach(session => {

      // These properties are required, if they don't exist,
      // the session data was corrupted and should be skipped
      let name, model, workspaceDir;
      try {
        name = session[0];
        model = session[1].model;
        workspaceDir = session[1].workspace.directory;
      } catch(error) {
        return
      }
      
      // These are optional and the rest of the render method
      // should be robust to undefined values
      const suffix = session[1].workspace.suffix;
      const status = session[1]['status'];
      const description = session[1]['description'];
      const datetime = session[1]['humanTime'];

      const headerStyle = {
        backgroundColor: STATUS_COLOR_MAP[status] || 'rgba(23, 162, 184, 0.7)'
      }
      recentButtons.push(
        <Card className="text-left session-card"
          as="button"
          key={name}
          onClick={() => this.props.loadState(session[1]['statefile'])}>
          <Card.Body>
            <Card.Header as="h4" style={headerStyle}>
              {model}
              {status === 'running' && 
                <Spinner as='span' animation='border' size='sm' role='status' aria-hidden='true'/>
              }
            </Card.Header>
            <Card.Title>
              <span className='text-heading'>{'Workspace: '}</span>
              <span className='text-mono'>{workspaceDir}</span>
            </Card.Title>
            <Card.Title>
              <span className='text-heading'>{ suffix && 'Suffix: ' }</span>
              <span className='text-mono'>{suffix}</span>
            </Card.Title>
            <Card.Text>{description || <em>no description</em>}</Card.Text>
          <Card.Footer className="text-muted">{datetime}</Card.Footer>
          </Card.Body>
        </Card>
      );
    });

    return (
      <Container>
        <h4>Recent Sessions:</h4>
        {recentButtons.length
          ? <CardGroup className='session-card-group'>
              {recentButtons}
            </CardGroup>
          : <div>
              No recent sessions yet.<br></br> 
              Try the <b>Load</b> button to load a sample data json file
            </div>
        }
      </Container>
    );
  }
}

RecentInvestJobs.propTypes = {
  loadState: HomeTab.propTypes.loadState,
  recentSessions: HomeTab.propTypes.recentSessions
}