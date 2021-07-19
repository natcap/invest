import React from 'react';
import PropTypes from 'prop-types';

import Button from 'react-bootstrap/Button';
import Table from 'react-bootstrap/Table';
import Card from 'react-bootstrap/Card';
import Container from 'react-bootstrap/Container';
import Col from 'react-bootstrap/Col';
import Row from 'react-bootstrap/Row';

import OpenButton from '../OpenButton';
import InvestJob from '../../InvestJob';

const logger = window.Workbench.getLogger(__filename.split('/').slice(-2).join('/'));

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
    const { value } = event.target;
    const { investList, openInvestModel } = this.props;
    const modelRunName = investList[value].internal_name;
    const job = new InvestJob({
      modelRunName: modelRunName,
      modelHumanName: value
    });
    openInvestModel(job);
  }

  render() {
    const { investList, recentJobs } = this.props;
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
              value={model}
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
        <Col md={5} className="invest-list-table">
          <Table size="sm">
            <tbody>
              {investButtons}
            </tbody>
          </Table>
        </Col>
        <Col className="recent-job-card-col">
          <RecentInvestJobs
            openInvestModel={this.props.openInvestModel}
            recentJobs={recentJobs}
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
  openInvestModel: PropTypes.func.isRequired,
  recentJobs: PropTypes.arrayOf(
    PropTypes.shape({
      modelRunName: PropTypes.string.isRequired,
      modelHumanName: PropTypes.string.isRequired,
      argsValues: PropTypes.object,
      logfile: PropTypes.string,
      status: PropTypes.string,
    })
  ).isRequired,
};

/**
 * Renders a button for each recent invest job.
 */
class RecentInvestJobs extends React.PureComponent {
  constructor(props) {
    super(props);
    this.handleClick = this.handleClick.bind(this);
  }

  handleClick(jobMetadata) {
    this.props.openInvestModel(new InvestJob(jobMetadata));
  }

  render() {
    // Buttons to load each recently saved state
    const recentButtons = [];
    const { recentJobs } = this.props;
    recentJobs.forEach((job) => {
      recentButtons.push(
        <Card
          className="text-left recent-job-card"
          as="button"
          key={job.workspaceHash}
          onClick={() => this.handleClick(job)}
        >
          <Card.Body>
            <Card.Header>
              <span className="header-title">{job.modelHumanName}</span>
            </Card.Header>
            <Card.Title>
              <span className="text-heading">{'Workspace: '}</span>
              <span className="text-mono">{job.argsValues.workspace_dir}</span>
            </Card.Title>
            <Card.Title>
              <span className="text-heading">{'Suffix: '}</span>
              <span className="text-mono">{job.argsValues.results_suffix}</span>
            </Card.Title>
            <Card.Text>{job.description || <em>no description</em>}</Card.Text>
            <Card.Footer className="text-muted">
              <span>{job.humanTime}</span>
              <span className="float-right">
                <em>{job.stdErr || ''}</em>
              </span>
            </Card.Footer>
          </Card.Body>
        </Card>
      );
    });

    return (
      <Container>
        <div className="mb-1">
          <h4 className="d-inline-block">
            Recent runs:
          </h4>
          <OpenButton
            className="float-right"
            openInvestModel={this.props.openInvestModel}
            batchUpdateArgs={this.props.batchUpdateArgs}
          />
        </div>
        {recentButtons.length
          ? (
            <React.Fragment>
              {recentButtons}
            </React.Fragment>
          )
          : (
            <div>
              Try the <b>Open</b> button to setup a model from a sample 
              datastack file (.json) or from an invest model's logfile (.txt)
            </div>
          )}
      </Container>
    );
  }
}

RecentInvestJobs.propTypes = {
  recentJobs: PropTypes.arrayOf(
    PropTypes.shape({
      modelRunName: PropTypes.string.isRequired,
      modelHumanName: PropTypes.string.isRequired,
      argsValues: PropTypes.object,
      logfile: PropTypes.string,
      status: PropTypes.string,
    })
  ).isRequired,
  openInvestModel: PropTypes.func.isRequired,
};
