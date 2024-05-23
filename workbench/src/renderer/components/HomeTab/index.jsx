import React from 'react';
import PropTypes from 'prop-types';

import Badge from 'react-bootstrap/Badge';
import ListGroup from 'react-bootstrap/ListGroup';
import Card from 'react-bootstrap/Card';
import Container from 'react-bootstrap/Container';
import Col from 'react-bootstrap/Col';
import Row from 'react-bootstrap/Row';
import { useTranslation } from 'react-i18next';

import OpenButton from '../OpenButton';
import InvestJob from '../../InvestJob';
import PluginModal from '../PluginModal';

const { logger } = window.Workbench;

/**
 * Renders a table of buttons for each invest model and
 * a list of cards for each cached invest job.
 */
export default class HomeTab extends React.Component {
  constructor(props) {
    super(props);
    this.handleClick = this.handleClick.bind(this);
  }

  handleClick(value) {
    const { investList, openInvestModel } = this.props;
    const job = new InvestJob({
      modelRunName: value,
      modelHumanName: investList[value].model_name,
      type: investList[value].type,
    });
    openInvestModel(job);
  }

  render() {
    const { recentJobs, investList, openInvestModel, updateInvestList } = this.props;

    let sortedModelIds = {};
    if (investList) {
      // sort the model list alphabetically, by the model title,
      // and with special placement of CBC Preprocessor before CBC model.
      sortedModelIds = Object.keys(investList).sort(
        (a, b) => {
          if (investList[a].model_name > investList[b].model_name) {
            return 1;
          }
          if (investList[b].model_name > investList[a].model_name) {
            return -1;
          }
          return 0;
        }
      );
      const cbcpElement = 'coastal_blue_carbon_preprocessor';
      const cbcIdx = sortedModelIds.indexOf('coastal_blue_carbon');
      const cbcpIdx = sortedModelIds.indexOf(cbcpElement);
      if (cbcIdx > -1 && cbcpIdx > -1) {
        sortedModelIds.splice(cbcpIdx, 1); // remove it
        sortedModelIds.splice(cbcIdx, 0, cbcpElement); // insert it
      }
    }

    // A button in a table row for each model
    const investButtons = [];
    sortedModelIds.forEach((modelId) => {
      const modelName = investList[modelId].model_name;
      let badge;
      if (investList[modelId].type === 'plugin') {
        badge = <Badge className="mr-1" variant="secondary">Plugin</Badge>;
      }
      investButtons.push(
        <ListGroup.Item
          key={modelName}
          name={modelName}
          action
          onClick={() => this.handleClick(modelId)}
        >
          { badge }
          <span className="invest-button">{modelName}</span>
        </ListGroup.Item>
      );
    });

    return (
      <Row>
        <Col md={6} className="invest-list-container">
          <ListGroup className="invest-list-group">
            {investButtons}
          </ListGroup>
        </Col>
        <PluginModal
          updateInvestList={updateInvestList}
        />
        <Col className="recent-job-card-col">
          <RecentInvestJobs
            openInvestModel={openInvestModel}
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
      model_name: PropTypes.string,
      type: PropTypes.string,
    })
  ).isRequired,
  updateInvestList: PropTypes.func.isRequired,
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
function RecentInvestJobs(props) {
  const { recentJobs, openInvestModel } = props;
  const { t } = useTranslation();

  const handleClick = (jobMetadata) => {
    try {
      openInvestModel(new InvestJob(jobMetadata));
    } catch (error) {
      logger.debug(error);
    }
  };

  const recentButtons = [];
  recentJobs.forEach((job) => {
    if (job && job.argsValues && job.modelHumanName) {
      let badge;
      if (job.type === 'plugin') {
        badge = <Badge className="mr-1" variant="secondary">Plugin</Badge>;
      }
      recentButtons.push(
        <Card
          className="text-left recent-job-card"
          as="button"
          key={job.hash}
          onClick={() => handleClick(job)}
        >
          <Card.Body>
            <Card.Header>
              {badge}
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
            <Card.Footer className="text-muted">
              <span className="timestamp">{job.humanTime}</span>
              <span className="status">
                {(job.status === 'success'
                  ? <span className="status-success">{t('Model Complete')}</span>
                  : <span className="status-error">{job.status}</span>
                )}
              </span>
            </Card.Footer>
          </Card.Body>
        </Card>
      );
    }
  });

  return (
    <>
      <Container>
        <Row>
          <Col className="recent-header-col">
            {recentButtons.length
              ? (
                <h4>
                  {t('Recent runs:')}
                </h4>
              )
              : (
                <div className="default-text">
                  {t("Set up a model from a sample datastack file (.json) " +
                     "or from an InVEST model's logfile (.txt): ")}
                </div>
              )}
          </Col>
          <Col className="open-button-col">
            <OpenButton
              className="mr-2"
              openInvestModel={openInvestModel}
            />
          </Col>
        </Row>
      </Container>
      {recentButtons}
    </>
  );
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
