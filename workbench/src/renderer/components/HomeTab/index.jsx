import React, { useState } from 'react';
import PropTypes from 'prop-types';

import Form from 'react-bootstrap/Form';
import Badge from 'react-bootstrap/Badge';
import Button from 'react-bootstrap/Button';
import Spinner from 'react-bootstrap/Spinner';
import ListGroup from 'react-bootstrap/ListGroup';
import Card from 'react-bootstrap/Card';
import Container from 'react-bootstrap/Container';
import Col from 'react-bootstrap/Col';
import Row from 'react-bootstrap/Row';
import Modal from 'react-bootstrap/Modal';
import { useTranslation } from 'react-i18next';
import { MdOutlineAdd } from 'react-icons/md';

import { ipcMainChannels } from '../../../main/ipcMainChannels';
import OpenButton from '../OpenButton';
import InvestJob from '../../InvestJob';

const { logger } = window.Workbench;
const { ipcRenderer } = window.Workbench.electron;

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
      investButtons.push(
        <ListGroup.Item
          key={modelName}
          action
          onClick={() => this.handleClick(modelId)}
        >
          {
            (investList[modelId].type === 'core' ? <></> : <Badge className="mr-1" variant="secondary">Plugin</Badge>)
          }
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
    })
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
      const modelType = ipcRenderer.sendSync(ipcMainChannels.GET_SETTING, `models.${job.modelRunName}.type`);
      if (modelType === 'plugin') {
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

function PluginModal(props) {
  const { updateInvestList } = props;
  const [showAddPluginModal, setShowAddPluginModal] = useState(false);
  const [url, setURL] = useState(undefined);
  const [loading, setLoading] = useState(false);

  const handleAboutClose = () => setShowAddPluginModal(false);
  const handleAboutOpen = () => setShowAddPluginModal(true);
  const handleSubmit = () => {
    setLoading(true);
    ipcRenderer.invoke(ipcMainChannels.ADD_PLUGIN, url).then(() => {
      setLoading(false);
      setShowAddPluginModal(false);
      updateInvestList();
    });
  };
  const handleChange = (event) => {
    setURL(event.currentTarget.value);
  };

  const { t } = useTranslation();

  return (
    <React.Fragment>
      <Button onClick={handleAboutOpen}>
        <MdOutlineAdd className="mr-1" />
        {t('Add a plugin')}
      </Button>

      <Modal show={showAddPluginModal} onHide={handleAboutClose}>
        <Modal.Header>
          <Modal.Title>{t('Add a plugin')}</Modal.Title>
        </Modal.Header>
        {loading && (
          <Spinner animation="border" role="status">
            <span className="sr-only">Loading...</span>
          </Spinner>
        )}
        <Modal.Body>
          <Form>
            <Form.Group className="mb-3" controlId="formBasicEmail">
              <Form.Label>Git URL</Form.Label>
              <Form.Control
                name="url"
                type="text"
                placeholder={t('Enter Git URL')}
                onChange={handleChange}
              />
            </Form.Group>

            <Button
              name="submit"
              onClick={handleSubmit}
            >
              {t('Add')}
            </Button>
          </Form>
        </Modal.Body>
      </Modal>
    </React.Fragment>
  );
}

PluginModal.propTypes = {
  updateInvestList: PropTypes.func.isRequired,
};
