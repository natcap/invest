import React, { useEffect, useRef, useState } from 'react';
import PropTypes from 'prop-types';

import Form from 'react-bootstrap/Form';
import Button from 'react-bootstrap/Button';
import Spinner from 'react-bootstrap/Spinner';
import ListGroup from 'react-bootstrap/ListGroup';
import Card from 'react-bootstrap/Card';
import Container from 'react-bootstrap/Container';
import Col from 'react-bootstrap/Col';
import Row from 'react-bootstrap/Row';
import Modal from 'react-bootstrap/Modal';
import { useTranslation } from 'react-i18next';
import { MdOutlineAdd } from "react-icons/md";

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
    this.state = {
      sortedModels: []
    };
    this.handleClick = this.handleClick.bind(this);
  }

  componentDidMount() {
    // sort the model list alphabetically, by the model title,
    // and with special placement of CBC Preprocessor before CBC model.
    const sortedModels = Object.keys(this.props.investList).sort();
    const cbcpElement = 'Coastal Blue Carbon Preprocessor';
    const cbcIdx = sortedModels.indexOf('Coastal Blue Carbon');
    const cbcpIdx = sortedModels.indexOf(cbcpElement);
    if (cbcIdx > -1 && cbcpIdx > -1) {
      sortedModels.splice(cbcpIdx, 1); // remove it
      sortedModels.splice(cbcIdx, 0, cbcpElement); // insert it
    }
    this.setState({
      sortedModels: sortedModels
    });
  }

  handleClick(value) {
    const { investList, openInvestModel } = this.props;
    const modelRunName = investList[value].model_name;
    const job = new InvestJob({
      modelRunName: modelRunName,
      modelHumanName: value
    });
    openInvestModel(job);
  }

  render() {
    const { recentJobs } = this.props;
    const { sortedModels } = this.state;
    // A button in a table row for each model
    const investButtons = [];
    sortedModels.forEach((model) => {
      investButtons.push(
        <ListGroup.Item
          key={model}
          className="invest-button"
          title={model}
          action
          onClick={() => this.handleClick(model)}
        >
          {model}
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
        <PluginModal />
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
      model_name: PropTypes.string,
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
function RecentInvestJobs(props) {
  const { recentJobs, openInvestModel } = props;
  const { t, i18n } = useTranslation();

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
      recentButtons.push(
        <Card
          className="text-left recent-job-card"
          as="button"
          key={job.hash}
          onClick={() => handleClick(job)}
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
      <React.Fragment>
        {recentButtons}
      </React.Fragment>
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
  const [aboutShow, setAboutShow] = useState(false);
  const [url, setURL] = useState(undefined);
  const [loading, setLoading] = useState(false);

  const handleAboutClose = () => setAboutShow(false);
  const handleAboutOpen = () => setAboutShow(true);
  const handleSubmit = (event) => {
    console.log('add plugin', url);
    setLoading(true);
    ipcRenderer.invoke(
      ipcMainChannels.ADD_PLUGIN, url
    ).then((result) => {
      setLoading(false);
      console.log(result)
    })
  }
  const handleChange = (event) => {
    setURL(event.currentTarget.value);
  }

  const { t, i18n } = useTranslation();

  return (
    <React.Fragment>
      <Button onClick={handleAboutOpen}>
        <MdOutlineAdd className="mr-1" />
        {'Add a plugin'}
      </Button>

      <Modal show={aboutShow} onHide={handleAboutClose}>
        <Modal.Header>
          <Modal.Title>{t('Add a plugin')}</Modal.Title>
        </Modal.Header>
        {loading &&
          <Spinner animation="border" role="status">
            <span className="visually-hidden">Loading...</span>
          </Spinner>
        }
        <Modal.Body>
          <Form>
            <Form.Group className="mb-3" controlId="formBasicEmail">
              <Form.Label>Git URL</Form.Label>
              <Form.Control
                type="text"
                placeholder="Enter Git URL"
                onChange={handleChange}/>
            </Form.Group>

            <Button onClick={handleSubmit}>
              Add
            </Button>
          </Form>
        </Modal.Body>
      </Modal>
    </React.Fragment>
  );
}

PluginModal.propTypes = {
};
