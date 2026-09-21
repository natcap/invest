import React from 'react';
import PropTypes from 'prop-types';

import { useTranslation } from 'react-i18next';
import {
  BsTrash3,
} from 'react-icons/bs';

import Badge from 'react-bootstrap/Badge';
import Card from 'react-bootstrap/Card';
import Button from 'react-bootstrap/Button';

/**
 * Renders a card for a recent job. Clicking the card opens a model tab.
 *
 */
export default function RecentJobCard(props) {
  const {
    job,
    deleteJob,
    handleClick,
  } = props;
  const { t } = useTranslation();

  let badge;
  if (job.type === 'plugin') {
    badge = <Badge className="me-1" bg="secondary">Plugin</Badge>;
  }
  return (
    <Card
      className="col-12 text-start recent-job-card me-2 w-100"
    >
      <Card.Header>
        <div className="badge-container">
          {badge}
        </div>
        <span className="header-title">{job.modelTitle}</span>
        <Button
          variant="outline-light"
          onClick={() => deleteJob(job.hash)}
          className="float-end border-0"
          aria-label="delete"
        >
          <BsTrash3 size="1.5rem" />
        </Button>
      </Card.Header>
      <Card.Body
        className="text-start border-0"
        as="button"
        onClick={() => handleClick(job)}
      >
        <Card.Title>
          <span className="text-heading">{'Workspace: '}</span>
          <span className="text-mono">{job.argsValues.workspace_dir}</span>
        </Card.Title>
        <Card.Title>
          <span className="text-heading">{'Suffix: '}</span>
          <span className="text-mono">{job.argsValues.results_suffix}</span>
        </Card.Title>
        <Card.Footer>
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

RecentJobCard.propTypes = {
  job: PropTypes.shape({
    type: PropTypes.string.isRequired,
    modelTitle: PropTypes.string.isRequired,
    argsValues: PropTypes.shape({
      workspace_dir: PropTypes.string.isRequired,
      results_suffix: PropTypes.string.isRequired,
    }).isRequired,
    status: PropTypes.string,
    hash: PropTypes.string,
    humanTime: PropTypes.string,
  }).isRequired,
  handleClick: PropTypes.func.isRequired,
  deleteJob: PropTypes.func.isRequired,
};
