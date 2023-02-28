import React from 'react';
import PropTypes from 'prop-types';
import Alert from 'react-bootstrap/Alert';
import { useTranslation } from 'react-i18next';
import ProgressBar from 'react-bootstrap/ProgressBar';

import Expire from '../Expire';

export default function DownloadProgressBar(props) {
  const [nComplete, nTotal] = props.downloadedNofN;
  const { t, i18n } = useTranslation();
  if (nComplete === 'failed') {
    return (
      <Expire
        className="d-inline"
        delay={props.expireAfter}
      >
        <Alert
          className="d-inline"
          variant="danger"
        >
          {t("Download Failed")}
        </Alert>
      </Expire>
    );
  }
  if (nComplete === nTotal) {
    return (
      <Expire
        className="d-inline"
        delay={props.expireAfter}
      >
        <Alert
          className="d-inline"
          variant="success"
        >
          {t("Download Complete")}
        </Alert>
      </Expire>
    );
  }
  return (
    <ProgressBar
      animated
      max={1}
      now={(nComplete + 1) / nTotal}
      label={
        t('Downloading {{number}} of {{nTotal}}',
          {number: nComplete + 1, nTotal: nTotal}
        )
      }
    />
  );
}

DownloadProgressBar.propTypes = {
  downloadedNofN: PropTypes.arrayOf(
    PropTypes.oneOfType([PropTypes.number, PropTypes.string])
  ).isRequired,
  expireAfter: PropTypes.number.isRequired,
};
