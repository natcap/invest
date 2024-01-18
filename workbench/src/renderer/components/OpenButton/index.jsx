import React from 'react';
import PropTypes from 'prop-types';

import Button from 'react-bootstrap/Button';
import OverlayTrigger from 'react-bootstrap/OverlayTrigger';
import Tooltip from 'react-bootstrap/Tooltip';
import { withTranslation } from 'react-i18next';

import InvestJob from '../../InvestJob';
import { fetchDatastackFromFile } from '../../server_requests';
import { ipcMainChannels } from '../../../main/ipcMainChannels';

const { ipcRenderer } = window.Workbench.electron;
const { logger } = window.Workbench;

/**
 * Render a button that loads args from a datastack, parameterset, or logfile.
 * Opens a native OS filesystem dialog to browse to a file.
 */
class OpenButton extends React.Component {
  constructor(props) {
    super(props);
    this.browseFile = this.browseFile.bind(this);
  }

  async browseFile() {
    const { t } = this.props;
    const data = await ipcRenderer.invoke(ipcMainChannels.SHOW_OPEN_DIALOG);
    if (!data.canceled) {
      let datastack;
      try {
        datastack = await fetchDatastackFromFile({ filepath: data.filePaths[0] });
      } catch (error) {
        logger.error(error);
        alert(
          t(
            'No InVEST model data can be parsed from the file:\n {{filepath}}',
            { filepath: data.filePaths[0] }
          )
        );
        return;
      }
      const job = new InvestJob({
        modelRunName: datastack.model_run_name,
        modelHumanName: datastack.model_human_name,
        argsValues: datastack.args,
      });
      this.props.openInvestModel(job);
    }
  }

  render() {
    const { t } = this.props;
    const tipText = t('Browse to a datastack (.json) or InVEST logfile (.txt)');
    return (
      <OverlayTrigger
        placement="left"
        delay={{ show: 250, hide: 400 }}
        overlay={<Tooltip>{tipText}</Tooltip>}
      >
        <Button
          className={this.props.className}
          onClick={this.browseFile}
          variant="outline-dark"
        >
          {t("Open")}
        </Button>
      </OverlayTrigger>
    );
  }
}

OpenButton.propTypes = {
  openInvestModel: PropTypes.func.isRequired,
};

export default withTranslation()(OpenButton);
