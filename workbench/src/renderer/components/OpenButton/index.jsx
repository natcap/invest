import React from 'react';
import PropTypes from 'prop-types';

import Button from 'react-bootstrap/Button';
import OverlayTrigger from 'react-bootstrap/OverlayTrigger';
import Tooltip from 'react-bootstrap/Tooltip';
import { withTranslation } from 'react-i18next';

import InvestJob from '../../InvestJob';
import { fetchDatastackFromFile } from '../../server_requests';
import { openDatastack } from '../../utils';
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
    const { investList, openInvestModel } = this.props;
    const data = await ipcRenderer.invoke(ipcMainChannels.SHOW_OPEN_DIALOG);
    if (!data.canceled) {
      let datastack;
      try {
        datastack = await openDatastack(data.filePaths[0]);
      } catch (error) {
        alert(error);
      }
      if (!datastack) {
        return; // the open dialog was cancelled
      }
      const job = new InvestJob({
        modelID: datastack.model_id,
        modelTitle: investList[datastack.model_id].modelTitle,
        argsValues: datastack.args,
        type: investList[datastack.model_id].type,
      });
      openInvestModel(job);
    }
  }

  render() {
    const { t, className } = this.props;
    const tipText = t(
      `Open an InVEST model by loading input parameters from
      a .json, .tgz, or InVEST logfile (.txt)`);
    return (
      <OverlayTrigger
        placement="right"
        delay={{ show: 250, hide: 400 }}
        overlay={<Tooltip>{tipText}</Tooltip>}
      >
        <Button
          className={className}
          onClick={this.browseFile}
          variant="outline-primary"
        >
          {t('Browse to a datastack or InVEST logfile')}
        </Button>
      </OverlayTrigger>
    );
  }
}

OpenButton.propTypes = {
  openInvestModel: PropTypes.func.isRequired,
  investList: PropTypes.shape({
    modelTitle: PropTypes.string,
    type: PropTypes.string,
  }).isRequired,
  t: PropTypes.func.isRequired,
  className: PropTypes.string.isRequired,
};

export default withTranslation()(OpenButton);
