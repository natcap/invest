import React from 'react';
import PropTypes from 'prop-types';

import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import Container from 'react-bootstrap/Container';

import { Virtuoso } from 'react-virtuoso';

import { ipcMainChannels } from '../../../main/ipcMainChannels';

const { ipcRenderer } = window.Workbench.electron;
const logger = window.Workbench.getLogger('LogTab');

/**
 * A component wrapping a virtualized list for optimal renders
 * of arbitrarily long lists of text data, like an invest logfile.
 */
function LogDisplay(props) {
  return (
    <Col
      className="text-break"
      id="log-display"
    >
      <Virtuoso
        followOutput
        atBottomThreshold={1000} // no adverse effect from large value
        overscan={{ main: 400, reverse: 400 }}
        // 3 props above all help keep scroll at bottom during rapid updates
        style={{ height: '100%' }}
        totalCount={props.logdata.length}
        components={{
          Footer: () => <div className="log-footer" />
        }}
        itemContent={
          (index) => (
            <div
              className={props.logdata[index][1]}
            >
              {props.logdata[index][0]}
            </div>
          )
        }
      />
    </Col>
  );
}

LogDisplay.propTypes = {
  logdata: PropTypes.arrayOf(
    PropTypes.arrayOf(PropTypes.string)
  ).isRequired,
};

export default class LogTab extends React.Component {
  constructor(props) {
    super(props);
    this.validationTimer = null;
    this.cache = [];
    this.state = {
      logdata: [],
    };

    this.updateState = this.updateState.bind(this);
    this.debouncedLogUpdate = this.debouncedLogUpdate.bind(this);
  }

  componentDidMount() {
    const { logfile, executeClicked, tabID } = this.props;
    // This channel is replied to by the invest process stdout listener
    // And by the logfile reader.
    ipcRenderer.on(`invest-stdout-${tabID}`, (data) => {
      this.debouncedLogUpdate(data);
    });
    if (!executeClicked && logfile) {
      ipcRenderer.send(
        ipcMainChannels.INVEST_READ_LOG,
        logfile,
        tabID,
      );
    }
  }

  componentDidUpdate(prevProps) {
    // If we're re-running a model after loading a recent run,
    // we should clear out the logdata when the new run is launched.
    if (this.props.executeClicked && !prevProps.executeClicked) {
      this.setState({ logdata: [] });
    }
  }

  componentWillUnmount() {
    ipcRenderer.removeAllListeners(`invest-stdout-${this.props.tabID}`);
    clearTimeout(this.validationTimer);
  }

  updateState() {
    this.setState((state) => ({
      logdata: state.logdata.concat(this.cache)
    }));
    this.cache = [];
  }

  debouncedLogUpdate(data) {
    if (this.validationTimer) {
      clearTimeout(this.validationTimer);
    }
    this.cache = [...this.cache, data];
    // updates every 10ms will appear to be real-time
    this.validationTimer = setTimeout(this.updateState, 10);
  }

  render() {
    console.log(this.state.logdata)
    return (
      <Container fluid>
        <Row>
          <LogDisplay logdata={this.state.logdata} />
        </Row>
      </Container>
    );
  }
}

LogTab.propTypes = {
  logfile: PropTypes.string,
  executeClicked: PropTypes.bool.isRequired,
  tabID: PropTypes.string.isRequired,
};
