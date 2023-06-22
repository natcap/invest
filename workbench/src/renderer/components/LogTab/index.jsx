import React, { useEffect, useRef } from 'react';
import PropTypes from 'prop-types';

import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import Container from 'react-bootstrap/Container';

import { ipcMainChannels } from '../../../main/ipcMainChannels';

const { ipcRenderer } = window.Workbench.electron;

function LogDisplay(props) {
  const ref = useRef();

  useEffect(() => {
    ref.current.scrollTop = ref.current.scrollHeight;
  }, [props.logdata]);

  return (
    <Col
      className="text-break"
      id="log-display"
      ref={ref}
    >
      {
        props.logdata.map(([line, cls], idx) => (
          <div
            className={cls}
            key={idx}
          >
            {line}
          </div>
        ))
      }
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
    this.state = {
      logdata: [],
    };

    this.updateLog = this.updateLog.bind(this);
  }

  componentDidMount() {
    const { logfile, executeClicked, tabID } = this.props;
    // This channel is replied to by the invest process stdout listener
    // And by the logfile reader.
    ipcRenderer.on(`invest-stdout-${tabID}`, (data) => {
      this.updateLog(data);
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
  }

  updateLog(data) {
    this.setState((state) => ({
      logdata: state.logdata.concat([data])
    }));
  }

  render() {
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
