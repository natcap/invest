import React, { useEffect, useRef } from 'react';
import ReactDom from 'react-dom';
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

export default function LogTab(props) {
  const [logdata, setLogdata] = React.useState([]);
  let validationTimer = null;
  let cache = [];

  React.useEffect(() => {
    const { logfile, executeClicked, tabID } = props;
    // This channel is replied to by the invest process stdout listener
    // And by the logfile reader.
    ipcRenderer.on(`invest-stdout-${tabID}`, (data) => {
      debouncedLogUpdate(data);
    });
    if (!executeClicked && logfile) {
      ipcRenderer.send(
        ipcMainChannels.INVEST_READ_LOG,
        logfile,
        tabID,
      );
    }
    return () => {
      ipcRenderer.removeAllListeners(`invest-stdout-${props.tabID}`);
      clearTimeout(validationTimer);
    };
  }, [props.executeClicked]);


  // componentDidUpdate(prevProps) {
  //   // If we're re-running a model after loading a recent run,
  //   // we should clear out the logdata when the new run is launched.
  //   if (this.props.executeClicked && !prevProps.executeClicked) {
  //     this.setState({ logdata: [] });
  //   }
  // }


  function updateState() {
    // flushSync will override react18 batched updates
    // and force state updates to happen now. We're managing
    // the rate of updates ourselves in this component via
    // debouncedLogUpdate.
    ReactDom.flushSync(() => {
      setLogdata(logdata.concat(cache));
    });
    cache = [];
  }

  /*
   * Cache incoming logger messages in real-time, then move them to react state
   * in batches, at some longer interval. This limits the number of renders in
   * the event of very high volume logging, while still allowing the browser to
   * appear that it is updating in real-time.
   */
  function debouncedLogUpdate(data) {
    if (validationTimer) {
      clearTimeout(validationTimer);
    }
    cache.push(data);
    // updates every 10ms will appear to be real-time
    validationTimer = setTimeout(updateState, 10);
  }

  return (
    <Container fluid>
      <Row>
        <LogDisplay logdata={logdata} />
      </Row>
    </Container>
  );
}

LogTab.propTypes = {
  logfile: PropTypes.string,
  executeClicked: PropTypes.bool.isRequired,
  tabID: PropTypes.string.isRequired,
};
