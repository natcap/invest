import React, { useEffect, useRef, useState } from 'react';
import ReactDom from 'react-dom';
import PropTypes from 'prop-types';

import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import Container from 'react-bootstrap/Container';

import { ipcMainChannels } from '../../../main/ipcMainChannels';

const { ipcRenderer } = window.Workbench.electron;

function LogDisplay(props) {
  const ref = useRef();

  const [autoScroll, setAutoScroll] = useState(true);
  const [prevScrollTop, setPrevScrollTop] = useState(0);

  // A scroll event doesn't tell us whether it was initiated by a user or by code,
  // so we assume all scroll events are user-initiated unless otherwise specified.
  const [userInitiatedScroll, setUserInitiatedScroll] = useState(true);

  let scrollHandlerTimer;

  // `scrollOffsetThreshold` is used to determine when user has scrolled to bottom of window.
  // It includes a buffer to account for imprecision stemming from rounding errors and/or
  // scroll event throttling. 24px (the height of one line of log output) seems to be
  // large enough to detect scroll events we want, and small enough to avoid false positives.
  const scrollOffsetThreshold = 24;

  useEffect(() => {
    if (autoScroll) {
      // Setting `ref.current.scrollTop` will fire a scroll event, which will
      // result in a call to `handleScroll`. To avoid unnecessary operations in
      // `handleScroll`, we flag the next scroll event as _not_ user-initiated.
      setUserInitiatedScroll(false);
      ref.current.scrollTop = ref.current.scrollHeight;
    }
  }, [props.logdata]);

  // Check scroll direction or position IFF scroll event was user-initiated.
  // Always update `prevScrollTop` and reset `userInitiatedScroll` to `true`.
  const handleScroll = () => {
    if (scrollHandlerTimer) {
      clearTimeout(scrollHandlerTimer);
    }
    scrollHandlerTimer = setTimeout(() => {
      const currentScrollTop = ref.current.scrollTop;
      if (userInitiatedScroll) {
        if (autoScroll) {
          // If user has scrolled up, halt auto-scrolling.
          const scrollingUp = (currentScrollTop < prevScrollTop);
          if (scrollingUp) {
            setAutoScroll(false);
          }
        } else {
          // If user has scrolled back to the bottom, resume auto-scrolling.
          const currentScrollOffset = ref.current.scrollHeight - currentScrollTop;
          if (Math.abs(ref.current.offsetHeight - currentScrollOffset) <= scrollOffsetThreshold) {
            setAutoScroll(true);
          }
        }
      }
      setPrevScrollTop(currentScrollTop);
      setUserInitiatedScroll(true);
    }, 10);
  };

  return (
    <Col
      className="text-break"
      id="log-display"
      ref={ref}
      onScroll={handleScroll}
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
    // flushSync will override react18 batched updates
    // and force state updates to happen now. We're managing
    // the rate of updates ourselves in this component via
    // debouncedLogUpdate.
    ReactDom.flushSync(() => {
      this.setState((state) => ({
        logdata: state.logdata.concat(this.cache)
      }));
    });
    this.cache = [];
  }

  /*
   * Cache incoming logger messages in real-time, then move them to react state
   * in batches, at some longer interval. This limits the number of renders in
   * the event of very high volume logging, while still allowing the browser to
   * appear that it is updating in real-time.
   */
  debouncedLogUpdate(data) {
    if (this.validationTimer) {
      clearTimeout(this.validationTimer);
    }
    this.cache.push(data);
    // updates every 10ms will appear to be real-time
    this.validationTimer = setTimeout(this.updateState, 10);
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
