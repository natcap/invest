import React, {useState} from 'react';
import {InvestJob} from './components';
import Tabs from 'react-bootstrap/Tabs';
import Tab from 'react-bootstrap/Tab';

function ControlledTabs() {
  const [tabkey, setTab] = useState('setup');

  return (
    <Tabs id="controlled-tab-example" activeKey={tabkey} onSelect={k => setTab(k)}>
      <Tab eventKey="setup" title="Setup">
        <InvestJob />
      </Tab>
      <Tab eventKey="log" title="Log">
        <InvestJob />
      </Tab>
      <Tab eventKey="viz" title="Viz" disabled>
        <InvestJob />
      </Tab>
    </Tabs>
  );
}

export default class App extends React.Component {
  render() {
  	return (
			<ControlledTabs />
    );
	};
}
