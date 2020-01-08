import React from 'react';
import {InvestJob} from './InvestJob';
import { getInvestList } from './getInvestList'

export default class App extends React.Component {

  constructor(props) {
    super(props);
    this.state = {
      investList: {}
    };
  }

  async componentDidMount() {
    const investList = await getInvestList();
    this.setState({investList: investList});
  }

  render() {
    return (
      <InvestJob 
        investList={this.state.investList}
      />
    );
  }
}
