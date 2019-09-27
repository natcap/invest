import React, { Component } from 'react';
import fs from 'fs';
import path from 'path';

export default class Viz extends Component {

  render() {

    let html = 'report not found';
    const reportPath = path.join(this.props.workspace, 'report.html');
    if (fs.existsSync(reportPath)) {
      html = fs.readFileSync(reportPath, 'utf8');
    }
    return (
      <div>
        <div><p dangerouslySetInnerHTML={{__html: html}}/></div>
      </div>
    );
  }
}
