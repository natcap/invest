import React, { Component } from 'react';
import fs from 'fs';
import path from 'path';

export default class Viz extends Component {

  render() {

    let html = 'report not found';
    const basename = 'report';
    const ext = '.html';
    let reportName = basename + ext;

    if (this.props.workspace.suffix) {
      reportName = basename + '_' + this.props.workspace.suffix + ext;
    }
    const reportPath = path.join(this.props.workspace.directory, reportName);
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
