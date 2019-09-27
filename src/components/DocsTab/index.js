import fs from 'fs';
import React from 'react';

export class DocsTab extends React.Component {

  render () {
    let html = 'Local docs not found';
    if (fs.existsSync(this.props.docs)) {
      html = fs.readFileSync(this.props.docs, 'utf8');
    }

    return(
        <div><p dangerouslySetInnerHTML={{__html: html}}/></div>
      );
  }
}