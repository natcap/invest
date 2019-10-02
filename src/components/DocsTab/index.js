import fs from 'fs';
import React from 'react';

export class DocsTab extends React.Component {

  render () {
    let html = 'Local docs not found';
    if (fs.existsSync(this.props.docs)) {
      html = fs.readFileSync(this.props.docs, 'utf8');
    }

    // TODO: this clearly isn't the best way to embed User's Guide html.
    // For one thing, relative paths to resources aren't found because
    // the app looks for them with respect to it's own root.
    return(
        <div><p dangerouslySetInnerHTML={{__html: html}}/></div>
      );
  }
}