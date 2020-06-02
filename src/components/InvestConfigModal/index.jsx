import fs from 'fs';
import path from 'path';
import glob from 'glob';
import { spawnSync } from 'child_process';

import React from 'react';
import PropTypes from 'prop-types';
import { remote } from 'electron';

import Col from 'react-bootstrap/Col';
import Row from 'react-bootstrap/Row';
import Form from 'react-bootstrap/Form';
import Button from 'react-bootstrap/Button';
import Modal from 'react-bootstrap/Modal';

const INVEST_REGISTRY_PATH = path.join(
      remote.app.getPath('userData'), 'invest_registry.json')
const MAC_EXE = ['invest', 'server']  // same patterns for linux
const WIN_EXE = ['invest.exe', 'server.exe']


function findInvestExecutables(dir) {
  const patterns = process.platform === 'win32' ? WIN_EXE : MAC_EXE
  const investPattern = patterns[0]
  const serverPattern = patterns[1]
  
  const files = glob.sync(dir + `/**/${investPattern}`)
  if (files.length === 0) {
    alert(`No InVEST installation found in ${dir}`)
    return
  }
  let registry = {}
  files.forEach(file => {
    if (!fs.lstatSync(file).isDirectory()) {
      fs.access(file, fs.constants.X_OK, err => {
        if (err) { console.log(err); return }
      })
      let version;
      console.log(file)
      const invest = spawnSync(file, ['--version'])
      version = `${invest.stdout}`.trim()
      console.log(version)
      if (invest.stderr) {
        console.log(`${invest.stderr}`)
      }
      if (version) {
        const serverExe = path.join(path.dirname(file), serverPattern)
        fs.access(serverExe, fs.constants.X_OK, err => {
          if (err) { 
            console.log(err)
            alert(
              `InVEST version ${version} was found, but is
              not compatible with this desktop client.`)
            return 
          }
        })
        registry[version] = {
          invest: file,
          server: serverExe
        }
        console.log(registry)
      }
    }
  })
  return registry
}

export default class InvestConfigModal extends React.Component {

  constructor(props) {
    super(props);
    this.state = {
      show: true,
      version: this.props.investVersion,
      registry: this.props.investRegistry
    }

    this.handleClose = this.handleClose.bind(this);
    this.handleShow = this.handleShow.bind(this);
    this.handleSubmit = this.handleSubmit.bind(this);
    this.handleChange = this.handleChange.bind(this);
    this.selectFile = this.selectFile.bind(this);
  }

  handleClose() {
    this.setState({
      show: false,
    });
  }

  handleShow() {
    this.setState({show: true});
  }

  handleSubmit(event) {
    // submit the version and exe paths to be appended
    // to the invest registry (or switched to 'active'.
    // Write them to the json,
    // Re-launch the app
    event.preventDefault();
    console.log(event.target.value)
    console.log(this.state.registry)
    let updatedRegistry;
    if (fs.existsSync(INVEST_REGISTRY_PATH)) {
      const investRegistry = JSON.parse(fs.readFileSync(INVEST_REGISTRY_PATH))
      updatedRegistry = Object.assign(this.state.registry, investRegistry)
    } else {
      updatedRegistry = this.state.registry
    }
    updatedRegistry['active'] = this.state.version
    fs.writeFileSync(
      INVEST_REGISTRY_PATH, JSON.stringify(updatedRegistry, null, 2), 'utf8', (err) => {
      if (err) {
        console.log(err);
      }
    })
    remote.app.relaunch()
    remote.app.exit(0)
    
  }

  handleChange(event) {
    // change the select box
    this.setState({version: event.target.value})
  }

  async selectFile(event) {
    /** Handle clicks on browse-button inputs */
    const data = await remote.dialog.showOpenDialog(
      { properties: ['openDirectory'] })
    if (data.filePaths.length) {
      const registry = await findInvestExecutables(data.filePaths[0])
      const updatedRegistry = Object.assign({}, this.state.registry)
      updatedRegistry.registry = Object.assign(updatedRegistry.registry, registry)
      console.log(updatedRegistry)
      this.setState({
        registry: updatedRegistry,
        version: Object.keys(updatedRegistry.registry)[0]
      })
    } else {
      console.log('browse dialog was cancelled')
    }
  }

  render() {
    let versionSelect;
    if (Object.keys(this.state.registry.registry).length) {
      versionSelect = <React.Fragment>
        <Form.Label htmlFor="invest-select">Select InVEST Version:</Form.Label>
        <Form.Control
          id="invest-select"
          as="select"
          name="investSelect"
          value={this.state.version}
          onChange={this.handleChange}>
          {Object.keys(this.state.registry['registry']).map(opt =>
            <option value={opt} key={opt}>{opt}</option>
          )}
        </Form.Control>
      </React.Fragment>
    } else {
      versionSelect = <React.Fragment>
        <h5>Select one of these options:</h5>
      </React.Fragment>
    }

    return (
    
        <Modal centered size="lg" backdrop="static"
          show={this.state.show}
          onHide={this.handleClose}>
          <Form>
            <Modal.Header>
              <Modal.Title>InVEST Configuration</Modal.Title>
            </Modal.Header>
            <Modal.Body>
              <Form.Group as={Row}>
                <Col sm="6">
                  {versionSelect}
                </Col>
                <Col sm="6">
                  <Form.Label htmlFor="invest-download">
                    <h4>1. Download & install InVEST</h4>
                  </Form.Label>
                  <Button block
                    variant="outline-primary"
                    onClick={() => {}}>
                    Download
                  </Button>
                  <Form.Label htmlFor="invest-find" className="mt-4">
                    <h4>2. Use an existing InVEST installation</h4>
                  </Form.Label>
                  <Button block
                    id="invest-find"
                    variant="outline-primary"
                    onClick={this.selectFile}>
                    Browse
                  </Button>
                  <span>
                    <em>Select a folder on your system where InVEST was installed.</em>
                  </span>
                </Col>
              </Form.Group>
            </Modal.Body>
            <Modal.Footer className="justify-content-md-center">
              <Button size="lg"
                variant="primary"
                onClick={this.handleSubmit}
                type="submit"
                disabled={!this.state.version}>
                Restart Application with Selected Version
              </Button>
            </Modal.Footer>
          </Form>
        </Modal>
    )
  }
}