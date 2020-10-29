import crypto from 'crypto';

import localforage from 'localforage';

import { getLogger } from './logger';

const logger = getLogger(__filename.split('/').slice(-1)[0]);

const INDEX = 'sortedJobs';

/**
 * Create an object to hold properties associated with an Invest Job.
 *
 * @param  {string} modelRunName - invest model name to be passed to `invest run`
 * @param  {string} modelHumanName - the colloquial name of the invest model
 * @param  {object} argsValues - an invest "args dictionary" with initial values
 * @param  {object} workspace - with keys for invest workspace directory and suffix
 * @param  {string} logfile - path to an existing invest logfile
 * @param  {string} status - indicates how the job exited, if it's a recent job.
 */
export default class Job {
  static async init() {
    if (!await localforage.getItem(INDEX)) {
      localforage.setItem(INDEX, []);
    }
  }

  static async getJobStore() {
    let jobArray = [];
    const sortedKeys = await localforage.getItem(INDEX);
    if (sortedKeys) {
      jobArray = await Promise.all(sortedKeys.map(
        (key) => localforage.getItem(key)
      ));
    }
    return jobArray;
  }

  constructor(
    modelRunName, modelHumanName, argsValues, workspace, logfile, status
  ) {
    this.metadata = {};
    if (workspace && modelRunName) {
      this.metadata.workspaceHash = crypto.createHash('sha1').update(
        `${modelRunName}${JSON.stringify(workspace)}`
      ).digest('hex');
    }
    this.metadata.modelRunName = modelRunName;
    this.metadata.modelHumanName = modelHumanName;
    this.metadata.argsValues = argsValues;
    this.metadata.workspace = workspace;
    this.metadata.logfile = logfile;
    this.metadata.status = status;
    this.metadata.humanTime = new Date().toLocaleString();
    this.metadata.systemTime = new Date().getTime();

    this.save = this.save.bind(this);
    this.setProperty = this.setProperty.bind(this);
  }

  setProperty(key, value) {
    this.metadata[key] = value;
  }

  async save() {
    this.metadata.humanTime = new Date().toLocaleString();
    this.metadata.systemTime = new Date().getTime();
    const sortedKeys = await localforage.getItem(INDEX);
    // If this key already exists, make sure not to duplicate it,
    // and make sure to move it to the front
    const idx = sortedKeys.indexOf(this.metadata.workspaceHash);
    if (idx > 0) {
      sortedKeys.splice(idx, 1);
    }
    sortedKeys.unshift(this.metadata.workspaceHash);
    localforage.setItem(INDEX, sortedKeys);
    localforage.setItem(
      this.metadata.workspaceHash, this.metadata
    );
    return Job.getJobStore();
  }
}
