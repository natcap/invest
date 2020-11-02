import crypto from 'crypto';
import localforage from 'localforage';

import { getLogger } from './logger';

const logger = getLogger(__filename.split('/').slice(-1)[0]);

const KEYS_ARRAY = 'sortedHashArray';

/**
 * Create an object to hold properties associated with an Invest Job.
 *
 */
export default class Job {
  /* If none exists, init an empty array for the sorted workspace hashes */
  static async initDB() {
    localforage.setItem(KEYS_ARRAY, []);
    // const keys = await localforage.getItem(KEYS_ARRAY);
    // if (!keys) {
    //   localforage.setItem(KEYS_ARRAY, []);
    // }
  }

  static async getJobStore() {
    let jobArray = [];
    const sortedKeys = await localforage.getItem(KEYS_ARRAY);
    if (sortedKeys) {
      logger.debug(`cached jobs include ${sortedKeys}`);
      jobArray = await Promise.all(sortedKeys.map(
        (key) => localforage.getItem(key)
      ));
    }
    return jobArray;
  }

  static async clearStore() {
    await localforage.clear();
  }

  /**
  * @param  {string} modelRunName - (required) invest model name to be passed to `invest run`
  * @param  {string} modelHumanName - (required) the colloquial name of the invest model
  * @param  {object} argsValues - an invest "args dictionary" with initial values
  * @param  {object} workspace - with keys for invest workspace directory and suffix
  * @param  {string} logfile - path to an existing invest logfile
  * @param  {string} status - indicates how the job exited, if it's a recent job.
  */
  constructor(
    {
      modelRunName,
      modelHumanName,
      argsValues,
      workspace,
      logfile,
      status,
    }
  ) {
    this.metadata = {};
    this.metadata.modelRunName = modelRunName || '';
    this.metadata.modelHumanName = modelHumanName || '';
    this.metadata.argsValues = argsValues;
    this.metadata.workspace = workspace;
    this.metadata.logfile = logfile;
    this.metadata.status = status;
    this.metadata.humanTime = new Date().toLocaleString();

    this.save = this.save.bind(this);
    this.setProperty = this.setProperty.bind(this);
    this.setWorkspaceHash = this.setWorkspaceHash.bind(this);

    if (workspace) {
      this.setWorkspaceHash();
    }
  }

  setWorkspaceHash() {
    if (this.metadata.workspace && this.metadata.modelRunName) {
      this.metadata.workspaceHash = crypto.createHash('sha1').update(
        `${this.metadata.modelRunName}${JSON.stringify(this.metadata.workspace)}`
      ).digest('hex');
    } else {
      throw Error('cannot hash a workspace that does not exist');
    }
  }

  setProperty(key, value) {
    this.metadata[key] = value;
    if (key === 'workspace') {
      this.setWorkspaceHash();
    }
  }

  async save() {
    if (!this.metadata.workspaceHash) {
      throw Error('cannot save a job that has no workspaceHash');
    }
    this.metadata.humanTime = new Date().toLocaleString();
    let sortedKeys = await localforage.getItem(KEYS_ARRAY);
    if (!sortedKeys) {
      await Job.initDB();
      sortedKeys = await localforage.getItem(KEYS_ARRAY);
    }
    // If this key already exists, make sure not to duplicate it,
    // and make sure to move it to the front
    const idx = sortedKeys.indexOf(this.metadata.workspaceHash);
    if (idx > -1) {
      sortedKeys.splice(idx, 1);
    }
    sortedKeys.unshift(this.metadata.workspaceHash);
    localforage.setItem(KEYS_ARRAY, sortedKeys);
    localforage.setItem(
      this.metadata.workspaceHash, this.metadata
    );
    return Job.getJobStore();
  }
}
