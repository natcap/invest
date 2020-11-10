import crypto from 'crypto';
import localforage from 'localforage';

import { getLogger } from './logger';

const logger = getLogger(__filename.split('/').slice(-1)[0]);

const HASH_ARRAY_KEY = 'workspaceHashes';
const MAX_CACHED_JOBS = 30;

/**
 * Create an object to hold properties associated with an Invest Job.
 *
 */
export default class InvestJob {
  static get max_cached_jobs() {
    return MAX_CACHED_JOBS;
  }

  /* If none exists, init an empty array for the sorted workspace hashes */
  static async initDB() {
    const keys = await localforage.getItem(HASH_ARRAY_KEY);
    if (!keys) {
      localforage.setItem(HASH_ARRAY_KEY, []);
    }
  }

  /* Return an array of job metadata objects, ordered by most recently saved */
  static async getJobStore() {
    let jobArray = [];
    const sortedKeys = await localforage.getItem(HASH_ARRAY_KEY);
    if (sortedKeys) {
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
   * @param {object} obj - the metadata property
   * @param {string} obj.modelRunName - name to be passed to `invest run`
   * @param {string} obj.modelHumanName - colloquial name of the invest model
   * @param {object} obj.argsValues - an invest "args dict" with initial values
   * @param {string} obj.logfile - path to an existing invest logfile
   * @param {'running'|'success'|'error'} obj.status - status of the invest process
   */
  constructor(
    {
      modelRunName,
      modelHumanName,
      argsValues,
      logfile,
      status,
    }
  ) {
    this.metadata = {};
    this.metadata.modelRunName = modelRunName;
    this.metadata.modelHumanName = modelHumanName;
    this.metadata.argsValues = argsValues;
    this.metadata.logfile = logfile;
    this.metadata.status = status;
    this.metadata.workspaceHash = null;

    this.save = this.save.bind(this);
    this.setProperty = this.setProperty.bind(this);
    this.setWorkspaceHash = this.setWorkspaceHash.bind(this);
  }

  setWorkspaceHash() {
    if (this.metadata.argsValues.workspace_dir
        && this.metadata.modelRunName) {
      this.metadata.workspaceHash = crypto.createHash('sha1').update(
        `${this.metadata.modelRunName}
         ${JSON.stringify(this.metadata.argsValues.workspace_dir)}
         ${JSON.stringify(this.metadata.argsValues.results_suffix)}`
      ).digest('hex');
    } else {
      throw Error(
        'Cannot hash a job that is missing workspace or modelRunName properties'
      );
    }
  }

  setProperty(key, value) {
    this.metadata[key] = value;
  }

  async save() {
    if (!this.metadata.workspaceHash) {
      this.setWorkspaceHash();
    }
    this.metadata.humanTime = new Date().toLocaleString();
    let sortedKeys = await localforage.getItem(HASH_ARRAY_KEY);
    if (!sortedKeys) {
      await InvestJob.initDB();
      sortedKeys = await localforage.getItem(HASH_ARRAY_KEY);
    }
    // If this key already exists, make sure not to duplicate it,
    // and make sure to move it to the front
    const idx = sortedKeys.indexOf(this.metadata.workspaceHash);
    if (idx > -1) {
      sortedKeys.splice(idx, 1);
    }
    sortedKeys.unshift(this.metadata.workspaceHash);
    if (sortedKeys.length > MAX_CACHED_JOBS) {
      // only 1 key is ever added at a time, so only 1 item to remove
      const lastKey = sortedKeys.pop();
      localforage.removeItem(lastKey);
    }
    await localforage.setItem(HASH_ARRAY_KEY, sortedKeys);
    await localforage.setItem(
      this.metadata.workspaceHash, this.metadata
    );
    return InvestJob.getJobStore();
  }
}
