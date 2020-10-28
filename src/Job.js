import fs from 'fs';
import path from 'path';
import crypto from 'crypto';

import localforage from 'localforage';

import { fileRegistry } from './constants';
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
    const sortedKeys = await localforage.getItem(INDEX);
    console.log(sortedKeys);
    if (!await localforage.getItem(INDEX)) {
      localforage.setItem(INDEX, []);
    }
  }

  static async getJobStore() {
    const jobs = [];
    const sortedKeys = await localforage.getItem(INDEX);
    console.log(sortedKeys);
    if (sortedKeys) {
      console.log(sortedKeys);
      sortedKeys.forEach(async (workspaceHash) => {
        jobs.push(await localforage.getItem(workspaceHash));
      });
    }
    return jobs;
    // const store = localforage;
    // console.log(store);
    // const parsedStore = Object.values(store).forEach(
    //   (obj) => JSON.parse(obj)
    // );
    // const sorted = parsedStore.sort(
    //   (a, b) => b[1].systemTime - a[1].systemTime
    // );
    // return sorted;
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
    // const jsonContent = JSON.stringify(this.metadata);
    // const filepath = path.join(
    //   fileRegistry.CACHE_DIR, `${this.workspaceHash}.json`
    // );
    // fs.writeFile(filepath, jsonContent, 'utf8', (err) => {
    //   if (err) {
    //     logger.error('An error occured while writing JSON Object to File.');
    //     return logger.error(err.stack);
    //   }
    // });
    this.metadata.humanTime = new Date().toLocaleString();
    this.metadata.systemTime = new Date().getTime();
    const sortedKeys = await localforage.getItem(INDEX);
    sortedKeys.unshift(this.metadata.workspaceHash);
    localforage.setItem(INDEX, sortedKeys);
    localforage.setItem(
      this.metadata.workspaceHash, this.metadata
    );
    // const store = window.localStorage;
    // const sortedMetadata = Object.entries(store).sort(
    //   (a, b) => b[1].systemTime - a[1].systemTime
    // );
    return Job.getJobStore();

    // const jobMetadata = {};
    // jobMetadata[this.workspaceHash] = {
    //   model: this.modelHumanName,
    //   workspace: this.workspace,
    //   humanTime: new Date().toLocaleString(),
    //   systemTime: new Date().getTime(),
    //   jobDataPath: filepath,
    // };
    // return jobMetadata;
  }
}
