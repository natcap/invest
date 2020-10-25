import fs from 'fs';
import path from 'path';
import crypto from 'crypto';

import { fileRegistry } from './constants';
import { getLogger } from './logger';

const logger = getLogger(__filename.split('/').slice(-1)[0]);

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
  constructor(
    modelRunName, modelHumanName, argsValues, workspace, logfile, status
  ) {
    if (workspace && modelRunName) {
      this.workspaceHash = crypto.createHash('sha1').update(
        `${modelRunName}${JSON.stringify(workspace)}`
      ).digest('hex');
    }
    this.modelRunName = modelRunName;
    this.modelHumanName = modelHumanName;
    this.argsValues = argsValues;
    this.workspace = workspace;
    this.logfile = logfile;
    this.status = status;
  }

  save() {
    const jsonContent = JSON.stringify(this);
    const filepath = path.join(
      fileRegistry.CACHE_DIR, `${this.workspaceHash}.json`
    );
    fs.writeFile(filepath, jsonContent, 'utf8', (err) => {
      if (err) {
        logger.error('An error occured while writing JSON Object to File.');
        return logger.error(err.stack);
      }
    });
    const jobMetadata = {};
    jobMetadata[this.workspaceHash] = {
      model: this.modelHumanName,
      workspace: this.workspace,
      humanTime: new Date().toLocaleString(),
      systemTime: new Date().getTime(),
      jobDataPath: filepath,
    };
    return jobMetadata;
  }
}
