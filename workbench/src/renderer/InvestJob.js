import localforage from 'localforage';

const { crypto, path } = window.Workbench;
const logger = window.Workbench.getLogger('InvestJob.js');

const HASH_ARRAY_KEY = 'workspaceHashes';
const MAX_CACHED_JOBS = 30;
const investJobStore = localforage.createInstance({
  name: 'InvestJobs'
});

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
    const workspaceHashes = await investJobStore.getItem(HASH_ARRAY_KEY);
    if (!workspaceHashes) {
      investJobStore.setItem(HASH_ARRAY_KEY, []);
    }
  }

  /* Return an array of job metadata objects, ordered by most recently saved */
  static async getJobStore() {
    let jobArray = [];
    const sortedWorkspaceHashes = await investJobStore.getItem(HASH_ARRAY_KEY);
    if (sortedWorkspaceHashes) {
      jobArray = await Promise.all(sortedWorkspaceHashes.map(
        (key) => investJobStore.getItem(key)
      ));
    }
    return jobArray;
  }

  static async clearStore() {
    await investJobStore.clear();
    return InvestJob.getJobStore();
  }

  static getWorkspaceHash(modelRunName, workspaceDir, resultsSuffix) {
    if (workspaceDir && modelRunName) {
      const workspaceHash = crypto.sha1hash(
        `${modelRunName}
         ${JSON.stringify(path.resolve(workspaceDir))}
         ${JSON.stringify(resultsSuffix)}`
      );
      return workspaceHash;
    }
    throw Error(
      'Cannot hash a job that is missing workspace or modelRunName properties'
    );
  }

  static async saveJob(job) {
    if (!job.workspaceHash) {
      job.workspaceHash = this.getWorkspaceHash(
        job.modelRunName,
        job.argsValues.workspace_dir,
        job.argsValues.resultsSuffix
      );
    }
    const isoDate = new Date().toISOString().split('T')[0];
    const localTime = new Date().toTimeString().split(' ')[0];
    job.humanTime = `${isoDate} ${localTime}`;
    let sortedWorkspaceHashes = await investJobStore.getItem(HASH_ARRAY_KEY);
    if (!sortedWorkspaceHashes) {
      await InvestJob.initDB();
      sortedWorkspaceHashes = await investJobStore.getItem(HASH_ARRAY_KEY);
    }
    // If this key already exists, make sure not to duplicate it,
    // and make sure to move it to the front
    const idx = sortedWorkspaceHashes.indexOf(job.workspaceHash);
    if (idx > -1) {
      sortedWorkspaceHashes.splice(idx, 1);
    }
    sortedWorkspaceHashes.unshift(job.workspaceHash);
    if (sortedWorkspaceHashes.length > MAX_CACHED_JOBS) {
      // only 1 key is ever added at a time, so only 1 item to remove
      const lastKey = sortedWorkspaceHashes.pop();
      investJobStore.removeItem(lastKey);
    }
    await investJobStore.setItem(HASH_ARRAY_KEY, sortedWorkspaceHashes);
    await investJobStore.setItem(job.workspaceHash, job);
    return InvestJob.getJobStore();
  }

  /**
   * @param {object} obj - with the following properties
   * @param {string} obj.modelRunName - name to be passed to `invest run`
   * @param {string} obj.modelHumanName - colloquial name of the invest model
   * @param {object} obj.argsValues - an invest "args dict" with initial values
   * @param {string} obj.logfile - path to an existing invest logfile
   * @param {string} obj.status - one of 'running'|'error'|'success'
   * @param {string} obj.finalTraceback - final & most relevant line of stderr
   */
  constructor(
    {
      modelRunName,
      modelHumanName,
      argsValues,
      logfile,
      status,
      finalTraceback,
    }
  ) {
    if (!modelRunName || !modelHumanName) {
      throw new Error(
        'Cannot create instance of InvestJob without modelRunName and modelHumanName properties')
    }
    this.modelRunName = modelRunName;
    this.modelHumanName = modelHumanName;
    this.argsValues = argsValues;
    this.logfile = logfile;
    this.status = status;
    this.finalTraceback = finalTraceback;
    this.workspaceHash = null;
  }
}
