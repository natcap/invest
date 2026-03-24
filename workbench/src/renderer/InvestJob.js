import localforage from 'localforage';

const HASH_ARRAY_KEY = 'jobHashes';
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
    const jobHashes = await investJobStore.getItem(HASH_ARRAY_KEY);
    if (!jobHashes) {
      investJobStore.setItem(HASH_ARRAY_KEY, []);
    }
  }

  /* Return an array of job metadata objects, ordered by most recently saved */
  static async getJobStore() {
    let jobArray = [];
    const sortedJobHashes = await investJobStore.getItem(HASH_ARRAY_KEY);
    if (sortedJobHashes) {
      jobArray = await Promise.all(sortedJobHashes.map(
        (key) => investJobStore.getItem(key)
      ));
    }
    // Migrate old-style jobs
    // We can eventually remove this code once it's likely that most users
    // will have updated and ran a newer version of the workbench
    jobArray = await Promise.all(jobArray.map(async (job) => {
      if (job.modelID === undefined) {
        job.modelID = job.modelRunName;
        job.modelTitle = job.modelHumanName;
        delete job.modelRunName;
        delete job.modelHumanName;
        await investJobStore.setItem(job.hash, job);
      }
      return job;
    }));
    return jobArray;
  }

  static async clearStore() {
    await investJobStore.clear();
    return InvestJob.getJobStore();
  }

  static async deleteJob(hash) {
    await investJobStore.removeItem(hash);
    // also remove item from the array that tracks the order of the jobs
    const sortedJobHashes = await investJobStore.getItem(HASH_ARRAY_KEY);
    const idx = sortedJobHashes.indexOf(hash);
    if (idx > -1) {
      sortedJobHashes.splice(idx, 1); // remove one item only
    }
    await investJobStore.setItem(HASH_ARRAY_KEY, sortedJobHashes);
    return InvestJob.getJobStore();
  }

  static async saveJob(job) {
    job.hash = window.crypto.getRandomValues(
      new Uint32Array(1)
    ).toString();
    const isoDate = new Date().toISOString().split('T')[0];
    const localTime = new Date().toTimeString().split(' ')[0];
    job.humanTime = `${isoDate} ${localTime}`;
    let sortedJobHashes = await investJobStore.getItem(HASH_ARRAY_KEY);
    if (!sortedJobHashes) {
      await InvestJob.initDB();
      sortedJobHashes = await investJobStore.getItem(HASH_ARRAY_KEY);
    }
    sortedJobHashes.unshift(job.hash);
    if (sortedJobHashes.length > MAX_CACHED_JOBS) {
      // only 1 key is ever added at a time, so only 1 item to remove
      const lastKey = sortedJobHashes.pop();
      investJobStore.removeItem(lastKey);
    }
    await investJobStore.setItem(job.hash, job);
    await investJobStore.setItem(HASH_ARRAY_KEY, sortedJobHashes);
    return InvestJob.getJobStore();
  }

  /**
   * @param {object} obj - with the following properties
   * @param {string} obj.modelID - name to be passed to `invest run`
   * @param {string} obj.modelTitle - colloquial name of the invest model
   * @param {object} obj.argsValues - an invest "args dict" with initial values
   * @param {string} obj.logfile - path to an existing invest logfile
   * @param {string} obj.htmlfile - path to an existing html report
   * @param {string} obj.status - one of 'running'|'error'|'success'
   * @param {string} obj.type - 'plugin' or 'core'
   */
  constructor(
    {
      modelID,
      modelTitle,
      argsValues,
      logfile,
      htmlfile,
      status,
      type,
    }
  ) {
    if (!modelID || !modelTitle) {
      throw new Error(
        'Cannot create instance of InvestJob without modelID and modelTitle properties'
      );
    }
    this.modelID = modelID;
    this.modelTitle = modelTitle;
    this.argsValues = argsValues;
    this.logfile = logfile;
    this.htmlfile = htmlfile;
    this.status = status;
    this.type = type;
    this.hash = null;
  }
}
