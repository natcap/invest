import crypto from 'crypto';

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
export default function Job(
  modelRunName, modelHumanName,
  argsValues, workspace, logfile, status
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

  // this.foo = function( {
  // 	return bar
  // })
}
