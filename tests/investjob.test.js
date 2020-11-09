import InvestJob from '../src/InvestJob';

describe('InvestJob', () => {
  const baseArgsValues = {
    workspace_dir: 'myspace',
    results_suffix: '!!',
  };

  afterEach(() => {
    InvestJob.clearStore();
  });

  test('workspaceHash collisions only when expected', async () => {
    const modelName = 'foo';
    const job1 = new InvestJob({
      modelRunName: modelName,
      modelHumanName: 'Foo',
    });
    job1.setProperty('argsValues', baseArgsValues);
    job1.setWorkspaceHash();
    const job2 = new InvestJob({
      modelRunName: modelName,
      modelHumanName: 'Foo',
    });
    job2.setProperty('argsValues', baseArgsValues);
    job2.setWorkspaceHash();
    expect(job1.metadata.workspaceHash).toBe(job2.metadata.workspaceHash);

    const job3 = new InvestJob({
      modelRunName: 'carbon',
      modelHumanName: 'Foo',
    });
    job3.setProperty('argsValues', baseArgsValues);
    job3.setWorkspaceHash();
    // A different model with the same workspace should not collide
    expect(job1.metadata.workspaceHash).not.toBe(job3.metadata.workspaceHash);
  });

  test('save method errors if no workspace is set', async () => {
    const job = new InvestJob({
      modelRunName: 'foo',
      modelHumanName: 'Foo',
    });
    job.setProperty('argsValues', { results_suffix: 'foo' });
    await expect(job.save()).rejects.toThrow(
      'Cannot hash a job that is missing workspace or modelRunName properties'
    );
  });

  test('save method works with no pre-existing database', async () => {
    const job = new InvestJob({
      modelRunName: 'foo',
      modelHumanName: 'Foo',
    });
    job.setProperty('argsValues', baseArgsValues);
    const recentJobs = await job.save();
    expect(recentJobs[0]).toBe(job.metadata);
  });

  test('save method returns job store in sorted order', async () => {
    const job1 = new InvestJob({
      modelRunName: 'foo',
      modelHumanName: 'Foo',
    });
    job1.setProperty('argsValues', baseArgsValues);
    let recentJobs = await job1.save();
    expect(recentJobs[0]).toBe(job1.metadata);

    const job2 = new InvestJob({
      modelRunName: 'foo2',
      modelHumanName: 'Foo2',
    });
    job2.setProperty('argsValues', baseArgsValues);
    recentJobs = await job2.save();
    expect(recentJobs).toHaveLength(2);
    // The most recently saved job should always be first
    expect(recentJobs[0]).toBe(job2.metadata);
    recentJobs = await job1.save();
    expect(recentJobs[0]).toBe(job1.metadata);
    expect(recentJobs).toHaveLength(2);
  });

  test('save on the same instance overwrites entry', async () => {
    const job1 = new InvestJob({
      modelRunName: 'foo',
      modelHumanName: 'Foo',
    });
    job1.setProperty('argsValues', baseArgsValues);
    let recentJobs = await job1.save();
    expect(recentJobs).toHaveLength(1);
    recentJobs = await job1.save();
    expect(recentJobs).toHaveLength(1);
  });

  test('getJobStore before and after any jobs exist', async () => {
    let recentJobs = await InvestJob.getJobStore();
    expect(recentJobs).toHaveLength(0);
    const job1 = new InvestJob({
      modelRunName: 'foo',
      modelHumanName: 'Foo',
    });
    job1.setProperty('argsValues', baseArgsValues);
    await job1.save();
    recentJobs = await InvestJob.getJobStore();
    expect(recentJobs).toHaveLength(1);
  });

  test('clearStore clears the store', async () => {
    const job1 = new InvestJob({
      modelRunName: 'foo',
      modelHumanName: 'Foo',
    });
    job1.setProperty('argsValues', baseArgsValues);
    await job1.save();
    let recentJobs = await InvestJob.getJobStore();
    expect(recentJobs).toHaveLength(1);
    await InvestJob.clearStore();
    recentJobs = await InvestJob.getJobStore();
    expect(recentJobs).toHaveLength(0);
  });
});
