import Job from '../src/Job';

describe('Job', () => {
  
  afterEach(() => {
    Job.clearStore();
  })

  test('setting a workspace generates a workspaceHash', async () => {
    const job = new Job({
      modelRunName: 'foo',
      modelHumanName: 'Foo'
    });
    job.setProperty('workspace', {directory: 'myspace'})
    expect(typeof job.metadata.workspaceHash).toBe('string');
  })

  test('workspaceHash collisions only when expected', async () => {
    const modelName = 'foo';
    const workspace = {directory: 'myspace', suffix: '!!'}
    const job1 = new Job({
      modelRunName: modelName,
      modelHumanName: 'Foo'
    });
    job1.setProperty('workspace', workspace);
    const job2 = new Job({
      modelRunName: modelName,
      modelHumanName: 'Foo'
    });
    job2.setProperty('workspace', workspace);
    expect(job1.metadata.workspaceHash).toBe(job2.metadata.workspaceHash);

    const job3 = new Job({
      modelRunName: 'carbon',
      modelHumanName: 'Foo'
    });
    job3.setProperty('workspace', workspace);
    // A different model with the same workspace should not collide
    expect(job1.metadata.workspaceHash).not.toBe(job3.metadata.workspaceHash);
  })

  test('save method errors if no workspace is set', async () => {
    const job = new Job({
      modelRunName: 'foo',
      modelHumanName: 'Foo'
    });
    try {
      const recentJobs = await job.save();
    } catch (error) {
      expect(error.message).toBe('cannot save a job that has no workspaceHash');
    }
  })

  test('save method works with no pre-existing database', async () => {
    const job = new Job({
      modelRunName: 'foo',
      modelHumanName: 'Foo'
    });
    job.setProperty('workspace', {directory: 'myspace'});
    const recentJobs = await job.save();
    expect(recentJobs[0]).toBe(job.metadata);
  })

  test('save method returns job store in sorted order', async () => {
    const job1 = new Job({
      modelRunName: 'foo',
      modelHumanName: 'Foo'
    })
    job1.setProperty('workspace', {directory: 'myspace'});
    let recentJobs = await job1.save();
    expect(recentJobs[0]).toBe(job1.metadata);

    const job2 = new Job({
      modelRunName: 'foo2',
      modelHumanName: 'Foo2'
    });
    job2.setProperty('workspace', {directory: 'myspace'});
    recentJobs = await job2.save();
    expect(recentJobs).toHaveLength(2);
    // The most recently saved job should always be first
    expect(recentJobs[0]).toBe(job2.metadata);
    recentJobs = await job1.save();
    expect(recentJobs[0]).toBe(job1.metadata);
    expect(recentJobs).toHaveLength(2);
  })

  test('save on the same instance overwrites entry', async () => {
    const job1 = new Job({
      modelRunName: 'foo',
      modelHumanName: 'Foo'
    });
    job1.setProperty('workspace', {directory: 'myspace'});
    let recentJobs = await job1.save();
    expect(recentJobs).toHaveLength(1);
    recentJobs = await job1.save();
    expect(recentJobs).toHaveLength(1);
  })

  test('getJobStore before and after any jobs exist', async () => {
    let recentJobs = await Job.getJobStore();
    expect(recentJobs).toHaveLength(0);
    const job1 = new Job({
      modelRunName: 'foo',
      modelHumanName: 'Foo'
    });
    job1.setProperty('workspace', {directory: 'myspace'});
    await job1.save();
    recentJobs = await Job.getJobStore();
    expect(recentJobs).toHaveLength(1);
  })

  test('clearStore clears the store', async () => {
    const job1 = new Job({
      modelRunName: 'foo',
      modelHumanName: 'Foo'
    });
    job1.setProperty('workspace', {directory: 'myspace'});
    await job1.save();
    let recentJobs = await Job.getJobStore();
    expect(recentJobs).toHaveLength(1);
    await Job.clearStore();
    recentJobs = await Job.getJobStore();
    expect(recentJobs).toHaveLength(0);
  })
})