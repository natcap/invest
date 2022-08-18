import InvestJob from '../../src/renderer/InvestJob';

describe('InvestJob', () => {
  const baseArgsValues = {
    workspace_dir: 'myspace',
    results_suffix: '!!',
  };

  afterEach(() => {
    InvestJob.clearStore();
  });

  test('constructor errors on missing required properties', async () => {
    const errorMessage = 'Cannot create instance of InvestJob without modelRunName and modelHumanName properties'
    // Arrow func allows error to be caught and asserted on
    // https://jestjs.io/docs/expect#tothrowerror
    expect(() => new InvestJob({}))
      .toThrow(errorMessage);
    expect(() => new InvestJob({ modelRunName: 'carbon' }))
      .toThrow(errorMessage);
    expect(() => new InvestJob({ modelHumanName: 'Carbon' }))
      .toThrow(errorMessage);
  });

  test('save method works with no pre-existing database', async () => {
    const job = new InvestJob({
      modelRunName: 'foo',
      modelHumanName: 'Foo',
      argsValues: baseArgsValues,
    });
    const recentJobs = await InvestJob.saveJob(job);
    expect(recentJobs[0]).toBe(job);
  });

  test('save method returns job store in sorted order', async () => {
    const job1 = new InvestJob({
      modelRunName: 'foo',
      modelHumanName: 'Foo',
      argsValues: baseArgsValues,
    });
    let recentJobs = await InvestJob.saveJob(job1);
    expect(recentJobs[0]).toBe(job1);

    const job2 = new InvestJob({
      modelRunName: 'foo2',
      modelHumanName: 'Foo2',
      argsValues: baseArgsValues,
    });
    recentJobs = await InvestJob.saveJob(job2);
    expect(recentJobs).toHaveLength(2);
    // The most recently saved job should always be first
    expect(recentJobs[0]).toBe(job2);
    recentJobs = await InvestJob.saveJob(job1);
    expect(recentJobs[0]).toBe(job1);
    expect(recentJobs).toHaveLength(3);
  });

  test('save on the same job adds new entry', async () => {
    const job1 = new InvestJob({
      modelRunName: 'foo',
      modelHumanName: 'Foo',
      argsValues: baseArgsValues
    });
    let recentJobs = await InvestJob.saveJob(job1);
    expect(recentJobs).toHaveLength(1);
    recentJobs = await InvestJob.saveJob(job1);
    expect(recentJobs).toHaveLength(2);
  });

  test('save never accumulates more jobs than max allowed', async () => {
    let i = 0;
    const saves = [];
    while (i < InvestJob.max_cached_jobs) {
      i += 1;
      saves.push(
        InvestJob.saveJob(new InvestJob({
          modelRunName: `model${i}`,
          modelHumanName: 'Foo',
          argsValues: { workspace_dir: 'foo' }
        }))
      );
    }
    await Promise.all(saves);
    let jobsArray = await InvestJob.getJobStore();
    expect(jobsArray).toHaveLength(InvestJob.max_cached_jobs);
    jobsArray = await InvestJob.saveJob(new InvestJob({
      modelRunName: 'a new model',
      modelHumanName: 'Foo',
      argsValues: { workspace_dir: 'foo' },
    }));
    expect(jobsArray).toHaveLength(InvestJob.max_cached_jobs);
  });

  test('getJobStore before and after any jobs exist', async () => {
    let recentJobs = await InvestJob.getJobStore();
    expect(recentJobs).toHaveLength(0);
    const job1 = new InvestJob({
      modelRunName: 'foo',
      modelHumanName: 'Foo',
      argsValues: baseArgsValues,
    });
    await InvestJob.saveJob(job1);
    recentJobs = await InvestJob.getJobStore();
    expect(recentJobs).toHaveLength(1);
  });

  test('clearStore clears the store', async () => {
    const job1 = new InvestJob({
      modelRunName: 'foo',
      modelHumanName: 'Foo',
      argsValues: baseArgsValues,
    });
    let recentJobs = await InvestJob.saveJob(job1);
    expect(recentJobs).toHaveLength(1);
    recentJobs = await InvestJob.clearStore();
    expect(recentJobs).toHaveLength(0);
  });
});
