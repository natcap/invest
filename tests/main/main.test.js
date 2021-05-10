import fs from 'fs';
import path from 'path';
import { app } from 'electron';

import {
  checkFirstRun,
  APP_HAS_RUN_TOKEN
} from '../../src/main/setupCheckFirstRun';

describe('checkFirstRun', () => {
  const tokenPath = path.join(app.getPath(), APP_HAS_RUN_TOKEN);
  beforeEach(() => {
    try {
      fs.unlinkSync(tokenPath);
    } catch {}
  });

  afterAll(() => {
    try {
      fs.unlinkSync(tokenPath);
    } catch {}
  });

  it('should return true & create token if token does not exist', () => {
    expect(fs.existsSync(tokenPath)).toBe(false);
    expect(checkFirstRun()).toBe(true);
    expect(fs.existsSync(tokenPath)).toBe(true);
  });

  it('should return false if token already exists', () => {
    fs.writeFileSync(tokenPath, '');
    expect(checkFirstRun()).toBe(false);
  });
});
