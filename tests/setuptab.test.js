import fs from 'fs';
import path from 'path';
import React from 'react';
// import ReactTestUtils from 'react-dom/test-utils';
import renderer from 'react-test-renderer';
import { SetupTab } from '../src/components/SetupTab';

// generated this file from `invest getspec carbon --json`
const ARGS_SPEC = JSON.parse(
	fs.readFileSync(
		path.join(__dirname, './data/carbon_args_spec.json'), 'utf8'));

test('SetupTab: a model input form', () => {
  const component = renderer.create(
    <SetupTab
    	args={ARGS_SPEC.args}
      argsValid={false}
      modulename={ARGS_SPEC.module}
      updateArg={() => {}}
      batchUpdateArgs={() => {}}
      investValidate={() => {}}
      argsValuesFromSpec={() => {}}
      investExecute={() => {}}
    />
  );
  let tree = component.toJSON();
  expect(tree).toMatchSnapshot();
})
