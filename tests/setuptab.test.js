import { remote } from 'electron';
import React from 'react';
import {
  createEvent, fireEvent, render, waitFor, within
} from '@testing-library/react';
import '@testing-library/jest-dom';

import SetupTab from '../src/components/SetupTab';
import {
  fetchDatastackFromFile, fetchValidation,
} from '../src/server_requests';

jest.mock('../src/server_requests');

const MODULE = 'carbon';

function renderSetupFromSpec(baseSpec, uiSpec = {}) {
  // some ARGS_SPEC boilerplate that is not under test,
  // but is required by PropType-checking
  const spec = { ...baseSpec };
  if (!spec.modelName) { spec.modelName = 'Eco Model'; }
  if (!spec.module) { spec.module = 'natcap.invest.dot'; }

  const { ...utils } = render(
    <SetupTab
      pyModuleName={spec.module}
      modelName={spec.modelName}
      argsSpec={spec.args}
      uiSpec={uiSpec}
      argsInitValues={undefined}
      investExecute={() => {}}
      argsToJsonFile={() => {}}
      nWorkers="-1"
      sidebarSetupElementId="foo"
      sidebarFooterElementId="foo"
      isRunning={false}
    />
  );
  return utils;
}

describe('Arguments form input types', () => {
  const validationMessage = 'invalid because';
  let baseSpec;

  afterEach(() => {
    jest.resetAllMocks();
  });

  beforeEach(() => {
    baseSpec = {
      args: {
        arg: {
          name: 'foo',
          type: undefined, // varies by test
          required: undefined,
          about: 'this is about foo',
        },
      },
    };
    fetchValidation.mockResolvedValue(
      [[Object.keys(baseSpec.args), validationMessage]]
    );
  });

  test.each([
    ['directory'],
    ['csv'],
    ['vector'],
    ['raster'],
  ])('render a text input & browse button for a %s', async (type) => {
    const spec = { ...baseSpec };
    spec.args.arg.type = type;
    const { findByText, findByLabelText } = renderSetupFromSpec(spec);
    const input = await findByLabelText(RegExp(`${spec.args.arg.name}`));
    expect(input).toHaveAttribute('type', 'text');
    expect(await findByText('Browse')).toBeInTheDocument();
  });

  test.each([
    ['freestyle_string'],
    ['number'],
  ])('render a text input for a %s', async (type) => {
    const spec = { ...baseSpec };
    spec.args.arg.type = type;
    const { findByLabelText } = renderSetupFromSpec(spec);
    const input = await findByLabelText(RegExp(`${spec.args.arg.name}`));
    expect(input).toHaveAttribute('type', 'text');
  });

  test('render an unchecked radio button for a boolean', async () => {
    const spec = { ...baseSpec };
    spec.args.arg.type = 'boolean';
    const { findByLabelText } = renderSetupFromSpec(spec);
    const input = await findByLabelText(RegExp(`${spec.args.arg.name}`));
    expect(input).toHaveAttribute('type', 'radio');
    expect(input).not.toBeChecked();
  });

  test('render a select input for an option_string', async () => {
    const spec = { ...baseSpec };
    spec.args.arg.type = 'option_string';
    spec.args.arg.validation_options = {
      options: ['a', 'b']
    };
    const { findByLabelText } = renderSetupFromSpec(spec);
    const input = await findByLabelText(RegExp(`${spec.args.arg.name}`));
    expect(input).toHaveValue('a');
    expect(input).not.toHaveValue('b');
  });

  test('expect the info dialog contains text about input', async () => {
    const spec = { ...baseSpec };
    spec.args.arg.type = 'directory';
    const { findByText } = renderSetupFromSpec(spec);
    fireEvent.click(await findByText('i'));
    // expect(true).toBe(true);
    expect(await findByText(spec.args.arg.about)).toBeInTheDocument();
  });
});

describe('Arguments form interactions', () => {
  const validationMessage = 'invalid because';
  let spec;

  afterEach(() => {
    jest.resetAllMocks();
  });

  beforeEach(() => {
    spec = {
      args: {
        arg: {
          name: 'foo',
          type: undefined, // varies by test
          required: undefined, // varies by test
          about: 'this is about foo',
        },
      },
    };
    fetchValidation.mockResolvedValue(
      [[Object.keys(spec.args), validationMessage]]
    );
  });

  test('Browse button populates an input', async () => {
    spec.args.arg.type = 'csv';
    const { findByText, findByLabelText } = renderSetupFromSpec(spec);

    const input = await findByLabelText(RegExp(`${spec.args.arg.name}`));
    expect(input).toHaveAttribute('type', 'text');
    expect(await findByText('Browse')).toBeInTheDocument();

    // Browsing for a file
    const filepath = 'grilled_cheese.csv';
    let mockDialogData = { filePaths: [filepath] };
    remote.dialog.showOpenDialog.mockResolvedValue(mockDialogData);
    fireEvent.click(await findByText('Browse'));
    await waitFor(() => {
      expect(input).toHaveValue(filepath);
    });

    // Browse again, but cancel it and expect the previous value
    mockDialogData = { filePaths: [] }; // empty array is a mocked 'Cancel'
    remote.dialog.showOpenDialog.mockResolvedValue(mockDialogData);
    fireEvent.click(await findByText('Browse'));
    await waitFor(() => {
      expect(input).toHaveValue(filepath);
    });
  });

  test('Change value & get feedback on a required input', async () => {
    spec.args.arg.type = 'directory';
    spec.args.arg.required = true;
    const { findByText, findByLabelText, queryByText } = renderSetupFromSpec(spec);

    const input = await findByLabelText(RegExp(`${spec.args.arg.name}`));

    // A required input with no value is invalid (red X), but
    // feedback does not display until the input has been touched.
    expect(input).toHaveClass('is-invalid');
    expect(queryByText(RegExp(validationMessage))).toBeNull();

    fireEvent.change(input, { target: { value: 'foo' } });
    await waitFor(() => {
      expect(input).toHaveValue('foo');
      expect(input).toHaveClass('is-invalid');
    });
    expect(await findByText(RegExp(validationMessage)))
      .toBeInTheDocument();

    fetchValidation.mockResolvedValue([]); // now make input valid
    fireEvent.change(input, { target: { value: 'mydir' } });
    await waitFor(() => {
      expect(input).toHaveClass('is-valid');
      expect(queryByText(RegExp(validationMessage))).toBeNull();
    });
  });

  test('Focus on required input & get validation feedback', async () => {
    spec.args.arg.type = 'csv';
    spec.args.arg.required = true;
    const { findByText, findByLabelText, queryByText } = renderSetupFromSpec(spec);

    const input = await findByLabelText(RegExp(`${spec.args.arg.name}`));
    expect(input).toHaveClass('is-invalid');
    expect(queryByText(RegExp(validationMessage))).toBeNull();

    await fireEvent.focus(input);
    await waitFor(() => {
      expect(input).toHaveClass('is-invalid');
    });
    expect(await findByText(RegExp(validationMessage)))
      .toBeInTheDocument();
  });

  test('Focus on optional input & get valid display', async () => {
    spec.args.arg.type = 'csv';
    spec.args.arg.required = false;
    fetchValidation.mockResolvedValue([]);
    const { findByLabelText } = renderSetupFromSpec(spec);

    const input = await findByLabelText(RegExp(`${spec.args.arg.name}`));

    // An optional input with no value is valid, but green check
    // does not display until the input has been touched.
    expect(input).not.toHaveClass('is-valid', 'is-invalid');

    await fireEvent.focus(input);
    await waitFor(() => {
      expect(input).toHaveClass('is-valid');
    });
  });
});

describe('UI spec functionality', () => {
  beforeAll(() => {
    fetchValidation.mockResolvedValue([]);
  });

  afterAll(() => {
    jest.resetAllMocks();
  });

  test('A UI spec with a boolean controller arg', async () => {
    const spec = {
      module: 'natcap.invest.dummy',
      args: {
        controller: {
          name: 'Afoo',
          type: 'boolean',
        },
        arg2: {
          name: 'Bfoo',
          type: 'number',
        },
        arg3: {
          name: 'Cfoo',
          type: 'number',
        },
        arg4: {
          name: 'Dfoo',
          type: 'number',
        },
        arg5: {
          name: 'Efoo',
          type: 'number',
        },
      },
    };

    const uiSpec = {
      controller: {
        ui_control: ['arg2', 'arg3', 'arg4'],
      },
      arg2: {
        ui_option: 'disable',
      },
      arg3: {
        ui_option: 'hide',
      },
      arg4: {
        ui_option: 'foo', // an invalid option should be ignored
      },
    };
    // arg5 is deliberately missing to demonstrate that that is okay.

    const {
      findByLabelText, findByTestId
    } = renderSetupFromSpec(spec, uiSpec);
    const controller = await findByLabelText(
      RegExp(`${spec.args.controller.name}`)
    );
    const arg2 = await findByLabelText(RegExp(`${spec.args.arg2.name}`));
    const arg3 = await findByLabelText(RegExp(`${spec.args.arg3.name}`));
    const arg4 = await findByLabelText(RegExp(`${spec.args.arg4.name}`));
    const arg5 = await findByLabelText(RegExp(`${spec.args.arg5.name}`));
    // The 'hide' style is applied to the whole Form.Group which
    // includes the Label and Input. Right now, the only good way
    // to query the Form.Group node is using a data-testid property.
    const arg3Group = await findByTestId('group-arg3');

    await waitFor(() => {
      // Boolean Radios should default to "false" when a spec is loaded,
      // so controlled inputs should be hidden/disabled.
      expect(arg2).toBeDisabled();
      expect(arg3Group).toHaveClass('arg-hide');

      // This input is controlled, but has an invalid option
      // so is not actually controlled.
      expect(arg4).toBeVisible();
      expect(arg4).toBeEnabled();
      // This input is not controlled.
      expect(arg5).toBeVisible();
      expect(arg5).toBeEnabled();
    });
    // fireEvent.change doesn't trigger the change handler but .click does
    // even though React demands an onChange handler for controlled checkbox inputs.
    // https://github.com/testing-library/react-testing-library/issues/156
    fireEvent.click(controller, { target: { value: 'true' } });
    // controller.click();

    // Now everything should be visible/enabled.
    await waitFor(() => {
      expect(arg2).toBeEnabled();
      expect(arg2).toBeVisible();
      expect(arg3).toBeEnabled();
      expect(arg3).toBeVisible();
      expect(arg4).toBeEnabled();
      expect(arg4).toBeVisible();
      expect(arg5).toBeEnabled();
      expect(arg5).toBeVisible();
    });
  });

  test('expect non-boolean controller can disable/hide optional inputs', async () => {
    // Normally the UI options are loaded from a seperate spec on disk
    // that is merged with ARGS_SPEC. But for testing, it's convenient
    // to just use one spec. And it works just the same.
    const spec = {
      args: {
        controller: {
          name: 'afoo',
          type: 'csv',
          ui_control: ['arg2'],
        },
        arg2: {
          name: 'bfoo',
          type: 'number',
          ui_option: 'disable',
        },
      },
    };

    const { findByLabelText } = renderSetupFromSpec(spec);
    const controller = await findByLabelText(
      RegExp(`${spec.args.controller.name}`)
    );
    const arg2 = await findByLabelText(
      RegExp(`${spec.args.arg2.name}`)
    );

    // The optional input should be disabled while the controlling input
    // has a falsy value (undefined or '')
    await waitFor(() => {
      expect(arg2).toBeDisabled();
    });

    fireEvent.change(controller, { target: { value: 'foo.csv' } });
    // Now everything should be enabled.
    await waitFor(() => {
      expect(arg2).toBeEnabled();
    });
  });

  test('Grouping and sorting of args', async () => {
    const spec = {
      module: 'natcap.invest.dummy',
      args: {
        arg1: {
          name: 'A',
          type: 'boolean',
        },
        arg2: {
          name: 'B',
          type: 'number',
        },
        arg3: {
          name: 'C',
          type: 'number',
        },
        arg4: {
          name: 'D',
          type: 'number',
        },
        arg5: {
          name: 'E',
          type: 'number',
        },
        arg6: {
          name: 'F',
          type: 'number',
        },
      },
    };

    const uiSpec = {
      arg1: { order: 2 },
      arg2: { order: 1.1 },
      arg3: { order: 1 },
      arg4: { order: 0 },
      arg5: {}, // order is deliberately missing, it should end up last.
      arg6: { order: 'hidden' }, // should not be included in the setup form
    };

    const { findByTestId } = renderSetupFromSpec(spec, uiSpec);
    const form = await findByTestId('setup-form');

    await waitFor(() => {
      // The form should have one child node per arg group
      // 2 of the 5 args share a group and 1 arg is hidden
      expect(form.childNodes).toHaveLength(4);
      // Input nodes should be in the order defined in uiSpec
      expect(form.childNodes[0])
        .toHaveTextContent(RegExp(`${spec.args.arg4.name}`));
      expect(form.childNodes[1].childNodes[0])
        .toHaveTextContent(RegExp(`${spec.args.arg3.name}`));
      expect(form.childNodes[1].childNodes[1])
        .toHaveTextContent(RegExp(`${spec.args.arg2.name}`));
      expect(form.childNodes[2])
        .toHaveTextContent(RegExp(`${spec.args.arg1.name}`));
      expect(form.childNodes[3])
        .toHaveTextContent(RegExp(`${spec.args.arg5.name}`));
    });
  });
});

test('Validation payload is well-formatted', async () => {
  const spec = {
    args: {
      a: {
        name: 'afoo',
        type: 'freestyle_string',
      },
      b: {
        name: 'bfoo',
        type: 'number',
      },
      c: {
        name: 'cfoo',
        type: 'csv',
      },
    },
  };

  // Mocking to return the payload so we can assert we always send
  // correct payload to this endpoint.
  fetchValidation.mockImplementation(
    (payload) => payload
  );

  renderSetupFromSpec(spec);
  await waitFor(() => {
    const expectedKeys = ['model_module', 'args'];
    const payload = fetchValidation.mock.results[0].value;
    expectedKeys.forEach((key) => {
      expect(Object.keys(payload)).toContain(key);
    });
  });
  fetchValidation.mockReset();
});

test('Check spatial overlap feedback is well-formatted', async () => {
  const spec = {
    args: {
      vector: {
        name: 'vvvvvv',
        type: 'vector',
      },
      raster: {
        name: 'rrrrrr',
        type: 'raster',
      },
    },
  };
  const vectorValue = './vector.shp';
  const expectedVal1 = '-84.9';
  const vectorBox = `[${expectedVal1}, 19.1, -69.1, 29.5]`;
  const rasterValue = './raster.tif';
  const expectedVal2 = '-79.0198012081401';
  const rasterBox = `[${expectedVal2}, 26.481559513537064, -78.37173806200593, 27.268061760228512]`;
  const message = `Bounding boxes do not intersect: ${vectorValue}: ${vectorBox} | ${rasterValue}: ${rasterBox}`;

  fetchValidation.mockResolvedValue([[Object.keys(spec.args), message]]);

  const { findByLabelText } = renderSetupFromSpec(spec);
  const vectorInput = await findByLabelText(spec.args.vector.name);
  const rasterInput = await findByLabelText(spec.args.raster.name);

  fireEvent.change(vectorInput, { target: { value: vectorValue } });
  fireEvent.change(rasterInput, { target: { value: rasterValue } });

  // Feedback on each input should only include the bounding box
  // of that single input.
  const vectorGroup = vectorInput.closest('div');
  await waitFor(() => {
    expect(within(vectorGroup).getByText(RegExp(expectedVal1)))
      .toBeInTheDocument();
    expect(within(vectorGroup).queryByText(RegExp(expectedVal2)))
      .toBeNull();
  });

  const rasterGroup = rasterInput.closest('div');
  await waitFor(() => {
    expect(within(rasterGroup).getByText(RegExp(expectedVal2)))
      .toBeInTheDocument();
    expect(within(rasterGroup).queryByText(RegExp(expectedVal1)))
      .toBeNull();
  });
  fetchValidation.mockReset();
});

test('SetupTab: test dragover of a datastack/logfile', async () => {
  const spec = {
    module: `natcap.invest.${MODULE}`,
    args: {
      arg1: {
        name: 'Workspace',
        type: 'directory',
      },
      arg2: {
        name: 'AOI',
        type: 'vector',
      },
    },
  };

  fetchValidation.mockResolvedValue(
    [[Object.keys(spec.args), 'invalid because']]
  );

  const mockDatastack = {
    module_name: spec.module,
    args: {
      arg1: 'circle',
      arg2: 'square',
    },
  };
  fetchDatastackFromFile.mockResolvedValue(mockDatastack);

  const { findByLabelText, findByTestId } = renderSetupFromSpec(spec);
  const setupForm = await findByTestId('setup-form');

  // This should work but doesn't due to lack of dataTransfer object in jsdom:
  // https://github.com/jsdom/jsdom/issues/1568
  // const dropEvent = new Event('drop',
  //   { dataTransfer: { files: ['foo.txt'] }
  // })
  // fireEvent.drop(setupForm, dropEvent)

  // Below is a patch similar to the one noted here:
  // https://github.com/testing-library/react-testing-library/issues/339
  const fileDropEvent = createEvent.drop(setupForm);
  const fileArray = ['foo.txt'];
  Object.defineProperty(fileDropEvent, 'dataTransfer', {
    value: { files: fileArray }
  });
  fireEvent(setupForm, fileDropEvent);

  expect(await findByLabelText(RegExp(`${spec.args.arg1.name}`)))
    .toHaveValue(mockDatastack.args.arg1);
  expect(await findByLabelText(RegExp(`${spec.args.arg2.name}`)))
    .toHaveValue(mockDatastack.args.arg2);
  fetchValidation.mockReset();
});
