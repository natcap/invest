import { ipcRenderer } from 'electron';
import React from 'react';
import {
  createEvent, fireEvent, render, waitFor, within
} from '@testing-library/react';
import '@testing-library/jest-dom';

import SetupTab from '../src/components/SetupTab';
import ArgInput from '../src/components/SetupTab/ArgInput';
import {
  fetchDatastackFromFile, fetchValidation,
} from '../src/server_requests';

jest.mock('../src/server_requests');

const MODULE = 'carbon';

function renderSetupFromSpec(baseSpec, uiSpec) {
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
  let baseSpec, uiSpec;

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

    uiSpec = {order: [Object.keys(baseSpec.args)]}
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
    const { findByText, findByLabelText } = renderSetupFromSpec(spec, uiSpec);
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
    const { findByLabelText } = renderSetupFromSpec(spec, uiSpec);
    const input = await findByLabelText(RegExp(`${spec.args.arg.name}`));
    expect(input).toHaveAttribute('type', 'text');
  });

  test('render an unchecked radio button for a boolean', async () => {
    const spec = { ...baseSpec };
    spec.args.arg.type = 'boolean';
    const { findByLabelText } = renderSetupFromSpec(spec, uiSpec);
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
    const { findByLabelText } = renderSetupFromSpec(spec, uiSpec);
    const input = await findByLabelText(RegExp(`${spec.args.arg.name}`));
    expect(input).toHaveValue('a');
    expect(input).not.toHaveValue('b');
  });

  test('expect the info dialog contains text about input', async () => {
    const spec = { ...baseSpec };
    spec.args.arg.type = 'directory';
    const { findByText } = renderSetupFromSpec(spec, uiSpec);
    fireEvent.click(await findByText('i'));
    expect(await findByText(spec.args.arg.about)).toBeInTheDocument();
  });
});

describe('Arguments form interactions', () => {
  const validationMessage = 'invalid because';
  let spec, uiSpec;

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

    uiSpec = {order: [Object.keys(spec.args)]}
    fetchValidation.mockResolvedValue(
      [[Object.keys(spec.args), validationMessage]]
    );
  });

  test('Browse button populates an input', async () => {
    spec.args.arg.type = 'csv';
    const { findByText, findByLabelText } = renderSetupFromSpec(spec, uiSpec);

    const input = await findByLabelText(RegExp(`${spec.args.arg.name}`));
    expect(input).toHaveAttribute('type', 'text');
    expect(await findByText('Browse')).toBeInTheDocument();

    // Browsing for a file
    const filepath = 'grilled_cheese.csv';
    let mockDialogData = { filePaths: [filepath] };
    ipcRenderer.invoke.mockResolvedValue(mockDialogData);
    fireEvent.click(await findByText('Browse'));
    await waitFor(() => {
      expect(input).toHaveValue(filepath);
    });

    // Browse again, but cancel it and expect the previous value
    mockDialogData = { filePaths: [] }; // empty array is a mocked 'Cancel'
    ipcRenderer.invoke.mockResolvedValue(mockDialogData);
    fireEvent.click(await findByText('Browse'));
    await waitFor(() => {
      expect(input).toHaveValue(filepath);
    });
  });

  test('Change value & get feedback on a required input', async () => {
    spec.args.arg.type = 'directory';
    spec.args.arg.required = true;
    const { findByText, findByLabelText, queryByText } = renderSetupFromSpec(spec, uiSpec);

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
    const { findByText, findByLabelText, queryByText } = renderSetupFromSpec(spec, uiSpec);

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
    const { findByLabelText } = renderSetupFromSpec(spec, uiSpec);

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

  test('A UI spec with conditionally enabled args', async () => {
    const spec = {
      module: 'natcap.invest.dummy',
      args: {
        arg1: {
          name: 'Afoo',
          type: 'boolean',
        },
        arg2: {
          name: 'Bfoo',
          type: 'boolean',
        },
        arg3: {
          name: 'Cfoo',
          type: 'number',
        },
        arg4: {
          name: 'Dfoo',
          type: 'number',
        }
      }
    };

    const uiSpec = {
      order: [Object.keys(spec.args)],
      enabledFunctions: {
        // enabled if arg1 is sufficient
        arg2: (state => state.argsEnabled['arg1'] && !!state.argsValues['arg1'].value),
        // enabled if arg1 and arg2 are sufficient
        arg3: (state => state.argsEnabled['arg1'] && !!state.argsValues['arg1'].value &&
                       (state.argsEnabled['arg2'] && !!state.argsValues['arg2'].value)),
        // enabled if arg1 is sufficient and arg2 is not sufficient
        arg4: (state => state.argsEnabled['arg1'] && !!state.argsValues['arg1'].value &&
                      !(state.argsEnabled['arg2'] && !!state.argsValues['arg2'].value))
      }
    };

    const { findByLabelText } = renderSetupFromSpec(spec, uiSpec);
    const arg1 = await findByLabelText(RegExp(`${spec.args.arg1.name}`));
    const arg2 = await findByLabelText(RegExp(`${spec.args.arg2.name}`));
    const arg3 = await findByLabelText(RegExp(`${spec.args.arg3.name}`));
    const arg4 = await findByLabelText(RegExp(`${spec.args.arg4.name}`));

    await waitFor(() => {
      // Boolean Radios should default to "false" when a spec is loaded,
      // so controlled inputs should be hidden/disabled.
      expect(arg2).toBeDisabled();
      expect(arg3).toBeDisabled();
      expect(arg4).toBeDisabled();
    });
    // fireEvent.change doesn't trigger the change handler but .click does
    // even though React demands an onChange handler for controlled checkbox inputs.
    // https://github.com/testing-library/react-testing-library/issues/156
    fireEvent.click(arg1, { target: { value: 'true' } });

    // Check how the state changes as we click the checkboxes
    await waitFor(() => {
      expect(arg2).toBeEnabled();
      expect(arg3).toBeDisabled();
      expect(arg4).toBeEnabled();
    });

    fireEvent.click(arg2, { target: { value: 'true' } });
    await waitFor(() => {
      expect(arg2).toBeEnabled();
      expect(arg3).toBeEnabled();
      expect(arg4).toBeDisabled();
    });
  });

  test('expect dropdown options can be dynamic', async () => {
    const mockGetVectorColumnNames = (state => {
      if (state.argsValues.arg1.value) {
        return ['Field1'];
      } else {
        return [];
      }
    });
    const spec = {
      args: {
        arg1: {
          name: 'afoo',
          type: 'vector'
        },
        arg2: {
          name: 'bfoo',
          type: 'option_string',
          validation_options: {
            options: []
          }
        }
      },
    };
    const uiSpec = {
      order: [Object.keys(spec.args)],
      dropdownFunctions: {
        arg2: mockGetVectorColumnNames
      }
    };
    const { findByLabelText, findByText, queryByText } = renderSetupFromSpec(spec, uiSpec);
    const arg1 = await findByLabelText(RegExp(`${spec.args.arg1.name}`));
    let option = await queryByText('Field1');
    expect(option).toBeNull();

    // check that the dropdown option appears when the text field gets a value
    fireEvent.change(arg1, { target: { value: 'a vector'}});
    option = await findByText('Field1');  // will raise an error if not found
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
      // intentionally leaving out arg6, it should not be in the setup form
      order: [['arg4'], ['arg3', 'arg2'], ['arg1'], ['arg5']]
    };

    const { findByTestId, queryByText } = renderSetupFromSpec(spec, uiSpec);
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
    const arg6 = await queryByText('F');
    expect(arg6).toBeNull();
  });
});

describe('Misc form validation stuff', () => {
  afterEach(() => {
    fetchValidation.mockReset();
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

    const uiSpec = {order: [Object.keys(spec.args)]}

    // Mocking to return the payload so we can assert we always send
    // correct payload to this endpoint.
    fetchValidation.mockImplementation(
      (payload) => payload
    );

    renderSetupFromSpec(spec, uiSpec);
    await waitFor(() => {
      const expectedKeys = ['model_module', 'args'];
      const payload = fetchValidation.mock.results[0].value;
      expectedKeys.forEach((key) => {
        expect(Object.keys(payload)).toContain(key);
      });
    });
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
    const uiSpec = {order: [Object.keys(spec.args)]}
    const vectorValue = './vector.shp';
    const expectedVal1 = '-84.9';
    const vectorBox = `[${expectedVal1}, 19.1, -69.1, 29.5]`;
    const rasterValue = './raster.tif';
    const expectedVal2 = '-79.0198012081401';
    const rasterBox = `[${expectedVal2}, 26.481559513537064, -78.37173806200593, 27.268061760228512]`;
    const message = `Bounding boxes do not intersect: ${vectorValue}: ${vectorBox} | ${rasterValue}: ${rasterBox}`;

    fetchValidation.mockResolvedValue([[Object.keys(spec.args), message]]);

    const { findByLabelText } = renderSetupFromSpec(spec, uiSpec);
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
  });
});

describe('Form drag-and-drop', () => {
  afterEach(() => {
    fetchValidation.mockReset();
    fetchDatastackFromFile.mockReset();
  });

  test('Dragover of a datastack/logfile updates all inputs', async () => {
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
    const uiSpec = {order: [Object.keys(spec.args)]}
    fetchDatastackFromFile.mockResolvedValue(mockDatastack);

    const { findByLabelText, findByTestId } = renderSetupFromSpec(spec, uiSpec);
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
    // `dataTransfer.files` normally returns a `FileList` object. Since we are
    // defining our own dataTransfer.files we are also creating an object
    // with properties that mimic FileList object
    const fileValue = {};
    Object.defineProperties(fileValue, {
      path: { value: 'foo.json' },
      length: { value: 1 },
    });
    Object.defineProperty(fileDropEvent, 'dataTransfer', {
      value: { files: [fileValue] }
    });
    fireEvent(setupForm, fileDropEvent);

    expect(await findByLabelText(RegExp(`${spec.args.arg1.name}`)))
      .toHaveValue(mockDatastack.args.arg1);
    expect(await findByLabelText(RegExp(`${spec.args.arg2.name}`)))
      .toHaveValue(mockDatastack.args.arg2);
  });

  test('Drag enter/drop of a datastack sets .dragging class', async () => {
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
    const uiSpec = {order: [Object.keys(spec.args)]}
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

    const { findByLabelText, findByTestId } = renderSetupFromSpec(spec, uiSpec);
    const setupForm = await findByTestId('setup-form');

    const fileDragEvent = createEvent.dragEnter(setupForm);
    // `dataTransfer.files` normally returns a `FileList` object. Since we are
    // defining our own dataTransfer.files we are also creating an object
    // with properties that mimic FileList object
    const fileValue = {};
    Object.defineProperties(fileValue, {
      path: { value: 'foo.json' },
      length: { value: 1 },
    });
    Object.defineProperty(fileDragEvent, 'dataTransfer', {
      value: { files: [fileValue] }
    });
    fireEvent(setupForm, fileDragEvent);

    expect(setupForm).toHaveClass("dragging");

    const fileDropEvent = createEvent.drop(setupForm);
    Object.defineProperty(fileDropEvent, 'dataTransfer', {
      value: { files: [fileValue] }
    });
    fireEvent(setupForm, fileDropEvent);

    expect(await findByLabelText(RegExp(`${spec.args.arg1.name}`)))
      .toHaveValue(mockDatastack.args.arg1);
    expect(await findByLabelText(RegExp(`${spec.args.arg2.name}`)))
      .toHaveValue(mockDatastack.args.arg2);
    expect(setupForm).not.toHaveClass("dragging");
  });

  test('Drag enter/leave of a datastack sets .dragging class', async () => {
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
    const uiSpec = {order: [Object.keys(spec.args)]}
    fetchValidation.mockResolvedValue(
      [[Object.keys(spec.args), 'invalid because']]
    );

    const { findByLabelText, findByTestId } = renderSetupFromSpec(spec, uiSpec);
    const setupForm = await findByTestId('setup-form');

    const fileDragEnterEvent = createEvent.dragEnter(setupForm);
    // `dataTransfer.files` normally returns a `FileList` object. Since we are
    // defining our own dataTransfer.files we are also creating an object
    // with properties that mimic FileList object
    const fileValue = {};
    Object.defineProperties(fileValue, {
      path: { value: 'foo.json' },
      length: { value: 1 },
    });
    Object.defineProperty(fileDragEnterEvent, 'dataTransfer', {
      value: { files: [fileValue] }
    });
    fireEvent(setupForm, fileDragEnterEvent);

    expect(setupForm).toHaveClass("dragging");

    const fileDragLeaveEvent = createEvent.dragLeave(setupForm);
    fireEvent(setupForm, fileDragLeaveEvent);

    expect(setupForm).not.toHaveClass("dragging");
  });

  test('Drag enter/drop of a file sets .input-dragging class on input', async () => {
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
    const uiSpec = {order: [Object.keys(spec.args)]}
    fetchValidation.mockResolvedValue(
      [[Object.keys(spec.args), 'invalid because']]
    );

    const { findByLabelText, findByTestId } = renderSetupFromSpec(spec, uiSpec);
    const setupForm = await findByTestId('setup-form');
    const setupInput = await findByLabelText(RegExp(`${spec.args.arg1.name}`));

    const fileDragEvent = createEvent.dragEnter(setupInput);
    // `dataTransfer.files` normally returns a `FileList` object. Since we are
    // defining our own dataTransfer.files we are also creating an object
    // with properties that mimic FileList object
    const fileValue = {};
    Object.defineProperties(fileValue, {
      path: { value: 'foo.txt' },
      length: { value: 1 },
    });
    Object.defineProperty(fileDragEvent, 'dataTransfer', {
      value: { files: [fileValue] }
    });
    fireEvent(setupInput, fileDragEvent);

    expect(setupForm).not.toHaveClass("dragging");
    expect(setupInput).toHaveClass("input-dragging");

    const fileDropEvent = createEvent.drop(setupInput);
    Object.defineProperty(fileDropEvent, 'dataTransfer', {
      value: { files: [fileValue] }
    });
    fireEvent(setupInput, fileDropEvent);

    expect(setupInput).not.toHaveClass("input-dragging");
    expect(setupForm).not.toHaveClass("dragging");
    expect(setupInput).toHaveValue("foo.txt");
  });

  test('Drag enter/leave of a file sets .input-dragging class on input', async () => {
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
    const uiSpec = {order: [Object.keys(spec.args)]}
    fetchValidation.mockResolvedValue(
      [[Object.keys(spec.args), 'invalid because']]
    );

    const { findByLabelText, findByTestId } = renderSetupFromSpec(spec, uiSpec);
    const setupInput = await findByLabelText(RegExp(`${spec.args.arg1.name}`));

    const fileDragEnterEvent = createEvent.dragEnter(setupInput);
    // `dataTransfer.files` normally returns a `FileList` object. Since we are
    // defining our own dataTransfer.files we are also creating an object
    // with properties that mimic FileList object
    const fileValue = {};
    Object.defineProperties(fileValue, {
      path: { value: 'foo.txt' },
      length: { value: 1 },
    });
    Object.defineProperty(fileDragEnterEvent, 'dataTransfer', {
      value: { files: [fileValue] }
    });
    fireEvent(setupInput, fileDragEnterEvent);

    expect(setupInput).toHaveClass("input-dragging");

    const fileDragLeaveEvent = createEvent.dragLeave(setupInput);
    Object.defineProperty(fileDragLeaveEvent, 'dataTransfer', {
      value: { files: [fileValue] }
    });
    fireEvent(setupInput, fileDragLeaveEvent);

    expect(setupInput).not.toHaveClass("input-dragging");
  });

  test('Drag and drop on a disabled input element.', async () => {
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
    const uiSpec = {
      order: [Object.keys(spec.args)],
      enabledFunctions: {
        arg2: (state => false)  // make this arg always disabled
      }
    };

    fetchValidation.mockResolvedValue(
      [[Object.keys(spec.args), 'invalid because']]
    );

    const { findByLabelText, findByTestId } = renderSetupFromSpec(spec, uiSpec);
    const setupInput = await findByLabelText(RegExp(`${spec.args.arg2.name}`));

    const fileDragEnterEvent = createEvent.dragEnter(setupInput);
    // `dataTransfer.files` normally returns a `FileList` object. Since we are
    // defining our own dataTransfer.files we are also creating an object
    // with properties that mimic FileList object
    const fileValue = {};
    Object.defineProperties(fileValue, {
      path: { value: 'foo.shp' },
      length: { value: 1 },
    });
    Object.defineProperty(fileDragEnterEvent, 'dataTransfer', {
      value: { files: [fileValue] }
    });
    fireEvent(setupInput, fileDragEnterEvent);

    expect(setupInput).not.toHaveClass("input-dragging");

    const fileDropEvent = createEvent.drop(setupInput);
    Object.defineProperty(fileDropEvent, 'dataTransfer', {
      value: { files: [fileValue] }
    });
    fireEvent(setupInput, fileDropEvent);

    expect(setupInput).not.toHaveClass("input-dragging");
    expect(setupInput).toHaveValue("");
  });

});
