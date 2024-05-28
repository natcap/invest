import { ipcRenderer } from 'electron';
import React from 'react';
import {
  createEvent, fireEvent, render, waitFor, within
} from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';

import SetupTab from '../../src/renderer/components/SetupTab';
import {
  fetchDatastackFromFile,
  fetchValidation,
  fetchArgsEnabled,
  getDynamicDropdowns
} from '../../src/renderer/server_requests';
import setupOpenExternalUrl from '../../src/main/setupOpenExternalUrl';
import { removeIpcMainListeners } from '../../src/main/main';
import { ipcMainChannels } from '../../src/main/ipcMainChannels';

jest.mock('../../src/renderer/server_requests');

const MODULE = 'carbon';

const VALIDATION_MESSAGE = 'invalid because';
const BASE_MODEL_SPEC = {
  args: {
    arg: {
      name: 'foo',
      type: undefined, // varies by test
      required: undefined,
      about: 'this is about foo',
    },
  },
  ui_spec: {
    order: [['arg']]
  },
};

/**
 * Create a base args spec containing one arg of a given type.
 *
 * @param {string} type - any invest arg type
 * @returns {object} - a simple args spec
 */
function baseArgsSpec(type) {
  // make a deep copy so we don't edit the original
  const spec = JSON.parse(JSON.stringify(BASE_MODEL_SPEC));
  spec.args.arg.type = type;
  if (type === 'number') {
    spec.args.arg.units = 'foo unit';
  }
  return spec;
}
const BASE_ARGS_ENABLED = {}
Object.keys(BASE_MODEL_SPEC.args).forEach((arg) => {
  BASE_ARGS_ENABLED[arg] = true;
});

/**
 * Render a SetupTab component given the necessary specs.
 *
 * @param {object} baseSpec - an invest model spec
 * @returns {object} - containing the test utility functions returned by render
 */
function renderSetupFromSpec(baseSpec, initValues = undefined) {
  // some MODEL_SPEC boilerplate that is not under test,
  // but is required by PropType-checking
  const spec = { ...baseSpec };
  if (!spec.modelName) { spec.modelName = 'Eco Model'; }
  if (!spec.pyname) { spec.pyname = 'natcap.invest.dot'; }
  if (!spec.userguide) { spec.userguide = 'foo.html'; }
  const { ...utils } = render(
    <SetupTab
      pyModuleName={spec.pyname}
      userguide={spec.userguide}
      modelId={spec.modelName}
      argsSpec={spec.args}
      uiSpec={spec.ui_spec}
      argsInitValues={initValues}
      investExecute={() => {}}
      nWorkers="-1"
      sidebarSetupElementId="foo"
      sidebarFooterElementId="foo"
      executeClicked={false}
      switchTabs={() => {}}
    />
  );
  return utils;
}

describe('Arguments form input types', () => {
  beforeEach(() => {
    fetchValidation.mockResolvedValue(
      [[Object.keys(BASE_MODEL_SPEC.args), VALIDATION_MESSAGE]]
    );
    fetchArgsEnabled.mockResolvedValue(BASE_ARGS_ENABLED);
  });

  test.each([
    ['directory'],
    ['csv'],
    ['vector'],
    ['raster'],
    ['file'],
  ])('render a text input & browse button for a %s', async (type) => {
    const spec = baseArgsSpec(type);

    const {
      findByLabelText, findByRole,
    } = renderSetupFromSpec(spec);

    const input = await findByLabelText(RegExp(`^${spec.args.arg.name}`));
    expect(input).toHaveAttribute('type', 'text');
    expect(await findByRole('button', { name: /browse for/ }))
      .toBeInTheDocument();
  });

  test.each([
    ['freestyle_string'],
    ['ratio'],
    ['percent'],
    ['integer'],
  ])('render a text input for a %s', async (type) => {
    const spec = baseArgsSpec(type);
    const { findByLabelText } = renderSetupFromSpec(spec);
    const input = await findByLabelText(RegExp(`^${spec.args.arg.name}$`));
    expect(input).toHaveAttribute('type', 'text');
  });

  test('render a text input with unit label for a number', async () => {
    const spec = baseArgsSpec('number');
    const { findByLabelText } = renderSetupFromSpec(spec);
    const input = await findByLabelText(`${spec.args.arg.name} (${spec.args.arg.units})`);
    expect(input).toHaveAttribute('type', 'text');
  });

  test('render an unchecked toggle switch for a boolean', async () => {
    const spec = baseArgsSpec('boolean');
    const { findByLabelText } = renderSetupFromSpec(spec);
    const input = await findByLabelText(`${spec.args.arg.name}`);
    // for some reason, the type is still checkbox when it renders as a switch
    expect(input).toHaveAttribute('type', 'checkbox');
    expect(input).not.toBeChecked();
  });

  test('render a toggle with a value', async () => {
    const spec = baseArgsSpec('boolean');
    const { findByLabelText } = renderSetupFromSpec(spec, { arg: true });
    const input = await findByLabelText(`${spec.args.arg.name}`);
    // for some reason, the type is still checkbox when it renders as a switch
    expect(input).toBeChecked();
  });

  test('render a select input for an option_string dict', async () => {
    const spec = baseArgsSpec('option_string');
    spec.args.arg.options = {
      a: {'display_name': 'Option A'},
      b: {'display_name': 'Option B'}
    };
    const { findByLabelText } = renderSetupFromSpec(spec);
    const input = await findByLabelText(`${spec.args.arg.name}`);
    expect(input).toHaveDisplayValue('Option A')
    expect(input).toHaveValue('a');
    expect(input).not.toHaveValue('b');
  });

  test('render a select input for an option_string list', async () => {
    const spec = baseArgsSpec('option_string');
    spec.args.arg.options = ['a', 'b'];
    const { findByLabelText } = renderSetupFromSpec(spec);
    const input = await findByLabelText(`${spec.args.arg.name}`);
    expect(input).toHaveValue('a');
    expect(input).not.toHaveValue('b');
  });

  test('initial arg values can contain extra args', async () => {
    const spec = baseArgsSpec('number');
    const displayedValue = '1';
    const missingValue = '0';
    const initArgs = {
      [Object.keys(spec.args)[0]]: displayedValue,
      paramZ: missingValue, // paramZ is not in the ARGS_SPEC
    };

    const { findByLabelText, queryByText } = renderSetupFromSpec(spec, initArgs);
    const input = await findByLabelText(`${spec.args.arg.name} (${spec.args.arg.units})`);
    await waitFor(() => expect(input).toHaveValue(displayedValue));
    expect(queryByText(missingValue)).toBeNull();
  });
});

describe('Arguments form interactions', () => {
  beforeEach(() => {
    fetchValidation.mockResolvedValue(
      [[Object.keys(BASE_MODEL_SPEC.args), VALIDATION_MESSAGE]]
    );
    fetchArgsEnabled.mockResolvedValue(BASE_ARGS_ENABLED);
    setupOpenExternalUrl();
  });

  afterEach(() => {
    removeIpcMainListeners();
  });

  test('Browse button populates an input', async () => {
    const spec = baseArgsSpec('csv');
    const {
      findByRole, findByLabelText,
    } = renderSetupFromSpec(spec);

    const input = await findByLabelText(`${spec.args.arg.name}`);
    expect(input).toHaveAttribute('type', 'text');
    expect(await findByRole('button', { name: /browse for/ }))
      .toBeInTheDocument();

    // Browsing for a file
    const filepath = 'grilled_cheese.csv';
    let mockDialogData = { filePaths: [filepath] };
    ipcRenderer.invoke.mockResolvedValue(mockDialogData);
    await userEvent.click(await findByRole('button', { name: /browse for/ }));
    await waitFor(() => {
      expect(input).toHaveValue(filepath);
    });

    // Browse again, but cancel it and expect the previous value
    mockDialogData = { filePaths: [] }; // empty array is a mocked 'Cancel'
    ipcRenderer.invoke.mockResolvedValue(mockDialogData);
    await userEvent.click(await findByRole('button', { name: /browse for/ }));
    await waitFor(() => {
      expect(input).toHaveValue(filepath);
    });
  });

  test('Browse button populates an input - test click on child svg', async () => {
    const spec = baseArgsSpec('csv');
    const {
      findByRole, findByLabelText,
    } = renderSetupFromSpec(spec);

    const filepath = 'grilled_cheese.csv';
    const mockDialogData = { filePaths: [filepath] };
    ipcRenderer.invoke.mockResolvedValue(mockDialogData);
    const btn = await findByRole('button', { name: /browse for/ });
    // Click on a target element nested within the button to make
    // sure the handler still works correctly.
    await userEvent.click(btn.querySelector('svg'));
    expect(await findByLabelText(`${spec.args.arg.name}`))
      .toHaveValue(filepath);
  });

  test('Change value & get feedback on a required input', async () => {
    const spec = baseArgsSpec('directory');
    spec.args.arg.required = true;
    const {
      findByText, findByLabelText, queryByText,
    } = renderSetupFromSpec(spec);

    const input = await findByLabelText(`${spec.args.arg.name}`);

    // A required input with no value is invalid (red X), but
    // feedback does not display until the input has been touched.
    expect(input).toHaveClass('is-invalid');
    expect(queryByText(RegExp(VALIDATION_MESSAGE))).toBeNull();

    await userEvent.type(input, 'foo');
    await waitFor(() => {
      expect(input).toHaveValue('foo');
      expect(input).toHaveClass('is-invalid');
    });
    expect(await findByText(RegExp(VALIDATION_MESSAGE)))
      .toBeInTheDocument();

    fetchValidation.mockResolvedValue([]); // now make input valid
    await userEvent.type(input, 'mydir');
    await waitFor(() => {
      expect(input).toHaveClass('is-valid');
      expect(queryByText(RegExp(VALIDATION_MESSAGE))).toBeNull();
    });
  });

  test('Type fast & confirm validation waits for pause in typing', async () => {
    const spy = jest.spyOn(SetupTab.WrappedComponent.prototype, 'investValidate');
    const spec = baseArgsSpec('directory');
    spec.args.arg.required = true;
    const { findByLabelText } = renderSetupFromSpec(spec);

    const input = await findByLabelText(`${spec.args.arg.name}`);
    spy.mockClear(); // it was already called once on render

    // Fast typing, expect only 1 validation call
    await userEvent.type(input, 'foo', { delay: 0 });
    await waitFor(() => {
      expect(spy).toHaveBeenCalledTimes(1);
    }, 500); // debouncedValidate waits for 200ms
  });

  test('Type slow & confirm validation waits for pause in typing', async () => {
    const spy = jest.spyOn(SetupTab.WrappedComponent.prototype, 'investValidate');
    const spec = baseArgsSpec('directory');
    spec.args.arg.required = true;
    const { findByLabelText } = renderSetupFromSpec(spec);

    const input = await findByLabelText(`${spec.args.arg.name}`);
    spy.mockClear(); // it was already called once on render

    // Slow typing, expect validation call after each character
    // debouncedValidate is set at 200ms, delay more than that per char.
    await userEvent.type(input, 'foo', { delay: 250 });
    await waitFor(() => {
      expect(spy).toHaveBeenCalledTimes(3);
    }, 2000);
  });

  test('Focus on required input & get validation feedback', async () => {
    const spec = baseArgsSpec('csv');
    spec.args.arg.required = true;
    const {
      findByText, findByLabelText, queryByText,
    } = renderSetupFromSpec(spec);

    const input = await findByLabelText(spec.args.arg.name);
    expect(input).toHaveClass('is-invalid');
    expect(queryByText(RegExp(VALIDATION_MESSAGE))).toBeNull();

    await userEvent.click(input);
    await waitFor(() => {
      expect(input).toHaveClass('is-invalid');
    });
    expect(await findByText(RegExp(VALIDATION_MESSAGE)))
      .toBeInTheDocument();
  });

  test('Focus on optional input & get valid display', async () => {
    const spec = baseArgsSpec('csv');
    spec.args.arg.required = false;
    fetchValidation.mockResolvedValue([]);
    const { findByLabelText } = renderSetupFromSpec(spec);

    const input = await findByLabelText(`${spec.args.arg.name} (optional)`);

    // An optional input with no value is valid, but green check
    // does not display until the input has been touched.
    expect(input).not.toHaveClass('is-valid', 'is-invalid');

    await userEvent.click(input);
    await waitFor(() => {
      expect(input).toHaveClass('is-valid');
    });
  });

  test('Open info dialog, expect text & link', async () => {
    const spy = jest.spyOn(ipcRenderer, 'send')
      .mockImplementation(() => Promise.resolve());
    const spec = baseArgsSpec('directory');
    const { findByText, findByRole } = renderSetupFromSpec(spec);
    await userEvent.click(await findByRole('button', { name: /info about/ }));
    expect(await findByText(spec.args.arg.about)).toBeInTheDocument();
    const link = await findByRole('link', { name: /user guide/ });
    await userEvent.click(link);
    await waitFor(() => {
      const calledChannels = spy.mock.calls.map(call => call[0]);
      expect(calledChannels).toContain(ipcMainChannels.OPEN_LOCAL_HTML);
    });
  });
});

describe('UI spec functionality', () => {
  beforeEach(() => {
    fetchValidation.mockResolvedValue([]);
    fetchArgsEnabled.mockResolvedValue({
      arg1: true, arg2: true, arg3: true, arg4: true, arg5: true, arg6: true
    });
  });

  test('A UI spec with conditionally enabled args', async () => {
    const spec = {
      pyname: 'natcap.invest.dummy',
      args: {
        arg1: {
          name: 'Afoo',
          type: 'boolean',
        },
        arg2: {
          name: 'Bfoo',
          type: 'boolean',
        },
      },
      ui_spec: {
        order: [['arg1', 'arg2']],
      }
    };
    fetchArgsEnabled.mockResolvedValue({
      arg1: false, arg2: true
    })

    const { findByLabelText } = renderSetupFromSpec(spec);
    const arg1 = await findByLabelText(`${spec.args.arg1.name}`);
    const arg2 = await findByLabelText(`${spec.args.arg2.name}`);

    await waitFor(() => {
      expect(arg1).toBeDisabled();
      expect(arg2).toBeEnabled();
    });
  });

  test('expect dropdown options can be dynamic', async () => {
    getDynamicDropdowns.mockResolvedValue({ arg2: ['Field1'] });
    const spec = {
      args: {
        arg1: {
          name: 'afoo',
          type: 'vector',
        },
        arg2: {
          name: 'bfoo',
          type: 'option_string',
          options: {},
        },
      },
      ui_spec: {
        order: [['arg1', 'arg2']],
        dropdown_functions: {
          arg2: 'function to retrieve arg1 column names',
        },
      },
    };

    const {
      findByLabelText, findByText, queryByText,
    } = renderSetupFromSpec(spec);
    const arg1 = await findByLabelText(`${spec.args.arg1.name}`);
    let option = await queryByText('Field1');
    expect(option).toBeNull();

    // check that the dropdown option appears when the text field gets a value
    await userEvent.type(arg1, 'a vector');
    option = await findByText('Field1'); // will raise an error if not found
  });

  test('Grouping and sorting of args', async () => {
    const spec = {
      pyname: 'natcap.invest.dummy',
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
      ui_spec: {
        // intentionally leaving out arg6, it should not be in the setup form
        order: [['arg4'], ['arg3', 'arg2'], ['arg1'], ['arg5']],
      }
    };

    const { findByTestId, queryByText } = renderSetupFromSpec(spec);
    const form = await findByTestId('setup-form');

    await waitFor(() => {
      // The form should have one child node per arg group
      // 2 of the 5 args share a group and 1 arg is hidden
      expect(form.childNodes).toHaveLength(4);
      // Input nodes should be in the order defined in the spec
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
      ui_spec: {
        order: [['a', 'b', 'c']]
      }
    };

    // Mocking to return the payload so we can assert we always send
    // correct payload to this endpoint.
    fetchValidation.mockImplementation(
      (payload) => payload
    );
    fetchArgsEnabled.mockResolvedValue({ a: true, b: true, c: true });

    renderSetupFromSpec(spec);
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
      ui_spec: {
        order: [['vector', 'raster']]
      }
    };
    const vectorValue = './vector.shp';
    const expectedVal1 = '-84.9';
    const vectorBox = `[${expectedVal1}, 19.1, -69.1, 29.5]`;
    const rasterValue = './raster.tif';
    const expectedVal2 = '-79.0198012081401';
    const rasterBox = `[${expectedVal2}, 26.481559513537064, -78.37173806200593, 27.268061760228512]`;
    const message = `Not all of the spatial layers overlap each other. All bounding boxes must intersect: ${vectorValue}: ${vectorBox} | ${rasterValue}: ${rasterBox}`;
    const newPrefix = 'Not all of the spatial layers overlap each other. Bounding box:';
    const vectorMessage = new RegExp(`${newPrefix}\\s*\\[${expectedVal1}`);
    const rasterMessage = new RegExp(`${newPrefix}\\s*\\[${expectedVal2}`);

    fetchValidation.mockResolvedValue([[Object.keys(spec.args), message]]);
    fetchArgsEnabled.mockResolvedValue({ vector: true, raster: true });
    const { findByLabelText } = renderSetupFromSpec(spec);
    const vectorInput = await findByLabelText(spec.args.vector.name);
    const rasterInput = await findByLabelText(RegExp(`^${spec.args.raster.name}`));
    await userEvent.type(vectorInput, vectorValue);
    await userEvent.type(rasterInput, rasterValue);

    // Feedback on each input should only include the bounding box
    // of that single input.
    const vectorGroup = vectorInput.closest('.input-group');
    await waitFor(() => {
      expect(within(vectorGroup).getByText(vectorMessage))
        .toBeInTheDocument();
      expect(within(vectorGroup).queryByText(rasterMessage))
        .toBeNull();
    });

    const rasterGroup = rasterInput.closest('.input-group');
    await waitFor(() => {
      expect(within(rasterGroup).getByText(rasterMessage))
        .toBeInTheDocument();
      expect(within(rasterGroup).queryByText(vectorMessage))
        .toBeNull();
    });
  });
});

describe('Form drag-and-drop', () => {
  test('Dragover of a datastack/logfile updates all inputs', async () => {
    const spec = {
      pyname: `natcap.invest.${MODULE}`,
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
      ui_spec: {
        order: [['arg1', 'arg2']]
      },
    };
    fetchValidation.mockResolvedValue(
      [[Object.keys(spec.args), VALIDATION_MESSAGE]]
    );
    fetchArgsEnabled.mockResolvedValue({
      arg1: true, arg2: true
    });

    const mockDatastack = {
      module_name: spec.pyname,
      args: {
        arg1: 'circle',
        arg2: 'square',
      },
    };
    fetchDatastackFromFile.mockResolvedValue(mockDatastack);

    const {
      findByLabelText, findByTestId,
    } = renderSetupFromSpec(spec);
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
      value: { files: [fileValue] },
    });
    fireEvent(setupForm, fileDropEvent);

    expect(await findByLabelText(`${spec.args.arg1.name}`))
      .toHaveValue(mockDatastack.args.arg1);
    expect(await findByLabelText(`${spec.args.arg2.name}`))
      .toHaveValue(mockDatastack.args.arg2);
  });

  test('Drag enter/drop of a datastack sets .dragging class', async () => {
    const spec = {
      pyname: `natcap.invest.${MODULE}`,
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
      ui_spec: {
        order: [['arg1', 'arg2']]
      },
    };
    fetchValidation.mockResolvedValue(
      [[Object.keys(spec.args), VALIDATION_MESSAGE]]
    );
    fetchArgsEnabled.mockResolvedValue({
      arg1: true, arg2: true
    });

    const mockDatastack = {
      module_name: spec.pyname,
      args: {
        arg1: 'circle',
        arg2: 'square',
      },
    };
    fetchDatastackFromFile.mockResolvedValue(mockDatastack);

    const {
      findByLabelText, findByTestId,
    } = renderSetupFromSpec(spec);
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
      value: { files: [fileValue] },
    });
    fireEvent(setupForm, fileDragEvent);

    expect(setupForm).toHaveClass('dragging');

    const fileDropEvent = createEvent.drop(setupForm);
    Object.defineProperty(fileDropEvent, 'dataTransfer', {
      value: { files: [fileValue] },
    });
    fireEvent(setupForm, fileDropEvent);

    expect(await findByLabelText(`${spec.args.arg1.name}`))
      .toHaveValue(mockDatastack.args.arg1);
    expect(await findByLabelText(`${spec.args.arg2.name}`))
      .toHaveValue(mockDatastack.args.arg2);
    expect(setupForm).not.toHaveClass('dragging');
  });

  test('Drag enter/leave of a datastack sets .dragging class', async () => {
    const spec = {
      pyname: `natcap.invest.${MODULE}`,
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
      ui_spec: {
        order: [['arg1', 'arg2']]
      },
    };
    fetchValidation.mockResolvedValue(
      [[Object.keys(spec.args), VALIDATION_MESSAGE]]
    );
    fetchArgsEnabled.mockResolvedValue({
      arg1: true, arg2: true
    });

    const { findByTestId } = renderSetupFromSpec(spec);
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
      value: { files: [fileValue] },
    });
    fireEvent(setupForm, fileDragEnterEvent);

    expect(setupForm).toHaveClass('dragging');

    const fileDragLeaveEvent = createEvent.dragLeave(setupForm);
    fireEvent(setupForm, fileDragLeaveEvent);

    expect(setupForm).not.toHaveClass('dragging');
  });

  test('Drag enter/drop of a file sets .input-dragging class on input', async () => {
    const spec = {
      pyname: `natcap.invest.${MODULE}`,
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
      ui_spec: {
        order: [['arg1', 'arg2']]
      },
    };
    fetchValidation.mockResolvedValue(
      [[Object.keys(spec.args), VALIDATION_MESSAGE]]
    );
    fetchArgsEnabled.mockResolvedValue({
      arg1: true, arg2: true
    });

    const {
      findByLabelText, findByTestId,
    } = renderSetupFromSpec(spec);
    const setupForm = await findByTestId('setup-form');
    const setupInput = await findByLabelText(`${spec.args.arg1.name}`);

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
      value: { files: [fileValue] },
    });
    fireEvent(setupInput, fileDragEvent);

    expect(setupForm).not.toHaveClass('dragging');
    expect(setupInput).toHaveClass('input-dragging');

    const fileDropEvent = createEvent.drop(setupInput);
    Object.defineProperty(fileDropEvent, 'dataTransfer', {
      value: { files: [fileValue] },
    });
    fireEvent(setupInput, fileDropEvent);

    expect(setupInput).not.toHaveClass('input-dragging');
    expect(setupForm).not.toHaveClass('dragging');
    expect(setupInput).toHaveValue('foo.txt');
  });

  test('Drag enter/leave of a file sets .input-dragging class on input', async () => {
    const spec = {
      pyname: `natcap.invest.${MODULE}`,
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
      ui_spec: {
        order: [['arg1', 'arg2']]
      },
    };
    fetchValidation.mockResolvedValue(
      [[Object.keys(spec.args), VALIDATION_MESSAGE]]
    );
    fetchArgsEnabled.mockResolvedValue({
      arg1: true, arg2: true
    });

    const { findByLabelText } = renderSetupFromSpec(spec);
    const setupInput = await findByLabelText(`${spec.args.arg1.name}`);

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
      value: { files: [fileValue] },
    });
    fireEvent(setupInput, fileDragEnterEvent);

    expect(setupInput).toHaveClass('input-dragging');

    const fileDragLeaveEvent = createEvent.dragLeave(setupInput);
    Object.defineProperty(fileDragLeaveEvent, 'dataTransfer', {
      value: { files: [fileValue] },
    });
    fireEvent(setupInput, fileDragLeaveEvent);

    expect(setupInput).not.toHaveClass('input-dragging');
  });

  test('Drag and drop on a disabled input element.', async () => {
    const spec = {
      pyname: `natcap.invest.${MODULE}`,
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
      ui_spec: {
        order: [['arg1', 'arg2']]
      },
    };

    fetchValidation.mockResolvedValue(
      [[Object.keys(spec.args), VALIDATION_MESSAGE]]
    );
    fetchArgsEnabled.mockResolvedValue({
      arg1: true, arg2: false
    });
    const { findByLabelText } = renderSetupFromSpec(spec);
    const setupInput = await findByLabelText(`${spec.args.arg2.name}`);

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
      value: { files: [fileValue] },
    });
    fireEvent(setupInput, fileDragEnterEvent);

    expect(setupInput).not.toHaveClass('input-dragging');

    const fileDropEvent = createEvent.drop(setupInput);
    Object.defineProperty(fileDropEvent, 'dataTransfer', {
      value: { files: [fileValue] },
    });
    fireEvent(setupInput, fileDropEvent);

    expect(setupInput).not.toHaveClass('input-dragging');
    expect(setupInput).toHaveValue('');
  });
});
