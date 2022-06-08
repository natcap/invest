import { ipcRenderer, shell } from 'electron';
import React from 'react';
import {
  createEvent, fireEvent, render, waitFor, within
} from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';

import SetupTab from '../../src/renderer/components/SetupTab';
import {
  fetchDatastackFromFile, fetchValidation
} from '../../src/renderer/server_requests';
import setupOpenExternalUrl from '../../src/main/setupOpenExternalUrl';
import { removeIpcMainListeners } from '../../src/main/main';

jest.mock('../../src/renderer/server_requests');

const MODULE = 'carbon';

const VALIDATION_MESSAGE = 'invalid because';
const BASE_ARGS_SPEC = {
  args: {
    arg: {
      name: 'foo',
      type: undefined, // varies by test
      required: undefined,
      about: 'this is about foo',
    },
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
  const spec = JSON.parse(JSON.stringify(BASE_ARGS_SPEC));
  spec.args.arg.type = type;
  if (type === 'number') {
    spec.args.arg.units = 'foo unit';
  }
  return spec;
}
const UI_SPEC = { order: [Object.keys(BASE_ARGS_SPEC.args)] };

/**
 * Render a SetupTab component given the necessary specs.
 *
 * @param {object} baseSpec - an invest args spec for a model
 * @param {object} uiSpec - an invest UI spec for the same model
 * @returns {object} - containing the test utility functions returned by render
 */
function renderSetupFromSpec(baseSpec, uiSpec) {
  // some ARGS_SPEC boilerplate that is not under test,
  // but is required by PropType-checking
  const spec = { ...baseSpec };
  if (!spec.modelName) { spec.modelName = 'Eco Model'; }
  if (!spec.pyname) { spec.pyname = 'natcap.invest.dot'; }
  if (!spec.userguide) { spec.userguide = 'foo.html'; }
  const { ...utils } = render(
    <SetupTab
      pyModuleName={spec.pyname}
      userguide={spec.userguide}
      modelName={spec.modelName}
      argsSpec={spec.args}
      uiSpec={uiSpec}
      argsInitValues={undefined}
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
      [[Object.keys(BASE_ARGS_SPEC.args), VALIDATION_MESSAGE]]
    );
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
    } = renderSetupFromSpec(spec, UI_SPEC);

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
    const { findByLabelText } = renderSetupFromSpec(spec, UI_SPEC);
    const input = await findByLabelText(RegExp(`^${spec.args.arg.name}$`));
    expect(input).toHaveAttribute('type', 'text');
  });

  test('render a text input with unit label for a number', async () => {
    const spec = baseArgsSpec('number');
    const { findByLabelText } = renderSetupFromSpec(spec, UI_SPEC);
    const input = await findByLabelText(`${spec.args.arg.name} (${spec.args.arg.units})`);
    expect(input).toHaveAttribute('type', 'text');
  });

  test('render an unchecked radio button for a boolean', async () => {
    const spec = baseArgsSpec('boolean');
    const { findByLabelText } = renderSetupFromSpec(spec, UI_SPEC);
    const input = await findByLabelText(`${spec.args.arg.name}`);
    expect(input).toHaveAttribute('type', 'radio');
    expect(input).not.toBeChecked();
  });

  test('render a select input for an option_string dict', async () => {
    const spec = baseArgsSpec('option_string');
    spec.args.arg.options = {
      a: 'about a',
      b: 'about b',
    };
    const { findByLabelText } = renderSetupFromSpec(spec, UI_SPEC);
    const input = await findByLabelText(`${spec.args.arg.name}`);
    expect(input).toHaveValue('a');
    expect(input).not.toHaveValue('b');
  });

  test('render a select input for an option_string list', async () => {
    const spec = baseArgsSpec('option_string');
    spec.args.arg.options = ['a', 'b'];
    const { findByLabelText } = renderSetupFromSpec(spec, UI_SPEC);
    const input = await findByLabelText(`${spec.args.arg.name}`);
    expect(input).toHaveValue('a');
    expect(input).not.toHaveValue('b');
  });
});

describe('Arguments form interactions', () => {
  beforeEach(() => {
    fetchValidation.mockResolvedValue(
      [[Object.keys(BASE_ARGS_SPEC.args), VALIDATION_MESSAGE]]
    );
    setupOpenExternalUrl();
  });

  afterEach(() => {
    removeIpcMainListeners();
  });

  test('Browse button populates an input', async () => {
    const spec = baseArgsSpec('csv');
    const {
      findByRole, findByLabelText,
    } = renderSetupFromSpec(spec, UI_SPEC);

    const input = await findByLabelText(`${spec.args.arg.name}`);
    expect(input).toHaveAttribute('type', 'text');
    expect(await findByRole('button', { name: /browse for/ }))
      .toBeInTheDocument();

    // Browsing for a file
    const filepath = 'grilled_cheese.csv';
    let mockDialogData = { filePaths: [filepath] };
    ipcRenderer.invoke.mockResolvedValue(mockDialogData);
    userEvent.click(await findByRole('button', { name: /browse for/ }));
    await waitFor(() => {
      expect(input).toHaveValue(filepath);
    });

    // Browse again, but cancel it and expect the previous value
    mockDialogData = { filePaths: [] }; // empty array is a mocked 'Cancel'
    ipcRenderer.invoke.mockResolvedValue(mockDialogData);
    userEvent.click(await findByRole('button', { name: /browse for/ }));
    await waitFor(() => {
      expect(input).toHaveValue(filepath);
    });
  });

  test('Browse button populates an input - test click on child svg', async () => {
    const spec = baseArgsSpec('csv');
    const {
      findByRole, findByLabelText,
    } = renderSetupFromSpec(spec, UI_SPEC);

    const filepath = 'grilled_cheese.csv';
    const mockDialogData = { filePaths: [filepath] };
    ipcRenderer.invoke.mockResolvedValue(mockDialogData);
    const btn = await findByRole('button', { name: /browse for/ });
    // Click on a target element nested within the button to make
    // sure the handler still works correctly.
    userEvent.click(btn.querySelector('svg'));
    expect(await findByLabelText(`${spec.args.arg.name}`))
      .toHaveValue(filepath);
  });

  test('Change value & get feedback on a required input', async () => {
    const spec = baseArgsSpec('directory');
    spec.args.arg.required = true;
    const {
      findByText, findByLabelText, queryByText,
    } = renderSetupFromSpec(spec, UI_SPEC);

    const input = await findByLabelText(`${spec.args.arg.name}`);

    // A required input with no value is invalid (red X), but
    // feedback does not display until the input has been touched.
    expect(input).toHaveClass('is-invalid');
    expect(queryByText(RegExp(VALIDATION_MESSAGE))).toBeNull();

    userEvent.type(input, 'foo');
    await waitFor(() => {
      expect(input).toHaveValue('foo');
      expect(input).toHaveClass('is-invalid');
    });
    expect(await findByText(RegExp(VALIDATION_MESSAGE)))
      .toBeInTheDocument();

    fetchValidation.mockResolvedValue([]); // now make input valid
    userEvent.type(input, 'mydir');
    await waitFor(() => {
      expect(input).toHaveClass('is-valid');
      expect(queryByText(RegExp(VALIDATION_MESSAGE))).toBeNull();
    });
  });

  test('Type fast & confirm validation waits for pause in typing', async () => {
    const spy = jest.spyOn(SetupTab.prototype, 'investValidate');
    const spec = baseArgsSpec('directory');
    spec.args.arg.required = true;
    const { findByLabelText } = renderSetupFromSpec(spec, UI_SPEC);

    const input = await findByLabelText(`${spec.args.arg.name}`);
    spy.mockClear(); // it was already called once on render

    // Fast typing, expect only 1 validation call
    userEvent.type(input, 'foo', { delay: 0 });
    await waitFor(() => {
      expect(spy).toHaveBeenCalledTimes(1);
    }, 500); // debouncedValidate waits for 200ms
  });

  test('Type slow & confirm validation waits for pause in typing', async () => {
    const spy = jest.spyOn(SetupTab.prototype, 'investValidate');
    const spec = baseArgsSpec('directory');
    spec.args.arg.required = true;
    const { findByLabelText } = renderSetupFromSpec(spec, UI_SPEC);

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
    } = renderSetupFromSpec(spec, UI_SPEC);

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
    const { findByLabelText } = renderSetupFromSpec(spec, UI_SPEC);

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
    const spec = baseArgsSpec('directory');
    const { findByText, findByRole } = renderSetupFromSpec(spec, UI_SPEC);
    userEvent.click(await findByRole('button', { name: /info about/ }));
    expect(await findByText(spec.args.arg.about)).toBeInTheDocument();
    const link = await findByRole('link', { name: /user guide/ });
    userEvent.click(link);
    await waitFor(() => {
      expect(shell.openExternal).toHaveBeenCalledTimes(1);
    });
  });
});

describe('UI spec functionality', () => {
  beforeEach(() => {
    fetchValidation.mockResolvedValue([]);
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
        arg3: {
          name: 'Cfoo',
          type: 'number',
        },
        arg4: {
          name: 'Dfoo',
          type: 'number',
        },
      },
    };
    // mock some validation state so that we can test that it only
    // displays when an input is enabled.
    fetchValidation.mockResolvedValue([[['arg4'], VALIDATION_MESSAGE]]);

    const uiSpec = {
      order: [Object.keys(spec.args)],
      enabledFunctions: {
        // enabled if arg1 is sufficient
        arg2: ((state) => state.argsEnabled.arg1 && !!state.argsValues.arg1.value),
        // enabled if arg1 and arg2 are sufficient
        arg3: ((state) => state.argsEnabled.arg1 && !!state.argsValues.arg1.value
                       && (state.argsEnabled.arg2 && !!state.argsValues.arg2.value)),
        // enabled if arg1 is sufficient and arg2 is not sufficient
        arg4: ((state) => state.argsEnabled.arg1 && !!state.argsValues.arg1.value
                      && !(state.argsEnabled.arg2 && !!state.argsValues.arg2.value)),
      },
    };

    const { findByLabelText } = renderSetupFromSpec(spec, uiSpec);
    const arg1 = await findByLabelText(`${spec.args.arg1.name}`);
    const arg2 = await findByLabelText(`${spec.args.arg2.name}`);
    const arg3 = await findByLabelText(`${spec.args.arg3.name}`);
    const arg4 = await findByLabelText(`${spec.args.arg4.name}`);

    await waitFor(() => {
      // Boolean Radios should default to "false" when a spec is loaded,
      // so controlled inputs should be hidden/disabled.
      expect(arg2).toBeDisabled();
      expect(arg3).toBeDisabled();
      expect(arg4).toBeDisabled();
    });

    // Check how the state changes as we click the checkboxes
    userEvent.click(arg1);
    await waitFor(() => {
      expect(arg2).toBeEnabled();
      expect(arg3).toBeDisabled();
      expect(arg4).toBeEnabled();
      expect(arg4).toHaveClass('is-invalid');
    });

    userEvent.click(arg2);
    await waitFor(() => {
      expect(arg2).toBeEnabled();
      expect(arg3).toBeEnabled();
      expect(arg4).toBeDisabled();
      // the disabled input's validation result has not changed,
      // but the validation state should be hidden on disabled inputs.
      expect(arg4).not.toHaveClass('is-invalid');
      expect(arg4).not.toHaveClass('is-valid');
    });
  });

  test('expect dropdown options can be dynamic', async () => {
    // the real getVectorColumnNames returns a Promise
    const mockGetVectorColumnNames = ((state) => new Promise(
      (resolve) => {
        if (state.argsValues.arg1.value) {
          resolve(['Field1']);
        }
        resolve([]);
      }
    ));
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
    };
    const uiSpec = {
      order: [Object.keys(spec.args)],
      dropdownFunctions: {
        arg2: mockGetVectorColumnNames,
      },
    };
    const {
      findByLabelText, findByText, queryByText,
    } = renderSetupFromSpec(spec, uiSpec);
    const arg1 = await findByLabelText(`${spec.args.arg1.name}`);
    let option = await queryByText('Field1');
    expect(option).toBeNull();

    // check that the dropdown option appears when the text field gets a value
    userEvent.type(arg1, 'a vector');
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
    };

    const uiSpec = {
      // intentionally leaving out arg6, it should not be in the setup form
      order: [['arg4'], ['arg3', 'arg2'], ['arg1'], ['arg5']],
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

    const uiSpec = { order: [Object.keys(spec.args)] };

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
    const uiSpec = { order: [Object.keys(spec.args)] };
    const vectorValue = './vector.shp';
    const expectedVal1 = '-84.9';
    const vectorBox = `[${expectedVal1}, 19.1, -69.1, 29.5]`;
    const rasterValue = './raster.tif';
    const expectedVal2 = '-79.0198012081401';
    const rasterBox = `[${expectedVal2}, 26.481559513537064, -78.37173806200593, 27.268061760228512]`;
    const message = `Bounding boxes do not intersect: ${vectorValue}: ${vectorBox} | ${rasterValue}: ${rasterBox}`;
    const newPrefix = 'Bounding box does not intersect at least one other:';
    const vectorMessage = new RegExp(`${newPrefix}\\s*\\[${expectedVal1}`);
    const rasterMessage = new RegExp(`${newPrefix}\\s*\\[${expectedVal2}`);

    fetchValidation.mockResolvedValue([[Object.keys(spec.args), message]]);

    const { findByLabelText } = renderSetupFromSpec(spec, uiSpec);
    const vectorInput = await findByLabelText(spec.args.vector.name);
    const rasterInput = await findByLabelText(RegExp(`^${spec.args.raster.name}`));
    userEvent.type(vectorInput, vectorValue);
    userEvent.type(rasterInput, rasterValue);

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
    };
    fetchValidation.mockResolvedValue(
      [[Object.keys(spec.args), VALIDATION_MESSAGE]]
    );

    const mockDatastack = {
      module_name: spec.pyname,
      args: {
        arg1: 'circle',
        arg2: 'square',
      },
    };
    const uiSpec = { order: [Object.keys(spec.args)] };
    fetchDatastackFromFile.mockResolvedValue(mockDatastack);

    const {
      findByLabelText, findByTestId,
    } = renderSetupFromSpec(spec, uiSpec);
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
    };
    const uiSpec = { order: [Object.keys(spec.args)] };
    fetchValidation.mockResolvedValue(
      [[Object.keys(spec.args), VALIDATION_MESSAGE]]
    );

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
    } = renderSetupFromSpec(spec, uiSpec);
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
    };
    const uiSpec = { order: [Object.keys(spec.args)] };
    fetchValidation.mockResolvedValue(
      [[Object.keys(spec.args), VALIDATION_MESSAGE]]
    );

    const { findByTestId } = renderSetupFromSpec(spec, uiSpec);
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
    };
    const uiSpec = { order: [Object.keys(spec.args)] };
    fetchValidation.mockResolvedValue(
      [[Object.keys(spec.args), VALIDATION_MESSAGE]]
    );

    const {
      findByLabelText, findByTestId,
    } = renderSetupFromSpec(spec, uiSpec);
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
    };
    const uiSpec = { order: [Object.keys(spec.args)] };
    fetchValidation.mockResolvedValue(
      [[Object.keys(spec.args), VALIDATION_MESSAGE]]
    );

    const { findByLabelText } = renderSetupFromSpec(spec, uiSpec);
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
    };
    const uiSpec = {
      order: [Object.keys(spec.args)],
      enabledFunctions: {
        arg2: (() => false), // make this arg always disabled
      },
    };

    fetchValidation.mockResolvedValue(
      [[Object.keys(spec.args), VALIDATION_MESSAGE]]
    );

    const { findByLabelText } = renderSetupFromSpec(spec, uiSpec);
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
