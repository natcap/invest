import unittest
import sys
import shutil
import os
import platform
import tempfile

from PyQt4 import QtGui
from PyQt4.QtTest import QTest
from PyQt4.QtCore import Qt

import natcap.invest.iui
from natcap.invest.iui import modelui
from natcap.invest.iui import fileio


# If we're on Windows, we know that the model JSON files are located in the
# InVEST root.  If we're on Linux, they're in the iui module, along with this
# test file.
if platform.system() == 'Windows':
    FILE_BASE = '.'
else:
    FILE_BASE = os.path.dirname(os.path.abspath(__file__))

# Define a new exception class for when the worspace args_id is not found in the
# json definition
class WorkspaceNotFound(Exception): pass

def locate_workspace_element(ui):
    """Locate the workspace element, denoted by the args id 'workspace_dir'.

        ui - an instance of natcap.invest.iui.modelui.ModelUI to query.

        Raises a WorkspaceNotFound exception if the workspace is not found.
        If the workspace is found, returns a pointer to the matching element.
        """

    for element_id, element in ui.allElements.iteritems():
        try:
            args_id = element.attributes['args_id']
        except KeyError:
            # A KeyError is triggered when the element doesn't have an args_id.
            pass

        # If the retrieved arg id is the definde workspace, return the element.
        if args_id == 'workspace_dir':
            return element

    # If we can't find the workspace dir element, raise a helpful exception
    # indicating that it must be clearly identifiable.
    raise WorkspaceNotFound('The workspace must be identified by the '
        'args_id "workspace_dir"')


class ModelUITest(unittest.TestCase):
    """This test class exists to contain click-through tests for all InVEST 3.0
    models that use the ModelUI user interface.

    For each test, the following class-level attributes are created, which might
    be useful to a tester:
        self.workspace - the workspace folder that will be used for the model.
            This folder is created using the tempfile module before each test,
            and is deleted after each test.
        self.app - an instance of QtGui.QApplication, required for the
            functioning of Qt.

    Here is an example test, where no user interaction is required:

    >>> def test_example(self):
    >>>     # Define a path to the example json file.  If your json file is located
    >>>     # in the iui directory, use the FILE_BASE folder, defined above.
    >>>     # If your json file is located elsewhere on disk, you may define the
    >>>     # file however you like.
    >>>     file_path_to_json = os.path.join(FILE_BASE, 'example.json')
    >>>     model_ui = modelui.ModelUI(file_path_to_json, use_gui=True)
    >>>
    >>>     # Define a list of files relative to the output path to check once
    >>>     # the tool has finished running.  All paths in this list will be
    >>>     # asserted to exist.  This is case-sensitive, as it will be run on
    >>>     # both Windows and Linux.
    >>>     files_to_check = [
    >>>         'output/file_1.tif',
    >>>         'output/file_2.tif',
    >>>         'intermediate/file_a.tif',
    >>>         'intermediate/file_b.tif'
    >>>     ]
    >>>     
    >>>     # Run a helper function to actually click through the modelui and
    >>>     # check that the required files exist.  An exception will be raised if
    >>>     # a problem is found.
    >>>     self.click_through_model(model_ui, files_to_check)



    If a particular model requires user input, you may need to interact with the
    `model_ui` object before calling `self.click_through_model()`.  The
    interaction here will probably differ based on the interaction required.

    Here is an example from the Carbon model's clickthrough test, where an
    element needs to be selected and its state altered.

    >>> # Define the file path, as needed for the model_ui
    >>> file_path = os.path.join(FILE_BASE, 'carbon_biophysical.json')
    >>> model_ui = modelui.ModelUI(file_path, use_gui=True)
    >>>
    >>> # since we need to test both carbon biophysical and valuation, we need
    >>> # to check the checkbox to actually trigger the calculation of
    >>> # sequestration so that the valuation component can be run.
    >>> checkbox = model_ui.allElements['calc_sequestration']
    >>> checkbox.setValue(True)
    >>> QTest.qWait(500)  # so that validation can finish for enabled elements.

    Let's go into this in detail a bit.
    ===============================
    SELECTING AN ELEMENT:
    >>> checkbox = model_ui.allElements['calc_sequestration']

    In the above example, we can see that the `model_ui` object has an
    attribute called `allElements`.  This attribute is a flat dictionary
    mapping {"element_id (from JSON)": element_pointer}.  To locate an
    element pointer, you must query this dictionary with the element ID
    specified in the UI's JSON object.


    SETTING THE ELEMENT VALUE:
    >>> checkbox.setValue(True)
    >>> QTest.qWait(500)  # so that validation can finish for enabled elements.

    Most elements have a `setValue()` function, that takes a sensible input and
    appropriately sets the value of the widget element.  In the case of a
    checkbox element, it takes a boolean indicating the check state.  In the
    case of a file or folder element, it will take a string URI and set the
    field.  IMPORTANT: affecting the state of the UI will cause a cascade of
    other elements to in turn be affected, which will frequently cause
    validation to occur.  When validation happens, the UI is only informed of
    validation completion once every 50 ms.  So when you set the value of an
    element, it is CRITICAL that you call `QTest.qWait()` (with a reasonable
    number of milliseconds as the argument to qWait) to allow validation to
    complete for all affected elements.


    Also note that if you are trying to test multiple UIs in sequence, this is
    definitely possible to do.  See the `test_carbon()` function, below, for an
    example of how this can be done.

    Finally, if you have any questions about the UI, or if you're experiencing
    difficulty with some component of this test or the UI, email James ... it's
    probably my fault anyways :).
    """

    def click_through_model(self, model_ui, files_to_check):
        """Click through a standard modelui interface.

            model_ui - an instance of natcap.invest.iui.modelui.ModelUI
            files_to_check - a list of strings.  Each string must be a URI
                relative to the workspace.  All files at these URIs are required
                to exist.  An AssertionError will be thrown if a file does not
                exist.

        Returns nothing."""

        workspace_element = locate_workspace_element(model_ui)

        workspace_element.setValue(self.workspace)
        QTest.qWait(100)  # Wait for the test workspace to validate

        # Assert that the test workspace didn't get a validation error.
        self.assertEqual(workspace_element.has_error(), False)

        # Assert that there are no default data validation errors.
        validation_errors = model_ui.errors_exist()
        self.assertEqual(validation_errors, [], 'Validation errors '
            'exist for %s inputs. Errors: %s' % (len(validation_errors),
            validation_errors))

        # Click the 'Run' button and see what happens now.
        QTest.mouseClick(model_ui.runButton, Qt.MouseButton(1))

        # Now that the run button has been pressed, we need to check the state
        # of the operation dialog to see if it has finished completing.  This
        # check is done at half-secong intervals.
        while not model_ui.operationDialog.backButton.isEnabled():
            QTest.qWait(500)

        # Once the test has finished, click the back button on the dialog to
        # return toe the UI.
        QTest.mouseClick(model_ui.operationDialog.backButton, Qt.MouseButton(1))

        missing_files = []
        for filepath in files_to_check:
            full_filepath = os.path.join(self.workspace, filepath)
            if not os.path.exists(full_filepath):
                missing_files.append(filepath)

        self.assertEqual(missing_files, [], 'Some expected files were not '
            'found: %s' % missing_files)

    def setUp(self):
        # Before the test is run, we need to move the existing lastrun folder to
        # a temp folder.
        self.workspace = tempfile.mkdtemp()
        self.settings_folder = natcap.invest.iui.fileio.settings_folder()
        self.settings_folder_name = os.path.basename(self.settings_folder)
        self.temp_folder = tempfile.mkdtemp()
        try:
            shutil.move(self.settings_folder, self.temp_folder)
        except IOError:
            # Thrown when the folder at self.settings_folder does not exist.
            # Create a sample folder inside self.settings_folder that has the
            # correct name.
            os.mkdir(os.path.join(self.temp_folder, self.settings_folder_name))

        self.app = QtGui.QApplication(sys.argv)

    def tearDown(self):
        self.app.exit()
        self.app = None
        try:
            # Remove the workspace directory for the next test.
            shutil.rmtree(self.workspace)
        except OSError:
            # Thrown when there's no workspace to remove.
            pass

        # Restore the settings folder to its previous location, but only after
        # we delete the folder that has been created in its place.
        saved_settings_dir = os.path.join(self.temp_folder, self.settings_folder_name)
        shutil.rmtree(self.settings_folder)
        shutil.move(saved_settings_dir, self.settings_folder)

    def test_pollination(self):
        file_path = os.path.join(FILE_BASE, 'pollination.json')
        model_ui = modelui.ModelUI(file_path, True)

        files_to_check = [
            'intermediate/frm_Apis_cur.tif',
            'intermediate/hf_Apis_cur.tif',
            'intermediate/hn_Apis_cur.tif',
            'intermediate/sup_Apis_cur.tif',
            'intermediate/frm_Bombus_cur.tif',
            'intermediate/hf_Bombus_cur.tif',
            'intermediate/hn_Bombus_cur.tif',
            'intermediate/sup_Bombus_cur.tif',
            'output/frm_avg_cur.tif',
            'output/sup_tot_cur.tif'
        ]

        self.click_through_model(model_ui, files_to_check)

    def test_carbon(self):
        file_path = os.path.join(FILE_BASE, 'carbon_biophysical.json')
        model_ui = modelui.ModelUI(file_path, True)

        # since we need to test both carbon biophysical and valuation, we need
        # to check the checkbox to actually trigger the calculation of
        # sequestration so that the valuation component can be run.
        checkbox = model_ui.allElements['calc_sequestration']
        checkbox.setValue(True)
        #QTest.mouseClick(checkbox, Qt.MouseButton(1))
        QTest.qWait(500)  # so that validation can finish for enabled elements.

        files_to_check = [
            'Output/tot_C_cur.tif',
            'Output/tot_C_fut.tif',
            'Output/sequest.tif',
            'Intermediate/bio_hwp_cur.tif',
            'Intermediate/bio_hwp_fut.tif',
            'Intermediate/c_hwp_cur.tif',
            'Intermediate/c_hwp_fut.tif',
            'Intermediate/vol_hwp_cur.tif',
            'Intermediate/vol_hwp_fut.tif'
        ]
        self.click_through_model(model_ui, files_to_check)


        # Now that we've run the carbon biophysical model and checked that all
        # required files exist, do the same steps with carbon valuation.
        file_path = os.path.join(FILE_BASE, 'carbon_valuation.json')
        valuation_ui = modelui.ModelUI(file_path, True)

        # The sequestration file field needs to be set to contain the URI of the
        # sequestration raster output from the biophysical model.
        sequest_element = valuation_ui.allElements['sequest_uri']
        sequest_element.setValue(os.path.join(self.workspace, 'Output',
            'sequest.tif'))

        # only one output file to check!
        files_to_check = [
            'Output/value_seq.tif'
        ]
        self.click_through_model(valuation_ui, files_to_check)

if __name__ == '__main__':
    # This call to unittest.main() runs all unittests in this test suite, much
    # as we would expect Nose to do when we give it a file with tests to run.
    unittest.main()
