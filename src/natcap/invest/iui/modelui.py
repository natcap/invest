import sys
import os
import platform
from optparse import OptionParser
import logging
import multiprocessing
import tempfile

from PyQt4 import QtGui, QtCore

import base_widgets

CMD_FOLDER = '.'

# Set up logging for the modelUI
import natcap.invest
import natcap.invest.iui
LOGGER = natcap.invest.iui.get_ui_logger('modelUI')

class ModelUIRegistrar(base_widgets.ElementRegistrar):
    def __init__(self, root_ptr):
        super(ModelUIRegistrar, self).__init__(root_ptr)

        changes = {'file': base_widgets.FileEntry,
                   'folder': base_widgets.FileEntry,
                   'text': base_widgets.YearEntry
                    }

        self.update_map(changes)

class ModelUI(base_widgets.ExecRoot):
    def __init__(self, uri, main_window):
        """Constructor for the DynamicUI class, a subclass of DynamicGroup.
            DynamicUI loads all setting from a JSON object at the provided URI
            and recursively creates all elements.

            uri - the string URI to the JSON configuration file.
            main_window - an instance of base_widgets.MainWindow

            returns an instance of DynamicUI."""

        #the top buttonbox needs to be initialized before super() is called,
        #since super() also creates all elements based on the user's JSON config
        #this is important because QtGui displays elements in the order in which
        #they are added.
        layout = QtGui.QVBoxLayout()

        self.links = QtGui.QLabel()
        self.links.setOpenExternalLinks(True)
        self.links.setAlignment(QtCore.Qt.AlignRight)
        layout.addWidget(self.links)

        registrar = ModelUIRegistrar(self)
        self.okpressed = False

        base_widgets.ExecRoot.__init__(self, uri, layout, registrar, main_window)

        self.layout().setSizeConstraint(QtGui.QLayout.SetMinimumSize)

        try:
            title = self.attributes['label']
        except KeyError:
            title = 'InVEST'
        window_title = "%s" % (title)
        main_window.setWindowTitle(window_title)

        self.addLinks()

    def addLinks(self):
        links = []
        try:
            architecture = platform.architecture()[0]
            links.append('InVEST Version %s (%s)' % (natcap.invest.__version__,
                architecture))
        except AttributeError:
            links.append('InVEST Version UNKNOWN')

        try:
            doc_uri = 'file:///' + os.path.abspath(self.attributes['localDocURI'])
            links.append('<a href=\"%s\">Model documentation</a>' % doc_uri)
        except KeyError:
            # Thrown if attributes['localDocURI'] is not present
            print 'Attribute localDocURI not found for this model; skipping.'

        feedback_uri = 'http://forums.naturalcapitalproject.org/'
        links.append('<a href=\"%s\">Report an issue</a>' % feedback_uri)

        self.links.setText(' | '.join(links))

    def queueOperations(self):
        modelArgs = self.assembleOutputDict()
        self.operationDialog.exec_controller.add_operation('model',
                                                   modelArgs,
                                                   self.attributes['targetScript'])


def getFlatDefaultArgumentsDictionary(args):
    flatDict = {}
    if isinstance(args, dict):
        if 'args_id' in args and 'defaultValue' in args:
            flatDict[args['args_id']] = args['defaultValue']
        if 'elements' in args:
            flatDict.update(getFlatDefaultArgumentsDictionary(args['elements']))
    elif isinstance(args, list):
        for element in args:
            flatDict.update(getFlatDefaultArgumentsDictionary(element))

    return flatDict


def main(uri, use_gui=True):
    multiprocessing.freeze_support()
    # get the existing QApplication instance, or creating a new one if
    # necessary.
    app = QtGui.QApplication.instance()
    if app is None:
        app = QtGui.QApplication(sys.argv)

#    validate(json_args)

    # Check to see if the URI exists in the current directory.  If not, assume
    # it exists in the directory where this module exists.
    if not os.path.exists(uri):
        file_path = os.path.dirname(os.path.abspath(__file__))
        uri = os.path.join(file_path, os.path.basename(uri))

        # If the URI still doesn't exist, raise a helpful exception.
        if not os.path.exists(uri):
            raise Exception('Can\'t find the file %s.'%uri)

    window = base_widgets.MainWindow(ModelUI, uri)
    window.show()
    if use_gui:
        result = app.exec_()
    else:
        window.ui.resetParametersToDefaults()
        from PyQt4.QtTest import QTest
        # check to see if the model's target default workspace exists.  If it
        # does, set a new workspace.
        workspace_element = window.ui.allElements['workspace']
        target_workspace = workspace_element.value()
        if os.path.exists(target_workspace):
            new_path = tempfile.mkdtemp(
                dir=os.path.dirname(target_workspace),
                prefix="%s_" % os.path.basename(target_workspace))
            workspace_element.setValue(new_path)

        # if we're running NDR, we need to check one (or both) of the nutrient
        # checkboxes.
        if window.ui.attributes['modelName'] == 'nutrient':
            phosphorus_element = window.ui.allElements['calc_p']
            phosphorus_element.setValue(True)
        elif window.ui.attributes['modelName'] == 'wind_energy':
            window.ui.allElements['foundation_cost'].setValue('12')
            window.ui.allElements['discount_rate'].setValue('0.12')


        # wait for 100ms for events to process before clicking run.
        # Validation will be the most common event that the Qt main loop will
        # need to process, but other events include dynamic updating of element
        # interactivity and the extraction of fieldnames from tables and
        # vectors to use in a TableDropdown.
        # 100ms was chosen through experimentation.
        QTest.qWait(100)
        window.ui.runButton.click()

        # When operations finish (whether on success or failure), the back
        # button of the run dialog is enabled.  If the application has been
        # exited, the main window will not be visible.
        while not window.ui.operationDialog.backButton.isEnabled() \
                and window.isVisible():
            QTest.qWait(50)

        thread_failed = False
        if window.ui.operationDialog.exec_controller.thread_failed:
            thread_failed = True

        window.ui.operationDialog.closeWindow()

        # exit not-so-peacefully if we're running in test mode AND the thread
        # failed.  I'm assuming --test is not an oft-used option!
        # The other possible case for failure is if there are validation errors
        # and the application was quit.  We should display a failure if this is
        # the case.
        if thread_failed or window.ui.errors_exist():
            sys.exit(1)

if __name__ == '__main__':
    #Optparse module is deprecated since python 2.7.  Using here since OSGeo4W
    #is version 2.5.
    parser = OptionParser()
    parser.add_option('-t', '--test', action='store_false', default=True, dest='test')
    parser.add_option('-d', '--debug', action='store_true', default=False, dest='debug')
    (options, uri) = parser.parse_args(sys.argv)
    print uri

    if options.debug == True:
        level = logging.NOTSET
    else:
        level = logging.WARNING
    LOGGER.setLevel(level)
    LOGGER.debug('Level set to %s, option_passed = %s', level, options.debug)

    main(uri[1], options.test)

