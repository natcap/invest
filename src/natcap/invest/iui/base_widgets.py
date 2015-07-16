import sys
import imp
import os
import time
import traceback
import platform
import logging
import threading

from PyQt4 import QtGui, QtCore

try:
    QString = QtCore.QString
except AttributeError:
    # For when we can't use API version 1
    QString = unicode

import natcap.invest.iui
import iui_validator
import executor
import registrar
import fileio

#This is for something
CMD_FOLDER = '.'
INVEST_ROOT = './'
IUI_DIR = os.path.dirname(os.path.abspath(__file__))
ENCODING = sys.getfilesystemencoding()

# This data object is intended to be a thread-safe way to interact with data
# values that may be altered while our multi-threaded UI is running.
class Data(object):
    def __init__(self):
        self.data = {
            'last_dir': ''
        }
        self.lock = threading.Lock()

    def __getitem__(self, key):
        self.lock.acquire()
        data = self.data[key]
        self.lock.release()
        return data

    def __setitem__(self, key, value):
        self.lock.acquire()
        self.data[key] = value
        self.lock.release()

DATA = Data()  # common data is stored here, protected by thread-safe methods

class DynamicElement(QtGui.QWidget):
    """Create an object containing the skeleton of most functionality in the
        UI Interpreter's related classes.  It is not invoked directly by other
        IUI classes, but is instead used as a base class for almost all classes
        in the UI interpreter.  A diagram of this class heirarchy can be found
        at https://docs.google.com/drawings/d/13QZ6SsUwvoBPjvr0gf_X1X20sc35tLTr9oedX1vaUh8/edit

        DynamicElement serves as a base class for DynamicGroup and
        DynamicPrimitive.  The functions and object data it declares are shared
        by all subclasses."""

    def __init__(self, attributes):
        """This is the constructor for the DynamicElement object.

            attributes - a Python dictionary with the attributes of the element
                taken from the input JSON file.

            returns a DynamicElement object
            """

        #DynamicElement inherits QtGui.QWidget, so we'll call the constructor
        #for QWidget here.
        QtGui.QWidget.__init__(self)

        # Save a local logger instance with the logger name reflecting the class
        # we're in.
        try:
            self.LOGGER = natcap.invest.iui.get_ui_logger('bw.%s.%s' %
                (self.__class__.__name__, attributes['id'][0:10]))
        except KeyError as e:
            print "cant' find ID %s %s" % (str(attributes), self.__class__.__name__)
            raise e
        self.LOGGER.setLevel(logging.ERROR)

        self.LOGGER.debug('Initializing element %s', attributes['id'])
        #save a copy of the user-defined attributes for this element.  Based
        # on the specification of the JSON config file, the attributes array
        #may contain the attributes for other, to-be-created, elements.
        self.attributes = attributes

        #We initialize self.required as False here, since some widgets may not
        #actually be input elements.  This ensures that all widgets will be
        #marked optional unless specified by the user.
        if 'required' in self.attributes:
            self.required = self.attributes['required']

        #These attributes are set in self.updateLinks()
        self.root = None #a pointer to the root of this UI
        self.enabledBy = None #a pointer to an element, if an ID is specified.
        self.disabledBy = None
        self.enables = [] #a list of pointers
        self.disables = [] #a list of pointers
        self.isDisabled = False #is the object disabled by an object that's 'activated'?
        self.requiredIf = [] #a list of pointers
        self.triggers = {}  # a dictionary of trigger strings -> pointers

        #initialize the elements array in case the user has defined sub-elements
        self.elements = []

    def getRoot(self):
        """Search up the Qt widget tree until we find the root element, which
            is by definition an instance of the class DynamicUI.  A pointer to
            the root is usually saved to the local object as self.root.

            returns a pointer to the instance of DynamicUI"""

        parent = self.parentWidget()
        if issubclass(parent.__class__, Root):
            return parent
        else:
            return parent.getRoot()

    def requirementsMet(self):
        return True

    def updateLinks(self, rootPointer):
        """Update dependency links for this element.

            All elements (although usually only subclasses of DynamicPrimitive)
            have the ability to be enabled by other elements"""

        self.root = rootPointer
        #enabledBy is only a single string ID
        if 'enabledBy' in self.attributes:
            idString = self.attributes['enabledBy']
            self.enabledBy = self.root.find_element_ptr(idString)
            self.enabledBy.enables.append(self)

        # disabledBy is similar to enabledBy; it's a single string ID
        if 'disabledBy' in self.attributes:
            idString = self.attributes['disabledBy']
            self.disabledBy = self.root.find_element_ptr(idString)
            self.disabledBy.disables.append(self)


        #requiredIf is a list
        if 'requiredIf' in self.attributes:
            for idString in self.attributes['requiredIf']:
                elementPointer = self.root.find_element_ptr(idString)
                self.requiredIf.append(elementPointer)

    def isRequired(self):
        """Check to see if this element is required.

            An element is required if element.required == True or if the element
            has a 'requiredIf' attribute list and at least one of the elements
            in the 'requiredIf' list is required.

            returns a boolean"""

        if hasattr(self, 'required'):
            return self.required
        else:
            for element in self.requiredIf:
                if element.requirementsMet():
                    return True
        return False

    def setState(self, state, includeSelf=True, recursive=True):
        # If the object is being disabled by another element,
        # don't let it get enabled. A 'disabledBy' should override
        # an 'enabledBy'.
        if state and self.isDisabled:
            return

        self.LOGGER.debug('Setting state in %s', self.attributes['id'])
        self.LOGGER.debug('state=%s, includeSelf=%s, recursive=%s',
            state, includeSelf, recursive)
        if includeSelf:
            self.setEnabled(state)
            for element in self.elements:
                element.setEnabled(state)

        if recursive:
            try:
                state = state and self.requirementsMet() and not self.has_error()
            except AttributeError:
                # Thrown when this element does not have self.has_error()
                state = state and self.requirementsMet()

            for element in self.enables:
                element.setState(state)

            for element in self.disables:
                element.isDisabled = state
                element.setState(not state)

    def getLabel(self):
        raise DeprecationWarning
        if 'inheritLabelFrom' in self.attributes:
            target_id = self.attributes['inheritLabelFrom']
            return self.root.allElements[target_id].attributes['label']
        elif 'label' in self.attributes:
            id = self.attributes['label']
            if id in self.root.allElements:
                return self.root.allElements[id].attributes['label']
            else:
                return self.attributes['label']
        else:
            return ''

    def getElementsDictionary(self):
        return {self.attributes['id']: self}

    def set_element_state(self, state):
        """Set the element's state from state.  State is assumed to come from
        self.get_element_state()."""

        self.setValue(state)

    def get_element_state(self):
        """This function gets the relevant element state information for setting
        a lastrun dictionary.  It defaults to just return the value of the
        element (whatever its return value is), but can be overridden to return
        more specific element information.  Information saved in this way can be
        set by using the function 'set_element_state'."""

        return self.value()

class DynamicGroup(DynamicElement):
    """Creates an object intended for grouping other elements together.

        DynamicGroup is a subclass of DynamicElement and thus inherits all
        attributes and functions of the DynamicElement class.

        DynamicUI, Container, CollapsibleContainer and List are all
        subclasses of DynamicGroup.

        The DynamicGroup object allows other elements to be grouped together
        using any arbitrary layout mechanism compatible with Qt.  If a custom
        layout manager is used, it may be necessary to revisit the
        DynamicGroup.createElements() function to define exactly how the
        elements created are to be added to the new layout.

        As all possible grouping objects in this Interpreter subclass
        DynamicGroup, if a new widget is to be added, it must likewise be added
        to the if-elif block in createElements.  The element will not be created
        if there is no corresponding entry in createElements()
        """

    def __init__(self, attributes, layout=QtGui.QVBoxLayout(), registrar=None):
        """Constructor for the DynamicGroup class.
            Most object construction has been abstracted to the DynamicElement
            class.  The defining feature of a DynamicGroup from a DynamicElement
            is that a DynamicGroup has a layout and can contain elements if
            they have been defined by the user.

            attributes - a python dictionary with the attributes for this group
                parsed from the user-defined JSON object.
            layout - a layout mechanism compatible with Qt4 or a subclass of
                such a layout manager.
            registrar=None - An (optional) instance of
                base_widgets.ElementRegistrar.  Required if creating elements
                within this DynamicGroup.

            returns an instance of DynamicGroup"""

        #create the object by initializing its superclass, DynamicElement.
        DynamicElement.__init__(self, attributes)

        #set the layout for this group(a subclass of DynamicElement, which
        #itself is a subclass of QtGui.QWidget) so that we can add widgets
        #as necessary.
        self.setLayout(layout)

        self.registrar = registrar
        if registrar != None:
            self.createElements(attributes['elements'])

    def createElements(self, elementsArray, start_index=0):
        """Create the elements defined in elementsArray as widgets within
            this current grouping widget.  All elements are created as widgets
            within this grouping widget's layout.

            elementsArray - a python array of elements, where each element is a
                python dictionary with string keys to each of its attributes as
                defined in the input JSON file.
            start_index=0 - a python int representing the row to start appending
                to.

            no return value"""

        #We initialize a counter here to keep track of which row we occupy
        # in this iteration of the loop.  Used excluseively when the layout
        #manager is an instance of QtGui.QGridLayout(), as both the row and
        #column indices are required in QGridLayout's addWidget() method.
        i = start_index

        #loop through all entries in the input elementsArray and create the
        #appropriate elements.  As new element classes are created, they must
        #likewise be entered here to be created.
        for values in elementsArray:
            widget = self.registrar.eval(values['type'], values)

            # Check to see if the widget has a valid size hint in this layout.
            # If so, be sure that the widget's minimum size is set to the size
            # hint.  This addresses a longstanding issue with widget sizes in a
            # QGridLayout.
            if widget.sizeHint().isValid():
                widget.setMinimumSize(widget.sizeHint())

            #If an unusual layoutManager has been declared, it may be necessary
            #to add a new clause to this conditional block.
            if isinstance(self.layout(), QtGui.QGridLayout):
                j = 0
                for subElement in widget.elements:
                    if subElement.sizeHint().isValid():
                        subElement.setMinimumSize(subElement.sizeHint())
                    self.layout().addWidget(subElement, i, j)
                    j += 1
            else:
                self.layout().addWidget(widget)


            #the self.elements array is used for maintaining a list of pointers
            #to elements associated with this group.
            self.elements.append(widget)
            i += 1

    def getElementsDictionary(self):
        """Assemble a flat dictionary of all elements contained in this group.

            This function loops through the self.elements array and attempts to
            retrieve a dictionary for each sub-element.  If a sub-element
            dictionary can be retrieved, it is concatenated with the existing
            dictionary.

            Such a flat dictionary structure is convenient for iterating over
            all elements in the UI.

            returns a python dictionary mapping widget id (a string) to an
                element pointer."""

        outputDict = {}
        outputDict[self.attributes['id']] = self
        for element in self.elements:
            #Create an entry in the output dictionary for the current element
            #outputDict[element.attributes['id']] = element #superceded by
            #dynamicElement.getElementsDictionary()
            try:
                outputDict.update(element.getElementsDictionary())
            except AttributeError:
                pass

        return outputDict

    def getOutputValue(self):
        if 'args_id' in self.attributes and self.isEnabled():
            return self.value()

    def value(self):
        """TO BE IMPLEMENTED"""
        return True

    def setValue(self, value):
        """TO BE IMPLEMENTED"""
        pass

class DynamicPrimitive(DynamicElement):
    """DynamicPrimitive represents the class of all elements that can be listed
        individually in the JSON file that themselves cannot group other
        elements.  As such, DynamicPrimitive is the superclass of all input
        elements.

        DynamicText and CheckBox inherit DynamicPrimitive, and FileEntry,
        YearEntry inherit DynamicText (thus also inheriting DynamicPrimitive).

        There are two defining attributes of DynamicPrimitive:
         - self.elements (a python Array)
         - self.attributes['args_id'] (a string)

        self.elements is an array of the widgets that make up this element.  By
        default, this is set to [self], implying that a subclass has at least
        one widget.  In cases where a subclass of DynamicPrimitive has multiple
        widgets, the elements array is used to determine the order in which the
        elements are added to the GUI in DynamicGroup.createElements().

        self.attributes['args_id'] is an optional string provided by the user
        that enables the construction of an arguments dictionary in python that
        will be passed to the specified python program.  The args_id must
        conform with the API specified by the desired model.

        Note that all implemented instances of DynamicPrimitive must implement
        their own setValue(value) function, specific to the target QWidget.
         - self.setValue(value) is a function that allows a developer to specify
             the value of the current element, depending on how the value needs
             to be set based on the class and class elements and the type of
             the value.
        """

    def __init__(self, attributes):
        """Constructor for the DynamicPrimitive class.
            Because DynamicPrimitive inherits DynamicElement, most of the obect
            construction has been abstracted away to the DynamicElement
            constructor or to the superclass of DynamicElement, QWidget.

            attributes - a python dictionary with attributes for this element.
                Attribute keys are defined in the JSON schema declaration in
                this python file.

            returns an instance of DynamicPrimitive"""

        super(DynamicPrimitive, self).__init__(attributes)

        # Prepend all elements with a validation button that will contain either
        # an icon indicating validation status or nothing if no validation is
        # taking place for this element.  Also, the button should not be
        # pressable unless the button has an icon, so defaulting to disabled.
        try:
            help_text = self.attributes['helpText']
        except KeyError:
            help_text = None

        try:
            label = self.attributes['label']
        except KeyError:
            label = ''
        self.info_button = InformationButton(label, help_text)
        self.error_button = ErrorButton(label)

        self.elements = [self.error_button]
        if help_text is not None:
            self.elements.append(self.info_button)
        self.elements.append(self)

        if 'validateAs' in self.attributes:
            validator_type = self.attributes['validateAs']['type']
            self.validator = iui_validator.Validator(validator_type)
            self.timer = QtCore.QTimer()
        else:
            self.validator = None

    def setState(self, state, includeSelf=True, recursive=True):
        if state == False:
            self.error_button.deactivate()
            self.setBGcolorSatisfied(True)
        else:
            self.validate()

        DynamicElement.setState(self, state, includeSelf, recursive)

    def resetValue(self):
        """If a default value has been specified, reset this element to its
            default.  Otherwise, leave the element alone.

            returns nothing."""

        if 'defaultValue' in self.attributes:
            self.setValue(self.attributes['defaultValue'])

        if 'enabled' in self.attributes:
            self.setState(self.attributes['enabled'])

    def setValue(self, value):
        pass

    def getElementsDictionary(self):
        """Assemble a python dictionary mapping this object's string ID to its
            pointer.

            self.getElementsDictionary is called to build a flat dictionary of
            all elements in the entire UI.  Considering that subclasses of
            DynamicPrimitive are the most atomic elements the user can control
            in the JSON file, it follows that subclasses of DynamicPrimitive
            should return the most primitive such dictionary.

            returns a python dict mapping string ID -> this object's pointer."""

        return {self.attributes['id'] : self}

    def cast_value(self):
        try:
            return self.root.type_registrar.eval(self.attributes['dataType'],
                self.value())
        except KeyError:
            return unicode(self.value())

    def getOutputValue(self):
        """Return the output value of this element, applying any necessary
            filters.  This function is intended to be called when assembling the
            output dictionary.  Returns the appropriate output value of this
            element."""
        if 'args_id' in self.attributes and self.isEnabled():
            value = self.value()

            # Check to see if the element should be passed if it's empty.
            try:
                if len(value) == 0:
                    try:
                        if self.attributes['returns']['ifEmpty'] == 'pass':
                            return None
                    except KeyError:
                        pass
            except TypeError:
                #if value is a boolean can't do length, just ignore
                pass

            if (value != '' and value != None and not isinstance(value, dict) and
                not isinstance(value, list)):
                return self.cast_value()
            return value

    def set_error(self, error, state):
        if error == None or error == '':
            msg = ''
        else:
            msg = str(error)

        satisfied = False
        if state == 'warning' or state == 'pass' or state == None:
            satisfied = True

        # If the current element is not enabled, we do not want to
        # display the error message until the element is enabled.
        if not self.isEnabled():
            satisfied = True
            state = None

        self.setBGcolorSatisfied(satisfied)
        self.error_button.set_error(msg, state)

    def has_error(self):
        if self.error_button.error_state == 'error':
            return True
        return False

    def has_warning(self):
        if self.error_button.error_state == 'warning':
            return True
        return False

    def validate(self):
        if self.isRequired() and not self.requirementsMet():
            self.set_error('Element is required', 'error')
            DynamicElement.setState(self, False, False, True)
        else:
            # Assume that validation passes until we are proven otherwise.
            self.set_error(None, None)

            # If the root element has not yet been set, we should just return since
            # validation will fail anyways.
            if self.root == None:
                self.LOGGER.warn("No root defined.  Skipping validation for %s",
                    self.attributes['id'])
                return

            if self.isEnabled() and self.validator != None and\
            self.validator.thread_finished():
                # We actually only want to validate input if the element's
                # requirements are met.  If not, we still want to simulate the
                # completion of the validation thread by performing the
                # post-completion operations.
                if self.requirementsMet():
                    rendered_dict = self.root.assembler.assemble(self.value(),
                        self.attributes['validateAs'])
                    self.validator.validate(rendered_dict)
                    self.timer.timeout.connect(self.check_validation_error)
                    self.timer.start(50)
                else:
                    self.check_validation_error()

    def check_validation_error(self):
        if self.validator.thread_finished():
            self.timer.stop()
            error, state = self.validator.get_error()

            # If this element's requirements are not met, no validation status
            # should be displayed.  Otherwise, set the error as normal.
            if not self.requirementsMet():
                state = None
            else:
                if state == None:
                    state = 'pass'
            self.set_error(error, state)

            # Toggle dependent elements based on the results of this validation
            enable = not self.has_error() and self.requirementsMet()
            DynamicElement.setState(self, enable, includeSelf=False,
                recursive=True)

class InformationButton(QtGui.QPushButton):
    """This class represents the information that a user will see when pressing
        the information button.  This specific class simply represents an object
        that has a couple of string attributes that may be changed at will, and
        then constructed into a cohesive string by calling self.build_contents.

        Note that this class supports the presentation of an error message.  If
        the error message is to be shown to the end user, it must be set after
        the creation of the InformationPopup instance by calling
        self.set_error().
        """
    def __init__(self, title, body_text=''):
        """This function initializes the InformationPopup class.
            title - a python string.  The title of the element.
            body_text - a python string.  The body of the text

            returns nothing."""

        QtGui.QPushButton.__init__(self)
        self.title = title
        self.body_text = body_text
        self.pressed.connect(self.show_info_popup)
        self.setFlat(True)
        self.setIcon(QtGui.QIcon(os.path.join(IUI_DIR, 'info.png')))
        self.setSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)

        # If the user has set "helpText": null in JSON, deactivate.
        if body_text == None:
            self.deactivate()

    def show_info_popup(self):
        """Show the information popup.  This manually (programmatically) enters
            What's This? mode and spawns the tooltip at the location of trigger,
            the element that triggered this function.
            """

        self.setWhatsThis(self.build_contents())  # set popup text
        QtGui.QWhatsThis.enterWhatsThisMode()
        QtGui.QWhatsThis.showText(self.pos(), self.whatsThis(), self)

    def deactivate(self):
        """Visually disable the button: set it to be flat, disable it, and clear
            its icon."""
        self.setFlat(True)
        self.setEnabled(False)
        self.setIcon(QtGui.QIcon(''))

    def set_title(self, title_text):
        """Set the title of the InformationPopup text.  title_text is a python
            string."""
        self.title = title_text

    def set_body(self, body_string):
        """Set the body of the InformationPopup.  body_string is a python
            string."""
        self.body_text = body_string

    def build_contents(self):
        """Take the python string components of this instance of
            InformationPopup, wrap them up in HTML as necessary and return a
            single string containing HTML markup.  Returns a python string."""
        width_table = '<table style="width:400px"></table>'
        title = '<h3 style="color:black">%s</h3><br/>' % (self.title)
        body = '<div style="color:black">%s</div>' % (self.body_text)

        return title + body + width_table

class ErrorButton(InformationButton):
    def __init__(self, title, body_text=''):
        """Initialize the ErrorPopup object.  Adding the self.error_text
        attribute.  Title and body_text are python strings."""
        InformationButton.__init__(self, title, body_text)
        self.error_text = ''
        self.error_state = None
        self.deactivate()

    def setEnabled(self, state):
        if state == False:
            self.setIcon(QtGui.QIcon(''))
        else:
            self.set_error(self.error_text, self.error_state)

        QtGui.QWidget.setEnabled(self, state)

    def set_error(self, error_string, state):
        """Set the error string of this InformationPopup and also set this
            button's icon according to the error contained in error_string.
            error_string is a python string."""

        self.error_text = error_string
        self.error_state = state
        button_is_flat = False
        button_icon = ''

        if state == 'error':
            button_icon = 'validate-fail.png'
        elif state == 'warning':
            button_icon = 'dialog-warning.png'
        elif state == 'pass':
            button_icon = 'validate-pass.png'

        if state == 'pass' or state == None:
            button_is_flat = True

        self.setIcon(QtGui.QIcon(os.path.join(IUI_DIR, button_icon)))
        self.setFlat(button_is_flat)
        QtGui.QWidget.setEnabled(self, True)  # enable the button; validation has completed

    def build_contents(self):
        """Take the python string components of this instance of
            InformationPopup, wrap them up in HTML as necessary and return a
            single string containing HTML markup.  Returns a python string."""
        width_table = '<table style="width:400px"></table>'
        title = '<h3 style="color:black">%s</h3><br/>' % (self.title)

        #### CHECK ERROR STATE TO DETERMINE TEXT
        if self.error_state == 'warning':
            color = 'orange'
            text = 'WARNING:'
        elif self.error_state == 'error':
            color = 'red'
            text = 'ERROR:'
        else:
            color = 'green'
            text = 'Validation successful'

        message = '<b style="color:%s">%s %s</b><br/>' % (color, text,
            self.error_text)

        body = '<div style="color:black">%s</div>' % (self.body_text)

        return str(title + message + body + width_table)

class LabeledElement(DynamicPrimitive):
    def __init__(self, attributes):
        DynamicPrimitive.__init__(self, attributes)
        self.label = QtGui.QLabel(attributes['label'])
        self.elements = [self.error_button, self.label, self.info_button]

    def addElement(self, element):
        self.elements.insert(len(self.elements)-1, element)

    def initState(self):
        if self.isEnabled():
            self.validate()

    def setState(self, state, includeSelf=True, recursive=True):
        DynamicPrimitive.setState(self, state, includeSelf, recursive)

    def isEnabled(self):
        #Labeled elements are designed to have more than one element, but in
        #case there isn't, the label still should have an enabled() attribute.
        if len(self.elements) == 0:
            return self.elements[0].isEnabled
        return self.elements[1].isEnabled()

    def setBGcolorSatisfied(self, satisfied=True):
        """Color the background of this element's label.

            satisfied=True - a boolean, indicating whether this element's
                requirements have been satisfied.

            returns nothing"""

        if satisfied:
            self.label.setStyleSheet("QWidget {}")
        else:
            self.label.setStyleSheet("QWidget { color: red }")


class StaticReturn(DynamicPrimitive):
    def __init__(self, attributes):
        DynamicPrimitive.__init__(self, attributes)

    def value(self):
        if 'returns' in self.attributes:
            return self.attributes['returns']

class Label(QtGui.QLabel, StaticReturn):
    def __init__(self, attributes):
        QtGui.QLabel.__init__(self)
        StaticReturn.__init__(self, attributes)
        self.setText(attributes['label'])
        self.setTextFormat(QtCore.Qt.RichText)
        self.setWordWrap(True)
        self.elements = [self.error_button, self, self.info_button]

class DynamicText(LabeledElement):
    """Creates an object containing a label and a sigle-line text field for
        user input.

        DynamicText is a subclass of DynamicPrimitive and thus inherits all its
        methods and attributes.

        FileEntry and YearEntry inherit DynamicText.

        As the superclass to a number of text-based elements, DynamicText
        implements a text-only option, namely validText.  DynamicText also
        implements the attribute defaultValue.
        """

    class TextField(QtGui.QLineEdit):
        def __init__(self):
            QtGui.QLineEdit.__init__(self)

            self.button = QtGui.QToolButton(self)
            self.button.setIcon(QtGui.QIcon(os.path.join(IUI_DIR,
                'validate-pass.png')))
            self.button.setStyleSheet('border: 0px; padding: 0px;')
            self.button.setVisible(False)

            # Only set the text margin if the icon button is visible.
            if self.button.isVisible():
                self.setTextMargins(0, 0, self.button.sizeHint().width() + 1, 0)

        def resizeEvent(self, event):
            """Reimplemented from QtGui.QLineEdit.resizeEvent.

            This reimplemented function allows me to move around the datatype
            button when the window is resized, before Qt handles all its normal
            resize event routines."""

            # We only care about moving the button around if the button is
            # visible.  If it isn't visible, we just call down to the
            # QLineEdit's resize event handler.
            if self.button.isVisible():
                buttonsize = self.button.sizeHint()
                framewidth = self.style().pixelMetric(QtGui.QStyle.PM_DefaultFrameWidth)
                self.button.move((self.rect().right() - framewidth -
                    buttonsize.width()), (self.rect().bottom() - buttonsize.height() +
                    1)/2)
            QtGui.QLineEdit.resizeEvent(self, event)

        def emit_textchanged(self, state=None):
            """This is a wrapper function that allows me to emit the textChanged
            signal more easily.  It simply emits the textChanged signal with the
            current contents of the textfield."""
            self.textChanged.emit(self.text())

        def contextMenuEvent(self, event=None):
            """Reimplemented from QtGui.QLineEdit.contextMenuEvent.

            This function allows me to make changes to the context menu when one
            is requested before I show the menu."""
            menu = self.createStandardContextMenu()
            refresh_action = QtGui.QAction('Refresh', menu)
            refresh_action.setIcon(QtGui.QIcon(os.path.join(IUI_DIR,
                'refresh.png')))
            refresh_action.triggered.connect(self.emit_textchanged)
            menu.addAction(refresh_action)
            menu.exec_(event.globalPos())

    def __init__(self, attributes):
        """Constructor for the DynamicText class.
            The defining features for this class have primarily to do with user
            interaction: a child of DynamicText can be required, can have
            defaultValue and can have valid text.

            attributes -a python dictionary of element attributes.

            returns a constructed DynamicText object."""

        #create the object by initializing its superclass, DynamicElement.
        super(DynamicText, self).__init__(attributes)

        #create the new Label widget and save it locally.  This label is
        #consistent across all included subclasses of DynamicText.
#        self.label = QtGui.QLabel(attributes['label'])

        #create the new textField widget and save it locally.  This textfield
        #is consistent across all included subclasses of DynamicText, though
        #the way the textfield is used may differ from class to class.
        if not hasattr(self, 'textField'):
            #self.textField = QtGui.QLineEdit()
            self.textField = self.TextField()

        #All subclasses of DynamicText must contain at least these two elements.
        self.addElement(self.textField)

        #Set self.required to the user's specification (could be True or False)
        if "required" in attributes:
            self.required = attributes['required']

        #If the user has defined some default text for this text field, insert
        #it into the text field.
        if "defaultValue" in attributes:
            self.setValue(attributes['defaultValue'])

        #If the user has defined a string regular expression of text the user is
        #allowed to input, set that validator up with the setValidateField()
        #function.
        if 'validText' in attributes:
            self.setValidateField(attributes['validText'])

        #Connect the textfield's textChanged signal to the toggle() function.
        self.textField.textChanged.connect(self.toggle)


    def toggle(self):
        """Toggle all elements associated with this element's ID.

            This function has several purposes:
              - It instructs the root element to update its requirement
                notification based on the current status of all elements
              - It sets the backgroundColor of this object's label if its
                completion requirements have not been met
              - It instructs the root element to toggle all other elements
                appropriately.

            returns nothing."""
        self.setBGcolorSatisfied(True)  # assume valid until validation fails
        self.error_button.deactivate()
        self.validate()

    def setValidateField(self, regexp):
        """Set input validation on the text field to conform with the input
            regular expression.  Validation takes place continuously, so the
            user will be unable to enter text in this field unless it conforms
            to the regexp.

            regexp - a string regular expression

            returns nothing"""

        regexpObj = QtCore.QRegExp(regexp)
        validator = QtGui.QRegExpValidator(regexpObj, self.textField)
        self.textField.setValidator(validator)

    def requirementsMet(self):
        """Determine whether the textfield is considered 'complete'.
            This is used to determine whether a dependent element should be
            enabled or disabled and may need to be reimplemented for a subclass
            as new text-based elements arise.

            As a basic form of completion, we assume that this field is
            satisfied when some text (any text) has been entered.

            returns a boolean"""

        try:
            input_length = len(self.value())
        except TypeError:
            # A TypeError is returned if self.value() returns None, which may
            # happen when the json-defined blank value is set to 'isEmpty':
            # 'pass'.
            input_length = 0

        if input_length > 0:
            return True
        return False

    def setBGcolorSatisfied(self, satisfied=True):
        """Color the background of this element's label.

            satisfied=True - a boolean, indicating whether this element's
                requirements have been satisfied.

            returns nothing"""

        if satisfied:
            self.label.setStyleSheet("QWidget {}")
            self.textField.setStyleSheet("QWidget {}")
        else:
            self.label.setStyleSheet("QWidget { color: red }")
            self.textField.setStyleSheet("QWidget { border: 1px solid red } ")

    def parentWidget(self):
        """Return the parent widget of one of the QWidgets of this object.

            Because DynamicText objects by definition have at least two widgets
            which individually could be added to separate layouts of separate
            widgets, it is necessary to specify which local widget we wish to
            identify as having the parent.

            In this case, self.textField has been selected.

            returns a pointer to an instance of a QWidget."""

        return self.textField.parentWidget()

    def isEnabled(self):
        """Check to see if this element is 'enabled'.

            This status is commonly used to determine whether other fields
            should be enabled or disabled (to allow or prevent the user from
            interacting with the widget)

            This is tested by checking the length of the string entered into
            self.textField.  Specific implementations may differ as appropriate
            to the subclass.

            returns a boolean."""

        return self.textField.isEnabled()

    def value(self):
        """Fetch the value of the user's input, stored in self.textField.

            returns a string."""
        value = self.textField.text()
        try:
            value = unicode(value, 'utf-8')
        except TypeError:
            # when casting unicode to unicode
            pass
        return value

    def setValue(self, text):
        """Set the value of self.textField.

            text - a string, the text to be inserted into self.textField.

            returns nothing."""

        self.textField.setText(text)
        self.toggle()  # Should cause validation to occur


class Container(QtGui.QGroupBox, DynamicGroup):
    """Class Container represents a QGroupBox (which is akin to the HTML widget
        'fieldset'.  It has a Vertical layout, but may be subclassed if a
        different layout is needed."""

    def __init__(self, attributes, registrar=None):
        """Constructor for the Container class.

            attributes - a python dictionary containing all attributes of this
                container, including its elements.  Elements are initialized in
                DynamicGroup.createElements().

            returns an instance of Container."""

        super(Container, self).__init__(attributes, QtGui.QVBoxLayout(), registrar)

        #set the title of the container
        self.setTitle(attributes['label'])
        self.LOGGER.debug('Title=%s', attributes['label'])

        if 'collapsible' in self.attributes:
            self.LOGGER.debug('Collapsible=%s', self.attributes['collapsible'])
            #this attribute of QtGui.QGroupBox determines whether the container
            #will sport a hide/reveal checkbox.
            self.setCheckable(self.attributes['collapsible'])
            self.setChecked(False)

            if self.attributes['collapsible'] == True:
                for element in self.elements:
                    self.LOGGER.debug('Hiding element %s', element)
                    element.setVisible(False)

                self.toggled.connect(self.toggleHiding)

            if 'style' in self.attributes:
                if self.attributes['style'] == 'arrows':
                    self.setStyleSheet('QGroupBox::indicator:unchecked {' +
                        'image: url(%s/dialog-yes-small.png);}'% IUI_DIR +
                        'QGroupBox::indicator:checked {' +
                        'image: url(%s/dialog-no-small.png);}'% IUI_DIR +
                        'QGroupBox::indicator:checked:pressed {' +
                        'image: url(%s/dialog-no-small.png);}'% IUI_DIR +
                        'QGroupBox::indicator:unchecked:pressed {' +
                        'image: url(%s/dialog-yes-small.png);}'% IUI_DIR +
                        'QGroupBox::indicator {width: 12px; height: 12px;}')

        if 'enabled' in self.attributes:
            self.LOGGER.debug('Setting enabled=%s', self.attributes['enabled'])
            self.setEnabled(self.attributes['enabled'])

    def toggleHiding(self, state):
        """Show or hide all sub-elements of container (if collapsible) as
            necessary.  This function is a callback for the toggled() signal.

            returns nothing."""

        self.LOGGER.debug('Toggling hiding of contained elements.')
        for element in self.elements:
            self.LOGGER.debug('Setting %s to state=%s', element, state)
            element.setVisible(state)
            element.setEnabled(state)

        self.LOGGER.debug('Setting minimum size from local sizeHint')
        self.setMinimumSize(self.sizeHint())

        self.LOGGER.debug('Calling down to DynamicGroup.setState')
        DynamicGroup.setState(self, state, includeSelf=False, recursive=True)

    def resetValue(self):
        if 'defaultValue' in self.attributes:
            self.setChecked(self.attributes['defaultValue'])

        self.setState(self.isEnabled() or self.isChecked(), includeSelf=False,
            recursive=True)

    def requirementsMet(self):
        """This function is used to return whether the container is enabled or
        not.  Used for determining whether other elements should be triggered.

        If the container is checkable, the check state is returned.  If the
        container is not checkable, True is returned."""
        if self.isCheckable():
            return self.isChecked()
        else:
            return True

    def value(self):
        return self.isChecked()

    def setValue(self, value):
        self.setChecked(value)

    def setState(self, state, includeSelf=False, recursive=True):
        """Reimplemented from Container.setState.  When this container is
        collapsible, we only want to set the state of contained elements when
        the collapsible container is open.  Otherwise, pass."""

        # if the checkbox is:
        #   - disabled,
        #   - checkable (collapsible), and
        #   - checked,
        # then, we want to set the checkbox to be unchecked.
        self.LOGGER.debug('State=%s.  Checkable=%s. Checked=%s', state,
            self.isCheckable(), self.isChecked())
        if state == False and self.isCheckable() and self.isChecked():
            self.setChecked(False)

        if self.isCheckable():
            # we always want to include the checkbox itself when this container is
            # toggled.
            includeSelf=True

        DynamicGroup.setState(self, state, includeSelf, recursive)


class MultiElement(Container):
    """Defines a class that allows the user to select an arbitrary number of the
    same input by providing an hyperlink by which to add another element.
    Validation applies as usual and the same validation is applied to all
    elements.  As a result, it is best to have a single multi-input element for
    each desired validation or input type as inputs cannot be mixed and matched.

    example JSON:
        "id": "multi-file",
        "type": "multiFile",
        "label": "Test multi-file",
        "sampleElement": {"id": "sample_id",
                          "type": "text",
                          "label": "Input raster",
                          "validateAs": {"type": "GDAL"}},
        "linkText": "Add another"
    """
    class MinusButton(QtGui.QPushButton):
        """This class defines the '-' button that is used by the MultiFile
        class."""
        def __init__(self, row_num, parent):
            QtGui.QPushButton.__init__(self)
            self.row_num = row_num
            self.parent = parent
            self.pressed.connect(self.remove_element)
            self.setIcon(QtGui.QIcon(os.path.join(IUI_DIR, 'list-remove.png')))

        def remove_element(self):
            """A callback that is triggered when the button is pressed."""
            self.parent.remove_element(self.row_num)

    def __init__(self, attributes, registrar=None):
        # If the user has not defined any extra elements to be added to this
        # group, assume that there are no elements.
        if 'elements' not in attributes:
            attributes['elements'] = []

        Container.__init__(self, attributes, registrar)
        self.file_def = attributes['sampleElement']

        if 'linkText' not in attributes:
            attributes['linkText'] = 'Add another'

        group_def = {'id': 'group',
                     'type': 'list',
                     'elements': []}

        self.multi_widget = GridList(group_def, registrar=self.registrar)
        self.layout().addWidget(self.multi_widget)
        self.multi_widget.setMinimumSize(self.multi_widget.sizeHint())

        self.create_element_link = QtGui.QLabel('<a href=\'google.com\'' +
            '>%s</a>' % attributes['linkText'])
        self.create_element_link.linkActivated.connect(self.add_element_callback)
        self.multi_widget.layout().addWidget(self.create_element_link,
            self.multi_widget.layout().rowCount(), 2)

        if 'defaultValue' in attributes:
            if not isinstance(attributes['defaultValue'], list):
                default_list = [attributes['defaultValue']]
            else:
                default_list = attributes['defaultValue']

            for default_value in default_list:
                self.LOGGER.debug('Setting default value of %s to "%s"',
                    attributes['id'], default_value)
                self.add_element(default_value)

    def add_element_callback(self, event=None):
        """Function wrapper for add_element.  event is expected to be a Qt
        event.  It is ignored."""
        self.add_element()

    def remove_element(self, row_num):
        """Remove the element located at row_num in the layout.  row_num is
            assumed to be 1-based, not zero-based."""

        for element in self.multi_widget.elements:
            element_row_num = element.elements[1].row_num
            if element_row_num == row_num:
                self.multi_widget.elements.remove(element)
                break

        for j in range(self.multi_widget.layout().columnCount()):
            sub_item = self.multi_widget.layout().itemAtPosition(row_num, j)
            sub_widget = sub_item.widget()
            self.multi_widget.layout().removeWidget(sub_widget)
            sub_widget.deleteLater()

        self.multi_widget.layout().setRowMinimumHeight(row_num, 0)
        self.multi_widget.setMinimumSize(self.multi_widget.sizeHint())
        self.setMinimumSize(self.sizeHint())
        self.update()

    def add_element(self, default_value=None):
        """Add another element entry using the default element json provided by
            the json configuration.  If default_value is not None, the value
            provided will be set as the element's value."""

        row_index = self.multi_widget.layout().rowCount()
        new_element = self.multi_widget.registrar.eval(self.file_def['type'],
            self.file_def)
        new_element.updateLinks(self.root)
        minus_button = self.MinusButton(row_index - 1, self)
        new_element.elements.insert(1, minus_button)

        # Only open the file dialog if a default value has not been provided by
        # the user.
        if default_value == None:
            try:
                # Open the file selection dialog.
                new_element.button.getFileName()
                add_element = bool(new_element.value())  # False if len(value) == 0
            except AttributeError:
                # Thrown if the element is not a FileEntry.  In this case, add the
                # element.
                add_element = True
        else:
            new_element.setValue(default_value)
            add_element = True

        self.LOGGER.debug('Adding element id:"%s default:%s, add_element=%s"',
            self.attributes['id'], default_value, add_element)
        if add_element:
            for subElement, col_index in zip(new_element.elements,\
                range(len(new_element.elements))):
                if subElement.sizeHint().isValid():
                    subElement.setMinimumSize(subElement.sizeHint())
                self.multi_widget.layout().addWidget(subElement, row_index - 1,
                    col_index)
            self.multi_widget.elements.append(new_element)
            self.multi_widget.layout().addWidget(self.create_element_link,
                row_index, 2)

            self.multi_widget.setMinimumSize(self.multi_widget.sizeHint())
            self.multi_widget.layout().setRowMinimumHeight(row_index - 1, 20)
            self.setMinimumSize(self.sizeHint())
            self.multi_widget.setMinimumSize(self.multi_widget.sizeHint())
            self.multi_widget.update()
            self.update()

    def value(self):
        """Return a python list of the values for all enclosed elements."""
        return [r.value() for r in self.multi_widget.elements]

    def setValue(self, values):
        """Set the local input values to values.  values should be a python
            list of values to be set.  A new element will be created for each
            item in values.   Returns nothing."""

        self.clear_elements()
        for value in values:
            self.add_element(value)

    def clear_elements(self):
        """Remove ALL elements in this multi-element."""
        # Need to make a deep copy of the elements so we don't have some
        # conflict with self.remove_element.
        elements = self.multi_widget.elements[:]
        for element in elements:
            self.remove_element(element.elements[1].row_num)

    def resetValue(self):
        """Reimplemented from the Container class.  Removes all rows in the
        multiElement and then creates new rows for each JSON-defined default
        value, if provided."""

        # Clear out all the elements.
        self.clear_elements()

        # Now that we've removed all the existing elements, add an element for
        # each default value the user provided.
        if 'defaultValue' in self.attributes:
            for user_value in self.attributes['defaultValue']:
                self.add_element(user_value)

class GridList(DynamicGroup):
    """Class GridList represents a DynamicGroup that has a QGridLayout as a
        layout manager."""

    def __init__(self, attributes, registrar=None):
        """Constructor for the GridList class.

            attributes -a python dictionary containing all attributes of this
                DynamicGroup, including the elements it contains.  Elements are
                initialized in DynamicGroup.createElements().

            returns an instance of the GridList class."""

        super(GridList, self).__init__(attributes, QtGui.QGridLayout(), registrar)

class FileEntry(DynamicText):
    class FileField(DynamicText.TextField):
        def __init__(self):
            DynamicText.TextField.__init__(self)
            self.setAcceptDrops(True)

        def dragEnterEvent(self, event=None):
            """Overriding the default dragEnterEvent function for when a file is
            dragged and dropped onto this qlineedit.  This reimplementation is
            necessary for the dropEvent function to work on Windows."""
            event.accept()

        def dropEvent(self, event=None):
            """Overriding the default Qt DropEvent function when a file is
            dragged and dropped onto this qlineedit."""
            path = event.mimeData().urls()[0].path()
            if platform.system() == 'Windows':
                path = path[1:]  # Remove the '/' ahead of disk letter
            self.setText(path)
            event.acceptProposedAction()

    """This object represents a file.  It has three components, all of which
        are subclasses of QtGui.QWidget: a label (QtGui.QLabel), a textfield
        for the URI (QtGui.QLineEdit), and a button to engage the file dialog
        (a custom FileButton object ... Qt doesn't have a 'FileWidget' to
        do this for us, hence the custom implementation).

        Note that the FileEntry object is also used for folder elements.  The
        only differentiation between the two is that actual file elements have
        attributes['type'] == 'file', whereas folder elements have
        attributes['type'] == 'folder'.  This type influences the type of dialog
        presented to the user when the 'open' button is clicked in the UI and it
        affects default validation for the element."""

    def __init__(self, attributes):
        """initialize the object"""
        # Set default validation based on whether this element is for a file or
        # a folder.

        if attributes['type'] == 'folder':
            validate_type = 'folder'
            default_permissions = 'r'
        else:  # type is assumed to be a file
            validate_type = 'exists'
            default_permissions = 'r'

        if 'validateAs' not in attributes:
            attributes['validateAs'] = {"type": validate_type}

        if 'permissions' not in attributes['validateAs']:
            attributes['validateAs']['permissions'] = default_permissions

        self.textField = self.FileField()

        super(FileEntry, self).__init__(attributes)

        try:
            filter_type = str(attributes['validateAs']['type'])
        except KeyError:
            filter_type = 'all'

        file_type = 'folder'
        if issubclass(self.__class__, FileEntry) and filter_type != 'folder':
            file_type = 'file'

        self.button = FileButton(attributes['label'], self.textField,
            file_type, filter_type)
        self.addElement(self.button)

        # Holy cow, this is hacky.  I'm trying to override the mousePressEvent
        # event callback, so instead of creating a new subclass of QLineEdit
        # (which takes some work back in DynamicText class), I'm just going to
        # tell Qt that the mousePressEvent function should call
        # self.click_in_field.  Is this kosher?
        self.textField.mousePressEvent = self.click_in_field

    def click_in_field(self, event=None):
        """Reimplemented event handler taking the place of
            QLineEdit.mousePressEvent.  Checks to see if there's a file in the
            textfield.  If so, we do nothing.  Otherwise (if the uri is blank),
            we should open up the file dialog for the user.  If the user
            right-clicks in the text field, pass the event on to the normal
            handler."""

        if len(self.textField.text()) == 0 and\
            event.button() == QtCore.Qt.LeftButton:
            self.button.getFileName()
        else:
            QtGui.QLineEdit.mousePressEvent(self.textField, event)

    def setValue(self, text):
        """Set the value of the uri field.  If parameter 'text' is an absolute
            path, set the textfield to its value.  If parameter 'text' is a
            relative path, set the textfield to be the absolute path of the
            input text, relative to the invest root.

            returns nothing."""

        # Expand a '~' in the parameter text if it is present.  Otherwise, this
        # returns the path as it was passed in.
        text = os.path.expanduser(text.encode(ENCODING))

        if os.path.isabs(text):
            self.textField.setText(text)
        else:
            # If the path was saved as blank, we should set the current text
            # field to be blank.  Otherwise, the path should be considered to be
            # relative to the InVEST root.
            if len(text) == 0:
                self.textField.setText('')
            else:
                self.textField.setText(os.path.abspath(INVEST_ROOT + text))
        self.toggle()

class YearEntry(DynamicText):
    """This represents all the components of a 'Year' line in the LULC box.
        The YearEntry object consists of two objects: a label (QtGui.QLabel)
        and a year (QtGui.QLineEdit).  The year also has a min and max width."""

    def __init__(self, attributes):
        super(YearEntry, self).__init__(attributes)
        self.addElement(QtGui.QWidget())  # adjust spacing to align info button

        #set the width attribute, if it's provided.
        if 'width' in self.attributes:
            self.textField.setMaximumWidth(self.attributes['width'])

class FileButton(QtGui.QPushButton):
    """This object is the button used in the FileEntry object that, when
        pressed, will open a file dialog (QtGui.QFileDialog).  The string URI
        returned by the QFileDialog will be set as the text of the provided
        URIField.

        Arguments:
        text - the string text title of the popup window.
        URIField - a QtGui.QLineEdit.  This object will receive the string URI
            from the QFileDialog."""

    def __init__(self, text, URIfield, filetype='file', filter='all'):
        super(FileButton, self).__init__()
        self.text = text
        self.setIcon(QtGui.QIcon(os.path.join(IUI_DIR, 'document-open.png')))
        self.URIfield = URIfield
        self.filetype = filetype
        self.filters = {"all": ["All files (* *.*)"],
                        "EXISTS": ["All files (* *.*)"],
                        "CSV": ["Comma separated value file (*.csv *.CSV)"],
                        "GDAL": ["[GDAL] Arc/Info Binary Grid (hdr.adf HDR.ADF hdr.ADF)",
                                 "[GDAL] Arc/Info ASCII Grid (*.asc *.ASC)",
                                 "[GDAL] GeoTiff (*.tif *.tiff *.TIF *.TIFF)"],
                        "OGR": ["[OGR] ESRI Shapefiles (*.shp *.SHP)"],
                        "DBF": ["[DBF] dBase legacy file (*dbf *.DBF)"]
                       }
        self.last_filter = QString()

        filters = self.filters['all']
        if filetype == 'file':
            if filter != 'all':
                try:
                    filters += self.filters[filter.upper()]
                except:
                    print 'Could not find filters for %s' % filter.upper()

        self.filter_string = QString(';;'.join(filters))

        #connect the button (self) with the filename function.
        self.clicked.connect(self.getFileName)

    def getFileName(self, filetype='file'):
        """Get the URI from the QFileDialog.

            If the user presses OK in the QFileDialog, the dialog returns the
            URI to the selected file.

            If the user presses 'Cancel' in the QFileDialog, the dialog returns
            ''.  As a result, we must first save the previous contents of the
            QLineEdit before we open the dialog.  Then, if the user presses
            'Cancel', we can restore the previous field contents."""

        try:
            oldText = unicode(self.URIfield.text(), 'utf-8')
        except TypeError:
            # thrown when we're decoding a unicode to unicode
            oldText = self.URIfield.text()
        filename = ''

        if len(oldText) == 0:
            # if there is no text in the file entry, get the last folder
            # visited.
            if len(DATA['last_dir']) > 0:
                default_folder = DATA['last_dir']
            else:
                default_folder = os.path.expanduser('~')
        elif os.path.isdir(oldText):
            # if there is currently some text in the file entry, get its folder
            default_folder = oldText
        else:
            default_folder = os.path.dirname(oldText)

        if self.filetype == 'folder':
            filename = QtGui.QFileDialog.getExistingDirectory(self, 'Select ' +
                self.text, default_folder)
            filter = self.last_filter
        else:
            file_dialog = QtGui.QFileDialog()
            filename, filter = file_dialog.getOpenFileNameAndFilter(self,
                'Select ' + self.text, default_folder,
                filter=self.filter_string, initialFilter = self.last_filter)
        self.last_filter = filter

        #Set the value of the URIfield.
        if filename == '':
            self.URIfield.setText(oldText)
        else:
            self.URIfield.setText(filename)
            try:
                filename = unicode(filename, 'utf-8')
            except TypeError:
                # when we're trying to decode a unicode to a unicode object,
                # just use the original object.
                pass
            if os.path.isdir(filename):
                DATA['last_dir'] = filename
            else:
                DATA['last_dir'] = os.path.dirname(filename)

class SliderSpinBox(DynamicPrimitive):
    """expects these attributes:
        label: string
        type: sliderSpinBox
        min: number
        max: number
        sliderSteps: int
        spinboxSteps: int
        """
    def __init__(self, attributes):
        super(SliderSpinBox, self).__init__(attributes)

        self.label = QtGui.QLabel(self.attributes['label'])
        self.slider = QtGui.QSlider()
        self.spinbox = QtGui.QDoubleSpinBox()
        self.elements = [self.label, self.slider, self.spinbox]

        slidermax = self.attributes['max'] * self.attributes['sliderSteps']
        slidermin = self.attributes['min'] * self.attributes['sliderSteps']
        self.slider.setMaximum(slidermax)
        self.slider.setMinimum(slidermin)
        self.slider.setOrientation(QtCore.Qt.Horizontal)

        steps = float(self.attributes['max']) / self.attributes['spinboxSteps']
        self.spinbox.setMinimum(self.attributes['min'])
        self.spinbox.setMaximum(self.attributes['max'])
        self.spinbox.setSingleStep(steps)

        self.slider.valueChanged.connect(self.setSpinbox)
        self.spinbox.valueChanged.connect(self.setSlider)

    def setSpinbox(self):
        sliderValue = int(self.slider.value())
        self.spinbox.setValue(float(sliderValue) / self.attributes['sliderSteps'])

    def setSlider(self):
        fieldValue = self.spinbox.value()
        self.slider.setValue(int(fieldValue * self.attributes['sliderSteps']))


class HideableElement(LabeledElement):
    def __init__(self, attributes):
        LabeledElement.__init__(self, attributes)
        self.checkbox = QtGui.QCheckBox(attributes['label'])
        self.checkbox.toggled.connect(self.toggleHiding)
        self.checkbox.toggled.connect(self.toggle)
        self.hideableElements = []

        #remove the label, as it is being subsumed by the new checkbox's label.
        self.elements.remove(self.label)
        self.elements.insert(0, self.checkbox)

        self.toggleHiding(False)

    def toggleHiding(self, checked):
        for element in self.hideableElements:
            if checked:
                element.show()
            else:
                element.hide()

    def requirementsMet(self):
        return self.checkbox.isChecked()

    def get_element_state(self):
        """Get the state of the HideableElement.  Since this is just a base
        class, we can only really return a dictionary with the checked value."""

        return {'checked': self.checkbox.isChecked()}

class HideableFileEntry(HideableElement, FileEntry):
    def __init__(self, attributes):
        HideableElement.__init__(self, attributes)
        FileEntry.__init__(self, attributes)
        self.elements = [self.error_button, self.checkbox, self.textField,
                         self.button, self.info_button]
        self.hideableElements = [self.textField, self.button]
        self.toggleHiding(False)

    def requirementsMet(self):
        if self.checkbox.isChecked():
            return FileEntry.requirementsMet(self)
        return False

    def isEnabled(self):
        """IsEnabled is a characteristic of QtGui.QWidget.  We need to override
            it here because the whether the element is enabled depends not just
            on whether this element is greyed out but whether its value should
            be retrieved (and the value should only be retrieved when the
            checkbox is checked)."""
        if not self.checkbox.isEnabled():  # If this element is actually disabled, False
            return False

        # If the user can interact with the element, return this element's check
        # state.
        return self.checkbox.isChecked()

    def get_element_state(self):
        hideable_state = HideableElement.get_element_state(self)
        hideable_state['value'] = self.value()

        return hideable_state

    def set_element_state(self, state):
        self.checkbox.setChecked(state['checked'])
        self.setValue(state['value'])

class Dropdown(LabeledElement):
    def __init__(self, attributes):
        LabeledElement.__init__(self, attributes)

        self.dropdown = QtGui.QComboBox()

        if 'width' in self.attributes:
            self.dropdown.setMaximumWidth(self.attributes['width'])

        for option in self.attributes['options']:
            self.dropdown.addItem(option)

        self.addElement(self.dropdown)
        self.addElement(QtGui.QWidget())

    def setValue(self, index):
        if isinstance(index, str) or isinstance(index, unicode):
            index = self.dropdown.findText(index)
            if index == -1: #returned if the index cannot be found
                index = 0

        self.dropdown.setCurrentIndex(index)

    def value(self):
        if self.dropdown.count == 0:
            #if there are no elements in the dropdown, don't return a value
            return None
        elif 'returns' in self.attributes:
            if self.attributes['returns'] == 'strings':
                return str(self.dropdown.currentText())
            elif 'mapValues' in self.attributes['returns']:
                text = str(self.dropdown.currentText())
                try:
                    #mapped_value = self.attributes['returns']['mapValues'][text]
                    #print(mapped_value, mapped_value.__class__.__name__)
                    return self.attributes['returns']['mapValues'][text]
                except KeyError:
                    return text
            else: #return the ordinal
                return self.dropdown.currentIndex()
        else:
            return str(self.dropdown.currentText())

    def get_element_state(self):
        # return the state of the dropdown, specifically the index.
        return self.dropdown.currentIndex()

class CheckBox(QtGui.QCheckBox, DynamicPrimitive):
    """This class represents a checkbox for our UI interpreter.  It has the
        ability to enable and disable other elements."""

    def __init__(self, attributes):
        """Constructor for the CheckBox class.

            attributes - a python dictionary containing all attributes of this
                checkbox as defined by the user in the json configuration file.

            returns an instance of CheckBox"""

#        super(CheckBox, self).__init__(attributes)
        # Automatically assume that the checkbox value should be cast to a
        # boolean if the user has not specified differently.
        if 'dataType' not in attributes:
            attributes['dataType'] = 'boolean'

        QtGui.QCheckBox.__init__(self)
        DynamicPrimitive.__init__(self, attributes)

        try:
            self.elements.remove(self.info_button)
        except ValueError:
            print "warning attempted to remove helpbutton that wasn't defined"
        self.elements.append(QtGui.QWidget())
        self.elements.append(QtGui.QWidget())
        self.elements.append(self.info_button)

        #set the text of the checkbox
        self.setText(attributes['label'])

        #connect the button to the toggle function.
        self.toggled.connect(self.toggle)

    def toggle(self, event=None):
        """Enable/disable all elements controlled by this element.

            returns nothing."""

        self.setState(self.value(), includeSelf=False)

#    def isEnabled(self):
#        """Check to see if this element is checked.
#
#            returns a boolean"""
#
#        return self.isChecked()

    def value(self):
        """Get the value of this checkbox.

            returns a boolean."""

        check_state = self.isChecked()
        if 'returns' in self.attributes:
            value_map = self.attributes['returns']['mapValues']
            try:
                return value_map[str(check_state)]
            except KeyError:
                return check_state
        else:
            return check_state

    def setValue(self, value):
        """Set the value of this element to value.

            value - a string or boolean representing

            returns nothing"""

        if isinstance(value, unicode) or isinstance(value, str):
            if value == 'True':
                value = True
            else:
                value = False

        self.setChecked(value)
        self.setState(value, includeSelf=False)

    def requirementsMet(self):
        return self.isChecked()

    def setBGcolorSatisfied(self, state):
        pass

class TableHandler(Dropdown):
    """This class defines a general-purpose class for handling dropdown-based
        column selection.  This class uses IUI's 'enabledBy' attribute to
        control the contents of the dropdown menu. This element's 'enabledBy'
        attribute must be set to the id of 0a file element that is validated
        appropriately."""
    def __init__(self, attributes):
        # initialize the options key if it doesn't already exist.
        if 'options' not in attributes:
            attributes['options'] = []

        Dropdown.__init__(self, attributes)
        self.loaded_state = None
        self.handler = None  # this should be set in an appropriate subclass.
        self.uri = ''

    def populate_fields(self):
        """Extract the fieldnames from the fileio table handler class for this
        instance of TableHandler.  Returns nothing, but populates the dropdown
        with the appropriate fieldnames.  If any options are present in the
        dropdown, they are cleared before the new column names are entered."""
        self.dropdown.clear()
        self.handler.update(self.enabledBy.value())
        field_names = self.handler.get_fieldnames(case='orig')
        for name in field_names:
            self.dropdown.addItem(name)

        # see if the current state of the dropdown matches the saved state
        if self.loaded_state != None:
            # check that the URI matches
            uri_matches = self.loaded_state['linked_uri'] == self.handler.uri

            # check that the index and key at that index matches
            key_at_index = str(self.dropdown.itemText(self.loaded_state['index']))
            key_matches = key_at_index == self.loaded_state['key']

            if uri_matches and key_matches:
                self.dropdown.setCurrentIndex(self.loaded_state['index'])

    def get_element_state(self):
        current_text = self.dropdown.currentText()
        try:
            current_text = unicode(current_text, 'utf-8')
        except TypeError:
            # when casting a unicode obj. to unicode, just use original.
            pass
        return {'key': current_text,
                'index': self.dropdown.currentIndex(),
                'linked_uri': self.handler.uri}

    def set_element_state(self, state):
        self.loaded_state = state
        self.dropdown.setCurrentIndex(state['index'])

    def setState(self, state, includeSelf=True, recursive=True):
        """Reimplemented from Dropdown.setState.  When state=False, the dropdown
        menu is cleared.  If state=True, the dropdown menu is populated with
        values from the corresponding table object."""

        # If an error exists in the enabledBy element, disable self.
        error = self.enabledBy.error_button.error_text
        if error != None and error != '':
            state = False

        Dropdown.setState(self, state, includeSelf, recursive)

        if state == False:
            self.dropdown.clear()
        else:
            self.populate_fields()

class CSVFieldDropdown(TableHandler):
    def __init__(self, attributes):
        TableHandler.__init__(self, attributes)
        self.handler = fileio.CSVHandler(self.uri)

class OGRFieldDropdown(TableHandler):
    def __init__(self, attributes):
        TableHandler.__init__(self, attributes)
        self.handler = fileio.OGRHandler(self.uri)

class OperationDialog(QtGui.QDialog):
    """ModelDialog is a class defining a modal window presented to the user
        while the model is running.  This modal window prevents the user from
        interacting with the main UI window while the model is processing and
        provides status updates for the model.

        This window is not configurable through the JSON configuration file."""

    def __init__(self, root):
        """Constructor for the ModelDialog class.

            root - a pointer to the parent window

            returns an instance of ModelDialog."""
        QtGui.QDialog.__init__(self)
#        Testator.__init__(self)

        self.root = root
        self.exec_controller = executor.Controller()

        #set window attributes
        self.setLayout(QtGui.QVBoxLayout())
        self.setWindowTitle("Running the model")
        self.resize(700, 400)
        center_window(self)
        self.setModal(True)

        self.cancel = False

        #create statusArea-related widgets for the window.
        self.statusAreaLabel = QtGui.QLabel('Messages:')
        self.statusArea = QtGui.QPlainTextEdit()
        self.statusArea.setReadOnly(True)
        self.cursor = self.statusArea.textCursor()

        #set the background color of the statusArea widget to be white.
        self.statusArea.setStyleSheet("QWidget { background-color: White }")

        #create an indeterminate progress bar.  According to the Qt
        #documentation, an indeterminate progress bar is created when a
        #QProgressBar's minimum and maximum are both set to 0.
        self.progressBar = QtGui.QProgressBar()
        self.progressBar.setMinimum(0)
        self.progressBar.setMaximum(0)
        self.progressBar.setTextVisible(False)

        self.messageArea = MessageArea()
        self.messageArea.clear()

        #Add the new widgets to the window
        self.layout().addWidget(self.statusAreaLabel)
        self.layout().addWidget(self.statusArea)
        self.layout().addWidget(self.messageArea)
        self.layout().addWidget(self.progressBar)


        #create Quit and Cancel buttons for the window
#        self.quitButton = QtGui.QPushButton(' Quit')
#        self.quitButton.setToolTip('Quit the application')
        self.backButton = QtGui.QPushButton(' Back')
        self.backButton.setToolTip('Return to parameter list')
#        self.cancelButton = QtGui.QPushButton(' Cancel')

        #add button icons
#        self.quitButton.setIcon(QtGui.QIcon(os.path.join(IUI_DIR,
#            'dialog-close.png')))
        self.backButton.setIcon(QtGui.QIcon(os.path.join(IUI_DIR,
            'dialog-ok.png')))
#        self.cancelButton.setIcon(QtGui.QIcon('dialog-cancel.png'))

        #disable the 'Back' button by default
        self.backButton.setDisabled(True)
#        self.quitButton.setDisabled(True)
#        self.cancelButton.setDisabled(False)

        #create the buttonBox (a container for buttons) and add the buttons to
        #the buttonBox.
        self.buttonBox = QtGui.QDialogButtonBox()
#        self.buttonBox.addButton(self.quitButton, QtGui.QDialogButtonBox.RejectRole)
        self.buttonBox.addButton(self.backButton, QtGui.QDialogButtonBox.AcceptRole)
#        self.buttonBox.addButton(self.cancelButton, QtGui.QDialogButtonBox.ResetRole)

        #connect the buttons to their callback functions.
        self.backButton.clicked.connect(self.closeWindow)
#        self.quitButton.clicked.connect(sys.exit)
#        self.cancelButton.clicked.connect(self.exec_controller.cancel_executor)

        #add the buttonBox to the window.
        self.layout().addWidget(self.buttonBox)

        self.timer = QtCore.QTimer()

    def showEvent(self, event):
        if self.exec_controller.is_finished() or not self.timer.isActive():
            QtCore.QTimer.singleShot(100, self.startExecutor)

    def startExecutor(self):
        self.statusArea.clear()
        self.start_buttons()

        self.exec_controller.start_executor()

        self.timer.timeout.connect(self.check_messages)
        self.timer.start(100)
        self.write('Initializing...\n')

    def check_messages(self):
        if not self.exec_controller.is_finished():
            message = self.exec_controller.get_message()
            if message != None:
                self.write(message)
            time.sleep(.1)
        else:
            self.finished()

    def start_buttons(self):
        self.progressBar.setMaximum(0) #start the progressbar.
        self.backButton.setDisabled(True)
#        self.quitButton.setDisabled(True)
#        self.cancelButton.setDisabled(False)

    def stop_buttons(self):
        self.progressBar.setMaximum(1) #stops the progressbar.
        self.backButton.setDisabled(False)
#        self.quitButton.setDisabled(False)
#        self.cancelButton.setDisabled(True)

    def write(self, text):
        """Write text.  If printing to the status area, also scrolls to the end
            of the text region after writing to it.  Otherwise, print to stdout.

            text - a string to be written to self.statusArea.

            returns nothing."""

        self.statusArea.insertPlainText(QString(text))
        self.cursor.movePosition(QtGui.QTextCursor.End)
        self.statusArea.setTextCursor(self.cursor)

    def finished(self):
        """Notify the user that model processing has finished.

            returns nothing."""

        self.timer.stop()
        self.stop_buttons()
        errors_found = self.exec_controller.thread_failed
        if errors_found:
            thread_exception = self.exec_controller.thread_exception
            self.messageArea.setText(str('<b>%s</b> encountered: <em>%s</em> <br/>' +
                'See the log for details.') % (thread_exception.__class__.__name__,
                str(thread_exception)))
        else:
            self.messageArea.setText('Model completed successfully.')
        self.messageArea.setError(errors_found)
        self.messageArea.show()
        self.cursor.movePosition(QtGui.QTextCursor.End)
        self.statusArea.setTextCursor(self.cursor)

    def closeEvent(self, event=None):
        """When a closeEvent is detected, run self.closeWindow().

            returns nothing."""

        event.ignore()

    def okPressed(self):
        """When self.runButton is pressed, halt the statusbar and close the
            window with a siccessful status code.

            returns nothing."""

        self.threadFinished()
        self.accept() # this is a built-in Qt signal.

    def closeWindow(self):
        """Close the window and ensure the modelProcess has completed.

            returns nothing."""

        self.messageArea.clear()
        self.cancel = False
        self.done(0)

    def cancelled(self):
        return self.cancel

class ElementAssembler(iui_validator.ValidationAssembler):
    def __init__(self, elements_ptr):
        iui_validator.ValidationAssembler.__init__(self)
        self.elements = elements_ptr

    def _get_value(self, element_id):
        """Takes a string element_id, returns the element's value, either strin
        g or int or boolean."""
        if element_id in self.elements:
            value = self.elements[element_id].value()
        else:
            value = element_id

        return value

class ScrollArea(QtGui.QScrollArea):
    def __init__(self, attributes, layout=QtGui.QVBoxLayout(), registrar=None):
        QtGui.QScrollArea.__init__(self)
        print self.width()

        self.body = DynamicGroup(attributes, layout, registrar)
        self.setWidget(self.body)
        self.updateScrollBorder()

    def updateScrollBorder(self, min=None, max=None):
        if min == None:
            min = self.verticalScrollBar().minimum()
        if max == None:
            max = self.verticalScrollBar().maximum()

        if min == 0 and max == 0:
            self.setStyleSheet("QScrollArea { border: None } ")
        else:
            self.setStyleSheet("")

    def getElementsDictionary(self):
        for id, element in self.body.getElementsDictionary().iteritems():
            print(id, element)
        return self.body.getElementsDictionary()

class MessageArea(QtGui.QLabel):
    def __init__(self):
        QtGui.QLabel.__init__(self)
        self.setWordWrap(True)
        self.setTextFormat(QtCore.Qt.RichText)
        self.messages = []

    def clear(self):
        """Clear all text and set the stylesheet to none."""

        self.hide()
        self.setText('')
        self.setStyleSheet('')

    def setText(self, text=None):
        if text == None:
            text = []
        else:
            text = [text + '<br/>']
        messages = text + self.messages
        string = "<br/>".join(messages)
        QtGui.QLabel.setText(self, string)

    def append(self, string):
        self.messages.append(string)
        self.setText()

    def setError(self, state):
        """Set the background color according to the error status passed in.

            state - a python boolean.  False if no error.  True if error.

            returns nothing."""

        if not state:
            self.setStyleSheet('QLabel { padding: 15px;' +
                'background-color: #d4efcc; border: 2px solid #3e895b;}')
        else:
            self.setStyleSheet('QLabel { padding: 15px;' +
                'background-color: #ebabb6; border: 2px solid #a23332;}')

class TabbedGroup(QtGui.QTabWidget, DynamicGroup):
    def __init__(self, attributes, registrar=None):
        super(TabbedGroup, self).__init__(attributes, QtGui.QVBoxLayout(), registrar)

        for index, tab in enumerate(self.elements):
            tab.set_index(index)
            self.addTab(tab, tab.attributes['label'])

        self.setMinimumSize(self.sizeHint())

    def set_active_tab(self):
        """Update the active tab in this tabbedgroup to be the leftmost enabled
            tab.  If there are no enabled tabs, the active tab should be the
            leftmost tab.

            returns nothing."""

        num_enabled = 0
        rightmost_tab_index = 0
        for tab_num in range(self.count() - 1, -1, -1):
            if self.isTabEnabled(tab_num):
                self.setCurrentIndex(tab_num)
                num_enabled += 1
        self.LOGGER.debug('%s tabs enabled', num_enabled)

        if num_enabled == 0 and self.count() > 0:
            self.setCurrentIndex(0)
            self.LOGGER.debug('Active tab=%s (leftmost)', 0)


class Tab(DynamicGroup):
    def __init__(self, attributes, registrar=None):
        layout = QtGui.QVBoxLayout()
        super(Tab, self).__init__(attributes, layout, registrar)
        self.index = None


        # default condenseElements to True if not otherwise specified.
        if 'condenseElements' not in self.attributes:
            self.attributes['condenseElements'] = True

        self.LOGGER.debug('Condense elements=%s',
            self.attributes['condenseElements'])
        if self.attributes['condenseElements'] == True:
            layout.insertStretch(-1)


    def set_index(self, index):
        self.index = index

    def setState(self, state, includeSelf=True, recursive=True):
        #disabling the tab itself automatically disables all its options.  No
        #need to recurse through all contained elements.
        #The parent of a tab is a QStackedWidget.  The parent of this
        #QStackedWidget is the TabbedGroup
        self.parent().parent().setTabEnabled(self.index, state)
        self.parent().parent().set_active_tab()

    def isEnabled(self):
        return self.parent().parent().isTabEnabled(self.index)


class Root(DynamicElement):
    def __init__(self, uri, layout, object_registrar):
        self.config_loader = fileio.JSONHandler(uri)
        attributes = self.config_loader.get_attributes()

        if not hasattr(self, 'super'):
            print(self, 'setting super to None')
            self.super = None

        self.obj_registrar = object_registrar

        self.find_and_replace(attributes)

        DynamicElement.__init__(self, attributes)
        self.type_registrar = registrar.DatatypeRegistrar()
        self.setLayout(layout)

        self.body = DynamicGroup(attributes, QtGui.QVBoxLayout(), object_registrar)

        if 'scrollable' in self.attributes:
            make_scrollable = self.attributes['scrollable']
        else:
            make_scrollable = True

        if make_scrollable:
            self.scrollArea = QtGui.QScrollArea()
            self.layout().addWidget(self.scrollArea)
            self.scrollArea.setWidget(self.body)
            self.scrollArea.setWidgetResizable(True)
            self.scrollArea.verticalScrollBar().rangeChanged.connect(self.updateScrollBorder)

            self.updateScrollBorder(self.scrollArea.verticalScrollBar().minimum(),
                                    self.scrollArea.verticalScrollBar().maximum())
        else:
            self.layout().addWidget(self.body)

        self.outputDict = {}
        self.allElements = self.body.getElementsDictionary()

        self.updateLinksLater = []

        for id, element in self.allElements.iteritems():
            try:
                element.updateLinks(self)
            except:
                # If an exception is thrown when trying to update links, assume
                # that it's an issue of not really knowing where the element is
                # located and try to address it once all other elements have
                # updated their links.
                self.updateLinksLater.append(element)

        for element in self.updateLinksLater:
            element.updateLinks(self)

        self.operationDialog = OperationDialog(self)
        self.assembler = ElementAssembler(self.allElements)

        self.embedded_uis = []

        if 'include_meta' not in self.attributes:
            self.attributes['include_meta'] = False

    def find_and_replace(self, attributes):
        """Initiates a recursive search and replace of the attributes
            dictionary according to the 'inheritFrom' capabilities of the JSON
            definition.

            attributes - a python dictionary representation of the JSON
                         configuration.

        Returns the rendered attributes dictionary."""

        self.attributes = attributes
        return self.find_inherited_elements(attributes)

    def find_value(self, inherit, current_dict=None):
        """Searches the given dictionary for values described in inherit.

            inherit - a python dictionary confirming to the attribute
                      inheritance properties of the JSON definition.
            current_dict - a python dictionary of the current scope of the
                           search.  if None (the default value), self.attributes
                           is used for the current_dict.

            Returns the value object requested by inherit if found.  Returns
            None if the requested object is not found."""

        if current_dict == None:
            current_dict = self.attributes

        if inherit['inheritFrom'] == current_dict['id']:
            return current_dict[inherit['useAttribute']]
        else:
            if 'elements' in current_dict:
                for element in current_dict['elements']:
                    value = self.find_value(inherit, element)
                    if value != None:
                        return value

    def find_inherited_elements(self, attributes):
        """Searches the input attributes dictionary for an inheritance object
            and initializes a search for the value requested by the inheritance
            object.

            attributes - a python dictionary representing an element.

            Returns the rendered attributes dictionary."""

        for key, value in attributes.iteritems():
            if isinstance(value, dict):
                if 'inheritFrom' in value:
                    if 'useAttribute' not in value:
                        value['useAttribute'] = key

                    if 'fromOtherUI' in value:
                        if str(value['fromOtherUI']) == 'super':
                            root_ptr = self.obj_registrar.root_ui
                            root_attrib = root_ptr.attributes
                            fetched_value = root_ptr.find_value(value, root_attrib)
                        else:
                            fetched_value = self.find_embedded_value(value)
                    else:
                        fetched_value = self.find_value(value)

                    attributes[key] = fetched_value
            elif key in ['elements', 'rows']: #guaranteed array of objects
                for element in value:
                    value = self.find_inherited_elements(element)
            elif key in ['args_id']: #list of strings or inheritance objects
                for index, element in enumerate(value):
                    if isinstance(element, dict):
                        value[index] = self.find_value(element)
        return attributes

    def find_embedded_value(self, inherit):
        #locate the configuration URI
        altered_inherit = {'inheritFrom': inherit['fromOtherUI'],
                           'useAttribute': 'configURI'}
        embedded_uri = self.find_value(altered_inherit)

        json_handler = fileio.JSONHandler(embedded_uri)

        #locate the value we want
        del inherit['fromOtherUI']
        return self.find_value(inherit, json_handler.get_attributes())

    def updateScrollBorder(self, min, max):
        if min == 0 and max == 0:
            self.scrollArea.setStyleSheet("QScrollArea { border: None } ")
        else:
            self.scrollArea.setStyleSheet("")

    def resetParametersToDefaults(self):
        """Reset all parameters to defaults provided in the configuration file.

            returns nothing"""

        for id, element in self.allElements.iteritems():
            if issubclass(element.__class__, DynamicPrimitive):
                element.resetValue()
            elif issubclass(element.__class__, Container):
                element.resetValue()
            elif issubclass(element.__class__, Root):
                element.resetParametersToDefaults()

    def errors_exist(self):
        """Check to see if any elements in this UI have errors.

           returns a list of tuples, where the first tuple entry is the element
           label and the second tuple entry is the element's error message.."""

        errors = []
        for id, element in self.allElements.iteritems():
            if issubclass(element.__class__, DynamicPrimitive):
                if element.isEnabled() and element.has_error():
                    try:
                        error_msg = element.error_button.error_text
                        errors.append((element.attributes['label'], error_msg))
                    except:
                        pass
            elif issubclass(element.__class__, EmbeddedUI):
                embedded_errors = element.errors_exist()
                if len(embedded_errors) > 0:
                    for error_tuple in embedded_errors:
                        errors.append(error_tuple)
        return errors

    def warnings_exist(self):
        """Check to see if any elements in this UI have warnings.

           returns a list of tuples, where the first tuple entry is the element
           label and the second tuple entry is the element's error message.."""

        warnings = []
        for id, element in self.allElements.iteritems():
            if issubclass(element.__class__, DynamicPrimitive):
                if element.has_warning() and element.isEnabled():
                    error_msg = element.error_button.error_text
                    warnings.append(element.attributes['label'])
        return warnings

    def queueOperations(self):
        #placeholder for custom implementations.
        #intended for the user to call executor.addOperation() as necessary
        #for the given model.
        return

    def assembleOutputDict(self):
        """Assemble an output dictionary for use in the target model

            Saves a python dictionary to self.outputDict.  This dictionary has
            the mapping: element args_id -> element value.  Values are converted
            to their appropriate dataType where specified in the JSON config
            file.

            returns a python dictionary"""

        #initialize the outputDict, in case it has been already written to
        #in a previous run.
        outputDict = {}

        for id, element in self.allElements.iteritems():
            always_return = False
            if 'returns' in element.attributes:
                try:
                    if 'alwaysReturn' in element.attributes['returns']:
                        always_return = element.attributes['returns']['alwaysReturn']
                except TypeError:
                    # Thrown when attributes['returns'] is not a dictionary.
                    # Assume that always_return should be False.
                    pass

            if element.isEnabled() or always_return:
                if 'args_id' in element.attributes:
                    element_value = element.getOutputValue()
                    if element_value != None:

                        args_id = element.attributes['args_id']
                        if not isinstance(args_id, list):
                            args_id = [args_id]

                        outputDict = self.set_dict_value(outputDict, args_id,
                            element_value)

        if self.attributes['include_meta'] is True:
            outputDict['_iui_meta'] = {
                'ui_state': self.get_element_state(),
                'lastrun_uri': self.last_run_handler.uri,
            }

        return outputDict

    def set_dict_value(self, dictionary, key_list, element_value):
        key, list = (key_list[0], key_list[1:])

        if len(list) > 0:
            if key not in dictionary:
                temp_dict = {}
            else:
                temp_dict = dictionary[key]

            dictionary[key] = self.set_dict_value(temp_dict, list, element_value)
        else:
            dictionary[key] = element_value
        return dictionary

    def find_embedded_elements(self):
        uis = []
        for id, element in self.allElements.iteritems():
            if issubclass(element.__class__, EmbeddedUI):
                uis.append(element)
        return uis

    def value(self):
        user_args = {}

        if self.isEnabled():
            #loop through all elements known to the UI, assemble into a dictionary
            #with the mapping element ID -> element value
            for id, element in self.allElements.iteritems():
                try:
                    value = element.value()
                    if value != None:
                        user_args[id] = value
                except:
                    pass
        return user_args

    def get_element_state(self):
        user_args = {}

        if self.isEnabled():
            #loop through all elements known to the UI, assemble into a dictionary
            #with the mapping element ID -> element value
            for id, element in self.allElements.iteritems():
                try:
                    value = element.get_element_state()
                    if value != None:
                        user_args[id] = value
                except:
                    pass
        return user_args

class EmbeddedUI(Root):
    def __init__(self, attributes, registrar):
        uri = attributes['configURI']
        layout = QtGui.QVBoxLayout()
        self.super = registrar.root_ui
        print('super_ui', self.super)
        Root.__init__(self, uri, layout, registrar)

        #removing the reference to self in self.allElements.  If a reference to
        #self is in self.allElements and self has an args_id, the args_id is
        #replicated at two levels: the embeddedUI level and in the super ui,
        #even though it should only be used in the superUI level.  This is a
        #bandaid fix.
        if self.attributes['id'] in self.allElements:
            del self.allElements[self.attributes['id']]
        self.body.layout().insertStretch(-1)

        self.attributes['args_id'] = attributes['args_id']

    def getOutputValue(self):
        return self.assembleOutputDict()

    def updateLinks(self, rootPointer):
        for element_id, element_ptr in self.allElements.iteritems():
            element_ptr.updateLinks(self)

    def find_element_ptr(self, element_id):
        """Return an element pointer if found.  None if not found."""
        #if the element id can be found in the current UI, return that
        #otherwise, get the element from this element's root.
        try:
            return self.allElements[element_id]
        except KeyError:
            return None

    def setValue(self, parameters):
        for element_id, value in parameters.iteritems():
            try:
                self.allElements[element_id].setValue(value)
            except KeyError:
                print 'Could not find %s in EmbUI %s' % (element_id,
                    self.attributes['id'])
            except Exception as e:
                print 'Error \'%s\' encountered setting %s to %s' %\
                    (e, element_id, value)
                print traceback.print_exc()

class MainWindow(QtGui.QMainWindow):
    def __init__(self, root_class, uri):
        QtGui.QMainWindow.__init__(self)
        self.ui = root_class(uri, self)
        self.setCentralWidget(self.ui)
        self.ui.setWindowSize()

        self.file_menu = QtGui.QMenu('&File')
        self.load_file_action = self.file_menu.addAction('&Load parameters from file ...')
        self.load_file_action.setShortcut(QtGui.QKeySequence("Ctrl+O"))
        self.save_file_action = self.file_menu.addAction('&Save parameters ...')
        self.save_file_action.setShortcut(QtGui.QKeySequence("Ctrl+S"))
        self.remove_lastrun = self.file_menu.addAction('&Clear cached runs ...')
        self.exit_action = self.file_menu.addAction('Exit')
        self.exit_action.setShortcut(QtGui.QKeySequence("Ctrl+Q"))
        self.menuBar().addMenu(self.file_menu)

        self.dev_menu = QtGui.QMenu('&Development')
        self.save_to_python = self.dev_menu.addAction('Save to &python script...')
        self.save_to_json = self.dev_menu.addAction('Save to archivable &JSON...')
        self.save_args_to_json = self.dev_menu.addAction('Save args dict to file...')
        self.menuBar().addMenu(self.dev_menu)

        self.exit_action.triggered.connect(self.ui.closeWindow)
        self.save_file_action.triggered.connect(self.ui.save_parameters_to_file)
        self.load_file_action.triggered.connect(self.ui.load_parameters_from_file)
        self.remove_lastrun.triggered.connect(self.ui.remove_lastrun)
        self.save_to_python.triggered.connect(self.ui.save_to_python)
        self.save_to_json.triggered.connect(self.ui.save_to_json)
        self.save_args_to_json.triggered.connect(self.ui.save_args_dict_to_file)

    def exec_(self):
        self.show()

class ExecRoot(Root):
    def __init__(self, uri, layout=None, object_registrar=None,
            main_window=None, version=None):
        if main_window == None:
            self.main_window = self
        else:
            self.main_window = main_window # a pointer

        if layout == None:
            layout = QtGui.QVBoxLayout()

        if object_registrar == None:
            object_registrar = ElementRegistrar(self)

        self.messageArea = MessageArea()
        self.messageArea.setError(False)
        Root.__init__(self, uri, layout, object_registrar)

        self.main_window.setWindowIcon(QtGui.QIcon(
            os.path.join(IUI_DIR, 'natcap_logo.png')))

        # Check to see if we should load the last run.  Defaults to false if the
        # user has not specified.
        try:
            use_lastrun = self.attributes['loadLastRun']
        except KeyError:
            use_lastrun = True
        self.lastRun = {}
        self.last_run_handler = fileio.LastRunHandler(self.attributes['modelName'],
            version)
        if use_lastrun:
            self.lastRun = self.last_run_handler.get_attributes()

        self.layout().addWidget(self.messageArea)
        self.addBottomButtons()
        self.setWindowSize()
        self.error_dialog = ErrorDialog()
        self.warning_dialog = WarningDialog()
        self.initElements()


    def save_to_json(self):
        """Save the current state of the UI to a python file after checking that
        there are no validation errors."""
        errors = self.errors_exist()
        if len(errors) > 0:
            self.error_dialog.set_messages(errors)
            self.error_dialog.exec_()
        else:
            warnings = self.warnings_exist()

            if len(warnings) > 0:
                self.warning_dialog.set_messages(warnings)
                exit_code = self.warning_dialog.exec_()

                # If the user pressed 'back' on the warning dialog, return to
                # the UI.
                if exit_code == 0:
                    return

            model = self.attributes['targetScript']
            model_name = model.split('.')[-1]

            filename = QtGui.QFileDialog.getSaveFileName(self, 'Select file to save...',
                '%s_archive.json' % model_name, filter = QString('JSON file' +
                ' (*.json);;All files (*.* *)'))
            try:
                filename = unicode(filename)
            except TypeError:
                # can't cast unicode to unicode
                pass

            if filename != '':
                arguments = self.assembleOutputDict()
                natcap.invest.iui.fileio.save_model_run_json(arguments, model, filename)

    def save_to_python(self):
        """Save the current state of the UI to a python file after checking that
        there are no validation errors."""
        errors = self.errors_exist()
        if len(errors) > 0:
            self.error_dialog.set_messages(errors)
            self.error_dialog.exec_()
        else:
            warnings = self.warnings_exist()

            if len(warnings) > 0:
                self.warning_dialog.set_messages(warnings)
                exit_code = self.warning_dialog.exec_()

                # If the user pressed 'back' on the warning dialog, return to
                # the UI.
                if exit_code == 0:
                    return

            model = self.attributes['targetScript']
            model_name = model.split('.')[-1]

            filename = QtGui.QFileDialog.getSaveFileName(self, 'Select file to save...',
                '%s_parameters.py' % model_name, QString(),
                QString('Python file (*.py);;All files (*.* *)'))
            try:
                filename = unicode(filename)
            except TypeError:
                # can't cast unicode to unicode
                pass

            if filename != '':
                arguments = self.assembleOutputDict()
                natcap.invest.iui.fileio.save_model_run(arguments, model, filename)

    def find_element_ptr(self, element_id):
        """Return an element pointer if found.  None if not found."""
        #if the element id can be found in the current UI, return that
        #otherwise, get the element from this element's root.
        try:
            return self.allElements[element_id]
        except KeyError:
            if not hasattr(self, 'embedded_uis'):
                raise RuntimeError(('Tried to locate element %s, but it does '
                    'exist in any known UIs') % element_id)

            if self.embedded_uis == []:
                self.embedded_uis = self.find_embedded_elements()

            for embedded_ui in self.embedded_uis:
                emb_ptr = embedded_ui.find_element_ptr(element_id)
                if emb_ptr != None:
                    return emb_ptr

    def save_args_dict_to_file(self):
        model = self.attributes['targetScript']
        model_name = model.split('.')[-1]

        filename = QtGui.QFileDialog.getSaveFileName(self, 'Select file to save...',
            '%s_parameters.json' % model_name, filter = QString('InVEST Parameter file' +
            ' (*.json);;All files (*.* *)'))
        filename = str(filename)
        if filename != '':
            save_handler = fileio.JSONHandler(filename)
            save_handler.write_to_disk(self.assembleOutputDict())
            print 'parameters written to %s' % filename
            basename = os.path.basename(filename)
            self.messageArea.append('Parameters saved to %s' % basename)

    def save_parameters_to_file(self):
        model = self.attributes['targetScript']
        model_name = model.split('.')[-1]

        filename = QtGui.QFileDialog.getSaveFileName(self, 'Select file to save...',
            '%s_parameters.json' % model_name, QString(),
            QString('InVEST Parameter file (*.json);;All files (*.* *)'))
        filename = str(filename)
        if filename != '':
            save_handler = fileio.JSONHandler(filename)
            save_handler.write_to_disk(self.get_element_state())
            print 'parameters written to %s' % filename
            basename = os.path.basename(filename)
            self.messageArea.append('Parameters saved to %s' % basename)

    def load_parameters_from_file(self):
        filename = QtGui.QFileDialog.getOpenFileName(self, 'Select file to load...',
            filter = QString('InVEST Parameter file' +
            ' (*.json);;All files (*.* *)'))
        filename = str(filename)
        if filename != '':
            load_handler = fileio.JSONHandler(filename)
            attributes = load_handler.get_attributes()
            self.load_elements_from_save(attributes)
            basename = os.path.basename(filename)
            self.messageArea.append('Parameters loaded from %s' % basename)


    def saveLastRun(self):
        """Saves the current values of all input elements to a JSON object on
            disc.

            returns nothing"""

        self.last_run_handler.write_to_disk(self.get_element_state())

    def load_elements_from_save(self, save_dict):
        for id, value in save_dict.iteritems():
            try:
                element = self.allElements[str(id)]
                element.set_element_state(value)
            except Exception as e:
                print traceback.print_exc()
                self.LOGGER.warn(('Error \'%s\' when setting lastrun value '
                    '%s to %s'), e, value, str(id))

    def initElements(self):
        """Set the enabled/disabled state and text from the last run for all
            elements

            returns nothing"""

        if self.lastRun == {}:
            self.resetParametersToDefaults()
        else:
            self.load_elements_from_save(self.lastRun)

            if hasattr(self, 'messageArea'):
                self.messageArea.setText('Parameters have been loaded from the' +
                    ' most recent run of this model.  <a href=\'default\'>' +
                    ' Reset to defaults</a>')
                try:
                    self.messageArea.linkActivated.disconnect()
                except TypeError:
                    # Raised if we can't disconnect any signals
                    pass
                self.messageArea.linkActivated.connect(self.resetParametersToDefaults)

    def remove_lastrun(self):
        dialog = WarningDialog()
        dialog.setWindowTitle('Clear cached runs?')
        dialog.set_title('Clear cached runs?')
        dialog.body.setText('Are you sure you want to clear any automatically'
            ' cached runs?  This will not affect any of your saved parameter'
            ' sets.')
        dialog.ok_button.setText('Yes')
        exit_code = dialog.exec_()

        if exit_code != 0:
            self.last_run_handler.delete()
            self.lastRun = {}
            self.messageArea.setText('Cached runs have been cleared')

    def setWindowSize(self):
        #this groups all elements together at the top, leaving the
        #buttons at the bottom of the window.
        if 'width' in self.attributes:
            width = self.attributes['width']
        else:
            width = 700

        if 'height' in self.attributes:
            height = self.attributes['height']
        else:
            height = 400

        if 'label' in self.attributes:
            self.setWindowTitle(self.attributes['label'])

        # Check the height and width set in the UI.  Shrink them if they extend
        # beyond the available height and width of the window manager.
        screen_geometry = QtGui.QDesktopWidget().availableGeometry()
        screen_height = screen_geometry.height()
        screen_width = screen_geometry.width()
        if width > screen_width:
            width = screen_width - 50

        if height > screen_height:
            height = screen_height - 50

        self.LOGGER.debug('Width=%s, height=%s', width, height)
        self.main_window.resize(width, height)
        center_window(self.main_window)

    def resetParametersToDefaults(self):
        Root.resetParametersToDefaults(self)
        reset_text = 'Parameters reset to defaults.  '
        if self.lastRun != {}:
            reset_text += str('<a href=\'reset\'>Restore parameters from' +
                ' your last run</a>')
        self.messageArea.setText(reset_text)
        try:
            self.messageArea.linkActivated.disconnect()
        except TypeError:
            # Thrown when we can't disconnect any slots from this signal
            pass
        self.messageArea.linkActivated.connect(self.initElements)

    def okPressed(self):
        """A callback, run when the user presses the 'OK' button.

            returns nothing."""

        errors = self.errors_exist()
        if len(errors) == 0:
            warnings = self.warnings_exist()

            if len(warnings) > 0:
                self.warning_dialog.set_messages(warnings)
                exit_code = self.warning_dialog.exec_()

                # If the user pressed 'back' on the warning dialog, return to
                # the UI.
                if exit_code == 0:
                    return

            #Check if workspace has an output directory, prompt the user that
            #it will be overwritten
            try:
                uri = self.allElements['workspace'].textField.text()
                try:
                    uri = unicode(uri)
                except TypeError:
                    # can't cast unicode to unicode`
                    pass
                if os.path.isdir(os.path.join(uri,'output')) or \
                        os.path.isdir(os.path.join(uri,'Output')):
                    dialog = WarningDialog()
                    dialog.setWindowTitle('Output Exists')
                    dialog.set_title('Output Exists')
                    dialog.set_icon('dialog-information-2.png')
                    dialog.body.setText('The directory workspace/output ' +
                        'exists.  Are you sure you want overwrite output ' +
                        'from previous model run? %s' % str())

                    dialog.ok_button.setText('Run Model')
                    dialog.ok_button.setIcon(QtGui.QIcon(os.path.join(IUI_DIR,
                        'dialog-ok.png')))
                    exit_code = dialog.exec_()
                    # An exit code of 0 means go back.
                    if exit_code == 0:
                        return
                    # A non-0 exit code means go go go, so just fall through
            except KeyError:
                #A keyerror means that 'workspace' isn't in the ui elements
                #concievable since this is a general framework.  Since this is
                #a little hacky this serves as a mechansim to handle the
                #workspace #cases as well as a point of reference if we
                #ever try to refactor
                pass

            # Check to see if the user has specified whether we should save the
            # last run.  If the user has not specified, assume that the last run
            # should be saved.
            try:
                save_lastrun = self.attributes['saveLastRun']
            except KeyError:
                save_lastrun = True

            if save_lastrun:
                self.saveLastRun()

            self.queueOperations()
            self.runProgram()
        else:
            self.error_dialog.set_messages(errors)
            self.error_dialog.exec_()


    def runProgram(self):
        # Caling show() here allows the call to be non-blocking, whereas calling
        # operationDialog.exec_() is a blocking call.  I expect that this may
        # cause some problems along the way, but it works as I want it to for
        # testing at least.
        self.operationDialog.show()

        if self.operationDialog.cancelled():
            QtCore.QCoreApplication.instance().exit()

    def closeWindow(self):
        """Terminates the application.

            This function is called when the user closes the window by any of
            Qt's recognized methods (often including closing the window via the
            little x in the corner of the window and/or pressing the Esc key).

            returns nothing"""

        dialog = WarningDialog()
        dialog.setWindowTitle('Are you sure you want to quit?')
        dialog.set_title('Really quit?')
        dialog.set_icon('dialog-information-2.png')
        dialog.body.setText('You will lose any changes to your parameter fields.')
        dialog.ok_button.setText('Quit')

        exit_code = dialog.exec_()
        # An exit code of 0 means cancel.
        # If the user pressed OK, the program should quit
        if exit_code != 0:
            self.main_window.close()

    def closeEvent(self, event=None):
        """Terminates the application. This function is a Qt-defined callback
            for when the window is closed.

            returns nothing"""

        self.closeWindow()

    def addBottomButtons(self):
        """Assembles buttons and connects their callbacks.

            returns nothing."""

        self.runButton = QtGui.QPushButton(' Run')
        self.runButton.setIcon(QtGui.QIcon(os.path.join(IUI_DIR,
            'dialog-ok.png')))

        self.cancelButton = QtGui.QPushButton(' Quit')
        self.cancelButton.setIcon(QtGui.QIcon(os.path.join(IUI_DIR,
            'dialog-close.png')))

        self.resetButton = QtGui.QPushButton(' Reset')
        self.resetButton.setIcon(QtGui.QIcon(os.path.join(IUI_DIR,
            'edit-undo.png')))

        #create the buttonBox (a container for buttons)
        self.buttonBox = QtGui.QDialogButtonBox()
        self.buttonBox.addButton(self.runButton, QtGui.QDialogButtonBox.AcceptRole)
        self.buttonBox.addButton(self.cancelButton, QtGui.QDialogButtonBox.RejectRole)
        self.buttonBox.addButton(self.resetButton, QtGui.QDialogButtonBox.ResetRole)

        #connect the buttons to their functions.
        self.runButton.clicked.connect(self.okPressed)
        self.cancelButton.clicked.connect(self.closeWindow)
        self.resetButton.clicked.connect(self.resetParametersToDefaults)

        #add the buttonBox to the window.
        self.layout().addWidget(self.buttonBox)

class InfoDialog(QtGui.QDialog):
    def __init__(self):
        QtGui.QDialog.__init__(self)
        self.messages = []
        self.resize(400, 200)
        self.setWindowTitle('Errors exist!')
        self.setLayout(QtGui.QVBoxLayout())
        self.icon = QtGui.QLabel()
        self.icon.setStyleSheet('QLabel { padding: 10px }')
        self.set_icon('dialog-error.png')
        self.icon.setSizePolicy(QtGui.QSizePolicy.Fixed,
            QtGui.QSizePolicy.Fixed)
        self.title = QtGui.QLabel()
        self.set_title('Whoops!')
        self.title.setStyleSheet('QLabel { font: bold 18px }')
        self.body = QtGui.QLabel()
        self.body.setWordWrap(True)
        self.ok_button = QtGui.QPushButton('OK')
        self.ok_button.clicked.connect(self.accept)

        error_widget = QtGui.QWidget()
        error_widget.setLayout(QtGui.QHBoxLayout())
        error_widget.layout().addWidget(self.icon)
        self.layout().addWidget(error_widget)

        body_widget = QtGui.QWidget()
        error_widget.layout().addWidget(body_widget)
        body_widget.setLayout(QtGui.QVBoxLayout())
        body_widget.layout().addWidget(self.title)
        body_widget.layout().addWidget(self.body)

        self.button_box = QtGui.QDialogButtonBox()
        self.button_box.addButton(self.ok_button, QtGui.QDialogButtonBox.AcceptRole)
        self.layout().addWidget(self.button_box)

    def set_icon(self, uri):
        self.icon.setPixmap(QtGui.QPixmap(os.path.join(IUI_DIR, uri)))

    def set_title(self, title):
        self.title.setText(title)

    def set_messages(self, message_list):
        self.messages = message_list

class WarningDialog(InfoDialog):
    def __init__(self):
        InfoDialog.__init__(self)
        self.set_title('Warning...')
        self.set_icon('dialog-warning-big.png')
        self.body.setText('Some inputs cannot be validated and may cause ' +
           'this program to fail.  Continue anyways?')
        self.no_button = QtGui.QPushButton('Back')
        self.no_button.clicked.connect(self.reject)
        self.button_box.addButton(self.no_button, QtGui.QDialogButtonBox.RejectRole)

class ErrorDialog(InfoDialog):
    def __init__(self):
        InfoDialog.__init__(self)
        self.set_title('Whoops!')

    def showEvent(self, event=None):
        label_string = '<ul>'
        for element_tuple in self.messages:
            label_string += '<li>%s: %s</li>' % element_tuple
        label_string += '</ul>'

        num_messages = len(self.messages)
        if num_messages == 1:
            num_error_string = 'is 1 error'
        else:
            num_error_string = 'are %s errors' % num_messages

        self.body.setText(str("There %s that must be resolved" +
            " before this tool can be run:%s") % (num_error_string, label_string))
        self.body.setMinimumSize(self.body.sizeHint())

class ElementRegistrar(registrar.Registrar):
    def __init__(self, root_ptr):
        registrar.Registrar.__init__(self)
        self.root_ui = root_ptr
        updates = {'container' : Container,
                   'list': GridList,
                   'file': FileEntry,
                   'folder': FileEntry,
                   'text': YearEntry,
                   'sliderSpinBox': SliderSpinBox,
                   'hideableFileEntry': HideableFileEntry,
                   'dropdown': Dropdown,
                   'embeddedUI': EmbeddedUI,
                   'checkbox': CheckBox,
                   'scrollGroup': ScrollArea,
                   'OGRFieldDropdown': OGRFieldDropdown,
                   'hiddenElement': StaticReturn,
                   'multi': MultiElement,
                   'label': Label,
                   'tabbedGroup': TabbedGroup,
                   'tab': Tab
                   }
        self.update_map(updates)

    def eval(self, type, op_values):
        widget = registrar.Registrar.get_func(self, type)
        if (issubclass(widget, DynamicGroup) or issubclass(widget, EmbeddedUI)
            or issubclass(widget, ScrollArea)):
            return widget(op_values, registrar=self)
        else:
            return widget(op_values)


def center_window(window_ptr):
    """Center a window on whatever screen it appears.

            window_ptr - a pointer to a Qt window, whether an application or a
                QDialog.

        returns nothing."""
    geometry = window_ptr.frameGeometry()
    center = QtGui.QDesktopWidget().availableGeometry().center()
    geometry.moveCenter(center)
    window_ptr.move(geometry.topLeft())


if __name__ == "__main__":
    reg = Registrar()
    reg.create('checkbox', {'label': 'lala'})
