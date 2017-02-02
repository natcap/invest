
import logging
import threading
import os
import contextlib
from datetime import datetime

from qtpy import QtWidgets
import natcap.invest
from natcap.ui import inputs

LOG_FMT = "%(asctime)s %(name)-18s %(levelname)-8s %(message)s"
DATE_FMT = "%m/%d/%Y %H:%M:%S "
LOGGER = logging.getLogger(__name__)


class ThreadFilter(logging.Filter):
    """When used, this filters out log messages that were recorded from other
    threads.  This is especially useful if we have logging coming from several
    concurrent threads.
    Arguments passed to the constructor:
        thread_name - the name of the thread to identify.  If the record was
            reported from this thread name, it will be passed on.
    """
    def __init__(self, thread_name):
        logging.Filter.__init__(self)
        self.thread_name = thread_name

    def filter(self, record):
        if record.threadName == self.thread_name:
            return True
        return False


@contextlib.contextmanager
def log_to_file(logfile):
    if os.path.exists(logfile):
        LOGGER.warn('Logfile %s exists and will be overwritten', logfile)

    handler = logging.FileHandler(logfile, 'w', encoding='UTF-8')
    formatter = logging.Formatter(LOG_FMT, DATE_FMT)
    thread_filter = ThreadFilter(threading.current_thread().name)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.NOTSET)  # capture everything
    root_logger.addHandler(handler)
    handler.addFilter(thread_filter)
    handler.setFormatter(formatter)
    yield handler
    handler.close()
    root_logger.removeHandler(handler)


class Model(object):
    label = None
    target = None
    validator = None
    localdoc = None

    def __init__(self):
        self.window = QtWidgets.QWidget()
        self.window.setLayout(QtWidgets.QVBoxLayout())
        if self.label:
            self.window.setWindowTitle(self.label)
        self.links = QtWidgets.QLabel()
        self.window.layout().addWidget(self.links)
        self.form = inputs.Form()
        self.window.layout().addWidget(self.form)
        self.run_dialog = inputs.FileSystemRunDialog()

        # start with workspace and suffix inputs
        self.workspace = inputs.Folder(args_key='workspace_dir',
                                       label='workspace',
                                       required=True)
        self.suffix = inputs.Text(args_key='suffix',
                                  label='Results suffix',
                                  required=False)
        self.suffix.textfield.setMaximumWidth(150)
        self.add_input(self.workspace)
        self.add_input(self.suffix)

        self.form.submitted.connect(self.execute)

    def _make_links(self, qlabel):
        links = ['Version ' + natcap.invest.__version__]

        try:
            doc_uri = 'file://' + os.path.abspath(self.localdoc)
            links.append('<a href=\"%s\">Model documentation</a>' % doc_uri)
        except AttributeError:
            # When self.localdoc is None, documentation is undefined.
            LOGGER.info('Skipping docs link; undefined.')

        feedback_uri = 'http://forums.naturalcapitalproject.org/'
        links.append('<a href=\"%s\">Report an issue</a>' % feedback_uri)

        qlabel.setText(' | '.join(links))

    def add_input(self, input):
        # Add the model's validator if it hasn't already been set.
        if hasattr(input, 'validator') and input.validator is None:
            input.validator = self.validator
        self.form.add_input(input)

    def execute(self, logfile=None, tempdir=None):
        args = self.assemble_args()

        if not os.path.exists(args['workspace_dir']):
            os.makedirs(args['workspace_dir'])

        if not logfile:
            logfile = os.path.join(
                args['workspace_dir'],
                'InVEST-{modelname}-log-{timestamp}.txt'.format(
                    modelname='-'.join(self.label.split(' ')),
                    timestamp=datetime.now().strftime("%Y-%m-%d--%H_%M_%S")))

        if not tempdir:
            tempdir = os.path.join(args['workspace_dir'], 'tmp')
            os.makedirs(tempdir)

        self.form.run(target=self.target,
                      kwargs={'args': args},
                      logfile=logfile,
                      tempdir=tempdir,
                      window_title='Running %s' % self.label,
                      out_folder=args['workspace_dir'])

    def assemble_args(self):
        raise NotImplementedError

    def run(self):
        self.window.show()
        self.window.raise_()  # raise window to top of stack.
        inputs.QT_APP.exec_()
