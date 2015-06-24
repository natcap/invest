import codecs
import os
import shutil
import imp

import natcap.invest.testing
import pygeoprocessing.geoprocessing


def file_has_class(test_file_uri, test_class_name):
    """Check that a python test file contains a class.

        test_file_uri - a URI to a python file containing test classes.
        test_class_name - a string, the class name we're looking for.

        Returns True if the class is found, False otherwise."""

    test_file = codecs.open(test_file_uri, mode='r', encoding='utf-8')
    try:
        module = imp.load_source('model', test_file_uri)
        cls_attr = getattr(module, test_class_name)
        return True
    except AttributeError:
        # Everything imported properly, but we didn't find the test class and
        # test function we wanted.
        return False
    except ImportError:
        # We couldn't import everything necessary (such as with
        # invest_test_core), so we need to loop line by line to check and see if
        # the class has the test required.
        for line in test_file:
            if line.startswith('class %s(' % test_class_name):
                test_file.close()
                return True
        return False

def class_has_test(test_file_uri, test_class_name, test_func_name):
    """Check that a python test file contains the given class and function.

        test_file_uri - a URI to a python file containing test classes.
        test_class_name - a string, the class name we're looking for.
        test_func_name - a string, the test function name we're looking for.
            This function should be located within the target test class.

        Returns True if the function is found within the class, False otherwise."""
    test_file = codecs.open(test_file_uri, mode='r', encoding='utf-8')
    try:
        module = imp.load_source('model', test_file_uri)
        cls_attr = getattr(module, test_class_name)
        func_attr = getattr(cls_attr, test_func_name)
        return True
    except AttributeError:
        # Everything imported properly, but we didn't find the test class and
        # test function we wanted.
        return False
    except ImportError:
        # We couldn't import everything necessary (such as with
        # invest_test_core), so we need to loop line by line to check and see if
        # the class has the test required.
        in_class = False
        for line in test_file:
            if line.startswith('class %s(' % test_class_name):
                in_class = True
            elif in_class:
                if line.startswith('class '):
                    # We went through the whole class and didn't find the
                    # function.
                    return False
                elif line.lstrip().startswith('def %s(self):' % test_func_name):
                    # We found the function within this class!
                    return True
        return False


def add_test_to_class(file_uri, test_class_name, test_func_name,
        in_archive_uri, out_archive_uri, module):
    """Add a test function to an existing test file.  The test added is a
    regression test using the natcap.invest.testing.regression archive
    decorator.

        file_uri - URI to the test file to modify.
        test_class_name - string. The test class name to modify.  If the test class
            already exists, the test function will be added to the test class.
            If not, the new class will be created.
        test_func_name - string.  The name of the test function to write.  If a
            test function by this name already exists in the target class, the
            function will not be written.
        in_archive_uri - URI to the input archive.
        out_archive_uri - URI to the output archive.
        module - string module, whose execute function will be run in the test
            (e.g. 'natcap.invest.pollination.pollination')

    WARNING: The input test file is overwritten with the new test file.

    Returns nothing."""

    test_file = codecs.open(file_uri, 'r', encoding='utf-8')

    temp_file_uri = pygeoprocessing.geoprocessing.temporary_filename()
    new_file = codecs.open(temp_file_uri, 'w+', encoding='utf-8')

    cls_exists = file_has_class(file_uri, test_class_name)
    test_exists = class_has_test(file_uri, test_class_name, test_func_name)

    if test_exists:
        print ('WARNING: %s.%s exists.  Not writing a new test.' %
            (test_class_name, test_func_name))
        return

    def _import():
        return 'import natcap.invest.testing\n'

    def _test_class(test_class):
        return 'class %s(natcap.invest.testing.GISTest):\n' % test_class

    def _archive_reg_test(test_name, module, in_archive, out_archive, cur_dir):
        in_archive = os.path.relpath(in_archive, cur_dir)
        out_archive = os.path.relpath(out_archive, cur_dir)
        return('    @natcap.invest.testing.regression(\n' +\
               '        input_archive="%s",\n' % in_archive +\
               '        workspace_archive="%s")\n' % out_archive +\
               '    def %s(self):\n' % test_name +\
               '        %s.execute(self.args)\n' % module +\
               '\n')

    if cls_exists == False:
        for line in test_file:
            new_file.write(line.rstrip() + '\n')

        new_file.write('\n')
        new_file.write(_import())
        new_file.write(_test_class(test_class_name))
        new_file.write(_archive_reg_test(test_func_name, module,
            in_archive_uri, out_archive_uri, os.path.dirname(file_uri)))
    else:
        import_written = False
        for line in test_file:
            if (not(line.startswith('import') or line.startswith('from')) and not
                import_written):
                new_file.write(_import())
                import_written = True

            new_file.write(line.rstrip() + '\n')
            if 'class %s(' % test_class_name in line:
                new_file.write(_archive_reg_test(test_func_name, module,
                    in_archive_uri, out_archive_uri, os.path.dirname(file_uri)))

    test_file = None
    new_file = None

    # delete the old file
    os.remove(file_uri)
    print 'removed %s' % file_uri

    # copy the new file over the old one.
    shutil.copyfile(temp_file_uri, file_uri)
    print 'copying %s to %s' % (temp_file_uri, file_uri)

