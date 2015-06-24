import unittest
import natcap.invest.iui.modelui_test

loader = unittest.TestLoader()
suite = loader.loadTestsFromModule(natcap.invest.iui.modelui_test)
unittest.TextTestRunner(verbosity=2).run(suite)

