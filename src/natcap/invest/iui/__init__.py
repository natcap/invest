import logging

# Set up logging for the modelUI
# I haven't been able to figure out why, but for some reason I have to add a new
# StreamHandler to the LOGGER object for information to be printed to stdout.  I
# can't figure out why this is necessary here and not necessary in our other
# models, where calling `logging.basicConfig` is sufficient.

# Format string and the date format are shared by the basic configuration as
# well as the added streamhandler.
format_string = '%(asctime)s %(name)-30s %(funcName)-20s %(levelname)-8s %(message)s'
date_format = '%m/%d/%Y %H:%M:%S '

# Do the basic configuration of logging here.  This is required in addition to
# adding the streamHandler below.
logging.basicConfig(format=format_string, level=logging.DEBUG,
        datefmt=date_format)

# Create a formatter and streamhandler to format and print messages to stdout.
formatter = logging.Formatter(format_string, date_format)
handler = logging.StreamHandler()
handler.setFormatter(formatter)

def get_ui_logger(name):
    #Get the logging object for this level and add the handler we just created.
    logger = logging.getLogger(name)
#    logger.addHandler(handler)
    return logger

