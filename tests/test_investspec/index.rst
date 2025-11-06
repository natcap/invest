custom sphinx role demo
***********************


Here are examples of all the types:
-----------------------------------

- :investspec:`test_module number_input`
- :investspec:`test_module ratio_input`
- :investspec:`test_module percent_input`
- :investspec:`test_module integer_input`
- :investspec:`test_module boolean_input`
- :investspec:`test_module freestyle_string_input`
- :investspec:`test_module option_string_input`
- :investspec:`test_module raster_input`
- :investspec:`test_module another_raster_input`
- :investspec:`test_module vector_input`
- :investspec:`test_module csv_input`
- :investspec:`test_module directory_input`


You can access any attribute or nested arg by a period-separated series of keys. Here is a nested CSV in the directory:
-----------------------------------------------------------------------------------------------------------------------

:investspec:`test_module directory_input.contents.baz`


Here is a raster column in the CSV:
-----------------------------------

:investspec:`test_module directory_input.contents.baz.columns.raster_path`
