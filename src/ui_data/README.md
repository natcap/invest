# Explanation of UI spec JSON format

For each model, there must be a UI spec file in this directory with the name `natcap.invest.<model name>.json`.
The JSON file has two required top-level keys:
* `"order"`: A 2D list of arg keys that describes the order in which input fields should appear in the UI.
    Each nested list represents a section. How sections are rendered is determined by the `ArgsForm` component; 
    currently there is some extra margin around each section to visually separate them.

    Only input fields that are listed in the `"order"` section will be rendered. To permanently hide an input,
    you can just leave it out of this list. Each arg key listed here must have a value in the model's `ARGS_SPEC`.

* `"argsOptions"`: A mapping from arg keys to configuration options for displaying the corresponding input fields.
    Each arg key is optional; if no configuration options apply, you can leave it out of this mapping.
    Currently two configuration options are recognized:

        * `"control_targets"`: A list of arg keys whose rendering is affected by the state of this arg key.
            Each arg key in this list must also exist in `"argsOptions"` and have a `"control_option"` property.
            
        * `"control_option"`: A string that determines the CSS style to apply to this arg's input field, conditional
            on the state of the controlling field. If `"control_option": "x"`, then the style `arg-x` is applied.
            Currently `"disable"`, `"hide"`, and `"group"` are available.
