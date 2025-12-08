import os

def assert_complete_execute(workspace_dir, raw_args, model_spec, **kwargs):
    args = model_spec.preprocess_inputs(raw_args)
    if 'save_file_registry' in kwargs and kwargs['save_file_registry']:
        if not os.path.exists(
            os.path.join(workspace_dir,
                         f'file_registry{args["results_suffix"]}.json')):
                raise AssertionError('file registry json file does not exist')
    if model_spec.reporter:
        if not os.path.exists(
            os.path.join(workspace_dir,
                         f'{model_spec.model_id}_report{args["results_suffix"]}.html')):
                raise AssertionError('model report html file does not exist')

