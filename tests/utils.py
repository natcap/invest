import os


def assert_complete_execute(raw_args, model_spec, **kwargs):
    """Assert that post-processing functions completed.

    This assertion can be used after calling ``model_spec.execute`` with
    various options to assert that expected files exist.

    Args:
        raw_args (dict): the args dict passed to ``execute``
        model_spec (natcap.invest.spec.ModelSpec): the model's specification
        kwargs (dict): kwargs that can be passed to ``execute``.

    Raises:
        AssertionError if expected files do not exist.
    """
    args = model_spec.preprocess_inputs(raw_args)
    if 'save_file_registry' in kwargs and kwargs['save_file_registry']:
        if not os.path.exists(
            os.path.join(args['workspace_dir'],
                         f'file_registry{args["results_suffix"]}.json')):
            raise AssertionError('file registry json file does not exist')
    if 'generate_report' in kwargs and kwargs['generate_report']:
        if not os.path.exists(
            os.path.join(args['workspace_dir'],
                         f'{model_spec.model_id}_report{args["results_suffix"]}.html')):
            raise AssertionError('model report html file does not exist')
