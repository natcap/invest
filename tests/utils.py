import os


def assert_complete_execute(workspace_dir, raw_args, model_spec, **kwargs):
    """Assert that post-processing functions completed.

    This assertion can be used after calling ``model_spec.execute`` with
    various options to assert that expected files exist.

    Args:
        workspace_dir (str): path to invest model output workspace
        raw_args (dict): the args dict passed to ``execute``
        model_spec (natcap.invest.spec.ModelSpec)
        kwargs (dict): kwargs that can be passed to ``execute``.

    Raises:
        AssertionError if expected files do not exist.
    """
    args = model_spec.preprocess_inputs(raw_args)
    if 'save_file_registry' in kwargs and kwargs['save_file_registry']:
        if not os.path.exists(
            os.path.join(workspace_dir,
                         f'file_registry{args["results_suffix"]}.json')):
            raise AssertionError('file registry json file does not exist')
    if 'generate_report' in kwargs and kwargs['generate_report']:
        if not os.path.exists(
            os.path.join(workspace_dir,
                         f'{model_spec.model_id}_report{args["results_suffix"]}.html')):
            raise AssertionError('model report html file does not exist')
