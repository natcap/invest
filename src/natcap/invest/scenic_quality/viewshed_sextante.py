def viewshed(input_uri, output_uri, coordinates, obs_elev=1.75, tgt_elev=0.0,
             max_dist=-1, refraction_coeff=0.14286, memory=500, stream_dir=None,
             consider_curvature=False, consider_refraction=False, boolean_mode=False,
             elevation_mode=False, verbose=False, quiet=False):
    """
    http://grass.osgeo.org/grass70/manuals/r.viewshed.html
    """

    args_string = ''
    for pred, flag in [(consider_curvature,'-c'), (consider_refraction, '-r'),
                       (boolean_mode, '-b'), (elevation_mode, '-e'),
                       (verbose,'--verbose'), (quiet, '--quiet')]:
        if pred:
            args_string += flag + ' '

    
