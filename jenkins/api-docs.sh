#!/bin/bash

ENV=doc_env
paver env --clear \           # clear out an existing env if it already exists
    --system-site-packages \  # Grant the new env access to the system python
    --with-invest \           # Install natcap.invest to the new repo
    --envdir=$ENV             # Create the env at this dir.
    -r requirements-docs.txt  # Install sphinx into the env.
source $ENV/bin/activate
paver build_docs --skip-guide
