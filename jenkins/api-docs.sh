#!/bin/bash

ENV=doc_env
paver env \
    --system-site-packages \  # Grant the new env access to the system python
    --with-invest \           # Install natcap.invest to the new repo
    --envdir=$ENV             # Create the env at this dir.
    -r requirements-docs.txt  # Install sphinx into the env.
source $ENV/bin/activate
paver build_docs --skip-guide
