import textwrap

import tomli


with open('pyproject.toml', 'rb') as f:
  pyproject_toml = tomli.load(f)
with open('requires-build.yml', 'w') as f:
  f.write(textwrap.dedent("""
  channels:
  - conda-forge
  - nodefaults
  dependencies:
  """))
  for requirement in pyproject_toml['build-system']['requires']:
    f.write(f'  - {req}')
