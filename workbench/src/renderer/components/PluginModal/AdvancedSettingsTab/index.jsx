import React, { useEffect, useState } from 'react';
import PropTypes from 'prop-types';

import Button from 'react-bootstrap/Button';
import Form from 'react-bootstrap/Form';
import { useTranslation } from 'react-i18next';
import { MdFolderOpen } from 'react-icons/md';

import { ipcMainChannels } from '../../../../main/ipcMainChannels';

const { ipcRenderer } = window.Workbench.electron;

export default function AdvancedSettingsTab(props) {
  const {
    plugins,
    dragOverHandler,
    dragEnterHandler,
    dragLeavingHandler,
    selectFile,
    selectDirectory,
    getDroppedFilePath,
  } = props;
  const [condaPath, setCondaPath] = useState('');
  const [pluginEnvs, setPluginEnvs] = useState({});

  useEffect(() => {
    Promise.all([
      ipcRenderer.invoke(ipcMainChannels.GET_SETTING, 'micromamba'),
      ipcRenderer.invoke(ipcMainChannels.GET_SETTING, 'userDefinedMicromamba')
    ]).then(([micromamba, userDefinedMicromamba]) => {
      setCondaPath(userDefinedMicromamba || micromamba);
    });
    ipcRenderer.invoke(
      ipcMainChannels.GET_SETTING, 'plugins'
    ).then((data) => setPluginEnvs(
      Object.fromEntries(
        Object.keys(data).map(
          (pluginID) => [pluginID, data[pluginID].userDefinedEnv || data[pluginID].env]
        )
      )
    ))
  }, []);

  const resetCondaPath = () => {
    ipcRenderer.invoke(
      ipcMainChannels.GET_SETTING, 'micromamba'
    ).then((data) => {
      setCondaPath(data);
    });
  };

  const saveCondaPath = () => {
    ipcRenderer.send(
      ipcMainChannels.SET_SETTING, 'userDefinedMicromamba', condaPath
    );
  };

  const resetPluginEnv = (pluginID) => {
    ipcRenderer.invoke(
      ipcMainChannels.GET_SETTING, `plugins.${pluginID}.env`
    ).then((value) => {
      setPluginEnvs({...pluginEnvs, [pluginID]: value});
    });
  };

  const savePluginEnvs = () => {
    Object.entries(pluginEnvs).forEach(([pluginID, envPath]) => {
      ipcRenderer.send(
        ipcMainChannels.SET_SETTING, `plugins.${pluginID}.userDefinedEnv`, envPath
      );
    });
  };

  const { t } = useTranslation();

  return (
    <>
      <Form aria-labelledby="configure-conda-form-title" aria-describedby="conda-executable-description">
        <Form.Group>
          <h5 id="configure-conda-form-title" className="mb-3">{t('Configure conda executable (Advanced)')}</h5>
          <Form.Text
            as="span"
            id="conda-executable-description"
            className="plugin-form-text mb-3"
          >
            {t('InVEST is distributed with a copy of micromamba, a conda-like '
              + 'package manager that is used to manage plugin environments. '
              + 'If you have conda or mamba installed elsewhere on the system, '
              + 'you can configure InVEST to use that executable instead. This '
              + 'may be useful if you run into limitations of the included '
              + 'micromamba distribution. You can enter an absolute path, or '
              + 'the name of an executable that is on the system PATH.')}
          </Form.Text>
          <Form.Label htmlFor="condaPath">{t('Conda or mamba executable')}</Form.Label>
          <div className="d-flex flex-nowrap w-100">
            <Form.Control
              id="condaPath"
              type="text"
              value={condaPath || ''}
              onChange={(event) => setCondaPath(event.target.value)}
              onDragOver={dragOverHandler}
              onDragEnter={dragEnterHandler}
              onDragLeave={dragLeavingHandler}
              onDrop={(event) => {
                const droppedPath = getDroppedFilePath(event);
                if (droppedPath) {
                  setCondaPath(droppedPath);
                }
              }}
              className="me-1"
            />
            <Button
              aria-label="browse for conda executable"
              className="browse-button ms-1 me-1"
              variant="outline-dark"
              onClick={async (event) => setCondaPath(await selectFile(event) || condaPath)}
            >
              <MdFolderOpen />
            </Button>
            <Button
              className="text-nowrap ms-1"
              onClick={resetCondaPath}
            >
              {t('Reset')}
            </Button>
          </div>
          <Button onClick={saveCondaPath} className="text-nowrap mt-3">
            {t('Save')}
          </Button>
        </Form.Group>
      </Form>
      <hr />
      <Form aria-labelledby="configure-plugin-envs-form-title" aria-describedby="plugin-env-description">
        <Form.Group>
        <h5 id="configure-plugin-envs-form-title" className="mb-3">{t('Configure plugin environments (Advanced)')}</h5>
        <Form.Text
            as="span"
            id="plugin-env-description"
            className="plugin-form-text mb-3"
          >
            {t('InVEST creates a separate conda environment for each installed '
              + 'plugin. You may override this and provide a path to a different '
              + 'conda environment, which may be useful for development and '
              + 'debugging.')}
          </Form.Text>
        {Object.keys(plugins).map((pluginID) => (
          <Form.Group key={`${pluginID}-env-group`}>
            <Form.Label htmlFor={pluginID}>
              {pluginID}
            </Form.Label>
            <div
              className="d-flex flex-nowrap w-100 mb-1"
            >
              <Form.Control
                id={pluginID}
                type="text"
                value={pluginEnvs[pluginID]}
                onChange={(event) => setPluginEnvs(
                  {...pluginEnvs, [pluginID]: event.target.value}
                )}
                onDragOver={dragOverHandler}
                onDragEnter={dragEnterHandler}
                onDragLeave={dragLeavingHandler}
                onDrop={(event) => {
                  const droppedPath = getDroppedFilePath(event);
                  if (droppedPath) {
                    setPluginEnvs({
                      ...pluginEnvs,
                      [pluginID]: droppedPath,
                    });
                  }
                }}
                className="me-1"
              />
              <Button
                aria-label="browse for env"
                className="browse-button ms-1 me-2"
                variant="outline-dark"
                onClick={async (event) => setPluginEnvs({
                  ...pluginEnvs,
                  [pluginID]: await selectDirectory(event) || pluginEnvs[pluginID]
                })}
              >
                <MdFolderOpen />
              </Button>
              <Button
                onClick={() => resetPluginEnv(pluginID)}
                className="text-nowrap"
              >
                {t('Reset')}
              </Button>
            </div>
          </Form.Group>
        ))}
        {Object.keys(pluginEnvs).length
          ? <Button
              onClick={savePluginEnvs}
              className="text-nowrap mt-3">
                {t('Save')}
            </Button>
          : <p>{t('No plugins to configure.')}</p>
        }
      </Form.Group>
      </Form>
    </>
  );
}