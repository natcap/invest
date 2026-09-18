import React from 'react';

import Accordion from 'react-bootstrap/Accordion';
import { useTranslation } from 'react-i18next';

import { openLinkInBrowser } from '../../../utils';

export default function AboutTab() {
  const { t } = useTranslation();

  const pluginRegistryURL = "https://natcap.github.io/invest-plugin-registry/";
  const pluginSubmissionURL = "https://natcap.github.io/invest-plugin-registry/docs/submission_process.html";
  const registryGitHubURL = "https://github.com/natcap/invest-plugin-registry/";
  const securityReportURL = "https://github.com/natcap/invest-plugin-registry/security";
  const pluginDocsURL = "https://invest.readthedocs.io/en/latest/plugins.html";

  return (
    <>
      <h5>{t(`About Plugins:`)}</h5>
      <Accordion>
        <Accordion.Item eventKey="0">
          <Accordion.Header>{t(`What Is a Plugin?`)}</Accordion.Header>
          <Accordion.Body>
            <p className="plugin-small-text">
            {t(`Conceptually, an InVEST plugin is an ecosystem services model or performs `
                + `functions related to ecosystem services modeling. A plugin could be a new `
                + `stand-alone model, a variation on a model that already exists in core InVEST, `
                + `or a workflow that composes multiple models to solve a domain-specific problem. `
                + `A plugin could also perform pre-processing steps to prepare data for use in `
                + `modeling, or post-processing steps on the outputs from another model.`)}
            </p>
            <p className="plugin-small-text">
              {t(`Like the core InVEST models, it takes in data of various formats (usually `
                  + `including some geospatial data), processes that data, and produces output files `
                  + `that contain the results. Unlike the core models, a plugin is not 'official', `
                  + `i.e., not reviewed or maintained by NatCap. Plugins may be developed, used, and `
                  + `distributed totally independently of the natcap/invest repo and the Natural `
                  + `Capital Alliance.`)}
            </p>
          </Accordion.Body>
        </Accordion.Item>
        <Accordion.Item eventKey="1">
          <Accordion.Header>{t(`Installing Plugins`)}</Accordion.Header>
          <Accordion.Body>
            <p className="plugin-small-text">
              {t(`You can use the Plugin Registry tab to browse for plugins included on the InVEST `
                + `Plugin Registry, a listing of community-contributed plugins. If you find one you `
                + `would like to install, you can do so directly from the Workbench using the "Install" `
                + `button.`)}
            </p>
            <p className="plugin-small-text">
              {t(` Installation will take a few minutes. Once it completes, the "Install" form will `
                + `be replaced by a message saying "This plugin is installed! You can then close the `
                + `modal and launch the plugin from the list of models.`)}
            </p>
            <p className="plugin-small-text">
              {t(`If you want to install a plugin that isn't listed on the Registry, and either have the `
                + `link to a git repo where it is hosted or have the code on your local machine, you can `
                + `install it using the Manual Install tab.`)}
            </p>
          </Accordion.Body>
        </Accordion.Item>
        <Accordion.Item eventKey="2">
          <Accordion.Header>{t(`Reporting Issues`)}</Accordion.Header>
          <Accordion.Body>
            <h6>General Plugin Bugs:</h6>
            <p className="plugin-small-text">
              {t(`If you encounter an issue while running a plugin, please report it to the maintainers `
                + `of the plugin. If you have installed a plugin from the Registry, its Registry listing `
                + `should include a link to the plugin's Issue Tracker. This is the recommended avenue `
                + `for reporting bugs.`)}
            </p>
            <h6>Security Vulnerabilities:</h6>
            <p className="plugin-small-text">
              {t(`The inclusion of a plugin in the InVEST Plugin Registry does not imply or guarantee `
                + `anything about the plugin's quality, suitability, or security, and the Natural `
                + `Capital Alliance reserves the right to remove any plugin from the Registry that is `
                + `believed to be a security risk. If you believe a plugin poses a security risk, please `
                + `go to the `)}
              <a
                href={securityReportURL}
                title={securityReportURL}
                aria-label={t(`Security and Quality tab of Registry GitHub repo (opens in web browser)`)}
                onClick={openLinkInBrowser}
              >{t(`Security and Quality tab`)}</a>
              {t(` of the Registry GitHub Repo and use the "Report a vulnerability" button to file a report.`)}
            </p>
          </Accordion.Body>
        </Accordion.Item>
        <Accordion.Item eventKey="3">
          <Accordion.Header>{t(`Developing a Plugin`)}</Accordion.Header>
          <Accordion.Body>
            <p className="plugin-small-text">
              {t(`Have an idea for a plugin or an existing script that you'd like to turn into `
                + `one? Take a look at the `)}
              <a
                href={pluginDocsURL}
                title={pluginDocsURL}
                aria-label={t(`Plugins Developer's Guide (opens in web browser)`)}
                onClick={openLinkInBrowser}
              >{t(`Plugin Developer Docs`)}</a>
              {t(` to learn more.`)}
            </p>
            <p className="plugin-small-text">
              {t(`If you've created an InVEST plugin and would like to make it available to `
                + `others in the community, the best way to make it discoverable is to submit it `
                + `for inclusion in the Registry. Take a look at the `)}
              <a
                href={pluginSubmissionURL}
                title={pluginSubmissionURL}
                aria-label={t(`Plugin Submission Docs (opens in web browser)`)}
                onClick={openLinkInBrowser}
              >{t(`Plugin Submission Docs`)}</a>
              {t(` for a detailed guide.`)}
            </p>
          </Accordion.Body>
        </Accordion.Item>
      </Accordion>
      <h5 className="mt-4 mb-3">{t(`About the Tabs in This Modal:`)}</h5>
      <Accordion>
        <Accordion.Item eventKey="0">
          <Accordion.Header>{t(`Plugin Registry`)}</Accordion.Header>
          <Accordion.Body>
            <p className="plugin-small-text">
              {t(`You can use the Plugin Registry tab to browse and install plugins listed on the `
                + ` InVEST Plugin Registry from directly within the Workbench.`)}
            </p>
            <p className="plugin-small-text">
              {t(`The `)}
              <a
                href={pluginRegistryURL}
                title={pluginRegistryURL}
                aria-label={t(`Plugin Registry (opens in web browser)`)}
                onClick={openLinkInBrowser}
              >{t(`Plugin Registry`)}</a>
              {t(` is a hub for community-contributed plugins. The goal of the Plugin Registry is `
                + `to make it easy for members of the InVEST community to both share the plugins `
                + `they have created and discover plugins created by others.`)}
            </p>
          </Accordion.Body>
        </Accordion.Item>
        <Accordion.Item eventKey="1">
          <Accordion.Header>{t(`Installed Plugins`)}</Accordion.Header>
          <Accordion.Body>
            <p className="plugin-small-text">
            {t(`This Installed Plugins tab provides a list of all plugins you have installed `
              + `in the Workbench. The listing includes the plugin version, as well as information `
              + `about the installation source. You can use this tab to uninstall plugins from the `
              + `Workbench.`)}
            </p>
          </Accordion.Body>
        </Accordion.Item>
        <Accordion.Item eventKey="2">
          <Accordion.Header>{t(`Manual Install`)}</Accordion.Header>
          <Accordion.Body>
            <p className="plugin-small-text">
              {t(`The Manual Install tab allows you to install a plugin via a git URL or local file `
                + `path. This is useful if you are working on developing a plugin, or if you want to `
                + `install a plugin (or a version of a plugin) that is not available on the Plugin `
                + `Registry.`)}
            </p>
          </Accordion.Body>
        </Accordion.Item>
        <Accordion.Item eventKey="3">
          <Accordion.Header>{t(`Advanced Settings`)}</Accordion.Header>
          <Accordion.Body>
            <p className="plugin-small-text">
              {t(`The Advanced Settings tab allows you to manually configure the conda executable `
                + `used by InVEST for plugin management, as well as each plugin's conda environment. `
                + `These options are primarily geared towards plugin developers; unless you encounter `
                + `a problem with micromamba when installing plugins, you likely won't need to adjust `
                + `these settings.`)}
            </p>
          </Accordion.Body>
        </Accordion.Item>
      </Accordion>
    </>
  )
}