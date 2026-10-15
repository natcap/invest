import React from 'react';

import { MdOpenInNew } from "react-icons/md";
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
      <h5>{t(`About Plugins`)}</h5>
      <p>
        {t(`Conceptually, an InVEST plugin is an ecosystem services model or performs `
          + `functions related to ecosystem services modeling. Unlike the core models, `
          + `a plugin is not 'official', i.e., not reviewed or maintained by NatCap. `
          + `Plugins may be developed, used, and distributed independently of the `
          + `natcap/invest repo and the Natural Capital Alliance.`)}
      </p>
      <p className="mb-0">{t(`External Links:`)}</p>
      <p>
        <a
            href={pluginDocsURL}
            title={pluginDocsURL}
            aria-label={t(`(opens in web browser)`)}
            onClick={openLinkInBrowser}
          >
            {t(`Plugin Developer's Guide`)}
            <MdOpenInNew
              aria-label={t("(opens in web browser)")}
            />
          </a> | <a
            href={pluginRegistryURL}
            title={pluginRegistryURL}
            aria-label={t(`(opens in web browser)`)}
            onClick={openLinkInBrowser}
          >
            {t(`Plugin Registry site`)}
            <MdOpenInNew
              aria-label={t("(opens in web browser)")}
            />
          </a> | <a
            href={registryGitHubURL}
            title={registryGitHubURL}
            aria-label={t(`(opens in web browser)`)}
            onClick={openLinkInBrowser}
          >
            {t(`Plugin Registry GitHub Repo`)}
            <MdOpenInNew
              aria-label={t("(opens in web browser)")}
            />
          </a>
      </p>
      <hr />
      <p className="bold-text mt-3">{t(`Installing a Plugin`)}</p>
      <p>
        {t(`You can use the Plugin Registry tab to browse for plugins included on the InVEST `
          + `Plugin Registry, a listing of community-contributed plugins. If you find one you `
          + `would like to install, you can do so directly from the Workbench using the "Install" `
          + `button.`)}
      </p>
      <p>
        {t(`Installation will take a few minutes. Once it completes, the "Install" form will `
          + `be replaced by a message saying "This plugin is installed!" You can then close the `
          + `modal and launch the plugin from the list of models.`)}
      </p>
      <p>
        {t(`If you want to install a plugin that isn't listed on the Registry, and have either the `
          + `link to a git repo where it is hosted or the code on your local machine, you can `
          + `install it using the Manual Install tab.`)}
      </p>
      <hr />
      <p className="bold-text">{t(`Reporting Issues`)}</p>
      <p>{t(`General Plugin Bugs:`)}</p>
      <p>
        {t(`If you encounter an issue while running a plugin, please report it to the maintainers `
          + `of the plugin. If you have installed a plugin from the Registry, its Registry listing `
          + `should include a link to the plugin's Issue Tracker. This is the recommended avenue `
          + `for reporting bugs.`)}
      </p>
      <p>{t(`Security Vulnerabilities:`)}</p>
      <p>
        {t(`The inclusion of a plugin in the InVEST Plugin Registry does not imply or guarantee `
          + `anything about the plugin's quality, suitability, or security, and the Natural `
          + `Capital Alliance reserves the right to remove any plugin from the Registry that is `
          + `believed to be a security risk. If you believe a plugin poses a security risk, please `
          + `go to the Security and Quality tab of the Registry GitHub repo and use the "report `
          + `a vulnerability" button to file a report: `)}
        <a
          href={securityReportURL}
          title={securityReportURL}
          aria-label={t(`(opens in web browser)`)}
          onClick={openLinkInBrowser}
        >
          {t(`Security and Quality tab`)}
          <MdOpenInNew
            aria-label={t("(opens in web browser)")}
          />
        </a>
      </p>
      <hr />
      <p className="bold-text">{t(`Developing a Plugin`)}</p>
      <p>
      {t(`Have an idea for a plugin or an existing script that you'd like to turn into `
        + `one? Take a look at the plugin developer docs to learn more: `)}
        <a
          href={pluginDocsURL}
          title={pluginDocsURL}
          aria-label={t(`(opens in web browser)`)}
          onClick={openLinkInBrowser}
        >
          {t(`Developer's Guide`)}
          <MdOpenInNew
            aria-label={t("(opens in web browser)")}
          />
        </a>
      </p>
      <p>
        {t(`If you've created an InVEST plugin and would like to make it available to `
          + `others in the community, the best way to make it discoverable is to submit it `
          + `for inclusion in the Registry. Detailed instructions for submission: `)}
        <a
          href={pluginSubmissionURL}
          title={pluginSubmissionURL}
          aria-label={t(`(opens in web browser)`)}
          onClick={openLinkInBrowser}
        >
          {t(`Submission Guide`)}
          <MdOpenInNew
            aria-label={t("(opens in web browser)")}
          />
        </a>
      </p>
    </>
  )
}
