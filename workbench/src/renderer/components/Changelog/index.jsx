import React, { useEffect, useState } from 'react';
import PropTypes from 'prop-types';
import Button from 'react-bootstrap/Button';
import Modal from 'react-bootstrap/Modal';
import { MdClose } from 'react-icons/md';
import { useTranslation } from 'react-i18next';

import pkg from '../../../../package.json';
import { ipcMainChannels } from '../../../main/ipcMainChannels';

const { ipcRenderer } = window.Workbench.electron;
const { logger } = window.Workbench;

export default function Changelog(props) {
  const { t } = useTranslation();
  const [htmlContent, setHtmlContent] = useState('');

  // Load HTML from external file (which is generated by Python build process).
  useEffect(() => {
    async function loadHtml() {
      const baseUrl = await ipcRenderer.invoke(ipcMainChannels.BASE_URL);
      const response = await fetch(`${baseUrl}/changelog.html`);
      if (!response.ok) {
        logger.debug(`Error fetching changelog HTML: ${response.status} ${response.statusText}`);
        return;
      }
      try {
        const htmlString = await response.text();
        // Find the section whose heading explicitly matches the current version.
        const versionStr = pkg.version;
        const escapedVersionStr = versionStr.split('.').join('\\.');
        const sectionRegex = new RegExp(
          `<section.*?>[\\s]*?<h1>${escapedVersionStr}\\b[\\s\\S]*?</h1>[\\s\\S]*?</section>`
        );
        const sectionMatches = htmlString.match(sectionRegex);
        if (sectionMatches && sectionMatches.length) {
          let latestVersionSection = sectionMatches[0];
          const linkRegex = /<a\shref/g;
          // Ensure all links open in a new window and are styled with a relevant icon.
          latestVersionSection = latestVersionSection.replaceAll(
            linkRegex,
            '<a target="_blank" class="link-external" href'
          );
          setHtmlContent({
            __html: latestVersionSection
          });
        }
      } catch(error) {
        logger.debug(error);
      }
    }
    loadHtml();
  }, []);

  // Once HTML content has loaded, set up links to open in browser
  // (instead of in an Electron window).
  useEffect(() => {
    const openLinkInBrowser = (event) => {
      event.preventDefault();
      ipcRenderer.send(
        ipcMainChannels.OPEN_EXTERNAL_URL, event.currentTarget.href
      );
    };
    document.querySelectorAll('.link-external').forEach(link => {
      link.addEventListener('click', openLinkInBrowser);
    });
  }, [htmlContent]);

  return (
    <Modal
      show={props.show && htmlContent !== ''}
      onHide={props.close}
      size="lg"
      aria-labelledby="changelog-modal-title"
    >
      <Modal.Header>
        <Modal.Title id="changelog-modal-title">
          {t('New in this version')}
        </Modal.Title>
        <Button
          variant="secondary-outline"
          onClick={props.close}
          className="float-right"
          aria-label="Close modal"
        >
          <MdClose />
        </Button>
      </Modal.Header>
      {/* Setting inner HTML in this way is OK because
      the HTML content is controlled by our build process
      and not, for example, sourced from user input. */}
      <Modal.Body
        dangerouslySetInnerHTML={htmlContent}
      >
      </Modal.Body>
    </Modal>
  );
}

Changelog.propTypes = {
  show: PropTypes.bool.isRequired,
  close: PropTypes.func.isRequired,
};