import React from 'react';

import { useTranslation } from 'react-i18next';

import Button from 'react-bootstrap/Button';
import { MdOpenInNew } from 'react-icons/md';

import { openLinkInBrowser } from '../../../utils';

export default function NeedsMSVC(props) {
  const {
    downloadMSVC
  } = props;

  const { t } = useTranslation();

  return (
    <>
      <h5>
        {t('Microsoft Visual C++ Redistributable must be installed!')}
      </h5>
      <p>
        {t('Plugin features require the ')}
        <a
          href="https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist"
          title="https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist"
          onClick={openLinkInBrowser}
        >
          {t('Microsoft Visual C++ Redistributable')}
          <MdOpenInNew
            aria-label={t("(opens in web browser)")}
          />
        </a>
        {t('. You must download and install the redistributable before continuing.')}
      </p>
      <Button
        className="mt-3"
        onClick={downloadMSVC}
      >
        {t('Continue to download and install')}
      </Button>
    </>
  )
}
