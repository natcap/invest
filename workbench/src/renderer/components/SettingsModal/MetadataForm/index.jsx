import React, { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';

import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import Alert from 'react-bootstrap/Alert';
import Button from 'react-bootstrap/Button';
import Form from 'react-bootstrap/Form';

import {
  getGeoMetaMakerProfile,
  setGeoMetaMakerProfile,
} from '../../../server_requests';

import { openLinkInBrowser } from '../../../utils';

function AboutMetadataDiv() {
  const { t } = useTranslation();

  return (
    <div>
      <h4>{t('Metadata for InVEST results')}</h4>
      <p>
        {t(`InVEST models create metadata files that describe each dataset
        created by the model. These are the "*.yml", or YAML, files
        in the output workspace after running a model.`)}
      </p>
      <p>
        {t(`Open a YAML file in a text editor to read the metadata and even add
        to it. Metadata includes descriptions of fields in tables,
        the bands in a raster, and other useful information.`)}
      </p>
      <p>
        {t(`Some properties of the metadata are configurable here. You may
        save information about the data author (you) and data license
        information. These details are included in all metadata documents
        created by InVEST and by GeoMetaMaker. This information is optional,
        it never leaves your computer unless you share your data and metadata,
        and you may modify it here anytime.`)}
      </p>
      <p>
        {t('InVEST uses GeoMetaMaker to generate metadata. Learn more about')}
        <a
          href="https://github.com/natcap/geometamaker"
          onClick={openLinkInBrowser}
        >GeoMetaMaker on Github</a>.
      </p>
    </div>
  );
}

function FormRow(label, value, handler) {
  return (
    <Row>
      <Col sm="4">
        <Form.Label>{label}</Form.Label>
      </Col>
      <Col sm="8">
        <Form.Control
          type="text"
          value={value || ''}
          onChange={(e) => handler(e.currentTarget.value)}
        />
      </Col>
    </Row>
  );
}

/**
 * A form for submitting GeoMetaMaker profile data.
 */
export default function MetadataForm() {
  const { t } = useTranslation();

  const [contactName, setContactName] = useState('');
  const [contactEmail, setContactEmail] = useState('');
  const [contactOrg, setContactOrg] = useState('');
  const [contactPosition, setContactPosition] = useState('');
  const [licenseTitle, setLicenseTitle] = useState('');
  const [licenseURL, setLicenseURL] = useState('');
  const [alertMsg, setAlertMsg] = useState('');
  const [alertError, setAlertError] = useState(false);
  const [showInfo, setShowInfo] = useState(false);

  useEffect(() => {
    async function loadProfile() {
      const profile = await getGeoMetaMakerProfile();
      if (profile && profile.contact) {
        setContactName(profile.contact.individual_name);
        setContactEmail(profile.contact.email);
        setContactOrg(profile.contact.organization);
        setContactPosition(profile.contact.position_name);
      }
      if (profile && profile.license) {
        setLicenseTitle(profile.license.title);
        setLicenseURL(profile.license.path);
      }
    }
    loadProfile();
  }, []);

  const handleSubmit = async (event) => {
    event.preventDefault();
    const { message, error } = await setGeoMetaMakerProfile({
      contact: {
        individual_name: contactName,
        email: contactEmail,
        organization: contactOrg,
        position_name: contactPosition,
      },
      license: {
        title: licenseTitle,
        path: licenseURL,
      },
    });
    setAlertMsg(message);
    setAlertError(error);
  };

  return (
    <div id="metadata-form">
      {
        (showInfo)
          ? <AboutMetadataDiv />
          : (
            <Form onSubmit={handleSubmit} onChange={() => setAlertMsg('')}>
              <fieldset>
                <legend>{t('Contact Information')}</legend>
                <Form.Group controlId="name">
                  {FormRow(t('Full name'), contactName, setContactName)}
                </Form.Group>
                <Form.Group controlId="email">
                  {FormRow(t('Email address'), contactEmail, setContactEmail)}
                </Form.Group>
                <Form.Group controlId="job-title">
                  {FormRow(t('Job title'), contactPosition, setContactPosition)}
                </Form.Group>
                <Form.Group controlId="organization">
                  {FormRow(t('Organization name'), contactOrg, setContactOrg)}
                </Form.Group>
              </fieldset>
              <fieldset>
                <legend>{t('Data License Information')}</legend>
                <Form.Group controlId="license-title">
                  {FormRow(t('Title'), licenseTitle, setLicenseTitle)}
                </Form.Group>
                <Form.Group controlId="license-url">
                  {FormRow('URL', licenseURL, setLicenseURL)}
                </Form.Group>
              </fieldset>
              <Form.Row>
                <Button
                  type="submit"
                  variant="primary"
                  className="my-1 py2 mx-2"
                >
                  {t('Save Metadata')}
                </Button>
                {
                  (alertMsg) && (
                    <Alert
                      className="my-1 py-2"
                      variant={alertError ? 'danger' : 'success'}
                    >
                      {alertMsg}
                    </Alert>
                  )
                }
              </Form.Row>
            </Form>
          )
      }
      <Button
        variant="outline-secondary"
        className="my-1 py2 mx-2 info-toggle"
        onClick={() => setShowInfo((prevState) => !prevState)}
      >
        {showInfo ? t('Hide Info') : t('More Info')}
      </Button>
    </div>
  );
}
