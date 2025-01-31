import React, { useState, useEffect, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import Alert from 'react-bootstrap/Alert';
import Button from 'react-bootstrap/Button';
import Form from 'react-bootstrap/Form';

import Expire from '../../Expire';
import {
  getGeoMetaMakerProfile,
  setGeoMetaMakerProfile,
} from '../../../server_requests';

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
  const [alertKey, setAlertKey] = useState(0);

  useEffect(() => {
    async function loadProfile() {
      const profile = await getGeoMetaMakerProfile();
      if (profile.contact) {
        setContactName(profile.contact.individual_name);
        setContactEmail(profile.contact.email);
        setContactOrg(profile.contact.organization);
        setContactPosition(profile.contact.position_name);
      }
      if (profile.license) {
        setLicenseTitle(profile.license.title);
        setLicenseURL(profile.license.path);
      }
    }
    loadProfile();
  }, []);

  const handleSubmit = useCallback(async (event) => {
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
    const key = window.crypto.getRandomValues(new Uint16Array(1))[0].toString();
    setAlertKey(key);
  }, []);

  return (
    <Form onSubmit={handleSubmit} id="metadata-form">
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
          Save Metadata
        </Button>
        {
          (alertMsg) && (
          <Expire
            key={alertKey}
            delay={4000}
          >
            <Alert
              className="my-1 py-2"
              variant={alertError ? 'danger' : 'success'}
            >
              {t(alertMsg)}
            </Alert>
          </Expire>
          )
        }
      </Form.Row>
    </Form>
  );
}
