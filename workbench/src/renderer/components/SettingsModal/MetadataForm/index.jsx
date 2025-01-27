import React, { useState, useEffect, } from 'react';
import PropTypes from 'prop-types';

import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import Button from 'react-bootstrap/Button';
import Form from 'react-bootstrap/Form';

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
          value={value}
          onChange={(e) => handler(e.currentTarget.value)}
        />
      </Col>
    </Row>
  );
}

export default function MetadataForm() {
  const [contactName, setContactName] = useState('');
  const [contactEmail, setContactEmail] = useState('');
  const [contactOrg, setContactOrg] = useState('');
  const [contactPosition, setContactPosition] = useState('');
  const [licenseTitle, setLicenseTitle] = useState('');
  const [licenseURL, setLicenseURL] = useState('');

  useEffect(() => {
    async function loadProfile() {
      const profile = await getGeoMetaMakerProfile();
      console.log(profile);
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

  function handleSubmit(event) {
    event.preventDefault();
    setGeoMetaMakerProfile({
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
  }

  return (
    <Form onSubmit={handleSubmit} id="metadata-form">
      <fieldset>
        <legend>Contact Information</legend>
        <Form.Group controlId="name">
          {FormRow('Name', contactName, setContactName)}
        </Form.Group>
        <Form.Group controlId="email">
          {FormRow('Email address', contactEmail, setContactEmail)}
        </Form.Group>
        <Form.Group controlId="job-title">
          {FormRow('Job title', contactPosition, setContactPosition)}
        </Form.Group>
        <Form.Group controlId="organization">
          {FormRow('Organization name', contactOrg, setContactOrg)}
        </Form.Group>
      </fieldset>
      <fieldset>
        <legend>Data License Information</legend>
        <Form.Group controlId="license-title">
          {FormRow('Title', licenseTitle, setLicenseTitle)}
        </Form.Group>
        <Form.Group controlId="license-url">
          {FormRow('URL', licenseURL, setLicenseURL)}
        </Form.Group>
      </fieldset>
      <Button variant="primary" type="submit">
        Save Metadata
      </Button>
    </Form>
  );
}
