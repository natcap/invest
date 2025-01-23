import React from 'react';
import PropTypes from 'prop-types';

import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import Button from 'react-bootstrap/Button';
import Form from 'react-bootstrap/Form';

function FormRow(label) {
  return (
    <Row>
      <Col sm="4"><Form.Label>{label}</Form.Label></Col>
      <Col sm="8"><Form.Control /></Col>
    </Row>
  );
}

export default function MetadataForm(props) {

  function handleSubmit(event) {
    event.preventDefault();
  }

  return (
    <Form>
      <Form.Group controlId="name">
        {FormRow('Name')}
      </Form.Group>
      <Form.Group controlId="email">
        {FormRow('Email address')}
      </Form.Group>
      <Form.Group controlId="job title">
        {FormRow('Job title')}
      </Form.Group>
      <Form.Group controlId="organization">
        {FormRow('Organization name')}
      </Form.Group>
      <Form.Group controlId="license title">
        {FormRow('Data license title')}
      </Form.Group>
      <Form.Group controlId="license url">
        {FormRow('Data license URL')}
      </Form.Group>

      <Button variant="primary" onClick={handleSubmit}>
        Submit Metadata
      </Button>
    </Form>
  );
}
