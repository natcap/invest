import PropTypes from 'prop-types';
import Button from 'react-bootstrap/Button';
import Modal from 'react-bootstrap/Modal';
import { MdClose } from 'react-icons/md';
import { useTranslation } from 'react-i18next';

export default function Changelog(props) {
    const { t } = useTranslation();
    return (
        <Modal
            show={props.show}
            onHide={props.close}
        >
            <Modal.Header>
                <Modal.Title>{t('New in this version')}</Modal.Title>
                <Button
                    variant="secondary-outline"
                    onClick={props.close}
                    className="float-right"
                    aria-label="Close modal"
                >
                    <MdClose />
                </Button>
            </Modal.Header>
            <Modal.Body>
                <ul>
                    <li>
                        General
                        <ul>
                            <li>Updated something</li>
                            <li>Updated something else</li>
                        </ul>
                    </li>
                    <li>
                        Workbench
                        <ul>
                            <li>Added a feature</li>
                            <li>Fixed a bug</li>
                        </ul>
                    </li>
                    <li>
                        Etc.
                    </li>
                </ul>
            </Modal.Body>
        </Modal>
    )
}

Changelog.propTypes = {
    show: PropTypes.bool.isRequired,
    close: PropTypes.func.isRequired,
};
