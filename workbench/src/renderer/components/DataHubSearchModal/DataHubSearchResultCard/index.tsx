import Button from 'react-bootstrap/Button';
import { useTranslation } from 'react-i18next';
import { PiCaretCircleDown, PiCaretCircleUp } from 'react-icons/pi';

import type { DataHubSearchResult } from '../models';
import { openLinkInBrowser } from '../../../utils';

interface DataHubSearchResultCardProps {
  datasetDetails: DataHubSearchResult,
  expanded: boolean,
  onToggleExpanded: () => void,
  onSelect: (url: string, collections: string[]) => void,
}

export default function DataHubSearchResultCard(
  props: DataHubSearchResultCardProps
) {
  const {
    datasetDetails,
    expanded,
    onToggleExpanded,
    onSelect,
  } = props;

  const { t } = useTranslation();

  const {
    id, title, description, tags, places, collections, license,
    author, lastUpdated, created, datasetUrl, webpageUrl
  } = datasetDetails;

  const descriptionPreviewLength = 430;
  const descriptionPreview = (
    description.length > descriptionPreviewLength
    ? (description.slice(0, descriptionPreviewLength) + '…')
    : description
  );

  return (
    <div className={`search-result ${expanded ? 'search-result-expanded' : ''}`}>
      <div className="search-result-header">
        <h3 className="h5 m-0" id={`${id}-title`}>{title}</h3>
        <div className="search-result-controls">
          <Button
            aria-describedby={`${id}-title`}
            aria-expanded={expanded}
            aria-controls={`${id}-details`}
            onClick={onToggleExpanded}
            variant="secondary"
            className="details-button"
          >
            {expanded ? <PiCaretCircleUp /> : <PiCaretCircleDown />}
            {t('Details')}
          </Button>
          <Button
            onClick={() => onSelect(datasetUrl, collections)}
            aria-describedby={`${id}-title`}
          >
            {t('Select')}
          </Button>
        </div>
      </div>
      {
        expanded &&
        <div id={`${id}-details`}>
          <p>
            {descriptionPreview}
          </p>
          <dl className="search-result-metadata">
            <dt>{t('Tags')}</dt>
            <dd>{tags.join(', ')}</dd>
            <dt>{t('Places')}</dt>
            <dd>{places.join(', ')}</dd>
            {
              collections.length > 0 &&
              <>
                <dt>{t('Collections')}</dt>
                <dd>{collections.join(', ')}</dd>
              </>
            }
            <dt>{t('License')}</dt>
            <dd>{license}</dd>
            <dt>{t('Author')}</dt>
            <dd>{author}</dd>
            <dt>{t('Last Updated')}</dt>
            <dd>{lastUpdated}</dd>
            <dt>{t('Created')}</dt>
            <dd>{created}</dd>
            <dt>{t('Full Details and Preview')}</dt>
            <dd>
              <a
                href={webpageUrl}
                onClick={openLinkInBrowser}
              >
                {t(`${title} (opens in web browser)`)}
              </a>
            </dd>
          </dl>
        </div>
      }
    </div>
  );
}
